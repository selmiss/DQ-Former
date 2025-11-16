"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn
import logging
from torch.cuda.amp import autocast as autocast
from peft import get_peft_model, LoraConfig, TaskType

from models.configuration import MolLLaMAConfig
from models.DQ_former_encoder import DQMolLLaMAEncoder
from models.mol_llama_encoder import MolLLaMAEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, GenerationMixin, BitsAndBytesConfig, LlamaForCausalLM

from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
from torch_geometric.data import Data, Batch
from data_provider.mol_dataset import smiles2graph, get_unimol_data
from data_provider.collaters import Mol3DCollater
import numpy as np
from safetensors.torch import load_file as load_safetensors
from pathlib import Path

logger = logging.getLogger(__name__)
# Set to ERROR level to suppress warnings (INFO < WARNING < ERROR < CRITICAL)
logger.setLevel(logging.ERROR)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def unlock_new_token_embeddings(embedding_layer, new_token_ids, init="mean"):
    """
    Unfreeze the rows in embedding_layer corresponding to new_token_ids,
    并可选初始化这些 token 的 embedding 向量。

    Args:
        embedding_layer (nn.Embedding): from self.llm.get_input_embeddings()
        new_token_ids (List[int]): 新增 token 的 ID 列表
        init (str or None): 初始化策略。可选：
            - "mean": initialize with the mean of the original vocabulary
            - "zero": initialize to 0 vector
            - None: do not initialize (keep default random)
    """
    # 1. 全部冻结
    embedding_layer.weight.requires_grad = False

    # 2. 有需要的话初始化新增行
    with torch.no_grad():
        if init == "mean":
            old_vocab_size = embedding_layer.weight.shape[0] - len(new_token_ids)
            avg_vec = embedding_layer.weight[:old_vocab_size].mean(dim=0)
            for idx in new_token_ids:
                embedding_layer.weight[idx].copy_(avg_vec)
        elif init == "zero":
            for idx in new_token_ids:
                embedding_layer.weight[idx].zero_()

    # 3. 单独解冻这些行
    for idx in new_token_ids:
        embedding_layer.weight[idx].requires_grad = True

    logger.info(f"✅ Unfrozen {len(new_token_ids)} tokens: {new_token_ids}")


class MolLLaMAPreTrainedModel(PreTrainedModel):
    config_class = MolLLaMAConfig
    base_model_prefix = 'mllm'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"encoder.graph_encoder",
        r"llm."
    ]

class DQMolLLaMA(MolLLaMAPreTrainedModel):
    def __init__(
        self,
        config: MolLLaMAConfig,
        vocab_size=None,
        torch_dtype="float16",
        enable_flash=True,
        add_ids=None,
        local_q_only=False,
        freeze_llm=False,
        brics_gids_enable=False,
        entropy_gids_enable=False,
        enable_blending=False,
        load_ckpt_before_peft=False,
        ckpt_path=None,
    ):
        super().__init__(config)
        self.config = config
        ## Intialize encoder
        if enable_blending:
            config.graph_encoder_config.encoder_types = ['unimol', 'moleculestm']
        self.encoder = DQMolLLaMAEncoder(
            graph_encoder_config = config.graph_encoder_config,
            blending_module_config = config.blending_module_config,
            qformer_config = config.qformer_config,
            brics_gids_enable = brics_gids_enable,
            entropy_gids_enable = entropy_gids_enable,
            enable_blending = enable_blending,
        )
        self.local_q_only = local_q_only
        self.brics_gids_enable = brics_gids_enable
        self.entropy_gids_enable = entropy_gids_enable
        self.postprocess_encoder()
        ## Initialize LLM
        if torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        # -------------------------- train llm ----------------------------------
        if not freeze_llm:
            logger.info(f"Loading LLM model: {config.llm_config.llm_model}")
            if enable_flash:
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        config.llm_config.llm_model,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                    )
                    logger.info("Using flash attention")
                except TypeError:
                    # Some architectures may not accept attn_implementation
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        config.llm_config.llm_model,
                        torch_dtype=torch_dtype,
                    )
            else:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    config.llm_config.llm_model,
                    torch_dtype=torch_dtype,
                )
            self.llm.resize_token_embeddings(vocab_size)
            
            # Create llm_proj BEFORE loading checkpoint so it can be loaded properly
            self.llm_proj = nn.Linear(self.encoder.Qformer.config.hidden_size, 
                                        self.llm.config.hidden_size)
            
            # Load checkpoint before PEFT if requested
            if load_ckpt_before_peft and ckpt_path:
                logger.info(f"🔧 Loading checkpoint BEFORE PEFT model creation: {ckpt_path}")
                self._load_checkpoint_before_peft(ckpt_path)
            
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                        inference_mode=False,
                                        r=config.llm_config.lora_config.r,
                                        lora_alpha=config.llm_config.lora_config.lora_alpha,
                                        lora_dropout=config.llm_config.lora_config.lora_dropout,
                                        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
            self.peft_config = peft_config
            
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()

        # -------------------------- frozen llm ----------------------------------

        else:
        # 1. 加载基座模型
            logger.info(f"Loading LLM model: {config.llm_config.llm_model}")
            self.llm = LlamaForCausalLM.from_pretrained(
                config.llm_config.llm_model,
                # quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2" if enable_flash else None,
                # device_map="auto",
            )

            # 2. 如果你自己扩充过词表，仍然可以保留这一行
            self.llm.resize_token_embeddings(vocab_size)

            # 3. 冻结 & eval
            self.llm.eval()                # 关闭 dropout / LayerNorm 统计更新
            for p in self.llm.parameters():  
                p.requires_grad = False    # 明确告诉框架"别把梯度算进去"

            if add_ids is not None:
                embed = self.llm.get_input_embeddings()
                unlock_new_token_embeddings(embed, add_ids, init="mean")
            
            # Create llm_proj for frozen LLM case too
            self.llm_proj = nn.Linear(self.encoder.Qformer.config.hidden_size, 
                                        self.llm.config.hidden_size)

    def postprocess_encoder(self):
        self.encoder.Qformer.cls = None
        self.encoder.Qformer.bert.embeddings.word_embeddings = None
        self.encoder.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.encoder.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.encoder.graph_proj = None
        self.encoder.text_proj = None
        self.encoder.gtm_head = None

    def inject_queries(
        self,
        query_output: torch.Tensor,          # [B, Q_total, D]
        text_embeds: torch.Tensor,           # [B, L, D]
        mol_token_flag: torch.Tensor,        # [B, L]  bool
        attention_mask: torch.Tensor,        # [B, L]  0/1, 左 padding
        labels: torch.Tensor,                # [B, L]  int  (可含 -100)
        max_pos: int,                        # llm.config.max_position_embeddings
        local_q_only: bool = False,          # 是否只使用 local Q
    ):
        ignore_index = -100
        B, _, D = query_output.shape
        query_output = query_output.to(text_embeds.dtype)

        embeds_list, mask_list, label_list, new_lengths = [], [], [], []

        for i in range(B):
            flag_i = mol_token_flag[i].nonzero(as_tuple=False).squeeze()  # True 位置
            q_i    = query_output[i]                                     # [Q_total, D]
            n_true = flag_i.numel()                                      # 全局 Q 数
            n_q    = q_i.size(0)                                         # 全部 Q 数
            pad_left = (attention_mask[i] == 0).sum().item()             # 原左侧 0 数

            assert n_q >= n_true, f"第 {i} 个样本 Q 数 {n_q} < True 数 {n_true}"

            # --- 1. 写入全局 Q（覆盖） ---
            x = text_embeds[i]
            if labels is not None:
                l = labels[i]
                if not local_q_only:                                      # view
                    x[flag_i] = q_i[:n_true]
                l[flag_i] = ignore_index
            else:
                if not local_q_only:
                    x[flag_i] = q_i[:n_true]

            # --- 2. 插入 local Q（在最后一个 True 右侧） ---
            local_q = q_i[n_true:]                                       # 可能为空
            if local_q.numel():
                insert_pos = flag_i[-1].item() + 1
                x = torch.cat([x[:insert_pos], local_q, x[insert_pos:]], dim=0)
                if labels is not None:
                    local_lbl = torch.full((local_q.size(0),), ignore_index, dtype=l.dtype, device=l.device)
                    l = torch.cat([l[:insert_pos], local_lbl, l[insert_pos:]], dim=0)
            # --- 3. 生成对应 attention mask ---
            cur_len = x.size(0)
            ones_len = cur_len - pad_left
            cur_mask = torch.cat([
                torch.zeros(pad_left, dtype=torch.long, device=x.device),
                torch.ones(ones_len, dtype=torch.long, device=x.device)
            ], dim=0)                                                    # [cur_len]

            embeds_list.append(x)
            mask_list.append(cur_mask)
            if labels is not None:
                label_list.append(l)
            else:
                label_list.append(None)
            new_lengths.append(cur_len)

        # --- 4. pad / 截断到批内最大 & max_pos ---
        max_len = min(max(new_lengths), max_pos)
        padded_embeds, padded_mask, padded_labels = [], [], []

        for emb, m, l in zip(embeds_list, mask_list, label_list):
            emb = emb[:max_len]
            m   = m[:max_len]
            if labels is not None:
                l   = l[:max_len]            # 同样截断 label

            if emb.size(0) < max_len:        # 右侧 pad
                pad_len = max_len - emb.size(0)
                emb_pad = torch.zeros(pad_len, D, dtype=emb.dtype, device=emb.device)
                m_pad   = torch.zeros(pad_len,     dtype=m.dtype,   device=m.device)
                if labels is not None:
                    l_pad = torch.full((pad_len,), ignore_index, dtype=l.dtype, device=l.device)

                emb = torch.cat([emb, emb_pad], dim=0)
                m   = torch.cat([m,   m_pad],   dim=0)
                if labels is not None:
                    l   = torch.cat([l,   l_pad],   dim=0)

            padded_embeds.append(emb)
            padded_mask.append(m)
            if labels is not None:
                padded_labels.append(l)
            else:
                padded_labels.append(None)


        text_embeds = torch.stack(padded_embeds, dim=0)     # [B, max_len, D]
        attention_mask = torch.stack(padded_mask, dim=0)    # [B, max_len]
        if labels is not None:
            labels = torch.stack(padded_labels, dim=0)          # [B, max_len]

        return text_embeds, attention_mask, labels, max_len



    def forward(self, graph_batch, text_batch):
        # brics_gids and entropy_gids are now in graph_batch
        # The encoder will extract them automatically
        _, _, query_output = self.encoder(graph_batch)
        query_output = self.llm_proj(query_output.last_hidden_state) #[batch_size,num_query_token,dim]

        inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]

        if hasattr(text_batch, 'labels'):
            labels = text_batch.labels
        else:
            labels = None

        inputs_embeds, attention_mask, labels, _ = self.inject_queries(
            query_output=query_output,
            text_embeds=inputs_embeds,
            mol_token_flag=text_batch.mol_token_flag,
            max_pos=self.llm.config.max_position_embeddings,
            labels=labels,
            attention_mask=text_batch.attention_mask,
            local_q_only=self.local_q_only,
        )

        # Align dtypes (e.g. Half vs BFloat16) to avoid runtime errors when using quantized models
        # inputs_embeds[text_batch.mol_token_flag] = query_output.flatten(0, 1) # [batch_size, max_len, dim]
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
            use_cache=False,
        )
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        graph_batch,
        text_batch,
        do_sample=False,
        num_beams=1,
        max_length=None,
        min_length=1,
        max_new_tokens=512,
        min_new_tokens=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        top_p=None,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        # 1. 图→Query
        # brics_gids and entropy_gids are already in graph_batch, no need to pass separately
        _, _, query_output = self.encoder(graph_batch)
        query_output = self.llm_proj(query_output.last_hidden_state)  # [B,Q,D]

        # 2. 原文本 embedding
        inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids)

        # 3. 复用 inject_queries，labels 可以传 None 或 text_batch.labels
        if hasattr(text_batch, 'labels'):
            labels = text_batch.labels
        else:
            labels = None

        inputs_embeds, attention_mask, _, _ = self.inject_queries(
            query_output=query_output,
            text_embeds=inputs_embeds,
            mol_token_flag=text_batch.mol_token_flag,
            attention_mask=text_batch.attention_mask,
            labels=labels,               # 或 None，取决于你函数实现
            max_pos=self.llm.config.max_position_embeddings,
            local_q_only=self.local_q_only,
        )

        # 4. 直接调 generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs

    @torch.no_grad()
    def generate_with_smiles(
        self,
        smiles_list,
        text_batch,
        do_sample=False,
        num_beams=1,
        max_length=None,
        min_length=1,
        max_new_tokens=1024,
        min_new_tokens=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        top_p=None,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
        brics_gids=None,
        entropy_gids=None,
    ):
        graph_batch = get_mol_graphs(smiles_list, self.encoder.unimol_dictionary, self.device)
        outputs = self.generate(
            graph_batch=graph_batch,
            text_batch=text_batch,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            brics_gids=brics_gids,
            entropy_gids=entropy_gids,
        )
        return outputs

    def _load_checkpoint_before_peft(self, ckpt_path):
        """
        Internal method to load checkpoint before PEFT model creation.
        Only loads encoder and llm_proj weights, skipping LLM weights.
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
        """
        logger.info(f"Loading encoder and projector from checkpoint: {ckpt_path}")
        
        path = Path(ckpt_path)
        
        # Detect file type and load accordingly
        if path.suffix == '.safetensors':
            logger.info("Detected safetensors format")
            state_dict_raw = load_safetensors(ckpt_path)
        else:
            logger.info("Detected PyTorch checkpoint format")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict_raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Determine prefix from keys
        first_key = list(state_dict_raw.keys())[0]
        if 'mol_llama.' in first_key:
            prefix_len = 10
            prefix = "mol_llama."
        elif 'model.' in first_key:
            prefix_len = 6
            prefix = "model."
        else:
            prefix_len = 0
            prefix = ""
        
        # Extract only encoder and llm_proj weights (skip LLM)
        state_dict = {}
        for k, v in state_dict_raw.items():
            if k.startswith(prefix):
                k_stripped = k[prefix_len:]
                state_dict[k_stripped] = v
        
        logger.info(f"Found {len(state_dict)} encoder/projector parameters to load")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        # Filter expected missing keys
        expected_missing = []
        for k in missing_keys:
            if 'position_ids' in k or k.startswith("encoder.graph_encoder.") or k.startswith("encoder.static_q_mask"):
                expected_missing.append(k)
            else:
                assert False, f"Unexpected missing key: {k}"
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected missing keys: {unexpected_keys}")
            assert False, f"Unexpected missing keys: {unexpected_keys}"
        logger.info(f"✅ Successfully loaded encoder and projector weights (LLM will be initialized separately)")

    def load_from_ckpt(self, ckpt_path, lora_init=False):
        """
        Load checkpoint from either PyTorch checkpoint or HuggingFace safetensors.
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
        """
        logger.info(f"Loading from checkpoint: {ckpt_path}")
        
        path = Path(ckpt_path)

        if lora_init:
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                        inference_mode=False,
                                        r=self.config.llm_config.lora_config.r,
                                        lora_alpha=self.config.llm_config.lora_config.lora_alpha,
                                        lora_dropout=self.config.llm_config.lora_config.lora_dropout,
                                        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
            self.llm = get_peft_model(self.llm, peft_config)
        
        # Detect file type and load accordingly
        if path.suffix == '.safetensors':
            # Load HuggingFace safetensor format
            logger.info("Detected safetensors format")
            state_dict_raw = load_safetensors(ckpt_path)
        else:
            # Load PyTorch checkpoint format
            logger.info("Detected PyTorch checkpoint format")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Some checkpoints save state_dict directly, others wrap it
            state_dict_raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Determine prefix from keys
        first_key = list(state_dict_raw.keys())[0]
        if 'mol_llama.' in first_key:
            prefix_len = 10
            prefix = "mol_llama."
        elif 'model.' in first_key:
            prefix_len = 6
            prefix = "model."
        else:
            prefix_len = 0
            prefix = ""
        
        # Extract relevant state dict with prefix removal
        state_dict = {k[prefix_len:]:v for k,v in state_dict_raw.items() if k.startswith(prefix)}
        
        logger.info(f"Found {len(state_dict)} parameters to load")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"
        for k in missing_keys:
            if 'position_ids' in k: continue
            if not (k.startswith("encoder.graph_encoder.") or k.startswith("llm.")) or k.startswith("encoder.static_q_mask"):
                logger.warning(f"❌ Unexpected missing key: {k}")
            else:
                logger.warning(f"Key: {k}, make sure this key is loaded before.")
            assert k.startswith("encoder.graph_encoder.") or \
                k.startswith("llm.") or k.startswith("encoder.static_q_mask")
        
        logger.info(f"✅ Successfully loaded weights from {ckpt_path}")
        
    
    def load_from_stage1_ckpt(self, ckpt_path):
        """
        Load stage1 checkpoint from either PyTorch Lightning checkpoint or HuggingFace safetensors.
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
        """
        logger.info(f"Loading from stage1 checkpoint: {ckpt_path}")
        
        path = Path(ckpt_path)
        
        # Detect file type and load accordingly
        if path.suffix == '.safetensors':
            # Load HuggingFace safetensor format
            logger.info("Detected safetensors format")
            state_dict_raw = load_safetensors(ckpt_path)
        else:
            # Load PyTorch Lightning checkpoint format
            logger.info("Detected PyTorch checkpoint format")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Some checkpoints save state_dict directly, others wrap it
            state_dict_raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Extract encoder parameters
        # Handle different possible prefixes: "encoder.", "model.encoder.", etc.
        state_dict = {}
        for k, v in state_dict_raw.items():
            if k.startswith("encoder."):
                # Remove "encoder." prefix (8 chars)
                state_dict[k[8:]] = v
            elif k.startswith("model.encoder."):
                # Remove "model.encoder." prefix (14 chars)
                state_dict[k[14:]] = v
        
        if not state_dict:
            logger.warning(f"No encoder weights found. Available keys: {list(state_dict_raw.keys())[:5]}...")
        
        logger.info(f"Found {len(state_dict)} encoder parameters to load")
        
        
        # Load state dict into encoder
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False, assign=True)
        
        assert len(unexpected_keys) == 0, f"Unexpected keys found: {unexpected_keys}"
        
        # Validate missing keys - only graph_encoder keys are allowed to be missing
        for k in missing_keys:
            assert k.startswith("graph_encoder."), f"Missing unexpected key: {k}"
        
        logger.info(f"✅ Successfully loaded encoder weights from {ckpt_path}")

 
def gen_3d_conformation_from_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()

        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=8, pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=False)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=8)
        except:
            pass
        mol = Chem.RemoveHs(mol)
    except:
        return None, None
    if mol.GetNumConformers() == 0:
        return None, None

    if num_atoms != mol.GetNumAtoms():
        return None, None

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = np.array(mol.GetConformer().GetPositions())
    return atoms, coordinates


def gen_3d_conformation_from_openbabel(smiles):
    mol = pybel.readstring('smi', smiles)
    mol.make3D(forcefield='mmff94', steps=10000)
    mol.OBMol.DeleteHydrogens()

    atomic_nums = [atom.atomicnum for atom in mol.atoms]
    pt = Chem.GetPeriodicTable()
    atoms = [pt.GetElementSymbol(atomic_num) for atomic_num in atomic_nums]
    coordinates = np.array([atom.coords for atom in mol.atoms])
    return atoms, coordinates


def gen_3d_conformation_from_libraries(smiles):
    atoms, coordinates = gen_3d_conformation_from_rdkit(smiles)
    if atoms is None or coordinates is None:
        atoms, coordinates = gen_3d_conformation_from_openbabel(smiles)

    return atoms, coordinates


def get_mol_graphs(smiles_list, dictionary, device):
    data_graphs = defaultdict(list)
    for idx, smiles in enumerate(tqdm(smiles_list, desc='Processing Molecules...')):
        atoms, coordinates = gen_3d_conformation_from_libraries(smiles)

        if atoms is None or coordinates is None:
            logger.warning(f"Invalid SMILES for {idx}-th SMILES: {smiles}")
            continue

        data_graphs['unimol'].append(
            get_unimol_data(atoms, coordinates, dictionary, remove_Hs=True))

        graph = smiles2graph(smiles)
        data_graphs['moleculestm'].append(Data(x=graph['node_feat'], 
                                        edge_index=graph['edge_index'], 
                                        edge_attr=graph['edge_feat']))

    d3_collater = Mol3DCollater(dictionary.pad())
    graph_batch = {}
    graph_batch['unimol'] = d3_collater(data_graphs['unimol'])
    graph_batch['moleculestm'] = Batch.from_data_list(data_graphs['moleculestm'])

    for key in graph_batch.keys():
        if key == 'unimol':
            for key_ in graph_batch[key].keys():
                graph_batch[key][key_] = graph_batch[key][key_].to(device)
        elif key == 'moleculestm':
            graph_batch[key] = graph_batch[key].to(device)
        
    return graph_batch