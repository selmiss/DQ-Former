import json
import os
from torch.utils.data import Dataset
from data_provider.mol_dataset import smiles2graph, get_unimol_data
from collections import defaultdict
from torch_geometric.data import Data, Batch
from data_provider.tokenization_utils import batch_tokenize_messages_list
from data_provider.collaters import Mol3DCollater
import numpy as np
from tqdm import tqdm
import warnings

class ZeroshotDataset(Dataset):
    def __init__(self, data_dir, split, prompt_type, 
                unimol_dictionary, only_llm=False, meta_filename=None, jsonl_filename=None, lazy_jsonl=True):
        super().__init__()

        # New format only: input is a directory containing a metadata JSON and a data JSONL
        self.only_llm = only_llm

        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")

        # Resolve meta and jsonl paths
        meta_path = None
        data_jsonl_path = None

        if meta_filename is not None:
            candidate = os.path.join(data_dir, meta_filename)
            if os.path.isfile(candidate):
                meta_path = candidate
            else:
                raise FileNotFoundError(f"Meta JSON not found: {candidate}")
        if jsonl_filename is not None:
            candidate = os.path.join(data_dir, jsonl_filename)
            if os.path.isfile(candidate):
                data_jsonl_path = candidate
            else:
                raise FileNotFoundError(f"Data JSONL not found: {candidate}")

        # Auto-discover if not provided
        if meta_path is None:
            preferred = ['meta.json', 'info.json', 'prompt.json']
            for name in preferred:
                candidate = os.path.join(data_dir, name)
                if os.path.isfile(candidate):
                    meta_path = candidate
                    break
            if meta_path is None:
                for name in os.listdir(data_dir):
                    if name.endswith('.json') and not name.endswith('.jsonl'):
                        meta_path = os.path.join(data_dir, name)
                        break
        if data_jsonl_path is None:
            preferred = ['data.jsonl', 'content.jsonl']
            for name in preferred:
                candidate = os.path.join(data_dir, name)
                if os.path.isfile(candidate):
                    data_jsonl_path = candidate
                    break
            if data_jsonl_path is None:
                for name in os.listdir(data_dir):
                    if name.endswith('.jsonl'):
                        data_jsonl_path = os.path.join(data_dir, name)
                        break

        if meta_path is None or data_jsonl_path is None:
            raise FileNotFoundError(f"Could not locate both meta JSON and data JSONL in directory: {data_dir}")

        # Load meta
        meta = json.load(open(meta_path, 'r'))
        self.prompts = meta['prompts'][prompt_type]
        self.split = meta['split']
        if 'positive_label' in meta and 'negative_label' in meta:
            self.positive_label = meta['positive_label'].lower()
            self.negative_label = meta['negative_label'].lower()
        else:
            self.positive_label = 'positive'
            self.negative_label = 'negative'
            warnings.warn(f"Positive and negative labels not found in meta.json, using 'positive' and 'negative' as default.")

        target_indices = self.split[split]

        # Build index for JSONL
        if lazy_jsonl:
            self.jsonl_path = data_jsonl_path
            target_set = set(target_indices)
            index_to_offset = {}
            print('load data from:', data_jsonl_path)
            with open(data_jsonl_path, 'rb') as f:
                offset = f.tell()
                for idx, _ in enumerate(f):
                    if idx in target_set:
                        index_to_offset[idx] = offset
                    offset = f.tell()
            missing = [i for i in target_indices if i not in index_to_offset]
            if missing:
                raise ValueError(f"Some split indices are out of range for JSONL file: {missing[:5]}")
            self._selected_offsets = [index_to_offset[i] for i in target_indices]
            self._length = len(self._selected_offsets)
            self.use_lazy_jsonl = True
        else:
            records = []
            print('load data from:', data_jsonl_path)
            with open(data_jsonl_path, 'r') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            self.data_list = [records[i] for i in target_indices]
            self._length = len(self.data_list)
            self.use_lazy_jsonl = False

        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"

        self.unimol_dictionary = unimol_dictionary

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if getattr(self, 'use_lazy_jsonl', False):
            offset = self._selected_offsets[idx]
            with open(self.jsonl_path, 'rb') as f:
                f.seek(offset)
                line = f.readline()
            data = json.loads(line.decode('utf-8'))
        else:
            data = self.data_list[idx]
        
        data_graphs = defaultdict(list)
        data_graphs['unimol'].append(
            get_unimol_data(data['atoms'], np.array(data['coordinates'][0]), self.unimol_dictionary))

        graphs = smiles2graph(data['smiles'])

        data_graphs['moleculestm'].append(
            Data(x=graphs['node_feat'], 
                edge_index=graphs['edge_index'], 
                edge_attr=graphs['edge_feat'])
        )

        # Prepare the prompt
        system_prompt = self.prompts['system']
        if self.only_llm:
            user_prompt = self.prompts['user'].replace('<mol>', data['smiles'])
        else:
            user_prompt = self.prompts['user'].replace('<mol>', self.mol_prompt)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return data_graphs, messages, data['answer'], data['smiles'], data['brics_gids'] if 'brics_gids' in data else None, data['entropy_gids'] if 'entropy_gids' in data else None

class ZeroshotCollater():
    def __init__(self, tokenizer, unimol_dictionary, llm_version, only_llm=False):
        self.tokenizer = tokenizer
        self.llm_version = llm_version
        self.d3_collater = Mol3DCollater(unimol_dictionary.pad())
        self.only_llm = only_llm
    def __call__(self, batch):
        data_graphs, messages_list, answers, smiles, brics_gids, entropy_gids = zip(*batch)

        graph_batch = {}
        data_unimol = []
        for data in data_graphs:
            data_unimol.extend(data['unimol'])
        graph_batch['unimol'] = self.d3_collater(data_unimol)
        data_moleculestm = []
        for data in data_graphs:
            data_moleculestm.extend(data['moleculestm'])
        graph_batch['moleculestm'] = Batch.from_data_list(data_moleculestm)

        if self.only_llm:
            # Use native tokenizer chat template for standard LLM mode
            # This avoids needing mol_token_id and uses the model's built-in formatting
            # Benefits:
            # 1. No modification to pretrained tokenizer vocabulary
            # 2. Uses model's official chat template (e.g., Llama3, Qwen, etc.)
            # 3. Ensures compatibility with model's expected input format
            text_inputs = []
            for messages in messages_list:
                # Apply chat template to format messages (system, user, assistant)
                text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                text_inputs.append(text)
            
            # Batch tokenize with left padding (for batch generation)
            # Note: padding_side='left' should already be set in inference.py
            tokenized = self.tokenizer(
                text_inputs,
                padding='longest',
                truncation=False,
                return_tensors='pt',
                add_special_tokens=False,  # Template already added them
            )
            # Add dummy labels for consistency (not used in generation)
            tokenized['labels'] = tokenized['input_ids'].clone()
            text_batch = tokenized
        else:
            tokenized = batch_tokenize_messages_list(messages_list, self.tokenizer, 
                                                    self.llm_version, padding_side='left')
            text_batch = tokenized

        return graph_batch, text_batch, answers, smiles, brics_gids, entropy_gids

    