import os
from typing import Any, Dict
import json
import re
from collections import defaultdict

import torch
from torch import optim
import pytorch_lightning as pl
from transformers import AutoTokenizer, BertTokenizerFast, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch_geometric.data import Batch

from models.mol_llama import MolLLaMA, DQMolLLaMA
from trainer.optims import LinearWarmupCosineLRScheduler



from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class MoleculeQATrainer(pl.LightningModule):
    def __init__(self, vocab_size, model_config, train_config, use_dq_encoder=False, torch_dtype=None):
        super().__init__()
        self.train_config = train_config
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        print(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_baseline', False):
            # LLM baseline - only use language model without molecular encoders
            print("Using LLM baseline (text-only)")
            if train_config.enable_flash:
                self.model = LlamaForCausalLM.from_pretrained(
                    train_config.llm_model_path, 
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2"
                )
                print("Using flash attention for LLM baseline")
            else:
                self.model = LlamaForCausalLM.from_pretrained(
                    train_config.llm_model_path, 
                    torch_dtype=torch_dtype
                )
            
            self.model.resize_token_embeddings(vocab_size)
            
            # Apply LoRA if not freezing LLM
            if not train_config.freeze_llm:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_config.llm_config.lora_config.r,
                    lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                    lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                )
                self.model = get_peft_model(self.model, peft_config)
                print("Applied LoRA to LLM baseline")
            
            self.is_llm_baseline = True
        elif use_dq_encoder:
            self.is_llm_baseline = False
            self.model = DQMolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = train_config.freeze_llm,
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = train_config.enable_blending,
            )
        else:
            self.is_llm_baseline = False
            model_config.graph_encoder_config.encoder_types = ['unimol', 'moleculestm']
            self.model = MolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = train_config.freeze_llm,
            ).from_pretrained(train_config.ckpt_path)

        self.test_step_outputs = []

    def load_from_ckpt(self, ckpt_path):
        self.model.load_from_ckpt(ckpt_path)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = text_batch.input_ids.size(0)
        ###============== Overall Loss ===================###
        
        if self.is_llm_baseline:
            # For LLM baseline, only use text_batch for forward pass
            output = self.model(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                labels=text_batch.input_ids
            )
            loss = output.loss
        else:
            # Standard molecular + text training
            output = self.model(graph_batch, text_batch, other_infos)
            loss = output['loss']
            
        # Show step-wise loss on the progress bar without cross-rank reduction
        self.log("train_loss", loss, batch_size=batch_size, sync_dist=False, logger=True, prog_bar=True, on_step=True, on_epoch=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=False, on_step=True, on_epoch=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        
        if self.is_llm_baseline:
            # For LLM baseline, only use text for generation
            responses = self.model.generate(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
        else:
            # Standard molecular + text generation
            responses = self.model.generate(
                graph_batch, 
                text_batch,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = [self.tokenizer.eos_token_id],
                brics_gids = other_infos['brics_gids'],
                entropy_gids = other_infos['entropy_gids'],
            )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        original_texts = self.tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)
        pattern = r"[Aa]nswer:"

        # Generate further if the output does not contain "Answer:"
        no_format_indices = []
        new_texts = []
        for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
            if not re.search(pattern, generated_text):
                no_format_indices.append(idx)
                new_texts.append(original_text + generated_text + "\n\nAnswer: ")
        if len(no_format_indices) > 0:
            new_graph_batch = {"unimol": {}, "moleculestm": {}}
            # new_graph_batch = {"unimol": {}}
            new_text_batch = {}
            for k, v in graph_batch['unimol'].items():
                new_graph_batch['unimol'][k] = v[no_format_indices]
            new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))

            new_text_batch = self.tokenizer(
                new_texts,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(self.device)
            new_text_batch.mol_token_flag = (new_text_batch.input_ids == self.tokenizer.mol_token_id).to(self.device)

            new_responses = self.model.generate(
                new_graph_batch, 
                new_text_batch,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = [self.tokenizer.eos_token_id],
                brics_gids = other_infos['brics_gids'],
                entropy_gids = other_infos['entropy_gids'],
            )
            new_generated_texts = self.tokenizer.batch_decode(new_responses, skip_special_tokens=True)

            for _, i in enumerate(no_format_indices):
                generated_texts[i] += "\n\nAnswer: " + new_generated_texts[_]



        for response, answer, task in zip(generated_texts, other_infos['answer'], other_infos['task']):
            self.test_step_outputs.append({
                'response': response,
                'answer': answer,
                'task': task
            })

        return responses

    def on_validation_epoch_end(self):
        outputs = self.test_step_outputs

        world_size = getattr(self.trainer, "world_size", 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, outputs)
            all_outputs = [item for sublist in gathered for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            corrects, results = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_test_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_test_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        
        if self.is_llm_baseline:
            # For LLM baseline, only use text for generation
            responses = self.model.generate(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.tokenizer.eos_token_id],
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
        else:
            # Standard molecular + text generation
            responses = self.model.generate(
                graph_batch, 
                text_batch,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = [self.tokenizer.eos_token_id],
                brics_gids = other_infos['brics_gids'],
                entropy_gids = other_infos['entropy_gids'],
            )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        original_texts = self.tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)
        pattern = r"[Aa]nswer:"

        # Generate further if the output does not contain "Answer:"
        no_format_indices = []
        new_texts = []
        for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
            if not re.search(pattern, generated_text):
                no_format_indices.append(idx)
                new_texts.append(original_text + generated_text + "\n\nAnswer: ")
        if len(no_format_indices) > 0:
            new_graph_batch = {"unimol": {}, "moleculestm": {}}
            # new_graph_batch = {"unimol": {}}
            new_text_batch = {}
            for k, v in graph_batch['unimol'].items():
                new_graph_batch['unimol'][k] = v[no_format_indices]
            new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))

            new_text_batch = self.tokenizer(
                new_texts,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(self.device)
            new_text_batch.mol_token_flag = (new_text_batch.input_ids == self.tokenizer.mol_token_id).to(self.device)

            new_responses = self.model.generate(
                new_graph_batch, 
                new_text_batch,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = [self.tokenizer.eos_token_id],
                brics_gids = other_infos['brics_gids'],
                entropy_gids = other_infos['entropy_gids'],
            )
            new_generated_texts = self.tokenizer.batch_decode(new_responses, skip_special_tokens=True)

            for _, i in enumerate(no_format_indices):
                generated_texts[i] += "\n\nAnswer: " + new_generated_texts[_]



        for response, answer, task in zip(generated_texts, other_infos['answer'], other_infos['task']):
            self.test_step_outputs.append({
                'response': response,
                'answer': answer,
                'task': task
            })

        return responses

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        world_size = getattr(self.trainer, "world_size", 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, outputs)
            all_outputs = [item for sublist in gathered for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            corrects, results = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, "test_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(self.logger.log_dir, "test_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)
                

    def compute_metrics(self, outputs):
        results = defaultdict(list)
        corrects = {}

        for output in outputs:
            task = output['task']
            response = output['response']
            answer = output['answer'].replace("Answer: ", "")
            prediction = response.split("Answer: ")[-1].strip()
            if 'A' in prediction:
                prediction = 'A'
            elif 'B' in prediction:
                prediction = 'B'
            elif 'C' in prediction:
                prediction = 'C'
            elif 'D' in prediction:
                prediction = 'D'
            else:
                prediction = 'None'

            correct = 1 if prediction == answer else 0
            results[task].append({
                'response': response,
                'answer': answer,
                'prediction': prediction,
                'correct': correct
            })

            if task not in corrects:
                corrects[task] = {'correct': 0, 'total': 0}
            corrects[task]['correct'] += correct
            corrects[task]['total'] += 1

        tasks = list(corrects.keys())
        for task in tasks:
            correct = corrects[task]['correct']
            total = corrects[task]['total']
            accuracy = correct / total * 100
            corrects[task]['accuracy'] = accuracy

        # Calculate overall accuracy
        overall_correct = sum([corrects[task]['correct'] for task in tasks])
        overall_total = sum([corrects[task]['total'] for task in tasks])
        overall_accuracy = overall_correct / overall_total * 100
        corrects['overall'] = {'correct': overall_correct, 'total': overall_total, 'accuracy': overall_accuracy}

        return corrects, results

class MoleculeGENQATrainer(pl.LightningModule):
    def __init__(self, vocab_size, model_config, train_config, use_dq_encoder=False, torch_dtype=None):
        super().__init__()
        self.train_config = train_config
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        print(f"use_dq_encoder: {use_dq_encoder}")

        if use_dq_encoder:
            self.model = DQMolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = train_config.freeze_llm,
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = train_config.enable_blending,
            )
        else:
            self.model = MolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = train_config.freeze_llm,
            )

        self.test_step_outputs = []

    def load_from_ckpt(self, ckpt_path):
        self.model.load_from_ckpt(ckpt_path)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = text_batch.input_ids.size(0)
        ###============== Overall Loss ===================###
        output = self.model(graph_batch, text_batch, other_infos)
        loss = output['loss']

        self.log("train_loss", loss, batch_size=batch_size, sync_dist=False, logger=True, prog_bar=True, on_step=True, on_epoch=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=False, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        responses = self.model.generate(
            graph_batch,
            text_batch,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            brics_gids=other_infos['brics_gids'],
            entropy_gids=other_infos['entropy_gids'],
        )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
            self.test_step_outputs.append({
                'prediction': pred_text,
                'ground_truth': gt_text,
            })

        return responses

    def on_validation_epoch_end(self):
        outputs = self.test_step_outputs

        world_size = getattr(self.trainer, "world_size", 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, outputs)
            all_outputs = [item for sublist in gathered for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            metrics, per_sample = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_caption_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_caption_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        responses = self.model.generate(
            graph_batch,
            text_batch,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            brics_gids=other_infos['brics_gids'],
            entropy_gids=other_infos['entropy_gids'],
        )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
            self.test_step_outputs.append({
                'prediction': pred_text,
                'ground_truth': gt_text,
            })

        return responses

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        world_size = getattr(self.trainer, "world_size", 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, outputs)
            all_outputs = [item for sublist in gathered for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            metrics, per_sample = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, "caption_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(self.logger.log_dir, "caption_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                

    def compute_metrics(self, outputs):
        # Prepare pairs
        pairs = [(o['ground_truth'], o['prediction']) for o in outputs]
        per_sample = [{'ground_truth': gt, 'prediction': pred} for gt, pred in pairs]

        # Tokenizer as in reference implementation
        text_tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')

        references = []
        hypotheses = []
        meteor_scores = []

        for gt, pred in pairs:
            gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=512, padding='max_length')
            gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

            pred_tokens = text_tokenizer.tokenize(pred, truncation=True, max_length=512, padding='max_length')
            pred_tokens = list(filter(('[PAD]').__ne__, pred_tokens))
            pred_tokens = list(filter(('[CLS]').__ne__, pred_tokens))
            pred_tokens = list(filter(('[SEP]').__ne__, pred_tokens))

            references.append([gt_tokens])
            hypotheses.append(pred_tokens)

            mscore = meteor_score([gt_tokens], pred_tokens)
            meteor_scores.append(mscore)

        bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores = []
        for gt, pred in pairs:
            rs = scorer.score(pred, gt)
            rouge_scores.append(rs)

        rouge_1 = float(np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]))
        rouge_2 = float(np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]))
        rouge_l = float(np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]))
        meteor_avg = float(np.mean(meteor_scores)) if len(meteor_scores) > 0 else 0.0

        metrics = {
            'BLEU-2': float(bleu2),
            'BLEU-4': float(bleu4),
            'ROUGE-1': rouge_1,
            'ROUGE-2': rouge_2,
            'ROUGE-L': rouge_l,
            'METEOR': meteor_avg,
        }

        return metrics, per_sample

class MoleculePropertyQATrainer(pl.LightningModule):
    def __init__(self, vocab_size, model_config, train_config, use_dq_encoder=False, torch_dtype=None):
        super().__init__()
        self.train_config = train_config
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        print(f"use_dq_encoder: {use_dq_encoder}")

        if use_dq_encoder:
            self.model = DQMolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = train_config.freeze_llm,
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = train_config.enable_blending,
            )
        else:
            self.model = MolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = train_config.freeze_llm,
            )

        self.test_step_outputs = []

    def load_from_ckpt(self, ckpt_path):
        self.model.load_from_ckpt(ckpt_path)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = text_batch.input_ids.size(0)
        ###============== Overall Loss ===================###
        output = self.model(graph_batch, text_batch, other_infos)
        loss = output['loss']

        self.log("train_loss", loss, batch_size=batch_size, sync_dist=False, logger=True, prog_bar=True, on_step=True, on_epoch=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=False, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        responses = self.model.generate(
            graph_batch,
            text_batch,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            brics_gids=other_infos['brics_gids'],
            entropy_gids=other_infos['entropy_gids'],
        )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
            self.test_step_outputs.append({
                'prediction': pred_text,
                'ground_truth': gt_text,
            })

        return responses

    def on_validation_epoch_end(self):
        outputs = self.test_step_outputs

        world_size = getattr(self.trainer, "world_size", 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, outputs)
            all_outputs = [item for sublist in gathered for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            metrics, per_sample = self.compute_metrics(all_outputs)
            # Log MAE for validation (avoid distributed sync inside rank-0-only block)
            if 'MAE' in metrics:
                self.log("val_mae", float(metrics['MAE']), prog_bar=True, sync_dist=False)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_regression_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_regression_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        responses = self.model.generate(
            graph_batch,
            text_batch,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            brics_gids=other_infos['brics_gids'],
            entropy_gids=other_infos['entropy_gids'],
        )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
            self.test_step_outputs.append({
                'prediction': pred_text,
                'ground_truth': gt_text,
            })

        return responses

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        world_size = getattr(self.trainer, "world_size", 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, outputs)
            all_outputs = [item for sublist in gathered for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            metrics, per_sample = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, "regression_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(self.logger.log_dir, "regression_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                

    def compute_metrics(self, outputs):
        # Extract last numeric value from prediction text and ground-truth text, compute MAE
        def _extract_last_number(text: Any):
            try:
                # If already numeric
                if isinstance(text, (int, float, np.number)):
                    return float(text)
                s = str(text)
                # Match floats/ints with optional sign and scientific notation
                matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                if not matches:
                    return None
                return float(matches[-1])
            except Exception:
                return None

        per_sample = []
        abs_errors = []
        total = 0
        valid = 0

        for o in outputs:
            gt_text = o['ground_truth']
            pred_text = o['prediction']
            gt_val = _extract_last_number(gt_text)
            pred_val = _extract_last_number(pred_text)
            total += 1
            record = {
                'ground_truth_text': gt_text,
                'prediction_text': pred_text,
                'ground_truth': gt_val,
                'prediction': pred_val,
            }
            per_sample.append(record)
            if gt_val is not None and pred_val is not None:
                abs_errors.append(abs(pred_val - gt_val))
                valid += 1

        mae = float(np.mean(abs_errors)) if len(abs_errors) > 0 else float('nan')
        metrics = {
            'MAE': mae,
            'valid_count': int(valid),
            'total_count': int(total),
        }

        return metrics, per_sample
