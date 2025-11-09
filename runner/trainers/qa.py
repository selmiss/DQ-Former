import os
from typing import Any, Dict, Optional, Union, Tuple, List
import json
import re
from collections import defaultdict

import torch
from torch import optim, nn
from transformers import (
    BertTokenizerFast, 
    AutoModelForCausalLM,
    Trainer,
    get_cosine_schedule_with_warmup,
)
from transformers.utils import logging
from peft import LoraConfig, get_peft_model, TaskType
from torch_geometric.data import Batch

from models.mol_llama import MolLLaMA, DQMolLLaMA

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np

logger = logging.get_logger(__name__)


class MoleculeQATrainer(Trainer):
    """Trainer for molecule QA tasks (multiple choice questions)."""
    def __init__(self, vocab_size, model_config, train_config, tokenizer, use_dq_encoder=False, torch_dtype=None, **kwargs):
        self.train_config = train_config
        self.tokenizer = tokenizer
        
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        logger.info(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_baseline', False):
            # LLM baseline - only use language model without molecular encoders
            logger.info("Using LLM baseline: ", train_config.llm_model_path)
            # Use AutoModelForCausalLM to support Gemma, Qwen, Mistral, LLaMA, etc.
            if train_config.enable_flash:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                    )
                    logger.info("Using flash attention for LLM baseline")
                except TypeError:
                    # Some architectures may not accept attn_implementation
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
            
            model.resize_token_embeddings(vocab_size)
            
            # Apply LoRA if not freezing LLM
            if not getattr(train_config, 'freeze_llm', False):
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_config.llm_config.lora_config.r,
                    lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                    lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                )
                model = get_peft_model(model, peft_config)
                logger.info("Applied LoRA to LLM baseline")
            
            self.is_llm_baseline = True
        elif use_dq_encoder:
            self.is_llm_baseline = False
            if hasattr(train_config, 'llm_model_path'):
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = DQMolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = getattr(train_config, 'enable_blending', False),
            )
        else:
            self.is_llm_baseline = False
            model_config.graph_encoder_config.encoder_types = ['unimol', 'moleculestm']
            if hasattr(train_config, 'llm_model_path'):
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = MolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
            ).from_pretrained(train_config.ckpt_path)

        self.test_step_outputs = []
        
        # Initialize parent Trainer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def _get_eos_token_ids(self):
        ids = []
        try:
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                ids.append(self.tokenizer.eos_token_id)
        except Exception:
            pass
        # Try a few common end-of-turn markers across model families
        for tok in ["<|eot_id|>", "<eos_token>", "<end_of_turn>", "<|endoftext|>", "<eos>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        # De-duplicate while preserving order
        ids = list(dict.fromkeys(ids))
        return ids if len(ids) > 0 else None

    def load_from_ckpt(self, ckpt_path):
        if hasattr(self.model, 'load_from_ckpt'):
            self.model.load_from_ckpt(ckpt_path)
        else:
            # Load checkpoint manually
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        Override to use custom optimizer/scheduler from train_config.
        """
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.train_config.init_lr, 
                weight_decay=self.train_config.weight_decay
            )
            self.optimizer = optimizer
        
        if self.lr_scheduler is None:
            if self.train_config.scheduler == 'linear_warmup_cosine_lr':
                warmup_steps = min(num_training_steps, self.train_config.warmup_steps)
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif self.train_config.scheduler == 'None':
                self.lr_scheduler = None
        
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for training.
        
        Args:
            model: The model to compute loss for
            inputs: Dict with 'graph_batch', 'text_batch', 'brics_gids', 'entropy_gids', 'other_infos'
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        """
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        # brics_gids and entropy_gids are now in graph_batch and will be extracted by the model
        
        if self.is_llm_baseline:
            # For LLM baseline, only use text_batch for forward pass
            output = model(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                labels=text_batch.input_ids
            )
            loss = output.loss
        else:
            # Standard molecular + text training
            output = model(graph_batch, text_batch)
            loss = output['loss'] if isinstance(output, dict) else output.loss
        
        return (loss, output) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation/prediction step.
        """
        has_labels = "labels" in inputs or "text_batch" in inputs
        inputs = self._prepare_inputs(inputs)
        
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        brics_gids = inputs.get('brics_gids', None)
        entropy_gids = inputs.get('entropy_gids', None)
        other_infos = inputs.get('other_infos', {})
        
        with torch.no_grad():
            if self.is_llm_baseline:
                # For LLM baseline, only use text for generation
                eos_ids = self._get_eos_token_ids()
                gen_kwargs = {
                    'input_ids': text_batch.input_ids,
                    'attention_mask': text_batch.attention_mask,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'max_new_tokens': 512,
                    'do_sample': True,
                    'temperature': 0.7,
                }
                if eos_ids is not None:
                    gen_kwargs['eos_token_id'] = eos_ids
                responses = model.generate(**gen_kwargs)
            else:
                # Standard molecular + text generation
                responses = model.generate(
                    graph_batch, 
                    text_batch,
                    pad_token_id = self.tokenizer.pad_token_id,
                    eos_token_id = [self.tokenizer.eos_token_id],
                    brics_gids = brics_gids,
                    entropy_gids = entropy_gids,
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
            
            if len(no_format_indices) > 0 and not self.is_llm_baseline:
                new_graph_batch = {"unimol": {}, "moleculestm": {}}
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
                ).to(self.args.device)
                new_text_batch.mol_token_flag = (new_text_batch.input_ids == self.tokenizer.mol_token_id).to(self.args.device)

                new_responses = model.generate(
                    new_graph_batch, 
                    new_text_batch,
                    pad_token_id = self.tokenizer.pad_token_id,
                    eos_token_id = [self.tokenizer.eos_token_id],
                    brics_gids = brics_gids,
                    entropy_gids = entropy_gids,
                )
                new_generated_texts = self.tokenizer.batch_decode(new_responses, skip_special_tokens=True)

                for _, i in enumerate(no_format_indices):
                    generated_texts[i] += "\n\nAnswer: " + new_generated_texts[_]

            # Store outputs for metrics computation
            for response, answer, task in zip(generated_texts, other_infos['answer'], other_infos['task']):
                self.test_step_outputs.append({
                    'response': response,
                    'answer': answer,
                    'task': task
                })
        
        loss = None
        if not prediction_loss_only and has_labels:
            with torch.no_grad():
                loss = self.compute_loss(model, inputs, return_outputs=False)
        
        return (loss, None, None)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute custom metrics after prediction loop.
        """
        self.test_step_outputs = []  # Reset outputs
        
        # Run standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Gather outputs from all processes
        if self.args.local_rank in [-1, 0]:
            corrects, results = self.compute_metrics_qa(self.test_step_outputs)
            
            # Save results
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)
            
            # Add accuracy to output metrics
            if 'overall' in corrects:
                output[f"{metric_key_prefix}_accuracy"] = corrects['overall']['accuracy']
        
        return output
    
    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """
        Override predict to compute custom metrics after prediction loop.
        """
        self.test_step_outputs = []  # Reset outputs
        
        # Run standard prediction
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        # Gather outputs from all processes
        if self.args.local_rank in [-1, 0]:
            corrects, results = self.compute_metrics_qa(self.test_step_outputs)
            
            # Save results
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)
        
        return output
                

    def compute_metrics_qa(self, outputs):
        """Compute metrics for QA task (multiple choice)."""
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
                'correct': correct,
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


class MoleculeGENQATrainer(Trainer):
    """Trainer for molecule generation/captioning tasks."""
    def __init__(self, vocab_size, model_config, train_config, tokenizer, use_dq_encoder=False, torch_dtype=None, **kwargs):
        self.train_config = train_config
        self.tokenizer = tokenizer
        
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        logger.info(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_baseline', False):
            logger.info("Using LLM baseline: ", train_config.llm_model_path)
            if train_config.enable_flash:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                    )
                    logger.info("Using flash attention for LLM baseline")
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
            model.resize_token_embeddings(vocab_size)
            if not getattr(train_config, 'freeze_llm', False):
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_config.llm_config.lora_config.r,
                    lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                    lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                )
                model = get_peft_model(model, peft_config)
                logger.info("Applied LoRA to LLM baseline")
            self.is_llm_baseline = True
        elif use_dq_encoder:
            self.is_llm_baseline = False
            if hasattr(train_config, 'llm_model_path'):
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = DQMolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = getattr(train_config, 'enable_blending', False),
            )
        else:
            self.is_llm_baseline = False
            if hasattr(train_config, 'llm_model_path'):
                model_config.llm_config.llm_model = train_config.llm_model_path
            # Align encoder types with MoleculeQATrainer defaults
            if hasattr(model_config, 'graph_encoder_config'):
                model_config.graph_encoder_config.encoder_types = ['unimol', 'moleculestm']
            model = MolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
            )

        self.test_step_outputs = []
        
        # Initialize parent Trainer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def load_from_ckpt(self, ckpt_path):
        if hasattr(self.model, 'load_from_ckpt'):
            self.model.load_from_ckpt(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and scheduler for HF Trainer."""
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.train_config.init_lr, 
                weight_decay=self.train_config.weight_decay
            )
            self.optimizer = optimizer
        
        if self.lr_scheduler is None:
            if self.train_config.scheduler == 'linear_warmup_cosine_lr':
                warmup_steps = min(num_training_steps, self.train_config.warmup_steps)
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif self.train_config.scheduler == 'None':
                self.lr_scheduler = None
        
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for training."""
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        # brics_gids and entropy_gids are now in graph_batch and will be extracted by the model
        
        if getattr(self, 'is_llm_baseline', False):
            output = model(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                labels=text_batch.input_ids
            )
            loss = output.loss
        else:
            output = model(graph_batch, text_batch)
            loss = output['loss'] if isinstance(output, dict) else output.loss

        return (loss, output) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation/prediction step for generation."""
        has_labels = "labels" in inputs or "text_batch" in inputs
        inputs = self._prepare_inputs(inputs)
        
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        brics_gids = inputs.get('brics_gids', None)
        entropy_gids = inputs.get('entropy_gids', None)
        other_infos = inputs.get('other_infos', {})
        
        with torch.no_grad():
            if getattr(self, 'is_llm_baseline', False):
                eos_ids = self._get_eos_token_ids()
                gen_kwargs = {
                    'input_ids': text_batch.input_ids,
                    'attention_mask': text_batch.attention_mask,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'max_new_tokens': 512,
                    'do_sample': True,
                    'temperature': 0.7,
                }
                if eos_ids is not None:
                    gen_kwargs['eos_token_id'] = eos_ids
                responses = model.generate(**gen_kwargs)
            else:
                responses = model.generate(
                    graph_batch,
                    text_batch,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.eos_token_id],
                    brics_gids=brics_gids,
                    entropy_gids=entropy_gids,
                )
            generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
                self.test_step_outputs.append({
                    'prediction': pred_text,
                    'ground_truth': gt_text,
                })
        
        loss = None
        if not prediction_loss_only and has_labels:
            with torch.no_grad():
                loss = self.compute_loss(model, inputs, return_outputs=False)
        
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to compute custom metrics."""
        self.test_step_outputs = []
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.local_rank in [-1, 0]:
            metrics, per_sample = self.compute_metrics_gen(self.test_step_outputs)
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_caption_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_caption_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            
            # Add metrics to output
            for k, v in metrics.items():
                output[f"{metric_key_prefix}_{k}"] = v
        
        return output

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """Override predict to compute custom metrics."""
        self.test_step_outputs = []
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.local_rank in [-1, 0]:
            metrics, per_sample = self.compute_metrics_gen(self.test_step_outputs)
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_caption_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_caption_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output
                

    def compute_metrics_gen(self, outputs):
        """Compute metrics for generation/captioning task."""
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

    def _get_eos_token_ids(self):
        ids = []
        try:
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                ids.append(self.tokenizer.eos_token_id)
        except Exception:
            pass
        for tok in ["<|eot_id|>", "<eos_token>", "<end_of_turn>", "<|endoftext|>", "<eos>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        ids = list(dict.fromkeys(ids))
        return ids if len(ids) > 0 else None


class MoleculePropertyQATrainer(Trainer):
    """Trainer for molecule property prediction/regression tasks."""
    def __init__(self, vocab_size, model_config, train_config, tokenizer, use_dq_encoder=False, torch_dtype=None, **kwargs):
        self.train_config = train_config
        self.tokenizer = tokenizer
        
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        logger.info(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_baseline', False):
            logger.info("Using LLM baseline: ", train_config.llm_model_path)
            if train_config.enable_flash:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                    )
                    logger.info("Using flash attention for LLM baseline")
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
            model.resize_token_embeddings(vocab_size)
            if not getattr(train_config, 'freeze_llm', False):
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_config.llm_config.lora_config.r,
                    lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                    lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                )
                model = get_peft_model(model, peft_config)
                logger.info("Applied LoRA to LLM baseline")
            self.is_llm_baseline = True
        elif use_dq_encoder:
            self.is_llm_baseline = False
            if hasattr(train_config, 'llm_model_path'):
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = DQMolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = getattr(train_config, 'enable_blending', False),
            )
        else:
            self.is_llm_baseline = False
            if hasattr(train_config, 'llm_model_path'):
                model_config.llm_config.llm_model = train_config.llm_model_path
            if hasattr(model_config, 'graph_encoder_config'):
                model_config.graph_encoder_config.encoder_types = ['unimol', 'moleculestm']
            model = MolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
            )

        self.test_step_outputs = []
        
        # Initialize parent Trainer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def load_from_ckpt(self, ckpt_path):
        if hasattr(self.model, 'load_from_ckpt'):
            self.model.load_from_ckpt(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and scheduler for HF Trainer."""
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.train_config.init_lr, 
                weight_decay=self.train_config.weight_decay
            )
            self.optimizer = optimizer
        
        if self.lr_scheduler is None:
            if self.train_config.scheduler == 'linear_warmup_cosine_lr':
                warmup_steps = min(num_training_steps, self.train_config.warmup_steps)
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif self.train_config.scheduler == 'None':
                self.lr_scheduler = None
        
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for training."""
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        # brics_gids and entropy_gids are now in graph_batch and will be extracted by the model
        
        if getattr(self, 'is_llm_baseline', False):
            output = model(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                labels=text_batch.input_ids
            )
            loss = output.loss
        else:
            output = model(graph_batch, text_batch)
            loss = output['loss'] if isinstance(output, dict) else output.loss

        return (loss, output) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation/prediction step for property prediction."""
        has_labels = "labels" in inputs or "text_batch" in inputs
        inputs = self._prepare_inputs(inputs)
        
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        brics_gids = inputs.get('brics_gids', None)
        entropy_gids = inputs.get('entropy_gids', None)
        other_infos = inputs.get('other_infos', {})
        
        with torch.no_grad():
            if getattr(self, 'is_llm_baseline', False):
                eos_ids = self._get_eos_token_ids()
                gen_kwargs = {
                    'input_ids': text_batch.input_ids,
                    'attention_mask': text_batch.attention_mask,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'max_new_tokens': 512,
                    'do_sample': True,
                    'temperature': 0.7,
                }
                if eos_ids is not None:
                    gen_kwargs['eos_token_id'] = eos_ids
                responses = model.generate(**gen_kwargs)
            else:
                responses = model.generate(
                    graph_batch,
                    text_batch,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.eos_token_id],
                    brics_gids=brics_gids,
                    entropy_gids=entropy_gids,
                )
            generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
                self.test_step_outputs.append({
                    'prediction': pred_text,
                    'ground_truth': gt_text,
                })
        
        loss = None
        if not prediction_loss_only and has_labels:
            with torch.no_grad():
                loss = self.compute_loss(model, inputs, return_outputs=False)
        
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to compute custom metrics."""
        self.test_step_outputs = []
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.local_rank in [-1, 0]:
            metrics, per_sample = self.compute_metrics_regression(self.test_step_outputs)
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_regression_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_regression_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            
            # Add MAE to output metrics
            if 'MAE' in metrics:
                output[f"{metric_key_prefix}_mae"] = metrics['MAE']
        
        return output

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """Override predict to compute custom metrics."""
        self.test_step_outputs = []
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.local_rank in [-1, 0]:
            metrics, per_sample = self.compute_metrics_regression(self.test_step_outputs)
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_regression_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_regression_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output
                

    def compute_metrics_regression(self, outputs):
        """Compute metrics for regression/property prediction task."""
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

    def _get_eos_token_ids(self):
        ids = []
        try:
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                ids.append(self.tokenizer.eos_token_id)
        except Exception:
            pass
        for tok in ["<|eot_id|>", "<eos_token>", "<end_of_turn>", "<|endoftext|>", "<eos>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        ids = list(dict.fromkeys(ids))
        return ids if len(ids) > 0 else None

