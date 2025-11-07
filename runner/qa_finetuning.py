"""
MoleculeQA Finetuning Script using HuggingFace Transformers.
Follows the same pattern as finetuning.py for consistency.

This script supports three types of MoleculeQA tasks:
- qa: Multiple choice question answering
- generation: Molecule captioning/description generation
- property: Property value prediction (regression)
"""

import os
import torch
import warnings
import argparse
import yaml
from easydict import EasyDict as edict
from typing import Dict, Optional
from collections import defaultdict

from transformers import (
    Trainer,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
import wandb

from peft import get_peft_model, LoraConfig, TaskType

from utils.configuration_mol_llama import MolLLaMAConfig
from runner.training_args import (
    ModelArguments,
    DataTrainingArguments,
    parse_args_from_yaml,
)
from data_provider.moleculeqa_dataset import create_moleculeqa_datasets
from data_provider.finetune_dataset import determine_llm_version

# Import trainers for metrics computation
from runner.trainers.qa import (
    MoleculeQATrainer,
    MoleculeGENQATrainer,
    MoleculePropertyQATrainer,
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
## for A100 gpus
torch.set_float32_matmul_precision("medium")


def main(model_args, training_args, data_config, test_mode=False, resume_from=None):
    """Main MoleculeQA finetuning function.
    
    Args:
        model_args: ModelArguments parsed by HfArgumentParser
        training_args: HuggingFace TrainingArguments parsed directly from YAML
        data_config: DataTrainingArguments parsed by HfArgumentParser
        test_mode: Whether to use small dataset for testing
        resume_from: Path to checkpoint to resume from
    """
    torch.manual_seed(0)
    
    # Get BASE_DIR for DeepSpeed config paths
    BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Initialize tokenizer
    if model_args.llm_backbone is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.llm_backbone, padding_side="left"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "DongkiKim/Mol-Llama-3.1-8B-Instruct",
            padding_side="left",
        )

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<mol>", "<graph>"]})
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    mol_id = tokenizer.convert_tokens_to_ids("<mol>")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

    # Create model config from parsed arguments
    model_config = MolLLaMAConfig(
        qformer_config={
            "use_flash_attention": model_args.use_flash_attention,
            "use_dq_encoder": model_args.use_dq_encoder,
            "num_query_tokens": model_args.num_query_tokens,
            "embed_dim": model_args.embed_dim,
            "cross_attention_freq": model_args.cross_attention_freq,
            "enable_lora": model_args.enable_lora_qformer,
        },
        graph_encoder_config={"local_q_only": model_args.local_q_only},
        blending_module_config={
            "num_layers": model_args.num_layers,
            "num_heads": model_args.num_heads,
            "enable_blending": model_args.enable_blending,
        },
    )
    
    if model_args.llm_backbone is not None:
        model_config.llm_config.llm_model = model_args.llm_backbone

    # Determine torch dtype from training_args
    if training_args.bf16:
        torch_dtype = "bfloat16"
    elif training_args.fp16:
        torch_dtype = "float16"
    else:
        torch_dtype = "float32"

    # Determine LLM version from model config
    llm_version = determine_llm_version(model_config.llm_config.llm_model, default="llama3")

    # Load unimol_dictionary first (before creating datasets)
    # Following the same pattern as DQ_former_encoder.py init_unimol_encoder
    print("Loading UniMol dictionary...")
    from huggingface_hub import hf_hub_download
    from utils.unicore import Dictionary
    
    unimol_config = model_config.graph_encoder_config.unimol_config
    unimol_dictionary_path = hf_hub_download(
        repo_id=unimol_config.repo_id,
        filename=unimol_config.dictionary_filename,
    )
    unimol_dictionary = Dictionary.load(unimol_dictionary_path)
    unimol_dictionary.add_symbol("[MASK]", is_special=True)
    print(f"✅ Loaded UniMol dictionary with {len(unimol_dictionary)} symbols")
    
    # Create datasets using the dictionary
    print("Creating MoleculeQA datasets...")
    
    # Determine limits for test mode
    train_limit = 100 if test_mode else None
    val_limit = 50 if test_mode else None
    test_limit = 50 if test_mode else None
    
    datasets, data_collator = create_moleculeqa_datasets(
        tokenizer=tokenizer,
        llama_version=llm_version,
        root=data_config.root,
        unimol_dictionary=unimol_dictionary,
        encoder_types=model_config.graph_encoder_config.encoder_types,
        mol_type=getattr(data_config, 'mol_type', 'mol'),
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=test_limit,
        brics_gids_enable=model_args.brics_gids_enable,
        entropy_gids_enable=model_args.entropy_gids_enable,
        use_cache=True,
    )
    
    train_dataset = datasets['train']
    val_dataset = datasets['test']
    test_dataset = datasets['test']
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Calculate total training steps for distributed training
    num_samples = len(train_dataset)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    samples_per_gpu = num_samples // num_gpus if num_gpus > 1 else num_samples
    steps_per_epoch = samples_per_gpu // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    max_steps = steps_per_epoch * training_args.num_train_epochs
    
    # Update max_steps
    training_args.max_steps = max_steps
    
    # Determine task type and use appropriate trainer
    task_type = getattr(data_config, 'task_type', 'qa')
    print(f"Task type: {task_type}")
    
    # Select appropriate trainer based on task type
    if task_type == 'generation':
        trainer_class = MoleculeGENQATrainer
    elif task_type == 'property':
        trainer_class = MoleculePropertyQATrainer
    else:  # default to 'qa'
        trainer_class = MoleculeQATrainer
    
    # Initialize WandB
    if training_args.local_rank in [-1, 0]:
        from dataclasses import asdict
        wandb.init(
            project="MoleculeQA-Finetuning",
            name=training_args.run_name,
            config={
                "training": training_args.to_dict(),
                "model": asdict(model_args),
                "data": asdict(data_config),
            },
            mode="offline",
        )
    
    # Initialize Trainer with task-specific config and actual datasets
    from easydict import EasyDict
    train_config = EasyDict({
        'init_lr': training_args.learning_rate,
        'weight_decay': training_args.weight_decay,
        'warmup_steps': training_args.warmup_steps,
        'max_epochs': training_args.num_train_epochs,
        'min_lr': 1e-7,
        'warmup_lr': 1e-6,
        'scheduler': 'linear_warmup_cosine_lr',
        'precision': 'bf16-mixed' if training_args.bf16 else ('fp16' if training_args.fp16 else 'fp32'),
        'enable_flash': model_args.enable_flash,
        'freeze_llm': model_args.freeze_llm,
        'brics_gids_enable': model_args.brics_gids_enable,
        'entropy_gids_enable': model_args.entropy_gids_enable,
        'enable_blending': model_args.enable_blending,
    })
    
    # Create trainer with actual datasets (no workaround needed!)
    trainer = trainer_class(
        vocab_size=len(tokenizer),
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        use_dq_encoder=model_args.use_dq_encoder,
        torch_dtype=torch_dtype,
        # HF Trainer arguments
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if training_args.eval_strategy != "no" else None,
        data_collator=data_collator,
    )
    
    # Load from pretrained checkpoint (Stage 2 full model) if provided
    # The trainer has already created the model, so we just load weights into it
    if model_args.model_name_or_path:
        print(f"Loading from pretrained checkpoint: {model_args.model_name_or_path}")
        trainer.load_from_ckpt(model_args.model_name_or_path)
    
    # Apply LoRA to Qformer if enabled (after checkpoint loading)
    if model_args.enable_lora_qformer:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=model_config.qformer_config.lora_config.r,
            lora_alpha=model_config.qformer_config.lora_config.lora_alpha,
            lora_dropout=model_config.qformer_config.lora_config.lora_dropout,
            target_modules=[
                'query', 'key', 'value', 'output.dense',
                'intermediate.dense', 'output.dense'
            ],
        )
        trainer.model.encoder.Qformer = get_peft_model(trainer.model.encoder.Qformer, peft_config)
        print("LoRA enabled for Qformer")
    
    # Check for existing checkpoints or resume_from parameter
    ckpt_path = None
    if resume_from is not None:
        if resume_from == "last":
            candidate = os.path.join(training_args.output_dir, "checkpoint-last")
            ckpt_path = candidate if os.path.exists(candidate) else None
            if ckpt_path is None:
                # Try get_last_checkpoint
                ckpt_path = get_last_checkpoint(training_args.output_dir)
        else:
            ckpt_path = resume_from if os.path.exists(resume_from) else None
    else:
        # Auto-detect last checkpoint
        if os.path.isdir(training_args.output_dir):
            ckpt_path = get_last_checkpoint(training_args.output_dir)
    
    # Train
    if ckpt_path is not None:
        print(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.train(resume_from_checkpoint=ckpt_path)
    else:
        print("No resuming checkpoint found, starting finetuning from model")
        trainer.train()
    
    # Evaluate on validation set if available
    if val_dataset is not None and len(val_dataset) > 0:
        print("\nEvaluating on validation set...")
        eval_results = trainer.evaluate()
        print(f"Validation results: {eval_results}")
    
    # Test on test set
    if test_dataset is not None and len(test_dataset) > 0:
        print("\nTesting on test set...")
        test_results = trainer.predict(test_dataset)
        print(f"Test results saved to {training_args.output_dir}")
    
    # Save final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    
    # Close WandB
    if training_args.local_rank in [-1, 0]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoleculeQA Finetuning with HF Transformers")
    parser.add_argument(
        "--model_config_path", type=str, default="configs/moleculeqa/dqw2d/model_config.yaml",
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--training_config_path", type=str, default="configs/moleculeqa/dqw2d/training_config.yaml",
        help="Path to training configuration YAML file (HF TrainingArguments compatible)"
    )
    parser.add_argument(
        "--data_config_path", type=str, default="configs/moleculeqa/dqw2d/data_config.yaml",
        help="Path to data configuration YAML file"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help="Use small dataset for testing",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help='Checkpoint path or "last" to resume from latest',
    )
    parser.add_argument(
        "--deepspeed_stage",
        type=int,
        default=2,
        choices=[2, 3],
        help="DeepSpeed ZeRO stage (2 or 3)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed launcher)",
    )

    args = parser.parse_args()
    
    # Get BASE_DIR for paths
    BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Determine output directory from training config
    with open(args.training_config_path, 'r') as f:
        training_config_preview = yaml.load(f, Loader=yaml.FullLoader)
    
    # Setup output directory
    run_name = training_config_preview.get('run_name', 'default_run')
    output_dir = f"checkpoints/{run_name}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup DeepSpeed config based on command-line argument
    deepspeed_config = None
    if args.deepspeed_stage == 3:
        deepspeed_config = os.path.join(BASE_DIR, "configs/deepspeed/ds_config_zero3.json")
    else:
        deepspeed_config = os.path.join(BASE_DIR, "configs/deepspeed/ds_config_zero2.json")
    
    if not os.path.exists(deepspeed_config):
        print(f"Warning: DeepSpeed config not found at {deepspeed_config}")
        deepspeed_config = None
    else:
        print(f"Using DeepSpeed ZeRO-{args.deepspeed_stage} config: {deepspeed_config}")

    # Parse arguments from YAML files using HfArgumentParser
    model_args, training_args, data_config = parse_args_from_yaml(
        model_config_path=args.model_config_path,
        data_config_path=args.data_config_path,
        training_config_path=args.training_config_path,
        output_dir=output_dir,
        deepspeed_config=deepspeed_config,
    )

    print("-" * 60)
    detected_num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"batch_size: {training_args.per_device_train_batch_size}\tnum_devices: {detected_num_devices}\taccumulate_grad_batches: {training_args.gradient_accumulation_steps}"
    )
    print(
        f"Total batch size: {training_args.per_device_train_batch_size * detected_num_devices * training_args.gradient_accumulation_steps}"
    )
    print("-" * 60)
    
    task_type = getattr(data_config, 'task_type', 'qa')
    mol_type = getattr(data_config, 'mol_type', 'mol')
    print(f"Task Type: {task_type}")
    print(f"Mol Type: {mol_type}")
    print(f"Data Root: {data_config.root}")
    print("-" * 60)

    if args.test_mode:
        print("TEST MODE: Using small dataset for quick testing")
    
    main(
        model_args,
        training_args,
        data_config,
        test_mode=args.test_mode,
        resume_from=args.resume_from,
    )

