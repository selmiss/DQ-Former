import os
import torch
import warnings
import argparse
import yaml
from dataclasses import dataclass, field, asdict
from easydict import EasyDict as edict
from typing import Dict, Optional
from collections import defaultdict

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
import wandb

from peft import get_peft_model, LoraConfig, TaskType

from utils.configuration_mol_llama import MolLLaMAConfig
from trainer.finetuning_hf import FinetuningModel
from data_provider.stage2_hf_dm import create_stage2_dataset

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
## for A100 gpus
torch.set_float32_matmul_precision("medium")


def parse_tasks(tasks):
    """Parse task string into dictionary."""
    tasks = tasks.split(",")
    out = defaultdict(list)
    for task in tasks:
        split = task.split("_")
        if len(split) == 1:
            out[task] = []
        elif len(split) == 2:
            out[task.split("_")[0]].append(task.split("_")[1])
    return out


def edict_to_dict(config):
    """Convert an EasyDict object to a regular dictionary."""
    if isinstance(config, edict):
        return {k: edict_to_dict(v) for k, v in config.items()}
    else:
        return config


@dataclass
class DataFinetuningArguments:
    """
    Arguments pertaining to what data we are going to input our model for finetuning.
    """
    
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during finetuning."}
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading."}
    )
    root: str = field(
        default="data/Mol-LLaMA-Instruct/",
        metadata={"help": "Root directory containing the dataset files."}
    )
    data_types: list = field(
        default_factory=list,
        metadata={"help": "List of data types to use for training (e.g., ['moleculenet', 'llm_qa'])."}
    )
    
    @classmethod
    def from_config(cls, data_config):
        """Create DataFinetuningArguments from data configuration dict."""
        return cls(
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            root=data_config.root,
            data_types=list(data_config.data_types) if hasattr(data_config, 'data_types') else [],
        )


@dataclass
class ModelFinetuningArguments:
    """
    Arguments pertaining to finetuning configuration that map to HuggingFace TrainingArguments.
    """
    
    # Training configuration
    output_dir: str = field(
        default="checkpoints/default",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of finetuning epochs to perform."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )
    
    # Batch and gradient
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    
    # Optimizer
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "The initial learning rate for AdamW optimizer."}
    )
    weight_decay: float = field(
        default=0.05,
        metadata={"help": "Weight decay to apply (if not zero) to all layers except bias and LayerNorm weights."}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of steps used for a linear warmup from 0 to learning_rate."}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. Options: linear, cosine, polynomial, constant, etc."}
    )
    
    # Precision
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training."}
    )
    
    # Checkpointing and evaluation
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use. Options: no, steps, epoch."}
    )
    save_steps: int = field(
        default=1,
        metadata={"help": "Save checkpoint every X epochs (if save_strategy='epoch') or steps."}
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints. Default: None (keep all)."}
    )
    eval_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to use. Options: no, steps, epoch."}
    )
    eval_steps: int = field(
        default=1,
        metadata={"help": "Run evaluation every X steps (if eval_strategy='steps')."}
    )
    val_check_interval: float = field(
        default=1.0,
        metadata={"help": "How often to check the validation set. Use float for fraction of training epoch or int for number of batches."}
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether to load the best model found during training at the end of training."}
    )
    
    # Logging
    logging_dir: str = field(
        default="checkpoints/default/logs",
        metadata={"help": "TensorBoard log directory."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."}
    )
    report_to: list = field(
        default_factory=lambda: ["wandb"],
        metadata={"help": "The list of integrations to report the results and logs to. Options: wandb, tensorboard, etc."}
    )
    run_name: str = field(
        default="default",
        metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    
    # Data loading
    dataloader_num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."}
    )
    dataloader_persistent_workers: bool = field(
        default=True,
        metadata={"help": "If True, data loader workers are kept alive between batches. Significantly faster for large datasets."}
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory in data loaders for faster GPU transfer."}
    )
    dataloader_drop_last: bool = field(
        default=True,
        metadata={"help": "Whether to drop the last incomplete batch. Important for consistent batch sizes in distributed training."}
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=2,
        metadata={"help": "Number of batches loaded in advance by each worker. None means default (2 in PyTorch)."}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not required by the model when using a Dataset. Must be False for custom collators."}
    )
    
    # DeepSpeed
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file (json). If provided, enables DeepSpeed integration."}
    )
    
    # Distributed training
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "When using distributed training, find unused parameters. Should be False for most cases."}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank. Set automatically by torch.distributed.launch."}
    )
    
    # Seeds
    seed: int = field(
        default=0,
        metadata={"help": "Random seed for initialization and data shuffling."}
    )
    data_seed: int = field(
        default=0,
        metadata={"help": "Random seed to be used with data samplers."}
    )
    
    @classmethod
    def from_configs(cls, train_config, data_config, checkpoint_dir: str, 
                     deepspeed_config: Optional[str] = None, 
                     max_steps: int = -1):
        """
        Create ModelFinetuningArguments from train_config and data_config.
        
        Args:
            train_config: Training configuration (EasyDict)
            data_config: Data configuration (EasyDict)
            checkpoint_dir: Directory for checkpoints
            deepspeed_config: Path to DeepSpeed config or None
            max_steps: Maximum training steps
        """
        # Determine precision
        if train_config.precision == 'bf16-mixed' or train_config.precision == 'bf16':
            fp16, bf16 = False, True
        elif train_config.precision == '16' or train_config.precision == '16-mixed':
            fp16, bf16 = True, False
        else:
            fp16, bf16 = False, False
        
        # Adjust num_workers for multi-GPU to prevent CPU contention
        # With N GPUs, each process spawns num_workers, so total = N * num_workers
        # This can saturate CPU and become a bottleneck
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        original_workers = data_config.num_workers
        
        if num_gpus > 1:
            # Scale down workers per GPU to keep total workers reasonable
            # Aim for 8-12 total workers across all GPUs
            adjusted_workers = max(2, min(original_workers, 12 // num_gpus))
            if adjusted_workers != original_workers:
                print(f"⚠️  Multi-GPU detected: Adjusting num_workers from {original_workers} to {adjusted_workers} per GPU")
                print(f"   Total workers: {num_gpus} GPUs × {adjusted_workers} workers = {num_gpus * adjusted_workers} workers")
                print(f"   (This prevents CPU contention in data loading)")
            num_workers = adjusted_workers
        else:
            num_workers = original_workers
        
        return cls(
            output_dir=checkpoint_dir,
            num_train_epochs=train_config.max_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=data_config.batch_size,
            per_device_eval_batch_size=data_config.batch_size,
            gradient_accumulation_steps=train_config.accumulate_grad_batches,
            learning_rate=train_config.init_lr,
            weight_decay=train_config.weight_decay,
            warmup_steps=train_config.warmup_steps,
            lr_scheduler_type="cosine",
            fp16=fp16,
            bf16=bf16,
            save_strategy="epoch",
            save_steps=getattr(train_config, 'save_every_n_epochs', 1),
            save_total_limit=None,
            eval_strategy="no",  # Can be changed to "epoch" or "steps" if needed
            eval_steps=1,
            val_check_interval=getattr(train_config, 'val_check_interval', 1.0),
            load_best_model_at_end=False,
            logging_dir=f"{checkpoint_dir}/logs",
            logging_steps=10,
            report_to=["wandb"],
            run_name=train_config.filename,
            dataloader_num_workers=num_workers,  # Adjusted for multi-GPU
            dataloader_persistent_workers=True,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            dataloader_prefetch_factor=2,
            remove_unused_columns=False,
            deepspeed=deepspeed_config,
            ddp_find_unused_parameters=False,
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=0,
            data_seed=0,
        )
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        args_dict = {k: v for k, v in asdict(self).items() if v is not None or k != 'deepspeed'}
        # Remove val_check_interval as it's not a TrainingArguments parameter
        args_dict.pop('val_check_interval', None)
        return TrainingArguments(**args_dict)


class FinetuningTrainer(Trainer):
    """
    Custom Trainer for Finetuning that handles the specific data format and loss computation.
    Uses pure HuggingFace data format without custom wrappers.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for training/evaluation.
        
        Args:
            model: The model
            inputs: Dictionary with graph_batch, text_batch, brics_gids, entropy_gids
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in the batch (for newer transformers versions)
            
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        graph_batch = inputs['graph_batch']
        text_batch = inputs['text_batch']
        brics_gids = inputs.get('brics_gids', None)
        entropy_gids = inputs.get('entropy_gids', None)
        
        # Construct other_infos for the model
        other_infos = {
            'brics_gids': brics_gids,
            'entropy_gids': entropy_gids,
        }
        
        # Forward pass
        output = model(
            graph_batch=graph_batch,
            text_batch=text_batch,
            other_infos=other_infos,
            return_dict=True
        )
        
        loss = output['loss']
        
        if return_outputs:
            return loss, output
        else:
            return loss


def main(model_config, train_config, data_config, test_mode=False, resume_from=None):
    """Main finetuning function."""
    torch.manual_seed(0)
    
    # Get BASE_DIR for DeepSpeed config paths
    BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize tokenizer
    if getattr(train_config, "llm_backbone", None) is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.llm_backbone, padding_side="left"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "DongkiKim/Mol-Llama-3.1-8B-Instruct",
            padding_side="left",
        )

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<mol>"]})
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    mol_id = tokenizer.convert_tokens_to_ids("<mol>")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

    if getattr(train_config, "llm_backbone", None) is not None:
        model_config.llm_config.llm_model = train_config.llm_backbone

    # Initialize model
    model = FinetuningModel(
        vocab_size=len(tokenizer),
        model_config=model_config,
        train_config=train_config,
        add_ids=[mol_id, pad_id],
    )

    # Load from Stage 1 checkpoint
    model.load_from_stage1_ckpt(train_config.stage1_path)

    # Apply LoRA to Qformer if enabled
    if train_config.enable_lora_qformer:
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
        model.model.encoder.Qformer = get_peft_model(model.model.encoder.Qformer, peft_config)
        print("LoRA enabled for Qformer")

    # Determine LLM version
    if "Llama-2" in model_config.llm_config.llm_model:
        llm_version = "llama2"
    elif "Llama-3" in model_config.llm_config.llm_model:
        llm_version = "llama3"
    elif "Qwen3" in model_config.llm_config.llm_model:
        llm_version = "qwen3"
    elif "Ministral" in model_config.llm_config.llm_model:
        llm_version = "mistral"
    elif "gemma" in model_config.llm_config.llm_model:
        llm_version = "gemma"
    else:
        raise ValueError(
            f"Unsupported model type. Choose 'llama2', 'llama3', 'qwen3', 'mistral', or 'gemma'."
        )

    # Create dataset and collator using pure HuggingFace approach
    train_dataset, data_collator = create_stage2_dataset(
        tokenizer=tokenizer,
        llm_version=llm_version,
        root=data_config.root,
        unimol_dictionary=model.model.encoder.unimol_dictionary,
        encoder_types=model_config.graph_encoder_config.encoder_types,
        data_types=data_config.data_types,
        test_mode=test_mode,
        brics_gids_enable=train_config.brics_gids_enable,
        entropy_gids_enable=train_config.entropy_gids_enable,
    )
    
    # Calculate total training steps for distributed training
    # Note: HF Trainer will create DataLoader internally with DistributedSampler
    num_samples = len(train_dataset)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # With distributed training, each GPU processes num_samples/num_gpus samples
    # Each step processes batch_size samples per GPU
    # After accumulate_grad_batches micro-steps, we do one optimizer step
    samples_per_gpu = num_samples // num_gpus if num_gpus > 1 else num_samples
    steps_per_epoch = samples_per_gpu // (data_config.batch_size * train_config.accumulate_grad_batches)
    max_steps = steps_per_epoch * train_config.max_epochs
    
    print(f"Training configuration:")
    print(f"  Total samples: {num_samples}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Samples per GPU: {samples_per_gpu}")
    print(f"  Batch size per GPU: {data_config.batch_size}")
    print(f"  Gradient accumulation steps: {train_config.accumulate_grad_batches}")
    print(f"  Effective batch size per GPU: {data_config.batch_size * train_config.accumulate_grad_batches}")
    print(f"  Global effective batch size: {data_config.batch_size * train_config.accumulate_grad_batches * num_gpus}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Max steps: {max_steps}")
    
    # Setup checkpoint directory
    checkpoint_dir = f"checkpoints/{train_config.filename}/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup DeepSpeed config if needed
    deepspeed_config = None
    if train_config.strategy_name == "deepspeed":
        deepspeed_stage = getattr(train_config, 'deepspeed_stage', 2)
        
        # Path to DeepSpeed config file
        if deepspeed_stage == 3:
            deepspeed_config_path = os.path.join(BASE_DIR, "configs/deepspeed/ds_config_zero3.json")
        else:
            # Use fast config without CPU offloading for better performance
            deepspeed_config_path = os.path.join(BASE_DIR, "configs/deepspeed/ds_config_zero2.json")
        
        # Check if config file exists
        if os.path.exists(deepspeed_config_path):
            print(f"Using DeepSpeed config: {deepspeed_config_path}")
            deepspeed_config = deepspeed_config_path
        else:
            print(f"Warning: DeepSpeed config not found at {deepspeed_config_path}")
    
    # Create ModelFinetuningArguments from configs
    model_training_args = ModelFinetuningArguments.from_configs(
        train_config=train_config,
        data_config=data_config,
        checkpoint_dir=checkpoint_dir,
        deepspeed_config=deepspeed_config,
        max_steps=max_steps,
    )
    
    # Convert to HuggingFace TrainingArguments
    training_args = model_training_args.to_training_arguments()
    
    # Initialize WandB
    if training_args.local_rank in [-1, 0]:
        wandb.init(
            project="Finetuning",  # Changed from "Stage2"
            name=train_config.filename,
            config={
                "train": edict_to_dict(train_config),
                "model": edict_to_dict(model_config),
                "data": edict_to_dict(data_config),
            },
            mode="offline",
        )
    
    # Initialize Trainer with pure HuggingFace components
    trainer = FinetuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Check for existing checkpoints or resume_from parameter
    ckpt_path = None
    if resume_from is not None:
        if resume_from == "last":
            candidate = os.path.join(checkpoint_dir, "checkpoint-last")
            ckpt_path = candidate if os.path.exists(candidate) else None
            if ckpt_path is None:
                # Try get_last_checkpoint
                ckpt_path = get_last_checkpoint(checkpoint_dir)
        else:
            ckpt_path = resume_from if os.path.exists(resume_from) else None
    else:
        # Auto-detect last checkpoint
        if os.path.isdir(checkpoint_dir):
            ckpt_path = get_last_checkpoint(checkpoint_dir)
    
    # Train
    if ckpt_path is not None:
        print(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.train(resume_from_checkpoint=ckpt_path)
    else:
        print("No checkpoint found, starting training from scratch")
        trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(checkpoint_dir, "final_model"))
    
    # Close WandB
    if training_args.local_rank in [-1, 0]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning (Stage 2) with HF Transformers")
    parser.add_argument(
        "--train_config_path", type=str, default="configs/stage2_dqw2d/train_config.yaml"
    )
    parser.add_argument(
        "--data_config_path", type=str, default="configs/stage2_dqw2d/data_config.yaml"
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
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed launcher)",
    )

    args = parser.parse_args()

    # Load configs
    data_config = edict(yaml.load(open(args.data_config_path), Loader=yaml.FullLoader))
    train_config = edict(
        yaml.load(open(args.train_config_path), Loader=yaml.FullLoader)
    )
    
    # Create model config
    model_config = MolLLaMAConfig(
        qformer_config={
            "use_flash_attention": train_config.use_flash_attention,
            "use_dq_encoder": train_config.use_dq_encoder,
            "num_query_tokens": getattr(train_config, "num_query_tokens", 8),
            "embed_dim": getattr(train_config, "embed_dim", 256),
            "cross_attention_freq": getattr(train_config, "cross_attention_freq", 2),
            "enable_lora": getattr(train_config, "enable_lora_qformer", False),
        },
        graph_encoder_config={"local_q_only": getattr(train_config, "local_q_only", False)},
        blending_module_config={
            "num_layers": getattr(train_config, "num_layers", 4),
            "num_heads": getattr(train_config, "num_heads", 8),
            "enable_blending": getattr(train_config, "enable_blending", False),
        },
    )

    print("-" * 60)
    detected_num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"batch_size: {data_config.batch_size}\tnum_devices: {detected_num_devices}\taccumulate_grad_batches: {train_config.accumulate_grad_batches}"
    )
    print(
        f"Total batch size: {data_config.batch_size * detected_num_devices * train_config.accumulate_grad_batches}"
    )
    print("-" * 60)
    print(f"Data Types:")
    for data_type in data_config.data_types:
        print(f"  - {data_type}")
    print("-" * 60)

    if args.test_mode:
        print("TEST MODE: Using small dataset for quick testing")
    
    main(
        model_config,
        train_config,
        data_config,
        test_mode=args.test_mode,
        resume_from=args.resume_from,
    )

