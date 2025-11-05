import os
import torch
import warnings
import argparse
import yaml
import glob
from dataclasses import dataclass, field, asdict
from easydict import EasyDict as edict
from typing import Dict, Optional

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
import wandb

from utils.configuration_mol_llama import MolLLaMAConfig
from trainer.stage1_hf import Stage1Model
from data_provider.stage1_hf_dm import create_stage1_datasets

## for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
## for A100 gpus
torch.set_float32_matmul_precision(
    "medium"
)  # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def edict_to_dict(config):
    """
    Convert an EasyDict object to a regular dictionary.
    """
    if isinstance(config, edict):
        return {k: edict_to_dict(v) for k, v in config.items()}
    else:
        return config


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    batch_size: int = field(
        default=48,
        metadata={"help": "Batch size per device during training and evaluation."}
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading."}
    )
    text_max_len: int = field(
        default=512,
        metadata={"help": "Maximum length of text sequences (IUPAC names)."}
    )
    root: str = field(
        default="data/Mol-LLaMA-Instruct/",
        metadata={"help": "Root directory containing the dataset files."}
    )
    
    @classmethod
    def from_config(cls, data_config):
        """Create DataTrainingArguments from data configuration dict."""
        return cls(
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            text_max_len=data_config.text_max_len,
            root=data_config.root,
        )


@dataclass
class ModelTrainingArguments:
    """
    Arguments pertaining to training configuration that map to HuggingFace TrainingArguments.
    """
    
    # Training configuration
    output_dir: str = field(
        default="checkpoints/default",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    num_train_epochs: int = field(
        default=6,
        metadata={"help": "Total number of training epochs to perform."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )
    
    # Batch and gradient
    per_device_train_batch_size: int = field(
        default=48,
        metadata={"help": "Batch size per GPU/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=48,
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
        default=500,
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
        default=3,
        metadata={"help": "Save checkpoint every X epochs (if save_strategy='epoch')."}
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints. Default: None (keep all)."}
    )
    eval_strategy: str = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use. Options: no, steps, epoch."}
    )
    eval_steps: int = field(
        default=3,
        metadata={"help": "Run evaluation every X epochs (if evaluation_strategy='epoch')."}
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
        Create ModelTrainingArguments from train_config and data_config.
        
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
            save_steps=train_config.save_every_n_epochs,
            save_total_limit=None,
            eval_strategy="epoch",
            eval_steps=train_config.check_val_every_n_epoch,
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
        # Convert dataclass to dict, excluding None values for deepspeed
        args_dict = {k: v for k, v in asdict(self).items() if v is not None or k != 'deepspeed'}
        return TrainingArguments(**args_dict)


class LossLoggingCallback(TrainerCallback):
    """
    Callback to log individual loss components (loss_gtc, loss_gtm, loss_lm).
    """
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log loss components if available."""
        if logs is not None and state.is_world_process_zero:
            # The logs will contain our custom metrics
            pass
        return control




class Stage1Trainer(Trainer):
    """
    Custom Trainer for Stage1 that handles the specific data format and loss computation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = {
            'train_loss_gtc': [],
            'train_loss_gtm': [],
            'train_loss_lm': [],
            'val_loss_gtc': [],
            'val_loss_gtm': [],
            'val_loss_lm': [],
        }
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for training/evaluation.
        
        Args:
            model: The model
            inputs: Dictionary with graph_batch, text_batch, etc.
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in the batch (for newer transformers versions)
            
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        graph_batch = inputs['graph_batch']
        text_batch = inputs['text_batch']
        brics_gids = inputs.get('brics_gids', None)
        entropy_gids = inputs.get('entropy_gids', None)
        
        # Forward pass
        loss_dict = model(
            graph_batch=graph_batch,
            text_batch=text_batch,
            brics_gids=brics_gids,
            entropy_gids=entropy_gids,
            return_dict=True
        )
        
        loss = loss_dict['loss']
        
        # Store loss components for logging
        if self.args.local_rank in [-1, 0]:
            prefix = 'train' if model.training else 'val'
            self.loss_history[f'{prefix}_loss_gtc'].append(float(loss_dict['loss_gtc']))
            self.loss_history[f'{prefix}_loss_gtm'].append(float(loss_dict['loss_gtm']))
            self.loss_history[f'{prefix}_loss_lm'].append(float(loss_dict['loss_lm']))
        
        if return_outputs:
            return loss, loss_dict
        else:
            return loss
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log metrics including custom loss components.
        
        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for timing calculations (added in newer transformers)
        """
        # Add loss components to logs
        if self.state.is_world_process_zero:
            if self.loss_history['train_loss_gtc']:
                logs['train_loss_gtc'] = sum(self.loss_history['train_loss_gtc']) / len(self.loss_history['train_loss_gtc'])
                logs['train_loss_gtm'] = sum(self.loss_history['train_loss_gtm']) / len(self.loss_history['train_loss_gtm'])
                logs['train_loss_lm'] = sum(self.loss_history['train_loss_lm']) / len(self.loss_history['train_loss_lm'])
                # Clear history after logging
                self.loss_history['train_loss_gtc'] = []
                self.loss_history['train_loss_gtm'] = []
                self.loss_history['train_loss_lm'] = []
            
            if self.loss_history['val_loss_gtc']:
                logs['val_loss_gtc'] = sum(self.loss_history['val_loss_gtc']) / len(self.loss_history['val_loss_gtc'])
                logs['val_loss_gtm'] = sum(self.loss_history['val_loss_gtm']) / len(self.loss_history['val_loss_gtm'])
                logs['val_loss_lm'] = sum(self.loss_history['val_loss_lm']) / len(self.loss_history['val_loss_lm'])
                # Clear history after logging
                self.loss_history['val_loss_gtc'] = []
                self.loss_history['val_loss_gtm'] = []
                self.loss_history['val_loss_lm'] = []
        
        # Call parent class log (pass start_time if provided for newer transformers versions)
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


def main(model_config, train_config, data_config, test_mode=False):
    """Main training function."""
    torch.manual_seed(0)
    
    # Get BASE_DIR for DeepSpeed config paths
    BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize model
    model = Stage1Model(model_config, train_config)
    
    # Create datasets using HuggingFace-compatible implementation with caching
    train_dataset, val_dataset, data_collator = create_stage1_datasets(
        unimol_dictionary=(
            model.encoder.unimol_dictionary
            if "unimol" in model_config.graph_encoder_config.encoder_types
            else None
        ),
        scibert_tokenizer=model.encoder.scibert_tokenizer,
        encoder_types=model_config.graph_encoder_config.encoder_types,
        text_max_len=data_config.text_max_len,
        root=data_config.root,
        test_mode=test_mode,
        brics_gids_enable=train_config.brics_gids_enable,
        entropy_gids_enable=train_config.entropy_gids_enable,
        use_cache=True,
    )
    
    # Calculate total training steps for distributed training
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    samples_per_gpu = len(train_dataset) // num_gpus if num_gpus > 1 else len(train_dataset)
    steps_per_epoch = samples_per_gpu // (data_config.batch_size * train_config.accumulate_grad_batches)
    max_steps = steps_per_epoch * train_config.max_epochs
    
    print(f"Training configuration:")
    print(f"  Total train samples: {len(train_dataset)}")
    print(f"  Total val samples: {len(val_dataset)}")
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
            deepspeed_config_path = os.path.join(BASE_DIR, "configs/deepspeed/ds_config_zero2.json")
        
        # Check if config file exists
        if os.path.exists(deepspeed_config_path):
            print(f"Using DeepSpeed config: {deepspeed_config_path}")
            deepspeed_config = deepspeed_config_path
        else:
            print(f"Warning: DeepSpeed config not found at {deepspeed_config_path}")
    
    # Create ModelTrainingArguments from configs
    model_training_args = ModelTrainingArguments.from_configs(
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
            project="stage1_v2",
            name=train_config.filename,
            config={
                "train": edict_to_dict(train_config),
                "model": edict_to_dict(model_config),
                "data": edict_to_dict(data_config),
            },
            mode="offline",  # Change to "online" if you want online logging
        )
    
    # Create callbacks
    callbacks = [LossLoggingCallback()]
    
    # Initialize Trainer
    trainer = Stage1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Check for existing checkpoints
    last_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
    
    # Train
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found, starting training from scratch")
        trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(checkpoint_dir, "final_model"))
    
    # Close WandB
    if training_args.local_rank in [-1, 0]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 training with HF Transformers")
    parser.add_argument(
        "--train_config_path", type=str, default="configs/stage1_dqw2d/train_config.yaml"
    )
    parser.add_argument(
        "--data_config_path", type=str, default="configs/stage1_dqw2d/data_config.yaml"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help="Use small dataset for testing",
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
    
    # Create model config with Flash Attention support
    model_config = MolLLaMAConfig(
        qformer_config={
            "use_flash_attention": train_config.use_flash_attention,
            "use_dq_encoder": train_config.use_dq_encoder,
            "num_query_tokens": getattr(train_config, "num_query_tokens", 8),
            "embed_dim": getattr(train_config, "embed_dim", 256),
            "cross_attention_freq": getattr(train_config, "cross_attention_freq", 2),
        },
        graph_encoder_config={"local_q_only": train_config.local_q_only},
        blending_module_config={
            "num_layers": getattr(train_config, "num_layers", 4),
            "num_heads": getattr(train_config, "num_heads", 8),
            "enable_blending": getattr(train_config, "enable_blending", False),
        },
    )
    
    if train_config.enable_blending:
        model_config.graph_encoder_config.encoder_types = ["unimol", "moleculestm"]
        print(f"Caution: Using blending module" + "-" * 10)

    print("-" * 60)
    detected_num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"batch_size: {data_config.batch_size}\tnum_devices: {detected_num_devices}\taccumulate_grad_batches: {train_config.accumulate_grad_batches}"
    )
    print(
        f"Total batch size: {data_config.batch_size * detected_num_devices * train_config.accumulate_grad_batches}"
    )
    if args.test_mode:
        print("TEST MODE: Using small dataset for quick testing")
    print("-" * 60)

    main(model_config, train_config, data_config, test_mode=args.test_mode)

