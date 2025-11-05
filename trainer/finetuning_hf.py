import torch
import torch.nn as nn
from typing import Dict, Optional

from models.mol_llama import MolLLaMA, DQMolLLaMA


def load_ignore_unexpected(model, state_dict):
    """Load state dict ignoring unexpected keys."""
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    """Extract module state dict from full state dict."""
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class FinetuningModel(nn.Module):
    """
    Finetuning Model (Stage 2) for Hugging Face Trainer.
    Converted from PyTorch Lightning LightningModule.
    """
    def __init__(self, vocab_size, model_config, train_config, add_ids):
        super().__init__()
        self.train_config = train_config
        
        # Determine torch dtype from precision setting
        if train_config.precision == 'bf16-mixed':
            torch_dtype = "bfloat16"
        elif train_config.precision == '16':
            torch_dtype = "float16"
        elif train_config.precision == '32':
            torch_dtype = "float32"
        else:
            torch_dtype = "bfloat16"  # default

        # Choose model based on configuration
        if model_config.qformer_config.use_dq_encoder:
            enable_blending = getattr(train_config, 'enable_blending', False)
            print(f"Using DQMolLLaMA, enable_blending: {enable_blending}")
            self.model = DQMolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype=torch_dtype,
                enable_flash=train_config.enable_flash,
                add_ids=add_ids,
                freeze_llm=getattr(train_config, 'freeze_llm', False),
                brics_gids_enable=train_config.brics_gids_enable,
                entropy_gids_enable=train_config.entropy_gids_enable,
                enable_blending=enable_blending,
            )
        else:
            print("Using MolLLaMA")
            self.model = MolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype=torch_dtype,
                enable_flash=train_config.enable_flash,
                freeze_llm=getattr(train_config, 'freeze_llm', False),
                enable_blending=getattr(train_config, 'enable_blending', False),
            )

    def load_from_stage1_ckpt(self, ckpt_path):
        """Load encoder weights from Stage 1 checkpoint."""
        self.model.load_from_stage1_ckpt(ckpt_path)

    def forward(
        self,
        graph_batch: Dict,
        text_batch: Dict,
        other_infos: Dict,
        return_dict: bool = True,
    ):
        """
        Forward pass that computes the loss.
        
        Args:
            graph_batch: Dictionary containing graph data
            text_batch: Dictionary containing text data  
            other_infos: Dictionary with additional information
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'loss' or just loss tensor
        """
        output = self.model(graph_batch, text_batch, other_infos)
        
        if return_dict:
            return output
        else:
            return output['loss']


def precision2dtype(precision):
    """Convert precision string to torch dtype."""
    if precision == '16':
        return torch.float16
    elif precision == '32':
        return torch.float32
    elif precision.find('bf16') >= 0:
        return torch.bfloat16
    else:
        raise NotImplementedError(f"Precision {precision} not implemented")

