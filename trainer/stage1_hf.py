import torch
import torch.nn as nn
from typing import Dict, Optional

from models.mol_llama_encoder import MolLLaMAEncoder
from models.DQ_former_encoder import DQMolLLaMAEncoder


class Stage1Model(nn.Module):
    """
    Stage1 Model for Hugging Face Trainer.
    Converted from PyTorch Lightning LightningModule.
    """
    def __init__(self, model_config, train_config):
        super().__init__()
        self.train_config = train_config
        
        # Choose encoder based on configuration
        if model_config.qformer_config.use_dq_encoder:
            print("Using DQMolLLaMAEncoder")
            self.encoder = DQMolLLaMAEncoder(
                graph_encoder_config=model_config.graph_encoder_config,
                blending_module_config=model_config.blending_module_config,
                qformer_config=model_config.qformer_config,
                temperature=train_config.temperature,
                tune_gnn=train_config.tune_gnn,
                enable_blending=train_config.enable_blending,
                brics_gids_enable=train_config.brics_gids_enable,
                entropy_gids_enable=train_config.entropy_gids_enable,
            )
        else:
            print("Using MolLLaMAEncoder")
            self.encoder = MolLLaMAEncoder(
                graph_encoder_config=model_config.graph_encoder_config,
                blending_module_config=model_config.blending_module_config,
                qformer_config=model_config.qformer_config,
                temperature=train_config.temperature,
                tune_gnn=train_config.tune_gnn,
                enable_blending=train_config.enable_blending,
            )
    
    def forward(
        self,
        graph_batch: Dict,
        text_batch: Dict,
        brics_gids: Optional[torch.Tensor] = None,
        entropy_gids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass that computes the loss.
        
        Args:
            graph_batch: Dictionary containing graph data
            text_batch: Dictionary containing text data
            brics_gids: BRICS group IDs (optional)
            entropy_gids: Entropy group IDs (optional)
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'loss' and individual loss components
        """
        loss_dict = self.encoder.compute_loss(graph_batch, text_batch, brics_gids, entropy_gids)
        
        if return_dict:
            return loss_dict
        else:
            return loss_dict['loss']
    
    def get_encoder(self):
        """Get the encoder for inference or checkpoint loading."""
        return self.encoder


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

