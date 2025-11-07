from transformers import Trainer
from typing import Dict, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl

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

class PretrainTrainer(Trainer):
    """
    Custom Trainer for Pretraining that handles the specific data format and loss computation.
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