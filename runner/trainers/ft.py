from transformers import Trainer

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
            inputs: Dictionary with graph_batch (containing brics_gids, entropy_gids), text_batch
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in the batch (for newer transformers versions)
            
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        graph_batch = inputs['graph_batch']
        text_batch = inputs['text_batch']
        # brics_gids and entropy_gids are now in graph_batch and will be extracted by the model
        
        # Forward pass
        output = model(
            graph_batch=graph_batch,
            text_batch=text_batch,
            return_dict=True
        )
        
        loss = output['loss']
        
        if return_outputs:
            return loss, output
        else:
            return loss

