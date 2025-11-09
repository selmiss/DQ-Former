"""
HuggingFace-compatible dataset loader for preprocessed Stage 2 finetuning data.
Supports loading from local JSONL files or HuggingFace Hub.

Usage example:

    # From local JSONL file
    dataset, collator = create_hf_finetune_dataset(
        data_path='/path/to/preprocessed_finetune_data.jsonl',
        tokenizer=tokenizer,
        llm_version='llama3',
        pad_idx=0,
        encoder_types=['unimol', 'moleculestm']
    )
    
    # From HuggingFace Hub
    dataset, collator = create_hf_finetune_dataset(
        data_path='username/dataset-name',
        tokenizer=tokenizer,
        llm_version='llama3',
        pad_idx=0,
        encoder_types=['unimol', 'moleculestm'],
        load_from_hub=True
    )
"""
import os
import json
import logging
from typing import List, Optional
import torch
from torch_geometric.data import Data, Batch
from datasets import load_dataset
import numpy as np
from data_provider.collaters import Mol3DCollater
from data_provider.tokenization_utils import batch_tokenize_messages_list

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def deserialize_unimol_data(data_dict):
    """Convert serialized UniMol data back to tensors."""
    return {
        'src_tokens': torch.LongTensor(data_dict['src_tokens']),
        'src_edge_type': torch.LongTensor(data_dict['src_edge_type']),
        'src_distance': torch.FloatTensor(data_dict['src_distance'])
    }


def deserialize_moleculestm_data(data_dict):
    """Convert serialized MoleculeSTM data back to PyG Data object."""
    return Data(
        x=torch.LongTensor(data_dict['node_feat']),
        edge_index=torch.LongTensor(data_dict['edge_index']),
        edge_attr=torch.LongTensor(data_dict['edge_attr'])
    )


class HFFinetuneDataset:
    """
    HuggingFace-compatible wrapper for preprocessed finetuning data.
    
    This class wraps a HuggingFace Dataset and provides deserialization
    of graph data on-the-fly during iteration.
    """
    
    def __init__(self, hf_dataset, encoder_types):
        """
        Args:
            hf_dataset: HuggingFace Dataset instance
            encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
        """
        self.hf_dataset = hf_dataset
        self.encoder_types = encoder_types
        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """
        Get item at index and deserialize graph data.
        
        Returns:
            Dictionary with:
                - data_graphs: Dict with encoder-specific graph data
                - messages: List of message dicts (with roles and content)
                - brics_gids: List or None
                - entropy_gids: List or None
        """
        item = self.hf_dataset[idx]
        
        # Deserialize graph data
        data_graphs = {}
        graph_data = json.loads(item['graph_data']) if isinstance(item['graph_data'], str) else item['graph_data']
        
        if 'unimol' in self.encoder_types and 'unimol' in graph_data:
            data_graphs['unimol'] = [deserialize_unimol_data(graph_data['unimol'])]
        
        if 'moleculestm' in self.encoder_types and 'moleculestm' in graph_data:
            data_graphs['moleculestm'] = [deserialize_moleculestm_data(graph_data['moleculestm'])]
        
        # Parse messages (convert from JSON if needed)
        messages = json.loads(item['messages']) if isinstance(item['messages'], str) else item['messages']
        
        # Replace <mol> tokens with mol_prompt in user messages
        for msg in messages:
            if msg['role'] == 'user':
                msg['content'] = msg['content'].replace('<mol>', self.mol_prompt)
        
        return {
            'data_graphs': data_graphs,
            'messages': messages,
            'brics_gids': item.get('brics_gids', None),
            'entropy_gids': item.get('entropy_gids', None),
        }


class HFFinetuneCollator:
    """
    Data collator for HF-loaded preprocessed Stage 2 finetuning data.
    Compatible with the original FinetuneCollator interface.
    """
    
    def __init__(self, tokenizer, llm_version, pad_idx, encoder_types):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            llm_version: LLM version (llama2, llama3, qwen3, mistral, gemma)
            pad_idx: Padding index for UniMol
            encoder_types: List of encoder types
        """
        self.tokenizer = tokenizer
        self.llm_version = llm_version
        self.encoder_types = encoder_types
        if 'unimol' in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)
    
    def __call__(self, batch):
        """
        Collate a batch of preprocessed samples.
        
        Args:
            batch: List of dictionaries from HFFinetuneDataset
            
        Returns:
            Dictionary with graph_batch, text_batch, brics_gids, entropy_gids
        """
        # Extract fields from batch
        data_graphs = [item['data_graphs'] for item in batch]
        messages_list = [item['messages'] for item in batch]
        brics_gids_list = [item['brics_gids'] for item in batch]
        entropy_gids_list = [item['entropy_gids'] for item in batch]
        
        # Process graph data
        graph_batch = {}
        if 'unimol' in self.encoder_types:
            data_unimol = []
            for data in data_graphs:
                data_unimol.extend(data['unimol'])
            graph_batch['unimol'] = self.d3_collater(data_unimol)
        
        if 'moleculestm' in self.encoder_types:
            data_moleculestm = []
            for data in data_graphs:
                data_moleculestm.extend(data['moleculestm'])
            graph_batch['moleculestm'] = Batch.from_data_list(data_moleculestm)
        
        # Process text data (tokenize based on LLM version)
        text_batch = batch_tokenize_messages_list(
            messages_list,
            self.tokenizer,
            self.llm_version,
            padding_side='left'
        )
        
        # Return all data
        result = {
            'graph_batch': graph_batch,
            'text_batch': text_batch,
        }
        
        # Only add brics_gids if at least one is not None
        if any(g is not None for g in brics_gids_list):
            result['brics_gids'] = brics_gids_list
        else:
            result['brics_gids'] = None
        
        # Only add entropy_gids if at least one is not None
        if any(g is not None for g in entropy_gids_list):
            result['entropy_gids'] = entropy_gids_list
        else:
            result['entropy_gids'] = None
        
        return result


def create_hf_finetune_dataset(
    data_path: str,
    tokenizer,
    llm_version: str,
    pad_idx: int,
    encoder_types: List[str],
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    val_ratio: float = 0.01,
    random_seed: int = 42,
):
    """
    Factory function to create HF-based finetuning dataset and collator.
    
    Automatically handles three cases:
    1. Local directory with *.jsonl files (loads all JSONL files)
    2. Local JSONL file or pattern (e.g., 'data/finetune/*.jsonl')
    3. HuggingFace Hub repository (e.g., 'username/dataset-name')
    
    Args:
        data_path: Path to data - can be:
                   - Local directory: 'data/finetune/' (loads all *.jsonl)
                   - Local file: 'data/finetune/train.jsonl'
                   - Wildcard pattern: 'data/finetune/*.jsonl'
                   - HuggingFace Hub repo: 'username/dataset-name'
        tokenizer: HuggingFace tokenizer
        llm_version: LLM version (llama2, llama3, qwen3, mistral, gemma)
        pad_idx: Padding index for UniMol
        encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
        cache_dir: Optional cache directory for HuggingFace datasets
        streaming: If True, use streaming mode (useful for very large datasets)
        val_ratio: Validation set ratio if only 'train' split exists (default: 0.01)
        random_seed: Random seed for splitting (default: 42)
    
    Returns:
        tuple: (train_dataset, val_dataset, collator)
    """
    logger.info("=" * 80)
    logger.info("Loading preprocessed finetuning dataset using HuggingFace datasets")
    logger.info(f"Data path: {data_path}")
    logger.info(f"LLM version: {llm_version}")
    logger.info(f"Encoder types: {encoder_types}")
    logger.info(f"Streaming: {streaming}")
    
    # HuggingFace load_dataset automatically handles:
    # - Local directories (loads as dataset)
    # - HuggingFace Hub repositories (downloads and loads)
    # We only need special handling for local files/patterns
    
    logger.info(f"Loading dataset: {data_path}")
    dataset = load_dataset(data_path, cache_dir=cache_dir, streaming=streaming)
    
    # Extract train/val splits from loaded dataset
    if 'train' in dataset and 'validation' in dataset:
        logger.info("✅ Using train/validation splits")
        train_hf_dataset = dataset['train']
        val_hf_dataset = dataset['validation']
    elif 'train' in dataset:
        # Only single 'train' split - do fast random split
        logger.warning(f"Only 'train' split found, randomly splitting with val_ratio={val_ratio}")
        all_data = dataset['train']
        
        if not streaming:
            total_size = len(all_data)
            val_size = int(total_size * val_ratio)
            
            # Create shuffled indices
            import numpy as np
            np.random.seed(random_seed)
            indices = np.random.permutation(total_size)
            
            train_indices = indices[val_size:].tolist()
            val_indices = indices[:val_size].tolist()
            
            train_hf_dataset = all_data.select(train_indices)
            val_hf_dataset = all_data.select(val_indices)
            logger.info(f"✅ Split into {len(train_hf_dataset)} train, {len(val_hf_dataset)} val")
        else:
            # For streaming, can't do random split efficiently
            logger.warning("Streaming mode with single split - using same data for both train/val")
            train_hf_dataset = all_data
            val_hf_dataset = all_data
    else:
        raise ValueError("Dataset must have 'train' split or 'train'+'validation' splits")
    
    # Wrap HF datasets
    train_dataset = HFFinetuneDataset(train_hf_dataset, encoder_types)
    val_dataset = HFFinetuneDataset(val_hf_dataset, encoder_types)
    
    # Create collator
    collator = HFFinetuneCollator(
        tokenizer=tokenizer,
        llm_version=llm_version,
        pad_idx=pad_idx,
        encoder_types=encoder_types
    )
    
    logger.info(f"✅ Datasets ready!")
    if not streaming:
        logger.info(f"   Train: {len(train_dataset)} samples")
        logger.info(f"   Val: {len(val_dataset)} samples")
    logger.info("=" * 80)
    
    return train_dataset, val_dataset, collator