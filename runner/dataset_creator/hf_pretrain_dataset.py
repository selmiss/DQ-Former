"""
HuggingFace-compatible dataset loader for preprocessed Stage 1 pretraining data.
Supports loading from local JSONL files or HuggingFace Hub.

Usage example:

    # From local JSONL file
    train_dataset, val_dataset, collator = create_hf_pretrain_datasets(
        data_path='/path/to/preprocessed_data.jsonl',
        tokenizer=scibert_tokenizer,
        text_max_len=128,
        pad_idx=0,
        encoder_types=['unimol', 'moleculestm']
    )
    
    # From HuggingFace Hub
    train_dataset, val_dataset, collator = create_hf_pretrain_datasets(
        data_path='username/dataset-name',
        tokenizer=scibert_tokenizer,
        text_max_len=128,
        pad_idx=0,
        encoder_types=['unimol', 'moleculestm'],
        load_from_hub=True
    )
"""
import os
import json
import logging
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from datasets import load_dataset, Dataset as HFDataset

from data_provider.collaters import Mol3DCollater

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
    # Only include x, edge_index, edge_attr - no extra attributes
    # Let PyG auto-infer num_nodes from x.size(0) to avoid collation issues
    edge_attr = torch.LongTensor(data_dict['edge_attr'])
    # Handle empty edge_attr: reshape [0] to [0, 3] for molecules without edges
    if edge_attr.numel() == 0:
        edge_attr = edge_attr.new_zeros((0, 3))
    return Data(
        x=torch.LongTensor(data_dict['node_feat']),
        edge_index=torch.LongTensor(data_dict['edge_index']),
        edge_attr=edge_attr
    )


class HFPretrainDataset:
    """
    HuggingFace-compatible wrapper for preprocessed pretraining data.
    
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
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """
        Get item at index and deserialize graph data.
        
        Returns:
            Dictionary with:
                - data_graphs: Dict with encoder-specific graph data
                - iupac_name: String
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
        
        # Ensure iupac_name is always a valid string (safeguard against None or non-string values)
        iupac_name = item.get('iupac_name', '')
        if iupac_name is None or not isinstance(iupac_name, str):
            logger.warning(
                f"⚠️  Sample at index {idx} has invalid iupac_name (type: {type(iupac_name).__name__}). "
                f"Replacing with empty string, which may cause training issues!"
            )
            iupac_name = ''
        
        return {
            'data_graphs': data_graphs,
            'iupac_name': iupac_name,
            'brics_gids': item.get('brics_gids', None),
            'entropy_gids': item.get('entropy_gids', None),
        }


class HFPretrainCollator:
    """
    Data collator for HF-loaded preprocessed Stage 1 data.
    Compatible with the original PretainCollator interface.
    """
    
    def __init__(self, tokenizer, text_max_len, pad_idx, encoder_types):
        """
        Args:
            tokenizer: HuggingFace tokenizer (e.g., SciBERT)
            text_max_len: Maximum text sequence length
            pad_idx: Padding index for UniMol
            encoder_types: List of encoder types
        """
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.encoder_types = encoder_types
        if 'unimol' in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)
    
    def __call__(self, batch):
        """
        Collate a batch of preprocessed samples.
        
        Args:
            batch: List of dictionaries from HFPretrainDataset
            
        Returns:
            Dictionary with graph_batch, text_batch, iupac_names, brics_gids, entropy_gids
        """
        # Extract fields from batch
        data_graphs = [item['data_graphs'] for item in batch]
        # Ensure all iupac_names are valid strings (extra safeguard)
        iupac_names = []
        invalid_count = 0
        for i, item in enumerate(batch):
            name = item['iupac_name']
            if not name or not isinstance(name, str):
                invalid_count += 1
                iupac_names.append('')
            else:
                iupac_names.append(str(name))
        
        if invalid_count > 0:
            logger.warning(
                f"⚠️  Found {invalid_count}/{len(batch)} samples with invalid iupac_name in batch. "
                f"Replaced with empty strings, which may cause training to fail!"
            )
        
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
        
        # Process text data
        text_batch = self.tokenizer(
            iupac_names,
            truncation=True,
            padding='max_length',
            max_length=self.text_max_len,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        # Add brics_gids and entropy_gids to graph_batch
        # Only add if at least one is not None
        if any(g is not None for g in brics_gids_list):
            graph_batch['brics_gids'] = brics_gids_list
        else:
            graph_batch['brics_gids'] = None
        
        if any(g is not None for g in entropy_gids_list):
            graph_batch['entropy_gids'] = entropy_gids_list
        else:
            graph_batch['entropy_gids'] = None
        
        # Return all data
        result = {
            'graph_batch': graph_batch,
            'text_batch': text_batch,
            'iupac_names': iupac_names,
        }
        
        return result


def create_hf_pretrain_datasets(
    data_path: str,
    tokenizer,
    text_max_len: int,
    pad_idx: int,
    encoder_types: List[str],
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    val_ratio: float = 0.01,
    random_seed: int = 42,
):
    """
    Factory function to create HF-based train/val datasets and collator.
    
    Automatically handles three cases:
    1. Local directory with train.jsonl/val.jsonl files
    2. Local JSONL file or pattern (e.g., 'data/pretrain/*.jsonl')
    3. HuggingFace Hub repository (e.g., 'username/dataset-name')
    
    Args:
        data_path: Path to data - can be:
                   - Local directory: 'data/pretrain/' (must have train.jsonl/val.jsonl)
                   - Local file: 'data/pretrain/data.jsonl'
                   - Wildcard pattern: 'data/pretrain/*.jsonl'
                   - HuggingFace Hub repo: 'username/dataset-name'
        tokenizer: HuggingFace tokenizer (e.g., SciBERT)
        text_max_len: Maximum text sequence length
        pad_idx: Padding index for UniMol
        encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
        cache_dir: Optional cache directory for HuggingFace datasets
        streaming: If True, use streaming mode (useful for very large datasets)
        val_ratio: Validation ratio if only single 'train' split exists (default: 0.01)
        random_seed: Random seed for splitting (default: 42)
    
    Returns:
        tuple: (train_dataset, val_dataset, collator)
    """
    logger.info("=" * 80)
    logger.info("Loading preprocessed datasets using HuggingFace datasets")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Encoder types: {encoder_types}")
    logger.info(f"Streaming: {streaming}")
    
    
    # HuggingFace load_dataset automatically handles:
    logger.info(f"Loading dataset: {data_path}")
    dataset = load_dataset(data_path, cache_dir=cache_dir, streaming=streaming)
    
    # Extract train/val splits from loaded dataset
    if 'train' in dataset and 'validation' in dataset:
        logger.info("✅ Using train/validation splits")
        train_hf_dataset = dataset['train']
        val_hf_dataset = dataset['validation']
    elif 'train' in dataset:
        # Only single 'train' split - do random split based on val_ratio
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
    train_dataset = HFPretrainDataset(train_hf_dataset, encoder_types)
    val_dataset = HFPretrainDataset(val_hf_dataset, encoder_types)
    
    # Create collator
    collator = HFPretrainCollator(
        tokenizer=tokenizer,
        text_max_len=text_max_len,
        pad_idx=pad_idx,
        encoder_types=encoder_types
    )
    
    logger.info(f"✅ Datasets ready!")
    if not streaming:
        logger.info(f"   Train: {len(train_dataset)} samples")
        logger.info(f"   Val: {len(val_dataset)} samples")
    logger.info("=" * 80)
    
    return train_dataset, val_dataset, collator

