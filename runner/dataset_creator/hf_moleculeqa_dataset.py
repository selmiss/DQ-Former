"""
HuggingFace-compatible dataset loader for preprocessed MoleculeQA data.
Supports loading from local JSONL files or HuggingFace Hub.

Usage example:

    # From local directory
    datasets, collator = create_hf_moleculeqa_datasets(
        data_path='data/moleculeqa_preprocessed/',
        tokenizer=tokenizer,
        llm_version='llama3',
        pad_idx=0,
        encoder_types=['unimol', 'moleculestm']
    )
    
    # From HuggingFace Hub
    datasets, collator = create_hf_moleculeqa_datasets(
        data_path='username/moleculeqa-preprocessed',
        tokenizer=tokenizer,
        llm_version='llama3',
        pad_idx=0,
        encoder_types=['unimol', 'moleculestm']
    )
"""
import os
import json
import logging
from typing import List, Optional, Dict
from collections import defaultdict
import torch
from torch_geometric.data import Data, Batch
from datasets import load_dataset

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


class HFMoleculeQADataset:
    """
    HuggingFace-compatible wrapper for preprocessed MoleculeQA data.
    
    This class wraps a HuggingFace Dataset and provides deserialization
    of graph data on-the-fly during iteration.
    """
    
    def __init__(self, hf_dataset, encoder_types, mol_type='mol', do_infer=False):
        """
        Args:
            hf_dataset: HuggingFace Dataset instance
            encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
            mol_type: Molecular representation type
            do_infer: If True, don't include assistant responses in messages
        """
        self.hf_dataset = hf_dataset
        self.encoder_types = encoder_types
        self.mol_type = mol_type
        self.do_infer = do_infer
        
        # Prompts for different representations
        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
        self.graph_prompt = "<graph>" * 28
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """
        Get item at index and deserialize graph data.
        
        Returns:
            Dictionary with:
                - data_graphs: Dict with encoder-specific graph data
                - messages: List of message dicts (with roles and content)
                - cid: Compound ID
                - task: Task category
                - answer: Ground truth answer
                - smiles: SMILES string
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
        
        # Parse conversations (convert from JSON if needed)
        conversations = json.loads(item['conversations']) if isinstance(item['conversations'], str) else item['conversations']
        
        # Build messages
        messages = []
        messages.append({"role": "system", "content": item['system']})
        
        # Process conversations (typically only 1 for MoleculeQA)
        answer = None
        for turn in conversations:
            # Replace <mol> with appropriate representation
            if self.mol_type == 'mol':
                user_prompt = turn['user'].replace('<mol>', self.mol_prompt)
            elif self.mol_type == 'SMILES':
                user_prompt = turn['user'].replace('<mol>', item['smiles'])
            elif self.mol_type == "SMILES,mol":
                user_prompt = turn['user'].replace('<mol>', item['smiles'][:96] + ' ' + self.mol_prompt)
            elif self.mol_type == 'SELFIES':
                user_prompt = turn['user'].replace('<mol>', item.get('selfies', ''))
            elif self.mol_type == "SMILES,<graph>":
                user_prompt = turn['user'].replace('<mol>', item['smiles'][:128] + ' ' + self.graph_prompt)
            else:
                user_prompt = turn['user'].replace('<mol>', self.mol_prompt)
            
            messages.append({"role": "user", "content": user_prompt})
            
            # Only add assistant response if not inference
            if not self.do_infer:
                messages.append({"role": "assistant", "content": turn['assistant']})
            
            answer = turn['assistant']
        
        return {
            'data_graphs': data_graphs,
            'messages': messages,
            'brics_gids': item.get('brics_gids', None),
            'entropy_gids': item.get('entropy_gids', None),
            # Additional info for metrics computation
            'cid': item['cid'],
            'task': item.get('category', None),
            'answer': answer,
            'smiles': item['smiles'],
        }


class HFMoleculeQACollator:
    """
    Data collator for HF-loaded preprocessed MoleculeQA data.
    Compatible with the original MoleculeQAHFCollator interface.
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
            batch: List of dictionaries from HFMoleculeQADataset
            
        Returns:
            Dictionary with graph_batch, text_batch, other_infos, brics_gids, entropy_gids
        """
        # Extract fields from batch
        data_graphs = [item['data_graphs'] for item in batch]
        messages_list = [item['messages'] for item in batch]
        brics_gids_list = [item['brics_gids'] for item in batch]
        entropy_gids_list = [item['entropy_gids'] for item in batch]
        
        # Extract other info for metrics computation
        other_infos = defaultdict(list)
        for item in batch:
            other_infos['cid'].append(item['cid'])
            other_infos['task'].append(item['task'])
            other_infos['answer'].append(item['answer'])
            other_infos['smiles'].append(item['smiles'])
        
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
            'other_infos': dict(other_infos),  # Convert defaultdict to dict
        }
        
        return result


def create_hf_moleculeqa_datasets(
    data_path: str,
    tokenizer,
    llm_version: str,
    pad_idx: int,
    encoder_types: List[str],
    mol_type: str = 'mol',
    cache_dir: Optional[str] = None,
    streaming: bool = False,
):
    """
    Factory function to create HF-based MoleculeQA datasets and collator.
    
    Automatically handles:
    1. Local directory with train.jsonl/test.jsonl files
    2. HuggingFace Hub repository (e.g., 'username/dataset-name')
    
    Args:
        data_path: Path to data - can be:
                   - Local directory: 'data/moleculeqa_preprocessed/'
                   - HuggingFace Hub repo: 'username/moleculeqa-preprocessed'
        tokenizer: HuggingFace tokenizer
        llm_version: LLM version (llama2, llama3, qwen3, mistral, gemma)
        pad_idx: Padding index for UniMol
        encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
        mol_type: Molecular representation type ('mol', 'SMILES', etc.)
        cache_dir: Optional cache directory for HuggingFace datasets
        streaming: If True, use streaming mode (useful for very large datasets)
    
    Returns:
        tuple: (datasets_dict, collator)
        where datasets_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    """
    logger.info("=" * 80)
    logger.info("Loading preprocessed MoleculeQA dataset using HuggingFace datasets")
    logger.info(f"Data path: {data_path}")
    logger.info(f"LLM version: {llm_version}")
    logger.info(f"Mol type: {mol_type}")
    logger.info(f"Encoder types: {encoder_types}")
    logger.info(f"Streaming: {streaming}")
    
    # HuggingFace load_dataset automatically handles local/Hub
    logger.info(f"Loading dataset: {data_path}")
    dataset = load_dataset(data_path, cache_dir=cache_dir, streaming=streaming)
    
    # Create datasets for each split
    datasets = {}
    
    if 'train' in dataset:
        train_hf_dataset = dataset['train']
        datasets['train'] = HFMoleculeQADataset(
            train_hf_dataset, 
            encoder_types, 
            mol_type=mol_type,
            do_infer=False  # Training includes assistant responses
        )
        logger.info(f"✅ Train dataset: {len(datasets['train'])} samples")
    
    if 'validation' in dataset:
        val_hf_dataset = dataset['validation']
        datasets['val'] = HFMoleculeQADataset(
            val_hf_dataset,
            encoder_types,
            mol_type=mol_type,
            do_infer=True  # Validation is inference mode
        )
        logger.info(f"✅ Val dataset: {len(datasets['val'])} samples")
    elif 'test' in dataset:
        # Use test split as validation if no validation split exists
        test_hf_dataset = dataset['test']
        datasets['val'] = HFMoleculeQADataset(
            test_hf_dataset,
            encoder_types,
            mol_type=mol_type,
            do_infer=True
        )
        logger.info(f"✅ Val dataset (from test split): {len(datasets['val'])} samples")
    
    if 'test' in dataset:
        test_hf_dataset = dataset['test']
        datasets['test'] = HFMoleculeQADataset(
            test_hf_dataset,
            encoder_types,
            mol_type=mol_type,
            do_infer=True  # Test is inference mode
        )
        logger.info(f"✅ Test dataset: {len(datasets['test'])} samples")
    
    # Create collator
    collator = HFMoleculeQACollator(
        tokenizer=tokenizer,
        llm_version=llm_version,
        pad_idx=pad_idx,
        encoder_types=encoder_types
    )
    
    logger.info("✅ Datasets ready!")
    logger.info("=" * 80)
    
    return datasets, collator

