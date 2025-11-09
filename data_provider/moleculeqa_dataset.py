"""
MoleculeQA Data Module for HuggingFace Transformers.
Pure HuggingFace implementation aligned with stage2_hf_dm.py pattern.
"""
import os
import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from datasets import load_dataset
from torch_geometric.data import Batch
from transformers import BatchEncoding

from data_provider.collaters import Mol3DCollater
from data_provider.tokenization_utils import batch_tokenize_messages_list
from utils.cache_utils import (
    get_cache_dir,
    get_cache_path,
    is_cache_valid,
    load_cache,
    save_cache
)

# Get cache directory from environment variable (absolute path)
CACHE_DIR = get_cache_dir()


class MoleculeQAHFDataset(Dataset):
    """
    HuggingFace-compatible dataset for MoleculeQA.
    Aligned with Stage2HFDataset pattern.
    """
    def __init__(self, json_path, mol_dataset, mol_type='mol', do_infer=False):
        super().__init__()
        self.instruction_dataset = load_dataset("json", data_files=[json_path])['train']
        self.mol_dataset = mol_dataset
        self.mol_type = mol_type
        self.do_infer = do_infer
        
        # Prompts for different representations
        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
        self.graph_prompt = "<graph>" * 28
    
    def __len__(self):
        return len(self.instruction_dataset)
    
    def __getitem__(self, index):
        text_data = self.instruction_dataset[index]
        
        cid = text_data['cid']
        data_graphs, data_others = self.mol_dataset[cid]
        
        messages = []
        messages.append({"role": "system", "content": text_data['system']})
        
        # Handle conversations (typically only 1 for MoleculeQA)
        assert len(text_data['conversations']) == 1, "Only one conversation is expected in the dataset."
        
        for turn in text_data['conversations']:
            # Replace <mol> with appropriate representation
            if self.mol_type == 'mol':
                user_prompt = turn['user'].replace('<mol>', self.mol_prompt)
            elif self.mol_type == 'SMILES':
                user_prompt = turn['user'].replace('<mol>', text_data['smiles'])
            elif self.mol_type == "SMILES,mol":
                user_prompt = turn['user'].replace('<mol>', text_data['smiles'][:96] + ' ' + self.mol_prompt)
            elif self.mol_type == 'SELFIES':
                user_prompt = turn['user'].replace('<mol>', text_data['selfies'])
            elif self.mol_type == "SMILES,<graph>":
                user_prompt = turn['user'].replace('<mol>', text_data['smiles'][:128] + ' ' + self.graph_prompt)
            else:
                user_prompt = turn['user'].replace('<mol>', self.mol_prompt)
            
            messages.append({"role": "user", "content": user_prompt})
            
            # Only add assistant response if not inference
            if not self.do_infer:
                messages.append({"role": "assistant", "content": turn['assistant']})
            
            answer = turn['assistant']
        
        # Return a flat dictionary for easier collation (same pattern as Stage2)
        return {
            'data_graphs': data_graphs,
            'messages': messages,
            'brics_gids': data_others.get('brics_gids', None),
            'entropy_gids': data_others.get('entropy_gids', None),
            # Additional info for metrics computation
            'cid': cid,
            'task': text_data.get('category', None),
            'answer': answer,
            'smiles': text_data['smiles'],
        }


class MoleculeQAHFCollator:
    """
    Pure HuggingFace data collator for MoleculeQA.
    Returns only tensors to avoid Accelerate device placement issues.
    Aligned with Stage2HFCollator pattern.
    """
    def __init__(self, tokenizer, llama_version, pad_idx, encoder_types):
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.encoder_types = encoder_types
        if 'unimol' in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)
    
    def __call__(self, batch):
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dictionaries from MoleculeQAHFDataset
            
        Returns:
            Dictionary with graph_batch, text_batch, brics_gids, entropy_gids, other_infos (all tensors/lists)
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
        
        # Process text data
        tokenized = batch_tokenize_messages_list(
            messages_list, 
            self.tokenizer,
            self.llama_version, 
            padding_side='left'
        )
        text_batch = tokenized
        
        # Return tensors only - same pattern as Stage2HFCollator
        result = {
            'graph_batch': graph_batch,
            'text_batch': text_batch,
            'other_infos': dict(other_infos),  # Convert defaultdict to dict
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


def create_moleculeqa_dataset(
    tokenizer,
    llama_version,
    root,
    unimol_dictionary,
    encoder_types,
    mol_type='mol',
    split='train',
    do_infer=False,
    brics_gids_enable=False,
    entropy_gids_enable=False,
    use_cache=True,
    limit_samples=None,
):
    """
    Factory function to create MoleculeQA dataset and collator for a single split.
    Aligned with create_stage2_dataset pattern.
    
    Args:
        tokenizer: Tokenizer for text processing
        llm_version: LLM version (llama2, llama3, etc.)
        root: Root directory for data files (absolute path)
        unimol_dictionary: UniMol dictionary
        encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
        mol_type: Molecular representation type
        split: Data split ('train', 'test', 'val')
        do_infer: If True, don't include assistant responses
        brics_gids_enable: Enable BRICS group IDs
        entropy_gids_enable: Enable entropy group IDs
        use_cache: If True, cache molecule dataset for faster loading (default: True)
        limit_samples: Optional limit on number of samples
    
    Returns:
        tuple: (dataset, collator)
    """
    from data_provider.mol_dataset import MolDataset_cid
    
    # Create cache directory if it doesn't exist
    if use_cache:
        cache_dir = get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        print(f'Cache directory: {cache_dir}')
    
    # Determine which molecule file to use
    if split == 'train':
        mol_file = 'train_mol.json'
    else:  # test or val
        mol_file = 'test_mol.json'
    
    # Get paths
    mol_file_path = os.path.join(root, mol_file)
    if not os.path.exists(mol_file_path):
        raise FileNotFoundError(f"Molecule file not found: {mol_file_path}")
    
    cache_path = get_cache_path(root, mol_file, encoder_types)
    
    # Try to load from cache
    mol_dataset = None
    
    if use_cache and is_cache_valid(cache_path, mol_file_path):
        mol_dataset = load_cache(cache_path, verbose=True)
    else:
        if use_cache and os.path.exists(cache_path):
            print(f'⚠️  Cache is outdated (source file modified), rebuilding...')
    
    # If not loaded from cache, create and cache it
    if mol_dataset is None:
        print(f'Loading molecule data from: {mol_file_path}')
        data_list = json.load(open(mol_file_path))
        mol_dataset = MolDataset_cid(data_list, unimol_dictionary, encoder_types)
        
        # Save to cache
        if use_cache:
            save_cache(mol_dataset, cache_path, verbose=True)
    
    # Load instruction dataset (HuggingFace datasets has built-in caching)
    json_path = os.path.join(root, f'{split}.json')
    print(f'Loading instruction data from: {json_path}')
    
    dataset = MoleculeQAHFDataset(
        json_path=json_path,
        mol_dataset=mol_dataset,
        mol_type=mol_type,
        do_infer=do_infer
    )
    
    # Apply limit if specified
    if limit_samples is not None:
        limit = max(0, int(limit_samples))
        if limit < len(dataset):
            # Create a limited version by selecting indices
            dataset.instruction_dataset = dataset.instruction_dataset.select(range(limit))
            print(f'⚠️  Limited dataset to {limit} samples')
    
    print(f'✅ Dataset ready with {len(dataset)} samples')
    
    collator = MoleculeQAHFCollator(
        tokenizer=tokenizer,
        llama_version=llama_version,
        pad_idx=unimol_dictionary.pad(),
        encoder_types=encoder_types
    )
    
    return dataset, collator


def create_moleculeqa_datasets(
    tokenizer,
    llama_version,
    root,
    unimol_dictionary,
    encoder_types,
    mol_type='mol',
    train_limit=None,
    val_limit=None,
    test_limit=None,
    brics_gids_enable=False,
    entropy_gids_enable=False,
    use_cache=True,
):
    """
    Factory function to create train/val/test datasets for MoleculeQA.
    Returns datasets and a single collator (collator is the same for all splits).
    
    Args:
        tokenizer: Tokenizer for text processing
        llama_version: LLM version (llama2, llama3, etc.)
        root: Root directory for data files
        unimol_dictionary: UniMol dictionary
        encoder_types: List of encoder types
        mol_type: Molecular representation type
        train_limit: Optional limit on training samples
        val_limit: Optional limit on validation samples
        test_limit: Optional limit on test samples
        brics_gids_enable: Enable BRICS group IDs
        entropy_gids_enable: Enable entropy group IDs
        use_cache: If True, cache molecule dataset
    
    Returns:
        tuple: (datasets_dict, collator)
        where datasets_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    """
    # Create train dataset
    train_dataset, collator = create_moleculeqa_dataset(
        tokenizer=tokenizer,
        llama_version=llama_version,
        root=root,
        unimol_dictionary=unimol_dictionary,
        encoder_types=encoder_types,
        mol_type=mol_type,
        split='train',
        do_infer=False,
        brics_gids_enable=brics_gids_enable,
        entropy_gids_enable=entropy_gids_enable,
        use_cache=use_cache,
        limit_samples=train_limit,
    )
    
    # Create val dataset (using test split, inference mode)
    val_dataset, _ = create_moleculeqa_dataset(
        tokenizer=tokenizer,
        llama_version=llama_version,
        root=root,
        unimol_dictionary=unimol_dictionary,
        encoder_types=encoder_types,
        mol_type=mol_type,
        split='test',
        do_infer=True,
        brics_gids_enable=brics_gids_enable,
        entropy_gids_enable=entropy_gids_enable,
        use_cache=use_cache,
        limit_samples=val_limit,
    )
    
    # Create test dataset (using test split, inference mode)
    test_dataset, _ = create_moleculeqa_dataset(
        tokenizer=tokenizer,
        llama_version=llama_version,
        root=root,
        unimol_dictionary=unimol_dictionary,
        encoder_types=encoder_types,
        mol_type=mol_type,
        split='test',
        do_infer=True,
        brics_gids_enable=brics_gids_enable,
        entropy_gids_enable=entropy_gids_enable,
        use_cache=use_cache,
        limit_samples=test_limit,
    )
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    return datasets, collator
