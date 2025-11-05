"""
Stage 2 Data Module for HuggingFace Transformers.
Pure HuggingFace implementation without PyTorch Lightning.
Includes data caching for faster loading using DATA_CACHE_DIR from environment.
"""
import os
import json
import torch
import pickle
from collections import defaultdict
from torch.utils.data import Dataset
from datasets import load_dataset
from torch_geometric.data import Batch
from transformers import BatchEncoding

from data_provider.collaters import Mol3DCollater
from data_provider.tokenization_utils import batch_tokenize_messages_list

# Get cache directory from environment variable (absolute path)
# Fallback to data/.cache if not set
CACHE_DIR = os.environ.get('DATA_CACHE_DIR', os.path.join(os.environ.get('DATA_DIR', './data'), '.cache'))


class Stage2HFDataset(Dataset):
    """
    HuggingFace-compatible dataset for Stage 2 finetuning.
    """
    def __init__(self, json_paths, mol_dataset):
        super().__init__()
        self.instruction_dataset = load_dataset("json", data_files=json_paths)['train']
        self.mol_dataset = mol_dataset
        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
    
    def __len__(self):
        return len(self.instruction_dataset)
    
    def __getitem__(self, index):
        text_data = self.instruction_dataset[index]
        
        cid = text_data['cid']
        data_graphs, data_others = self.mol_dataset[cid]
        
        messages = []
        messages.append({"role": "system", "content": text_data['system']})
        for turn in text_data['conversations']:
            messages.append({"role": "user", "content": turn['user'].replace('<mol>', self.mol_prompt)})
            messages.append({"role": "assistant", "content": turn['assistant']})
        
        # Return a flat dictionary for easier collation
        return {
            'data_graphs': data_graphs,
            'messages': messages,
            'brics_gids': data_others.get('brics_gids', None),
            'entropy_gids': data_others.get('entropy_gids', None),
        }


class Stage2HFCollator:
    """
    Pure HuggingFace data collator for Stage 2.
    Returns only tensors to avoid Accelerate device placement issues.
    """
    def __init__(self, tokenizer, llm_version, pad_idx, encoder_types):
        self.tokenizer = tokenizer
        self.llm_version = llm_version
        self.encoder_types = encoder_types
        if 'unimol' in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)
    
    def __call__(self, batch):
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dictionaries from Stage2HFDataset
            
        Returns:
            Dictionary with graph_batch, text_batch, brics_gids, entropy_gids (all tensors)
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
        
        # Process text data
        tokenized = batch_tokenize_messages_list(
            messages_list, 
            self.tokenizer,
            self.llm_version, 
            padding_side='left'
        )
        text_batch = tokenized
        
        # Return tensors only - brics_gids and entropy_gids as separate fields
        # This avoids nested dictionary issues with Accelerate
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


def create_stage2_dataset(
    tokenizer,
    llm_version,
    root,
    unimol_dictionary,
    encoder_types,
    data_types,
    test_mode=False,
    brics_gids_enable=False,
    entropy_gids_enable=False,
    use_cache=True,
):
    """
    Factory function to create Stage 2 dataset and collator.
    
    Args:
        tokenizer: Tokenizer for text processing
        llm_version: LLM version (llama2, llama3, etc.)
        root: Root directory for data files (absolute path)
        unimol_dictionary: UniMol dictionary
        encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
        data_types: List of data types to load
        test_mode: If True, use test dataset
        brics_gids_enable: Enable BRICS group IDs
        entropy_gids_enable: Enable entropy group IDs
        use_cache: If True, cache molecule dataset for faster loading (default: True)
    
    Returns:
        tuple: (dataset, collator)
    """
    from data_provider.mol_dataset import MolDataset_cid
    
    # Create cache directory if it doesn't exist
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f'Cache directory: {CACHE_DIR}')
    
    print('Loading molecule data...')
    
    # Determine which molecule file to use
    if test_mode:
        if brics_gids_enable or entropy_gids_enable:
            mol_file = 'pubchem-molecules-test_brics_entropy_gids.json'
        else:
            mol_file = 'pubchem-molecules-test.json'
        mol_dataset = None  # Don't load/cache for test mode
    else:
        if brics_gids_enable or entropy_gids_enable:
            mol_file = 'pubchem-molecules_brics_entropy_gids.json'
        else:
            mol_file = 'pubchem-molecules.json'
        
        # Create cache key based on file and encoder types
        cache_key = f"{mol_file.replace('.json', '')}_{'-'.join(encoder_types)}"
        cache_file = os.path.join(CACHE_DIR, f"mol_dataset_{cache_key}.pkl")
        
        # Try to load from cache
        mol_dataset = None
        if use_cache and os.path.exists(cache_file):
            print(f'Loading molecule dataset from cache: {cache_file}')
            try:
                with open(cache_file, 'rb') as f:
                    mol_dataset = pickle.load(f)
                print('✅ Molecule dataset loaded from cache')
            except Exception as e:
                print(f'⚠️  Failed to load cache ({e}), loading from scratch...')
                mol_dataset = None
        
        # If not loaded from cache, create and cache it
        if mol_dataset is None:
            mol_file_path = os.path.join(root, mol_file)
            print(f'Loading molecule data from: {mol_file_path}')
            data_list = json.load(open(mol_file_path))
            mol_dataset = MolDataset_cid(data_list, unimol_dictionary, encoder_types)
            
            # Save to cache
            if use_cache:
                print(f'Saving molecule dataset to cache: {cache_file}')
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(mol_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print('✅ Molecule dataset cached successfully')
                except Exception as e:
                    print(f'⚠️  Failed to save cache: {e}')
    
    # Load instruction dataset (HuggingFace datasets has built-in caching)
    json_paths = [os.path.join(root, f'{data_type}.json') for data_type in data_types]
    print(f'Loading instruction data from: {json_paths}')
    
    dataset = Stage2HFDataset(
        json_paths=json_paths,
        mol_dataset=mol_dataset
    )
    
    print(f'✅ Dataset ready with {len(dataset)} samples')
    
    collator = Stage2HFCollator(
        tokenizer=tokenizer,
        llm_version=llm_version,
        pad_idx=unimol_dictionary.pad(),
        encoder_types=encoder_types
    )
    
    return dataset, collator

