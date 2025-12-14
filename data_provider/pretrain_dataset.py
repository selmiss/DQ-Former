"""
Stage 1 Data Module for HuggingFace Transformers.
Pure HuggingFace implementation without PyTorch Lightning.
Includes data caching for faster loading using DATA_CACHE_DIR from environment.
"""
import os
import json
import torch
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from data_provider.collaters import Mol3DCollater

# Get cache directory from environment variable (absolute path)
# Fallback to data/.cache if not set
CACHE_DIR = os.environ.get('DATA_CACHE_DIR', os.path.join(os.environ.get('DATA_DIR', './data'), '.cache'))


class PretainDataset(Dataset):
    """
    HuggingFace-compatible dataset for Stage 1 pretraining.
    """
    def __init__(self, data_list, mol_dataset):
        super().__init__()
        self.data_list = data_list
        self.mol_dataset = mol_dataset
        self.cid2idx = {data['cid']: idx for idx, data in enumerate(self.data_list)}
        self.idx2cid = {v: k for k, v in self.cid2idx.items()}
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        cid = self.idx2cid[index]
        data_graphs, data_others = self.mol_dataset[cid]
        
        return {
            'data_graphs': data_graphs,
            'iupac_name': data_others['iupac_name'],
            'brics_gids': data_others.get('brics_gids', None),
            'entropy_gids': data_others.get('entropy_gids', None),
        }


class PretainCollator:
    """
    Pure HuggingFace data collator for Stage 1.
    Returns only tensors to avoid Accelerate device placement issues.
    """
    def __init__(self, tokenizer, text_max_len, pad_idx, encoder_types):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.encoder_types = encoder_types
        if 'unimol' in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)
    
    def __call__(self, batch):
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dictionaries from PretainDataset
            
        Returns:
            Dictionary with:
                - graph_batch: Dictionary containing encoder data, brics_gids, and entropy_gids
                - text_batch: Tokenized text data
                - iupac_names: List of IUPAC names
        """
        # Extract fields from batch
        data_graphs = [item['data_graphs'] for item in batch]
        iupac_names = [item['iupac_name'] for item in batch]
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


def create_pretrain_datasets(
    unimol_dictionary,
    scibert_tokenizer,
    encoder_types,
    text_max_len,
    root,
    test_mode=False,
    brics_gids_enable=False,
    entropy_gids_enable=False,
    use_cache=True,
):
    """
    Factory function to create Stage 1 train/val datasets and collator.
    
    Args:
        unimol_dictionary: UniMol dictionary
        scibert_tokenizer: SciBERT tokenizer
        encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
        text_max_len: Maximum text length
        root: Root directory for data files (absolute path)
        test_mode: If True, use test dataset
        brics_gids_enable: Enable BRICS group IDs
        entropy_gids_enable: Enable entropy group IDs
        use_cache: If True, cache molecule dataset for faster loading (default: True)
    
    Returns:
        tuple: (train_dataset, val_dataset, collator)
    """
    from data_provider.mol_dataset import MolDataset_cid
    
    # Create cache directory if it doesn't exist
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f'Cache directory: {CACHE_DIR}')
    
    # Determine which molecule file to use
    if test_mode:
        if brics_gids_enable or entropy_gids_enable:
            mol_file = 'pubchem-molecules-test_brics_entropy_gids.json'
        else:
            mol_file = 'pubchem-molecules-test.json'
        mol_dataset = None  # Don't load/cache for test mode
        data_list = json.load(open(os.path.join(root, mol_file)))
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
        else:
            # Load data_list for splitting
            mol_file_path = os.path.join(root, mol_file)
            data_list = json.load(open(mol_file_path))
    
    # Split data
    train_data_list = [data for data in data_list if data['split'] == 'pretrain']
    val_data_list = [data for data in data_list if data['split'] == 'valid']
    
    print(f'Creating train dataset with {len(train_data_list)} samples...')
    train_dataset = PretainDataset(
        data_list=train_data_list,
        mol_dataset=mol_dataset
    )
    
    print(f'Creating validation dataset with {len(val_data_list)} samples...')
    val_dataset = PretainDataset(
        data_list=val_data_list,
        mol_dataset=mol_dataset
    )
    
    print(f'✅ Datasets ready: {len(train_dataset)} train, {len(val_dataset)} val')
    
    collator = PretainCollator(
        tokenizer=scibert_tokenizer,
        text_max_len=text_max_len,
        pad_idx=unimol_dictionary.pad(),
        encoder_types=encoder_types
    )
    
    return train_dataset, val_dataset, collator

