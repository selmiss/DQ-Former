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
    # Only include x, edge_index, edge_attr - no extra attributes
    # Let PyG auto-infer num_nodes from x.size(0) to avoid collation issues
    # Note: preprocessing uses 'edge_feat' but we need 'edge_attr' for PyG
    edge_feat_key = 'edge_feat' if 'edge_feat' in data_dict else 'edge_attr'
    edge_attr = torch.LongTensor(data_dict[edge_feat_key])
    # Handle empty edge_attr: reshape [0] to [0, 3] for molecules without edges
    if edge_attr.numel() == 0:
        edge_attr = edge_attr.new_zeros((0, 3))
    return Data(
        x=torch.LongTensor(data_dict['node_feat']),
        edge_index=torch.LongTensor(data_dict['edge_index']),
        edge_attr=edge_attr
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
        
        Supports:
        - Single-molecule tasks (mol_qa): graph_data is a dict, smiles/selfies are strings
        - Multi-molecule tasks (reactions): graph_data/smiles/selfies are lists
        - Text-only tasks (mol_design): no graph_data field
        
        Returns:
            Dictionary with:
                - data_graphs: Dict with encoder-specific graph data (list per encoder)
                - messages: List of message dicts (with roles and content)
                - cid: Compound ID
                - task: Task category
                - answer: Ground truth answer
                - smiles: SMILES string or list
                - brics_gids: List or list of lists or None
                - entropy_gids: List or list of lists or None
        """
        item = self.hf_dataset[idx]
        
        # Deserialize graph data (if present)
        data_graphs = {}
        has_graph_data = 'graph_data' in item and item['graph_data'] is not None
        
        if has_graph_data:
            graph_data_raw = json.loads(item['graph_data']) if isinstance(item['graph_data'], str) else item['graph_data']
            
            # Check if multi-molecule (list of dicts) or single-molecule (single dict)
            is_multi_molecule = isinstance(graph_data_raw, list)
            
            if is_multi_molecule:
                # Multi-molecule: graph_data is a list of dicts
                if 'unimol' in self.encoder_types:
                    data_graphs['unimol'] = []
                    for graph_dict in graph_data_raw:
                        if 'unimol' in graph_dict:
                            data_graphs['unimol'].append(deserialize_unimol_data(graph_dict['unimol']))
                
                if 'moleculestm' in self.encoder_types:
                    data_graphs['moleculestm'] = []
                    for graph_dict in graph_data_raw:
                        if 'moleculestm' in graph_dict:
                            data_graphs['moleculestm'].append(deserialize_moleculestm_data(graph_dict['moleculestm']))
            else:
                # Single-molecule: graph_data is a single dict
                if 'unimol' in self.encoder_types and 'unimol' in graph_data_raw:
                    data_graphs['unimol'] = [deserialize_unimol_data(graph_data_raw['unimol'])]
                
                if 'moleculestm' in self.encoder_types and 'moleculestm' in graph_data_raw:
                    data_graphs['moleculestm'] = [deserialize_moleculestm_data(graph_data_raw['moleculestm'])]
        
        # Parse conversations (convert from JSON if needed)
        conversations = json.loads(item['conversations']) if isinstance(item['conversations'], str) else item['conversations']
        
        # Build messages
        messages = []
        messages.append({"role": "system", "content": item['system']})
        
        # Get smiles/selfies (handle both single string and list)
        smiles_data = item.get('smiles', '')
        selfies_data = item.get('selfies', '')
        is_multi_smiles = isinstance(smiles_data, list)
        
        # Process conversations (typically only 1 for MoleculeQA/reactions)
        answer = None
        for turn in conversations:
            user_prompt = turn['user']
            
            # Replace <mol> tokens with appropriate representation
            if '<mol>' in user_prompt:
                if self.mol_type == 'mol':
                    # Replace all <mol> with mol_prompt
                    user_prompt = user_prompt.replace('<mol>', self.mol_prompt)
                    
                elif self.mol_type == 'SMILES':
                    if is_multi_smiles:
                        # Multi-molecule: replace sequentially
                        for smiles in smiles_data:
                            user_prompt = user_prompt.replace('<mol>', smiles, 1)
                    else:
                        user_prompt = user_prompt.replace('<mol>', smiles_data)
                        
                elif self.mol_type == "SMILES,mol":
                    if is_multi_smiles:
                        # Multi-molecule: replace sequentially with SMILES + mol_prompt
                        for smiles in smiles_data:
                            user_prompt = user_prompt.replace('<mol>', smiles + ' ', 1)
                        user_prompt = user_prompt + 'Molecules Embedding: ' + self.mol_prompt
                    else:
                        user_prompt = user_prompt.replace('<mol>', smiles_data + ' ' + self.mol_prompt)
                        
                elif self.mol_type == 'SELFIES':
                    if is_multi_smiles:  # selfies will also be a list
                        # Multi-molecule: replace sequentially
                        for selfies in selfies_data:
                            user_prompt = user_prompt.replace('<mol>', selfies, 1)
                    else:
                        user_prompt = user_prompt.replace('<mol>', selfies_data)
                        
                elif self.mol_type == "SELFIES,mol":
                    if is_multi_smiles:  # selfies will also be a list
                        # Multi-molecule: replace sequentially with SELFIES + mol_prompt
                        for selfies in selfies_data:
                            user_prompt = user_prompt.replace('<mol>', selfies + ' ', 1)
                        user_prompt = user_prompt + 'Molecules Embedding: ' + self.mol_prompt
                    else:
                        user_prompt = user_prompt.replace('<mol>', selfies_data + ' ' + self.mol_prompt)
                        
                elif self.mol_type == "SMILES,<graph>":
                    if is_multi_smiles:
                        # Multi-molecule: replace sequentially with SMILES + graph_prompt
                        for smiles in smiles_data:
                            user_prompt = user_prompt.replace('<mol>', smiles + ' ' + self.graph_prompt, 1)
                    else:
                        user_prompt = user_prompt.replace('<mol>', smiles_data + ' ' + self.graph_prompt)
                else:
                    # Default: use mol_prompt
                    user_prompt = user_prompt.replace('<mol>', self.mol_prompt)
            
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
            'smiles': smiles_data,  # Can be string or list
        }


def batch_multi_molecule_data(moleculestm_list, unimol_list, brics_gids_list, entropy_gids_list):
    """
    Custom batching function for multi-molecule data.
    Concatenates multiple molecules into one disconnected graph with proper index offsetting.
    
    Args:
        moleculestm_list: List of PyG Data objects (MoleculeSTM graphs)
        unimol_list: List of UniMol dictionaries
        brics_gids_list: List of BRICS group ID lists
        entropy_gids_list: List of entropy group ID lists
    
    Returns:
        Tuple of (batched_moleculestm, batched_unimol, batched_brics_gids, batched_entropy_gids)
        Note: inner_cluster is already embedded in batched_moleculestm and batched_unimol
    """
    if not moleculestm_list and not unimol_list:
        return None, None, None, None
    
    # Batch MoleculeSTM graphs
    batched_moleculestm = None
    if moleculestm_list:
        # Concatenate node features
        x_list = [g.x for g in moleculestm_list]
        x_concat = torch.cat(x_list, dim=0)
        
        # Concatenate edges with proper node offset
        edge_index_list = []
        edge_attr_list = []
        inner_cluster = []
        node_offset = 0
        
        for mol_idx, g in enumerate(moleculestm_list):
            # Offset edge indices by cumulative node count
            edge_index_list.append(g.edge_index + node_offset)
            edge_attr_list.append(g.edge_attr)
            # Track which molecule each node belongs to
            n_nodes = g.x.size(0)
            inner_cluster.extend([mol_idx] * n_nodes)
            node_offset += n_nodes
        
        edge_index_concat = torch.cat(edge_index_list, dim=1)
        edge_attr_concat = torch.cat(edge_attr_list, dim=0)
        
        # Create disconnected graph
        batched_moleculestm = Data(
            x=x_concat,
            edge_index=edge_index_concat,
            edge_attr=edge_attr_concat,
            inner_cluster=torch.LongTensor(inner_cluster)
        )
    
    # Batch UniMol data
    batched_unimol = None
    if unimol_list:
        # Concatenate tokens along sequence dimension
        concat_tokens = torch.cat([m['src_tokens'] for m in unimol_list], dim=0)
        
        # Create block diagonal matrices for edge_type and distance
        n_atoms_list = [m['src_tokens'].size(0) for m in unimol_list]
        total_atoms = sum(n_atoms_list)
        
        concat_edge_type = torch.zeros(total_atoms, total_atoms, dtype=unimol_list[0]['src_edge_type'].dtype)
        concat_distance = torch.zeros(total_atoms, total_atoms, dtype=unimol_list[0]['src_distance'].dtype)
        
        offset = 0
        inner_cluster = []
        for mol_idx, (m, n_atoms) in enumerate(zip(unimol_list, n_atoms_list)):
            concat_edge_type[offset:offset+n_atoms, offset:offset+n_atoms] = m['src_edge_type'][:n_atoms, :n_atoms]
            concat_distance[offset:offset+n_atoms, offset:offset+n_atoms] = m['src_distance'][:n_atoms, :n_atoms]
            inner_cluster.extend([mol_idx] * n_atoms)
            offset += n_atoms
        
        batched_unimol = {
            'src_tokens': concat_tokens,
            'src_edge_type': concat_edge_type,
            'src_distance': concat_distance,
            'inner_cluster': torch.LongTensor(inner_cluster)
        }
    
    # Batch BRICS group IDs with proper group offset
    batched_brics_gids = None
    if brics_gids_list and any(g is not None for g in brics_gids_list):
        concatenated_gids = []
        group_offset = 0
        
        for gids in brics_gids_list:
            if gids is None:
                continue
            # Offset group IDs to avoid collision
            # Example: [0,0,1,1] + [0,0,1,1] -> [0,0,1,1,2,2,3,3]
            offset_gids = [g + group_offset for g in gids]
            concatenated_gids.extend(offset_gids)
            # Update offset for next molecule
            group_offset = max(offset_gids) + 1 if offset_gids else group_offset
        
        batched_brics_gids = concatenated_gids if concatenated_gids else None
    
    # Batch entropy group IDs with proper group offset
    batched_entropy_gids = None
    if entropy_gids_list and any(g is not None for g in entropy_gids_list):
        concatenated_gids = []
        group_offset = 0
        
        for gids in entropy_gids_list:
            if gids is None:
                continue
            # Offset group IDs to avoid collision
            offset_gids = [g + group_offset for g in gids]
            concatenated_gids.extend(offset_gids)
            # Update offset for next molecule
            group_offset = max(offset_gids) + 1 if offset_gids else group_offset
        
        batched_entropy_gids = concatenated_gids if concatenated_gids else None
    
    return batched_moleculestm, batched_unimol, batched_brics_gids, batched_entropy_gids


class HFMoleculeQACollator:
    """
    Data collator for HF-loaded preprocessed MoleculeQA data.
    Compatible with the original MoleculeQAHFCollator interface.
    """
    
    def __init__(self, tokenizer, llm_version, pad_idx, encoder_types, max_input_length=None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            llm_version: LLM version (llama2, llama3, qwen3, mistral, gemma)
            pad_idx: Padding index for UniMol
            encoder_types: List of encoder types
            max_input_length: Maximum input token length for truncation (None = no truncation)
        """
        self.tokenizer = tokenizer
        self.llm_version = llm_version
        self.encoder_types = encoder_types
        self.max_input_length = max_input_length
        if max_input_length is not None:
            logger.info(f"⚠️  Max input length set to {max_input_length} tokens. Sequences exceeding this will be truncated.")
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
        
        # Process graph data: batch multi-molecule inputs into single graphs per sample
        graph_batch = {}
        
        # First, for each sample, batch its molecules into one graph
        batched_data_graphs = []
        batched_brics_gids = []
        batched_entropy_gids = []
        
        for sample_idx, (data, brics, entropy) in enumerate(zip(data_graphs, brics_gids_list, entropy_gids_list)):
            # Extract molecules for this sample
            moleculestm_mols = data.get('moleculestm', []) if 'moleculestm' in self.encoder_types else []
            unimol_mols = data.get('unimol', []) if 'unimol' in self.encoder_types else []
            
            # Extract gids for this sample
            # If gids are list of lists (multi-molecule), extract each; otherwise use as-is
            if isinstance(brics, list) and len(brics) > 0 and isinstance(brics[0], list):
                # Multi-molecule
                brics_mol_list = brics
            elif brics is not None:
                # Single-molecule
                brics_mol_list = [brics]
            else:
                brics_mol_list = []
            
            if isinstance(entropy, list) and len(entropy) > 0 and isinstance(entropy[0], list):
                # Multi-molecule
                entropy_mol_list = entropy
            elif entropy is not None:
                # Single-molecule
                entropy_mol_list = [entropy]
            else:
                entropy_mol_list = []
            
            # Check if multi-molecule
            if len(moleculestm_mols) > 1 or len(unimol_mols) > 1:
                # Multi-molecule: batch them into one graph with proper offsetting
                batched_stm, batched_uni, batched_brics, batched_entropy = batch_multi_molecule_data(
                    moleculestm_mols, unimol_mols, brics_mol_list, entropy_mol_list
                )
                
                # Store batched data
                sample_data = {}
                if batched_stm is not None:
                    sample_data['moleculestm'] = batched_stm
                if batched_uni is not None:
                    sample_data['unimol'] = batched_uni
                
                batched_data_graphs.append(sample_data)
                batched_brics_gids.append(batched_brics)
                batched_entropy_gids.append(batched_entropy)
            else:
                # Single-molecule: use as-is
                sample_data = {}
                if moleculestm_mols:
                    sample_data['moleculestm'] = moleculestm_mols[0]
                if unimol_mols:
                    sample_data['unimol'] = unimol_mols[0]
                
                batched_data_graphs.append(sample_data)
                batched_brics_gids.append(brics_mol_list[0] if brics_mol_list else None)
                batched_entropy_gids.append(entropy_mol_list[0] if entropy_mol_list else None)
        
        # Now batch across samples using standard collaters
        if 'unimol' in self.encoder_types:
            data_unimol = [d['unimol'] for d in batched_data_graphs if 'unimol' in d]
            if data_unimol:
                # Check if any have inner_cluster (multi-molecule samples)
                inner_clusters = []
                clean_data = []
                for d in data_unimol:
                    if isinstance(d, dict) and 'inner_cluster' in d:
                        inner_clusters.append(d['inner_cluster'])
                        clean_d = {k: v for k, v in d.items() if k != 'inner_cluster'}
                        clean_data.append(clean_d)
                    else:
                        inner_clusters.append(None)
                        clean_data.append(d)
                
                # Collate
                graph_batch['unimol'] = self.d3_collater(clean_data)
                
                # Add inner_cluster back if present
                if any(ic is not None for ic in inner_clusters):
                    all_inner_cluster = []
                    inner_cluster_batch = []
                    for batch_idx, ic in enumerate(inner_clusters):
                        if ic is not None:
                            all_inner_cluster.append(ic)
                            inner_cluster_batch.extend([batch_idx] * len(ic))
                    
                    if all_inner_cluster:
                        graph_batch['unimol_inner_cluster'] = torch.cat(all_inner_cluster)
                        graph_batch['unimol_inner_cluster_batch'] = torch.LongTensor(inner_cluster_batch)
        
        if 'moleculestm' in self.encoder_types:
            data_moleculestm = [d['moleculestm'] for d in batched_data_graphs if 'moleculestm' in d]
            if data_moleculestm:
                # Extract inner_cluster separately (same pattern as unimol)
                # Some samples have inner_cluster (multi-molecule), others don't (single-molecule)
                inner_clusters = []
                clean_data = []
                for g in data_moleculestm:
                    if hasattr(g, 'inner_cluster'):
                        inner_clusters.append(g.inner_cluster)
                        # Create new Data without inner_cluster to avoid collation issues
                        clean_g = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
                        clean_data.append(clean_g)
                    else:
                        inner_clusters.append(None)
                        clean_data.append(g)
                
                # Batch without inner_cluster
                graph_batch['moleculestm'] = Batch.from_data_list(clean_data)
                
                # Add inner_cluster back manually if present
                if any(ic is not None for ic in inner_clusters):
                    all_inner_cluster = []
                    inner_cluster_batch = []
                    for batch_idx, ic in enumerate(inner_clusters):
                        if ic is not None:
                            all_inner_cluster.append(ic)
                            inner_cluster_batch.extend([batch_idx] * len(ic))
                    
                    if all_inner_cluster:
                        graph_batch['moleculestm'].inner_cluster = torch.cat(all_inner_cluster)
                        graph_batch['moleculestm'].inner_cluster_batch = torch.LongTensor(inner_cluster_batch)
        
        # Process text data (tokenize based on LLM version)
        text_batch = batch_tokenize_messages_list(
            messages_list,
            self.tokenizer,
            self.llm_version,
            padding_side='left',
            max_input_length=self.max_input_length
        )
        
        # Use batched gids (already properly offset for multi-molecule samples)
        # batched_brics_gids and batched_entropy_gids were created earlier
        graph_batch['brics_gids'] = batched_brics_gids if any(g is not None for g in batched_brics_gids) else None
        graph_batch['entropy_gids'] = batched_entropy_gids if any(g is not None for g in batched_entropy_gids) else None
        
        # Normalize inner_cluster storage to top-level (encoder-agnostic)
        # Extract from either encoder and store at top level for easier access
        if 'moleculestm' in graph_batch and hasattr(graph_batch['moleculestm'], 'inner_cluster'):
            graph_batch['inner_cluster'] = graph_batch['moleculestm'].inner_cluster
            graph_batch['inner_cluster_batch'] = graph_batch['moleculestm'].inner_cluster_batch
        elif 'unimol_inner_cluster' in graph_batch:
            graph_batch['inner_cluster'] = graph_batch['unimol_inner_cluster']
            graph_batch['inner_cluster_batch'] = graph_batch['unimol_inner_cluster_batch']
        
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
    max_input_length: Optional[int] = None,
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
        max_input_length: Maximum input token length for truncation (None = no truncation)
    
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
        encoder_types=encoder_types,
        max_input_length=max_input_length
    )
    
    logger.info("✅ Datasets ready!")
    logger.info("=" * 80)
    
    return datasets, collator

