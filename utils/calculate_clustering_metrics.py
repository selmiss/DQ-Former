"""
Calculate ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information) between clustering methods.

This script reads cluster IDs from a single JSONL file and compares FOUR clustering methods:
1. Key1 (e.g., entropy_gids)
2. Key2 (e.g., brics_gids)
3. RECAP (computed from SMILES using RDKit with timeout protection)
4. Random baseline (shuffled version of Key2)

ARI and NMI are calculated for EACH molecule individually for all pairwise comparisons,
then aggregated to show mean, median, std, etc. in matrix form.

Features:
- Progress bar showing: molecules processed, valid count, and skipped count
- RECAP fragmentation with 3-second timeout (skips slow molecules)

Example usage:
    python calculate_clustering_metrics.py \\
        --file data/train.jsonl \\
        --key1 entropy_ids \\
        --key2 brics_ids
"""
import argparse
import json
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from rdkit import Chem
from rdkit.Chem import Recap
import random
from tqdm import tqdm
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    """Exception raised when a timeout occurs."""
    pass


@contextmanager
def time_limit(seconds):
    """
    Context manager to limit execution time.
    Raises TimeoutException if the code block takes longer than specified seconds.
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    
    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def get_recap_cluster_ids(smiles: str, timeout_seconds: int = 3) -> Optional[List[int]]:
    """
    Generate RECAP-based cluster IDs for a molecule with timeout protection.
    Each atom is assigned a cluster ID based on which RECAP fragment it belongs to.
    
    If RECAP takes longer than timeout_seconds, returns None (molecule will be skipped).
    
    Args:
        smiles: SMILES string of the molecule
        timeout_seconds: Maximum time to spend on RECAP (default 3 seconds)
    
    Returns:
        List of cluster IDs in atom order, or None if fragmentation fails or times out
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        num_atoms = mol.GetNumAtoms()
        
        # Initialize all atoms to cluster 0
        cluster_ids = [0] * num_atoms
        
        # Use timeout protection for RECAP fragmentation
        try:
            with time_limit(timeout_seconds):
                # Perform RECAP fragmentation
                recap_tree = Recap.RecapDecompose(mol)
                
                # Get all leaf nodes (actual fragments)
                leaves = recap_tree.GetLeaves()
                
                if len(leaves) <= 1:
                    # No fragmentation occurred, all atoms in one cluster
                    return cluster_ids
                
                # Assign cluster IDs based on fragments
                cluster_id = 0
                assigned = set()
                
                for leaf_smiles in leaves.keys():
                    leaf_mol = Chem.MolFromSmiles(leaf_smiles)
                    if leaf_mol is None:
                        continue
                    
                    # Find matching substructure in original molecule
                    matches = mol.GetSubstructMatches(leaf_mol)
                    
                    if matches:
                        # Use the first match
                        for atom_idx in matches[0]:
                            if atom_idx not in assigned:
                                cluster_ids[atom_idx] = cluster_id
                                assigned.add(atom_idx)
                        
                        cluster_id += 1
                
                # Assign any unassigned atoms to their own clusters
                for i in range(num_atoms):
                    if i not in assigned:
                        cluster_ids[i] = cluster_id
                        cluster_id += 1
                
                return cluster_ids
                
        except TimeoutException:
            # RECAP took too long, return None to skip this molecule
            return None
        
    except Exception as e:
        return None


def shuffle_cluster_ids(cluster_ids: List[int], seed: Optional[int] = None) -> List[int]:
    """
    Shuffle cluster IDs randomly while maintaining the same structure.
    
    Args:
        cluster_ids: Original cluster ID list
        seed: Random seed for reproducibility
    
    Returns:
        Shuffled cluster ID list
    """
    shuffled = cluster_ids.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled)
    return shuffled


def load_and_calculate_per_molecule(file_path: str, key1: str, key2: str) -> Tuple[List[np.ndarray], List[np.ndarray], int, int, Dict]:
    """
    Load JSONL file and calculate ARI/NMI for each molecule with 4 clustering methods.
    Methods: key1, key2, RECAP (from SMILES with timeout), Random (shuffled key2)
    
    Calculates all pairwise comparisons and returns matrices.
    
    Args:
        file_path: Path to JSONL file
        key1: First key name containing cluster IDs
        key2: Second key name containing cluster IDs
    
    Returns:
        Tuple of (ari_matrices, nmi_matrices, total_lines, skipped_lines, method_names)
    """
    ari_matrices = []  # List of 4x4 matrices (one per molecule)
    nmi_matrices = []  # List of 4x4 matrices (one per molecule)
    total_lines = 0
    skipped_lines = 0
    
    method_names = {
        0: key1,
        1: key2,
        2: 'RECAP',  # Using RECAP fragmentation with timeout
        3: 'Random'
    }
    
    # First pass: count total lines for progress bar
    with open(file_path, "r") as f:
        total_lines_count = sum(1 for _ in f)
    
    pbar = tqdm(total=total_lines_count, desc="Processing molecules", unit="mol")
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            try:
                data = json.loads(line.strip())
                
                # Check if both keys exist
                if key1 not in data:
                    skipped_lines += 1
                    pbar.update(1)
                    pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                    continue
                
                if key2 not in data:
                    skipped_lines += 1
                    pbar.update(1)
                    pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                    continue
                
                if 'smiles' not in data:
                    skipped_lines += 1
                    pbar.update(1)
                    pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                    continue
                
                cluster_ids1 = data[key1]
                cluster_ids2 = data[key2]
                smiles = data['smiles']
                
                # Validate that both are lists
                if not isinstance(cluster_ids1, list):
                    skipped_lines += 1
                    pbar.update(1)
                    pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                    continue
                
                if not isinstance(cluster_ids2, list):
                    skipped_lines += 1
                    pbar.update(1)
                    pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                    continue
                
                # Check if lengths match
                if len(cluster_ids1) != len(cluster_ids2):
                    skipped_lines += 1
                    pbar.update(1)
                    pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                    continue
                
                # Skip if too few atoms (need at least 2 atoms for meaningful clustering comparison)
                if len(cluster_ids1) < 2:
                    skipped_lines += 1
                    pbar.update(1)
                    pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                    continue
                
                # Generate RECAP cluster IDs (with timeout protection)
                cluster_ids_recap = get_recap_cluster_ids(smiles, timeout_seconds=3)
                if cluster_ids_recap is None or len(cluster_ids_recap) != len(cluster_ids1):
                    skipped_lines += 1
                    pbar.update(1)
                    pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                    continue
                
                # Generate random baseline (shuffle key2)
                cluster_ids_random = shuffle_cluster_ids(cluster_ids2, seed=line_num)
                
                # Store all 4 clustering methods
                all_clusters = [cluster_ids1, cluster_ids2, cluster_ids_recap, cluster_ids_random]
                
                # Calculate pairwise ARI and NMI for all methods (4x4 matrix)
                ari_matrix = np.zeros((4, 4))
                nmi_matrix = np.zeros((4, 4))
                
                for i in range(4):
                    for j in range(4):
                        if i == j:
                            # Perfect match with itself
                            ari_matrix[i, j] = 1.0
                            nmi_matrix[i, j] = 1.0
                        else:
                            ari_matrix[i, j] = adjusted_rand_score(all_clusters[i], all_clusters[j])
                            nmi_matrix[i, j] = normalized_mutual_info_score(all_clusters[i], all_clusters[j])
                
                ari_matrices.append(ari_matrix)
                nmi_matrices.append(nmi_matrix)
                
                pbar.update(1)
                pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                
            except json.JSONDecodeError:
                skipped_lines += 1
                pbar.update(1)
                pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                continue
            except Exception as e:
                skipped_lines += 1
                pbar.update(1)
                pbar.set_postfix({'valid': len(ari_matrices), 'skipped': skipped_lines})
                continue
    
    pbar.close()
    
    return ari_matrices, nmi_matrices, total_lines, skipped_lines, method_names


def calculate_metrics(file_path: str, key1: str, key2: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, int]:
    """
    Calculate ARI and NMI between four clustering methods from a single file.
    Methods: key1, key2, RECAP (from SMILES with timeout), Random (shuffled key2)
    
    Calculates metrics for each molecule individually and aggregates into matrices.
    
    Args:
        file_path: Path to JSONL file
        key1: Key name for first clustering method
        key2: Key name for second clustering method
    
    Returns:
        Tuple of (ari_mean_matrix, ari_median_matrix, nmi_mean_matrix, nmi_median_matrix, method_names, count)
    """
    # Load data and calculate per-molecule metrics
    print(f"Loading data from {file_path}...")
    print(f"Calculating 4 methods: {key1}, {key2}, RECAP (timeout=3s), Random")
    ari_matrices, nmi_matrices, total_lines, skipped_lines, method_names = load_and_calculate_per_molecule(file_path, key1, key2)
    
    print(f"  Total records: {total_lines}")
    print(f"  Valid records: {len(ari_matrices)}")
    print(f"  Skipped records: {skipped_lines}")
    
    if len(ari_matrices) == 0:
        raise ValueError("No valid data found! All records were skipped.")
    
    # Stack all matrices and calculate mean and median
    ari_stack = np.stack(ari_matrices, axis=0)  # Shape: (n_molecules, 4, 4)
    nmi_stack = np.stack(nmi_matrices, axis=0)  # Shape: (n_molecules, 4, 4)
    
    ari_mean = np.mean(ari_stack, axis=0)  # Shape: (4, 4)
    ari_median = np.median(ari_stack, axis=0)  # Shape: (4, 4)
    nmi_mean = np.mean(nmi_stack, axis=0)  # Shape: (4, 4)
    nmi_median = np.median(nmi_stack, axis=0)  # Shape: (4, 4)
    
    return ari_mean, ari_median, nmi_mean, nmi_median, method_names, len(ari_matrices)


def print_matrix(matrix: np.ndarray, method_names: Dict, title: str) -> None:
    """
    Pretty print a comparison matrix.
    
    Args:
        matrix: 4x4 matrix of comparison values
        method_names: Dictionary mapping indices to method names
        title: Title for the matrix
    """
    print(f"\n{title}:")
    print("-" * 80)
    
    # Print header
    header = "       "
    for i in range(4):
        name = method_names[i]
        header += f"{name:>15s} "
    print(header)
    print("-" * 80)
    
    # Print rows
    for i in range(4):
        row_name = method_names[i]
        row_str = f"{row_name:>6s} "
        for j in range(4):
            row_str += f"{matrix[i, j]:>15.6f} "
        print(row_str)
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate ARI and NMI between four clustering methods (key1, key2, RECAP, Random)"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to JSONL file containing clustering methods and SMILES"
    )
    parser.add_argument(
        "--key1",
        required=True,
        help="Key name for first clustering method (e.g., 'entropy_gids')"
    )
    parser.add_argument(
        "--key2",
        required=True,
        help="Key name for second clustering method (e.g., 'brics_gids')"
    )
    
    args = parser.parse_args()
    
    try:
        ari_mean, ari_median, nmi_mean, nmi_median, method_names, count = calculate_metrics(
            args.file, args.key1, args.key2
        )
        
        print("\n" + "="*80)
        print("CLUSTERING COMPARISON RESULTS")
        print("="*80)
        print(f"File: {args.file}")
        print(f"Methods compared: {args.key1}, {args.key2}, RECAP (from SMILES, timeout=3s), Random (baseline)")
        print(f"Valid molecules analyzed: {count}")
        print("="*80)
        
        # Print ARI matrices
        print_matrix(ari_mean, method_names, "Adjusted Rand Index (ARI) - MEAN across molecules")
        print_matrix(ari_median, method_names, "Adjusted Rand Index (ARI) - MEDIAN across molecules")
        
        # Print NMI matrices
        print_matrix(nmi_mean, method_names, "Normalized Mutual Information (NMI) - MEAN across molecules")
        print_matrix(nmi_median, method_names, "Normalized Mutual Information (NMI) - MEDIAN across molecules")
        
        print("\n" + "="*80)
        print("INTERPRETATION:")
        print("="*80)
        print("ARI range: [-1, 1]")
        print("  - 1.0  = Identical clustering")
        print("  - 0.0  = Random clustering")
        print("  - < 0  = Worse than random")
        print()
        print("NMI range: [0, 1]")
        print("  - 1.0  = Perfect agreement")
        print("  - 0.0  = Independent clusterings")
        print()
        print("Each cell (i, j) shows the similarity between method i and method j.")
        print("Diagonal is always 1.0 (perfect self-similarity).")
        print("Matrix is symmetric.")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

