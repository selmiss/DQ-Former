import re
from pathlib import Path
from typing import List, Dict

# Longest-first regex for heavy atoms (expand as needed)
ATOM_REGEX = re.compile(
    r"Cl|Br|Si|Se|Na|Ca|Li|Mg|Al|Zn|Cu|Mn|Fe|Co|Ni|Ag|Au|Pt|Pd|Hg|Sn|Sb|Te|Xe|Kr|He|Ne|Ar|"
    r"[cnospb]|[BCNOFPSI]"
)
SPECIALS = ["<pad>", "<bos>", "<eos>"]

def tokenize_atoms(smiles: str) -> List[str]:
    toks = ATOM_REGEX.findall(smiles)
    out = []
    for t in toks:
        if len(t) == 1:
            out.append(t.upper())              # aromatic → uppercase
        else:
            out.append(t[0].upper() + t[1:].lower())
    return out

def build_vocab(smiles_list: List[str]) -> Dict[str,int]:
    atoms = set()
    for s in smiles_list:
        atoms.update(tokenize_atoms(s))
    vocab = SPECIALS + sorted(atoms)
    return {tok: i for i, tok in enumerate(vocab)}

def save_vocab(vocab: Dict[str,int], path: str):
    Path(path).write_text("\n".join(sorted(vocab, key=lambda k: vocab[k])))
