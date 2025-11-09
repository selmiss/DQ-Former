# entropy.py
# pip install transformers accelerate
import torch, math
from pathlib import Path
from transformers import GPT2LMHeadModel
from transformers.utils import logging
from runner.entropy_model.tokenizer import tokenize_atoms
from runner.entropy_model.model import build_tiny_model
import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)



def load_vocab(fp="vocab.txt"):
    toks = Path(fp).read_text().splitlines()
    return {t:i for i,t in enumerate(toks)}

def batch_encode(smiles_batch, vocab, add_eos=True, max_len=512):
    BOS, EOS, PAD = vocab["<bos>"], vocab["<eos>"], vocab["<pad>"]
    batch = []
    for s in smiles_batch:
        ids = [BOS] + [vocab.get(a, EOS) for a in tokenize_atoms(s)]
        if add_eos: ids.append(EOS)
        batch.append(ids[:max_len])
    L = max(len(x) for x in batch)
    pad = lambda x: x + [PAD]*(L-len(x))
    x = torch.tensor([pad(x) for x in batch], dtype=torch.long)      # [B, T]
    mask = (x != PAD).long()
    return x, mask

def next_atom_entropy_bits(logits):  # logits: [B, T-1, V]
    probs = logits.softmax(-1)
    ent_nats = -(probs * (probs.clamp_min(1e-12).log())).sum(-1)     # [B, T-1]
    return ent_nats / math.log(2)

def score_batch(smiles_batch, ckpt_dir, vocab_path="vocab.txt", device=None):
    vocab = load_vocab(vocab_path); PAD, BOS, EOS = vocab["<pad>"], vocab["<bos>"], vocab["<eos>"]
    model = build_tiny_model(len(vocab), PAD, BOS, EOS)
    model.load_state_dict(GPT2LMHeadModel.from_pretrained(ckpt_dir).state_dict(), strict=False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    max_len = int(getattr(model.config, "n_positions", 512))
    x, attn = batch_encode(smiles_batch, vocab, max_len=max_len)          # x: [B,T], attn: [B,T]
    x, attn = x.to(device), attn.to(device)
    with torch.no_grad():
        logits = model(input_ids=x, attention_mask=attn).logits  # [B,T,V]

    # scores for predicting the next token
    ent_bits = next_atom_entropy_bits(logits[:, :-1, :])  # [B,T-1]
    pmax     = logits[:, :-1, :].softmax(-1).amax(-1)     # [B,T-1]

    # trim outputs using mask (exclude last token since no "next" target)
    ent = []
    one_minus_pmax = []
    for i in range(x.size(0)):
        valid_len = attn[i].sum().item() - 1   # number of valid next-token predictions
        ent.append(ent_bits[i, :valid_len].cpu())
        one_minus_pmax.append((1 - pmax[i, :valid_len]).cpu())
    return ent, one_minus_pmax


def plot_entropy_from_atoms(atom_list, entropy, title=None, savepath=None):
    """
    Plot entropy values given atom symbols and entropy list.
    
    Parameters
    ----------
    atom_list : list[str]
        Atom symbols (e.g., ['N','C','C','O'])
    entropy : list[float]
        Entropy values per atom (must match length of atom_list)
    title : str, optional
        Title for the figure
    savepath : str, optional
        Path to save figure (png)
    """
    if len(atom_list) != len(entropy):
        raise ValueError("Atom list and entropy list must have same length.")
    
    x = list(range(len(atom_list)))
    fig, ax = plt.subplots(figsize=(12.5,3.2))
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    ax.plot(x, entropy, marker='^', color='pink', markeredgecolor='gray', markeredgewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(atom_list, rotation=0)
    ax.set_xlabel("Atoms")
    ax.set_ylabel("Entropy")
    if title:
        ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', axis='y')
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=500, bbox_inches="tight")
    return fig, ax



if __name__ == "__main__":
    # smiles = ["c1ccccc1O", "CC(=O)NCCC1=CNc2c1cccc2"]  # batch input
    smiles = [
    # "N(CC(=O)O)(CC(=O)O)CC(=O)O"
    # "O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CCN(CC(=O)O)CC(=O)O"
    "O=C(O)CN(CCN(CCN(CCN(CC(=O)O)CC(=O)O)CC(=O)O)CC(=O)O)CCN(CC(=O)O)CC(=O)O"
    ]

    ent, one_minus_pmax = score_batch(
        smiles_batch=smiles,
        ckpt_dir="checkpoints/entropy_model/checkpoint-4929",
        vocab_path="trainer/entropy_model/vocab.txt"
    )
    # Each row i corresponds to SMILES[i], each column t is entropy for predicting token at t+1
    logger.info("Entropy(bits) per position:\n", ent)
    logger.info("1 - p_max per position:\n", one_minus_pmax)
    # Calculate 75th percentile average of entropy values
    ent_75_avg = sum([torch.quantile(e, 0.75) for e in ent[0]]) / len(ent[0])
    logger.info("75th percentile average entropy:", ent_75_avg.item())

    atoms_demo = [
        "O","C","O","C","N","C","C","N","C","C",
        "N","C","C","N","C","C","O","O","C","C",
        "O","O","C","C","O","O","C","C","O","O",
        "C","C","N","C","C","O","O","C","C","O","O"
    ]
    # entropy_demo = ent[0][1:].tolist()
    entropy_demo = [1.0, 1.1, 1.6102, 1.1868, 1.6020, 1.2162, 1.1973, 1.5637,
        1.3162, 1.2973, 1.6723, 1.2562, 1.2473, 1.5746, 1.5440, 1.4642, 1.8367,
        1.9162, 1.3161, 1.4602, 1.8952, 1.9499, 1.3935, 1.3844, 1.8701, 1.9845,
        1.4033, 1.3952, 1.8863, 1.9669, 1.3700, 1.3571, 1.7053, 1.2418, 1.2737,
        1.7862, 1.8258, 1.2510, 1.1631, 1.7900, 1.8765]

    fig, ax = plot_entropy_from_atoms(atoms_demo, entropy_demo, title="", savepath="trainer/entropy_model/fig/entropy_atoms_demo.png")
