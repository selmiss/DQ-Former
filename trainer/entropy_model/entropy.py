# entropy.py
# pip install transformers accelerate
import torch, math
from pathlib import Path
from transformers import GPT2LMHeadModel
from trainer.entropy_model.tokenizer import tokenize_atoms
from trainer.entropy_model.model import build_tiny_model



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


if __name__ == "__main__":
    # smiles = ["c1ccccc1O", "CC(=O)NCCC1=CNc2c1cccc2"]  # batch input
    smiles = [
    "CCO",                    # ethanol
    "CC(=O)O",                # acetic acid
    "CC(C)N",                 # isopropylamine
    "CCN(CC)CC",              # triethylamine
    "c1ccccc1",               # benzene
    "c1ccccc1O",              # phenol
    "c1ccc(Cl)cc1",           # chlorobenzene
    "c1ccncc1",               # pyridine
    "CCc1ccccc1",             # ethylbenzene
    "CCOC(=O)C",              # ethyl acetate
    "C1CCCCC1",               # cyclohexane
    "c1cccc2ccccc12",         # naphthalene
    "c1cc2ccccn2c1",          # quinoline
    "CCSCC",                  # thioether
    "CCCl",                   # chloroethane
    "CCBr",                   # bromoethane
    "CCI",                    # iodoethane
    "CCF",                    # fluoroethane
    "B(O)(O)O",               # boric acid
    "O=P(O)(O)OCC",           # ethyl phosphate (P)
    "C[Si](C)(C)C",           # (contains Si)
    "CC[Se]CC",               # (contains Se)
    ]

    ent, one_minus_pmax = score_batch(
        smiles_batch=smiles,
        ckpt_dir="checkpoints/entropy_model/checkpoint-4929",
        vocab_path="trainer/entropy_model/vocab.txt"
    )
    # Each row i corresponds to SMILES[i], each column t is entropy for predicting token at t+1
    print("Entropy(bits) per position:\n", ent)
    print("1 - p_max per position:\n", one_minus_pmax)
