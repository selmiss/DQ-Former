import json, random, torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict
from tokenizer import tokenize_atoms

class SmilesAtomDataset(Dataset):
    def __init__(self, json_path: str, vocab: Dict[str,int], max_len: int = 128):
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]; self.bos_id = vocab["<bos>"]; self.eos_id = vocab["<eos>"]
        data = json.loads(open(json_path).read())
        self.samples = []
        for item in data:
            atoms = tokenize_atoms(item["smiles"])
            ids = [self.bos_id] + [vocab.get(a, self.eos_id) for a in atoms] + [self.eos_id]
            self.samples.append(ids[:max_len])
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

@dataclass
class PadCollator:
    pad_id: int
    def __call__(self, features):
        lens = [len(f["input_ids"]) for f in features]
        maxlen = max(lens)
        def pad(seq): return seq + [self.pad_id]*(maxlen - len(seq))
        input_ids = torch.tensor([pad(f["input_ids"].tolist()) for f in features])
        labels    = torch.tensor([pad(f["labels"].tolist()) for f in features])
        labels[labels == self.pad_id] = -100
        attn = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
