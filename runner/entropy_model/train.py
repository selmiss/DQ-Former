# pip install transformers datasets accelerate
import json, random, torch
from pathlib import Path
from transformers import Trainer, TrainingArguments
from transformers.utils import logging

from runner.entropy_model.tokenizer import build_vocab, tokenize_atoms, save_vocab
from runner.entropy_model.dataset import SmilesAtomDataset, PadCollator
from runner.entropy_model.model import build_tiny_model
import wandb
import os

logger = logging.get_logger(__name__)

JSON_PATH = f"{os.getenv('DATA_DIR')}/Mol-LLaMA-Instruct/pubchem-molecules.json"

# 1. Build vocab
logger.info("Start building vocab...")
all_smiles = [x["smiles"] for x in json.loads(open(JSON_PATH).read())]
vocab = build_vocab(all_smiles)
save_vocab(vocab, "vocab.txt")

# 2. Datasets
train_set = SmilesAtomDataset(JSON_PATH, vocab, max_len=128)
eval_set  = SmilesAtomDataset(JSON_PATH, vocab, max_len=128)
collator  = PadCollator(pad_id=vocab["<pad>"])

# 3. Model
model = build_tiny_model(len(vocab), vocab["<pad>"], vocab["<bos>"], vocab["<eos>"])

wandb.init(project="entropy_model")

# 4. Training
args = TrainingArguments(
    output_dir="checkpoints/entropy_model",
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="epoch",
    report_to="wandb",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=collator,
)

trainer.train()

# 5. Quick test generation
id2tok = {v:k for k,v in vocab.items()}
def encode_atoms(smiles):
    return torch.tensor([[vocab["<bos>"]] + [vocab.get(a, vocab["<eos>"]) for a in tokenize_atoms(smiles)]])
def decode_atoms(ids): return [id2tok[i] for i in ids if i not in (vocab["<bos>"],vocab["<eos>"],vocab["<pad>"])]

model.eval()
model.to('cuda')
prompt = "c1ccccc1O"   # phenol
inp = encode_atoms(prompt).to(model.device)
gen = model.generate(inp, max_new_tokens=8, do_sample=True, top_p=0.9, temperature=0.8, eos_token_id=vocab["<eos>"])
logger.info("Prompt:", tokenize_atoms(prompt))
logger.info("Pred:", decode_atoms(gen[0].tolist())[len(tokenize_atoms(prompt)):])
