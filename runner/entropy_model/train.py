# pip install transformers datasets accelerate
import json, random, torch
from pathlib import Path
from transformers import Trainer, TrainingArguments
from transformers.utils import logging

from runner.entropy_model.tokenizer import build_vocab, tokenize_atoms, save_vocab
from runner.entropy_model.dataset import SmilesAtomDataset, PadCollator
from runner.entropy_model.model import build_tiny_model, build_middle_model
import wandb
import os

# Configure logging to show INFO messages
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

JSON_PATH = f"{os.getenv('DATA_DIR')}/Mol-LLaMA-Instruct/pubchem-molecules.json"

# 1. Load JSON once
print(f"[INFO] Loading JSON from {JSON_PATH}...")
logger.info(f"Loading JSON from {JSON_PATH}...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)
print(f"[INFO] Loaded {len(data)} molecules")
logger.info(f"Loaded {len(data)} molecules")

# 2. Build vocab
print("[INFO] Building vocab...")
logger.info("Building vocab...")
all_smiles = [x["smiles"] for x in data]
vocab = build_vocab(all_smiles)
save_vocab(vocab, "vocab.txt")
print(f"[INFO] Vocab size: {len(vocab)}")
logger.info(f"Vocab size: {len(vocab)}")

# 3. Datasets (pass data directly instead of file path)
print("[INFO] Creating datasets...")
logger.info("Creating datasets...")
train_set = SmilesAtomDataset(data, vocab, max_len=128)
eval_set  = SmilesAtomDataset(data, vocab, max_len=128)
collator  = PadCollator(pad_id=vocab["<pad>"])
print(f"[INFO] Train size: {len(train_set)}, Eval size: {len(eval_set)}")
logger.info(f"Train size: {len(train_set)}, Eval size: {len(eval_set)}")

# 4. Model
print("[INFO] Building model...")
logger.info("Building model...")
# model = build_tiny_model(len(vocab), vocab["<pad>"], vocab["<bos>"], vocab["<eos>"])
model = build_middle_model(len(vocab), vocab["<pad>"], vocab["<bos>"], vocab["<eos>"])

# Log model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] Total parameters: {total_params:,}")
print(f"[INFO] Trainable parameters: {trainable_params:,}")
print(f"[INFO] Non-trainable parameters: {total_params - trainable_params:,}")
logger.info(f"Total parameters: {total_params:,}")
logger.info(f"Trainable parameters: {trainable_params:,}")
logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

wandb.init(project="entropy_model")

# 5. Training
print("[INFO] Starting training...")
logger.info("Starting training...")
args = TrainingArguments(
    output_dir="checkpoints/entropy_model_large",
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",  # Changed from evaluation_strategy (deprecated)
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

# 6. Quick test generation
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
