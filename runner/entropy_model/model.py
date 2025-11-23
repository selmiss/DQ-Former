import torch
from transformers import GPT2Config, GPT2LMHeadModel

def build_tiny_model(vocab_size: int, pad_id: int, bos_id: int, eos_id: int):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_ctx=512,
        n_layer=2, n_head=2, n_embd=128,
        bos_token_id=bos_id, eos_token_id=eos_id
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(vocab_size)
    model.config.pad_token_id = pad_id
    return model


def build_middle_model(vocab_size: int, pad_id: int, bos_id: int, eos_id: int):
    # Configuration for ~500M parameters
    # Similar to GPT2-medium/large hybrid: 32 layers, 16 heads, 1152 embedding dim
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_ctx=512,
        n_layer=32, n_head=16, n_embd=1152,
        bos_token_id=bos_id, eos_token_id=eos_id
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(vocab_size)
    model.config.pad_token_id = pad_id
    return model
