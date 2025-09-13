import os
from typing import Any, Dict
import json

import torch
from torch import optim
import pytorch_lightning as pl

from models.mol_llama import MolLLaMA, DQMolLLaMA
from trainer.optims import LinearWarmupCosineLRScheduler


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class Stage2Trainer(pl.LightningModule):
    def __init__(self, vocab_size, model_config, train_config, add_ids):
        super().__init__()
        self.train_config = train_config
        if train_config.precision == 'bf16-mixed':
            torch_dtype = "bfloat16"
        elif train_config.precision == '16':
            torch_dtype = "float16"
        elif train_config.precision == '32':
            torch_dtype = "float32"

        # Choose model based on configuration
        if model_config.qformer_config.use_dq_encoder:
            if hasattr(train_config, 'enable_blending'):
                print("Using DQMolLLaMA, enable_blending:", train_config.enable_blending)
            else:
                print("Using DQMolLLaMA, enable_blending: False")
                train_config.enable_blending = False
            self.model = DQMolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                add_ids = add_ids,
                freeze_llm = train_config.freeze_llm if train_config.freeze_llm else False,
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = train_config.enable_blending,
            )
        else:
            print("Using MolLLaMA")
            self.model = MolLLaMA(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = train_config.freeze_llm if train_config.freeze_llm else False,
                # add_ids = add_ids,
                enable_blending = train_config.enable_blending,
            )


    def load_from_stage1_ckpt(self, ckpt_path):
        self.model.load_from_stage1_ckpt(ckpt_path)        

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch              
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = text_batch.input_ids.size(0)
        ###============== Overall Loss ===================###
        output = self.model(graph_batch, text_batch, other_infos)
        loss = {'loss': output['loss']}

        self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss['loss']