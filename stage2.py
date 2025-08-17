import os
import argparse
import warnings
from collections import defaultdict
import yaml
from easydict import EasyDict as edict
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import WandbLogger

from transformers import AutoTokenizer

from utils.configuration_mol_llama import MolLLaMAConfig
from data_provider.stage2_dm import Stage2DM
from trainer.stage2 import Stage2Trainer

from utils.dist_funs import MyDeepSpeedStrategy
from peft import get_peft_model, LoraConfig, TaskType

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium')

def parse_tasks(tasks):
    tasks = tasks.split(',')
    out = defaultdict(list)
    for task in tasks:
        split = task.split('_')
        if len(split) == 1:
            out[task] = []
        elif len(split) == 2:
            out[task.split('_')[0]].append(task.split('_')[1])

    return out

def edict_to_dict(config):
    """
    Convert an EasyDict object to a regular dictionary.
    """
    if isinstance(config, edict):
        return {k: edict_to_dict(v) for k, v in config.items()}
    else:
        return config

# Added test_mode parameter
def main(model_config, train_config, data_config, test_mode=False):
    pl.seed_everything(0)

    tokenizer = AutoTokenizer.from_pretrained(
        'DongkiKim/Mol-Llama-3.1-8B-Instruct', 
        # use_fast=False, 
        padding_side='left'
    )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ["<mol>"]})
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    mol_id = tokenizer.convert_tokens_to_ids("<mol>")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

    model = Stage2Trainer(
        vocab_size = len(tokenizer), 
        model_config = model_config, 
        train_config = train_config,
        add_ids = [mol_id, pad_id]
    )

    model.load_from_stage1_ckpt(train_config.stage1_path)

    # -------------------------- LoRA 训练 ----------------------------------
    # q_lora_cfg = LoraConfig(
    #     task_type=TaskType.FEATURE_EXTRACTION,   # 通用特征提取任务
    #     inference_mode=False,                    # 训练态
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     target_modules=["query", "key", "value", "dense"],  # BERT风格的层名
    #     bias="none"
    # )

    # # 2. 套 LoRA（只包 Q-Former）
    # model.mol_llama.encoder.Qformer = get_peft_model(model.mol_llama.encoder.Qformer, q_lora_cfg)

    # # 3. 确认可训练参数
    # model.mol_llama.encoder.Qformer.print_trainable_parameters()
    # --------------------------------------------------------------
    
    args = {'train': edict_to_dict(train_config), 
            'model': edict_to_dict(model_config), 
            'data': edict_to_dict(data_config)}
    model.save_hyperparameters(args)

    if 'Llama-2' in model_config.llm_config.llm_model:
        llama_version = 'llama2'
    elif 'Llama-3' in model_config.llm_config.llm_model:
        llama_version = 'llama3'
    
    dm = Stage2DM(
        tokenizer=tokenizer,
        llama_version=llama_version,
        num_workers=data_config.num_workers,
        batch_size=data_config.batch_size,
        root=data_config.root,
        unimol_dictionary=model.mol_llama.encoder.unimol_dictionary, 
        encoder_types=model_config.graph_encoder_config.encoder_types, 
        data_types=data_config.data_types,
        test_mode=test_mode,
    )
    
    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="checkpoints/"+train_config.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=train_config.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    if len(train_config.devices) > 1:
        if train_config.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
    
    logger = WandbLogger(project="Mol-LLaMA", name=train_config.filename)

    trainer = Trainer(
        accelerator=train_config.accelerator,
        devices=train_config.devices,
        precision=train_config.precision,
        max_epochs=train_config.max_epochs,
        val_check_interval=train_config.val_check_interval,
        limit_val_batches=10,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)

# ----------------- Entry point -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2 training')
    parser.add_argument('--train_config_path', type=str, default='configs/stage2/train_config.yaml')
    parser.add_argument('--data_config_path', type=str, default='configs/stage2/data_config.yaml')
    parser.add_argument('--test_mode', default=False, action='store_true', help='Use small dataset for testing')

    args = parser.parse_args()

    model_config = MolLLaMAConfig({'use_flash_attention': True, 'use_dq_encoder': True})  # Enable Flash Attention for Qformer
    data_config = edict(yaml.load(open(args.data_config_path), Loader=yaml.FullLoader))
    train_config = edict(yaml.load(open(args.train_config_path), Loader=yaml.FullLoader))

    print('-'*60)
    print(f'batch_size: {data_config.batch_size}\tnum_devices: {len(train_config.devices)}\taccumulate_grad_batches: {train_config.accumulate_grad_batches}')
    print(f'Total batch size: {data_config.batch_size * len(train_config.devices) * train_config.accumulate_grad_batches}')
    print('-'*60)
    print(f'Data Types:')
    for data_type in data_config.data_types:
        print(f'  - {data_type}')
    print('-'*60)

    if args.test_mode:
        print('TEST MODE: Using small dataset for quick testing')
    main(model_config, train_config, data_config, test_mode=args.test_mode)

