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
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from transformers import AutoTokenizer

from utils.configuration_mol_llama import MolLLaMAConfig
from data_provider.moleculeqa_dataset import MoleculeQADM
from trainer.moleculeqa_trainer import MoleculeGENQATrainer, MoleculeQATrainer, MoleculePropertyQATrainer

from utils.dist_funs import MyDeepSpeedStrategy

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

def main(model_config, train_config, data_config):
    pl.seed_everything(0)
    
    tokenizer = AutoTokenizer.from_pretrained(
        'DongkiKim/Mol-Llama-3.1-8B-Instruct',
        # model_config.llm_config.llm_model, 
        use_fast=False, 
        padding_side='left'
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ["<mol>"]})
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    if train_config.precision == 'bf16-mixed':
        torch_dtype = torch.bfloat16
    else:
        raise NotImplementedError(f"Precision {train_config.precision} not supported.")

    if data_config.task == 'property':
        model = MoleculePropertyQATrainer(
            vocab_size = len(tokenizer), 
            model_config = model_config, 
            train_config = train_config,
            use_dq_encoder = train_config.use_dq_encoder,
            torch_dtype = torch_dtype
        )
    elif data_config.task == 'caption':   
        model = MoleculeGENQATrainer(
            vocab_size = len(tokenizer), 
            model_config = model_config, 
            train_config = train_config,
            use_dq_encoder = train_config.use_dq_encoder,
            torch_dtype = torch_dtype
        )
    elif data_config.task == 'qa':
        model = MoleculeQATrainer(
            vocab_size = len(tokenizer), 
            model_config = model_config,    
            train_config = train_config,
            use_dq_encoder = train_config.use_dq_encoder,
            torch_dtype = torch_dtype
        )
    else:
        raise ValueError(f"Task {data_config.task} not supported.")
    
    if not train_config.use_dq_encoder:
        model.mol_llama = model.mol_llama.from_pretrained(train_config.ckpt_path, torch_dtype=torch_dtype)
    else:
        # model.mol_llama = model.mol_llama.from_pretrained(train_config.ckpt_path, torch_dtype=torch_dtype)
        model.mol_llama.load_from_ckpt(train_config.ckpt_path)
    encoder = model.mol_llama.encoder
    model.mol_llama.llm.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer

    args = {'train': edict_to_dict(train_config), 
            'model': edict_to_dict(model_config), 
            'data': edict_to_dict(data_config)}
    model.save_hyperparameters(args)

    if 'Llama-2' in model_config.llm_config.llm_model:
        llama_version = 'llama2'
    elif 'Llama-3' in model_config.llm_config.llm_model:
        llama_version = 'llama3'
    
    dm = MoleculeQADM(
        tokenizer=tokenizer,
        llama_version=llama_version,
        num_workers=data_config.num_workers,
        batch_size=data_config.batch_size,
        root=data_config.root,
        unimol_dictionary=encoder.unimol_dictionary, 
        encoder_types=model_config.graph_encoder_config.encoder_types, 
        mol_type=data_config.mol_type if 'mol_type' in data_config else 'mol',
    )
    
    
    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="checkpoints/"+train_config.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=train_config.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    detected_num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if detected_num_devices > 1:
        if train_config.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
    
    # logger = WandbLogger(project="MoleculeQA", name=train_config.filename)
    csv_logger = CSVLogger(save_dir=f'./checkpoints/{train_config.filename}/')
    # logger = [logger, csv_logger]
    logger = csv_logger

    accelerator_arg = train_config.accelerator if detected_num_devices > 0 else 'cpu'
    devices_arg = detected_num_devices if detected_num_devices > 0 else 1

    trainer = Trainer(
        accelerator=accelerator_arg,
        devices=devices_arg,
        precision=train_config.precision,
        max_epochs=train_config.max_epochs,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        check_val_every_n_epoch=train_config.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=getattr(train_config, "log_every_n_steps", 2),
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoleculeQA Training or Test')
    parser.add_argument('--train_config_path', type=str, default='configs/moleculeqa/train_config.yaml')
    parser.add_argument('--data_config_path', type=str, default='configs/moleculeqa/data_config.yaml')

    args = parser.parse_args()

    model_config = MolLLaMAConfig()
    data_config = edict(yaml.load(open(args.data_config_path), Loader=yaml.FullLoader))
    train_config = edict(yaml.load(open(args.train_config_path), Loader=yaml.FullLoader))

    print('-'*60)
    detected_num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f'batch_size: {data_config.batch_size}\tnum_devices: {detected_num_devices}\taccumulate_grad_batches: {train_config.accumulate_grad_batches}')
    print(f'Total batch size: {data_config.batch_size * max(1, detected_num_devices) * train_config.accumulate_grad_batches}')
    print('-'*60)

    main( model_config, train_config, data_config )

