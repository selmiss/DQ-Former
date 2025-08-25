import os
import torch
import warnings
import argparse
import yaml
import glob
from easydict import EasyDict as edict

import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger


from utils.configuration_mol_llama import MolLLaMAConfig
from utils.dist_funs import MyDeepSpeedStrategy
from trainer.stage1 import Stage1Trainer
from data_provider.stage1_dm import Stage1DM
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler

## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

def edict_to_dict(config):
    """
    Convert an EasyDict object to a regular dictionary.
    """
    if isinstance(config, edict):
        return {k: edict_to_dict(v) for k, v in config.items()}
    else:
        return config


def find_latest_checkpoint(checkpoint_dir):
    """
    Automatically find the latest checkpoint in the checkpoint directory.
    Returns the path to the latest checkpoint or None if no checkpoint exists.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint files (both .ckpt and -last.ckpt)
    checkpoint_patterns = [
        os.path.join(checkpoint_dir, "*.ckpt"),
        os.path.join(checkpoint_dir, "*-last.ckpt")
    ]
    
    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        return None
    
    # Sort by modification time to get the latest
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def main(model_config, train_config, data_config, test_mode=False):
    pl.seed_everything(0)

    model = Stage1Trainer(model_config, train_config)
    
    args = {'train': edict_to_dict(train_config), 
            'model': edict_to_dict(model_config), 
            'data': edict_to_dict(data_config)}
    model.save_hyperparameters(args)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.shape}")
    #     else:
    #         print(f"{name}: {param.shape} (not trainable)")
    
    # exit()


    dm = Stage1DM(
        num_workers = data_config.num_workers, 
        batch_size = data_config.batch_size, 
        root = data_config.root,
        unimol_dictionary = model.encoder.unimol_dictionary if 'unimol' in model_config.graph_encoder_config.encoder_types else None, 
        scibert_tokenizer = model.encoder.scibert_tokenizer, 
        encoder_types = model_config.graph_encoder_config.encoder_types, 
        text_max_len = data_config.text_max_len,
        test_mode = test_mode,  # Add test mode parameter
    )
    
    checkpoint_dir = f"checkpoints/{train_config.filename}/"
    callbacks = [
        plc.ModelCheckpoint(dirpath=checkpoint_dir, 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=train_config.save_every_n_epochs, 
                                         save_top_k=-1,
                                         save_last=True,
                                         save_on_train_epoch_end=True)
    ]
    
    if len(train_config.devices) > 1:
        if train_config.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=False)
    else:
        strategy = 'auto'
        
    logger = CSVLogger(save_dir=f'./checkpoints/{train_config.filename}/')
    wandb_logger = WandbLogger(
        project="stage1_v2",              # 项目名
        name=train_config.filename,    # 实验名，可用 checkpoint 名作为 run name
        log_model=False,               # 是否自动上传模型
    )
    logger = [logger, wandb_logger]
    # profiler = AdvancedProfiler(dirpath="prof_log", filename="perf.txt")
    trainer = Trainer(
        accelerator=train_config.accelerator,
        devices=train_config.devices,
        precision=train_config.precision,
        max_epochs=train_config.max_epochs,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        check_val_every_n_epoch=train_config.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        limit_val_batches=10,
    )
    
    # Auto-detect and resume from latest checkpoint
    # latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    latest_checkpoint = None
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.fit(model, datamodule=dm, ckpt_path=latest_checkpoint)
    else:
        print("No checkpoint found, starting training from scratch")
        trainer.fit(model, datamodule=dm)
    # print("\n=== Profiler summary (first 5 lines) ===")
    # print(profiler.summary()[:5])   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1 training')
    parser.add_argument('--train_config_path', type=str, default='configs/stage1/train_config.yaml')
    parser.add_argument('--data_config_path', type=str, default='configs/stage1/data_config.yaml')
    parser.add_argument('--test_mode', default=False, action='store_true', help='Use small dataset for testing')

    args = parser.parse_args()

    # Create model config with Flash Attention support
    data_config = edict(yaml.load(open(args.data_config_path), Loader=yaml.FullLoader))
    train_config = edict(yaml.load(open(args.train_config_path), Loader=yaml.FullLoader))
    model_config = MolLLaMAConfig(
        qformer_config={'use_flash_attention': train_config.use_flash_attention, 'use_dq_encoder': train_config.use_dq_encoder},  # Enable Flash Attention for Qformer
        graph_encoder_config={'local_q_only': train_config.local_q_only}
    )

    print('-'*60)
    print(f'batch_size: {data_config.batch_size}\tnum_devices: {len(train_config.devices)}\taccumulate_grad_batches: {train_config.accumulate_grad_batches}')
    print(f'Total batch size: {data_config.batch_size * len(train_config.devices) * train_config.accumulate_grad_batches}')
    if args.test_mode:
        print('TEST MODE: Using small dataset for quick testing')
    print('-'*60)
    
    main(model_config, train_config, data_config, test_mode=args.test_mode)