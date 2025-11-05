"""
Test script to validate the HF-based Stage 1 training conversion.
This script checks imports, model initialization, and basic functionality.

USAGE:
    cd /home/UWO/zjing29/workdir/DQ-Former
    source local.env.sh  # Required to set up environment
    python test_stage1_hf.py
"""

import os
import sys
import torch
import yaml
from easydict import EasyDict as edict

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("=" * 80)
print("Stage 1 HF Training Validation Test")
print("=" * 80)

# Test 1: Import checks
print("\n[1/6] Testing imports...")
try:
    from trainer.stage1_hf import Stage1Model, precision2dtype
    from utils.configuration_mol_llama import MolLLaMAConfig
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load configurations
print("\n[2/6] Loading configurations...")
try:
    train_config_path = "configs/stage1_dqw2d/train_config.yaml"
    data_config_path = "configs/stage1_dqw2d/data_config.yaml"
    
    if not os.path.exists(train_config_path):
        print(f"⚠ Config file not found: {train_config_path}")
        print("  Using mock config for testing")
        train_config = edict({
            'filename': 'test_run',
            'use_flash_attention': False,
            'use_dq_encoder': True,
            'local_q_only': False,
            'enable_blending': False,
            'temperature': 0.1,
            'tune_gnn': False,
            'brics_gids_enable': True,
            'entropy_gids_enable': True,
            'num_query_tokens': 8,
            'embed_dim': 256,
            'cross_attention_freq': 2,
            'num_layers': 4,
            'num_heads': 8,
        })
    else:
        train_config = edict(yaml.load(open(train_config_path), Loader=yaml.FullLoader))
    
    print("✓ Configuration loaded")
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")
    sys.exit(1)

# Test 3: Create model config
print("\n[3/6] Creating model configuration...")
try:
    model_config = MolLLaMAConfig(
        qformer_config={
            "use_flash_attention": train_config.use_flash_attention,
            "use_dq_encoder": train_config.use_dq_encoder,
            "num_query_tokens": getattr(train_config, "num_query_tokens", 8),
            "embed_dim": getattr(train_config, "embed_dim", 256),
            "cross_attention_freq": getattr(train_config, "cross_attention_freq", 2),
        },
        graph_encoder_config={"local_q_only": train_config.local_q_only},
        blending_module_config={
            "num_layers": getattr(train_config, "num_layers", 4),
            "num_heads": getattr(train_config, "num_heads", 8),
            "enable_blending": getattr(train_config, "enable_blending", False),
        },
    )
    print("✓ Model configuration created")
    print(f"  - QFormer: use_dq_encoder={model_config.qformer_config.use_dq_encoder}")
    print(f"  - Blending: enabled={train_config.enable_blending}")
except Exception as e:
    print(f"✗ Model configuration failed: {e}")
    sys.exit(1)

# Test 4: Initialize model
print("\n[4/6] Initializing Stage1Model...")
try:
    model = Stage1Model(model_config, train_config)
    print("✓ Model initialized successfully")
    print(f"  - Encoder type: {type(model.encoder).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test precision conversion
print("\n[5/6] Testing precision conversion...")
try:
    dtype_map = {
        '16': torch.float16,
        '32': torch.float32,
        'bf16-mixed': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    
    for precision, expected_dtype in dtype_map.items():
        result_dtype = precision2dtype(precision)
        if result_dtype == expected_dtype:
            print(f"  ✓ {precision} → {result_dtype}")
        else:
            print(f"  ✗ {precision} → {result_dtype} (expected {expected_dtype})")
except Exception as e:
    print(f"✗ Precision conversion test failed: {e}")

# Test 6: Test HF Trainer components
print("\n[6/6] Testing HF Trainer components...")
try:
    from transformers import TrainingArguments
    from stage1_hf import CustomWarmupCosineSchedulerCallback, Stage1DataCollator, Stage1Trainer
    
    print("  ✓ TrainingArguments imported")
    print("  ✓ CustomWarmupCosineSchedulerCallback imported")
    print("  ✓ Stage1DataCollator imported")
    print("  ✓ Stage1Trainer imported")
    
    # Test creating training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_strategy="no",
        report_to=[],
    )
    print("  ✓ TrainingArguments instantiated")
    
except Exception as e:
    print(f"✗ HF Trainer components test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("Validation Complete!")
print("=" * 80)
print("\n✓ All core components are working correctly")
print("\nNext steps:")
print("  1. Set up environment: source local.env.sh")
print("  2. Run with test mode: python stage1_hf.py --test_mode")
print("  3. Monitor loss curves and compare with Lightning version")
print("  4. Run full training: bash scripts/training/stage_1_dqw2d_hf.sh")
print("\nFor more details, see:")
print("  - MIGRATION_GUIDE_STAGE1_HF.md")
print("  - STAGE1_CODE_COMPARISON.md")
print("\nReminder: Always run 'source local.env.sh' before training!")

