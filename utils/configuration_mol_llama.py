"""
Compatibility shim for old checkpoints.

This module redirects imports from the old location (utils.configuration_mol_llama)
to the new location (models.configuration).

This allows loading checkpoints that were saved with references to the old module path.
"""

# Import everything from the new location
from models.configuration import *

# Explicitly re-export for pickle compatibility
__all__ = [
    'MolLLaMAConfig',
    'QformerConfig', 
    'GraphEncoderConfig',
    'BlendingModuleConfig',
    'LLMConfig',
    'LoraConfig',
]

