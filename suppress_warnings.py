"""Suppress common harmless warnings in astronomy processing."""

import warnings
import os

def suppress_common_warnings():
    """Suppress warnings that don't affect functionality."""

    # Suppress specific warning categories
    warnings.filterwarnings('ignore', category=UserWarning, module='triton')
    warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
    warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='skimage')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')

    # Suppress astropy FITS warnings (deprecated keywords, NaN handling)
    warnings.filterwarnings('ignore', message='.*FITSFixedWarning.*')
    warnings.filterwarnings('ignore', message='.*Input data contains invalid values.*')
    warnings.filterwarnings('ignore', message='.*PCi_ja.*deprecated.*')

    # Suppress HuggingFace symlink warnings (Windows limitation)
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    # Suppress torch distributed warnings
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:torch.distributed'

    print("✓ Warnings suppressed for cleaner output\n")