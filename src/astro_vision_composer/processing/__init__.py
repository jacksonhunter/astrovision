"""Processing module for astronomical image alignment and enhancement.

This module provides Phase 2 & 4 functionality:
- WCSHandler: WCS validation and comparison
- Reprojector: Image alignment via reprojection
- Normalizer: Data scaling to normalized range
- Stretcher: Non-linear transformations
- Enhancer: Image enhancement (CLAHE, unsharp masking, etc.)
"""

from .wcs_handler import WCSHandler, WCSInfo
from .reprojector import Reprojector
from .normalizer import Normalizer
from .stretcher import Stretcher
from .enhancer import Enhancer

__all__ = [
    'WCSHandler',
    'WCSInfo',
    'Reprojector',
    'Normalizer',
    'Stretcher',
    'Enhancer',
]
