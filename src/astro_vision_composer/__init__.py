"""Astro Vision Composer - AI-guided astronomical image composition from FITS data.

This library combines astronomical data processing with vision model inference
to create visually stunning planetarium-style composite images.
"""

from .fits_processor import FITSImageProcessor
from .composite_generator import CompositeImageGenerator

# Try to import VisionGuidedCompositor (requires vision server)
try:
    from .vision_compositor import VisionGuidedCompositor
    _has_vision = True
except ImportError:
    VisionGuidedCompositor = None
    _has_vision = False

__version__ = "0.1.0"
__all__ = [
    "FITSImageProcessor",
    "CompositeImageGenerator",
]

if _has_vision:
    __all__.append("VisionGuidedCompositor")