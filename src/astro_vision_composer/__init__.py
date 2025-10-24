"""Astro Vision Composer - AI-guided astronomical image composition from FITS data.

This library combines astronomical data processing with vision model inference
to create visually stunning planetarium-style composite images.
"""

from .fits_processor import FITSImageProcessor
from .composite_generator import CompositeImageGenerator
from .vision_compositor import VisionGuidedCompositor

__version__ = "0.1.0"
__all__ = [
    "FITSImageProcessor",
    "CompositeImageGenerator",
    "VisionGuidedCompositor"
]