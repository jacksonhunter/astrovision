"""Postprocessing module for RGB composition and export.

This module provides Phase 3 & 4 functionality:
- ChannelMapper: RGB channel assignment by wavelength
- Compositor: RGB composite generation (Lupton, simple)
- ImageExporter: Save images in various formats
- HistoryTracker: Record processing history
- ColorBalancer: Color balance adjustment
- PreviewGenerator: Thumbnails and previews
"""

from .channel_mapper import ChannelMapper, ChannelMapping
from .compositor import Compositor
from .exporter import ImageExporter
from .history_tracker import HistoryTracker, ProcessingStep
from .color_balancer import ColorBalancer
from .preview import PreviewGenerator

__all__ = [
    'ChannelMapper',
    'ChannelMapping',
    'Compositor',
    'ImageExporter',
    'HistoryTracker',
    'ProcessingStep',
    'ColorBalancer',
    'PreviewGenerator',
]
