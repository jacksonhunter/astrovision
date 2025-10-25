"""Postprocessing module for RGB composition and export.

This module provides Phase 3 functionality:
- ChannelMapper: RGB channel assignment by wavelength
- Compositor: RGB composite generation (Lupton, simple)
- ImageExporter: Save images in various formats
- HistoryTracker: Record processing history
"""

from .channel_mapper import ChannelMapper, ChannelMapping
from .compositor import Compositor
from .exporter import ImageExporter
from .history_tracker import HistoryTracker, ProcessingStep

__all__ = [
    'ChannelMapper',
    'ChannelMapping',
    'Compositor',
    'ImageExporter',
    'HistoryTracker',
    'ProcessingStep',
]
