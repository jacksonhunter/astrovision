"""Postprocessing module for RGB composition and export.

This module provides Phase 3 & 4 functionality:
- ChannelMapper: RGB channel assignment by wavelength
- PaletteMapper: Narrowband palette mapping (Hubble, HOO, etc.)
- BandSelector: Multi-band selection from >3 filters
- Compositor: RGB composite generation (Lupton, simple)
- ImageExporter: Save images in various formats
- HistoryTracker: Record processing history
- ColorBalancer: Color balance adjustment
- PreviewGenerator: Thumbnails and previews
"""

from .channel_mapper import ChannelMapper, ChannelMapping
from .palette_mapper import PaletteMapper, PaletteMapping, ColorChannel
from .band_selector import BandSelector, BandSelection, SelectionStrategy
from .compositor import Compositor
from .exporter import ImageExporter
from .history_tracker import HistoryTracker, ProcessingStep
from .color_balancer import ColorBalancer
from .preview import PreviewGenerator

__all__ = [
    'ChannelMapper',
    'ChannelMapping',
    'PaletteMapper',
    'PaletteMapping',
    'ColorChannel',
    'BandSelector',
    'BandSelection',
    'SelectionStrategy',
    'Compositor',
    'ImageExporter',
    'HistoryTracker',
    'ProcessingStep',
    'ColorBalancer',
    'PreviewGenerator',
]
