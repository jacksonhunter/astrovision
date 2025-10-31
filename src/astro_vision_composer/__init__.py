"""Astro Vision Composer - Professional astronomical image processing toolkit.

A comprehensive FITS processing library with mission-aware data handling,
advanced image processing, and RGB composite generation.

Modern API (Phases 1-4):
- preprocessing: FITSLoader, MissionAdapter, QualityAssessor, Calibrator
- processing: WCSHandler, Reprojector, Normalizer, Stretcher, Enhancer
- postprocessing: ChannelMapper, Compositor, ColorBalancer, ImageExporter, HistoryTracker
- utilities: FITSMetadata
- pipeline: ProcessingPipeline (high-level orchestrator)
"""

# Import main components for convenience
# NOTE: Preprocessing imports disabled - deprecated module has broken paths
# from .preprocessing import FITSLoader, QualityAssessor, Calibrator
# from deprecated.old_api_src.mission_adapters import get_mission_adapter
from .processing import WCSHandler, Reprojector, Normalizer, Stretcher, Enhancer
from .postprocessing import ChannelMapper, Compositor, ColorBalancer, ImageExporter, HistoryTracker
from .utilities import FITSMetadata
from .pipeline import ProcessingPipeline

__version__ = "0.2.0-dev"
__all__ = [
    # Preprocessing
    "FITSLoader",
    "QualityAssessor",
    "Calibrator",
    "get_mission_adapter",
    # Processing
    "WCSHandler",
    "Reprojector",
    "Normalizer",
    "Stretcher",
    "Enhancer",
    # Postprocessing
    "ChannelMapper",
    "Compositor",
    "ColorBalancer",
    "ImageExporter",
    "HistoryTracker",
    # Utilities
    "FITSMetadata",
    # Pipeline
    "ProcessingPipeline",
]