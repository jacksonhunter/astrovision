"""Processing pipeline for astronomical image workflows.

This module provides the ProcessingPipeline class that orchestrates the complete
workflow from raw FITS files to RGB composites, implementing multiple validated
workflows for different use cases.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Literal, Union
from pathlib import Path
import numpy as np
import logging
import json
import pickle

logger = logging.getLogger(__name__)

# NOTE: Imports are lazy (inside methods) to avoid circular import with preprocessing
# preprocessing.fits_loader imports utilities.metadata
# utilities.pipeline imports preprocessing.FITSLoader
# This would create a circle if done at module level

# Import astropy visualization for advanced workflows (safe - external dependency)
from astropy.visualization import (
    ZScaleInterval, AsymmetricPercentileInterval,
    AsinhStretch, HistEqStretch, LinearStretch,
    LuptonAsinhZscaleStretch
)


WorkflowMode = Literal['scientific', 'sdss', 'aesthetic', 'narrowband', 'custom']


class ProcessingPipeline:
    """Flexible astronomical image processing pipeline.

    Implements multiple validated workflows:
    - **scientific**: Preserve photometry (ZScale + Asinh + Lupton/Simple)
    - **sdss**: Auto-optimized Lupton (ZScale + LuptonAsinhZscaleStretch)
    - **aesthetic**: Maximum impact (Percentile + HistEq + Lupton)
    - **narrowband**: Per-channel optimization (false-color)
    - **custom**: User controls everything

    Example:
        >>> pipeline = ProcessingPipeline(mode='scientific')
        >>> # Process FITS files through complete pipeline
        >>> rgb = pipeline.process_to_rgb(
        ...     fits_files=['r.fits', 'g.fits', 'b.fits'],
        ...     output_dir='output/'
        ... )
    """

    def __init__(self, mode: WorkflowMode = 'scientific'):
        """Initialize the ProcessingPipeline.

        Args:
            mode: Workflow mode (scientific, sdss, aesthetic, narrowband, custom)
        """
        # Lazy imports to avoid circular dependency
        from astro_vision_composer.preprocessing import FITSLoader, QualityAssessor
        from astro_vision_composer.processing import Normalizer, Stretcher
        from astro_vision_composer.postprocessing import (
            ChannelMapper, Compositor, ImageExporter, HistoryTracker
        )

        self.mode = mode
        self.history = HistoryTracker()

        # Initialize components
        self.loader = FITSLoader()
        self.quality_assessor = QualityAssessor()
        self.normalizer = Normalizer()
        self.stretcher = Stretcher()
        self.mapper = ChannelMapper()
        self.compositor = Compositor()
        self.exporter = ImageExporter()

        # Workflow state
        self.phase1_data = {}
        self.phase2_data = {}
        self.phase3_rgb = None

        logger.info(f"Pipeline initialized with mode: {mode}")

    def process_to_rgb(
        self,
        fits_files: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> np.ndarray:
        """Process FITS files through complete pipeline to RGB composite.

        This is the main entry point for end-to-end processing.

        Args:
            fits_files: List of FITS file paths (at least 3 for RGB)
            output_dir: Optional output directory for intermediate files
            **kwargs: Override workflow parameters (interval, stretch, etc.)

        Returns:
            RGB image array (float, [0, 1] range)

        Example:
            >>> pipeline = ProcessingPipeline(mode='scientific')
            >>> rgb = pipeline.process_to_rgb(
            ...     fits_files=['502nmos.fits', '656nmos.fits', '673nmos.fits'],
            ...     output_dir='output/'
            ... )
        """
        if len(fits_files) < 3:
            raise ValueError(f"Need at least 3 FITS files for RGB, got {len(fits_files)}")

        logger.info(f"Starting {self.mode} workflow with {len(fits_files)} files")

        # Phase 1: Load and calibrate (simplified for now)
        logger.info("Phase 1: Loading FITS files")
        self.phase1_data = self._load_fits_files(fits_files)

        # Phase 2: Normalize and stretch
        logger.info("Phase 2: Normalize and stretch")
        self.phase2_data = self._normalize_and_stretch(
            self.phase1_data,
            interval=kwargs.get('interval', 'auto'),
            stretch=kwargs.get('stretch', 'auto')
        )

        # Phase 3: Compose RGB
        logger.info("Phase 3: Compose RGB")
        self.phase3_rgb = self._compose_rgb(
            self.phase2_data,
            compositor=kwargs.get('compositor', 'auto'),
            stretch_object=kwargs.get('stretch_object', 'auto')
        )

        # Export if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._export_results(output_dir)

        return self.phase3_rgb

    def _load_fits_files(self, fits_files: List[Union[str, Path]]) -> Dict:
        """Load FITS files (Phase 1 simplified).

        Args:
            fits_files: List of FITS file paths

        Returns:
            Dictionary mapping filename to loaded data with metadata
        """
        loaded = {}

        for fits_file in fits_files:
            fits_path = Path(fits_file)
            logger.debug(f"Loading {fits_path.name}")

            # Load FITS (simplified - using astropy directly)
            from astropy.io import fits
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            # Extract basic metadata
            from astro_vision_composer.utilities.metadata import FITSMetadata
            metadata_extractor = FITSMetadata()
            metadata = metadata_extractor.extract_metadata(header)

            # Store
            basename = fits_path.stem
            loaded[basename] = {
                'data': data,
                'metadata': metadata,
                'header': header
            }

            logger.debug(f"  Loaded {basename}: shape={data.shape}, wavelength={metadata.wavelength}nm")

        self.history.record('load_fits', {'count': len(loaded)}, 'Phase1')
        return loaded

    def _normalize_and_stretch(
        self,
        phase1_data: Dict,
        interval: str = 'auto',
        stretch: str = 'auto'
    ) -> Dict:
        """Normalize and stretch data (Phase 2).

        Args:
            phase1_data: Output from Phase 1
            interval: Interval method ('auto', 'zscale', 'percentile', etc.)
            stretch: Stretch method ('auto', 'asinh', 'none', etc.)

        Returns:
            Dictionary with normalized and stretched data
        """
        # Get workflow-specific defaults
        if interval == 'auto':
            interval_obj = self._get_default_interval()
        else:
            interval_obj = self._parse_interval(interval)

        if stretch == 'auto':
            stretch_method = self._get_default_stretch()
        else:
            stretch_method = stretch

        processed = {}

        for name, item in phase1_data.items():
            data = item['data']

            # Normalize
            if isinstance(interval_obj, str) and interval_obj == 'per_channel':
                # Narrowband mode - different per channel (not implemented yet)
                logger.warning("Per-channel intervals not yet implemented, using zscale")
                normalized = self.normalizer.normalize(data, method='zscale')
            else:
                # Use interval object or method name
                if hasattr(interval_obj, 'get_limits'):
                    # It's an astropy interval object
                    vmin, vmax = interval_obj.get_limits(data[np.isfinite(data)])
                    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
                else:
                    # It's a method name
                    normalized = self.normalizer.normalize(data, method=interval_obj)

            # Stretch
            if stretch_method is None or stretch_method == 'none':
                # No stretching (for SDSS workflow)
                stretched = normalized
                logger.debug(f"  {name}: normalized only (no stretch)")
            elif isinstance(stretch_method, str) and stretch_method == 'per_channel':
                # Per-channel stretch (narrowband)
                logger.warning("Per-channel stretch not yet implemented, using asinh")
                stretched = self.stretcher.stretch(normalized, method='asinh', a=0.1)
            else:
                # Apply stretch with method-specific parameters
                if stretch_method in ('asinh', 'sinh'):
                    # Asinh/Sinh accept 'a' parameter
                    stretched = self.stretcher.stretch(normalized, method=stretch_method, a=0.1)
                elif stretch_method == 'power':
                    # Power accepts 'power' parameter
                    stretched = self.stretcher.stretch(normalized, method=stretch_method, power=2.0)
                elif stretch_method == 'histeq':
                    # HistEq doesn't need extra parameters (takes data internally)
                    stretched = self.stretcher.stretch(normalized, method=stretch_method)
                elif stretch_method == 'contrast_bias':
                    # ContrastBias accepts 'contrast' and 'bias' parameters
                    stretched = self.stretcher.stretch(normalized, method=stretch_method, contrast=1.5, bias=0.5)
                else:
                    # Other methods (linear, sqrt, squared, log) don't need parameters
                    stretched = self.stretcher.stretch(normalized, method=stretch_method)
                logger.debug(f"  {name}: normalized + {stretch_method} stretch")

            # Get objects for serialization
            interval_obj_save = self.normalizer.get_interval_object()
            stretch_obj_save = self.stretcher.get_stretch_object()

            processed[name] = {
                'normalized': normalized,
                'stretched': stretched,
                'metadata': item['metadata'],
                'interval_object': interval_obj_save,
                'stretch_object': stretch_obj_save
            }

        self.history.record('normalize_stretch', {
            'interval': str(interval),
            'stretch': str(stretch_method)
        }, 'Phase2')

        return processed

    def _compose_rgb(
        self,
        phase2_data: Dict,
        compositor: str = 'auto',
        stretch_object = 'auto'
    ) -> np.ndarray:
        """Compose RGB image (Phase 3).

        Args:
            phase2_data: Output from Phase 2
            compositor: Compositor method ('auto', 'lupton', 'simple')
            stretch_object: Stretch object for Lupton ('auto', 'linear', object)

        Returns:
            RGB array (float, [0, 1])
        """
        # Get wavelengths for chromatic mapping
        wavelengths = {
            name: item['metadata'].wavelength
            for name, item in phase2_data.items()
            if item['metadata'].wavelength is not None
        }

        if len(wavelengths) < 3:
            raise ValueError("Need wavelength info for at least 3 bands")

        # Map channels by wavelength
        mapping = self.mapper.auto_map_by_wavelength(wavelengths)
        logger.info(f"  Mapped: R={mapping.red}, G={mapping.green}, B={mapping.blue}")

        # Get stretched data for RGB
        r_data = phase2_data[mapping.red]['stretched']
        g_data = phase2_data[mapping.green]['stretched']
        b_data = phase2_data[mapping.blue]['stretched']

        # Determine compositor method
        if compositor == 'auto':
            compositor = self._get_default_compositor()

        # Determine stretch object for Lupton
        if stretch_object == 'auto':
            stretch_object = self._detect_stretch_object(phase2_data)

        # Create RGB composite
        if compositor == 'lupton':
            if stretch_object is None:
                # Use Lupton with auto-calculated stretch (SDSS workflow)
                from astropy.visualization import LuptonAsinhZscaleStretch
                stretch_obj = LuptonAsinhZscaleStretch(r_data, Q=8)
                logger.info("  Using Lupton with auto-calculated stretch (SDSS method)")
            elif isinstance(stretch_object, str) and stretch_object == 'linear':
                stretch_obj = LinearStretch()
                logger.info("  Using Lupton with LinearStretch (pre-stretched data)")
            else:
                stretch_obj = stretch_object
                logger.info(f"  Using Lupton with {type(stretch_obj).__name__}")

            rgb = self.compositor.create_lupton_rgb(
                r=r_data, g=g_data, b=b_data,
                stretch_object=stretch_obj,
                output_dtype=np.float64
            )

        elif compositor == 'simple':
            logger.info("  Using Simple RGB (channel stacking)")
            rgb = self.compositor.create_simple_rgb(
                r=r_data, g=g_data, b=b_data
            )

        else:
            raise ValueError(f"Unknown compositor: {compositor}")

        self.history.record('compose_rgb', {
            'compositor': compositor,
            'mapping': {'red': mapping.red, 'green': mapping.green, 'blue': mapping.blue}
        }, 'Phase3')

        return rgb

    def _get_default_interval(self):
        """Get default interval for workflow mode."""
        defaults = {
            'scientific': ZScaleInterval(),
            'sdss': ZScaleInterval(),
            'aesthetic': AsymmetricPercentileInterval(2, 99.5),
            'narrowband': 'per_channel',
            'custom': ZScaleInterval()
        }
        return defaults.get(self.mode, ZScaleInterval())

    def _get_default_stretch(self):
        """Get default stretch for workflow mode."""
        defaults = {
            'scientific': 'asinh',
            'sdss': None,  # No stretch, Lupton handles it
            'aesthetic': 'histeq',  # Histogram equalization for maximum visual impact
            'narrowband': 'per_channel',
            'custom': 'asinh'
        }
        return defaults.get(self.mode, 'asinh')

    def _get_default_compositor(self) -> str:
        """Get default compositor for workflow mode."""
        defaults = {
            'scientific': 'lupton',
            'sdss': 'lupton',
            'aesthetic': 'lupton',
            'narrowband': 'simple',
            'custom': 'simple'
        }
        return defaults.get(self.mode, 'lupton')

    def _detect_stretch_object(self, phase2_data: Dict):
        """Detect appropriate stretch object based on Phase 2 processing.

        If Phase 2 applied stretching, return LinearStretch (identity).
        If Phase 2 only normalized, return None (let Lupton auto-calculate).
        """
        # Check if any stretch was applied
        sample_item = next(iter(phase2_data.values()))

        if sample_item['stretch_object'] is not None:
            # Phase 2 applied stretch - use LinearStretch
            logger.debug("  Detected pre-stretched data → LinearStretch")
            return 'linear'
        else:
            # Phase 2 only normalized - let Lupton auto-calculate
            logger.debug("  Detected normalized-only data → Auto-calculate stretch")
            return None

    def _parse_interval(self, interval_spec):
        """Parse interval specification string to interval object."""
        if interval_spec == 'zscale':
            return ZScaleInterval()
        elif interval_spec == 'percentile':
            return AsymmetricPercentileInterval(1, 99)
        elif interval_spec == 'minmax':
            return 'minmax'
        else:
            return interval_spec

    def _export_results(self, output_dir: Path):
        """Export pipeline results.

        Args:
            output_dir: Output directory path
        """
        logger.info(f"Exporting results to {output_dir}")

        # Export RGB composite
        rgb_file = output_dir / f"rgb_composite_{self.mode}.png"
        self.exporter.save_png(self.phase3_rgb, rgb_file, bit_depth=8)
        logger.info(f"  Saved: {rgb_file.name}")

        # Export TIFF (archival quality)
        tiff_file = output_dir / f"rgb_composite_{self.mode}.tif"
        self.exporter.save_tiff(self.phase3_rgb, tiff_file, bit_depth=16)
        logger.info(f"  Saved: {tiff_file.name}")

        # Export processing history
        history_file = output_dir / f"processing_history_{self.mode}.txt"
        with open(history_file, 'w') as f:
            for step in self.history.get_history():
                f.write(f"{step.timestamp} [{step.component}] {step.operation}\n")
                if step.parameters:
                    for key, value in step.parameters.items():
                        f.write(f"  {key}: {value}\n")
        logger.info(f"  Saved: {history_file.name}")

        # Export workflow metadata
        metadata = {
            'workflow_mode': self.mode,
            'files_processed': len(self.phase1_data),
            'output_shape': self.phase3_rgb.shape,
            'output_dtype': str(self.phase3_rgb.dtype),
            'output_range': [float(self.phase3_rgb.min()), float(self.phase3_rgb.max())]
        }

        metadata_file = output_dir / f"workflow_metadata_{self.mode}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Saved: {metadata_file.name}")
