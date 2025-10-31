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
    ZScaleInterval, AsymmetricPercentileInterval, MinMaxInterval, PercentileInterval,
    AsinhStretch, HistEqStretch, LinearStretch, SqrtStretch, SquaredStretch, LogStretch,
    LuptonAsinhZscaleStretch, ImageNormalize
)


WorkflowMode = Literal['scientific', 'sdss', 'aesthetic', 'narrowband', 'custom', 'manual']


class ProcessingPipeline:
    """Flexible astronomical image processing pipeline.

    Implements multiple validated workflows:
    - **scientific**: Preserve photometry (ZScale + Asinh + Lupton/Simple)
    - **sdss**: Auto-optimized Lupton (ZScale + LuptonAsinhZscaleStretch)
    - **aesthetic**: Maximum impact (Percentile + HistEq + Lupton)
    - **narrowband**: Per-channel optimization (false-color)
    - **custom**: User controls everything
    - **manual**: Complete control with per-band normalization/stretching

    Example:
        >>> pipeline = ProcessingPipeline(mode='scientific')
        >>> # Process FITS files through complete pipeline
        >>> rgb = pipeline.process_to_rgb(
        ...     fits_files=['r.fits', 'g.fits', 'b.fits'],
        ...     output_dir='output/'
        ... )

        >>> # Manual mode with custom per-band processing
        >>> pipeline = ProcessingPipeline(mode='manual')
        >>> normalizations = [
        ...     ImageNormalize(interval=ZScaleInterval(), stretch=AsinhStretch(a=0.1)),
        ...     ImageNormalize(interval=PercentileInterval(99), stretch=LogStretch(a=1000)),
        ...     ImageNormalize(interval=MinMaxInterval(), stretch=LinearStretch())
        ... ]
        >>> rgb = pipeline.process_with_normalizations(
        ...     fits_files=['ha.fits', 'oiii.fits', 'sii.fits'],
        ...     normalizations=normalizations
        ... )
    """

    def __init__(
        self,
        mode: WorkflowMode = 'scientific',
        enable_experimental: bool = False,
        calibration_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the ProcessingPipeline.

        Args:
            mode: Workflow mode (scientific, sdss, aesthetic, narrowband, custom, manual)
            enable_experimental: Enable experimental/low-quality features (default: False)
            calibration_dir: Directory containing raw calibration files (bias, dark, flat)
                           If provided, enables automatic calibration of raw data

        .. warning::
            Experimental features (CLAHE, color balance) are disabled by default.
            Only enable if you understand the quality issues. See QUALITY.md for details.

        Example:
            >>> # For pre-calibrated data
            >>> pipeline = ProcessingPipeline(mode='scientific')
            >>>
            >>> # For raw CCD data with calibration frames
            >>> pipeline = ProcessingPipeline(
            ...     mode='scientific',
            ...     calibration_dir='raw_data/calibration/'
            ... )
        """
        # Lazy imports to avoid circular dependency
        from astro_vision_composer.processing import Normalizer, Stretcher
        from astro_vision_composer.postprocessing import (
            ChannelMapper, Compositor, ImageExporter, HistoryTracker
        )

        self.mode = mode
        self.enable_experimental = enable_experimental
        self.history = HistoryTracker()

        # Initialize components
        self.normalizer = Normalizer()
        self.stretcher = Stretcher()
        self.mapper = ChannelMapper()
        self.compositor = Compositor()
        self.exporter = ImageExporter()

        # Initialize calibration manager if calibration directory provided
        self.calib_manager = None
        if calibration_dir:
            try:
                from astro_vision_composer.preprocessing import CalibrationManager
                self.calib_manager = CalibrationManager(calibration_dir)
                logger.info(f"Calibration manager initialized: {calibration_dir}")
            except ImportError as e:
                logger.warning(
                    f"Could not initialize CalibrationManager: {e}. "
                    "Install ccdproc for raw data calibration: pip install ccdproc"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize CalibrationManager: {e}")

        # Workflow state
        self.phase1_data = {}
        self.phase2_data = {}
        self.phase3_rgb = None

        # Warn if experimental features are enabled
        if enable_experimental:
            import warnings
            warnings.warn(
                "Experimental features enabled. These include low-quality components "
                "(CLAHE, color balance) that may produce poor results. "
                "See QUALITY.md for known issues.",
                UserWarning,
                stacklevel=2
            )

        logger.info(f"Pipeline initialized with mode: {mode}, experimental: {enable_experimental}, calibration: {calibration_dir is not None}")

    def process_to_rgb(
        self,
        fits_input,
        output_dir: Optional[Union[str, Path]] = None,
        auto_calibrate: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Process FITS files through complete pipeline to RGB composite.

        This is the main entry point for end-to-end processing.

        Args:
            fits_input: FITS data input (at least 3 items for RGB). One of:
                - List[Union[str, Path, BytesIO, HDUList]]: Multiple FITS inputs
                - str/Path: Single file path (will be wrapped in list)
                - BytesIO: Single stream (will be wrapped in list)
                - HDUList: Single pre-opened FITS (will be wrapped in list)
            output_dir: Optional output directory for intermediate files
            auto_calibrate: Automatically apply calibration if CalibrationManager available
            **kwargs: Override workflow parameters (interval, stretch, etc.)

        Returns:
            RGB image array (float, [0, 1] range)

        Example:
            >>> # File paths (traditional)
            >>> pipeline = ProcessingPipeline(mode='scientific')
            >>> rgb = pipeline.process_to_rgb(
            ...     fits_input=['502nmos.fits', '656nmos.fits', '673nmos.fits'],
            ...     output_dir='output/'
            ... )
            >>>
            >>> # BytesIO streams (Mission Control integration)
            >>> from mission_control import MissionControl
            >>> mc = MissionControl()
            >>> mast = mc.create_service('mast')
            >>> streams = await mast.get_images_async(location, filters=['F850LP', 'F775W', 'F658N'])
            >>> rgb = pipeline.process_to_rgb(fits_input=streams, output_dir='output/')
        """
        # Normalize input to list for validation
        if not isinstance(fits_input, list):
            fits_input_list = [fits_input]
        else:
            fits_input_list = fits_input

        if len(fits_input_list) < 3:
            raise ValueError(f"Need at least 3 FITS inputs for RGB, got {len(fits_input_list)}")

        logger.info(f"Starting {self.mode} workflow with {len(fits_input_list)} inputs")

        # Phase 0: Load or create master calibrations
        if auto_calibrate and self.calib_manager:
            logger.info("Phase 0: Loading/creating master calibration frames")
            try:
                # Try to load cached calibrations first (FAST - <1 second)
                cache_loaded = self.calib_manager.load_cached_calibrations()

                if cache_loaded:
                    n_darks = len(self.calib_manager.calibrations.master_darks)
                    n_flats = len(self.calib_manager.calibrations.master_flats)
                    logger.info(f"✓ Loaded cached calibrations "
                               f"({n_darks} darks, {n_flats} flats)")
                else:
                    # Cache miss, create new ones (SLOW - 30-60 seconds)
                    logger.info("No cached calibrations found, creating from raw frames...")
                    self.calib_manager.create_master_calibrations()
                    logger.info("✓ Created fresh calibrations (cached for future runs)")
            except Exception as e:
                logger.warning(f"Could not load/create calibrations: {e}")

        # Phase 1: Load and calibrate
        logger.info("Phase 1: Loading FITS files")
        self.phase1_data = self._ensure_loaded_data(fits_input, auto_calibrate=auto_calibrate)

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
            lupton_workflow=kwargs.get('lupton_workflow', 'auto')
        )

        # Export if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._export_results(output_dir)

        return self.phase3_rgb

    def process_with_normalizations(
        self,
        fits_input,
        normalizations: List[ImageNormalize],
        compositor: str = 'simple',
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> np.ndarray:
        """Process FITS files with explicit per-band normalizations (manual mode).

        This method provides complete control over each band's processing,
        allowing different intervals and stretches for each channel.

        Args:
            fits_files: List of FITS file paths (must match normalizations length)
            normalizations: List of ImageNormalize objects, one per band
            compositor: RGB compositor method ('lupton' or 'simple')
            output_dir: Optional output directory for results
            **kwargs: Additional compositor options

        Returns:
            RGB image array (float, [0, 1] range)

        Example:
            >>> from astropy.visualization import (
            ...     ImageNormalize, ZScaleInterval, PercentileInterval,
            ...     AsinhStretch, LogStretch
            ... )
            >>> pipeline = ProcessingPipeline(mode='manual')
            >>> # Different processing per narrowband filter
            >>> normalizations = [
            ...     ImageNormalize(interval=ZScaleInterval(), stretch=AsinhStretch(a=0.1)),  # Ha
            ...     ImageNormalize(interval=PercentileInterval(99.5), stretch=LogStretch()),  # OIII
            ...     ImageNormalize(interval=PercentileInterval(98), stretch=AsinhStretch())   # SII
            ... ]
            >>> rgb = pipeline.process_with_normalizations(
            ...     fits_files=['ha.fits', 'oiii.fits', 'sii.fits'],
            ...     normalizations=normalizations,
            ...     compositor='simple'
            ... )
        """
        if len(fits_files) != len(normalizations):
            raise ValueError(f"Number of FITS files ({len(fits_files)}) must match "
                           f"number of normalizations ({len(normalizations)})")

        if len(fits_files) < 3:
            raise ValueError(f"Need at least 3 FITS files for RGB, got {len(fits_files)}")

        logger.info(f"Starting manual workflow with {len(fits_files)} files")

        # Phase 1: Load FITS files
        logger.info("Phase 1: Loading FITS files")
        self.phase1_data = self._load_fits_files(fits_files)

        # Phase 2: Apply custom normalizations
        logger.info("Phase 2: Applying custom normalizations per band")
        self.phase2_data = self._apply_manual_normalizations(
            self.phase1_data,
            normalizations
        )

        # Phase 3: Compose RGB
        logger.info("Phase 3: Compose RGB")
        self.phase3_rgb = self._compose_rgb(
            self.phase2_data,
            compositor=compositor,
            lupton_workflow='pre_stretched' if compositor == 'lupton' else 'auto'
        )

        # Export if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._export_results(output_dir)

        return self.phase3_rgb

    def process_with_arrays(
        self,
        fits_input,
        intervals: List,
        stretches: List,
        compositor: str = 'simple',
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> np.ndarray:
        """Process FITS files with separate interval and stretch arrays.

        Convenience method that constructs ImageNormalize objects from
        separate arrays of intervals and stretches.

        Args:
            fits_files: List of FITS file paths
            intervals: List of astropy interval objects
            stretches: List of astropy stretch objects
            compositor: RGB compositor method
            output_dir: Optional output directory
            **kwargs: Additional options

        Returns:
            RGB image array

        Example:
            >>> from astropy.visualization import (
            ...     ZScaleInterval, PercentileInterval,
            ...     AsinhStretch, LinearStretch
            ... )
            >>> pipeline = ProcessingPipeline(mode='manual')
            >>> intervals = [ZScaleInterval(), PercentileInterval(99.5), PercentileInterval(98)]
            >>> stretches = [AsinhStretch(), LinearStretch(), AsinhStretch()]
            >>> rgb = pipeline.process_with_arrays(
            ...     fits_files=['r.fits', 'g.fits', 'b.fits'],
            ...     intervals=intervals,
            ...     stretches=stretches
            ... )
        """
        if len(intervals) != len(stretches):
            raise ValueError("Intervals and stretches must have same length")

        if len(fits_files) != len(intervals):
            raise ValueError(f"Number of FITS files ({len(fits_files)}) must match "
                           f"number of intervals/stretches ({len(intervals)})")

        # Create ImageNormalize objects from arrays
        normalizations = []
        for i, (interval, stretch) in enumerate(zip(intervals, stretches)):
            # Load data to create normalization
            # Note: This is a simplified approach, actual implementation would be more efficient
            normalizations.append(ImageNormalize(interval=interval, stretch=stretch))

        return self.process_with_normalizations(
            fits_files=fits_files,
            normalizations=normalizations,
            compositor=compositor,
            output_dir=output_dir,
            **kwargs
        )

    def _apply_manual_normalizations(
        self,
        phase1_data: Dict,
        normalizations: List[ImageNormalize]
    ) -> Dict:
        """Apply manual normalizations to each band.

        Args:
            phase1_data: Loaded FITS data
            normalizations: List of ImageNormalize objects

        Returns:
            Dictionary with normalized data
        """
        processed = {}

        # Convert phase1_data keys to list to ensure order
        bands = list(phase1_data.keys())

        for i, (name, norm) in enumerate(zip(bands, normalizations)):
            data = phase1_data[name]['data']

            # Apply the normalization
            processed_data = norm(data)

            logger.debug(f"  {name}: Applied custom normalization {i+1}/{len(normalizations)}")

            processed[name] = {
                'normalized': processed_data,
                'stretched': processed_data,  # Already includes stretch from ImageNormalize
                'metadata': phase1_data[name]['metadata'],
                'norm_object': norm,
                'interval_object': norm.interval,
                'stretch_object': norm.stretch
            }

        self.history.record('manual_normalize', {
            'count': len(normalizations)
        }, 'Phase2')

        return processed

    def apply_enhancement(
        self,
        data: np.ndarray,
        method: str = 'unsharp_mask',
        **kwargs
    ) -> np.ndarray:
        """Apply enhancement techniques with quality warnings.

        Args:
            data: Input image data
            method: Enhancement method ('unsharp_mask', 'clahe', etc.)
            **kwargs: Method-specific parameters

        Returns:
            Enhanced image data

        Raises:
            RuntimeError: If experimental methods used without enabling

        Example:
            >>> # Safe enhancement
            >>> enhanced = pipeline.apply_enhancement(data, method='unsharp_mask', sigma=2.0)
            >>>
            >>> # Experimental (requires enable_experimental=True)
            >>> enhanced = pipeline.apply_enhancement(data, method='clahe')
        """
        # List of experimental/low-quality methods
        experimental_methods = ['clahe', 'white_balance', 'color_temperature']

        if method in experimental_methods and not self.enable_experimental:
            raise RuntimeError(
                f"Enhancement method '{method}' is experimental/low-quality and disabled by default. "
                f"To use it, initialize pipeline with enable_experimental=True and read QUALITY.md first."
            )

        # Import enhancer only when needed
        from astro_vision_composer.processing import Enhancer
        enhancer = Enhancer()

        # Apply the enhancement
        if method == 'unsharp_mask':
            return enhancer.unsharp_mask(data, **kwargs)
        elif method == 'clahe':
            import warnings
            warnings.warn(
                "CLAHE is a low-quality implementation. See QUALITY.md for issues.",
                UserWarning,
                stacklevel=2
            )
            return enhancer.apply_clahe(data, **kwargs)
        elif method == 'enhance_stars':
            return enhancer.enhance_stars(data, **kwargs)
        elif method == 'local_contrast':
            return enhancer.local_contrast_enhancement(data, **kwargs)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")


    def _ensure_loaded_data(
        self,
        fits_input,
        auto_calibrate: bool = True
    ):
        """Factory method for polymorphic FITS input handling.

        Accepts various input types and returns standardized Phase 1 data format.
        This enables Mission Control integration (BytesIO streams) while maintaining
        backward compatibility with file paths.

        Args:
            fits_input: One of:
                - List[Union[str, Path, BytesIO, HDUList]]: Multiple inputs (mixed types allowed)
                - Dict: Pre-loaded phase1_data (pass-through)
                - str/Path: Single file path
                - BytesIO: Single in-memory stream
                - HDUList: Single pre-opened FITS
            auto_calibrate: Apply calibration if CalibrationManager available (file paths only)

        Returns:
            Dictionary mapping name to loaded data with metadata

        Raises:
            TypeError: If input type is not supported
            ValueError: If list is empty
        """
        from io import BytesIO
        from astropy.io.fits import HDUList

        # Already processed data dict - pass through
        if isinstance(fits_input, dict) and any(k in fits_input for k in ['bands', 'data', 'metadata']):
            logger.info("Using pre-loaded data")
            return fits_input

        # Single input - wrap in list for uniform processing
        if not isinstance(fits_input, list):
            fits_input = [fits_input]

        if len(fits_input) == 0:
            raise ValueError("fits_input cannot be empty")

        # Process each item (handles mixed types)
        loaded = {}
        for i, item in enumerate(fits_input):
            if isinstance(item, (str, Path)):
                loaded.update(self._load_single_file(item, auto_calibrate=auto_calibrate))
            elif isinstance(item, BytesIO):
                loaded.update(self._load_from_stream(item, index=i))
            elif isinstance(item, HDUList):
                loaded.update(self._load_from_hdulist(item, index=i))
            else:
                raise TypeError(f"Unsupported input type at index {i}: {type(item).__name__}")

        calibrated = auto_calibrate and self.calib_manager is not None and any(
            isinstance(item, (str, Path)) for item in ([fits_input] if not isinstance(fits_input, list) else fits_input)
        )
        self.history.record('load_fits', {'count': len(loaded), 'calibrated': calibrated}, 'Phase1')
        return loaded

    def _load_single_file(self, fits_file, auto_calibrate: bool = True):
        """Load FITS from file path with optional calibration.

        Args:
            fits_file: Path to FITS file
            auto_calibrate: Apply calibration if CalibrationManager available

        Returns:
            Dictionary with single entry: {basename: {data, metadata, header}}
        """
        from astropy.io import fits

        fits_path = Path(fits_file)
        logger.debug(f"Loading {fits_path.name}")

        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        # Apply calibration if available
        if auto_calibrate and self.calib_manager:
            try:
                from ccdproc import CCDData

                # Use CalibrationManager's unit detection if available
                if hasattr(self.calib_manager, '_get_unit_from_header'):
                    unit = self.calib_manager._get_unit_from_header(fits_path)
                    logger.debug(f"  Detected unit: {unit}")
                else:
                    # Fallback: try BUNIT keyword, default to 'adu'
                    bunit = header.get('BUNIT', 'adu')
                    unit = bunit.lower() if isinstance(bunit, str) else 'adu'
                    logger.debug(f"  Using unit from BUNIT: {unit}")

                ccd = CCDData(data, unit=unit, header=header)

                # Apply calibration
                filter_name = header.get('FILTER', None)
                calibrated_ccd = self.calib_manager.calibrate(ccd, filter_name=filter_name)

                # Extract calibrated data
                data = calibrated_ccd.data
                header = calibrated_ccd.header
                logger.debug(f"  Applied calibration to {fits_path.name}")

            except Exception as e:
                logger.warning(f"Could not calibrate {fits_path.name}: {e}")

        result = self._extract_data_from_hdulist_data(data, header, fits_path.stem)
        return result

    def _load_from_stream(self, stream, index: int):
        """Load FITS from BytesIO stream (Mission Control integration).

        Args:
            stream: BytesIO object containing FITS data
            index: Index for naming (e.g., 'stream_0')

        Returns:
            Dictionary with single entry: {name: {data, metadata, header}}
        """
        from astropy.io import fits

        logger.debug(f"Loading stream {index}")

        # astropy.io.fits.open() natively supports BytesIO
        with fits.open(stream) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        result = self._extract_data_from_hdulist_data(data, header, f"stream_{index}")
        return result

    def _load_from_hdulist(self, hdul, index: int):
        """Load from pre-opened HDUList.

        Args:
            hdul: Opened astropy HDUList
            index: Index for naming (e.g., 'hdulist_0')

        Returns:
            Dictionary with single entry: {name: {data, metadata, header}}
        """
        logger.debug(f"Using HDUList {index}")

        # Don't close HDUList - caller owns it
        data = hdul[0].data
        header = hdul[0].header

        result = self._extract_data_from_hdulist_data(data, header, f"hdulist_{index}")
        return result

    def _extract_data_from_hdulist_data(self, data, header, name: str):
        """Extract metadata from data and header.

        Shared logic for file/stream/HDUList loading.

        Args:
            data: FITS data array
            header: FITS header
            name: Name/identifier for this data

        Returns:
            Dictionary with single entry: {name: {data, metadata, header}}
        """
        from astro_vision_composer.utilities.metadata import FITSMetadata

        # Extract metadata
        metadata_extractor = FITSMetadata()
        metadata = metadata_extractor.extract_metadata(header)

        logger.debug(f"  Loaded {name}: shape={data.shape}, wavelength={metadata.wavelength}nm")

        return {
            name: {
                'data': data,
                'metadata': metadata,
                'header': header
            }
        }

    def _load_fits_files(
        self,
        fits_input,
        auto_calibrate: bool = True
    ) -> Dict:
        """Load FITS files and optionally apply calibration (Phase 1).

        Args:
            fits_files: List of FITS file paths
            auto_calibrate: Apply calibration if CalibrationManager available

        Returns:
            Dictionary mapping filename to loaded data with metadata
        """
        return self._ensure_loaded_data(fits_files, auto_calibrate=auto_calibrate)

    def _normalize_and_stretch(
        self,
        phase1_data: Dict,
        interval: Union[str, List] = 'auto',
        stretch: Union[str, List] = 'auto'
    ) -> Dict:
        """Normalize and stretch data using astropy ImageNormalize (Phase 2).

        This method now uses astropy's ImageNormalize to combine normalization
        and stretching in a single operation, following best practices.

        Supports per-channel intervals and stretches for narrowband imaging.

        Args:
            phase1_data: Output from Phase 1
            interval: Interval method or list of interval objects (one per band)
                     - String: 'auto', 'zscale', 'percentile', 'per_channel'
                     - List: One interval object per band
            stretch: Stretch method or list of stretch objects (one per band)
                     - String: 'auto', 'asinh', 'none', etc.
                     - List: One stretch object per band

        Returns:
            Dictionary with normalized and stretched data
        """
        # Handle per-channel intervals/stretches (lists)
        is_per_channel = isinstance(interval, list) or isinstance(stretch, list)

        if is_per_channel:
            # Per-channel mode - validate list lengths
            band_names = list(phase1_data.keys())
            n_bands = len(band_names)

            # Convert to lists if not already
            if not isinstance(interval, list):
                # Single interval for all bands
                interval_objs = [self._parse_interval(interval) if interval != 'auto'
                                else self._get_default_interval()] * n_bands
            else:
                interval_objs = interval
                if len(interval_objs) != n_bands:
                    raise ValueError(
                        f"Number of intervals ({len(interval_objs)}) must match "
                        f"number of bands ({n_bands})"
                    )

            if not isinstance(stretch, list):
                # Single stretch for all bands
                stretch_objs = [self._parse_stretch(stretch) if stretch != 'auto'
                               else self._get_default_stretch_object()] * n_bands
            else:
                stretch_objs = stretch
                if len(stretch_objs) != n_bands:
                    raise ValueError(
                        f"Number of stretches ({len(stretch_objs)}) must match "
                        f"number of bands ({n_bands})"
                    )

            logger.info(f"Using per-channel normalization for {n_bands} bands")

            # Process each band with its specific interval/stretch
            processed = {}
            for i, (name, item) in enumerate(phase1_data.items()):
                data = item['data']
                interval_obj = interval_objs[i]
                stretch_obj = stretch_objs[i]

                # Create ImageNormalize for this band
                if stretch_obj is None:
                    norm = ImageNormalize(data, interval=interval_obj, stretch=LinearStretch())
                else:
                    norm = ImageNormalize(data, interval=interval_obj, stretch=stretch_obj)

                processed_data = norm(data)
                logger.info(
                    f"  {name}: {type(interval_obj).__name__} + {type(stretch_obj).__name__ if stretch_obj else 'LinearStretch'}"
                )

                processed[name] = {
                    'data': processed_data,
                    'metadata': item['metadata'],
                    'header': item['header'],
                    'normalization': norm
                }

            return processed

        # Single interval/stretch for all bands (original behavior)
        if interval == 'auto' or interval == 'per_channel':
            interval_obj = self._get_default_interval()
        else:
            interval_obj = self._parse_interval(interval)

        if stretch == 'auto':
            stretch_obj = self._get_default_stretch_object()
        else:
            stretch_obj = self._parse_stretch(stretch)

        processed = {}

        for name, item in phase1_data.items():
            data = item['data']

            # Create ImageNormalize object combining interval and stretch
            # This is the astropy best practice way
            if stretch_obj is None:
                # No stretching (for SDSS workflow - will use Lupton later)
                # Use LinearStretch (identity) with the interval
                norm = ImageNormalize(data, interval=interval_obj, stretch=LinearStretch())
                processed_data = norm(data)
                logger.debug(f"  {name}: normalized with {type(interval_obj).__name__}, no stretch")
            elif isinstance(stretch_obj, str) and stretch_obj == 'histeq':
                # Special case: HistEqStretch requires data to compute histogram
                histeq_stretch = HistEqStretch(data)
                norm = ImageNormalize(data, interval=interval_obj, stretch=histeq_stretch)
                processed_data = norm(data)
                logger.debug(f"  {name}: ImageNormalize with {type(interval_obj).__name__} + HistEqStretch")
                stretch_obj = histeq_stretch  # Store the actual object for later
            else:
                # Combined normalization + stretch in single ImageNormalize
                norm = ImageNormalize(data, interval=interval_obj, stretch=stretch_obj)
                processed_data = norm(data)
                logger.debug(f"  {name}: ImageNormalize with {type(interval_obj).__name__} + {type(stretch_obj).__name__}")

            # Store processed data and normalization objects
            processed[name] = {
                'normalized': processed_data,  # Keep naming for compatibility
                'stretched': processed_data,    # This is the final processed data
                'metadata': item['metadata'],
                'norm_object': norm,           # Store the ImageNormalize object
                'interval_object': interval_obj,
                'stretch_object': stretch_obj
            }

        self.history.record('normalize_stretch', {
            'interval': str(type(interval_obj).__name__),
            'stretch': str(type(stretch_obj).__name__) if stretch_obj else 'None'
        }, 'Phase2')

        return processed

    def _compose_rgb(
        self,
        phase2_data: Dict,
        compositor: str = 'auto',
        lupton_workflow: str = 'auto'
    ) -> np.ndarray:
        """Compose RGB image (Phase 3) with simplified Lupton workflows.

        Implements 3 canonical Lupton workflows explicitly:
        - 'pre_stretched': Data was stretched in Phase 2, use LinearStretch
        - 'sdss_auto': Let Lupton auto-calculate stretch from data
        - 'manual': Provide explicit stretch parameters

        Args:
            phase2_data: Output from Phase 2
            compositor: Compositor method ('auto', 'lupton', 'simple')
            lupton_workflow: Lupton workflow type ('auto', 'pre_stretched', 'sdss_auto', 'manual')

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

        # Create RGB composite
        if compositor == 'lupton':
            # Determine Lupton workflow
            if lupton_workflow == 'auto':
                # Auto-detect based on mode and Phase 2 processing
                if self.mode == 'sdss':
                    lupton_workflow = 'sdss_auto'
                elif phase2_data[mapping.red]['stretch_object'] is not None:
                    lupton_workflow = 'pre_stretched'
                else:
                    lupton_workflow = 'sdss_auto'

            # Implement the 3 canonical Lupton workflows
            if lupton_workflow == 'pre_stretched':
                # Workflow A: Pre-stretched data, use identity transform
                stretch_obj = LinearStretch()
                logger.info("  Lupton Workflow A: Pre-stretched data with LinearStretch (identity)")

            elif lupton_workflow == 'sdss_auto':
                # Workflow B: SDSS auto-calculated from data
                from astropy.visualization import LuptonAsinhZscaleStretch
                stretch_obj = LuptonAsinhZscaleStretch(r_data, Q=8)
                logger.info("  Lupton Workflow B: SDSS auto-calculated stretch")

            elif lupton_workflow == 'manual':
                # Workflow C: Manual Lupton stretch parameters
                from astropy.visualization import LuptonAsinhStretch
                stretch_obj = LuptonAsinhStretch(stretch=5, Q=8)
                logger.info("  Lupton Workflow C: Manual stretch parameters")

            else:
                raise ValueError(f"Unknown Lupton workflow: {lupton_workflow}")

            # Apply Lupton RGB composition
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
            'lupton_workflow': lupton_workflow if compositor == 'lupton' else 'N/A',
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

    def _get_default_stretch_object(self):
        """Get default stretch object for workflow mode.

        Returns astropy stretch objects instead of string names,
        following best practices for ImageNormalize.

        Note: HistEqStretch requires data, so it's returned as string 'histeq'
        and created later in _normalize_and_stretch() when data is available.
        """
        defaults = {
            'scientific': AsinhStretch(a=0.1),
            'sdss': None,  # No stretch, Lupton handles it
            'aesthetic': 'histeq',  # Special case - requires data to create
            'narrowband': AsinhStretch(a=0.1),  # Per-channel not implemented yet
            'custom': AsinhStretch(a=0.1)
        }
        return defaults.get(self.mode, AsinhStretch(a=0.1))

    def _parse_stretch(self, stretch_spec):
        """Parse stretch specification to astropy stretch object.

        Args:
            stretch_spec: String name or stretch object

        Returns:
            Astropy stretch object, string 'histeq', or None

        Note: 'histeq' is returned as string because HistEqStretch
        requires data and will be created later in _normalize_and_stretch().
        """
        if stretch_spec is None or stretch_spec == 'none':
            return None
        elif isinstance(stretch_spec, str):
            # Convert string to stretch object
            stretch_map = {
                'linear': LinearStretch(),
                'sqrt': SqrtStretch(),
                'squared': SquaredStretch(),
                'asinh': AsinhStretch(a=0.1),
                'sinh': AsinhStretch(a=0.1),  # Similar to asinh
                'log': LogStretch(a=1000),
                'histeq': 'histeq'  # Special case - return string, not object
            }
            return stretch_map.get(stretch_spec, AsinhStretch(a=0.1))
        else:
            # Assume it's already a stretch object
            return stretch_spec

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
