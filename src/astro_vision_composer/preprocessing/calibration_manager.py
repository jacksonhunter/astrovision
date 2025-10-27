"""Automatic calibration frame management using ccdproc.

This module provides the CalibrationManager class for automatically detecting,
combining, and applying calibration frames (bias, dark, flat) to raw CCD data.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import ccdproc for CCD calibration
try:
    import ccdproc
    from ccdproc import CCDData, Combiner, ImageFileCollection
    CCDPROC_AVAILABLE = True
except ImportError:
    CCDPROC_AVAILABLE = False
    logger.warning("ccdproc not installed. CalibrationManager will not work. Install with: pip install ccdproc")

from astropy import units as u
from astropy.nddata import StdDevUncertainty


@dataclass
class CalibrationFrames:
    """Container for master calibration frames."""
    master_bias: Optional[CCDData] = None
    master_dark: Optional[CCDData] = None
    master_flats: Dict[str, CCDData] = None  # key: filter name

    def __post_init__(self):
        if self.master_flats is None:
            self.master_flats = {}


class CalibrationManager:
    """Manage calibration frames for raw CCD data processing.

    This class automates the process of:
    1. Finding calibration files in a directory
    2. Combining multiple frames with outlier rejection
    3. Applying calibrations to science frames

    Uses ccdproc for proper error propagation and unit handling.

    Example:
        >>> # Auto-detect and create master frames
        >>> calib_mgr = CalibrationManager('raw_data/calibration/')
        >>> calib_mgr.create_master_calibrations()
        >>>
        >>> # Apply to science frame
        >>> science = CCDData.read('science.fits', unit='adu')
        >>> calibrated = calib_mgr.calibrate(science)
    """

    def __init__(
        self,
        calibration_dir: Path | str,
        master_cache_dir: Optional[Path | str] = None
    ):
        """Initialize CalibrationManager.

        Args:
            calibration_dir: Directory containing raw calibration files
            master_cache_dir: Directory to cache master calibration frames (optional)

        Raises:
            ImportError: If ccdproc is not installed
            ValueError: If calibration_dir does not exist
        """
        if not CCDPROC_AVAILABLE:
            raise ImportError(
                "ccdproc is required for CalibrationManager. "
                "Install with: pip install ccdproc"
            )

        self.calibration_dir = Path(calibration_dir)
        if not self.calibration_dir.exists():
            raise ValueError(f"Calibration directory not found: {calibration_dir}")

        self.master_cache_dir = Path(master_cache_dir) if master_cache_dir else None
        if self.master_cache_dir:
            self.master_cache_dir.mkdir(parents=True, exist_ok=True)

        # ImageFileCollection for finding files
        self.ic = ImageFileCollection(self.calibration_dir, keywords='*')

        # Storage for master calibration frames
        self.calibrations = CalibrationFrames()

        logger.info(f"CalibrationManager initialized: {calibration_dir}")

    def create_master_bias(
        self,
        sigma_clip: bool = True,
        clip_low: float = 3.0,
        clip_high: float = 3.0,
        method: str = 'average'
    ) -> CCDData:
        """Create master bias frame from individual bias frames.

        Args:
            sigma_clip: Apply sigma clipping to reject outliers
            clip_low: Lower sigma threshold for clipping
            clip_high: Upper sigma threshold for clipping
            method: Combination method ('average' or 'median')

        Returns:
            Master bias frame as CCDData

        Raises:
            ValueError: If no bias frames found

        Example:
            >>> master_bias = calib_mgr.create_master_bias(sigma_clip=True)
        """
        # Find bias frames
        bias_files = self.ic.files_filtered(imagetyp='BIAS', include_path=True)

        if len(bias_files) == 0:
            raise ValueError("No bias frames found in calibration directory")

        logger.info(f"Found {len(bias_files)} bias frames")

        # Load bias frames
        bias_list = []
        for bias_file in bias_files:
            ccd = CCDData.read(bias_file, unit='adu')
            bias_list.append(ccd)

        # Combine with sigma clipping
        combiner = Combiner(bias_list)

        if sigma_clip:
            combiner.sigma_clipping(
                low_thresh=clip_low,
                high_thresh=clip_high,
                func=np.ma.median
            )
            logger.debug(f"Sigma clipping: low={clip_low}, high={clip_high}")

        # Combine frames
        if method == 'average':
            master_bias = combiner.average_combine()
        elif method == 'median':
            master_bias = combiner.median_combine()
        else:
            raise ValueError(f"Unknown combination method: {method}")

        # Add metadata
        master_bias.header['COMBINED'] = len(bias_files)
        master_bias.header['COMBTYPE'] = method
        master_bias.header['SIGCLIP'] = sigma_clip

        logger.info(f"Created master bias from {len(bias_files)} frames using {method}")

        # Cache if directory specified
        if self.master_cache_dir:
            cache_file = self.master_cache_dir / 'master_bias.fits'
            master_bias.write(cache_file, overwrite=True)
            logger.debug(f"Cached master bias: {cache_file}")

        self.calibrations.master_bias = master_bias
        return master_bias

    def create_master_dark(
        self,
        exposure_time: Optional[float] = None,
        tolerance: float = 1.0,
        sigma_clip: bool = True,
        clip_low: float = 3.0,
        clip_high: float = 3.0,
        method: str = 'average',
        subtract_bias: bool = True
    ) -> CCDData:
        """Create master dark frame from individual dark frames.

        Args:
            exposure_time: Target exposure time (None = use all darks)
            tolerance: Exposure time tolerance in seconds
            sigma_clip: Apply sigma clipping
            clip_low: Lower sigma threshold
            clip_high: Upper sigma threshold
            method: Combination method
            subtract_bias: Subtract master bias from darks before combining

        Returns:
            Master dark frame as CCDData

        Raises:
            ValueError: If no dark frames found

        Example:
            >>> # Create master dark for 300s exposures
            >>> master_dark = calib_mgr.create_master_dark(exposure_time=300.0)
        """
        # Find dark frames
        if exposure_time is not None:
            # Match exposure time with tolerance
            dark_files = []
            for summary_row in self.ic.summary.iterrows():
                row = summary_row[1]
                if row['imagetyp'].upper() == 'DARK':
                    exp = row.get('exptime', None)
                    if exp and abs(exp - exposure_time) <= tolerance:
                        dark_files.append(self.calibration_dir / row['file'])
        else:
            # Use all dark frames
            dark_files = self.ic.files_filtered(imagetyp='DARK', include_path=True)

        if len(dark_files) == 0:
            msg = f"No dark frames found"
            if exposure_time:
                msg += f" for exposure time {exposure_time}s (±{tolerance}s)"
            raise ValueError(msg)

        logger.info(f"Found {len(dark_files)} dark frames")

        # Load dark frames
        dark_list = []
        for dark_file in dark_files:
            ccd = CCDData.read(dark_file, unit='adu')

            # Subtract bias if requested
            if subtract_bias and self.calibrations.master_bias is not None:
                ccd = ccdproc.subtract_bias(ccd, self.calibrations.master_bias)

            dark_list.append(ccd)

        # Combine
        combiner = Combiner(dark_list)

        if sigma_clip:
            combiner.sigma_clipping(
                low_thresh=clip_low,
                high_thresh=clip_high,
                func=np.ma.median
            )

        if method == 'average':
            master_dark = combiner.average_combine()
        elif method == 'median':
            master_dark = combiner.median_combine()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Add metadata
        master_dark.header['COMBINED'] = len(dark_files)
        master_dark.header['COMBTYPE'] = method
        master_dark.header['SIGCLIP'] = sigma_clip
        master_dark.header['BIASSUB'] = subtract_bias

        logger.info(f"Created master dark from {len(dark_files)} frames")

        # Cache
        if self.master_cache_dir:
            exp_str = f"_{int(exposure_time)}s" if exposure_time else ""
            cache_file = self.master_cache_dir / f'master_dark{exp_str}.fits'
            master_dark.write(cache_file, overwrite=True)

        self.calibrations.master_dark = master_dark
        return master_dark

    def create_master_flat(
        self,
        filter_name: str,
        sigma_clip: bool = True,
        clip_low: float = 3.0,
        clip_high: float = 3.0,
        method: str = 'average',
        subtract_bias: bool = True,
        subtract_dark: bool = False
    ) -> CCDData:
        """Create master flat frame for specific filter.

        Args:
            filter_name: Filter name to match (e.g., 'V', 'R', 'Ha')
            sigma_clip: Apply sigma clipping
            clip_low: Lower sigma threshold
            clip_high: Upper sigma threshold
            method: Combination method
            subtract_bias: Subtract master bias from flats
            subtract_dark: Subtract scaled dark from flats

        Returns:
            Master flat frame (normalized to 1.0)

        Raises:
            ValueError: If no flat frames found for filter

        Example:
            >>> master_flat_v = calib_mgr.create_master_flat('V')
        """
        # Find flat frames for this filter
        flat_files = self.ic.files_filtered(
            imagetyp='FLAT',
            filter=filter_name,
            include_path=True
        )

        if len(flat_files) == 0:
            raise ValueError(f"No flat frames found for filter: {filter_name}")

        logger.info(f"Found {len(flat_files)} flat frames for filter {filter_name}")

        # Load and calibrate flats
        flat_list = []
        for flat_file in flat_files:
            ccd = CCDData.read(flat_file, unit='adu')

            # Subtract bias
            if subtract_bias and self.calibrations.master_bias is not None:
                ccd = ccdproc.subtract_bias(ccd, self.calibrations.master_bias)

            # Subtract dark (scaled by exposure time)
            if subtract_dark and self.calibrations.master_dark is not None:
                ccd = ccdproc.subtract_dark(
                    ccd,
                    self.calibrations.master_dark,
                    exposure_time='exptime',
                    exposure_unit=u.second,
                    scale=True
                )

            flat_list.append(ccd)

        # CRITICAL: Scale flats to common level before combining
        # Each flat has different average level - normalize to median=1.0
        def inv_median(arr):
            """Scaling function: 1 / median."""
            return 1.0 / np.ma.median(arr)

        combiner = Combiner(flat_list)
        combiner.scaling = inv_median  # Scale each flat to median=1.0

        if sigma_clip:
            combiner.sigma_clipping(
                low_thresh=clip_low,
                high_thresh=clip_high,
                func=np.ma.median
            )

        if method == 'average':
            master_flat = combiner.average_combine()
        elif method == 'median':
            master_flat = combiner.median_combine()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize master flat to 1.0
        median_value = np.ma.median(master_flat.data)
        master_flat = master_flat.divide(median_value * master_flat.unit)

        # Add metadata
        master_flat.header['COMBINED'] = len(flat_files)
        master_flat.header['COMBTYPE'] = method
        master_flat.header['SIGCLIP'] = sigma_clip
        master_flat.header['FILTER'] = filter_name
        master_flat.header['BIASSUB'] = subtract_bias
        master_flat.header['DARKSUB'] = subtract_dark
        master_flat.header['FLATNORM'] = float(median_value)

        logger.info(f"Created master flat for {filter_name} from {len(flat_files)} frames")

        # Cache
        if self.master_cache_dir:
            cache_file = self.master_cache_dir / f'master_flat_{filter_name}.fits'
            master_flat.write(cache_file, overwrite=True)

        self.calibrations.master_flats[filter_name] = master_flat
        return master_flat

    def create_master_calibrations(
        self,
        bias: bool = True,
        dark: bool = True,
        flats: Optional[List[str]] = None
    ) -> CalibrationFrames:
        """Create all master calibration frames automatically.

        Args:
            bias: Create master bias
            dark: Create master dark
            flats: List of filter names for master flats (None = auto-detect)

        Returns:
            CalibrationFrames object with all masters

        Example:
            >>> # Create all calibrations automatically
            >>> calibs = calib_mgr.create_master_calibrations()
            >>>
            >>> # Create only bias and specific flats
            >>> calibs = calib_mgr.create_master_calibrations(
            ...     dark=False,
            ...     flats=['V', 'R', 'I']
            ... )
        """
        logger.info("Creating master calibration frames...")

        # Create master bias
        if bias:
            try:
                self.create_master_bias()
            except ValueError as e:
                logger.warning(f"Could not create master bias: {e}")

        # Create master dark
        if dark:
            try:
                self.create_master_dark()
            except ValueError as e:
                logger.warning(f"Could not create master dark: {e}")

        # Create master flats
        if flats is None:
            # Auto-detect filters from flat frames
            flats = set()
            for row in self.ic.summary.iterrows():
                if row[1]['imagetyp'].upper() == 'FLAT':
                    filter_val = row[1].get('filter', None)
                    if filter_val:
                        flats.add(filter_val)
            logger.info(f"Auto-detected filters: {flats}")

        for filter_name in flats:
            try:
                self.create_master_flat(filter_name)
            except ValueError as e:
                logger.warning(f"Could not create master flat for {filter_name}: {e}")

        logger.info("Master calibration creation complete")
        return self.calibrations

    def calibrate(
        self,
        science_frame: CCDData,
        filter_name: Optional[str] = None
    ) -> CCDData:
        """Apply all available calibrations to a science frame.

        Args:
            science_frame: Raw science CCDData
            filter_name: Filter name for flat field (auto-detect from header if None)

        Returns:
            Calibrated CCDData

        Example:
            >>> raw_science = CCDData.read('science.fits', unit='adu')
            >>> calibrated = calib_mgr.calibrate(raw_science, filter_name='V')
        """
        calibrated = science_frame

        # Auto-detect filter if not provided
        if filter_name is None and 'FILTER' in science_frame.header:
            filter_name = science_frame.header['FILTER']

        # Apply bias correction
        if self.calibrations.master_bias is not None:
            calibrated = ccdproc.subtract_bias(calibrated, self.calibrations.master_bias)
            logger.debug("Applied bias correction")

        # Apply dark correction
        if self.calibrations.master_dark is not None:
            calibrated = ccdproc.subtract_dark(
                calibrated,
                self.calibrations.master_dark,
                exposure_time='exptime',
                exposure_unit=u.second,
                scale=True
            )
            logger.debug("Applied dark correction")

        # Apply flat correction
        if filter_name and filter_name in self.calibrations.master_flats:
            calibrated = ccdproc.flat_correct(
                calibrated,
                self.calibrations.master_flats[filter_name]
            )
            logger.debug(f"Applied flat correction for {filter_name}")
        elif filter_name:
            logger.warning(f"No master flat available for filter: {filter_name}")

        return calibrated
