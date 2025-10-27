"""Automatic calibration frame management using ccdproc.

This module provides the CalibrationManager class for automatically detecting,
combining, and applying calibration frames (bias, dark, flat) to raw CCD data.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Union
from pathlib import Path
import numpy as np
import logging
from dataclasses import dataclass
import re

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
from astropy.io import fits


@dataclass
class CalibrationFrames:
    """Container for master calibration frames."""
    master_bias: Optional[CCDData] = None
    master_darks: Dict[float, CCDData] = None  # key: exposure time in seconds
    master_flats: Dict[str, CCDData] = None  # key: filter name
    bad_pixel_mask: Optional[np.ndarray] = None  # Boolean mask of bad pixels

    def __post_init__(self):
        if self.master_darks is None:
            self.master_darks = {}
        if self.master_flats is None:
            self.master_flats = {}

    def get_dark_for_exposure(self, exp_time: float, tolerance: float = 5.0) -> Optional[CCDData]:
        """Get dark frame for requested exposure time.

        Args:
            exp_time: Requested exposure time in seconds
            tolerance: Maximum acceptable difference in seconds

        Returns:
            Dark frame closest to requested exposure time, or None if none within tolerance

        Notes
        -----
        If exact match exists, returns it. Otherwise returns closest match
        within tolerance. This is safe because ccdproc.subtract_dark with
        scale=True can scale any dark to any exposure time.

        Examples
        --------
        >>> calibs = CalibrationFrames()
        >>> calibs.master_darks = {30.0: dark30, 60.0: dark60, 300.0: dark300}
        >>> # Exact match
        >>> dark = calibs.get_dark_for_exposure(60.0)  # Returns dark60
        >>> # Closest match within tolerance
        >>> dark = calibs.get_dark_for_exposure(65.0, tolerance=10.0)  # Returns dark60
        >>> # No match within tolerance
        >>> dark = calibs.get_dark_for_exposure(150.0, tolerance=5.0)  # Returns None
        """
        if not self.master_darks:
            return None

        # Check for exact match first
        if exp_time in self.master_darks:
            return self.master_darks[exp_time]

        # Find closest match
        closest_exp = min(self.master_darks.keys(), key=lambda x: abs(x - exp_time))
        diff = abs(closest_exp - exp_time)

        if diff <= tolerance:
            return self.master_darks[closest_exp]

        return None


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

    # Standard IMAGETYP variations across observatories
    IMAGETYP_KEYWORDS = ['imagetyp', 'obstype', 'frametype', 'frame', 'imagecat']

    def __init__(
        self,
        calibration_dir: Path | str,
        master_cache_dir: Optional[Path | str] = None,
        overscan_region: Optional[str] = None,
        trim_region: Optional[str] = None,
        overscan_axis: int = 1,
        gain: Optional[float] = None,
        readnoise: Optional[float] = None,
        mem_limit: float = 16e9
    ):
        """Initialize CalibrationManager.

        Args:
            calibration_dir: Directory containing raw calibration files
            master_cache_dir: Directory to cache master calibration frames (optional)
            overscan_region: FITS section for overscan (e.g., '[2049:2080, :]')
            trim_region: FITS section for trimming (e.g., '[1:2048, 1:2048]')
            overscan_axis: Axis along which overscan is subtracted (0 or 1, default: 1)
            gain: CCD gain in electron/adu (for uncertainty calculation, optional)
            readnoise: CCD read noise in electrons (for uncertainty calculation, optional)
            mem_limit: Memory limit in bytes for combining (default 16 GB)

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

        # Overscan and trim settings
        self.overscan_region = overscan_region
        self.trim_region = trim_region
        self.overscan_axis = overscan_axis

        # CCD characteristics for uncertainty calculation
        self.gain = gain * u.electron / u.adu if gain else None
        self.readnoise = readnoise * u.electron if readnoise else None

        # Memory limit for combining
        self.mem_limit = mem_limit

        # ImageFileCollection for finding files
        self.ic = ImageFileCollection(self.calibration_dir, keywords='*')

        # Storage for master calibration frames
        self.calibrations = CalibrationFrames()

        logger.info(f"CalibrationManager initialized: {calibration_dir}")

    def _get_unit_from_header(self, file_path: Path) -> u.Unit:
        """Determine data unit from FITS header.

        Args:
            file_path: Path to FITS file

        Returns:
            astropy Unit object (default: adu)
        """
        with fits.open(file_path) as hdul:
            bunit = hdul[0].header.get('BUNIT', 'adu').lower()
            # Normalize common variations
            unit_map = {
                'adu': u.adu,
                'adu/s': u.adu / u.s,
                'count': u.adu,
                'counts': u.adu,
                'electron': u.electron,
                'electrons': u.electron,
                'dn': u.adu,  # Data Number = ADU
                'adu/sec': u.adu / u.s
            }
            return unit_map.get(bunit, u.adu)

    def _get_imagetyp(self, header_dict: Dict) -> Optional[str]:
        """Get normalized IMAGETYP from header (case-insensitive).

        Args:
            header_dict: Dictionary of header values from ImageFileCollection

        Returns:
            Uppercase IMAGETYP value or None
        """
        for keyword in self.IMAGETYP_KEYWORDS:
            value = header_dict.get(keyword)
            if value:
                return str(value).upper().strip()
        return None

    def _match_filter_name(self, filter_value: str, target_filter: str) -> bool:
        """Flexible filter name matching.

        Handles variations like:
        - 'V' matches 'V-band', 'V_filter', 'Johnson V'
        - Case-insensitive
        - Substring matching

        Args:
            filter_value: Filter value from FITS header
            target_filter: Target filter to match

        Returns:
            True if filters match
        """
        filter_clean = filter_value.strip().upper()
        target_clean = target_filter.strip().upper()

        # Exact match
        if filter_clean == target_clean:
            return True

        # Substring match
        if target_clean in filter_clean or filter_clean in target_clean:
            return True

        return False

    def _preprocess_frame(
        self,
        file_path: Path,
        add_uncertainty: bool = False,
        reject_cosmics: bool = False
    ) -> CCDData:
        """Load and preprocess a single frame (overscan/trim/uncertainty/CR).

        Args:
            file_path: Path to FITS file
            add_uncertainty: Create uncertainty frame using gain/readnoise
            reject_cosmics: Apply cosmic ray rejection (LAcosmic)

        Returns:
            Preprocessed CCDData

        Notes
        -----
        This method implements proper ccdproc preprocessing workflow:
        1. Load with correct units
        2. Subtract overscan (if configured) using FITS section syntax
        3. Trim image (if configured)
        4. Add uncertainty frame (if requested and gain/readnoise available)
        5. Reject cosmic rays (if requested)

        The overscan subtraction uses ccdproc.subtract_overscan with fits_section
        parameter, which handles FITS→Python axis conversion automatically.

        See Also
        --------
        ccdproc.subtract_overscan : Overscan subtraction with FITS sections
        ccdproc.trim_image : Image trimming
        ccdproc.create_deviation : Uncertainty frame creation
        ccdproc.cosmicray_lacosmic : Cosmic ray rejection

        References
        ----------
        .. [1] ccdproc documentation: https://ccdproc.readthedocs.io/
        .. [2] "Reduction toolbox": https://ccdproc.readthedocs.io/en/latest/reduction_toolbox.html
        """
        # Determine unit from header
        unit = self._get_unit_from_header(file_path)
        ccd = CCDData.read(file_path, unit=unit)

        # Apply overscan subtraction if configured
        if self.overscan_region:
            try:
                # Use ccdproc's subtract_overscan with FITS section
                # This handles FITS→Python axis conversion automatically (safer than eval!)
                # From ccdproc docs: overscan_axis follows Python convention (0 or 1)
                ccd = ccdproc.subtract_overscan(
                    ccd,
                    fits_section=self.overscan_region,
                    overscan_axis=self.overscan_axis,
                    median=True,
                    model=None  # Can use models.Polynomial1D(1) for fitting
                )
                logger.debug(f"Subtracted overscan: {self.overscan_region} (axis={self.overscan_axis})")
            except Exception as e:
                logger.warning(f"Failed to subtract overscan: {e}")

        # Apply trim if configured
        if self.trim_region:
            try:
                ccd = ccdproc.trim_image(ccd, fits_section=self.trim_region)
                logger.debug(f"Trimmed image: {self.trim_region}")
            except Exception as e:
                logger.warning(f"Failed to trim image: {e}")

        # Add uncertainty frame if requested and gain/readnoise available
        if add_uncertainty and self.gain and self.readnoise:
            try:
                ccd = ccdproc.create_deviation(
                    ccd,
                    gain=self.gain,
                    readnoise=self.readnoise
                )
                logger.debug(f"Added uncertainty frame (gain={self.gain}, readnoise={self.readnoise})")
            except Exception as e:
                logger.warning(f"Failed to add uncertainty: {e}")

        # Reject cosmic rays if requested
        if reject_cosmics:
            try:
                # From ccdproc docs: Use gain-corrected image OR supply gain
                # We'll supply gain if available
                cr_kwargs = {}
                if self.gain:
                    # Extract numeric value from Quantity
                    cr_kwargs['gain'] = self.gain.value if hasattr(self.gain, 'value') else self.gain

                ccd = ccdproc.cosmicray_lacosmic(
                    ccd,
                    sigclip=5.0,
                    **cr_kwargs
                )
                logger.debug("Rejected cosmic rays (LAcosmic)")
            except Exception as e:
                logger.warning(f"Failed to reject cosmic rays: {e}")

        return ccd

    def _validate_master_frame(self, master: CCDData, frame_type: str) -> None:
        """Validate master calibration frame quality.

        Args:
            master: Master calibration frame
            frame_type: Type of frame ('bias', 'dark', 'flat')

        Raises:
            ValueError: If frame fails validation
        """
        # Check for all-NaN data
        if np.all(np.isnan(master.data)):
            raise ValueError(f"Master {frame_type} is all NaN!")

        # Check for zero variance (suspicious)
        if master.data.std() == 0:
            logger.warning(f"Master {frame_type} has zero variance - possible issue")

        # Check for reasonable statistics
        median = np.ma.median(master.data)
        std = np.ma.std(master.data)

        logger.debug(f"Master {frame_type} stats: median={median:.2f}, std={std:.2f}")

        # Frame-specific checks
        if frame_type == 'flat':
            # Flats should be normalized near 1.0
            if not (0.5 < median < 1.5):
                logger.warning(f"Master flat median={median:.2f}, expected ~1.0")

        elif frame_type in ('bias', 'dark'):
            # Bias/dark should be positive (ADU offset or dark current)
            if median < 0:
                logger.warning(f"Master {frame_type} has negative median={median:.2f}")

    def load_cached_calibrations(self) -> bool:
        """Load master calibration frames from cache.

        Returns:
            True if any calibrations loaded successfully

        Example:
            >>> calib_mgr = CalibrationManager('raw_data/calibration/',
            ...                                 master_cache_dir='masters/')
            >>> if calib_mgr.load_cached_calibrations():
            ...     print("Using cached calibrations")
            ... else:
            ...     calib_mgr.create_master_calibrations()
        """
        if not self.master_cache_dir or not self.master_cache_dir.exists():
            return False

        loaded_count = 0

        # Try to load master bias
        bias_path = self.master_cache_dir / 'master_bias.fits'
        if bias_path.exists():
            try:
                self.calibrations.master_bias = CCDData.read(bias_path)
                logger.info(f"Loaded cached master bias from {bias_path}")
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to load cached bias: {e}")

        # Try to load master dark(s) - now handles multiple exposure times
        for dark_file in self.master_cache_dir.glob('master_dark_*.fits'):
            try:
                master_dark = CCDData.read(dark_file)
                # Extract exposure time from header (more reliable than filename)
                exp_time = master_dark.header.get('EXPTIME', None)
                if exp_time is not None:
                    self.calibrations.master_darks[float(exp_time)] = master_dark
                    logger.info(f"Loaded cached master dark for {exp_time}s from {dark_file.name}")
                    loaded_count += 1
                else:
                    logger.warning(f"No EXPTIME in {dark_file.name}, skipping")
            except Exception as e:
                logger.warning(f"Failed to load cached dark from {dark_file.name}: {e}")

        if self.calibrations.master_darks:
            logger.info(f"Loaded {len(self.calibrations.master_darks)} dark exposure time(s): "
                       f"{sorted(self.calibrations.master_darks.keys())}s")

        # Try to load master flats
        for flat_file in self.master_cache_dir.glob('master_flat_*.fits'):
            try:
                master_flat = CCDData.read(flat_file)
                # Extract filter name from filename
                filter_name = flat_file.stem.replace('master_flat_', '')
                self.calibrations.master_flats[filter_name] = master_flat
                logger.info(f"Loaded cached master flat for {filter_name} from {flat_file}")
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to load cached flat: {e}")

        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} cached calibration frame(s)")
            return True
        return False

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

        # Load and preprocess bias frames
        bias_list = []
        for bias_file in bias_files:
            ccd = self._preprocess_frame(bias_file)
            bias_list.append(ccd)

        # Combine with sigma clipping
        combiner = Combiner(bias_list)
        combiner.mem_limit = self.mem_limit  # Memory-efficient combining

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
        master_bias.header['FRAMTYPE'] = 'MASTER_BIAS'

        # Validate master frame
        self._validate_master_frame(master_bias, 'bias')

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
            exposure_time: Target exposure time (None = create for all unique exposure times)
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

        Notes
        -----
        By default (subtract_bias=True), creates a bias-subtracted master dark.
        This is the recommended pattern when using exposure time scaling with
        ccdproc.subtract_dark(scale=True). The resulting dark contains only
        dark current, not bias offset.

        If no exact exposure time match is found, uses closest available darks
        within tolerance. This is safe because ccdproc can scale darks to any
        exposure time with scale=True.

        See Also
        --------
        create_master_dark_library : Create master darks for all exposure times
        calibrate : Apply calibrations to science frame

        Example:
            >>> # Create master dark for 300s exposures
            >>> master_dark = calib_mgr.create_master_dark(exposure_time=300.0)
        """
        # Find dark frames using flexible IMAGETYP matching
        if exposure_time is not None:
            # Match exposure time with tolerance
            dark_files = []
            for row in self.ic.summary:
                row_dict = {col: row[col] for col in self.ic.summary.colnames}
                imagetyp = self._get_imagetyp(row_dict)
                if imagetyp == 'DARK':
                    exp = row_dict.get('exptime', None)
                    if exp and abs(exp - exposure_time) <= tolerance:
                        dark_files.append(self.calibration_dir / row_dict['file'])
        else:
            # Use all dark frames
            dark_files = []
            for row in self.ic.summary:
                row_dict = {col: row[col] for col in self.ic.summary.colnames}
                imagetyp = self._get_imagetyp(row_dict)
                if imagetyp == 'DARK':
                    dark_files.append(self.calibration_dir / row_dict['file'])

        if len(dark_files) == 0:
            msg = f"No dark frames found"
            if exposure_time:
                msg += f" for exposure time {exposure_time}s (±{tolerance}s)"
            raise ValueError(msg)

        logger.info(f"Found {len(dark_files)} dark frames")

        # Load and preprocess dark frames
        dark_list = []
        for dark_file in dark_files:
            ccd = self._preprocess_frame(dark_file)

            # Subtract bias if requested
            if subtract_bias and self.calibrations.master_bias is not None:
                ccd = ccdproc.subtract_bias(ccd, self.calibrations.master_bias)

            dark_list.append(ccd)

        # Combine
        combiner = Combiner(dark_list)
        combiner.mem_limit = self.mem_limit  # Memory-efficient combining

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

        # Add metadata (CRITICAL: store exposure time for identification!)
        master_dark.header['COMBINED'] = len(dark_files)
        master_dark.header['COMBTYPE'] = method
        master_dark.header['SIGCLIP'] = sigma_clip
        master_dark.header['BIASSUB'] = subtract_bias
        master_dark.header['FRAMTYPE'] = 'MASTER_DARK'
        if exposure_time:
            master_dark.header['EXPTIME'] = exposure_time
            master_dark.header['exptime'] = exposure_time  # ccdproc requires lowercase

        # Validate master frame
        self._validate_master_frame(master_dark, 'dark')

        logger.info(f"Created master dark from {len(dark_files)} frames")

        # Cache
        if self.master_cache_dir:
            exp_str = f"_{int(exposure_time)}s" if exposure_time else "_all"
            cache_file = self.master_cache_dir / f'master_dark{exp_str}.fits'
            master_dark.write(cache_file, overwrite=True)
            logger.debug(f"Cached master dark: {cache_file}")

        # Store in dictionary by exposure time
        if exposure_time:
            self.calibrations.master_darks[exposure_time] = master_dark
        else:
            # Extract actual exposure time from header
            actual_exp = master_dark.header.get('EXPTIME', 0.0)
            if actual_exp > 0:
                self.calibrations.master_darks[actual_exp] = master_dark
            else:
                logger.warning("No EXPTIME in master dark header, using 0.0s as key")
                self.calibrations.master_darks[0.0] = master_dark

        return master_dark

    def create_master_dark_library(
        self,
        sigma_clip: bool = True,
        clip_low: float = 3.0,
        clip_high: float = 3.0,
        method: str = 'average',
        subtract_bias: bool = True
    ) -> Dict[float, CCDData]:
        """Create master darks for all unique exposure times found.

        This method auto-detects all unique exposure times in dark frames
        and creates a master dark for each one. This is the recommended
        approach for building a complete calibration library.

        Args:
            sigma_clip: Apply sigma clipping to reject outliers
            clip_low: Lower sigma threshold for clipping
            clip_high: Upper sigma threshold for clipping
            method: Combination method ('average' or 'median')
            subtract_bias: Subtract master bias from darks before combining

        Returns:
            Dictionary mapping exposure time (seconds) to master dark CCDData

        Raises:
            ValueError: If no dark frames found in calibration directory

        Notes
        -----
        This method calls create_master_dark() for each unique exposure time.
        If master bias is available and subtract_bias=True, it will be
        subtracted from darks before combining. This is the recommended
        pattern for scaling darks with ccdproc.subtract_dark(scale=True).

        See Also
        --------
        create_master_dark : Create master dark for single exposure time
        CalibrationFrames.get_dark_for_exposure : Get best matching dark

        Examples
        --------
        >>> calib_mgr.create_master_bias()  # Create bias first
        >>> dark_lib = calib_mgr.create_master_dark_library()
        >>> print(f"Created {len(dark_lib)} master darks")
        >>> print(f"Exposure times: {sorted(dark_lib.keys())}")
        """
        # Find all unique exposure times in dark frames
        exposure_times = set()
        for row in self.ic.summary:
            # Convert astropy Table row to dict for _get_imagetyp
            row_dict = {col: row[col] for col in self.ic.summary.colnames}
            imagetyp = self._get_imagetyp(row_dict)
            if imagetyp == 'DARK':
                exp = row_dict.get('exptime', None)
                if exp is not None and exp > 0:
                    exposure_times.add(float(exp))

        if not exposure_times:
            raise ValueError("No dark frames found in calibration directory")

        logger.info(f"Found {len(exposure_times)} unique dark exposure times: "
                   f"{sorted(exposure_times)} seconds")

        # Create master dark for each exposure time
        created_count = 0
        for exp_time in sorted(exposure_times):
            try:
                self.create_master_dark(
                    exposure_time=exp_time,
                    sigma_clip=sigma_clip,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    method=method,
                    subtract_bias=subtract_bias
                )
                created_count += 1
                logger.info(f"✓ Created master dark for {exp_time}s "
                           f"({created_count}/{len(exposure_times)})")
            except Exception as e:
                logger.warning(f"✗ Could not create master dark for {exp_time}s: {e}")

        if created_count == 0:
            raise ValueError("Failed to create any master darks")

        logger.info(f"Master dark library complete: {created_count}/{len(exposure_times)} "
                   f"exposure times")
        return self.calibrations.master_darks

    def create_bad_pixel_mask(
        self,
        sigma_threshold: float = 5.0,
        use_dark: bool = True,
        update_flats: bool = False
    ) -> np.ndarray:
        """Create bad pixel mask from master dark or bias.

        Identifies hot pixels (pixels significantly above median) as bad.
        Hot pixels are defined as pixels >sigma_threshold standard deviations
        above the median.

        Args:
            sigma_threshold: Number of standard deviations for bad pixel threshold
            use_dark: Use master dark (True) or bias (False) for detection
            update_flats: Apply mask to master flats (default: False)

        Returns:
            Boolean array where True = bad pixel

        Raises:
            ValueError: If no master dark or bias available

        Notes
        -----
        Hot pixels are typically identified in dark frames because dark current
        accumulates in hot pixels. The longer the exposure, the more obvious they
        become. Using the longest exposure dark frame is recommended.

        The mask can be applied to science frames by setting frame.mask = mask.
        ccdproc operations will then automatically exclude bad pixels from
        calculations.

        See Also
        --------
        calibrate : Apply calibrations including bad pixel mask

        Examples
        --------
        >>> # Create bad pixel mask from master dark
        >>> bad_pixels = calib_mgr.create_bad_pixel_mask(sigma_threshold=5.0)
        >>> print(f"Found {bad_pixels.sum()} bad pixels")
        >>>
        >>> # Apply to science frame
        >>> science_frame.mask = bad_pixels
        """
        if use_dark:
            # Use longest exposure dark (hot pixels more obvious)
            if not self.calibrations.master_darks:
                raise ValueError("Need master dark to create bad pixel mask. "
                               "Create darks first or set use_dark=False")

            # Get longest exposure dark
            max_exp = max(self.calibrations.master_darks.keys())
            dark_data = self.calibrations.master_darks[max_exp].data
            logger.info(f"Using {max_exp}s dark for bad pixel detection")
        else:
            if self.calibrations.master_bias is None:
                raise ValueError("Need master bias to create bad pixel mask. "
                               "Create bias first or set use_dark=True")
            dark_data = self.calibrations.master_bias.data

        # Identify hot pixels
        median = np.ma.median(dark_data)
        std = np.ma.std(dark_data)

        # Pixels > median + N*sigma are bad
        bad_pixel_mask = dark_data > (median + sigma_threshold * std)

        n_bad = bad_pixel_mask.sum()
        pct_bad = 100.0 * n_bad / bad_pixel_mask.size
        logger.info(f"Identified {n_bad} bad pixels ({pct_bad:.3f}%)")

        # Optionally update flats to mask bad pixels
        if update_flats and self.calibrations.master_flats:
            for filter_name, flat in self.calibrations.master_flats.items():
                if flat.mask is None:
                    flat.mask = bad_pixel_mask
                else:
                    flat.mask = flat.mask | bad_pixel_mask
                logger.debug(f"Updated {filter_name} flat with bad pixel mask")

        self.calibrations.bad_pixel_mask = bad_pixel_mask
        return bad_pixel_mask

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
        # Find flat frames for this filter using flexible matching
        flat_files = []
        for row in self.ic.summary:
            row_dict = {col: row[col] for col in self.ic.summary.colnames}
            imagetyp = self._get_imagetyp(row_dict)
            if imagetyp == 'FLAT':
                filter_val = row_dict.get('filter', None)
                if filter_val and self._match_filter_name(filter_val, filter_name):
                    flat_files.append(self.calibration_dir / row_dict['file'])

        if len(flat_files) == 0:
            raise ValueError(f"No flat frames found for filter: {filter_name}")

        logger.info(f"Found {len(flat_files)} flat frames for filter {filter_name}")

        # Load, preprocess, and calibrate flats
        flat_list = []
        for flat_file in flat_files:
            ccd = self._preprocess_frame(flat_file)

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
        combiner.mem_limit = self.mem_limit  # Memory-efficient combining

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
        master_flat.header['FRAMTYPE'] = 'MASTER_FLAT'

        # Validate master frame
        self._validate_master_frame(master_flat, 'flat')

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

        # Create master darks for all exposure times (recommended)
        if dark:
            try:
                self.create_master_dark_library()
            except ValueError as e:
                logger.warning(f"Could not create master dark library: {e}")

        # Create master flats
        if flats is None:
            # Auto-detect filters from flat frames using flexible IMAGETYP matching
            flats = set()
            for row in self.ic.summary:
                row_dict = {col: row[col] for col in self.ic.summary.colnames}
                imagetyp = self._get_imagetyp(row_dict)
                if imagetyp == 'FLAT':
                    filter_val = row_dict.get('filter', None)
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

        # Apply dark correction (use best matching dark for exposure time)
        exp_time = calibrated.header.get('EXPTIME', None)
        if exp_time is not None:
            # Get dark frame closest to science exposure time
            master_dark = self.calibrations.get_dark_for_exposure(
                exp_time,
                tolerance=10.0  # Accept darks within ±10 seconds
            )

            if master_dark:
                dark_exp = master_dark.header.get('EXPTIME', 'unknown')
                calibrated = ccdproc.subtract_dark(
                    calibrated,
                    master_dark,
                    exposure_time='exptime',
                    exposure_unit=u.second,
                    scale=True  # Scale dark to match science exposure time
                )

                if dark_exp == exp_time:
                    logger.debug(f"Applied dark correction (exact match: {exp_time}s)")
                else:
                    logger.debug(f"Applied dark correction (using {dark_exp}s dark "
                                f"scaled for {exp_time}s science)")
            else:
                logger.warning(f"No dark frame available for exposure time {exp_time}s "
                              f"(closest match >10s away)")
        elif self.calibrations.master_darks:
            logger.warning("No EXPTIME in science frame header, cannot match dark frame")

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
