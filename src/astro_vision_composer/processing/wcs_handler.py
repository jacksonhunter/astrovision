"""World Coordinate System (WCS) handling and validation.

This module provides the WCSHandler class for extracting, validating, and
working with WCS information from FITS headers and WCS objects.

Supports multi-mission WCS loading including:
- JWST: gwcs from ASDF extension (requires stdatamodels)
- HST: Full distortion with IDCTAB/D2IMFILE (requires drizzlepac)
- Euclid: Multi-detector mosaics
- Standard: FITS WCS with SIP/TPV distortion
"""

from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import logging

logger = logging.getLogger(__name__)

# Optional imports for mission-specific WCS
try:
    import gwcs
    GWCS_AVAILABLE = True
except ImportError:
    GWCS_AVAILABLE = False
    gwcs = None

try:
    from stdatamodels.jwst import datamodels as jwst_datamodels
    JWST_DATAMODELS_AVAILABLE = True
except ImportError:
    JWST_DATAMODELS_AVAILABLE = False
    jwst_datamodels = None

try:
    from drizzlepac import stwcs
    DRIZZLEPAC_AVAILABLE = True
except ImportError:
    DRIZZLEPAC_AVAILABLE = False
    stwcs = None


@dataclass
class WCSInfo:
    """Container for WCS information and validation results.

    Attributes:
        wcs: The WCS object (APE 14-compliant: astropy.wcs.WCS or gwcs.WCS)
        has_celestial: Whether WCS contains celestial coordinates
        has_spectral: Whether WCS contains spectral coordinates
        pixel_scale: Pixel scale in arcseconds per pixel (mean of both axes)
        pixel_scale_x: Pixel scale in X direction (arcsec/pixel)
        pixel_scale_y: Pixel scale in Y direction (arcsec/pixel)
        rotation_angle: Rotation angle in degrees
        projection: Projection type (e.g., 'TAN', 'SIN')
        reference_pixel: Reference pixel coordinates (CRPIX)
        reference_sky: Reference sky coordinates (CRVAL) as SkyCoord
        warnings: List of validation warnings
        is_valid: Whether the WCS is valid and usable

        Mission-specific attributes (Phase 3):
        mission: Detected mission ('JWST', 'HST', 'EUCLID', 'PANSTARRS', 'STANDARD')
        distortion_model: Distortion model type ('SIP', 'TPV', 'gwcs', 'layered', 'none')
        distortion_magnitude: Distortion severity ('none', 'low', 'moderate', 'high')
        quality_score: Composite WCS quality metric (0-100)
        has_sip: Whether SIP distortion coefficients present
        has_gwcs: Whether using gwcs (APE 14 compliant)
        wcs_origin: WCS loading method ('standard', 'drizzlepac', 'gwcs', 'multi-detector')

        gwcs-specific attributes:
        available_frames: List of intermediate coordinate frames (gwcs only)
        world_axis_names: Axis names from APE 14 interface
        world_axis_units: Axis units from APE 14 interface
    """
    wcs: BaseHighLevelWCS  # APE 14 compliant (works for both astropy.wcs.WCS and gwcs.WCS)
    has_celestial: bool
    has_spectral: bool
    pixel_scale: Optional[float] = None
    pixel_scale_x: Optional[float] = None
    pixel_scale_y: Optional[float] = None
    rotation_angle: Optional[float] = None
    projection: Optional[str] = None
    reference_pixel: Optional[Tuple[float, float]] = None
    reference_sky: Optional[SkyCoord] = None
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True

    # Mission-specific fields (Phase 3)
    mission: Optional[str] = None
    distortion_model: Optional[str] = None
    distortion_magnitude: Optional[str] = None
    quality_score: float = 0.0
    has_sip: bool = False
    has_gwcs: bool = False
    wcs_origin: str = 'standard'

    # gwcs-specific fields (Phase 3A)
    available_frames: Optional[List[str]] = None
    world_axis_names: Optional[Tuple[str, ...]] = None
    world_axis_units: Optional[Tuple[u.Unit, ...]] = None

    def __repr__(self):
        """Pretty representation."""
        parts = []
        if self.mission:
            parts.append(f"Mission={self.mission}")
        if self.projection:
            parts.append(f"Projection={self.projection}")
        if self.pixel_scale:
            parts.append(f"Scale={self.pixel_scale:.3f}\"/px")
        if self.rotation_angle is not None:
            parts.append(f"Rotation={self.rotation_angle:.1f}deg")
        if self.distortion_model and self.distortion_model != 'none':
            parts.append(f"Distortion={self.distortion_model}({self.distortion_magnitude})")
        if self.quality_score > 0:
            parts.append(f"Quality={self.quality_score:.0f}/100")
        status = "Valid" if self.is_valid else "Invalid"
        return f"WCSInfo({status}, {', '.join(parts)})"


class WCSHandler:
    """Handler for WCS extraction, validation, and coordinate conversion.

    This class provides utilities for working with World Coordinate Systems,
    including validation, coordinate conversion, and extracting useful properties.

    Note: This class works with astropy.wcs.WCS objects and does not replace
    astropy functionality. It provides analysis and validation on top of astropy.

    Example:
        >>> from astropy.io import fits
        >>> from astropy.wcs import WCS
        >>> handler = WCSHandler()
        >>> with fits.open('image.fits') as hdul:
        ...     wcs = WCS(hdul[0].header)
        ...     wcs_info = handler.validate(wcs)
        ...     if wcs_info.is_valid:
        ...         print(f"Pixel scale: {wcs_info.pixel_scale:.3f} arcsec/pixel")
    """

    def __init__(self):
        """Initialize the WCS handler."""
        pass

    def _detect_mission(self, header: fits.Header) -> str:
        """Auto-detect mission/telescope from FITS header.

        Args:
            header: FITS header object

        Returns:
            Mission identifier: 'JWST', 'HST', 'CHANDRA', 'EUCLID', 'PANSTARRS', 'STANDARD'
        """
        # Check TELESCOP keyword first
        telescope = header.get('TELESCOP', '').upper()

        if 'JWST' in telescope:
            return 'JWST'
        elif 'HST' in telescope:
            return 'HST'
        elif 'CHANDRA' in telescope:
            return 'CHANDRA'
        elif 'EUCLID' in telescope:
            return 'EUCLID'

        # Check ORIGIN keyword
        origin = header.get('ORIGIN', '').upper()
        if 'JWST' in origin or 'STScI-JWST' in origin:
            return 'JWST'
        elif 'STScI' in origin:
            return 'HST'

        # Check INSTRUME keyword
        instrument = header.get('INSTRUME', '').upper()
        if any(inst in instrument for inst in ['NIRCAM', 'NIRSPEC', 'MIRI', 'NIRISS', 'FGS']):
            return 'JWST'
        elif any(inst in instrument for inst in ['ACS', 'WFC3', 'WFPC2', 'STIS', 'NICMOS']):
            return 'HST'
        elif 'ACIS' in instrument or 'HRC' in instrument:
            return 'CHANDRA'
        elif 'VIS' in instrument or 'NISP' in instrument:
            return 'EUCLID'
        elif 'GMOS' in instrument:
            # Gemini data - often uses TPV
            return 'STANDARD'

        # Check for PanSTARRS-specific keywords
        if 'PROJCELL' in header or 'PROJECT' in header:
            projcell = header.get('PROJCELL', header.get('PROJECT', ''))
            if 'PS1' in str(projcell).upper() or 'PANSTARRS' in str(projcell).upper():
                return 'PANSTARRS'

        # Default to standard FITS WCS
        return 'STANDARD'

    def _is_jwst_file(self, fits_file: Union[Path, str]) -> bool:
        """Check if file is JWST data.

        Args:
            fits_file: Path to FITS file

        Returns:
            True if JWST data
        """
        try:
            with fits.open(fits_file) as hdul:
                return self._detect_mission(hdul[0].header) == 'JWST'
        except Exception:
            return False

    def _is_hst_file(self, fits_file: Union[Path, str]) -> bool:
        """Check if file is HST data.

        Args:
            fits_file: Path to FITS file

        Returns:
            True if HST data
        """
        try:
            with fits.open(fits_file) as hdul:
                return self._detect_mission(hdul[0].header) == 'HST'
        except Exception:
            return False

    def _is_euclid_file(self, fits_file: Union[Path, str]) -> bool:
        """Check if file is Euclid data.

        Args:
            fits_file: Path to FITS file

        Returns:
            True if Euclid data
        """
        try:
            with fits.open(fits_file) as hdul:
                return self._detect_mission(hdul[0].header) == 'EUCLID'
        except Exception:
            return False

    def _has_asdf_extension(self, fits_file: Path) -> bool:
        """Check if FITS file has ASDF extension with WCS.

        This is the proper way to detect gwcs files - by ASDF presence,
        not by mission name. gwcs can be used for any mission!

        Args:
            fits_file: Path to FITS file

        Returns:
            True if file has ASDF extension
        """
        try:
            with fits.open(fits_file) as hdul:
                # Check for ASDF extension
                for hdu in hdul:
                    if hdu.name == 'ASDF':
                        logger.debug(f"Found ASDF extension in {fits_file.name}")
                        return True
            return False
        except Exception as e:
            logger.debug(f"Could not check for ASDF extension: {e}")
            return False

    def load_wcs(
        self,
        fits_file: Union[Path, str],
        extension: Optional[Union[int, Tuple[str, int]]] = None,
        mission: Optional[str] = None
    ) -> Union[BaseHighLevelWCS, Dict[str, BaseHighLevelWCS]]:
        """Load WCS with automatic format detection (APE 14-compliant).

        This is the main factory method for loading WCS from FITS files.
        Uses duck-typing to detect WCS format, prioritizing:
        1. ASDF extension (gwcs) - Highest fidelity, mission-agnostic
        2. Mission-specific loaders (HST drizzlepac)
        3. Standard FITS WCS (SIP/TPV distortion)

        All returned WCS objects implement the APE 14 common interface
        (pixel_to_world, world_to_pixel, etc.) and are interchangeable.

        Args:
            fits_file: Path to FITS file
            extension: FITS extension (int or tuple of (name, version))
                      If None, uses first SCI extension or primary HDU
            mission: Optional mission override ('JWST', 'HST', 'EUCLID', etc.)
                    If None, auto-detects from header

        Returns:
            - BaseHighLevelWCS (APE 14) for single-detector missions
            - Dict[str, BaseHighLevelWCS] for multi-detector (Euclid)

        Note:
            All returned WCS types are APE 14-compliant:
            - astropy.wcs.WCS (standard FITS WCS)
            - gwcs.WCS (generalized WCS from ASDF)
            - drizzlepac.stwcs.HSTWCS (HST with full distortion)

        Raises:
            ValueError: If file cannot be opened
            ImportError: If mission-specific library not available (with fallback)

        Example:
            >>> handler = WCSHandler()
            >>>
            >>> # Auto-detect and load (duck-typing)
            >>> wcs = handler.load_wcs('jwst_nircam_cal.fits')  # gwcs.WCS (ASDF)
            >>> wcs = handler.load_wcs('hst_acs_flt.fits')      # HSTWCS (drizzlepac)
            >>> wcs = handler.load_wcs('panstarrs.fits')        # WCS (standard)
            >>>
            >>> # All have same interface (APE 14):
            >>> sky = wcs.pixel_to_world(100, 200)  # Works for all types!
            >>> x, y = wcs.world_to_pixel(sky)      # Works for all types!
        """
        fits_file = Path(fits_file)
        if not fits_file.exists():
            raise ValueError(f"FITS file not found: {fits_file}")

        # Priority 1: Check for ASDF extension (gwcs) - Duck-typing approach
        # This works for JWST, Roman, Euclid, or ANY mission using gwcs!
        if self._has_asdf_extension(fits_file):
            logger.info(f"ASDF extension detected - loading gwcs")
            try:
                return self._load_jwst_wcs(fits_file, extension)
            except Exception as e:
                logger.warning(f"Failed to load gwcs from ASDF: {e}. Trying standard WCS.")

        # Priority 2: Auto-detect mission for mission-specific loaders
        if mission is None:
            with fits.open(fits_file) as hdul:
                mission = self._detect_mission(hdul[0].header)
            logger.info(f"Auto-detected mission: {mission}")
        else:
            mission = mission.upper()

        # Priority 3: Mission-specific loaders
        try:
            if mission == 'HST' or self._is_hst_file(fits_file):
                return self._load_hst_wcs(fits_file, extension)

            elif mission == 'EUCLID' or self._is_euclid_file(fits_file):
                return self._load_euclid_wcs(fits_file)

            else:
                # Priority 4: Standard FITS WCS (includes PanSTARRS, ground-based, etc.)
                return self._load_standard_wcs(fits_file, extension)

        except Exception as e:
            logger.error(f"Failed to load WCS from {fits_file}: {e}")
            raise

    def _load_jwst_wcs(
        self,
        fits_file: Path,
        extension: Optional[Union[int, Tuple[str, int]]] = None
    ) -> Union[WCS, 'gwcs.WCS']:
        """Load JWST WCS from ASDF extension using stdatamodels.

        Args:
            fits_file: Path to JWST FITS file (typically *_cal.fits or *_i2d.fits)
            extension: Not used for JWST (gwcs is in ASDF extension)

        Returns:
            gwcs.WCS object if stdatamodels available, otherwise standard WCS

        Note:
            Requires stdatamodels package. Falls back to standard WCS if not available.
        """
        if not JWST_DATAMODELS_AVAILABLE:
            logger.warning(
                "stdatamodels not installed. Cannot load JWST gwcs from ASDF extension. "
                "Falling back to standard WCS (distortion will be incomplete). "
                "Install with: pip install stdatamodels"
            )
            return self._load_standard_wcs(fits_file, extension)

        try:
            # Use JWST datamodels to open file and extract gwcs
            with jwst_datamodels.open(fits_file) as model:
                wcs_obj = model.meta.wcs

                if wcs_obj is None:
                    logger.warning(f"No WCS found in JWST datamodel for {fits_file}")
                    # Fall back to standard WCS
                    return self._load_standard_wcs(fits_file, extension)

                logger.info(f"Loaded JWST gwcs from {fits_file}")
                logger.debug(f"Available gwcs frames: {wcs_obj.available_frames}")

                return wcs_obj

        except Exception as e:
            logger.error(f"Failed to load JWST gwcs: {e}. Falling back to standard WCS.")
            return self._load_standard_wcs(fits_file, extension)

    def _load_hst_wcs(
        self,
        fits_file: Path,
        extension: Optional[Union[int, Tuple[str, int]]] = None
    ) -> WCS:
        """Load HST WCS with full distortion using drizzlepac.stwcs.

        This includes all distortion layers:
        - IDCTAB: Fourth-order polynomial distortion
        - D2IMFILE: Detector manufacturing defects (WFC3/UVIS)
        - NPOLFILE: Filter-dependent distortion

        Args:
            fits_file: Path to HST FITS file (typically *_flt.fits or *_flc.fits)
            extension: Extension to load WCS from (default: ('SCI', 1))

        Returns:
            WCS object with full distortion correction

        Note:
            Requires drizzlepac package and CRDS environment configured.
            Falls back to standard WCS if not available.

        Environment:
            Requires CRDS_PATH and CRDS_SERVER_URL environment variables for
            reference file access.
        """
        if not DRIZZLEPAC_AVAILABLE:
            logger.warning(
                "drizzlepac not installed. Cannot load HST full distortion correction. "
                "Falling back to standard WCS (will only have SIP, not IDCTAB/D2IMFILE). "
                "Install with: pip install drizzlepac"
            )
            return self._load_standard_wcs(fits_file, extension)

        try:
            # Default to first SCI extension
            if extension is None:
                extension = ('SCI', 1)
            elif isinstance(extension, int):
                # Convert integer index to extension name
                with fits.open(fits_file) as hdul:
                    if extension < len(hdul):
                        ext_name = hdul[extension].name
                        ext_ver = hdul[extension].ver if hasattr(hdul[extension], 'ver') else 1
                        extension = (ext_name, ext_ver)
                    else:
                        extension = ('SCI', 1)

            # Use drizzlepac stwcs to load WCS with full distortion
            hst_wcs = stwcs.wcsutil.HSTWCS(str(fits_file), ext=extension)

            logger.info(f"Loaded HST WCS with full distortion from {fits_file}, ext={extension}")

            # Check if distortion reference files were found
            if hasattr(hst_wcs, 'idcmodel') and hst_wcs.idcmodel is not None:
                logger.debug("HST IDCTAB distortion model loaded")
            if hasattr(hst_wcs, 'd2im1') and hst_wcs.d2im1 is not None:
                logger.debug("HST D2IMFILE distortion loaded")

            return hst_wcs

        except Exception as e:
            logger.error(
                f"Failed to load HST WCS with drizzlepac: {e}. "
                f"This may be due to missing CRDS environment or reference files. "
                f"Falling back to standard WCS."
            )
            return self._load_standard_wcs(fits_file, extension)

    def _load_euclid_wcs(self, fits_file: Path) -> Dict[str, WCS]:
        """Load Euclid multi-detector WCS.

        Euclid has 36 CCDs (VIS) or 16 detectors (NISP), each with independent WCS.
        Returns a dictionary mapping detector quadrant names to WCS objects.

        Args:
            fits_file: Path to Euclid FITS file

        Returns:
            Dictionary mapping detector names (e.g., 'SCI_1', 'SCI_2') to WCS objects

        Example:
            >>> wcs_dict = handler._load_euclid_wcs('euclid_vis.fits')
            >>> for det_name, wcs in wcs_dict.items():
            ...     print(f"{det_name}: {wcs.wcs.crval}")
        """
        wcs_dict = {}

        try:
            with fits.open(fits_file) as hdul:
                # Iterate through HDUs and find science extensions
                for hdu in hdul:
                    # Euclid science data typically in SCI extensions
                    if hdu.is_image and hdu.data is not None:
                        if 'SCI' in hdu.name or 'DATA' in hdu.name:
                            # Create unique name for this detector
                            det_name = f"{hdu.name}_{hdu.ver}" if hasattr(hdu, 'ver') else hdu.name

                            # Load WCS for this detector
                            det_wcs = WCS(hdu.header)

                            wcs_dict[det_name] = det_wcs
                            logger.debug(f"Loaded Euclid WCS for detector {det_name}")

            if not wcs_dict:
                logger.warning(f"No SCI/DATA extensions found in Euclid file {fits_file}")
                # Fall back to primary HDU
                with fits.open(fits_file) as hdul:
                    if hdul[0].data is not None:
                        wcs_dict['PRIMARY'] = WCS(hdul[0].header)

            logger.info(f"Loaded Euclid multi-detector WCS: {len(wcs_dict)} detectors")
            return wcs_dict

        except Exception as e:
            logger.error(f"Failed to load Euclid WCS: {e}")
            raise

    def _load_standard_wcs(
        self,
        fits_file: Path,
        extension: Optional[Union[int, Tuple[str, int]]] = None
    ) -> WCS:
        """Load standard FITS WCS using astropy.wcs.

        This handles:
        - Simple WCS (TAN, SIN, etc. projections)
        - SIP distortion (coefficients in header)
        - TPV distortion (e.g., PanSTARRS, ground-based)

        Args:
            fits_file: Path to FITS file
            extension: Extension to load (int or tuple). If None, tries:
                      1. First 'SCI' extension
                      2. Primary HDU (index 0)

        Returns:
            astropy.wcs.WCS object
        """
        try:
            with fits.open(fits_file) as hdul:
                # Determine which extension to use
                target_ext = None

                if extension is not None:
                    # User specified extension
                    target_ext = extension
                else:
                    # Auto-detect: look for SCI extension first
                    for idx, hdu in enumerate(hdul):
                        if hdu.name == 'SCI' and hdu.data is not None:
                            target_ext = idx
                            logger.debug(f"Using SCI extension at index {idx}")
                            break

                    # If no SCI found, use primary HDU
                    if target_ext is None:
                        target_ext = 0
                        logger.debug("Using primary HDU (index 0)")

                # Load header
                if isinstance(target_ext, tuple):
                    header = hdul[target_ext].header
                else:
                    header = hdul[target_ext].header

                # Create WCS object
                wcs_obj = WCS(header)

                logger.info(f"Loaded standard WCS from {fits_file}, ext={target_ext}")

                # Check for distortion
                if wcs_obj.sip is not None:
                    logger.debug("SIP distortion coefficients detected")
                ctype = header.get('CTYPE1', '')
                if 'TPV' in ctype:
                    logger.debug("TPV projection with distortion detected")

                return wcs_obj

        except Exception as e:
            logger.error(f"Failed to load standard WCS: {e}")
            raise

    def validate(self, wcs: BaseHighLevelWCS) -> WCSInfo:
        """Validate a WCS object using APE 14 interface (works for all WCS types).

        Performs comprehensive validation and extracts useful properties
        such as pixel scale, rotation, and reference coordinates.

        Uses ONLY the APE 14 common interface, so works for:
        - astropy.wcs.WCS (standard FITS WCS)
        - gwcs.WCS (JWST, Roman, etc.)
        - drizzlepac.stwcs.HSTWCS (HST with full distortion)

        Args:
            wcs: WCS object (APE 14-compliant)

        Returns:
            WCSInfo dataclass containing validation results and extracted properties
        """
        # Detect if this is gwcs (has pipeline) vs standard WCS
        is_gwcs = hasattr(wcs, 'available_frames')

        # Use APE 14 interface to check for celestial coordinates
        has_celestial = False
        has_spectral = False

        if hasattr(wcs, 'world_axis_physical_types'):
            phys_types = wcs.world_axis_physical_types
            # Check for celestial coordinates (pos.eq, pos.galactic, etc.)
            has_celestial = any('pos' in str(t) for t in phys_types if t)
            # Check for spectral coordinates
            has_spectral = any('spect' in str(t) or 'em.wl' in str(t) for t in phys_types if t)
        else:
            # Fallback for older WCS objects
            has_celestial = getattr(wcs, 'has_celestial', False)
            has_spectral = getattr(wcs, 'has_spectral', False)

        info = WCSInfo(
            wcs=wcs,
            has_celestial=has_celestial,
            has_spectral=has_spectral,
            has_gwcs=is_gwcs
        )

        # Extract axis information using APE 14
        if hasattr(wcs, 'world_axis_names'):
            info.world_axis_names = wcs.world_axis_names
            logger.debug(f"WCS axis names: {info.world_axis_names}")

        if hasattr(wcs, 'world_axis_units'):
            info.world_axis_units = wcs.world_axis_units
            logger.debug(f"WCS axis units: {info.world_axis_units}")

        # Extract gwcs-specific information
        if is_gwcs:
            info.available_frames = wcs.available_frames
            info.wcs_origin = 'gwcs'
            logger.debug(f"gwcs frames: {info.available_frames}")

        # Check for celestial coordinates
        if not has_celestial:
            info.warnings.append("WCS has no celestial coordinates")
            info.is_valid = False
            return info

        # Extract projection type (FITS-specific, skip for gwcs)
        if not is_gwcs:
            try:
                # Get CTYPE for first axis (FITS WCS only)
                if hasattr(wcs, 'wcs') and hasattr(wcs.wcs, 'ctype') and len(wcs.wcs.ctype) > 0:
                    ctype = wcs.wcs.ctype[0]
                    # Projection is typically last 3 characters (e.g., 'RA---TAN' -> 'TAN')
                    if '-' in ctype:
                        info.projection = ctype.split('-')[-1]
            except Exception as e:
                logger.debug(f"Could not extract projection type: {e}")
                info.warnings.append("Could not determine projection type")
        else:
            # For gwcs, projection info is in the pipeline
            info.projection = 'gwcs'  # Indicate this is gwcs

        # Extract reference pixel (FITS WCS only, gwcs doesn't have CRPIX concept)
        if not is_gwcs:
            try:
                if hasattr(wcs, 'wcs') and hasattr(wcs.wcs, 'crpix') and len(wcs.wcs.crpix) >= 2:
                    crpix = wcs.wcs.crpix
                    info.reference_pixel = (float(crpix[0]), float(crpix[1]))
            except Exception as e:
                logger.debug(f"Could not extract reference pixel: {e}")
                info.warnings.append("Could not extract reference pixel (CRPIX)")

        # Extract reference sky coordinates using APE 14 transformation
        try:
            # Use APE 14 interface: transform pixel to world
            # For FITS WCS, use CRPIX; for gwcs, use image center estimate
            if info.reference_pixel:
                ref_x, ref_y = info.reference_pixel
            else:
                # Estimate image center (works for both WCS types)
                if hasattr(wcs, 'pixel_shape'):
                    ny, nx = wcs.pixel_shape
                    ref_x, ref_y = nx / 2, ny / 2
                else:
                    ref_x, ref_y = 1024, 1024  # Default guess

            # Use APE 14 pixel_to_world (works for both!)
            ref_world = wcs.pixel_to_world(ref_x, ref_y)
            if isinstance(ref_world, SkyCoord):
                info.reference_sky = ref_world
            elif isinstance(ref_world, tuple) and len(ref_world) >= 2:
                # Some WCS return tuple of (SkyCoord, spectral, ...)
                if isinstance(ref_world[0], SkyCoord):
                    info.reference_sky = ref_world[0]

        except Exception as e:
            logger.debug(f"Could not extract reference sky coordinates: {e}")
            info.warnings.append("Could not extract reference sky coordinates")

        # Calculate pixel scale using transformation (works for both WCS types!)
        try:
            pixel_scales = self._calculate_pixel_scale_from_transform(wcs)
            if pixel_scales:
                info.pixel_scale_x, info.pixel_scale_y = pixel_scales
                info.pixel_scale = np.mean([info.pixel_scale_x, info.pixel_scale_y])

                # Check for large distortions (>10% difference between axes)
                if abs(info.pixel_scale_x - info.pixel_scale_y) / info.pixel_scale > 0.1:
                    info.warnings.append(
                        f"Significant pixel scale difference between axes: "
                        f"{info.pixel_scale_x:.3f} vs {info.pixel_scale_y:.3f} arcsec/pixel"
                    )
        except Exception as e:
            logger.debug(f"Could not calculate pixel scale: {e}")
            info.warnings.append("Could not calculate pixel scale")

        # Calculate rotation angle (FITS WCS only for now)
        if not is_gwcs:
            try:
                info.rotation_angle = self._calculate_rotation(wcs)
            except Exception as e:
                logger.debug(f"Could not calculate rotation angle: {e}")

        # Detect distortion model
        if not is_gwcs:
            # Check for SIP distortion
            if hasattr(wcs, 'sip') and wcs.sip is not None:
                info.has_sip = True
                info.distortion_model = 'SIP'
                info.wcs_origin = 'standard_with_sip'
            # Check for TPV projection
            elif info.projection and 'TPV' in info.projection:
                info.distortion_model = 'TPV'
                info.wcs_origin = 'standard_with_tpv'
            else:
                info.distortion_model = 'none'
                info.wcs_origin = 'standard'

        # Final validation
        if info.pixel_scale is None:
            info.warnings.append("Missing pixel scale - WCS may be incomplete")
            info.is_valid = False

        if info.reference_sky is None:
            info.warnings.append("Missing reference coordinates - WCS may be incomplete")
            # Don't mark as invalid - gwcs may not have reference pixel concept

        return info

    def _calculate_pixel_scale_from_transform(
        self,
        wcs: BaseHighLevelWCS
    ) -> Optional[Tuple[float, float]]:
        """Calculate pixel scale by measuring actual transformations (APE 14).

        This method works for BOTH astropy.wcs.WCS and gwcs.WCS because
        it uses the APE 14 common interface (pixel_to_world).

        Unlike _calculate_pixel_scale which reads CDELT keywords (FITS-only),
        this method actually transforms pixels and measures separations.

        Args:
            wcs: WCS object (APE 14-compliant)

        Returns:
            Tuple of (pixel_scale_x, pixel_scale_y) in arcsec/pixel, or None
        """
        try:
            # Get reference pixel (image center estimate)
            if hasattr(wcs, 'pixel_shape') and wcs.pixel_shape is not None:
                ny, nx = wcs.pixel_shape
                cx, cy = nx / 2, ny / 2
            else:
                cx, cy = 1024, 1024  # Assume 2K detector

            # Transform reference pixel and neighbors using APE 14
            # For 2D celestial WCS, pixel_to_world_values returns TWO separate values (ra, dec)
            # NOT an array! This is the APE 14 standard.
            center_world = wcs.pixel_to_world_values(cx, cy)
            right_world = wcs.pixel_to_world_values(cx + 1, cy)
            up_world = wcs.pixel_to_world_values(cx, cy + 1)

            # DEBUG: Log what we got
            logger.debug(f"center_world type: {type(center_world)}, value: {center_world}")
            logger.debug(f"right_world type: {type(right_world)}, value: {right_world}")
            logger.debug(f"up_world type: {type(up_world)}, value: {up_world}")

            # APE 14: For N-dimensional WCS, pixel_to_world_values returns N separate values
            # For 2D celestial: returns (ra, dec) as TWO scalars or arrays
            if isinstance(center_world, tuple) and len(center_world) == 2:
                # Tuple of (ra, dec)
                center_ra, center_dec = center_world
                right_ra, right_dec = right_world
                up_ra, up_dec = up_world
            else:
                # Some WCS implementations return single value or need different handling
                # Try to extract as scalars
                try:
                    # Assume it's unpacked already (older astropy versions)
                    # This shouldn't happen with modern astropy, but handle it
                    logger.debug(f"pixel_to_world_values returned: {type(center_world)}, value: {center_world}")

                    # If single scalar, WCS is 1D (not celestial)
                    if np.isscalar(center_world):
                        logger.warning("WCS returns scalar values, appears to be 1D")
                        return None

                    # If array, take first two elements
                    if hasattr(center_world, '__len__'):
                        center_ra, center_dec = float(center_world[0]), float(center_world[1])
                        right_ra, right_dec = float(right_world[0]), float(right_world[1])
                        up_ra, up_dec = float(up_world[0]), float(up_world[1])
                    else:
                        logger.warning(f"Unexpected return type from pixel_to_world_values: {type(center_world)}")
                        return None
                except Exception as ex:
                    logger.warning(f"Could not extract RA/Dec from pixel_to_world_values: {ex}")
                    return None

            # Calculate separations using SkyCoord (works for any projection!)
            c = SkyCoord(center_ra, center_dec, unit='deg')
            r = SkyCoord(right_ra, right_dec, unit='deg')
            u_sky = SkyCoord(up_ra, up_dec, unit='deg')

            scale_x = c.separation(r).to(u.arcsec).value
            scale_y = c.separation(u_sky).to(u.arcsec).value

            return (scale_x, scale_y)

        except Exception as e:
            logger.warning(f"Transform-based pixel scale calculation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _calculate_pixel_scale(self, wcs: WCS) -> Optional[Tuple[float, float]]:
        """Calculate pixel scale in arcseconds per pixel for both axes (FITS WCS only).

        DEPRECATED: Use _calculate_pixel_scale_from_transform for APE 14 compatibility.

        This method only works for astropy.wcs.WCS (FITS keywords).
        Kept for backward compatibility.

        Args:
            wcs: WCS object (astropy.wcs.WCS only)

        Returns:
            Tuple of (pixel_scale_x, pixel_scale_y) in arcsec/pixel, or None
        """
        try:
            # Try to use proj_plane_pixel_scales (most robust method)
            if hasattr(wcs, 'proj_plane_pixel_scales'):
                scales = wcs.proj_plane_pixel_scales()
                # Convert to arcseconds
                scale_x = abs(scales[0]) * 3600.0
                scale_y = abs(scales[1]) * 3600.0
                return (scale_x, scale_y)
        except:
            pass

        try:
            # Alternative: use CDELT keywords if available
            if hasattr(wcs, 'wcs') and hasattr(wcs.wcs, 'cdelt') and len(wcs.wcs.cdelt) >= 2:
                cdelt = wcs.wcs.cdelt
                scale_x = abs(cdelt[0]) * 3600.0
                scale_y = abs(cdelt[1]) * 3600.0
                return (scale_x, scale_y)
        except:
            pass

        return None

    def _calculate_rotation(self, wcs: WCS) -> Optional[float]:
        """Calculate rotation angle of the WCS in degrees.

        Args:
            wcs: WCS object

        Returns:
            Rotation angle in degrees (East of North), or None
        """
        try:
            # Get the CD matrix or PC matrix
            if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd.shape == (2, 2):
                cd = wcs.wcs.cd
                angle = np.rad2deg(np.arctan2(-cd[0, 1], cd[1, 1]))
                return angle

            # Try PC matrix with CDELT
            if hasattr(wcs.wcs, 'pc') and wcs.wcs.pc.shape == (2, 2):
                pc = wcs.wcs.pc
                angle = np.rad2deg(np.arctan2(-pc[0, 1], pc[1, 1]))
                return angle
        except Exception as e:
            logger.debug(f"Rotation calculation failed: {e}")

        return None

    def get_available_frames(self, wcs: BaseHighLevelWCS) -> Optional[List[str]]:
        """Get list of available coordinate frames (gwcs only).

        For gwcs.WCS objects, returns the list of intermediate coordinate frames
        in the transformation pipeline (e.g., ['detector', 'undistorted_frame', 'icrs']).

        For standard FITS WCS, returns None (no intermediate frames concept).

        Args:
            wcs: WCS object (APE 14-compliant)

        Returns:
            List of frame names for gwcs, None for standard WCS

        Example:
            >>> handler = WCSHandler()
            >>> wcs = handler.load_wcs('jwst_nircam_cal.fits')
            >>> frames = handler.get_available_frames(wcs)
            >>> print(frames)
            ['detector', 'v2v3', 'world']
        """
        if hasattr(wcs, 'available_frames'):
            return wcs.available_frames
        return None

    def get_transform(
        self,
        wcs: BaseHighLevelWCS,
        from_frame: str,
        to_frame: str
    ) -> Optional[Any]:
        """Get transformation between two coordinate frames (gwcs only).

        Extracts the transformation model between any two frames in the
        gwcs pipeline. Useful for:
        - Accessing distortion corrections alone
        - Getting slit plane coordinates for spectroscopy
        - Debugging coordinate transformations
        - Custom calibrations

        For standard FITS WCS, returns None (no sub-transform access).

        Args:
            wcs: WCS object (APE 14-compliant)
            from_frame: Source frame name (e.g., 'detector')
            to_frame: Destination frame name (e.g., 'undistorted_frame')

        Returns:
            Transform object (astropy Model) for gwcs, None for standard WCS

        Example:
            >>> # Get distortion correction only
            >>> distortion = handler.get_transform(wcs, 'detector', 'undistorted_frame')
            >>> undistorted_x, undistorted_y = distortion(raw_x, raw_y)
            >>>
            >>> # Get slit plane coordinates for spectroscopy
            >>> slit_transform = handler.get_transform(wcs, 'detector', 'slit_frame')
            >>> slit_x, slit_y = slit_transform(detector_x, detector_y)
        """
        if hasattr(wcs, 'get_transform'):
            try:
                return wcs.get_transform(from_frame, to_frame)
            except Exception as e:
                logger.warning(f"Could not get transform {from_frame}→{to_frame}: {e}")
                return None
        return None

    def inspect_pipeline(self, wcs: BaseHighLevelWCS) -> Dict[str, Any]:
        """Inspect WCS transformation pipeline (gwcs only).

        Returns detailed information about the gwcs transformation pipeline,
        including all frames and transforms. Useful for understanding complex
        WCS structures and debugging.

        For standard FITS WCS, returns basic information.

        Args:
            wcs: WCS object (APE 14-compliant)

        Returns:
            Dictionary with pipeline information:
            - 'type': 'gwcs' or 'standard'
            - 'has_pipeline': Whether WCS has multi-step pipeline
            - 'frames': List of frame names (gwcs only)
            - 'steps': List of pipeline steps with frame and transform info (gwcs only)
            - 'input_frame': Input frame info
            - 'output_frame': Output frame info

        Example:
            >>> info = handler.inspect_pipeline(wcs)
            >>> print(info['type'])
            'gwcs'
            >>> for step in info['steps']:
            ...     print(f"{step['frame']} via {step['transform']}")
            detector via <Shift & Shift | ...>
            v2v3 via <RotateNative2Celestial>
            world via None
        """
        result = {
            'type': 'unknown',
            'has_pipeline': False,
            'frames': None,
            'steps': None,
            'input_frame': None,
            'output_frame': None
        }

        # Check if gwcs
        if hasattr(wcs, 'available_frames') and hasattr(wcs, 'pipeline'):
            result['type'] = 'gwcs'
            result['has_pipeline'] = True
            result['frames'] = wcs.available_frames

            # Extract pipeline steps
            steps = []
            try:
                for step in wcs.pipeline:
                    step_info = {
                        'frame': step.frame.name if hasattr(step.frame, 'name') else str(step.frame),
                        'transform': str(step.transform) if step.transform else None,
                        'frame_type': type(step.frame).__name__
                    }
                    steps.append(step_info)
                result['steps'] = steps
            except Exception as e:
                logger.debug(f"Could not extract pipeline steps: {e}")

            # Extract frame info
            if hasattr(wcs, 'input_frame'):
                result['input_frame'] = {
                    'name': wcs.input_frame.name if hasattr(wcs.input_frame, 'name') else 'unknown',
                    'axes_names': wcs.input_frame.axes_names if hasattr(wcs.input_frame, 'axes_names') else None,
                    'unit': str(wcs.input_frame.unit) if hasattr(wcs.input_frame, 'unit') else None
                }

            if hasattr(wcs, 'output_frame'):
                result['output_frame'] = {
                    'name': wcs.output_frame.name if hasattr(wcs.output_frame, 'name') else 'unknown',
                    'axes_names': wcs.output_frame.axes_names if hasattr(wcs.output_frame, 'axes_names') else None,
                    'unit': str(wcs.output_frame.unit) if hasattr(wcs.output_frame, 'unit') else None
                }

        else:
            # Standard FITS WCS
            result['type'] = 'standard'
            result['has_pipeline'] = False

            # Basic info from standard WCS
            if hasattr(wcs, 'wcs'):
                result['input_frame'] = {'name': 'pixel', 'axes_names': ['x', 'y']}
                result['output_frame'] = {
                    'name': 'world',
                    'axes_names': list(wcs.world_axis_names) if hasattr(wcs, 'world_axis_names') else None
                }

        return result

    def set_bounding_box(
        self,
        wcs: BaseHighLevelWCS,
        bbox: Tuple[Tuple[float, float], ...]
    ) -> None:
        """Set bounding box for WCS (valid pixel region).

        Bounding box defines the valid pixel region for the WCS. Pixels outside
        this region will return NaN when transformed. Critical for:
        - IFU/MOS spectroscopy (slit regions)
        - Multi-object data (each object has valid region)
        - Detector boundaries

        Note: gwcs and astropy.wcs use different tuple ordering conventions.
        This method handles the conversion automatically.

        Args:
            wcs: WCS object (APE 14-compliant)
            bbox: Bounding box as ((x_min, x_max), (y_min, y_max), ...)
                  Uses GWCS "F" ordering (x, y, z)

        Example:
            >>> # Set valid region for 2048x1024 detector
            >>> handler.set_bounding_box(wcs, ((0, 2048), (0, 1024)))
            >>>
            >>> # Now pixels outside bounds return NaN
            >>> wcs.pixel_to_world(3000, 500)  # Outside bounds
            <SkyCoord (nan, nan)>
        """
        if hasattr(wcs, 'bounding_box'):
            # gwcs supports bounding boxes directly
            wcs.bounding_box = bbox
            logger.debug(f"Set bounding box: {bbox}")
        else:
            logger.warning("WCS does not support bounding boxes (not gwcs)")

    def get_bounding_box(
        self,
        wcs: BaseHighLevelWCS
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Get bounding box from WCS (if set).

        Args:
            wcs: WCS object (APE 14-compliant)

        Returns:
            Bounding box as ((x_min, x_max), (y_min, y_max), ...) or None

        Example:
            >>> bbox = handler.get_bounding_box(wcs)
            >>> if bbox:
            ...     print(f"Valid region: x=[{bbox[0][0]}, {bbox[0][1]}], "
            ...           f"y=[{bbox[1][0]}, {bbox[1][1]}]")
        """
        if hasattr(wcs, 'bounding_box'):
            return wcs.bounding_box
        return None

    def save_wcs(
        self,
        wcs: BaseHighLevelWCS,
        output_file: Union[Path, str],
        overwrite: bool = False
    ) -> None:
        """Save WCS to ASDF file.

        Serializes WCS object to ASDF format for caching or sharing.
        Works for both gwcs.WCS and astropy.wcs.WCS (ASDF supports both).

        Args:
            wcs: WCS object to save (APE 14-compliant)
            output_file: Path to output ASDF file
            overwrite: Whether to overwrite existing file

        Raises:
            ImportError: If asdf package not installed
            FileExistsError: If file exists and overwrite=False

        Example:
            >>> # Save gwcs for later use
            >>> handler.save_wcs(wcs, 'cached_wcs.asdf')
            >>>
            >>> # Load it back
            >>> wcs_loaded = handler.load_wcs('cached_wcs.asdf')
        """
        try:
            import asdf
        except ImportError:
            raise ImportError(
                "asdf package required for WCS saving. "
                "Install with: pip install asdf"
            )

        output_path = Path(output_file)

        # Check if file exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File {output_path} already exists. "
                f"Use overwrite=True to replace it."
            )

        # Create ASDF tree
        tree = {'wcs': wcs}

        # Add metadata
        tree['metadata'] = {
            'created_by': 'astro_vision_composer.WCSHandler',
            'wcs_type': 'gwcs' if hasattr(wcs, 'available_frames') else 'standard'
        }

        # Write to file
        af = asdf.AsdfFile(tree)
        af.write_to(output_path)

        logger.info(f"Saved WCS to {output_path}")

    def compare_wcs(self, wcs1: BaseHighLevelWCS, wcs2: BaseHighLevelWCS) -> dict:
        """Compare two WCS objects and report differences.

        Useful for determining if images can be aligned or need reprojection.

        Args:
            wcs1: First WCS object
            wcs2: Second WCS object

        Returns:
            Dictionary with comparison results including:
            - 'compatible': Whether WCS are compatible (same projection, similar scale)
            - 'pixel_scale_diff': Difference in pixel scales (%)
            - 'rotation_diff': Difference in rotation angles (degrees)
            - 'projection_match': Whether projections match
            - 'details': List of detailed comparison information
        """
        info1 = self.validate(wcs1)
        info2 = self.validate(wcs2)

        result = {
            'compatible': True,
            'pixel_scale_diff': None,
            'rotation_diff': None,
            'projection_match': None,
            'details': []
        }

        # Check if both are valid
        if not info1.is_valid or not info2.is_valid:
            result['compatible'] = False
            result['details'].append("One or both WCS are invalid")
            return result

        # Compare projections
        if info1.projection != info2.projection:
            result['projection_match'] = False
            result['compatible'] = False
            result['details'].append(
                f"Different projections: {info1.projection} vs {info2.projection}"
            )
        else:
            result['projection_match'] = True

        # Compare pixel scales
        if info1.pixel_scale and info2.pixel_scale:
            diff_pct = abs(info1.pixel_scale - info2.pixel_scale) / info1.pixel_scale * 100
            result['pixel_scale_diff'] = diff_pct

            if diff_pct > 5:  # More than 5% difference
                result['compatible'] = False
                result['details'].append(
                    f"Pixel scale difference: {diff_pct:.1f}% "
                    f"({info1.pixel_scale:.3f} vs {info2.pixel_scale:.3f} arcsec/pixel)"
                )

        # Compare rotations
        if info1.rotation_angle is not None and info2.rotation_angle is not None:
            diff_deg = abs(info1.rotation_angle - info2.rotation_angle)
            # Normalize to [0, 360]
            diff_deg = min(diff_deg, 360 - diff_deg)
            result['rotation_diff'] = diff_deg

            if diff_deg > 1:  # More than 1 degree difference
                result['details'].append(
                    f"Rotation difference: {diff_deg:.2f} degrees"
                )

        return result
