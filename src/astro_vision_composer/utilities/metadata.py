"""FITS metadata extraction and validation.

This module provides the FITSMetadata class for extracting and validating
metadata from FITS file headers in a mission-aware manner.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from astropy.io import fits
import logging

logger = logging.getLogger(__name__)


@dataclass
class FITSMetadataResult:
    """Result of FITS metadata extraction.

    Attributes:
        instrument: Instrument name (e.g., 'NIRCam', 'ACS/WFC', 'NISP')
        filter_name: Filter/band name (e.g., 'F444W', 'g', 'H-alpha')
        wavelength: Central wavelength in nm (if available)
        exposure_time: Exposure time in seconds
        observation_date: Observation date string
        telescope: Telescope name
        target_name: Target object name
        mission: Mission name (e.g., 'JWST', 'HST', 'PanSTARRS')
        pixel_scale: Pixel scale in arcsec/pixel (if available)
        additional: Additional mission-specific metadata
        warnings: List of warning messages from validation
    """
    instrument: Optional[str] = None
    filter_name: Optional[str] = None
    wavelength: Optional[float] = None  # nm
    exposure_time: Optional[float] = None  # seconds
    observation_date: Optional[str] = None
    telescope: Optional[str] = None
    target_name: Optional[str] = None
    mission: Optional[str] = None
    pixel_scale: Optional[float] = None  # arcsec/pixel
    additional: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        """Pretty representation."""
        parts = []
        if self.mission:
            parts.append(f"Mission={self.mission}")
        if self.instrument:
            parts.append(f"Instrument={self.instrument}")
        if self.filter_name:
            parts.append(f"Filter={self.filter_name}")
        if self.wavelength:
            parts.append(f"λ={self.wavelength:.0f}nm")
        if self.exposure_time:
            parts.append(f"ExpTime={self.exposure_time:.1f}s")
        return f"FITSMetadata({', '.join(parts)})"


class FITSMetadata:
    """Extract and validate FITS file metadata.

    This class provides mission-aware metadata extraction from FITS headers,
    handling the different keyword conventions used by various observatories.

    Examples:
        >>> from astropy.io import fits
        >>> metadata = FITSMetadata()
        >>> with fits.open('example.fits') as hdul:
        ...     result = metadata.extract_metadata(hdul[0].header)
        ...     print(f"Filter: {result.filter_name}, Wavelength: {result.wavelength}nm")
    """

    # Common wavelengths for standard filters (nm)
    STANDARD_WAVELENGTHS = {
        # Optical (HST, ground-based)
        'F435W': 435, 'F475W': 475, 'F555W': 555, 'F606W': 606, 'F814W': 814,
        'U': 365, 'B': 445, 'V': 551, 'R': 658, 'I': 806,
        # PanSTARRS
        'g': 481, 'r': 617, 'i': 752, 'z': 866, 'y': 962,
        # JWST NIRCam
        'F070W': 704, 'F090W': 902, 'F115W': 1154, 'F150W': 1501,
        'F200W': 1989, 'F277W': 2762, 'F356W': 3568, 'F444W': 4421,
        # JWST MIRI
        'F560W': 5600, 'F770W': 7700, 'F1000W': 10000, 'F1130W': 11300,
        'F1280W': 12800, 'F1500W': 15000, 'F1800W': 18000, 'F2100W': 21000,
        # Common narrowband
        'H-alpha': 656, 'H-beta': 486, 'OIII': 501, 'SII': 672,
    }

    def __init__(self):
        """Initialize FITS metadata extractor."""
        pass

    def extract_metadata(
        self,
        header: fits.Header,
        extension_name: Optional[str] = None
    ) -> FITSMetadataResult:
        """Extract metadata from a FITS header.

        Args:
            header: FITS header object
            extension_name: Name of the extension (helps with mission identification)

        Returns:
            FITSMetadataResult containing extracted metadata
        """
        result = FITSMetadataResult()

        # Try to identify mission/telescope
        result.mission = self._identify_mission(header)
        result.telescope = self._get_keyword(header, ['TELESCOP', 'TELESC'])
        result.instrument = self._get_keyword(header, ['INSTRUME', 'INSTRU'])

        # Extract filter information
        result.filter_name = self._extract_filter(header)
        result.wavelength = self._get_wavelength(header, result.filter_name)

        # Observational parameters
        result.exposure_time = self._get_keyword(header, ['EXPTIME', 'EXPOSURE'], float)
        result.observation_date = self._get_keyword(header, ['DATE-OBS', 'DATE_OBS'])
        result.target_name = self._get_keyword(header, ['OBJECT', 'TARGNAME', 'TARGET'])

        # Pixel scale (from WCS if available)
        result.pixel_scale = self._get_pixel_scale(header)

        # Validate required fields
        self._validate_metadata(result)

        logger.debug(f"Extracted metadata: {result}")

        return result

    def _identify_mission(self, header: fits.Header) -> Optional[str]:
        """Identify the mission/observatory from header keywords."""
        telescope = self._get_keyword(header, ['TELESCOP', 'TELESC'], str, '').upper()

        if 'JWST' in telescope:
            return 'JWST'
        elif 'HST' in telescope or 'HUBBLE' in telescope:
            return 'HST'
        elif 'CHANDRA' in telescope:
            return 'Chandra'
        elif 'EUCLID' in telescope:
            return 'Euclid'
        elif 'PS1' in telescope or 'PANSTARRS' in telescope:
            return 'PanSTARRS'
        elif 'GALEX' in telescope:
            return 'GALEX'

        # Try to infer from instrument
        instrument = self._get_keyword(header, ['INSTRUME'], str, '').upper()
        if 'NIRCAM' in instrument or 'NIRSPEC' in instrument or 'MIRI' in instrument:
            return 'JWST'
        elif 'ACS' in instrument or 'WFC3' in instrument or 'WFPC' in instrument:
            return 'HST'
        elif 'NISP' in instrument or 'VIS' in instrument:
            return 'Euclid'

        return None

    def _extract_filter(self, header: fits.Header) -> Optional[str]:
        """Extract filter name from various possible keywords."""
        # Try standard keywords
        filter_name = self._get_keyword(header, ['FILTER', 'FILTNAM', 'FILTER1'])

        if filter_name:
            # Clean up filter names (remove extra characters)
            filter_name = filter_name.strip().upper()
            # Remove common prefixes/suffixes
            for prefix in ['CLEAR', 'F']:
                if filter_name.startswith(prefix) and len(filter_name) > len(prefix):
                    break  # Keep F### style names

            return filter_name

        return None

    def _get_wavelength(self, header: fits.Header, filter_name: Optional[str]) -> Optional[float]:
        """Get wavelength in nm, either from header or standard filter list."""
        # Try to get from header directly
        wave = self._get_keyword(header, ['PHOTPLAM', 'WAVELENG'], float)
        if wave:
            # Convert to nm if needed (PHOTPLAM is often in Angstroms)
            if wave > 100:  # Likely Angstroms
                wave = wave / 10.0
            return wave

        # Look up from standard filters
        if filter_name and filter_name in self.STANDARD_WAVELENGTHS:
            return self.STANDARD_WAVELENGTHS[filter_name]

        return None

    def _get_pixel_scale(self, header: fits.Header) -> Optional[float]:
        """Extract pixel scale from WCS keywords (arcsec/pixel)."""
        # Try CD matrix (most common)
        cd1_1 = self._get_keyword(header, ['CD1_1'], float)
        cd2_2 = self._get_keyword(header, ['CD2_2'], float)

        if cd1_1 is not None and cd2_2 is not None:
            # Average scale in degrees, convert to arcsec
            scale_deg = (abs(cd1_1) + abs(cd2_2)) / 2.0
            return scale_deg * 3600.0

        # Try CDELT (older style)
        cdelt1 = self._get_keyword(header, ['CDELT1'], float)
        cdelt2 = self._get_keyword(header, ['CDELT2'], float)

        if cdelt1 is not None and cdelt2 is not None:
            scale_deg = (abs(cdelt1) + abs(cdelt2)) / 2.0
            return scale_deg * 3600.0

        return None

    def _get_keyword(
        self,
        header: fits.Header,
        keywords: List[str],
        dtype: type = str,
        default: Any = None
    ) -> Any:
        """Try multiple keyword names, return first found value.

        Args:
            header: FITS header
            keywords: List of possible keyword names to try
            dtype: Expected data type
            default: Default value if not found

        Returns:
            Value of first found keyword, or default
        """
        for key in keywords:
            if key in header:
                try:
                    value = header[key]
                    if value is not None and value != '':
                        if dtype == float:
                            return float(value)
                        elif dtype == int:
                            return int(value)
                        else:
                            return str(value).strip()
                except (ValueError, TypeError):
                    continue

        return default

    def _validate_metadata(self, result: FITSMetadataResult) -> None:
        """Add warnings for missing critical metadata."""
        if result.filter_name is None:
            result.warnings.append("No filter information found")

        if result.wavelength is None and result.filter_name:
            result.warnings.append(f"Unknown wavelength for filter '{result.filter_name}'")

        if result.exposure_time is None:
            result.warnings.append("No exposure time found")

        if result.pixel_scale is None:
            result.warnings.append("No pixel scale information (WCS may be missing)")

    def validate_required(
        self,
        result: FITSMetadataResult,
        required_fields: List[str]
    ) -> bool:
        """Check if required fields are present.

        Args:
            result: Metadata result to validate
            required_fields: List of required field names

        Returns:
            True if all required fields are present and non-None
        """
        missing = []
        for field in required_fields:
            if not hasattr(result, field) or getattr(result, field) is None:
                missing.append(field)

        if missing:
            result.warnings.append(f"Missing required fields: {', '.join(missing)}")
            return False

        return True
