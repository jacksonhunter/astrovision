"""Mission-specific adapters for FITS files.

This module provides mission-specific adapters that handle the unique conventions
and structures of different astronomical observatories and instruments.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from astropy.io import fits
import logging

logger = logging.getLogger(__name__)


class MissionAdapter(ABC):
    """Abstract base class for mission-specific FITS adapters.

    Each mission (JWST, HST, PanSTARRS, etc.) has unique conventions for
    organizing data within FITS files. Adapters provide a unified interface
    to access science, error, and quality data regardless of mission-specific
    naming conventions.
    """

    @abstractmethod
    def identify_science_extension(self, hdul: fits.HDUList) -> Tuple[int, str, int]:
        """Identify the primary science extension.

        Args:
            hdul: Opened FITS HDUList

        Returns:
            Tuple of (extension_index, extension_name, extension_version)
        """
        pass

    @abstractmethod
    def get_error_extension(
        self,
        hdul: fits.HDUList,
        science_version: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """Get the error/uncertainty extension name and version.

        Args:
            hdul: Opened FITS HDUList
            science_version: Version of the science extension to match

        Returns:
            Tuple of (extension_name, version) or None if not available
        """
        pass

    @abstractmethod
    def get_quality_extension(
        self,
        hdul: fits.HDUList,
        science_version: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """Get the data quality mask extension name and version.

        Args:
            hdul: Opened FITS HDUList
            science_version: Version of the science extension to match

        Returns:
            Tuple of (extension_name, version) or None if not available
        """
        pass

    def interpret_quality_flags(self, dq_value: int) -> List[str]:
        """Interpret data quality bitmask flags.

        Args:
            dq_value: Integer DQ value (bitmask)

        Returns:
            List of human-readable flag descriptions
        """
        # Default implementation - override in subclasses with mission-specific flags
        return [f"FLAG_{i}" for i in range(16) if dq_value & (1 << i)]

    def get_wcs_characteristics(
        self,
        header: fits.Header,
        instrument: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get mission-specific WCS quality characteristics.

        This method extracts WCS-related information from FITS headers to assess
        the quality and characteristics of the astrometric solution. It is separate
        from DQ flags and focuses on geometric/astrometric properties.

        Implementation Strategy (Option A - Header-driven):
        - Extract distortion model info from headers (SIP, TPV, etc.)
        - No assumptions about what "should" be there
        - Let real data teach us the patterns

        Args:
            header: FITS header object
            instrument: Optional instrument name for instrument-specific logic

        Returns:
            Dictionary with:
            - 'distortion_model': 'SIP'|'TPV'|'polynomial'|'none'|'unknown'
            - 'distortion_magnitude': 'low'|'moderate'|'high'|'unknown'
            - 'astrometric_quality': 'excellent'|'good'|'fair'|'poor'|'unknown'
            - 'notes': List[str] - Additional information
            - 'warnings': List[str] - Potential issues
        """
        result = {
            'distortion_model': 'unknown',
            'distortion_magnitude': 'unknown',
            'astrometric_quality': 'unknown',
            'notes': [],
            'warnings': []
        }

        # Check for SIP (Simple Imaging Polynomial) distortion keywords
        if 'A_ORDER' in header or 'B_ORDER' in header:
            result['distortion_model'] = 'SIP'
            result['notes'].append('SIP distortion coefficients present')

            # Check order of SIP polynomial
            a_order = header.get('A_ORDER', 0)
            b_order = header.get('B_ORDER', 0)
            max_order = max(a_order, b_order)

            if max_order >= 4:
                result['distortion_magnitude'] = 'high'
                result['notes'].append(f'High-order SIP polynomial (order {max_order})')
            elif max_order >= 2:
                result['distortion_magnitude'] = 'moderate'
            elif max_order > 0:
                result['distortion_magnitude'] = 'low'

        # Check for TPV (Tangent Plane with PV distortion) projection
        ctype1 = header.get('CTYPE1', '')
        if 'TPV' in ctype1:
            result['distortion_model'] = 'TPV'
            result['notes'].append('TPV projection includes distortion model')
            # TPV typically handles moderate distortion well
            result['distortion_magnitude'] = 'low'

        # Check for other distortion-related keywords
        if 'DISTORT' in header or 'D2IMERR1' in header:
            result['notes'].append('Additional distortion keywords present')

        # If no distortion model detected, assume none
        if result['distortion_model'] == 'unknown':
            result['distortion_model'] = 'none'
            result['distortion_magnitude'] = 'none'

        # Default implementation - override in subclasses for mission-specific logic
        return result

    def get_all_extensions_info(self, hdul: fits.HDUList) -> Dict[str, Any]:
        """Get comprehensive information about all extensions in the file.

        Args:
            hdul: Opened FITS HDUList

        Returns:
            Dictionary with extension organization info
        """
        sci_idx, sci_name, sci_ver = self.identify_science_extension(hdul)
        err_ext = self.get_error_extension(hdul, sci_ver)
        dq_ext = self.get_quality_extension(hdul, sci_ver)

        return {
            'science': {'index': sci_idx, 'name': sci_name, 'version': sci_ver},
            'error': {'name': err_ext[0], 'version': err_ext[1]} if err_ext else None,
            'quality': {'name': dq_ext[0], 'version': dq_ext[1]} if dq_ext else None,
            'mission': self.__class__.__name__.replace('Adapter', '')
        }


class PanSTARRSAdapter(MissionAdapter):
    """Adapter for Pan-STARRS FITS files.

    Pan-STARRS data products typically have simple single-extension FITS files
    where the science data is in the primary HDU or a 'COMPRESSED_IMAGE' extension.
    They usually don't include separate error or DQ arrays in the standard products.
    """

    def identify_science_extension(self, hdul: fits.HDUList) -> Tuple[int, str, int]:
        """Identify science extension in Pan-STARRS FITS file.

        Pan-STARRS stack/warp images are typically:
        - Primary HDU with data, OR
        - 'COMPRESSED_IMAGE' extension (compressed FITS)

        Args:
            hdul: Opened FITS HDUList

        Returns:
            Tuple of (index, name, version)
        """
        # Check for COMPRESSED_IMAGE extension first (common in newer files)
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'COMPRESSED_IMAGE' and hdu.data is not None:
                logger.debug(f"Found PanSTARRS science data in COMPRESSED_IMAGE extension")
                return (idx, 'COMPRESSED_IMAGE', hdu.ver)

        # Fall back to Primary HDU
        if hdul[0].data is not None:
            logger.debug(f"Found PanSTARRS science data in PRIMARY HDU")
            return (0, 'PRIMARY', 1)

        # If primary is empty, look for first image extension
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                logger.debug(f"Found PanSTARRS science data in extension {idx} ({hdu.name})")
                return (idx, hdu.name, hdu.ver)

        raise ValueError("Could not identify science extension in PanSTARRS file")

    def get_error_extension(
        self,
        hdul: fits.HDUList,
        science_version: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """Get error extension for Pan-STARRS data.

        Standard PanSTARRS products typically don't include error arrays
        in the same file. They may be provided as separate files.

        Args:
            hdul: Opened FITS HDUList
            science_version: Not used for PanSTARRS

        Returns:
            None (PanSTARRS doesn't typically include error in same file)
        """
        # Check if there's a separate error extension (non-standard)
        for hdu in hdul:
            if hdu.name in ['ERR', 'ERROR', 'SIGMA', 'UNCERTAINTY']:
                logger.debug(f"Found non-standard error extension: {hdu.name}")
                return (hdu.name, hdu.ver)

        return None

    def get_quality_extension(
        self,
        hdul: fits.HDUList,
        science_version: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """Get quality extension for Pan-STARRS data.

        Standard PanSTARRS products typically don't include DQ arrays.

        Args:
            hdul: Opened FITS HDUList
            science_version: Not used for PanSTARRS

        Returns:
            None (PanSTARRS doesn't typically include DQ in same file)
        """
        # Check if there's a DQ extension (non-standard)
        for hdu in hdul:
            if hdu.name in ['DQ', 'MASK', 'FLAGS']:
                logger.debug(f"Found non-standard quality extension: {hdu.name}")
                return (hdu.name, hdu.ver)

        return None

    def get_wcs_characteristics(
        self,
        header: fits.Header,
        instrument: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get WCS characteristics for PanSTARRS data.

        PanSTARRS uses TPV projection which includes distortion corrections.
        Stack products have excellent astrometry from multi-epoch co-addition.

        Args:
            header: FITS header object
            instrument: Not used for PanSTARRS

        Returns:
            Dictionary with WCS characteristics
        """
        # Start with base implementation
        result = super().get_wcs_characteristics(header, instrument)

        # PanSTARRS-specific checks
        projcell = header.get('PROJCELL', header.get('PROJECT', ''))

        if 'stack' in str(projcell).lower():
            result['astrometric_quality'] = 'excellent'
            result['notes'].append(
                'PanSTARRS stack - multi-epoch co-addition provides excellent astrometry'
            )
        elif 'warp' in str(projcell).lower():
            result['astrometric_quality'] = 'good'
            result['notes'].append('PanSTARRS warp - single-epoch astrometry')
        else:
            result['astrometric_quality'] = 'good'

        # Check for TPV projection (typical for PanSTARRS)
        ctype1 = header.get('CTYPE1', '')
        if 'TPV' in ctype1 and result['distortion_model'] == 'none':
            result['distortion_model'] = 'TPV'
            result['distortion_magnitude'] = 'low'
            result['notes'].append('TPV projection handles field distortion')

        return result


class JWSTAdapter(MissionAdapter):
    """Adapter for James Webb Space Telescope (JWST) FITS files.

    JWST uses multi-extension FITS with standard imset structure:
    - SCI: Science data (multiple versions for integrations)
    - ERR: Error/uncertainty
    - DQ: Data quality bitmask
    - Additional extensions: VAR_POISSON, VAR_RNOISE, etc.
    """

    # JWST DQ flag definitions (from JWST pipeline)
    DQ_FLAGS = {
        0: "DO_NOT_USE",
        1: "SATURATED",
        2: "JUMP_DET",
        3: "DROPOUT",
        4: "OUTLIER",
        5: "PERSISTENCE",
        6: "AD_FLOOR",
        7: "RESERVED",
        8: "UNRELIABLE_ERROR",
        9: "NON_SCIENCE",
        10: "DEAD",
        11: "HOT",
        12: "WARM",
        13: "LOW_QE",
        14: "RC",
        15: "TELEGRAPH",
        16: "NONLINEAR",
        17: "BAD_REF_PIXEL",
        18: "NO_FLAT_FIELD",
        19: "NO_GAIN_VALUE",
        20: "NO_LIN_CORR",
        21: "NO_SAT_CHECK",
        22: "UNRELIABLE_BIAS",
        23: "UNRELIABLE_DARK",
        24: "UNRELIABLE_SLOPE",
        25: "UNRELIABLE_FLAT",
    }

    def identify_science_extension(self, hdul: fits.HDUList) -> Tuple[int, str, int]:
        """Identify science extension in JWST FITS file.

        JWST files use 'SCI' extensions. For files with multiple integrations,
        we return the first SCI extension (version 1).

        Args:
            hdul: Opened FITS HDUList

        Returns:
            Tuple of (index, 'SCI', 1)
        """
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'SCI':
                logger.debug(f"Found JWST science data in SCI extension, version {hdu.ver}")
                return (idx, 'SCI', hdu.ver)

        raise ValueError("Could not find SCI extension in JWST file")

    def get_error_extension(
        self,
        hdul: fits.HDUList,
        science_version: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """Get error extension for JWST data.

        Args:
            hdul: Opened FITS HDUList
            science_version: Version number to match (default: 1)

        Returns:
            Tuple of ('ERR', version) or None
        """
        version = science_version or 1

        try:
            # Check if ERR extension exists with matching version
            hdu = hdul[('ERR', version)]
            return ('ERR', version)
        except KeyError:
            logger.warning(f"ERR extension version {version} not found in JWST file")
            return None

    def get_quality_extension(
        self,
        hdul: fits.HDUList,
        science_version: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """Get quality extension for JWST data.

        Args:
            hdul: Opened FITS HDUList
            science_version: Version number to match (default: 1)

        Returns:
            Tuple of ('DQ', version) or None
        """
        version = science_version or 1

        try:
            hdu = hdul[('DQ', version)]
            return ('DQ', version)
        except KeyError:
            logger.warning(f"DQ extension version {version} not found in JWST file")
            return None

    def interpret_quality_flags(self, dq_value: int) -> List[str]:
        """Interpret JWST DQ bitmask.

        Args:
            dq_value: Integer DQ value

        Returns:
            List of flag descriptions
        """
        flags = []
        for bit, description in self.DQ_FLAGS.items():
            if dq_value & (1 << bit):
                flags.append(description)
        return flags

    def get_wcs_characteristics(
        self,
        header: fits.Header,
        instrument: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get WCS characteristics for JWST data.

        JWST pipeline includes comprehensive distortion corrections.
        WCS quality is generally excellent for all instruments.

        Args:
            header: FITS header object
            instrument: JWST instrument name (e.g., 'NIRCAM', 'MIRI', 'NIRSPEC')

        Returns:
            Dictionary with WCS characteristics
        """
        # Start with base implementation
        result = super().get_wcs_characteristics(header, instrument)

        # JWST data typically has excellent WCS
        result['astrometric_quality'] = 'excellent'
        result['notes'].append('JWST pipeline includes full distortion correction')

        # Instrument-specific checks
        inst = (instrument or header.get('INSTRUME', '')).upper()

        if 'NIRCAM' in inst:
            # NIRCam has low distortion with SIP corrections
            if result['distortion_model'] == 'SIP':
                result['distortion_magnitude'] = 'low'
                result['notes'].append('NIRCam SIP distortion correction applied')
            elif result['distortion_model'] == 'none':
                result['warnings'].append(
                    'NIRCam distortion keywords missing - '
                    'corners may have positional errors up to 0.5 pixels'
                )
                result['distortion_magnitude'] = 'moderate'

        elif 'MIRI' in inst:
            # MIRI has more significant field distortion
            result['distortion_magnitude'] = 'moderate'
            result['notes'].append(
                'MIRI has significant field distortion - '
                'use reproject_exact for photometry'
            )

        elif 'NIRISS' in inst or 'NIRSPEC' in inst:
            result['distortion_magnitude'] = 'low'
            result['notes'].append(f'{inst} distortion corrections applied by pipeline')

        # Check for calibration level
        cal_level = header.get('CAL_VER', '')
        if cal_level:
            result['notes'].append(f'JWST calibration pipeline version: {cal_level}')

        return result


class HSTAdapter(MissionAdapter):
    """Adapter for Hubble Space Telescope (HST) FITS files.

    HST uses similar multi-extension structure to JWST:
    - SCI: Science data
    - ERR: Error
    - DQ: Data quality flags
    """

    # HST DQ flag definitions (common across instruments)
    DQ_FLAGS = {
        0: "GOOD_PIXEL",
        1: "REED_SOLOMON",  # Not used in newer data
        2: "CALIBRATION_DEFECT",
        3: "BAD_IN_BADPIX_FILE",
        4: "BAD_VALUE_REMOVED",
        5: "SATURATED",
        6: "DEFECTIVE",
        7: "COSMIC_RAY",
        8: "ATODSAT",  # A-to-D saturated
        9: "WARM_PIXEL",
        10: "HOT_PIXEL",
        11: "UNSTABLE",
        12: "RESERVED",
        13: "SOURCE_REJECTED",
        14: "RESERVED",
        15: "RESERVED",
    }

    def identify_science_extension(self, hdul: fits.HDUList) -> Tuple[int, str, int]:
        """Identify science extension in HST FITS file."""
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'SCI':
                logger.debug(f"Found HST science data in SCI extension, version {hdu.ver}")
                return (idx, 'SCI', hdu.ver)

        raise ValueError("Could not find SCI extension in HST file")

    def get_error_extension(
        self,
        hdul: fits.HDUList,
        science_version: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """Get error extension for HST data."""
        version = science_version or 1

        try:
            hdu = hdul[('ERR', version)]
            return ('ERR', version)
        except KeyError:
            return None

    def get_quality_extension(
        self,
        hdul: fits.HDUList,
        science_version: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """Get quality extension for HST data."""
        version = science_version or 1

        try:
            hdu = hdul[('DQ', version)]
            return ('DQ', version)
        except KeyError:
            return None

    def interpret_quality_flags(self, dq_value: int) -> List[str]:
        """Interpret HST DQ bitmask."""
        flags = []
        for bit, description in self.DQ_FLAGS.items():
            if dq_value & (1 << bit):
                flags.append(description)
        return flags

    def get_wcs_characteristics(
        self,
        header: fits.Header,
        instrument: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get WCS characteristics for HST data.

        HST WCS quality varies by instrument and whether the data has been drizzled.
        SIP distortion corrections are standard for most modern HST data.

        Args:
            header: FITS header object
            instrument: HST instrument name (e.g., 'ACS', 'WFC3', 'WFPC2')

        Returns:
            Dictionary with WCS characteristics
        """
        # Start with base implementation
        result = super().get_wcs_characteristics(header, instrument)

        # Check if drizzled product
        drizcorr = header.get('DRIZCORR', '')
        if drizcorr == 'COMPLETE':
            result['astrometric_quality'] = 'excellent'
            result['notes'].append('Drizzled product - high quality WCS')
        else:
            result['astrometric_quality'] = 'good'

        # Instrument-specific checks
        inst = (instrument or header.get('INSTRUME', '')).upper()

        if 'ACS' in inst:
            result['notes'].append('ACS WCS includes SIP distortion correction')
            if result['distortion_model'] == 'SIP':
                result['distortion_magnitude'] = 'low'

        elif 'WFC3' in inst:
            # Check if UVIS or IR channel
            detector = header.get('DETECTOR', '').upper()
            if 'UVIS' in detector:
                result['distortion_magnitude'] = 'low'
                result['notes'].append('WFC3/UVIS has low geometric distortion')
            elif 'IR' in detector:
                result['distortion_magnitude'] = 'moderate'
                result['notes'].append('WFC3/IR has moderate geometric distortion')

        elif 'WFPC2' in inst:
            result['warnings'].append(
                'WFPC2 has 4 chips with different pixel scales '
                '(0.046 vs 0.0996 arcsec/pixel) - requires special handling'
            )
            result['distortion_magnitude'] = 'high'
            result['astrometric_quality'] = 'fair'

        elif 'STIS' in inst or 'NICMOS' in inst or 'FOS' in inst:
            result['notes'].append(f'{inst} astrometry')

        return result


# Factory function to get appropriate adapter
def get_mission_adapter(mission: str) -> MissionAdapter:
    """Get the appropriate mission adapter for a given mission name.

    Args:
        mission: Mission name ('JWST', 'HST', 'PanSTARRS', etc.)

    Returns:
        Appropriate MissionAdapter instance

    Raises:
        ValueError: If mission not recognized
    """
    mission_upper = mission.upper()

    adapters = {
        'JWST': JWSTAdapter,
        'HST': HSTAdapter,
        'PANSTARRS': PanSTARRSAdapter,
        'PS1': PanSTARRSAdapter,
    }

    if mission_upper in adapters:
        return adapters[mission_upper]()

    raise ValueError(
        f"Unsupported mission '{mission}'. "
        f"Supported missions: {', '.join(adapters.keys())}"
    )
