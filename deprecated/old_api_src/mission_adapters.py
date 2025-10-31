"""Production-ready mission adapter suite for all 24+ astronomical observatories.

Combines comprehensive mission coverage with robust error handling, WCS management,
and metadata preservation. Auto-registers all adapters on import.

Architecture:
- BaseMissionAdapter: Production-ready base with context managers, validation
- Concrete adapters: Mission-specific implementations for 24+ observatories
- Registry: Auto-registration system for factory pattern

Supported missions span space telescopes, ground observatories, all-sky surveys,
radio/sub-mm interferometry, and multi-mission aggregators.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, InvalidTransformError, NoConvergence
from astropy.io.fits import VerifyError
import astropy.units as u

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class MissionAdapterError(Exception):
    """Base exception for mission adapter errors."""
    pass


class WCSError(MissionAdapterError):
    """WCS-related errors."""
    pass


class DataValidationError(MissionAdapterError):
    """Data validation errors."""
    pass


# ============================================================================
# ENUMS AND DATACLASSES
# ============================================================================

class WCSQuality(Enum):
    """WCS astrometric quality levels with quantitative thresholds."""
    EXCELLENT = (0.01, "Sub-milliarcsecond (<10 mas) - space telescopes")
    GOOD = (0.05, "Milliarcsecond (10-50 mas) - modern ground+Gaia")
    FAIR = (0.3, "Sub-arcsecond (50-300 mas) - typical ground")
    POOR = (1.0, "Arc-second (>300 mas) - legacy data")
    UNKNOWN = (None, "Quality not assessed")

    def __init__(self, threshold_arcsec: Optional[float], description: str):
        self.threshold = threshold_arcsec
        self.description = description


class DistortionModel(Enum):
    """WCS distortion model types."""
    NONE = "none"
    SIP = "SIP"
    TPV = "TPV"
    GWCS = "gwcs"
    TDD = "TDD"
    LAYERED = "layered"
    POLYNOMIAL = "polynomial"
    UNKNOWN = "unknown"


@dataclass
class SaturationInfo:
    """Saturation characteristics for a detector/instrument."""
    ad_limit: int
    saturation_threshold: Optional[float]
    saturation_reference: str
    well_depth: Optional[float] = None
    linearity_limit: Optional[float] = None

    def is_saturated(self, value: float) -> bool:
        if self.saturation_threshold is not None:
            return value >= self.saturation_threshold
        return value >= self.ad_limit


# ============================================================================
# REGISTRY FOR AUTO-REGISTRATION
# ============================================================================

_ADAPTER_REGISTRY: Dict[str, type] = {}


def register_adapter(*mission_names: str):
    """Decorator to register mission adapter classes.

    Args:
        *mission_names: One or more mission names/aliases to register
    """
    def decorator(cls):
        for name in mission_names:
            _ADAPTER_REGISTRY[name.upper()] = cls
        return cls
    return decorator


def get_adapter_class(mission: str) -> type:
    """Get adapter class for a mission.

    Args:
        mission: Mission name (case-insensitive)

    Returns:
        Adapter class

    Raises:
        ValueError: If mission not registered
    """
    mission_upper = mission.upper()
    if mission_upper not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unsupported mission '{mission}'. "
            f"Supported: {', '.join(sorted(_ADAPTER_REGISTRY.keys()))}"
        )
    return _ADAPTER_REGISTRY[mission_upper]


# ============================================================================
# BASE MISSION ADAPTER
# ============================================================================

class BaseMissionAdapter(ABC):
    """Production-ready base class for mission-specific FITS adapters.

    Provides:
    - Context manager for automatic cleanup
    - Robust error handling and validation
    - WCS management
    - Metadata caching
    - Data quality masking
    """

    MISSION_NAME = "GENERIC"
    SUPPORTED_INSTRUMENTS = []
    REQUIRED_KEYWORDS = ['TELESCOP', 'INSTRUME']

    def __init__(self, strict_validation: bool = True):
        """Initialize mission adapter.

        Args:
            strict_validation: If True, raise exceptions on validation failures
        """
        self.strict_validation = strict_validation
        self.hdul: Optional[fits.HDUList] = None
        self.filename: Optional[str] = None
        self.wcs: Optional[WCS] = None
        self._metadata_cache: Dict[str, Any] = {}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure file cleanup."""
        self.close()
        return False

    def open(self, filename: Union[str, Path], memmap: bool = True) -> 'BaseMissionAdapter':
        """Open and validate FITS file.

        Args:
            filename: Path to FITS file
            memmap: Use memory mapping for large files

        Returns:
            self for method chaining
        """
        self.filename = str(filename)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.hdul = fits.open(
                    self.filename,
                    mode='readonly',
                    memmap=memmap,
                    ignore_missing_end=True
                )

            self._validate_structure()
            self._initialize_wcs()
            self._validate_mission_requirements()

            logger.info(f"Successfully opened {self.MISSION_NAME} file: {self.filename}")
            return self

        except Exception as e:
            logger.error(f"Error opening {filename}: {e}")
            if self.hdul is not None:
                self.hdul.close()
                self.hdul = None
            raise DataValidationError(f"Failed to open file: {e}") from e

    def close(self):
        """Close FITS file and release resources."""
        if self.hdul is not None:
            try:
                self.hdul.close()
            except Exception as e:
                logger.warning(f"Error closing file: {e}")
            finally:
                self.hdul = None
                self.wcs = None
                self._metadata_cache = {}

    def _validate_structure(self):
        """Validate basic FITS file structure."""
        if len(self.hdul) < 1:
            raise DataValidationError("Empty FITS file")

        # Find science extension
        sci_idx, sci_name, sci_ver = self.identify_science_extension(self.hdul)
        self.primary_hdu_index = sci_idx

        data = self.get_data()
        if data.size == 0:
            raise DataValidationError("Empty data array")
        if not np.isfinite(data).any():
            raise DataValidationError("All data values are NaN or Inf")

    def _validate_mission_requirements(self):
        """Validate mission-specific requirements."""
        header = self.get_header()

        # Check required keywords
        missing = [kw for kw in self.REQUIRED_KEYWORDS if kw not in header]
        if missing:
            msg = f"Missing required keywords: {missing}"
            if self.strict_validation:
                raise DataValidationError(msg)
            else:
                logger.warning(msg)

        # Check instrument support
        if self.SUPPORTED_INSTRUMENTS:
            instrument = header.get('INSTRUME', 'UNKNOWN')
            if instrument not in self.SUPPORTED_INSTRUMENTS:
                msg = f"Unsupported instrument: {instrument}"
                if self.strict_validation:
                    raise DataValidationError(msg)
                else:
                    logger.warning(msg)

    def _initialize_wcs(self):
        """Initialize WCS - default implementation."""
        try:
            header = self.get_header()
            self.wcs = WCS(header, fix=True, naxis=2)
            if self.wcs.wcs.naxis == 0:
                self.wcs = None
        except Exception as e:
            logger.warning(f"WCS initialization failed: {e}")
            self.wcs = None

    @abstractmethod
    def identify_science_extension(self, hdul: fits.HDUList) -> Tuple[int, str, int]:
        """Identify the primary science extension.

        Returns:
            Tuple of (extension_index, extension_name, extension_version)
        """
        pass

    @abstractmethod
    def get_error_extension(self, hdul: fits.HDUList,
                           science_version: Optional[int] = None) -> Optional[Tuple[str, int]]:
        """Get error/uncertainty extension."""
        pass

    @abstractmethod
    def get_quality_extension(self, hdul: fits.HDUList,
                             science_version: Optional[int] = None) -> Optional[Tuple[str, int]]:
        """Get data quality extension."""
        pass

    @abstractmethod
    def get_saturation_info(self, header: fits.Header) -> SaturationInfo:
        """Get saturation characteristics."""
        pass

    def get_data(self, hdu_index: Optional[int] = None) -> np.ndarray:
        """Get data array from FITS file."""
        if self.hdul is None:
            raise ValueError("No file opened")
        idx = hdu_index if hdu_index is not None else self.primary_hdu_index
        return self.hdul[idx].data

    def get_header(self, hdu_index: Optional[int] = None) -> fits.Header:
        """Get header from FITS file."""
        if self.hdul is None:
            raise ValueError("No file opened")
        idx = hdu_index if hdu_index is not None else self.primary_hdu_index
        return self.hdul[idx].header

    def get_wcs(self) -> Optional[WCS]:
        """Get WCS object for coordinate transformations."""
        return self.wcs

    def get_metadata(self, refresh: bool = False) -> Dict[str, Any]:
        """Extract standardized metadata."""
        if not refresh and self._metadata_cache:
            return self._metadata_cache

        header = self.get_header()
        metadata = {
            'mission': self.MISSION_NAME,
            'filename': self.filename,
            'telescope': header.get('TELESCOP', 'UNKNOWN'),
            'instrument': header.get('INSTRUME', 'UNKNOWN'),
            'filter': header.get('FILTER', header.get('FILTNAM1', 'UNKNOWN')),
            'object': header.get('OBJECT', 'UNKNOWN'),
            'date_obs': header.get('DATE-OBS', 'UNKNOWN'),
            'exptime': header.get('EXPTIME', 0.0),
            'wcs_available': self.wcs is not None,
        }

        self._metadata_cache = metadata
        return metadata


# ============================================================================
# SPACE TELESCOPES - OPTICAL/IR/UV
# ============================================================================

@register_adapter('JWST')
class JWSTAdapter(BaseMissionAdapter):
    """JWST - Near/Mid-IR flagship with gwcs."""

    MISSION_NAME = "JWST"
    SUPPORTED_INSTRUMENTS = ['NIRCAM', 'NIRISS', 'NIRSPEC', 'MIRI', 'FGS']
    REQUIRED_KEYWORDS = ['TELESCOP', 'INSTRUME', 'DETECTOR']

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'SCI':
                return (idx, 'SCI', hdu.ver)
        raise ValueError("Could not find SCI extension in JWST file")

    def get_error_extension(self, hdul, science_version=None):
        version = science_version or 1
        try:
            hdul[('ERR', version)]
            return ('ERR', version)
        except KeyError:
            return None

    def get_quality_extension(self, hdul, science_version=None):
        version = science_version or 1
        try:
            hdul[('DQ', version)]
            return ('DQ', version)
        except KeyError:
            return None

    def get_saturation_info(self, header):
        return SaturationInfo(
            ad_limit=65535,
            saturation_threshold=None,
            saturation_reference="CRDS",
            linearity_limit=0.8,
        )


@register_adapter('HST', 'HUBBLE')
class HSTAdapter(BaseMissionAdapter):
    """HST - Optical/UV/Near-IR workhorse with TDD."""

    MISSION_NAME = "HST"
    SUPPORTED_INSTRUMENTS = ['ACS', 'WFC3', 'WFPC2', 'STIS', 'NICMOS']
    REQUIRED_KEYWORDS = ['TELESCOP', 'INSTRUME']

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'SCI':
                return (idx, 'SCI', hdu.ver)
        raise ValueError("Could not find SCI extension in HST file")

    def get_error_extension(self, hdul, science_version=None):
        version = science_version or 1
        try:
            hdul[('ERR', version)]
            return ('ERR', version)
        except KeyError:
            return None

    def get_quality_extension(self, hdul, science_version=None):
        version = science_version or 1
        try:
            hdul[('DQ', version)]
            return ('DQ', version)
        except KeyError:
            return None

    def get_saturation_info(self, header):
        sat_threshold = header.get('SATURATE', None)
        return SaturationInfo(
            ad_limit=65535,
            saturation_threshold=sat_threshold,
            saturation_reference="header",
        )


@register_adapter('GALEX')
class GALEXAdapter(BaseMissionAdapter):
    """GALEX - UV imaging (NUV: 1750-2750Å, FUV: 1350-1750Å)."""

    MISSION_NAME = "GALEX"
    REQUIRED_KEYWORDS = ['TELESCOP']

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('SPITZER')
class SpitzerAdapter(BaseMissionAdapter):
    """Spitzer - Mid/Far-IR (retired 2020)."""

    MISSION_NAME = "Spitzer"
    REQUIRED_KEYWORDS = ['TELESCOP']

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        for hdu in hdul:
            if 'FUNC' in hdu.name or 'UNC' in hdu.name:
                return (hdu.name, hdu.ver)
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('HERSCHEL')
class HerschelAdapter(BaseMissionAdapter):
    """Herschel - Far-IR/sub-mm (retired 2013)."""

    MISSION_NAME = "Herschel"
    REQUIRED_KEYWORDS = ['TELESCOP']

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'image' or (hdu.data is not None and len(hdu.data.shape) >= 2):
                return (idx, hdu.name, hdu.ver)
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        for hdu in hdul:
            if 'error' in hdu.name.lower() or 'stddev' in hdu.name.lower():
                return (hdu.name, hdu.ver)
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=32767, saturation_threshold=None,
                            saturation_reference="instrument_spec")


# ============================================================================
# SPACE TELESCOPES - HIGH ENERGY
# ============================================================================

@register_adapter('CHANDRA')
class ChandraAdapter(BaseMissionAdapter):
    """Chandra - X-ray imaging."""

    MISSION_NAME = "Chandra"
    REQUIRED_KEYWORDS = ['TELESCOP']

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'EVENTS' and isinstance(hdu, fits.BinTableHDU):
                return (idx, 'EVENTS', hdu.ver)
        if hdul[0].data is not None and len(hdul[0].data.shape) >= 2:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=32767, saturation_threshold=None,
                            saturation_reference="ACIS_PILEUP")


@register_adapter('XMM', 'XMM-NEWTON')
class XMMNewtonAdapter(BaseMissionAdapter):
    """XMM-Newton - X-ray spectroscopy."""

    MISSION_NAME = "XMM-Newton"
    REQUIRED_KEYWORDS = ['TELESCOP']

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'EVENTS':
                return (idx, 'EVENTS', hdu.ver)
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=32767, saturation_threshold=None,
                            saturation_reference="pile_up_fraction")


@register_adapter('FERMI')
class FermiAdapter(BaseMissionAdapter):
    """Fermi - Gamma-ray all-sky."""

    MISSION_NAME = "Fermi"
    REQUIRED_KEYWORDS = ['TELESCOP']

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name in ['EVENTS', 'SKYMAP']:
                return (idx, hdu.name, hdu.ver)
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=0, saturation_threshold=None,
                            saturation_reference="photon_counting")


# ============================================================================
# GROUND OBSERVATORIES
# ============================================================================

@register_adapter('VLT', 'ESO')
class ESOVLTAdapter(BaseMissionAdapter):
    """ESO VLT - 8m class optical/IR."""

    MISSION_NAME = "ESO VLT"
    REQUIRED_KEYWORDS = ['TELESCOP']

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        for hdu in hdul:
            if 'STAT' in hdu.name or 'ERR' in hdu.name:
                return (hdu.name, hdu.ver)
        return None

    def get_quality_extension(self, hdul, science_version=None):
        for hdu in hdul:
            if 'QUAL' in hdu.name or 'DQ' in hdu.name:
                return (hdu.name, hdu.ver)
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('GEMINI', 'GEMINI-N', 'GEMINI-S')
class GeminiAdapter(BaseMissionAdapter):
    """Gemini - Twin 8.1m telescopes."""

    MISSION_NAME = "Gemini"
    REQUIRED_KEYWORDS = ['TELESCOP']

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'SCI':
                return (idx, 'SCI', hdu.ver)
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        version = science_version or 1
        try:
            hdul[('VAR', version)]
            return ('VAR', version)
        except KeyError:
            return None

    def get_quality_extension(self, hdul, science_version=None):
        version = science_version or 1
        try:
            hdul[('DQ', version)]
            return ('DQ', version)
        except KeyError:
            return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('CADC')
class CADCAdapter(BaseMissionAdapter):
    """CADC - Canadian multi-mission."""

    MISSION_NAME = "CADC"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('NOIRLAB', 'DECAM')
class NOIRLabAdapter(BaseMissionAdapter):
    """NOIRLab - DECam deep imaging."""

    MISSION_NAME = "NOIRLab"
    REQUIRED_KEYWORDS = []

    def __init__(self, strict_validation: bool = False):
        """Initialize NOIRLab adapter (relaxed validation by default)."""
        super().__init__(strict_validation)

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if 'COMPRESSED' in hdu.name:
                return (idx, hdu.name, hdu.ver)
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        for hdu in hdul:
            if 'MASK' in hdu.name or 'DQ' in hdu.name:
                return (hdu.name, hdu.ver)
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


# ============================================================================
# ALL-SKY SURVEYS
# ============================================================================

@register_adapter('PANSTARRS', 'PS1')
class PanSTARRSAdapter(BaseMissionAdapter):
    """PanSTARRS - Optical all-sky (grizy)."""

    MISSION_NAME = "PanSTARRS"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'COMPRESSED_IMAGE' and hdu.data is not None:
                return (idx, 'COMPRESSED_IMAGE', hdu.ver)
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('SDSS')
class SDSSAdapter(BaseMissionAdapter):
    """SDSS - ugriz all-sky."""

    MISSION_NAME = "SDSS"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('2MASS')
class TwoMASSAdapter(BaseMissionAdapter):
    """2MASS - Near-IR all-sky."""

    MISSION_NAME = "2MASS"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('WISE')
class WISEAdapter(BaseMissionAdapter):
    """WISE - Mid-IR all-sky."""

    MISSION_NAME = "WISE"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('UKIDSS')
class UKIDSSAdapter(BaseMissionAdapter):
    """UKIDSS - Near-IR legacy survey."""

    MISSION_NAME = "UKIDSS"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('ZTF')
class ZTFAdapter(BaseMissionAdapter):
    """ZTF - Time-domain optical."""

    MISSION_NAME = "ZTF"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        for hdu in hdul:
            if 'MASK' in hdu.name:
                return (hdu.name, hdu.ver)
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


# ============================================================================
# RADIO/SUB-MM
# ============================================================================

@register_adapter('ALMA')
class ALMAAdapter(BaseMissionAdapter):
    """ALMA - Sub-mm interferometry."""

    MISSION_NAME = "ALMA"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=0, saturation_threshold=None,
                            saturation_reference="N/A")


@register_adapter('CASDA', 'ASKAP')
class CASDAAdapter(BaseMissionAdapter):
    """CASDA - ASKAP L-band radio."""

    MISSION_NAME = "CASDA"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=0, saturation_threshold=None,
                            saturation_reference="N/A")


@register_adapter('MAGPIS', 'VLA')
class MAGPISAdapter(BaseMissionAdapter):
    """MAGPIS - VLA Galactic plane."""

    MISSION_NAME = "MAGPIS"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=0, saturation_threshold=None,
                            saturation_reference="N/A")


# ============================================================================
# SPECIAL & AGGREGATORS
# ============================================================================

@register_adapter('GAIA')
class GaiaAdapter(BaseMissionAdapter):
    """Gaia - Astrometry mission (catalog-based)."""

    MISSION_NAME = "Gaia"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if isinstance(hdu, fits.BinTableHDU):
                return (idx, hdu.name, hdu.ver)
        raise ValueError("Gaia data is catalog-based, not imaging")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=0, saturation_threshold=None,
                            saturation_reference="N/A")


@register_adapter('TESS')
class TESSAdapter(BaseMissionAdapter):
    """TESS - Time-domain exoplanet survey."""

    MISSION_NAME = "TESS"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if 'PIXELS' in hdu.name or hdu.data is not None:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        for hdu in hdul:
            if 'QUALITY' in hdu.name:
                return (hdu.name, hdu.ver)
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="instrument_spec")


@register_adapter('SKYVIEW')
class SkyViewAdapter(BaseMissionAdapter):
    """SkyView - Multi-survey aggregator."""

    MISSION_NAME = "SkyView"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="survey_dependent")


@register_adapter('ESASKY')
class ESASkyAdapter(BaseMissionAdapter):
    """ESASky - ESA multi-mission portal."""

    MISSION_NAME = "ESASky"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        if hdul[0].data is not None:
            return (0, 'PRIMARY', 1)
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None:
                return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=65535, saturation_threshold=None,
                            saturation_reference="mission_dependent")


@register_adapter('HEASARC')
class HEASARCAdapter(BaseMissionAdapter):
    """HEASARC - Multi-mission high-energy aggregator."""

    MISSION_NAME = "HEASARC"
    REQUIRED_KEYWORDS = []

    def identify_science_extension(self, hdul):
        for idx, hdu in enumerate(hdul):
            if hdu.name in ['EVENTS', 'IMAGE', 'PRIMARY']:
                if hdu.data is not None:
                    return (idx, hdu.name, hdu.ver)
        raise ValueError("No science extension found")

    def get_error_extension(self, hdul, science_version=None):
        return None

    def get_quality_extension(self, hdul, science_version=None):
        return None

    def get_saturation_info(self, header):
        return SaturationInfo(ad_limit=0, saturation_threshold=None,
                            saturation_reference="mission_dependent")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_mission_adapter(mission: str, **kwargs) -> BaseMissionAdapter:
    """Factory function to get mission adapter instance.

    Args:
        mission: Mission name (case-insensitive)
        **kwargs: Arguments passed to adapter constructor

    Returns:
        Initialized adapter instance (not opened)

    Raises:
        ValueError: If mission not recognized

    Example:
        >>> adapter = get_mission_adapter('JWST')
        >>> with adapter.open('file.fits') as f:
        ...     data = f.get_data()
    """
    adapter_class = get_adapter_class(mission)
    return adapter_class(**kwargs)
