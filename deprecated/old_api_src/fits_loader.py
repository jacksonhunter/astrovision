"""FITS file loading with mission-aware logic.

This module provides the FITSLoader class for loading FITS files with support
for multi-extension FITS (MEF), lazy loading, and automatic metadata extraction.
"""

from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import logging

from ..utilities.metadata import FITSMetadata, FITSMetadataResult

logger = logging.getLogger(__name__)


@dataclass
class FITSData:
    """Container for loaded FITS data.

    Attributes:
        filepath: Path to the FITS file
        science: Science data array
        error: Error/uncertainty array (if available)
        dq: Data quality mask array (if available)
        header: Primary or science extension header
        wcs: World Coordinate System object (if available)
        metadata: Extracted metadata
        extension_name: Name of the loaded extension (e.g., 'SCI', 'COMPRESSED_IMAGE')
        extension_version: Version number of the extension (for multi-imset files)
    """
    filepath: Path
    science: np.ndarray
    error: Optional[np.ndarray] = None
    dq: Optional[np.ndarray] = None
    header: Optional[fits.Header] = None
    wcs: Optional[WCS] = None
    metadata: Optional[FITSMetadataResult] = None
    extension_name: Optional[str] = None
    extension_version: Optional[int] = None

    @property
    def shape(self) -> tuple:
        """Shape of the science data array."""
        if self.science is None:
            return None
        return self.science.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the science array."""
        if self.science is None:
            return None
        return self.science.dtype

    def __repr__(self):
        """Pretty representation."""
        parts = [f"FITSData('{self.filepath.name}'"]
        if self.shape is not None:
            parts.append(f"shape={self.shape}")
        else:
            parts.append("lazy=True")
        if self.metadata and self.metadata.filter_name:
            parts.append(f"filter={self.metadata.filter_name}")
        if self.extension_name:
            ext = f"{self.extension_name}"
            if self.extension_version:
                ext += f"_{self.extension_version}"
            parts.append(f"ext={ext}")
        parts.append(f"has_error={self.error is not None}")
        parts.append(f"has_dq={self.dq is not None}")
        return ", ".join(parts) + ")"


class FITSLoader:
    """Load FITS files with mission-aware logic.

    This class provides intelligent FITS file loading that handles multi-extension
    FITS (MEF) files, automatically extracts metadata, and can identify science,
    error, and data quality arrays based on mission-specific conventions.

    Examples:
        >>> loader = FITSLoader()
        >>> data = loader.load('panstarrs_g.fits')
        >>> print(f"Loaded {data.metadata.filter_name} band: {data.shape}")

        >>> # Load specific extension from HST file
        >>> loader = FITSLoader(mission='HST')
        >>> data = loader.load('hst_acs.fits', extension=('SCI', 1))
    """

    def __init__(
        self,
        mission: Optional[str] = None,
        lazy: bool = False,
        memmap: bool = True
    ):
        """Initialize FITS loader.

        Args:
            mission: Mission name hint ('JWST', 'HST', 'PanSTARRS', etc.)
                    If None, will auto-detect from file headers
            lazy: If True, don't immediately load data arrays (faster inspection)
            memmap: Use memory mapping for large files (default True)
        """
        self.mission = mission
        self.lazy = lazy
        self.memmap = memmap
        self.metadata_extractor = FITSMetadata()

    def load(
        self,
        filepath: Union[str, Path],
        extension: Optional[Union[int, str, tuple]] = None,
        load_error: bool = True,
        load_dq: bool = True
    ) -> FITSData:
        """Load a FITS file and return a FITSData object.

        Args:
            filepath: Path to FITS file
            extension: Specific extension to load. Can be:
                      - int: Extension index (0 = primary)
                      - str: Extension name (e.g., 'SCI')
                      - tuple: (name, version) e.g., ('SCI', 1)
                      - None: Auto-detect science extension
            load_error: Load error array if available
            load_dq: Load data quality array if available

        Returns:
            FITSData object containing loaded data and metadata

        Raises:
            FileNotFoundError: If FITS file doesn't exist
            ValueError: If requested extension not found
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")

        logger.info(f"Loading FITS file: {filepath}")

        with fits.open(filepath, memmap=self.memmap, mode='readonly') as hdul:
            # Log file structure
            logger.debug(f"FITS structure:\n{hdul.info(output=False)}")

            # Determine which extension to load
            if extension is None:
                ext_info = self._identify_science_extension(hdul)
            else:
                ext_info = self._validate_extension(hdul, extension)

            ext_idx, ext_name, ext_ver = ext_info

            # Get the HDU
            if ext_name and ext_ver:
                hdu = hdul[(ext_name, ext_ver)]
            else:
                hdu = hdul[ext_idx]

            # Extract header and metadata
            header = hdu.header.copy()
            metadata = self.metadata_extractor.extract_metadata(header, ext_name)

            # Load science data
            if self.lazy:
                science = None
                logger.debug("Lazy loading enabled - science data not loaded")
            else:
                science = hdu.data
                if science is None:
                    raise ValueError(f"No data in extension {ext_idx} ({ext_name})")
                # Ensure it's a numpy array
                science = np.asarray(science)
                logger.debug(f"Loaded science data: shape={science.shape}, dtype={science.dtype}")

            # Try to extract WCS
            wcs = None
            try:
                wcs = WCS(header)
                if not wcs.has_celestial:
                    logger.warning("WCS found but has no celestial coordinates")
                    wcs = None
            except Exception as e:
                logger.debug(f"Could not extract WCS: {e}")

            # Try to load error array
            error = None
            if load_error and not self.lazy:
                error = self._load_companion_array(hdul, 'ERR', ext_ver)

            # Try to load DQ array
            dq = None
            if load_dq and not self.lazy:
                dq = self._load_companion_array(hdul, 'DQ', ext_ver)

        # Create FITSData object
        fits_data = FITSData(
            filepath=filepath,
            science=science,
            error=error,
            dq=dq,
            header=header,
            wcs=wcs,
            metadata=metadata,
            extension_name=ext_name,
            extension_version=ext_ver
        )

        logger.info(f"Successfully loaded: {fits_data}")

        return fits_data

    def list_extensions(self, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """List all extensions in a FITS file.

        Args:
            filepath: Path to FITS file

        Returns:
            List of dictionaries with extension info:
            [{'index': 0, 'name': 'PRIMARY', 'ver': 1, 'type': 'PrimaryHDU', 'shape': None}, ...]
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")

        extensions = []

        with fits.open(filepath, mode='readonly') as hdul:
            for idx, hdu in enumerate(hdul):
                ext_info = {
                    'index': idx,
                    'name': hdu.name,
                    'ver': hdu.ver,
                    'type': type(hdu).__name__,
                    'shape': hdu.data.shape if hdu.data is not None else None,
                    'dimensions': len(hdu.data.shape) if hdu.data is not None else 0
                }
                extensions.append(ext_info)

        return extensions

    def _identify_science_extension(self, hdul: fits.HDUList) -> tuple:
        """Identify the science extension in a FITS file.

        Args:
            hdul: Opened FITS HDUList

        Returns:
            Tuple of (index, name, version)
        """
        # Strategy 1: Look for 'SCI' extension (JWST, HST)
        for idx, hdu in enumerate(hdul):
            if hdu.name == 'SCI':
                return (idx, 'SCI', hdu.ver)

        # Strategy 2: Look for image data in first few extensions
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                # Found 2D or 3D data
                return (idx, hdu.name, hdu.ver)

        # Fallback: Use primary HDU
        logger.warning("Could not identify science extension, using primary HDU")
        return (0, 'PRIMARY', 1)

    def _validate_extension(
        self,
        hdul: fits.HDUList,
        extension: Union[int, str, tuple]
    ) -> tuple:
        """Validate and normalize extension specification.

        Args:
            hdul: Opened FITS HDUList
            extension: Extension specification

        Returns:
            Tuple of (index, name, version)

        Raises:
            ValueError: If extension not found
        """
        if isinstance(extension, int):
            # Extension by index
            if extension < 0 or extension >= len(hdul):
                raise ValueError(f"Extension index {extension} out of range (0-{len(hdul)-1})")
            hdu = hdul[extension]
            return (extension, hdu.name, hdu.ver)

        elif isinstance(extension, str):
            # Extension by name (use first version)
            for idx, hdu in enumerate(hdul):
                if hdu.name == extension:
                    return (idx, hdu.name, hdu.ver)
            raise ValueError(f"Extension '{extension}' not found in FITS file")

        elif isinstance(extension, tuple) and len(extension) == 2:
            # Extension by (name, version)
            ext_name, ext_ver = extension
            try:
                hdu = hdul[(ext_name, ext_ver)]
                for idx, h in enumerate(hdul):
                    if h is hdu:
                        return (idx, ext_name, ext_ver)
            except KeyError:
                raise ValueError(f"Extension {extension} not found in FITS file")

        else:
            raise ValueError(f"Invalid extension specification: {extension}")

    def _load_companion_array(
        self,
        hdul: fits.HDUList,
        array_type: str,
        version: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Load companion array (ERR, DQ, etc.) if it exists.

        Args:
            hdul: Opened FITS HDUList
            array_type: Type of array ('ERR', 'DQ', 'VAR_POISSON', etc.)
            version: Extension version to match

        Returns:
            Array data or None if not found
        """
        try:
            if version:
                hdu = hdul[(array_type, version)]
            else:
                # Find first occurrence
                for h in hdul:
                    if h.name == array_type:
                        hdu = h
                        break
                else:
                    return None

            if hdu.data is not None:
                logger.debug(f"Loaded {array_type} array: shape={hdu.data.shape}")
                return np.asarray(hdu.data)

        except (KeyError, IndexError):
            logger.debug(f"{array_type} array not found")

        return None

    def get_metadata(self, filepath: Union[str, Path]) -> FITSMetadataResult:
        """Extract metadata without loading full data arrays.

        This is faster than load() when you only need metadata.

        Args:
            filepath: Path to FITS file

        Returns:
            FITSMetadataResult object
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")

        with fits.open(filepath, mode='readonly') as hdul:
            # Try to get science extension header
            ext_info = self._identify_science_extension(hdul)
            ext_idx, ext_name, ext_ver = ext_info

            if ext_name and ext_ver:
                header = hdul[(ext_name, ext_ver)].header
            else:
                header = hdul[ext_idx].header

            metadata = self.metadata_extractor.extract_metadata(header, ext_name)

        return metadata
