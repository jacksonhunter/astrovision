"""FITS file processing for extracting and analyzing astronomical image bands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.stats import sigma_clipped_stats

logger = logging.getLogger(__name__)


class FITSImageProcessor:
    """Process FITS files and extract band information for composite imaging.

    This class handles loading FITS files, extracting different wavelength bands,
    and preparing them for vision-guided composition.
    """

    def __init__(self, fits_file: str | Path):
        """Initialize the FITS processor.

        Args:
            fits_file: Path to the FITS file to process
        """
        self.fits_file = Path(fits_file)
        self.hdulist = None
        self.wcs = None
        self.bands: Dict[str, np.ndarray] = {}
        self.band_metadata: Dict[str, dict] = {}

        if not self.fits_file.exists():
            raise FileNotFoundError(f"FITS file not found: {self.fits_file}")

        logger.info(f"Initialized FITS processor for: {self.fits_file}")

    def load_fits(self) -> None:
        """Load the FITS file and extract WCS information."""
        logger.info("Loading FITS file...")
        self.hdulist = fits.open(self.fits_file)

        # Try to get WCS from primary HDU or first image HDU
        for hdu in self.hdulist:
            if hdu.data is not None:
                try:
                    self.wcs = WCS(hdu.header)
                    logger.info(f"Loaded WCS from HDU: {hdu.name}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load WCS from {hdu.name}: {e}")

        logger.info(f"FITS file loaded with {len(self.hdulist)} HDUs")

    def extract_bands(self, band_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Extract image bands from the FITS file.

        Args:
            band_names: Optional list of specific band names to extract.
                       If None, extracts all available image HDUs.

        Returns:
            Dictionary mapping band names to numpy arrays
        """
        if self.hdulist is None:
            self.load_fits()

        logger.info("Extracting bands from FITS file...")

        for i, hdu in enumerate(self.hdulist):
            if hdu.data is None:
                continue

            # Determine band name
            band_name = hdu.name if hdu.name else f"Band_{i}"

            # Check if we should extract this band
            if band_names is not None and band_name not in band_names:
                continue

            # Extract band data
            data = hdu.data.astype(np.float64)

            # Calculate statistics
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)

            # Store band data and metadata
            self.bands[band_name] = data
            self.band_metadata[band_name] = {
                'shape': data.shape,
                'mean': float(mean),
                'median': float(median),
                'std': float(std),
                'min': float(np.nanmin(data)),
                'max': float(np.nanmax(data)),
                'filter': hdu.header.get('FILTER', 'Unknown'),
                'wavelength': hdu.header.get('WAVELENG', None),
                'exposure': hdu.header.get('EXPTIME', None)
            }

            logger.info(f"Extracted band '{band_name}': {data.shape}, "
                       f"filter={self.band_metadata[band_name]['filter']}")

        logger.info(f"Extracted {len(self.bands)} bands")
        return self.bands

    def get_band_info(self) -> Dict[str, dict]:
        """Get metadata about all extracted bands.

        Returns:
            Dictionary of band metadata including statistics and header info
        """
        return self.band_metadata

    def normalize_band(
        self,
        band_name: str,
        method: str = 'zscale',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> np.ndarray:
        """Normalize a band for visualization.

        Args:
            band_name: Name of the band to normalize
            method: Normalization method ('zscale', 'linear', 'manual')
            vmin: Manual minimum value (for 'manual' method)
            vmax: Manual maximum value (for 'manual' method)

        Returns:
            Normalized array (0-1 range)
        """
        if band_name not in self.bands:
            raise ValueError(f"Band '{band_name}' not found")

        data = self.bands[band_name].copy()

        if method == 'zscale':
            # Use ZScale for astronomical data
            interval = ZScaleInterval()
            vmin_calc, vmax_calc = interval.get_limits(data)
        elif method == 'manual':
            if vmin is None or vmax is None:
                raise ValueError("Must provide vmin and vmax for manual normalization")
            vmin_calc, vmax_calc = vmin, vmax
        else:  # linear
            # Use sigma-clipped stats
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            vmin_calc = median - 2 * std
            vmax_calc = median + 10 * std

        # Normalize to 0-1
        normalized = np.clip((data - vmin_calc) / (vmax_calc - vmin_calc), 0, 1)

        logger.info(f"Normalized band '{band_name}' using {method} method "
                   f"(vmin={vmin_calc:.2e}, vmax={vmax_calc:.2e})")

        return normalized

    def create_preview_rgb(
        self,
        r_band: str,
        g_band: str,
        b_band: str,
        normalize_method: str = 'zscale'
    ) -> np.ndarray:
        """Create a quick RGB preview from three bands.

        Args:
            r_band: Band name for red channel
            g_band: Band name for green channel
            b_band: Band name for blue channel
            normalize_method: Method to normalize each band

        Returns:
            RGB image array (H, W, 3) with values in 0-1 range
        """
        r = self.normalize_band(r_band, method=normalize_method)
        g = self.normalize_band(g_band, method=normalize_method)
        b = self.normalize_band(b_band, method=normalize_method)

        # Stack into RGB
        rgb = np.dstack([r, g, b])

        logger.info(f"Created RGB preview from bands: R={r_band}, G={g_band}, B={b_band}")

        return rgb

    def get_available_bands(self) -> List[str]:
        """Get list of available band names.

        Returns:
            List of band names that have been extracted
        """
        return list(self.bands.keys())

    def close(self) -> None:
        """Close the FITS file."""
        if self.hdulist is not None:
            self.hdulist.close()
            logger.info("Closed FITS file")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()