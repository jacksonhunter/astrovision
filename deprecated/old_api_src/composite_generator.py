"""Composite image generation with advanced processing for astronomical images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import exposure, filters
from skimage.morphology import white_tophat, black_tophat, disk

logger = logging.getLogger(__name__)


class CompositeImageGenerator:
    """Generate visually appealing composite images from multiple bands.

    This class handles the technical aspects of creating planetarium-style
    astronomical images with high contrast, detail enhancement, and color mapping.
    """

    def __init__(self):
        """Initialize the composite generator."""
        self.bands: Dict[str, np.ndarray] = {}
        self.processed_bands: Dict[str, np.ndarray] = {}
        logger.info("Initialized CompositeImageGenerator")

    def add_band(self, name: str, data: np.ndarray) -> None:
        """Add a normalized band for compositing.

        Args:
            name: Name/identifier for this band
            data: Normalized band data (0-1 range)
        """
        if data.ndim != 2:
            raise ValueError(f"Band data must be 2D, got shape {data.shape}")

        self.bands[name] = data.copy()
        logger.info(f"Added band '{name}' with shape {data.shape}")

    def enhance_contrast(
        self,
        band_name: str,
        method: str = 'adaptive',
        clip_limit: float = 0.03
    ) -> np.ndarray:
        """Enhance contrast in a band.

        Args:
            band_name: Name of the band to enhance
            method: Enhancement method ('adaptive', 'histogram', 'gamma')
            clip_limit: Clipping limit for adaptive histogram equalization

        Returns:
            Contrast-enhanced band
        """
        data = self.bands[band_name].copy()

        if method == 'adaptive':
            # Adaptive histogram equalization (CLAHE)
            # Convert to uint8 for OpenCV
            data_uint8 = (data * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(data_uint8).astype(np.float32) / 255.0

        elif method == 'histogram':
            # Standard histogram equalization
            enhanced = exposure.equalize_hist(data)

        elif method == 'gamma':
            # Gamma correction for mid-tones
            enhanced = exposure.adjust_gamma(data, gamma=0.8)

        else:
            raise ValueError(f"Unknown enhancement method: {method}")

        logger.info(f"Enhanced contrast for band '{band_name}' using {method}")
        return enhanced

    def enhance_detail(
        self,
        band_name: str,
        scale: float = 1.5,
        radius: int = 3
    ) -> np.ndarray:
        """Enhance fine details using unsharp masking.

        Args:
            band_name: Name of the band to enhance
            scale: Amount of enhancement (1.0 = no change, >1.0 = more detail)
            radius: Radius of the unsharp mask

        Returns:
            Detail-enhanced band
        """
        data = self.bands[band_name].copy()

        # Create blurred version
        blurred = ndimage.gaussian_filter(data, sigma=radius)

        # Unsharp mask: original + (original - blurred) * scale
        enhanced = data + (data - blurred) * (scale - 1.0)

        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 1)

        logger.info(f"Enhanced details for band '{band_name}' (scale={scale}, radius={radius})")
        return enhanced

    def enhance_stars(
        self,
        band_name: str,
        threshold_percentile: float = 99.0,
        boost: float = 1.5
    ) -> np.ndarray:
        """Enhance bright point sources (stars).

        Args:
            band_name: Name of the band to enhance
            threshold_percentile: Percentile threshold for star detection
            boost: Multiplicative boost factor for stars

        Returns:
            Star-enhanced band
        """
        data = self.bands[band_name].copy()

        # Detect bright sources
        threshold = np.percentile(data, threshold_percentile)
        star_mask = data > threshold

        # Boost stars
        enhanced = data.copy()
        enhanced[star_mask] = np.clip(data[star_mask] * boost, 0, 1)

        logger.info(f"Enhanced stars in band '{band_name}' "
                   f"(threshold={threshold:.3f}, boost={boost})")
        return enhanced

    def create_star_mask(
        self,
        band_name: str,
        threshold_percentile: float = 98.0,
        min_size: int = 2
    ) -> np.ndarray:
        """Create a binary mask of bright stars for separate processing.

        Args:
            band_name: Name of the band to analyze
            threshold_percentile: Percentile threshold for star detection
            min_size: Minimum size in pixels for star detection

        Returns:
            Binary mask (True = star, False = background)
        """
        data = self.bands[band_name].copy()

        # Threshold
        threshold = np.percentile(data, threshold_percentile)
        mask = data > threshold

        # Morphological operations to clean up mask
        selem = disk(min_size)
        mask = ndimage.binary_opening(mask, structure=selem)

        logger.info(f"Created star mask from band '{band_name}' "
                   f"with {np.sum(mask)} star pixels")
        return mask

    def create_cmyk_composite(
        self,
        cyan_band: str,
        magenta_band: str,
        yellow_band: str,
        black_band: str,
        enhance_contrast: bool = True,
        enhance_details: bool = True,
        enhance_stars: bool = True
    ) -> np.ndarray:
        """Create a CMYK composite image.

        Args:
            cyan_band: Band name for cyan channel
            magenta_band: Band name for magenta channel
            yellow_band: Band name for yellow channel
            black_band: Band name for black (detail) channel
            enhance_contrast: Apply contrast enhancement
            enhance_details: Apply detail enhancement
            enhance_stars: Apply star enhancement

        Returns:
            RGB image array (H, W, 3) in 0-1 range
        """
        logger.info("Creating CMYK composite...")

        # Process each channel
        c = self.bands[cyan_band].copy()
        m = self.bands[magenta_band].copy()
        y = self.bands[yellow_band].copy()
        k = self.bands[black_band].copy()

        # Apply enhancements
        if enhance_contrast:
            c = self.enhance_contrast(cyan_band)
            m = self.enhance_contrast(magenta_band)
            y = self.enhance_contrast(yellow_band)
            k = self.enhance_contrast(black_band)

        if enhance_details:
            c = self.enhance_detail(cyan_band)
            m = self.enhance_detail(magenta_band)
            y = self.enhance_detail(yellow_band)
            k = self.enhance_detail(black_band)

        if enhance_stars:
            c = self.enhance_stars(cyan_band)
            m = self.enhance_stars(magenta_band)
            y = self.enhance_stars(yellow_band)

        # Convert CMYK to RGB
        # RGB = (1-C) * (1-K), (1-M) * (1-K), (1-Y) * (1-K)
        r = (1.0 - c) * (1.0 - k)
        g = (1.0 - m) * (1.0 - k)
        b = (1.0 - y) * (1.0 - k)

        # Stack into RGB
        rgb = np.dstack([r, g, b])

        logger.info("CMYK composite created")
        return rgb

    def create_rgb_composite(
        self,
        r_band: str,
        g_band: str,
        b_band: str,
        enhance_contrast: bool = True,
        enhance_details: bool = True,
        enhance_stars: bool = True,
        color_balance: bool = True
    ) -> np.ndarray:
        """Create an RGB composite with enhancements.

        Args:
            r_band: Band name for red channel
            g_band: Band name for green channel
            b_band: Band name for blue channel
            enhance_contrast: Apply contrast enhancement
            enhance_details: Apply detail enhancement
            enhance_stars: Apply star enhancement
            color_balance: Apply color balancing

        Returns:
            RGB image array (H, W, 3) in 0-1 range
        """
        logger.info("Creating RGB composite...")

        # Get bands
        r = self.bands[r_band].copy()
        g = self.bands[g_band].copy()
        b = self.bands[b_band].copy()

        # Apply enhancements to each channel
        if enhance_contrast:
            r = exposure.equalize_adapthist(r, clip_limit=0.03)
            g = exposure.equalize_adapthist(g, clip_limit=0.03)
            b = exposure.equalize_adapthist(b, clip_limit=0.03)

        if enhance_details:
            r = r + (r - ndimage.gaussian_filter(r, 2)) * 0.5
            g = g + (g - ndimage.gaussian_filter(g, 2)) * 0.5
            b = b + (b - ndimage.gaussian_filter(b, 2)) * 0.5

        if enhance_stars:
            # Boost bright stars
            r_thresh = np.percentile(r, 99)
            g_thresh = np.percentile(g, 99)
            b_thresh = np.percentile(b, 99)

            r = np.where(r > r_thresh, np.clip(r * 1.3, 0, 1), r)
            g = np.where(g > g_thresh, np.clip(g * 1.3, 0, 1), g)
            b = np.where(b > b_thresh, np.clip(b * 1.3, 0, 1), b)

        # Clip to valid range
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)

        # Stack into RGB
        rgb = np.dstack([r, g, b])

        # Color balance
        if color_balance:
            rgb = self.apply_color_balance(rgb)

        logger.info("RGB composite created")
        return rgb

    def apply_color_balance(self, rgb: np.ndarray) -> np.ndarray:
        """Apply color balancing to an RGB image.

        Args:
            rgb: RGB image array

        Returns:
            Color-balanced RGB image
        """
        # Calculate percentiles for each channel
        r_med = np.median(rgb[:, :, 0])
        g_med = np.median(rgb[:, :, 1])
        b_med = np.median(rgb[:, :, 2])

        # Calculate scaling factors
        avg_med = (r_med + g_med + b_med) / 3.0
        r_scale = avg_med / (r_med + 1e-8)
        g_scale = avg_med / (g_med + 1e-8)
        b_scale = avg_med / (b_med + 1e-8)

        # Apply scaling
        balanced = rgb.copy()
        balanced[:, :, 0] = np.clip(rgb[:, :, 0] * r_scale, 0, 1)
        balanced[:, :, 1] = np.clip(rgb[:, :, 1] * g_scale, 0, 1)
        balanced[:, :, 2] = np.clip(rgb[:, :, 2] * b_scale, 0, 1)

        logger.info(f"Applied color balance (R={r_scale:.2f}, G={g_scale:.2f}, B={b_scale:.2f})")
        return balanced

    def save_composite(
        self,
        rgb: np.ndarray,
        output_path: str | Path,
        quality: int = 95
    ) -> None:
        """Save the composite image.

        Args:
            rgb: RGB image array (0-1 range)
            output_path: Output file path
            quality: JPEG quality (1-100) or PNG compression level
        """
        output_path = Path(output_path)

        # Convert to 8-bit
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(rgb_uint8, mode='RGB')

        # Save
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            img.save(output_path, 'JPEG', quality=quality)
        else:
            img.save(output_path, 'PNG', compress_level=9)

        logger.info(f"Saved composite to: {output_path}")