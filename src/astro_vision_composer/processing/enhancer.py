"""Image enhancement techniques for astronomical data.

This module provides the Enhancer class for various image enhancement techniques
including CLAHE, unsharp masking, and star highlighting.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)

# Try to import skimage for CLAHE
try:
    from skimage import exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available. Install with: pip install scikit-image")


class Enhancer:
    """Apply enhancement techniques to astronomical images.

    Provides methods for:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Unsharp masking
    - Star highlighting
    - Luminance masking

    Example:
        >>> enhancer = Enhancer()
        >>> # Apply unsharp mask to bring out detail
        >>> enhanced = enhancer.unsharp_mask(data, sigma=2.0, strength=1.5)
    """

    def __init__(self):
        """Initialize the Enhancer."""
        pass

    def apply_clahe(
        self,
        data: np.ndarray,
        kernel_size: Optional[int] = None,
        clip_limit: float = 0.01,
        nbins: int = 256
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        CLAHE enhances local contrast while preventing over-amplification of noise.
        It's particularly effective for bringing out faint structures.

        Args:
            data: Input image data (should be in [0, 1] range)
            kernel_size: Size of local region (None = auto-calculate from image size)
            clip_limit: Contrast limiting threshold (0.0-1.0, lower = less aggressive)
            nbins: Number of histogram bins

        Returns:
            Enhanced image data

        Raises:
            ImportError: If scikit-image is not installed

        Example:
            >>> enhanced = enhancer.apply_clahe(
            ...     normalized_data,
            ...     kernel_size=64,
            ...     clip_limit=0.01
            ... )
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required. Install with: pip install scikit-image")

        # Auto-calculate kernel size if not provided
        if kernel_size is None:
            # Use ~1/8 of image size
            kernel_size = max(data.shape) // 8
            kernel_size = max(8, min(kernel_size, 256))  # Clamp to reasonable range

        logger.info(f"Applying CLAHE: kernel_size={kernel_size}, clip_limit={clip_limit}")

        # Ensure data is in [0, 1] range
        data_clipped = np.clip(data, 0, 1)

        # Apply CLAHE
        enhanced = exposure.equalize_adapthist(
            data_clipped,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins
        )

        return enhanced

    def unsharp_mask(
        self,
        data: np.ndarray,
        sigma: float = 2.0,
        strength: float = 1.5
    ) -> np.ndarray:
        """Apply unsharp masking to enhance fine details.

        Unsharp masking subtracts a blurred version of the image to enhance edges
        and fine structures. This is one of the most commonly used enhancement
        techniques in astrophotography.

        Args:
            data: Input image data
            sigma: Gaussian blur sigma (larger = broader enhancement)
            strength: Enhancement strength (higher = more aggressive)

        Returns:
            Sharpened image data

        Example:
            >>> # Moderate sharpening
            >>> sharpened = enhancer.unsharp_mask(data, sigma=2.0, strength=1.5)
            >>>
            >>> # Aggressive sharpening
            >>> sharpened = enhancer.unsharp_mask(data, sigma=1.0, strength=2.5)
        """
        logger.info(f"Applying unsharp mask: sigma={sigma}, strength={strength}")

        # Create blurred version
        blurred = ndimage.gaussian_filter(data, sigma=sigma)

        # Unsharp mask: original + strength * (original - blurred)
        enhanced = data + strength * (data - blurred)

        # Clip to prevent artifacts
        enhanced = np.clip(enhanced, data.min(), data.max())

        return enhanced

    def enhance_stars(
        self,
        data: np.ndarray,
        sigma: float = 1.0,
        threshold: Optional[float] = None,
        strength: float = 2.0
    ) -> np.ndarray:
        """Enhance point sources (stars) in the image.

        Selectively enhances high-frequency components (stars) while leaving
        extended emission relatively unchanged.

        Args:
            data: Input image data
            sigma: Scale of star detection
            threshold: Brightness threshold for star detection (None = auto)
            strength: Enhancement strength

        Returns:
            Image with enhanced stars

        Example:
            >>> enhanced = enhancer.enhance_stars(
            ...     data,
            ...     sigma=1.0,
            ...     strength=2.0
            ... )
        """
        logger.info(f"Enhancing stars: sigma={sigma}, strength={strength}")

        # Detect stars using high-pass filter
        smoothed = ndimage.gaussian_filter(data, sigma=sigma)
        high_freq = data - smoothed

        # Auto-determine threshold if not provided
        if threshold is None:
            threshold = np.percentile(high_freq[high_freq > 0], 95)

        # Create mask for stars (bright point sources)
        star_mask = high_freq > threshold

        # Enhance only the star regions
        enhanced = data.copy()
        enhanced[star_mask] = data[star_mask] + strength * high_freq[star_mask]

        return np.clip(enhanced, 0, None)

    def create_luminance_mask(
        self,
        data: np.ndarray,
        low: float = 0.2,
        high: float = 0.8,
        smoothing: float = 5.0
    ) -> np.ndarray:
        """Create a luminance-based mask for selective enhancement.

        Luminance masks allow you to apply processing only to specific brightness
        ranges, commonly used to protect highlights or shadows.

        Args:
            data: Input image data
            low: Lower brightness threshold (0-1)
            high: Upper brightness threshold (0-1)
            smoothing: Gaussian smoothing sigma for soft edges

        Returns:
            Mask array with values in [0, 1]

        Example:
            >>> # Mask for midtones
            >>> mask = enhancer.create_luminance_mask(data, low=0.2, high=0.8)
            >>> # Apply processing only to masked regions
            >>> processed = data * (1 - mask) + enhanced * mask
        """
        logger.info(f"Creating luminance mask: low={low}, high={high}")

        # Normalize data to [0, 1]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)

        # Create mask
        mask = np.zeros_like(data)
        mask[(data_norm >= low) & (data_norm <= high)] = 1.0

        # Smooth mask for soft edges
        if smoothing > 0:
            mask = ndimage.gaussian_filter(mask, sigma=smoothing)

        return mask

    def apply_with_luminance_mask(
        self,
        data: np.ndarray,
        enhancement_func,
        low: float = 0.2,
        high: float = 0.8,
        smoothing: float = 5.0,
        **enhancement_kwargs
    ) -> np.ndarray:
        """Apply enhancement only to specific brightness range.

        Args:
            data: Input image data
            enhancement_func: Enhancement function to apply
            low: Lower brightness threshold
            high: Upper brightness threshold
            smoothing: Mask smoothing
            **enhancement_kwargs: Arguments for enhancement function

        Returns:
            Selectively enhanced image

        Example:
            >>> # Apply CLAHE only to midtones
            >>> enhanced = enhancer.apply_with_luminance_mask(
            ...     data,
            ...     enhancer.apply_clahe,
            ...     low=0.2,
            ...     high=0.8,
            ...     clip_limit=0.01
            ... )
        """
        # Create luminance mask
        mask = self.create_luminance_mask(data, low, high, smoothing)

        # Apply enhancement
        enhanced = enhancement_func(data, **enhancement_kwargs)

        # Blend based on mask
        result = data * (1 - mask) + enhanced * mask

        logger.info("Applied enhancement with luminance masking")
        return result

    def local_contrast_enhancement(
        self,
        data: np.ndarray,
        radius: int = 50,
        amount: float = 1.0
    ) -> np.ndarray:
        """Enhance local contrast using multiscale processing.

        Args:
            data: Input image data
            radius: Radius for local contrast calculation
            amount: Enhancement amount (1.0 = 100%)

        Returns:
            Contrast-enhanced image

        Example:
            >>> enhanced = enhancer.local_contrast_enhancement(
            ...     data,
            ...     radius=50,
            ...     amount=1.5
            ... )
        """
        logger.info(f"Applying local contrast enhancement: radius={radius}, amount={amount}")

        # Calculate local mean
        local_mean = ndimage.uniform_filter(data, size=radius)

        # Local contrast
        local_contrast = data - local_mean

        # Apply enhancement
        enhanced = data + amount * local_contrast

        return np.clip(enhanced, data.min(), data.max())
