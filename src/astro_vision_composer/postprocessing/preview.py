"""Preview and thumbnail generation for astronomical images.

This module provides the PreviewGenerator class for creating quick low-resolution
previews and thumbnails of astronomical images.
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import PIL for thumbnails
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available. Install with: pip install Pillow")


class PreviewGenerator:
    """Generate previews and thumbnails from astronomical images.

    Provides methods for creating quick, low-resolution versions of images
    for faster inspection and web display.

    Example:
        >>> generator = PreviewGenerator()
        >>> # Create thumbnail
        >>> thumbnail = generator.create_thumbnail(large_image, max_size=256)
    """

    def __init__(self):
        """Initialize the PreviewGenerator."""
        pass

    def create_thumbnail(
        self,
        data: np.ndarray,
        max_size: int = 256,
        method: str = 'bilinear'
    ) -> np.ndarray:
        """Create a thumbnail from image data.

        Args:
            data: Input image data (grayscale or RGB)
            max_size: Maximum dimension in pixels
            method: Resampling method ('bilinear', 'nearest', 'area')

        Returns:
            Thumbnail array

        Example:
            >>> thumb = generator.create_thumbnail(image, max_size=256)
            >>> print(f"Thumbnail shape: {thumb.shape}")
        """
        # Calculate new dimensions
        if len(data.shape) == 2:
            height, width = data.shape
        elif len(data.shape) == 3:
            height, width, channels = data.shape
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

        # Calculate scale factor
        scale = min(max_size / height, max_size / width)

        if scale >= 1.0:
            logger.debug("Image already smaller than max_size, no resize needed")
            return data

        new_height = int(height * scale)
        new_width = int(width * scale)

        logger.info(f"Creating thumbnail: {width}x{height} → {new_width}x{new_height}")

        # Resize using scipy or PIL
        try:
            from scipy import ndimage
            if method == 'bilinear':
                thumbnail = ndimage.zoom(data, (scale, scale) if len(data.shape) == 2 else (scale, scale, 1), order=1)
            elif method == 'nearest':
                thumbnail = ndimage.zoom(data, (scale, scale) if len(data.shape) == 2 else (scale, scale, 1), order=0)
            elif method == 'area':
                # Area method not directly available in scipy, use bilinear
                thumbnail = ndimage.zoom(data, (scale, scale) if len(data.shape) == 2 else (scale, scale, 1), order=1)
            else:
                raise ValueError(f"Unknown method: {method}")

        except ImportError:
            logger.warning("scipy not available, using simple slicing")
            stride = int(1 / scale)
            if len(data.shape) == 2:
                thumbnail = data[::stride, ::stride]
            else:
                thumbnail = data[::stride, ::stride, :]

        return thumbnail

    def generate_preview(
        self,
        data: np.ndarray,
        target_size: Tuple[int, int] = (512, 512),
        auto_stretch: bool = True
    ) -> np.ndarray:
        """Generate a preview with automatic stretching.

        Args:
            data: Input image data
            target_size: Target (width, height) in pixels
            auto_stretch: Apply automatic contrast stretch

        Returns:
            Preview array in [0, 1] range

        Example:
            >>> preview = generator.generate_preview(
            ...     raw_data,
            ...     target_size=(512, 512),
            ...     auto_stretch=True
            ... )
        """
        # Resize to target
        max_dim = max(target_size)
        resized = self.create_thumbnail(data, max_size=max_dim)

        if not auto_stretch:
            return resized

        # Auto-stretch using percentiles
        logger.debug("Applying auto-stretch to preview")
        vmin, vmax = np.percentile(resized[np.isfinite(resized)], [1, 99])
        stretched = np.clip((resized - vmin) / (vmax - vmin + 1e-10), 0, 1)

        return stretched

    def save_thumbnail(
        self,
        data: np.ndarray,
        filepath: str,
        max_size: int = 256,
        quality: int = 85
    ) -> Path:
        """Create and save a thumbnail image.

        Args:
            data: Input image data (should be in [0, 1] range for RGB)
            filepath: Output file path
            max_size: Maximum dimension
            quality: JPEG quality (1-100)

        Returns:
            Path to saved thumbnail

        Raises:
            ImportError: If PIL/Pillow is not installed

        Example:
            >>> generator.save_thumbnail(
            ...     rgb_image,
            ...     'thumbnail.jpg',
            ...     max_size=256
            ... )
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow required. Install with: pip install Pillow")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create thumbnail
        thumbnail = self.create_thumbnail(data, max_size=max_size)

        # Ensure data is in [0, 1] then convert to [0, 255]
        thumbnail_clipped = np.clip(thumbnail, 0, 1)
        thumbnail_8bit = (thumbnail_clipped * 255).astype(np.uint8)

        # Create PIL Image
        if len(thumbnail.shape) == 2:
            img = Image.fromarray(thumbnail_8bit, mode='L')
        elif len(thumbnail.shape) == 3:
            img = Image.fromarray(thumbnail_8bit, mode='RGB')
        else:
            raise ValueError(f"Unexpected array shape: {thumbnail.shape}")

        # Save
        img.save(filepath, quality=quality)

        logger.info(f"Saved thumbnail to {filepath}")
        return filepath

    def create_multi_resolution(
        self,
        data: np.ndarray,
        sizes: list = [128, 256, 512, 1024]
    ) -> dict:
        """Create multiple resolution versions of an image.

        Args:
            data: Input image data
            sizes: List of maximum dimensions

        Returns:
            Dictionary mapping sizes to thumbnail arrays

        Example:
            >>> previews = generator.create_multi_resolution(
            ...     image,
            ...     sizes=[128, 256, 512]
            ... )
            >>> small = previews[128]
            >>> medium = previews[256]
            >>> large = previews[512]
        """
        logger.info(f"Creating {len(sizes)} resolution versions")

        previews = {}
        for size in sorted(sizes):
            previews[size] = self.create_thumbnail(data, max_size=size)

        return previews

    def quick_display_info(
        self,
        data: np.ndarray
    ) -> dict:
        """Generate quick statistics for display purposes.

        Args:
            data: Input image data

        Returns:
            Dictionary with statistics

        Example:
            >>> info = generator.quick_display_info(image)
            >>> print(f"Min: {info['min']}, Max: {info['max']}, Mean: {info['mean']}")
        """
        finite_data = data[np.isfinite(data)]

        info = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min': float(np.min(finite_data)) if finite_data.size > 0 else None,
            'max': float(np.max(finite_data)) if finite_data.size > 0 else None,
            'mean': float(np.mean(finite_data)) if finite_data.size > 0 else None,
            'median': float(np.median(finite_data)) if finite_data.size > 0 else None,
            'std': float(np.std(finite_data)) if finite_data.size > 0 else None,
            'percentile_1': float(np.percentile(finite_data, 1)) if finite_data.size > 0 else None,
            'percentile_99': float(np.percentile(finite_data, 99)) if finite_data.size > 0 else None,
            'has_nan': bool(np.any(np.isnan(data))),
            'has_inf': bool(np.any(np.isinf(data))),
        }

        return info
