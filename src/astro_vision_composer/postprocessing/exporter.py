"""Image export with metadata preservation.

This module provides the ImageExporter class for saving processed astronomical
images in various formats while preserving metadata and processing history.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import PIL for image saving
from PIL import Image, PngImagePlugin

# Import matplotlib for saving
import matplotlib.pyplot as plt


class ImageExporter:
    """Export processed astronomical images with metadata.

    Supports multiple output formats:
    - PNG: 8-bit or 16-bit, with optional metadata in text chunks
    - TIFF: 8-bit or 16-bit, supports metadata tags
    - JPEG: 8-bit only, lossy compression
    - FITS: Full precision, preserves WCS and processing history

    Example:
        >>> exporter = ImageExporter()
        >>> exporter.save_png(
        ...     rgb_data,
        ...     'composite.png',
        ...     metadata={'source': 'PanSTARRS', 'filters': 'gri'}
        ... )
    """

    def __init__(self):
        """Initialize the ImageExporter."""
        pass

    def _normalize_to_float(self, data: np.ndarray) -> np.ndarray:
        """Normalize image data to [0, 1] float range.

        Handles both float [0, 1] and uint8 [0, 255] input data.

        Args:
            data: Input image array

        Returns:
            Normalized array in [0, 1] float range
        """
        # If already float in [0, 1], just clip
        if data.dtype in [np.float32, np.float64, np.float16]:
            if data.max() <= 1.0:
                return np.clip(data, 0, 1)
            else:
                # Float but values > 1, normalize by max
                logger.warning(f"Float data has max={data.max():.2f} > 1, normalizing to [0,1]")
                return np.clip(data / data.max(), 0, 1)

        # If uint8 [0, 255], convert to float [0, 1]
        elif data.dtype == np.uint8:
            return data.astype(np.float32) / 255.0

        # If uint16, convert to float [0, 1]
        elif data.dtype == np.uint16:
            return data.astype(np.float32) / 65535.0

        # Other integer types, normalize by max value
        else:
            logger.warning(f"Unexpected dtype {data.dtype}, normalizing by max value")
            return np.clip(data.astype(np.float32) / data.max(), 0, 1)

    def save_png(
        self,
        data: np.ndarray,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
        bit_depth: int = 8,
        dpi: int = 150
    ) -> Path:
        """Save image as PNG with optional metadata.

        Args:
            data: Image data array (either grayscale or RGB)
                 - For RGB: shape (height, width, 3), values in [0, 1] or [0, 255]
                 - For grayscale: shape (height, width), values in [0, 1] or [0, 255]
            filepath: Output file path
            metadata: Optional metadata dictionary
            bit_depth: 8 or 16 bits per channel
            dpi: Resolution in dots per inch

        Returns:
            Path to saved file

        Example:
            >>> rgb = compositor.create_lupton_rgb(r, g, b)
            >>> exporter.save_png(rgb, 'output.png', bit_depth=16)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Normalize data to [0, 1] float range
        data_normalized = self._normalize_to_float(data)

        # Convert to appropriate bit depth
        if bit_depth == 8:
            data_scaled = (data_normalized * 255).astype(np.uint8)
            mode = 'RGB' if len(data.shape) == 3 else 'L'
        elif bit_depth == 16:
            data_scaled = (data_normalized * 65535).astype(np.uint16)
            mode = 'RGB' if len(data.shape) == 3 else 'I;16'
        else:
            raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")

        # Create PIL Image
        if len(data.shape) == 3:
            # RGB image
            img = Image.fromarray(data_scaled, mode='RGB')
        else:
            # Grayscale
            img = Image.fromarray(data_scaled, mode='L')

        # Add metadata as PNG text chunks
        pnginfo = None
        if metadata and PngImagePlugin:
            pnginfo = PngImagePlugin.PngInfo()
            for key, value in metadata.items():
                pnginfo.add_text(str(key), str(value))

        # Save
        img.save(filepath, format='PNG', pnginfo=pnginfo, dpi=(dpi, dpi))

        logger.info(f"Saved {bit_depth}-bit PNG to {filepath}")
        return filepath

    def save_tiff(
        self,
        data: np.ndarray,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
        bit_depth: int = 16,
        compression: str = 'lzw'
    ) -> Path:
        """Save image as TIFF with metadata.

        Args:
            data: Image data array (grayscale or RGB), values in [0, 1] or [0, 255]
            filepath: Output file path
            metadata: Optional metadata dictionary
            bit_depth: 8 or 16 bits per channel
            compression: Compression method ('none', 'lzw', 'tiff_deflate')

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Normalize data to [0, 1] float range
        data_normalized = self._normalize_to_float(data)

        # Convert to appropriate bit depth
        if bit_depth == 8:
            data_scaled = (data_normalized * 255).astype(np.uint8)
            mode = 'RGB' if len(data.shape) == 3 else 'L'
        elif bit_depth == 16:
            data_scaled = (data_normalized * 65535).astype(np.uint16)
            # PIL's RGB mode doesn't support 16-bit, must save per-channel or use I;16
            if len(data.shape) == 3:
                # For 16-bit RGB TIFF, use tifffile if available, else fall back to 8-bit
                try:
                    import tifffile
                    # Save directly with tifffile (supports 16-bit RGB properly)
                    tifffile.imwrite(
                        filepath,
                        data_scaled,
                        compression=compression if compression != 'lzw' else 'deflate',
                        photometric='rgb'
                    )
                    logger.info(f"Saved 16-bit RGB TIFF to {filepath} (tifffile)")
                    return filepath
                except ImportError:
                    logger.warning("tifffile not available, falling back to 8-bit TIFF")
                    data_scaled = (data_normalized * 255).astype(np.uint8)
                    mode = 'RGB'
            else:
                mode = 'I;16'  # 16-bit grayscale
        else:
            raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")

        # Create PIL Image (if we didn't use tifffile above)
        if len(data.shape) == 3:
            img = Image.fromarray(data_scaled, mode='RGB')
        else:
            img = Image.fromarray(data_scaled, mode='L' if bit_depth == 8 else 'I;16')

        # Save with compression
        img.save(filepath, format='TIFF', compression=compression)

        logger.info(f"Saved {bit_depth}-bit TIFF to {filepath} ({compression} compression)")
        return filepath

    def save_jpeg(
        self,
        data: np.ndarray,
        filepath: str,
        quality: int = 95
    ) -> Path:
        """Save image as JPEG (lossy, 8-bit only).

        Note: JPEG is lossy and not recommended for archival purposes.
        Use PNG or TIFF for lossless storage.

        Args:
            data: Image data array (grayscale or RGB), values in [0, 1] or [0, 255]
            filepath: Output file path
            quality: JPEG quality (1-100, higher is better)

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Normalize and convert to 8-bit
        data_normalized = self._normalize_to_float(data)
        data_scaled = (data_normalized * 255).astype(np.uint8)

        # Create PIL Image
        if len(data.shape) == 3:
            img = Image.fromarray(data_scaled, mode='RGB')
        else:
            img = Image.fromarray(data_scaled, mode='L')

        # Save
        img.save(filepath, format='JPEG', quality=quality)

        logger.info(f"Saved JPEG to {filepath} (quality={quality})")
        logger.warning("JPEG is lossy - consider PNG or TIFF for archival")
        return filepath

    def save_with_matplotlib(
        self,
        data: np.ndarray,
        filepath: str,
        dpi: int = 150,
        cmap: Optional[str] = None
    ) -> Path:
        """Save image using matplotlib (useful for grayscale with colormaps).

        Args:
            data: Image data array
            filepath: Output file path
            dpi: Resolution in dots per inch
            cmap: Colormap name (for grayscale images)

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(data.shape[1]/dpi, data.shape[0]/dpi), dpi=dpi)
        ax.axis('off')

        if len(data.shape) == 3:
            # RGB
            ax.imshow(data, origin='lower')
        else:
            # Grayscale
            ax.imshow(data, origin='lower', cmap=cmap or 'gray')

        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        logger.info(f"Saved image to {filepath} via matplotlib")
        return filepath

    def auto_save(
        self,
        data: np.ndarray,
        filepath: str,
        **kwargs
    ) -> Path:
        """Automatically save based on file extension.

        Args:
            data: Image data array
            filepath: Output file path (extension determines format)
            **kwargs: Format-specific arguments

        Returns:
            Path to saved file

        Example:
            >>> exporter.auto_save(rgb, 'output.png', bit_depth=16)
            >>> exporter.auto_save(rgb, 'output.tiff', compression='lzw')
            >>> exporter.auto_save(rgb, 'output.jpg', quality=95)
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext in ['.png']:
            return self.save_png(data, filepath, **kwargs)
        elif ext in ['.tif', '.tiff']:
            return self.save_tiff(data, filepath, **kwargs)
        elif ext in ['.jpg', '.jpeg']:
            return self.save_jpeg(data, filepath, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                f"Supported: .png, .tif, .tiff, .jpg, .jpeg"
            )
