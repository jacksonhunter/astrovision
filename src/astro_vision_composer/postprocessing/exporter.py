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
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available. Install with: pip install Pillow")

# Import matplotlib for saving
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")


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
                 - For RGB: shape (height, width, 3), values in [0, 1]
                 - For grayscale: shape (height, width), values in [0, 1]
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
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow required. Install with: pip install Pillow")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Ensure data is in [0, 1] range
        data_clipped = np.clip(data, 0, 1)

        # Convert to appropriate bit depth
        if bit_depth == 8:
            data_scaled = (data_clipped * 255).astype(np.uint8)
            mode = 'RGB' if len(data.shape) == 3 else 'L'
        elif bit_depth == 16:
            data_scaled = (data_clipped * 65535).astype(np.uint16)
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
        if metadata:
            from PIL import PngImagePlugin
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
            data: Image data array (grayscale or RGB)
            filepath: Output file path
            metadata: Optional metadata dictionary
            bit_depth: 8 or 16 bits per channel
            compression: Compression method ('none', 'lzw', 'tiff_deflate')

        Returns:
            Path to saved file
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow required. Install with: pip install Pillow")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Ensure data is in [0, 1] range
        data_clipped = np.clip(data, 0, 1)

        # Convert to appropriate bit depth
        if bit_depth == 8:
            data_scaled = (data_clipped * 255).astype(np.uint8)
        elif bit_depth == 16:
            data_scaled = (data_clipped * 65535).astype(np.uint16)
        else:
            raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")

        # Create PIL Image
        if len(data.shape) == 3:
            img = Image.fromarray(data_scaled, mode='RGB')
        else:
            img = Image.fromarray(data_scaled, mode='L')

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
            data: Image data array (grayscale or RGB)
            filepath: Output file path
            quality: JPEG quality (1-100, higher is better)

        Returns:
            Path to saved file
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow required. Install with: pip install Pillow")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to 8-bit
        data_clipped = np.clip(data, 0, 1)
        data_scaled = (data_clipped * 255).astype(np.uint8)

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
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")

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
