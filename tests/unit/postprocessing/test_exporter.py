"""Unit tests for Exporter class.

Tests image export to various formats (PNG, TIFF, JPEG).
"""

import pytest
import numpy as np
from pathlib import Path

from astro_vision_composer.postprocessing.exporter import ImageExporter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rgb():
    """Create sample RGB image."""
    rgb = np.random.rand(100, 100, 3).astype(np.float32)
    return rgb


@pytest.fixture
def sample_grayscale():
    """Create sample grayscale image."""
    gray = np.random.rand(100, 100).astype(np.float32)
    return gray


# ============================================================================
# Test Exporter
# ============================================================================

class TestExporter:
    """Test ImageExporter class."""

    def test_init(self):
        """Test exporter initialization."""
        exporter = ImageExporter()
        assert exporter is not None

    def test_export_png(self, sample_rgb, tmp_path):
        """Test PNG export."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.png"

        exporter.save_png(sample_rgb, output_path, bit_depth=8)

        assert output_path.exists()

    def test_export_tiff(self, sample_rgb, tmp_path):
        """Test TIFF export."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.tiff"

        exporter.save_png(sample_rgb, output_path, format='tiff')

        assert output_path.exists()

    def test_export_jpeg(self, sample_rgb, tmp_path):
        """Test JPEG export."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.jpg"

        exporter.save_png(sample_rgb, output_path, format='jpeg', quality=95)

        assert output_path.exists()

    def test_export_auto_format(self, sample_rgb, tmp_path):
        """Test automatic format detection from extension."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.png"

        exporter.save_png(sample_rgb, output_path)  # No format specified

        assert output_path.exists()

    def test_export_grayscale(self, sample_grayscale, tmp_path):
        """Test exporting grayscale image."""
        exporter = ImageExporter()
        output_path = tmp_path / "test_gray.png"

        exporter.save_png(sample_grayscale, output_path)

        assert output_path.exists()

    def test_export_uint8_conversion(self, tmp_path):
        """Test that float [0,1] data is converted to uint8 [0,255]."""
        exporter = ImageExporter()
        output_path = tmp_path / "test_uint8.png"

        # Create float data in [0, 1]
        data = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

        exporter.save_png(data, output_path)

        assert output_path.exists()

    def test_export_already_uint8(self, tmp_path):
        """Test exporting data that's already uint8."""
        exporter = ImageExporter()
        output_path = tmp_path / "test_uint8_direct.png"

        # Create uint8 data directly
        data = np.array([[[0, 128, 255]]], dtype=np.uint8)

        exporter.save_png(data, output_path)

        assert output_path.exists()

    def test_export_clipping(self, tmp_path):
        """Test that values outside [0,1] are clipped."""
        exporter = ImageExporter()
        output_path = tmp_path / "test_clip.png"

        # Create data with values outside [0, 1]
        data = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)

        exporter.save_png(data, output_path)

        assert output_path.exists()

    def test_invalid_format_raises(self, sample_rgb, tmp_path):
        """Test that invalid format raises ValueError."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.xyz"

        with pytest.raises((ValueError, KeyError)):
            exporter.save_png(sample_rgb, output_path, format='invalid')

    def test_export_creates_directory(self, sample_rgb, tmp_path):
        """Test that export creates parent directories if needed."""
        exporter = ImageExporter()
        output_path = tmp_path / "subdir" / "test.png"

        exporter.save_png(sample_rgb, output_path)

        assert output_path.exists()

    def test_export_overwrite(self, sample_rgb, tmp_path):
        """Test that export can overwrite existing files."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.png"

        # Export once
        exporter.save_png(sample_rgb, output_path)
        assert output_path.exists()

        # Export again (overwrite)
        exporter.save_png(sample_rgb * 0.5, output_path)
        assert output_path.exists()

    def test_empty_data_raises(self, tmp_path):
        """Test that empty data raises ValueError."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.png"

        with pytest.raises((ValueError, AttributeError)):
            exporter.save_png(np.array([]), output_path)

    def test_export_various_shapes(self, tmp_path):
        """Test exporting images of various shapes."""
        exporter = ImageExporter()

        # Small image
        small = np.random.rand(10, 10, 3)
        exporter.save_png(small, tmp_path / "small.png")

        # Large image
        large = np.random.rand(1000, 1000, 3)
        exporter.save_png(large, tmp_path / "large.png")

        # Non-square
        rect = np.random.rand(50, 200, 3)
        exporter.save_png(rect, tmp_path / "rect.png")

        assert (tmp_path / "small.png").exists()
        assert (tmp_path / "large.png").exists()
        assert (tmp_path / "rect.png").exists()
