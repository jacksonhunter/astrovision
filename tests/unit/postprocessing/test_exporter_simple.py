"""Simplified unit tests for ImageExporter class."""

import pytest
import numpy as np
from pathlib import Path

from astro_vision_composer.postprocessing.exporter import ImageExporter


@pytest.fixture
def sample_rgb():
    """Create sample RGB image."""
    return np.random.rand(100, 100, 3).astype(np.float32)


class TestImageExporter:
    """Test ImageExporter class."""

    def test_init(self):
        """Test exporter initialization."""
        exporter = ImageExporter()
        assert exporter is not None

    def test_save_png(self, sample_rgb, tmp_path):
        """Test PNG save."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.png"

        exporter.save_png(sample_rgb, str(output_path))

        assert output_path.exists()

    def test_save_tiff(self, sample_rgb, tmp_path):
        """Test TIFF save."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.tiff"

        exporter.save_tiff(sample_rgb, str(output_path))

        assert output_path.exists()

    def test_save_jpeg(self, sample_rgb, tmp_path):
        """Test JPEG save."""
        exporter = ImageExporter()
        output_path = tmp_path / "test.jpg"

        exporter.save_jpeg(sample_rgb, str(output_path), quality=95)

        assert output_path.exists()

    def test_auto_save(self, sample_rgb, tmp_path):
        """Test auto_save with automatic format detection."""
        exporter = ImageExporter()

        # PNG
        png_path = tmp_path / "test_auto.png"
        exporter.auto_save(sample_rgb, str(png_path))
        assert png_path.exists()

        # TIFF
        tiff_path = tmp_path / "test_auto.tiff"
        exporter.auto_save(sample_rgb, str(tiff_path))
        assert tiff_path.exists()

    def test_various_shapes(self, tmp_path):
        """Test exporting images of various shapes."""
        exporter = ImageExporter()

        # Small RGB
        small = np.random.rand(10, 10, 3)
        exporter.save_png(small, str(tmp_path / "small.png"))

        # Large RGB
        large = np.random.rand(500, 500, 3)
        exporter.save_png(large, str(tmp_path / "large.png"))

        # Grayscale
        gray = np.random.rand(100, 100)
        exporter.save_png(gray, str(tmp_path / "gray.png"))

        assert (tmp_path / "small.png").exists()
        assert (tmp_path / "large.png").exists()
        assert (tmp_path / "gray.png").exists()
