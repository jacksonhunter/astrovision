"""Unit tests for ImageExporter class."""

import pytest
import numpy as np
from pathlib import Path


class TestImageExporter:
    """Test ImageExporter for saving images in various formats."""

    def test_save_png(self, image_exporter, temp_output_dir):
        """Test saving image as PNG."""
        rgb = np.random.rand(100, 100, 3)

        output_file = temp_output_dir / "test.png"
        image_exporter.auto_save(rgb, output_file, format='png')

        assert output_file.exists()
        assert output_file.suffix == '.png'

    def test_save_tiff(self, image_exporter, temp_output_dir):
        """Test saving image as TIFF."""
        rgb = np.random.rand(100, 100, 3)

        output_file = temp_output_dir / "test.tiff"
        image_exporter.auto_save(rgb, output_file, format='tiff')

        assert output_file.exists()

    def test_save_jpeg(self, image_exporter, temp_output_dir):
        """Test saving image as JPEG."""
        rgb = np.random.rand(100, 100, 3)

        output_file = temp_output_dir / "test.jpg"
        image_exporter.auto_save(rgb, output_file, format='jpeg', quality=95)

        assert output_file.exists()

    def test_save_with_metadata(self, image_exporter, history_tracker, temp_output_dir):
        """Test saving image with processing history."""
        rgb = np.random.rand(100, 100, 3)

        # Create some history
        history_tracker.record('normalize', {'method': 'zscale'})
        history_tracker.record('stretch', {'method': 'asinh'})

        output_file = temp_output_dir / "test_with_metadata.png"
        image_exporter.auto_save(rgb, output_file, history=history_tracker.get_history())

        assert output_file.exists()

    def test_auto_format_detection(self, image_exporter, temp_output_dir):
        """Test automatic format detection from filename."""
        rgb = np.random.rand(100, 100, 3)

        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            output_file = temp_output_dir / f"test{ext}"
            image_exporter.auto_save(rgb, output_file)
            assert output_file.exists()

    def test_8bit_conversion(self, image_exporter, temp_output_dir):
        """Test conversion to 8-bit for standard image formats."""
        # Float data in [0, 1]
        rgb = np.random.rand(100, 100, 3)

        output_file = temp_output_dir / "test_8bit.png"
        image_exporter.auto_save(rgb, output_file, bit_depth=8)

        assert output_file.exists()

    def test_16bit_tiff(self, image_exporter, temp_output_dir):
        """Test saving 16-bit TIFF for higher precision."""
        rgb = np.random.rand(100, 100, 3)

        output_file = temp_output_dir / "test_16bit.tiff"
        image_exporter.auto_save(rgb, output_file, bit_depth=16)

        assert output_file.exists()

    def test_grayscale_export(self, image_exporter, temp_output_dir):
        """Test exporting grayscale image."""
        gray = np.random.rand(100, 100)  # 2D array

        output_file = temp_output_dir / "test_gray.png"
        image_exporter.auto_save(gray, output_file)

        assert output_file.exists()

    def test_overwrite_protection(self, image_exporter, temp_output_dir):
        """Test overwrite protection."""
        rgb = np.random.rand(100, 100, 3)

        output_file = temp_output_dir / "test.png"

        # Save once
        image_exporter.auto_save(rgb, output_file)
        assert output_file.exists()

        # Save again (should either overwrite or raise error)
        # Behavior depends on implementation
        image_exporter.auto_save(rgb, output_file, overwrite=True)
        assert output_file.exists()

    def test_jpeg_quality_parameter(self, image_exporter, temp_output_dir):
        """Test JPEG quality parameter."""
        rgb = np.random.rand(100, 100, 3)

        file_high = temp_output_dir / "high_quality.jpg"
        file_low = temp_output_dir / "low_quality.jpg"

        image_exporter.auto_save(rgb, file_high, format='jpeg', quality=95)
        image_exporter.auto_save(rgb, file_low, format='jpeg', quality=50)

        # High quality should result in larger file
        assert file_high.stat().st_size > file_low.stat().st_size

    def test_value_clipping(self, image_exporter, temp_output_dir):
        """Test that values outside [0,1] are clipped."""
        rgb = np.random.rand(100, 100, 3) * 2 - 0.5  # Range [-0.5, 1.5]

        output_file = temp_output_dir / "test_clipped.png"

        # Should clip to [0, 1] without error
        image_exporter.auto_save(rgb, output_file)
        assert output_file.exists()

    def test_nan_handling(self, image_exporter, temp_output_dir):
        """Test handling of NaN values."""
        rgb = np.random.rand(100, 100, 3)
        rgb[50:60, 50:60, :] = np.nan

        output_file = temp_output_dir / "test_nan.png"

        # Should either handle NaNs or raise informative error
        try:
            image_exporter.auto_save(rgb, output_file)
            # If it succeeds, file should exist
            assert output_file.exists()
        except ValueError:
            # Or it might raise an error - also acceptable
            pass

    @pytest.mark.requires_noirlab_data
    def test_full_pipeline_export(self, image_exporter, compositor, fits_loader,
                                   normalizer, stretcher, edu008_data, temp_output_dir):
        """Test exporting result of full pipeline."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        # Process and composite
        channels = []
        for fits_file in edu008_data['fits_files'][:3]:
            fits_data = fits_loader.load(fits_file)
            normalized = normalizer.normalize(fits_data.data, method='zscale')
            stretched = stretcher.stretch(normalized, method='asinh', a=0.1)
            channels.append(stretched)

        rgb = compositor.create_lupton_rgb(
            channels[2], channels[1], channels[0],
            stretch=0.5, Q=8
        )

        # Export
        output_file = temp_output_dir / "composite.png"
        image_exporter.auto_save(rgb, output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 1000  # Should be reasonable size
