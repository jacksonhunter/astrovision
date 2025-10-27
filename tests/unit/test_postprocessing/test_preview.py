"""Unit tests for PreviewGenerator class."""

import pytest
import numpy as np


class TestPreviewGenerator:
    """Test PreviewGenerator for quick preview creation."""

    def test_generate_thumbnail(self, preview_generator):
        """Test thumbnail generation."""
        rgb = np.random.rand(1000, 1000, 3)

        thumbnail = preview_generator.create_thumbnail(rgb, size=(256, 256))

        assert thumbnail is not None
        assert thumbnail.shape[0] <= 256
        assert thumbnail.shape[1] <= 256
        assert thumbnail.shape[2] == 3

    def test_generate_preview(self, preview_generator):
        """Test preview generation at lower resolution."""
        rgb = np.random.rand(2000, 2000, 3)

        preview = preview_generator.create_preview(rgb, scale=0.25)

        assert preview is not None
        assert preview.shape[0] == 500  # 2000 * 0.25
        assert preview.shape[1] == 500

    def test_preserve_aspect_ratio(self, preview_generator):
        """Test that aspect ratio is preserved."""
        # 2:1 aspect ratio
        rgb = np.random.rand(1000, 2000, 3)

        preview = preview_generator.create_thumbnail(rgb, size=(256, 256),
                                                     preserve_aspect=True)

        # Should maintain 2:1 ratio, so height should be half width
        assert preview.shape[1] / preview.shape[0] == pytest.approx(2.0, rel=0.1)

    def test_multiple_sizes(self, preview_generator):
        """Test generating multiple preview sizes."""
        rgb = np.random.rand(1000, 1000, 3)

        sizes = [(128, 128), (256, 256), (512, 512)]

        previews = preview_generator.create_multiple_previews(rgb, sizes=sizes)

        assert len(previews) == 3
        for i, size in enumerate(sizes):
            assert previews[i].shape[0] <= size[0]
            assert previews[i].shape[1] <= size[1]

    def test_grayscale_preview(self, preview_generator):
        """Test preview generation for grayscale image."""
        gray = np.random.rand(1000, 1000)  # 2D array

        preview = preview_generator.create_thumbnail(gray, size=(256, 256))

        assert preview is not None
        assert preview.ndim == 2  # Still grayscale

    def test_fast_preview_mode(self, preview_generator):
        """Test fast preview mode (nearest neighbor)."""
        rgb = np.random.rand(2000, 2000, 3)

        fast_preview = preview_generator.create_preview(
            rgb, scale=0.5, interpolation='nearest'
        )

        assert fast_preview is not None
        assert fast_preview.shape[0] == 1000
        assert fast_preview.shape[1] == 1000

    def test_high_quality_preview(self, preview_generator):
        """Test high quality preview mode (bicubic/lanczos)."""
        rgb = np.random.rand(1000, 1000, 3)

        hq_preview = preview_generator.create_preview(
            rgb, scale=0.5, interpolation='lanczos'
        )

        assert hq_preview is not None

    def test_value_range_preservation(self, preview_generator):
        """Test that preview maintains value range."""
        rgb = np.random.rand(1000, 1000, 3)

        preview = preview_generator.create_thumbnail(rgb, size=(256, 256))

        assert preview.min() >= 0
        assert preview.max() <= 1

    def test_extreme_downscaling(self, preview_generator):
        """Test extreme downscaling (large image to tiny preview)."""
        rgb = np.random.rand(4000, 4000, 3)

        tiny = preview_generator.create_thumbnail(rgb, size=(64, 64))

        assert tiny is not None
        assert tiny.shape[0] <= 64
        assert tiny.shape[1] <= 64

    @pytest.mark.requires_noirlab_data
    def test_preview_of_real_composite(self, preview_generator, compositor,
                                       fits_loader, normalizer, stretcher, edu008_data):
        """Test preview generation of real composite."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        # Create composite
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

        # Generate preview
        preview = preview_generator.create_thumbnail(rgb, size=(512, 512))

        assert preview is not None
        assert preview.shape[0] <= 512
        assert preview.shape[1] <= 512
        assert 0 <= preview.min() <= preview.max() <= 1

    def test_save_preview(self, preview_generator, temp_output_dir):
        """Test saving preview to file."""
        rgb = np.random.rand(1000, 1000, 3)

        preview_file = temp_output_dir / "preview.png"

        preview_generator.save_thumbnail(
            rgb, preview_file, size=(256, 256)
        )

        assert preview_file.exists()
