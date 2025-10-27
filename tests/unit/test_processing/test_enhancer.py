"""Unit tests for Enhancer class."""

import pytest
import numpy as np


class TestEnhancer:
    """Test Enhancer for advanced image enhancement."""

    def test_clahe_enhancement(self, enhancer):
        """Test CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        data = np.random.rand(100, 100)

        enhanced = enhancer.apply_clahe(data, clip_limit=2.0, tile_size=8)

        assert enhanced is not None
        assert enhanced.shape == data.shape
        assert np.all(np.isfinite(enhanced))

    def test_unsharp_mask(self, enhancer):
        """Test unsharp masking."""
        # Create image with some features
        data = np.random.rand(100, 100) * 0.1
        data[40:60, 40:60] = 0.8  # Bright square

        enhanced = enhancer.apply_unsharp_mask(data, radius=2.0, amount=1.5)

        assert enhanced is not None
        assert enhanced.shape == data.shape
        # Unsharp mask should enhance edges
        assert enhanced.max() >= data.max()

    def test_star_highlighting(self, enhancer):
        """Test star highlighting enhancement."""
        # Create image with point sources
        data = np.random.rand(100, 100) * 0.1
        data[25, 25] = 1.0  # Star 1
        data[75, 75] = 1.0  # Star 2

        enhanced = enhancer.enhance_stars(data, sigma=2.0, amount=2.0)

        assert enhanced is not None
        assert enhanced.shape == data.shape

    def test_luminance_masking(self, enhancer):
        """Test luminance-based masking."""
        data = np.random.rand(100, 100)

        # Create luminance mask (protect bright regions)
        mask = enhancer.create_luminance_mask(data, threshold=0.7)

        assert mask is not None
        assert mask.shape == data.shape
        assert mask.dtype == bool or (mask.min() >= 0 and mask.max() <= 1)

    def test_enhancement_preserves_shape(self, enhancer):
        """Test that enhancement preserves array shape."""
        for shape in [(50, 50), (100, 200), (256, 256)]:
            data = np.random.rand(*shape)
            enhanced = enhancer.apply_clahe(data)
            assert enhanced.shape == shape

    def test_clahe_with_different_parameters(self, enhancer):
        """Test CLAHE with various parameters."""
        data = np.random.rand(100, 100)

        for clip_limit in [1.0, 2.0, 4.0]:
            for tile_size in [4, 8, 16]:
                enhanced = enhancer.apply_clahe(
                    data, clip_limit=clip_limit, tile_size=tile_size
                )
                assert enhanced is not None
                assert 0 <= enhanced.min() <= enhanced.max() <= 1

    def test_unsharp_mask_parameters(self, enhancer):
        """Test unsharp mask with different parameters."""
        data = np.random.rand(100, 100)

        for radius in [1.0, 2.0, 5.0]:
            for amount in [0.5, 1.0, 2.0]:
                enhanced = enhancer.apply_unsharp_mask(
                    data, radius=radius, amount=amount
                )
                assert enhanced is not None

    def test_blank_image(self, enhancer):
        """Test enhancement of blank image."""
        data = np.zeros((100, 100))

        # Should handle gracefully
        enhanced = enhancer.apply_clahe(data)
        assert enhanced is not None

    def test_uniform_image(self, enhancer):
        """Test enhancement of uniform image."""
        data = np.ones((100, 100)) * 0.5

        enhanced = enhancer.apply_clahe(data)
        assert enhanced is not None

    @pytest.mark.requires_noirlab_data
    def test_with_real_fits_data(self, enhancer, fits_loader, normalizer, edu008_data):
        """Test enhancement on real FITS data."""
        fits_file = edu008_data['fits_files'][0]
        fits_data = fits_loader.load(fits_file)

        # Normalize first
        normalized = normalizer.normalize(fits_data.data, method='zscale')

        # Test various enhancements
        clahe = enhancer.apply_clahe(normalized)
        unsharp = enhancer.apply_unsharp_mask(normalized)

        assert clahe is not None
        assert unsharp is not None
        assert clahe.shape == normalized.shape
        assert unsharp.shape == normalized.shape

    def test_enhancement_increases_contrast(self, enhancer):
        """Test that CLAHE increases local contrast."""
        # Create low-contrast image
        data = np.random.rand(100, 100) * 0.2 + 0.4  # Range 0.4-0.6

        enhanced = enhancer.apply_clahe(data, clip_limit=3.0)

        # Enhanced image should have wider range
        assert enhanced.max() - enhanced.min() > data.max() - data.min()
