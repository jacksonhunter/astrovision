"""Unit tests for Normalizer class."""

import pytest
import numpy as np


class TestNormalizer:
    """Test Normalizer interval selection methods."""

    def test_minmax_normalization(self, normalizer):
        """Test MinMax normalization."""
        data = np.random.rand(100, 100) * 100

        normalized = normalizer.normalize(data, method='minmax')

        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert normalized.min() < 0.1  # Should be close to 0
        assert normalized.max() > 0.9  # Should be close to 1

    def test_percentile_normalization(self, normalizer):
        """Test percentile-based normalization."""
        data = np.random.rand(100, 100) * 100

        normalized = normalizer.normalize(data, method='percentile', vmin=5, vmax=95)

        # Should clip values outside 5-95 percentile range
        assert normalized is not None
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_zscale_normalization(self, normalizer):
        """Test ZScale normalization."""
        # Create astronomical-like data with few bright sources
        data = np.random.normal(100, 10, (100, 100))
        data[20:30, 20:30] += 500  # Bright source

        normalized = normalizer.normalize(data, method='zscale')

        assert normalized is not None
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_manual_normalization(self, normalizer):
        """Test manual vmin/vmax normalization."""
        data = np.random.rand(100, 100) * 100

        normalized = normalizer.normalize(data, method='manual', vmin=25, vmax=75)

        # Values outside vmin-vmax should be clipped
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalization_preserves_shape(self, normalizer):
        """Test that normalization preserves array shape."""
        for shape in [(50, 50), (100, 100), (200, 150)]:
            data = np.random.rand(*shape)
            normalized = normalizer.normalize(data, method='minmax')
            assert normalized.shape == shape

    def test_blank_image(self, normalizer):
        """Test normalization of blank (all zeros) image."""
        data = np.zeros((100, 100))

        # Should handle gracefully
        normalized = normalizer.normalize(data, method='minmax')

        assert normalized is not None
        # All values should still be 0 (or handled appropriately)

    def test_uniform_image(self, normalizer):
        """Test normalization of uniform (constant value) image."""
        data = np.ones((100, 100)) * 50

        normalized = normalizer.normalize(data, method='minmax')

        # Should handle constant values gracefully
        assert normalized is not None

    def test_negative_values(self, normalizer):
        """Test normalization with negative pixel values."""
        data = np.random.normal(0, 50, (100, 100))  # Mean 0, includes negatives

        normalized = normalizer.normalize(data, method='minmax')

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_nan_handling(self, normalizer):
        """Test handling of NaN values."""
        data = np.random.rand(100, 100)
        data[50:60, 50:60] = np.nan

        # Should either handle NaNs or document expected behavior
        try:
            normalized = normalizer.normalize(data, method='minmax')
            # If successful, NaNs should either be preserved or masked
            assert normalized is not None
        except ValueError:
            # Or might raise error - also acceptable if documented
            pass

    def test_outlier_handling(self, normalizer):
        """Test that normalization handles outliers appropriately."""
        data = np.random.rand(100, 100)
        data[0, 0] = 1000  # Extreme outlier

        # Percentile method should handle outliers better than minmax
        norm_percentile = normalizer.normalize(data, method='percentile', vmin=1, vmax=99)
        norm_minmax = normalizer.normalize(data, method='minmax')

        # Percentile should give better dynamic range
        assert norm_percentile.std() > norm_minmax.std()

    @pytest.mark.requires_noirlab_data
    def test_with_real_fits_data(self, normalizer, fits_loader, edu008_data):
        """Test normalization on real FITS data."""
        fits_file = edu008_data['fits_files'][0]
        fits_data = fits_loader.load(fits_file)

        for method in ['minmax', 'percentile', 'zscale']:
            normalized = normalizer.normalize(fits_data.data, method=method)

            assert normalized is not None
            assert normalized.min() >= 0
            assert normalized.max() <= 1

    def test_all_methods_available(self, normalizer):
        """Test that all documented normalization methods work."""
        data = np.random.rand(100, 100) * 100

        methods = ['minmax', 'percentile', 'zscale', 'manual']

        for method in methods:
            if method == 'manual':
                result = normalizer.normalize(data, method=method, vmin=25, vmax=75)
            elif method == 'percentile':
                result = normalizer.normalize(data, method=method, vmin=5, vmax=95)
            else:
                result = normalizer.normalize(data, method=method)

            assert result is not None
            assert 0 <= result.min() <= result.max() <= 1
