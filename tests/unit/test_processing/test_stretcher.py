"""Unit tests for Stretcher class."""

import pytest
import numpy as np


class TestStretcher:
    """Test Stretcher non-linear transformation methods."""

    def test_linear_stretch(self, stretcher):
        """Test linear stretch (identity)."""
        data = np.random.rand(100, 100)

        stretched = stretcher.stretch(data, method='linear')

        # Linear should be essentially unchanged (after renormalization)
        assert stretched is not None
        assert stretched.shape == data.shape
        assert 0 <= stretched.min() <= stretched.max() <= 1

    def test_sqrt_stretch(self, stretcher):
        """Test square root stretch."""
        data = np.random.rand(100, 100)

        stretched = stretcher.stretch(data, method='sqrt')

        assert stretched is not None
        # sqrt brightens faint features
        assert stretched.mean() > data.mean()

    def test_squared_stretch(self, stretcher):
        """Test squared stretch."""
        data = np.random.rand(100, 100)

        stretched = stretcher.stretch(data, method='squared')

        assert stretched is not None
        # squared darkens faint features
        assert stretched.mean() < data.mean()

    def test_log_stretch(self, stretcher):
        """Test logarithmic stretch."""
        data = np.random.rand(100, 100)

        stretched = stretcher.stretch(data, method='log')

        assert stretched is not None
        assert np.all(np.isfinite(stretched))
        assert 0 <= stretched.min() <= stretched.max() <= 1

    def test_asinh_stretch(self, stretcher):
        """Test arcsinh stretch."""
        data = np.random.rand(100, 100)

        stretched = stretcher.stretch(data, method='asinh', a=0.1)

        assert stretched is not None
        assert np.all(np.isfinite(stretched))

    def test_sinh_stretch(self, stretcher):
        """Test sinh stretch."""
        data = np.random.rand(100, 100)

        stretched = stretcher.stretch(data, method='sinh', a=0.1)

        assert stretched is not None
        assert np.all(np.isfinite(stretched))

    def test_power_stretch(self, stretcher):
        """Test power-law stretch."""
        data = np.random.rand(100, 100)

        # Test different power values
        for power in [0.5, 1.0, 2.0, 3.0]:
            stretched = stretcher.stretch(data, method='power', power=power)

            assert stretched is not None
            assert 0 <= stretched.min() <= stretched.max() <= 1

    def test_histeq_stretch(self, stretcher):
        """Test histogram equalization stretch."""
        data = np.random.rand(100, 100)

        stretched = stretcher.stretch(data, method='histeq')

        assert stretched is not None
        # Histogram equalization should flatten the histogram
        # Output should be more uniformly distributed
        assert stretched.std() > data.std() * 0.8

    def test_contrast_bias_stretch(self, stretcher):
        """Test contrast-bias stretch."""
        data = np.random.rand(100, 100)

        # Test with different contrast/bias values
        stretched = stretcher.stretch(data, method='contrast_bias',
                                      contrast=1.5, bias=0.5)

        assert stretched is not None
        assert 0 <= stretched.min() <= stretched.max() <= 1

    def test_all_stretch_methods(self, stretcher):
        """Test all available stretch methods."""
        data = np.random.rand(100, 100)

        methods = ['linear', 'sqrt', 'squared', 'log', 'asinh', 'sinh',
                   'power', 'histeq', 'contrast_bias']

        for method in methods:
            if method == 'power':
                result = stretcher.stretch(data, method=method, power=2.0)
            elif method == 'histeq':
                result = stretcher.stretch(data, method=method)
            elif method == 'contrast_bias':
                result = stretcher.stretch(data, method=method, contrast=1.5, bias=0.5)
            elif method in ('asinh', 'sinh'):
                result = stretcher.stretch(data, method=method, a=0.1)
            else:
                result = stretcher.stretch(data, method=method)

            assert result is not None
            assert result.shape == data.shape
            assert np.all(np.isfinite(result))
            assert result.min() >= 0 and result.max() <= 1

    def test_stretch_preserves_shape(self, stretcher):
        """Test that stretching preserves array shape."""
        for shape in [(50, 50), (100, 200), (256, 256)]:
            data = np.random.rand(*shape)
            stretched = stretcher.stretch(data, method='asinh')
            assert stretched.shape == shape

    def test_blank_image(self, stretcher):
        """Test stretching blank (all zeros) image."""
        data = np.zeros((100, 100))

        # Should handle gracefully
        stretched = stretcher.stretch(data, method='sqrt')

        assert stretched is not None

    def test_uniform_image(self, stretcher):
        """Test stretching uniform (constant value) image."""
        data = np.ones((100, 100)) * 0.5

        stretched = stretcher.stretch(data, method='log')

        assert stretched is not None

    def test_negative_values(self, stretcher):
        """Test handling of negative values."""
        # Some methods (like log) can't handle negatives
        data = np.random.normal(0, 1, (100, 100))

        # asinh should handle negatives
        stretched = stretcher.stretch(data, method='asinh')
        assert stretched is not None

    def test_high_dynamic_range(self, stretcher):
        """Test stretching high dynamic range data."""
        # Create data with large dynamic range
        data = np.random.rand(100, 100) * 0.01  # Faint background
        data[45:55, 45:55] = np.random.rand(10, 10)  # Bright source

        # asinh is good for high dynamic range
        stretched = stretcher.stretch(data, method='asinh', a=0.01)

        assert stretched is not None
        # Should reveal both faint and bright features
        assert stretched.min() < 0.5
        assert stretched.max() > 0.5

    @pytest.mark.requires_noirlab_data
    def test_with_real_fits_data(self, stretcher, fits_loader, normalizer, edu008_data):
        """Test stretching on real FITS data."""
        fits_file = edu008_data['fits_files'][0]
        fits_data = fits_loader.load(fits_file)

        # Normalize first
        normalized = normalizer.normalize(fits_data.data, method='zscale')

        # Test multiple stretch methods
        for method in ['sqrt', 'log', 'asinh', 'histeq']:
            if method == 'asinh':
                stretched = stretcher.stretch(normalized, method=method, a=0.1)
            else:
                stretched = stretcher.stretch(normalized, method=method)

            assert stretched is not None
            assert 0 <= stretched.min() <= stretched.max() <= 1

    def test_histeq_with_2d_data(self, stretcher):
        """Test HistEq stretch handles 2D data correctly."""
        data = np.random.rand(200, 200)

        result = stretcher.stretch(data, method='histeq')

        assert result.shape == data.shape
        assert np.all(np.isfinite(result))
        assert 0 <= result.min() <= result.max() <= 1
