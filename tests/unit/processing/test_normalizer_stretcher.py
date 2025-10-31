"""Combined unit tests for Normalizer and Stretcher classes.

Tests image normalization and stretching operations.
"""

import pytest
import numpy as np

from astro_vision_composer.processing.normalizer import Normalizer
from astro_vision_composer.processing.stretcher import Stretcher


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample astronomical-like data."""
    np.random.seed(42)
    # Create data with realistic range
    data = np.random.randn(100, 100) * 100 + 1000
    return data.astype(np.float32)


@pytest.fixture
def normalized_data():
    """Create already-normalized data in [0, 1]."""
    np.random.seed(42)
    return np.random.rand(100, 100).astype(np.float32)


# ============================================================================
# Test Normalizer
# ============================================================================

class TestNormalizer:
    """Test Normalizer class."""

    def test_init(self):
        """Test normalizer initialization."""
        normalizer = Normalizer()
        assert normalizer is not None
        assert normalizer._last_interval_obj is None

    def test_normalize_zscale(self, sample_data):
        """Test normalization with ZScale (default)."""
        normalizer = Normalizer()
        normalized = normalizer.normalize(sample_data, method='zscale')

        assert normalized is not None
        assert normalized.shape == sample_data.shape
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

    def test_normalize_minmax(self, sample_data):
        """Test normalization with minmax."""
        normalizer = Normalizer()
        normalized = normalizer.normalize(sample_data, method='minmax')

        assert normalized is not None
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        # MinMax should use full [0, 1] range
        assert np.isclose(normalized.min(), 0, atol=0.01)
        assert np.isclose(normalized.max(), 1, atol=0.01)

    def test_normalize_percentile(self, sample_data):
        """Test normalization with percentile."""
        normalizer = Normalizer()
        normalized = normalizer.normalize(
            sample_data,
            method='percentile',
            lower=2.0,
            upper=98.0
        )

        assert normalized is not None
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

    def test_normalize_manual(self, sample_data):
        """Test normalization with manual interval."""
        normalizer = Normalizer()
        vmin, vmax = 900, 1100

        normalized = normalizer.normalize(
            sample_data,
            method='manual',
            vmin=vmin,
            vmax=vmax
        )

        assert normalized is not None
        # Values outside [vmin, vmax] should be clipped
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

    def test_empty_data_raises(self):
        """Test that empty data raises ValueError."""
        normalizer = Normalizer()
        with pytest.raises(ValueError, match="data is empty or None"):
            normalizer.normalize(np.array([]), method='zscale')

    def test_none_data_raises(self):
        """Test that None data raises ValueError."""
        normalizer = Normalizer()
        with pytest.raises(ValueError, match="data is empty or None"):
            normalizer.normalize(None, method='zscale')

    def test_all_nan_data(self):
        """Test normalization with all-NaN data."""
        normalizer = Normalizer()
        data = np.full((50, 50), np.nan)

        normalized = normalizer.normalize(data, method='zscale')

        # Should return zeros
        assert np.all(normalized == 0)

    def test_get_interval_object(self, sample_data):
        """Test getting interval object after normalization."""
        normalizer = Normalizer()

        # Before normalization
        assert normalizer.get_interval_object() is None

        # After normalization
        normalizer.normalize(sample_data, method='zscale')
        interval_obj = normalizer.get_interval_object()

        assert interval_obj is not None

    def test_get_interval_limits(self, sample_data):
        """Test getting interval limits without normalizing."""
        normalizer = Normalizer()

        vmin, vmax = normalizer.get_interval_limits(
            sample_data,
            method='zscale'
        )

        assert isinstance(vmin, float)
        assert isinstance(vmax, float)
        assert vmin < vmax

    def test_invalid_method_raises(self, sample_data):
        """Test that invalid method raises ValueError."""
        normalizer = Normalizer()

        with pytest.raises(ValueError, match="Unknown method"):
            normalizer.normalize(sample_data, method='invalid')

    def test_invalid_percentiles_raise(self, sample_data):
        """Test that invalid percentiles raise ValueError."""
        normalizer = Normalizer()

        with pytest.raises(ValueError, match="Invalid percentiles"):
            normalizer.normalize(sample_data, method='percentile', lower=50, upper=40)

    def test_manual_without_vmin_raises(self, sample_data):
        """Test that manual interval without vmin raises ValueError."""
        normalizer = Normalizer()

        with pytest.raises(ValueError, match="requires both vmin and vmax"):
            normalizer.normalize(sample_data, method='manual', vmax=1000)


# ============================================================================
# Test Stretcher
# ============================================================================

class TestStretcher:
    """Test Stretcher class."""

    def test_init(self):
        """Test stretcher initialization."""
        stretcher = Stretcher()
        assert stretcher is not None

    def test_stretch_linear(self, normalized_data):
        """Test linear stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(normalized_data, method='linear')

        assert stretched is not None
        assert stretched.shape == normalized_data.shape
        # Linear should be identity for normalized data
        assert np.allclose(stretched, normalized_data, atol=0.01)

    def test_stretch_sqrt(self, normalized_data):
        """Test sqrt stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(normalized_data, method='sqrt')

        assert stretched is not None
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_stretch_log(self, normalized_data):
        """Test log stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(normalized_data, method='log', a=1000)

        assert stretched is not None
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_stretch_asinh(self, normalized_data):
        """Test asinh stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(normalized_data, method='asinh', a=0.1)

        assert stretched is not None
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_stretch_sinh(self, normalized_data):
        """Test sinh stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(normalized_data, method='sinh', a=0.33)

        assert stretched is not None
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_stretch_power(self, normalized_data):
        """Test power stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(normalized_data, method='power', power=2.0)

        assert stretched is not None
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_stretch_squared(self, normalized_data):
        """Test squared stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(normalized_data, method='squared')

        assert stretched is not None
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_stretch_histeq(self, normalized_data):
        """Test histogram equalization stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(normalized_data, method='histeq')

        assert stretched is not None
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_stretch_contrast_bias(self, normalized_data):
        """Test contrast+bias stretch."""
        stretcher = Stretcher()
        stretched = stretcher.stretch(
            normalized_data,
            method='contrast_bias',
            contrast=1.5,
            bias=0.5
        )

        assert stretched is not None
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_empty_data_raises(self):
        """Test that empty data raises ValueError."""
        stretcher = Stretcher()

        with pytest.raises(ValueError, match="data is empty or None"):
            stretcher.stretch(np.array([]), method='linear')

    def test_none_data_raises(self):
        """Test that None data raises ValueError."""
        stretcher = Stretcher()

        with pytest.raises(ValueError, match="data is empty or None"):
            stretcher.stretch(None, method='linear')

    def test_invalid_method_raises(self, normalized_data):
        """Test that invalid method raises ValueError."""
        stretcher = Stretcher()

        with pytest.raises(ValueError, match="Unknown stretch method"):
            stretcher.stretch(normalized_data, method='invalid')

    def test_all_zeros(self):
        """Test stretching all-zero data."""
        stretcher = Stretcher()
        data = np.zeros((50, 50))

        stretched = stretcher.stretch(data, method='linear')
        assert np.all(stretched == 0)

    def test_all_ones(self):
        """Test stretching all-one data."""
        stretcher = Stretcher()
        data = np.ones((50, 50))

        stretched = stretcher.stretch(data, method='linear')
        assert np.allclose(stretched, 1.0)


# ============================================================================
# Test Integration
# ============================================================================

class TestNormalizerStretcherIntegration:
    """Test Normalizer and Stretcher working together."""

    def test_normalize_then_stretch(self, sample_data):
        """Test full workflow: normalize then stretch."""
        normalizer = Normalizer()
        stretcher = Stretcher()

        # Normalize
        normalized = normalizer.normalize(sample_data, method='zscale')

        # Stretch
        stretched = stretcher.stretch(normalized, method='asinh', a=0.1)

        assert stretched is not None
        assert stretched.shape == sample_data.shape
        assert np.all(stretched >= 0)
        assert np.all(stretched <= 1)

    def test_all_methods_combination(self, sample_data):
        """Test all normalization methods with all stretch methods."""
        normalizer = Normalizer()
        stretcher = Stretcher()

        norm_methods = ['zscale', 'minmax', 'percentile']
        stretch_methods = ['linear', 'sqrt', 'log', 'asinh']

        for norm_method in norm_methods:
            if norm_method == 'percentile':
                normalized = normalizer.normalize(
                    sample_data,
                    method=norm_method,
                    lower=2, upper=98
                )
            else:
                normalized = normalizer.normalize(sample_data, method=norm_method)

            for stretch_method in stretch_methods:
                if stretch_method == 'log':
                    stretched = stretcher.stretch(normalized, method=stretch_method, a=1000)
                elif stretch_method == 'asinh':
                    stretched = stretcher.stretch(normalized, method=stretch_method, a=0.1)
                else:
                    stretched = stretcher.stretch(normalized, method=stretch_method)

                assert stretched is not None
                assert np.all(stretched >= 0)
                assert np.all(stretched <= 1)
