"""Unit tests for Compositor class.

Tests RGB composite creation using Lupton and simple methods.
"""

import pytest
import numpy as np
from astropy.visualization import LinearStretch, LuptonAsinhStretch

from astro_vision_composer.postprocessing.compositor import Compositor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rgb_data():
    """Create sample RGB channel data."""
    np.random.seed(42)
    r = np.random.rand(100, 100).astype(np.float32)
    g = np.random.rand(100, 100).astype(np.float32) * 0.8
    b = np.random.rand(100, 100).astype(np.float32) * 0.6
    return r, g, b


@pytest.fixture
def normalized_rgb_data():
    """Create normalized RGB data (values in [0, 1])."""
    r = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)
    g = np.linspace(0, 0.8, 10000).reshape(100, 100).astype(np.float32)
    b = np.linspace(0, 0.6, 10000).reshape(100, 100).astype(np.float32)
    return r, g, b


@pytest.fixture
def narrowband_data():
    """Create synthetic narrowband data (Ha, OIII, SII)."""
    np.random.seed(42)
    ha = np.random.rand(100, 100).astype(np.float32)
    oiii = np.random.rand(100, 100).astype(np.float32) * 0.7
    sii = np.random.rand(100, 100).astype(np.float32) * 0.5
    return ha, oiii, sii


# ============================================================================
# Test Compositor Initialization
# ============================================================================

class TestCompositorInit:
    """Test Compositor initialization."""

    def test_init(self):
        """Test basic initialization."""
        compositor = Compositor()
        assert compositor is not None


# ============================================================================
# Test create_lupton_rgb()
# ============================================================================

class TestCreateLuptonRGB:
    """Test create_lupton_rgb() method."""

    def test_basic_lupton_rgb(self, sample_rgb_data):
        """Test basic Lupton RGB creation with default parameters."""
        r, g, b = sample_rgb_data
        compositor = Compositor()

        rgb = compositor.create_lupton_rgb(r, g, b)

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.float64
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 1)

    def test_lupton_with_custom_stretch(self, sample_rgb_data):
        """Test Lupton RGB with custom stretch and Q parameters."""
        r, g, b = sample_rgb_data
        compositor = Compositor()

        rgb = compositor.create_lupton_rgb(
            r, g, b,
            stretch=0.8,
            Q=12.0
        )

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)

    def test_lupton_with_minimum(self, normalized_rgb_data):
        """Test Lupton RGB with minimum parameter (black point)."""
        r, g, b = normalized_rgb_data
        compositor = Compositor()

        rgb = compositor.create_lupton_rgb(
            r, g, b,
            minimum=0.2,  # Values below 0.2 become black
            stretch=0.5,
            Q=8
        )

        assert rgb is not None
        # Lower values should be darker
        assert np.any(rgb < 0.1)

    def test_lupton_with_linear_stretch_object(self, normalized_rgb_data):
        """Test Lupton RGB with LinearStretch (identity, for pre-stretched data)."""
        r, g, b = normalized_rgb_data
        compositor = Compositor()

        rgb = compositor.create_lupton_rgb(
            r, g, b,
            stretch_object=LinearStretch()
        )

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)

    def test_lupton_with_custom_stretch_object(self, sample_rgb_data):
        """Test Lupton RGB with custom LuptonAsinhStretch object."""
        r, g, b = sample_rgb_data
        compositor = Compositor()

        stretch_obj = LuptonAsinhStretch(stretch=5, Q=10)
        rgb = compositor.create_lupton_rgb(
            r, g, b,
            stretch_object=stretch_obj
        )

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)

    def test_lupton_uint8_output(self, sample_rgb_data):
        """Test Lupton RGB with uint8 output."""
        r, g, b = sample_rgb_data
        compositor = Compositor()

        rgb = compositor.create_lupton_rgb(
            r, g, b,
            output_dtype=np.uint8
        )

        assert rgb is not None
        assert rgb.dtype == np.uint8
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 255)

    def test_lupton_shape_mismatch_raises(self):
        """Test that mismatched channel shapes raise ValueError."""
        compositor = Compositor()

        r = np.random.rand(100, 100)
        g = np.random.rand(100, 100)
        b = np.random.rand(50, 50)  # Different shape

        with pytest.raises(ValueError, match="All channels must have same shape"):
            compositor.create_lupton_rgb(r, g, b)


# ============================================================================
# Test create_simple_rgb()
# ============================================================================

class TestCreateSimpleRGB:
    """Test create_simple_rgb() method."""

    def test_basic_simple_rgb(self, sample_rgb_data):
        """Test basic simple RGB creation."""
        r, g, b = sample_rgb_data
        compositor = Compositor()

        rgb = compositor.create_simple_rgb(r, g, b)

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.float32
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 1)

    def test_simple_with_channel_scaling(self, sample_rgb_data):
        """Test simple RGB with per-channel scaling."""
        r, g, b = sample_rgb_data
        compositor = Compositor()

        rgb = compositor.create_simple_rgb(
            r, g, b,
            r_scale=1.2,
            g_scale=1.0,
            b_scale=0.8
        )

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)
        # Red channel should be boosted (but clipped at 1.0)
        assert np.all(rgb[:, :, 0] >= r)

    def test_simple_clipping(self):
        """Test that simple RGB clips values to [0, 1]."""
        compositor = Compositor()

        # Create data that will exceed [0, 1] after scaling
        r = np.ones((50, 50)) * 0.9
        g = np.ones((50, 50)) * 0.9
        b = np.ones((50, 50)) * 0.9

        rgb = compositor.create_simple_rgb(
            r, g, b,
            r_scale=2.0,  # Will exceed 1.0
            g_scale=2.0,
            b_scale=2.0
        )

        # Should be clipped to 1.0
        assert np.all(rgb <= 1.0)
        assert np.allclose(rgb, 1.0)

    def test_simple_shape_mismatch_raises(self):
        """Test that mismatched channel shapes raise ValueError."""
        compositor = Compositor()

        r = np.random.rand(100, 100)
        g = np.random.rand(100, 100)
        b = np.random.rand(50, 50)  # Different shape

        with pytest.raises(ValueError, match="All channels must have same shape"):
            compositor.create_simple_rgb(r, g, b)


# ============================================================================
# Test create_narrowband_composite()
# ============================================================================

class TestCreateNarrowbandComposite:
    """Test create_narrowband_composite() method."""

    def test_narrowband_lupton(self, narrowband_data):
        """Test narrowband composite with Lupton method."""
        ha, oiii, sii = narrowband_data
        compositor = Compositor()

        rgb = compositor.create_narrowband_composite(
            ha=ha,
            oiii=oiii,
            sii=sii,
            method='lupton',
            stretch=0.5,
            Q=10
        )

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)

    def test_narrowband_simple(self, narrowband_data):
        """Test narrowband composite with simple method."""
        ha, oiii, sii = narrowband_data
        compositor = Compositor()

        rgb = compositor.create_narrowband_composite(
            ha=ha,
            oiii=oiii,
            sii=sii,
            method='simple',
            r_scale=1.0,
            g_scale=1.0,
            b_scale=1.0
        )

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)

    def test_narrowband_invalid_method_raises(self, narrowband_data):
        """Test that invalid method raises ValueError."""
        ha, oiii, sii = narrowband_data
        compositor = Compositor()

        with pytest.raises(ValueError, match="Unknown method"):
            compositor.create_narrowband_composite(
                ha=ha, oiii=oiii, sii=sii,
                method='invalid'
            )


# ============================================================================
# Test create_from_mapping()
# ============================================================================

class TestCreateFromMapping:
    """Test create_from_mapping() method."""

    def test_from_mapping_lupton(self, sample_rgb_data):
        """Test create_from_mapping() with Lupton method."""
        from astro_vision_composer.postprocessing.channel_mapper import ChannelMapping

        r, g, b = sample_rgb_data
        compositor = Compositor()

        # Create mock mapping
        mapping = ChannelMapping(
            red='i',
            green='r',
            blue='g'
        )

        data = {'i': r, 'r': g, 'g': b}

        rgb = compositor.create_from_mapping(
            data=data,
            mapping=mapping,
            method='lupton',
            stretch=0.5,
            Q=8
        )

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)

    def test_from_mapping_simple(self, sample_rgb_data):
        """Test create_from_mapping() with simple method."""
        from astro_vision_composer.postprocessing.channel_mapper import ChannelMapping

        r, g, b = sample_rgb_data
        compositor = Compositor()

        mapping = ChannelMapping(
            red='i',
            green='r',
            blue='g'
        )

        data = {'i': r, 'r': g, 'g': b}

        rgb = compositor.create_from_mapping(
            data=data,
            mapping=mapping,
            method='simple'
        )

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)

    def test_from_mapping_invalid_method_raises(self, sample_rgb_data):
        """Test that invalid method raises ValueError."""
        from astro_vision_composer.postprocessing.channel_mapper import ChannelMapping

        r, g, b = sample_rgb_data
        compositor = Compositor()

        mapping = ChannelMapping(
            red='i', green='r', blue='g'
        )

        data = {'i': r, 'r': g, 'g': b}

        with pytest.raises(ValueError, match="Unknown method"):
            compositor.create_from_mapping(
                data=data, mapping=mapping, method='invalid'
            )


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestCompositorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_all_zeros(self):
        """Test compositor with all-zero channels."""
        compositor = Compositor()

        r = np.zeros((50, 50))
        g = np.zeros((50, 50))
        b = np.zeros((50, 50))

        rgb = compositor.create_lupton_rgb(r, g, b)
        assert rgb is not None
        assert rgb.shape == (50, 50, 3)

    def test_all_ones(self):
        """Test compositor with all-one channels."""
        compositor = Compositor()

        r = np.ones((50, 50))
        g = np.ones((50, 50))
        b = np.ones((50, 50))

        rgb = compositor.create_lupton_rgb(r, g, b)
        assert rgb is not None
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 1)

    def test_very_small_images(self):
        """Test compositor with very small images (10×10)."""
        compositor = Compositor()

        r = np.random.rand(10, 10)
        g = np.random.rand(10, 10)
        b = np.random.rand(10, 10)

        rgb = compositor.create_lupton_rgb(r, g, b)
        assert rgb.shape == (10, 10, 3)

    def test_single_channel_dominant(self):
        """Test compositor when one channel dominates."""
        compositor = Compositor()

        r = np.ones((50, 50))  # Dominant
        g = np.zeros((50, 50))
        b = np.zeros((50, 50))

        rgb = compositor.create_simple_rgb(r, g, b)

        # Red channel should be 1, others 0
        assert np.allclose(rgb[:, :, 0], 1.0)
        assert np.allclose(rgb[:, :, 1], 0.0)
        assert np.allclose(rgb[:, :, 2], 0.0)

    def test_negative_values_handled(self):
        """Test that negative values are handled (clipped to 0)."""
        compositor = Compositor()

        r = np.array([[-0.5, 0.5], [0.5, 1.0]])
        g = np.array([[0.0, 0.5], [0.5, 1.0]])
        b = np.array([[0.0, 0.5], [0.5, 1.0]])

        rgb = compositor.create_simple_rgb(r, g, b)

        # Negative values should be clipped to 0
        assert np.all(rgb >= 0)
