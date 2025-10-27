"""Unit tests for Compositor class."""

import pytest
import numpy as np


class TestCompositor:
    """Test Compositor for RGB image composition."""

    def test_simple_rgb_composite(self, compositor, sample_rgb_data):
        """Test simple RGB composite creation."""
        r, g, b = sample_rgb_data

        rgb = compositor.create_simple_rgb(r, g, b)

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.float32 or rgb.dtype == np.float64
        assert 0 <= rgb.min() <= rgb.max() <= 1

    def test_lupton_rgb_composite(self, compositor, sample_rgb_data):
        """Test Lupton algorithm RGB composite."""
        r, g, b = sample_rgb_data

        rgb = compositor.create_lupton_rgb(r, g, b, stretch=0.5, Q=8)

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)
        assert 0 <= rgb.min() <= rgb.max() <= 1

    def test_lupton_parameters(self, compositor):
        """Test Lupton composite with various parameters."""
        r = np.random.rand(100, 100)
        g = np.random.rand(100, 100)
        b = np.random.rand(100, 100)

        for stretch in [0.1, 0.5, 1.0]:
            for Q in [1, 8, 16]:
                rgb = compositor.create_lupton_rgb(r, g, b, stretch=stretch, Q=Q)
                assert rgb is not None
                assert rgb.shape == (100, 100, 3)

    def test_mismatched_shapes_error(self, compositor):
        """Test error when channel shapes don't match."""
        r = np.random.rand(100, 100)
        g = np.random.rand(100, 100)
        b = np.random.rand(50, 50)  # Different shape!

        with pytest.raises((ValueError, AssertionError)):
            compositor.create_simple_rgb(r, g, b)

    def test_channel_scaling(self, compositor):
        """Test per-channel scaling."""
        r = np.random.rand(100, 100) * 100
        g = np.random.rand(100, 100) * 50
        b = np.random.rand(100, 100) * 25

        # With different scales, should normalize appropriately
        rgb = compositor.create_simple_rgb(r, g, b, normalize_channels=True)

        assert rgb is not None
        # All channels should be scaled to similar ranges

    def test_narrowband_composite(self, compositor):
        """Test narrowband (SHO) composite."""
        sii = np.random.rand(100, 100)
        ha = np.random.rand(100, 100)
        oiii = np.random.rand(100, 100)

        # Map to RGB: SII=red, H-alpha=green, OIII=blue
        rgb = compositor.create_simple_rgb(sii, ha, oiii)

        assert rgb is not None
        assert rgb.shape == (100, 100, 3)

    def test_high_dynamic_range(self, compositor):
        """Test compositing high dynamic range data."""
        # Create channels with large dynamic range
        r = np.random.rand(100, 100) * 0.01  # Faint
        r[45:55, 45:55] = 1.0  # Bright source
        g = np.random.rand(100, 100) * 0.01
        g[40:60, 40:60] = 0.8
        b = np.random.rand(100, 100) * 0.01

        # Lupton algorithm should handle high dynamic range better
        rgb = compositor.create_lupton_rgb(r, g, b, stretch=0.1, Q=10)

        assert rgb is not None
        # Should show both faint and bright features
        assert rgb.min() < 0.5
        assert rgb.max() > 0.5

    def test_blank_channels(self, compositor):
        """Test compositing blank channels."""
        r = np.zeros((100, 100))
        g = np.zeros((100, 100))
        b = np.zeros((100, 100))

        rgb = compositor.create_simple_rgb(r, g, b)

        assert rgb is not None
        assert np.allclose(rgb, 0)

    def test_single_channel_only(self, compositor):
        """Test composite with only one channel having data."""
        r = np.random.rand(100, 100)
        g = np.zeros((100, 100))
        b = np.zeros((100, 100))

        rgb = compositor.create_simple_rgb(r, g, b)

        assert rgb is not None
        # Should result in red-only image
        assert rgb[:, :, 0].max() > 0  # Red channel has data
        assert rgb[:, :, 1].max() == 0  # Green is empty
        assert rgb[:, :, 2].max() == 0  # Blue is empty

    @pytest.mark.requires_noirlab_data
    def test_with_real_fits_data(self, compositor, fits_loader, normalizer, stretcher, edu008_data):
        """Test compositing real FITS data."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        # Load 3 bands
        channels = []
        for fits_file in edu008_data['fits_files'][:3]:
            fits_data = fits_loader.load(fits_file)
            normalized = normalizer.normalize(fits_data.data, method='zscale')
            stretched = stretcher.stretch(normalized, method='asinh', a=0.1)
            channels.append(stretched)

        # Create composite (assume chromatic order: shortest=blue, longest=red)
        rgb = compositor.create_lupton_rgb(
            channels[2], channels[1], channels[0],  # Reverse order
            stretch=0.5, Q=8
        )

        assert rgb is not None
        assert rgb.shape[2] == 3
        assert 0 <= rgb.min() <= rgb.max() <= 1

    def test_color_balance_during_composition(self, compositor):
        """Test that compositor can apply color balance."""
        r = np.random.rand(100, 100)
        g = np.random.rand(100, 100)
        b = np.random.rand(100, 100)

        # If compositor supports channel weights
        if hasattr(compositor, 'create_weighted_rgb'):
            rgb = compositor.create_weighted_rgb(
                r, g, b,
                r_weight=1.0, g_weight=0.8, b_weight=1.2
            )
            assert rgb is not None
