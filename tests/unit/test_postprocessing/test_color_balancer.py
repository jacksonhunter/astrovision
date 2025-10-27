"""Unit tests for ColorBalancer class."""

import pytest
import numpy as np


class TestColorBalancer:
    """Test ColorBalancer for color adjustment."""

    def test_white_balance(self, color_balancer):
        """Test white balance adjustment."""
        # Create RGB image with color cast
        rgb = np.random.rand(100, 100, 3)
        rgb[:, :, 0] *= 1.5  # Red bias

        balanced = color_balancer.white_balance(rgb)

        assert balanced is not None
        assert balanced.shape == rgb.shape
        # Red channel should be reduced
        assert balanced[:, :, 0].mean() < rgb[:, :, 0].mean()

    def test_adjust_saturation(self, color_balancer):
        """Test saturation adjustment."""
        rgb = np.random.rand(100, 100, 3)

        # Increase saturation
        saturated = color_balancer.adjust_saturation(rgb, factor=1.5)

        assert saturated is not None
        assert saturated.shape == rgb.shape

        # Decrease saturation
        desaturated = color_balancer.adjust_saturation(rgb, factor=0.5)

        assert desaturated is not None

    def test_adjust_brightness(self, color_balancer):
        """Test brightness adjustment."""
        rgb = np.random.rand(100, 100, 3) * 0.5  # Dark image

        brightened = color_balancer.adjust_brightness(rgb, factor=1.5)

        assert brightened is not None
        assert brightened.mean() > rgb.mean()

    def test_color_temperature(self, color_balancer):
        """Test color temperature adjustment."""
        rgb = np.random.rand(100, 100, 3)

        # Warmer (more red/yellow)
        warm = color_balancer.adjust_temperature(rgb, temperature=6500)

        # Cooler (more blue)
        cool = color_balancer.adjust_temperature(rgb, temperature=9000)

        assert warm is not None
        assert cool is not None
        # Warm should have more red, cool more blue
        assert warm[:, :, 0].mean() > cool[:, :, 0].mean()

    def test_channel_weights(self, color_balancer):
        """Test per-channel weighting."""
        rgb = np.random.rand(100, 100, 3)

        weighted = color_balancer.apply_channel_weights(
            rgb, r_weight=1.2, g_weight=1.0, b_weight=0.8
        )

        assert weighted is not None
        assert weighted.shape == rgb.shape
        # Red should be boosted, blue reduced
        assert weighted[:, :, 0].mean() > rgb[:, :, 0].mean()
        assert weighted[:, :, 2].mean() < rgb[:, :, 2].mean()

    def test_gray_world_assumption(self, color_balancer):
        """Test gray world white balance algorithm."""
        rgb = np.random.rand(100, 100, 3)
        rgb[:, :, 0] *= 1.5  # Red cast
        rgb[:, :, 2] *= 0.7  # Reduced blue

        if hasattr(color_balancer, 'gray_world_balance'):
            balanced = color_balancer.gray_world_balance(rgb)

            # Average of each channel should be more similar
            r_mean = balanced[:, :, 0].mean()
            g_mean = balanced[:, :, 1].mean()
            b_mean = balanced[:, :, 2].mean()

            # Means should be closer to each other than before
            assert abs(r_mean - g_mean) < abs(rgb[:, :, 0].mean() - rgb[:, :, 1].mean())

    def test_preserve_shape(self, color_balancer):
        """Test that color balancing preserves image shape."""
        for shape in [(50, 50, 3), (100, 200, 3), (256, 256, 3)]:
            rgb = np.random.rand(*shape)
            balanced = color_balancer.white_balance(rgb)
            assert balanced.shape == shape

    def test_value_range_preservation(self, color_balancer):
        """Test that values stay in [0, 1] range."""
        rgb = np.random.rand(100, 100, 3)

        balanced = color_balancer.white_balance(rgb)

        assert balanced.min() >= 0
        assert balanced.max() <= 1

    def test_saturation_extremes(self, color_balancer):
        """Test saturation adjustment with extreme values."""
        rgb = np.random.rand(100, 100, 3)

        # Full desaturation (grayscale)
        gray = color_balancer.adjust_saturation(rgb, factor=0.0)
        assert gray is not None
        # All channels should be equal (grayscale)
        assert np.allclose(gray[:, :, 0], gray[:, :, 1], atol=0.01)
        assert np.allclose(gray[:, :, 1], gray[:, :, 2], atol=0.01)

        # Maximum saturation
        super_sat = color_balancer.adjust_saturation(rgb, factor=3.0)
        assert super_sat is not None
        assert super_sat.min() >= 0
        assert super_sat.max() <= 1

    @pytest.mark.requires_noirlab_data
    def test_with_real_composite(self, color_balancer, compositor, fits_loader,
                                  normalizer, stretcher, edu008_data):
        """Test color balancing on real composite."""
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

        # Apply color balancing
        balanced = color_balancer.white_balance(rgb)

        assert balanced is not None
        assert balanced.shape == rgb.shape
        assert 0 <= balanced.min() <= balanced.max() <= 1
