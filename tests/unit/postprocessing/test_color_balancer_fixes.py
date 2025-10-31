"""Tests for color_balancer.py quality fixes.

Tests the fixed saturation method and deprecated methods.
"""

import numpy as np
import pytest
import warnings


class TestSaturationFixed:
    """Test the improved HSV-based saturation adjustment."""

    def test_saturation_hsv_conversion(self):
        """Test that saturation uses proper HSV conversion."""
        from astro_vision_composer.postprocessing import ColorBalancer

        balancer = ColorBalancer()

        # Create test RGB image (red gradient)
        rgb = np.zeros((100, 100, 3))
        rgb[:, :, 0] = np.linspace(0, 1, 100).reshape(1, -1)  # Red gradient

        # Apply saturation
        saturated = balancer.adjust_saturation(rgb, factor=1.5)

        # Verify output is valid
        assert saturated.shape == rgb.shape
        assert saturated.min() >= 0
        assert saturated.max() <= 1
        assert not np.any(np.isnan(saturated))

    def test_saturation_preserves_luminance(self):
        """Test that luminance (Value in HSV) is preserved."""
        from astro_vision_composer.postprocessing import ColorBalancer
        from skimage import color

        balancer = ColorBalancer()

        # Create colorful test image
        rgb = np.random.rand(50, 50, 3) * 0.5 + 0.25  # Values in [0.25, 0.75]

        # Get original luminance (Value in HSV)
        hsv_orig = color.rgb2hsv(rgb)
        value_orig = hsv_orig[:, :, 2]

        # Apply saturation
        saturated = balancer.adjust_saturation(rgb, factor=2.0)

        # Get new luminance
        hsv_new = color.rgb2hsv(saturated)
        value_new = hsv_new[:, :, 2]

        # Luminance should be approximately preserved
        # (small differences due to gamut clipping)
        np.testing.assert_allclose(value_orig, value_new, rtol=0.01, atol=0.01)

    def test_saturation_gamut_clipping(self):
        """Test that out-of-gamut colors are handled properly."""
        from astro_vision_composer.postprocessing import ColorBalancer

        balancer = ColorBalancer()

        # Create highly saturated test image (edge of gamut)
        rgb = np.zeros((50, 50, 3))
        rgb[:, :, 0] = 1.0  # Pure red
        rgb[:, :, 1] = 0.0
        rgb[:, :, 2] = 0.0

        # Boost saturation further (should clip gracefully)
        saturated = balancer.adjust_saturation(rgb, factor=2.0)

        # Should not have any invalid values
        assert np.all(saturated >= 0)
        assert np.all(saturated <= 1)
        assert not np.any(np.isnan(saturated))

    def test_saturation_desaturate_to_grayscale(self):
        """Test complete desaturation (factor=0) produces grayscale."""
        from astro_vision_composer.postprocessing import ColorBalancer

        balancer = ColorBalancer()

        # Create colorful image
        rgb = np.random.rand(50, 50, 3)

        # Desaturate completely
        gray = balancer.adjust_saturation(rgb, factor=0.0)

        # All channels should be equal (grayscale)
        np.testing.assert_allclose(gray[:, :, 0], gray[:, :, 1], rtol=1e-5)
        np.testing.assert_allclose(gray[:, :, 1], gray[:, :, 2], rtol=1e-5)

    def test_saturation_factor_one_unchanged(self):
        """Test that factor=1.0 leaves image unchanged."""
        from astro_vision_composer.postprocessing import ColorBalancer

        balancer = ColorBalancer()

        rgb = np.random.rand(50, 50, 3)
        result = balancer.adjust_saturation(rgb, factor=1.0)

        np.testing.assert_allclose(rgb, result, rtol=1e-5)

    def test_saturation_invalid_input_raises(self):
        """Test that invalid input raises ValueError."""
        from astro_vision_composer.postprocessing import ColorBalancer

        balancer = ColorBalancer()

        # Wrong shape
        with pytest.raises(ValueError, match="Expected RGB image"):
            balancer.adjust_saturation(np.random.rand(50, 50))  # 2D

        # Wrong number of channels
        with pytest.raises(ValueError, match="Expected RGB image"):
            balancer.adjust_saturation(np.random.rand(50, 50, 4))  # RGBA


class TestDeprecatedMethods:
    """Test that deprecated methods raise warnings but still work."""

    def test_white_balance_raises_deprecation_warning(self):
        """Test that white_balance raises DeprecationWarning."""
        from astro_vision_composer.postprocessing import ColorBalancer

        balancer = ColorBalancer()
        rgb = np.random.rand(50, 50, 3)

        with pytest.warns(DeprecationWarning, match="will be removed in v2.0"):
            result = balancer.white_balance(rgb)

        # Should still work (backward compatibility)
        assert result.shape == rgb.shape

    def test_color_temperature_raises_deprecation_warning(self):
        """Test that adjust_color_temperature raises DeprecationWarning."""
        from astro_vision_composer.postprocessing import ColorBalancer

        balancer = ColorBalancer()
        rgb = np.random.rand(50, 50, 3)

        with pytest.warns(DeprecationWarning, match="will be removed in v2.0"):
            result = balancer.adjust_color_temperature(rgb, temperature=0.2)

        # Should still work (backward compatibility)
        assert result.shape == rgb.shape

    def test_deprecated_methods_still_functional(self):
        """Test that deprecated methods still produce valid output."""
        from astro_vision_composer.postprocessing import ColorBalancer

        balancer = ColorBalancer()
        rgb = np.random.rand(50, 50, 3)

        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # white_balance
            wb_result = balancer.white_balance(rgb)
            assert wb_result.shape == rgb.shape
            assert wb_result.min() >= 0
            assert wb_result.max() <= 1

            # adjust_color_temperature
            ct_result = balancer.adjust_color_temperature(rgb, temperature=0.1)
            assert ct_result.shape == rgb.shape
            assert ct_result.min() >= 0
            assert ct_result.max() <= 1
