"""Tests for enhancer.py quality fixes.

Tests the enhanced CLAHE warnings, opt-in flag, and improved unsharp_mask.
"""

import numpy as np
import pytest


class TestCLAHEOptIn:
    """Test CLAHE explicit data loss acknowledgment."""

    def test_clahe_requires_explicit_flag(self):
        """Test that CLAHE requires i_accept_data_loss=True."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # Should raise ValueError without flag
        with pytest.raises(ValueError, match="CLAHE destroys photometric data"):
            enhancer.apply_clahe(data, i_accept_data_loss=False)

    def test_clahe_raises_error_without_flag(self):
        """Test that CLAHE raises error if flag not provided (defaults to False)."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # Default i_accept_data_loss=False should raise
        with pytest.raises(ValueError, match="i_accept_data_loss=True"):
            enhancer.apply_clahe(data)

    def test_clahe_works_with_flag(self):
        """Test that CLAHE works when flag is True."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # Should work with explicit acknowledgment
        result = enhancer.apply_clahe(data, i_accept_data_loss=True)

        assert result.shape == data.shape
        assert result.min() >= 0
        assert result.max() <= 1

    def test_clahe_error_message_helpful(self):
        """Test that error message provides clear guidance."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100)

        try:
            enhancer.apply_clahe(data)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Should mention alternatives
            assert "unsharp_mask" in error_msg or "PixInsight" in error_msg
            # Should explain how to proceed
            assert "i_accept_data_loss=True" in error_msg


class TestUnsharpMaskEnhanced:
    """Test improved unsharp_mask with multi-scale and luminance preservation."""

    def test_unsharp_single_scale_backward_compatible(self):
        """Test that single-scale (old API) still works."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # Old API: single sigma, single strength
        result = enhancer.unsharp_mask(data, sigma=2.0, strength=1.5)

        assert result.shape == data.shape
        assert not np.any(np.isnan(result))

    def test_unsharp_multiscale(self):
        """Test multi-scale sharpening with list of sigmas."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # Multi-scale: different radii
        result = enhancer.unsharp_mask(
            data,
            sigma=[1.0, 3.0, 5.0],
            strength=[0.5, 1.0, 0.3]
        )

        assert result.shape == data.shape
        assert not np.any(np.isnan(result))
        # Result should be different from original
        assert not np.allclose(result, data)

    def test_unsharp_multiscale_default_params(self):
        """Test that default params use multi-scale."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # No params - should use default multi-scale
        result = enhancer.unsharp_mask(data)

        assert result.shape == data.shape
        assert not np.any(np.isnan(result))

    def test_unsharp_preserves_data_range(self):
        """Test that unsharp mask preserves original data range."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25  # [0.25, 0.75]

        result = enhancer.unsharp_mask(data, sigma=2.0, strength=1.5)

        # Should not create values outside original range
        assert result.min() >= data.min() - 0.01  # Small tolerance
        assert result.max() <= data.max() + 0.01

    def test_unsharp_luminance_preservation_rgb(self):
        """Test luminance preservation for RGB images."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()

        # Create RGB image
        rgb = np.random.rand(50, 50, 3) * 0.5 + 0.25

        # Sharpen with luminance preservation
        result = enhancer.unsharp_mask(
            rgb,
            sigma=2.0,
            strength=1.5,
            preserve_luminance=True
        )

        assert result.shape == rgb.shape
        assert result.shape[2] == 3  # Still RGB
        assert not np.any(np.isnan(result))
        # Should be different (sharpened)
        assert not np.allclose(result, rgb)

    def test_unsharp_sigma_strength_length_mismatch_raises(self):
        """Test that mismatched sigma/strength lengths raise error."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100)

        with pytest.raises(ValueError, match="same length"):
            enhancer.unsharp_mask(
                data,
                sigma=[1.0, 3.0, 5.0],
                strength=[0.5, 1.0]  # Wrong length!
            )

    def test_unsharp_single_vs_multi_scale_different(self):
        """Test that multi-scale produces different result than single-scale."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # Single-scale
        single = enhancer.unsharp_mask(data, sigma=3.0, strength=1.0)

        # Multi-scale
        multi = enhancer.unsharp_mask(
            data,
            sigma=[1.0, 3.0, 5.0],
            strength=[0.33, 0.33, 0.34]  # Same total strength
        )

        # Results should be different (multi-scale has more scales)
        assert not np.allclose(single, multi)


class TestUnsharpMaskInternal:
    """Test internal _apply_multiscale_unsharp method."""

    def test_apply_multiscale_unsharp_accumulates(self):
        """Test that multiple scales accumulate correctly."""
        from astro_vision_composer.processing import Enhancer

        enhancer = Enhancer()
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # Apply manually
        sigma = np.array([2.0, 4.0])
        strength = np.array([1.0, 0.5])

        result = enhancer._apply_multiscale_unsharp(data, sigma, strength)

        assert result.shape == data.shape
        # Should enhance (values change)
        assert not np.allclose(result, data)
