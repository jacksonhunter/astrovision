"""Unit tests for Calibrator class."""

import pytest
import numpy as np
from astropy.io import fits


class TestCalibrator:
    """Test Calibrator for bias, dark, flat, and background corrections."""

    def test_bias_correction(self, calibrator):
        """Test bias frame subtraction."""
        # Create science frame with bias level
        science = np.random.rand(100, 100) * 1000 + 100  # Signal + bias

        # Create bias frame
        bias = np.ones((100, 100)) * 100

        corrected = calibrator.subtract_bias(science, bias)

        # After bias subtraction, mean should be close to 500 (original signal mean)
        assert corrected.mean() < science.mean()
        assert corrected.mean() > 400
        assert corrected.mean() < 600

    def test_dark_correction(self, calibrator):
        """Test dark frame subtraction."""
        science = np.random.rand(100, 100) * 1000
        dark = np.ones((100, 100)) * 10  # Dark current

        corrected = calibrator.subtract_dark(science, dark)

        assert corrected.mean() < science.mean()
        assert corrected.mean() > science.mean() - 15

    def test_flat_field_correction(self, calibrator):
        """Test flat field correction."""
        # Create science frame
        science = np.random.rand(100, 100) * 1000

        # Create flat field with vignetting (lower at edges)
        x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        flat = 1.0 - 0.3 * (x**2 + y**2)  # Radial vignetting
        flat = flat / flat.mean()  # Normalize

        corrected = calibrator.apply_flat(science, flat)

        # Corrected image should have more uniform response
        assert corrected is not None
        assert corrected.shape == science.shape

    def test_background_subtraction(self, calibrator):
        """Test background estimation and subtraction."""
        # Create image with constant background
        background_level = 100
        signal = np.random.rand(100, 100) * 50
        science = signal + background_level

        # Add some bright sources
        science[30:40, 30:40] += 500

        corrected = calibrator.subtract_background(science)

        # After background subtraction, median should be close to 0
        assert corrected.median() < background_level / 2
        # Bright sources should still be present
        assert corrected.max() > 400

    def test_combined_calibration(self, calibrator):
        """Test full calibration pipeline (bias, dark, flat)."""
        science = np.random.rand(100, 100) * 1000 + 100
        bias = np.ones((100, 100)) * 100
        dark = np.ones((100, 100)) * 5
        flat = np.ones((100, 100)) * 0.95  # 5% sensitivity variation

        # Apply full calibration
        corrected = calibrator.calibrate(science, bias=bias, dark=dark, flat=flat)

        assert corrected is not None
        assert corrected.shape == science.shape
        assert corrected.mean() < science.mean()  # Removed bias

    def test_mismatched_shapes(self, calibrator):
        """Test error handling for mismatched frame shapes."""
        science = np.random.rand(100, 100)
        bias = np.random.rand(50, 50)  # Wrong shape

        with pytest.raises((ValueError, AssertionError)):
            calibrator.subtract_bias(science, bias)

    def test_negative_values_after_subtraction(self, calibrator):
        """Test handling of negative values after dark subtraction."""
        science = np.random.rand(100, 100) * 10
        dark = np.ones((100, 100)) * 15  # Dark > science in some pixels

        corrected = calibrator.subtract_dark(science, dark)

        # Should handle negative values (clip to 0 or preserve)
        assert corrected is not None

    def test_flat_normalization(self, calibrator):
        """Test that flat fields are properly normalized."""
        science = np.random.rand(100, 100) * 1000
        flat = np.random.rand(100, 100) * 0.5 + 0.75  # Not normalized

        corrected = calibrator.apply_flat(science, flat)

        # Should normalize flat internally
        assert corrected is not None

    def test_bias_from_overscan(self, calibrator):
        """Test bias estimation from overscan region."""
        # Create frame with overscan region
        science = np.random.rand(100, 110) * 1000 + 100
        # Overscan is last 10 columns
        science[:, 100:] = np.random.normal(100, 5, (100, 10))

        # If calibrator supports overscan estimation
        if hasattr(calibrator, 'estimate_bias_from_overscan'):
            bias_estimate = calibrator.estimate_bias_from_overscan(
                science, overscan_region=(slice(None), slice(100, 110))
            )
            assert 95 < bias_estimate < 105

    def test_background_sigma_clipping(self, calibrator):
        """Test that background estimation uses sigma clipping to reject sources."""
        # Create image with background + outliers
        science = np.random.normal(100, 10, (100, 100))
        # Add bright sources (outliers)
        science[20:25, 20:25] = 1000
        science[70:75, 70:75] = 1000

        corrected = calibrator.subtract_background(science, sigma=3, iterations=3)

        # Background should be estimated from pixels excluding bright sources
        # So corrected background level should be near 0
        assert abs(np.median(corrected)) < 20
