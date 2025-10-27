"""Unit tests for QualityAssessor class."""

import pytest
import numpy as np


class TestQualityAssessor:
    """Test QualityAssessor statistical quality analysis."""

    def test_assess_quality_basic(self, quality_assessor):
        """Test basic quality assessment on random data."""
        data = np.random.rand(100, 100).astype(np.float32)

        report = quality_assessor.assess_quality(data)

        assert report is not None
        assert hasattr(report, 'snr')
        assert hasattr(report, 'saturated_fraction')
        assert hasattr(report, 'dynamic_range')
        assert hasattr(report, 'noise_estimate')

    def test_snr_calculation(self, quality_assessor):
        """Test SNR calculation."""
        # Create data with known properties
        signal = np.ones((100, 100)) * 1000
        noise = np.random.normal(0, 10, (100, 100))
        data = signal + noise

        report = quality_assessor.assess_quality(data)

        # SNR should be roughly signal/noise = 1000/10 = 100
        assert report.snr > 50  # Allow some margin
        assert report.snr < 200

    def test_saturation_detection(self, quality_assessor):
        """Test saturation detection."""
        # Create data with saturated pixels
        data = np.random.rand(100, 100) * 0.5
        data[10:20, 10:20] = 1.0  # Saturate 100 pixels (1% of 10000)

        report = quality_assessor.assess_quality(data, saturation_threshold=0.99)

        assert report.saturated_fraction > 0
        assert report.saturated_fraction <= 0.02  # ~1% saturated

    def test_no_saturation(self, quality_assessor):
        """Test case with no saturated pixels."""
        data = np.random.rand(100, 100) * 0.5  # Max value is 0.5

        report = quality_assessor.assess_quality(data, saturation_threshold=0.9)

        assert report.saturated_fraction == 0

    def test_noise_estimation(self, quality_assessor):
        """Test noise estimation."""
        # Create data with known noise level
        data = np.random.normal(100, 5, (100, 100))  # Mean=100, std=5

        report = quality_assessor.assess_quality(data)

        # Noise estimate should be close to 5
        assert 3 < report.noise_estimate < 10

    def test_dynamic_range_calculation(self, quality_assessor):
        """Test dynamic range calculation."""
        # Create data with known min/max
        data = np.zeros((100, 100))
        data[50, 50] = 100  # Single bright pixel

        report = quality_assessor.assess_quality(data)

        # Dynamic range should be large (log10(max/noise))
        assert report.dynamic_range > 0

    def test_blank_image(self, quality_assessor):
        """Test assessment of blank (all zeros) image."""
        data = np.zeros((100, 100))

        report = quality_assessor.assess_quality(data)

        # Should handle gracefully
        assert report is not None
        # SNR might be 0 or inf depending on implementation
        assert report.saturated_fraction == 0

    def test_uniform_image(self, quality_assessor):
        """Test assessment of uniform (constant value) image."""
        data = np.ones((100, 100)) * 50

        report = quality_assessor.assess_quality(data)

        assert report is not None
        assert report.noise_estimate < 1.0  # Very low noise

    def test_high_noise_image(self, quality_assessor):
        """Test assessment of very noisy image."""
        # Pure noise, no signal
        data = np.random.normal(0, 100, (100, 100))

        report = quality_assessor.assess_quality(data)

        assert report is not None
        assert report.noise_estimate > 50  # High noise

    @pytest.mark.requires_noirlab_data
    def test_with_real_fits_data(self, quality_assessor, fits_loader, edu008_data):
        """Test quality assessment on real FITS data."""
        fits_file = edu008_data['fits_files'][0]
        fits_data = fits_loader.load(fits_file)

        report = quality_assessor.assess_quality(fits_data.data)

        assert report is not None
        assert report.snr > 0
        assert 0 <= report.saturated_fraction <= 1.0
        assert report.dynamic_range > 0

    def test_nan_handling(self, quality_assessor):
        """Test handling of NaN values in data."""
        data = np.random.rand(100, 100)
        data[50:60, 50:60] = np.nan

        # Should either handle NaNs gracefully or document expected behavior
        try:
            report = quality_assessor.assess_quality(data)
            # If it succeeds, check results make sense
            assert report is not None
        except ValueError:
            # Or it might raise an error, which is also acceptable if documented
            pass

    def test_negative_values(self, quality_assessor):
        """Test handling of negative pixel values."""
        # Some astronomical data can have negative values after background subtraction
        data = np.random.normal(0, 10, (100, 100))  # Mean 0, so half negative

        report = quality_assessor.assess_quality(data)

        assert report is not None

    def test_custom_saturation_threshold(self, quality_assessor):
        """Test custom saturation threshold."""
        data = np.random.rand(100, 100)
        data[0:10, 0:10] = 0.95

        # With high threshold, shouldn't be saturated
        report1 = quality_assessor.assess_quality(data, saturation_threshold=0.99)
        assert report1.saturated_fraction == 0

        # With low threshold, should be saturated
        report2 = quality_assessor.assess_quality(data, saturation_threshold=0.90)
        assert report2.saturated_fraction > 0
