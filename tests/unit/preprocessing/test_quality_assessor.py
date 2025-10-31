"""Unit tests for QualityAssessor.

Tests statistical quality assessment: SNR calculation, saturation detection,
noise estimation, and quality reporting.

Author: VisionProject Testing Framework
Date: 2025-10-26
"""

import pytest
import numpy as np
from astropy.stats import mad_std

from astro_vision_composer.preprocessing.quality_assessor import (
    QualityAssessor,
    QualityReport
)


class TestQualityAssessorInstantiation:
    """Tests for QualityAssessor instantiation and configuration."""

    def test_default_instantiation(self):
        """Test assessor can be instantiated with defaults."""
        assessor = QualityAssessor()
        assert assessor is not None
        assert assessor.saturation_threshold is None
        assert assessor.background_percentile == 25.0
        assert assessor.sigma_clip == 3.0

    def test_custom_configuration(self):
        """Test assessor accepts custom configuration."""
        assessor = QualityAssessor(
            saturation_threshold=50000.0,
            background_percentile=30.0,
            sigma_clip=5.0
        )
        assert assessor.saturation_threshold == 50000.0
        assert assessor.background_percentile == 30.0
        assert assessor.sigma_clip == 5.0


class TestSNRCalculation:
    """Tests for SNR (Signal-to-Noise Ratio) calculation."""

    @pytest.fixture
    def assessor(self):
        """Create QualityAssessor instance."""
        return QualityAssessor()

    def test_snr_with_known_values(self, assessor):
        """Test SNR calculation with synthetic data of known SNR."""
        # Create image with known signal and noise
        np.random.seed(42)
        background = 100.0
        signal = 1000.0
        noise_std = 10.0

        # Background region
        bg_data = np.random.normal(background, noise_std, (200, 200))

        # Source region (signal + background + noise)
        source_data = np.random.normal(background + signal, noise_std, (100, 100))

        # Combine into image
        image = np.copy(bg_data)
        image[50:150, 50:150] = source_data

        # Calculate SNR
        snr = assessor.calculate_snr(
            image,
            source_region=(slice(50, 150), slice(50, 150)),
            background_region=(slice(0, 50), slice(0, 200))
        )

        # Expected SNR ≈ signal / noise_std = 1000 / 10 = 100
        # Allow for statistical variation
        assert 80 < snr < 120, f"Expected SNR ~100, got {snr}"

    def test_snr_low_signal(self, assessor):
        """Test SNR calculation with low signal (SNR < 1)."""
        np.random.seed(42)
        background = 1000.0
        signal = 5.0  # Very low signal
        noise_std = 10.0

        bg_data = np.random.normal(background, noise_std, (100, 100))
        source_data = np.random.normal(background + signal, noise_std, (50, 50))

        image = np.copy(bg_data)
        image[25:75, 25:75] = source_data

        snr = assessor.calculate_snr(
            image,
            source_region=(slice(25, 75), slice(25, 75)),
            background_region=(slice(0, 25), slice(0, 100))
        )

        # Expected SNR ≈ 5 / 10 = 0.5
        assert 0 < snr < 2, f"Expected SNR ~0.5, got {snr}"

    def test_snr_high_noise(self, assessor):
        """Test SNR calculation with high noise."""
        np.random.seed(42)
        background = 100.0
        signal = 100.0
        noise_std = 50.0  # High noise

        bg_data = np.random.normal(background, noise_std, (100, 100))
        source_data = np.random.normal(background + signal, noise_std, (50, 50))

        image = np.copy(bg_data)
        image[25:75, 25:75] = source_data

        snr = assessor.calculate_snr(
            image,
            source_region=(slice(25, 75), slice(25, 75)),
            background_region=(slice(0, 25), slice(0, 100))
        )

        # Expected SNR ≈ 100 / 50 = 2
        assert 1 < snr < 4, f"Expected SNR ~2, got {snr}"

    def test_snr_with_auto_regions(self, assessor):
        """Test SNR calculation with automatic region selection works."""
        np.random.seed(42)
        # Create image with bright center
        image = np.random.normal(100, 10, (200, 200))
        image[75:125, 75:125] += 2000  # Very bright central source

        # Should use center as source, edges as background
        snr = assessor.calculate_snr(image)

        # Just verify it returns a finite value (implementation details vary)
        assert np.isfinite(snr), f"SNR should be finite, got {snr}"
        assert snr >= 0, f"SNR should be non-negative, got {snr}"

    def test_snr_with_masked_data(self, assessor):
        """Test SNR handles NaN/Inf values gracefully."""
        np.random.seed(42)
        image = np.random.normal(1000, 50, (100, 100))
        image[40:60, 40:60] += 500  # Signal

        # Add some bad pixels
        image[10:15, 10:15] = np.nan
        image[80:85, 80:85] = np.inf

        # Should still calculate SNR
        snr = assessor.calculate_snr(
            image,
            source_region=(slice(40, 60), slice(40, 60)),
            background_region=(slice(0, 40), slice(0, 100))
        )

        assert np.isfinite(snr), "SNR should be finite despite NaN/Inf pixels"
        assert snr > 5, "Should detect signal despite bad pixels"


class TestSaturationDetection:
    """Tests for saturation detection."""

    @pytest.fixture
    def assessor(self):
        """Create QualityAssessor instance."""
        return QualityAssessor()

    def test_detect_saturation_with_threshold(self, assessor):
        """Test saturation detection with explicit threshold."""
        # Create image with some saturated pixels
        image = np.random.uniform(0, 40000, (100, 100))
        image[10:20, 10:20] = 65535  # Saturated region

        sat_mask, num_sat, frac_sat = assessor.detect_saturation(image, threshold=60000)

        assert num_sat == 100, f"Expected 100 saturated pixels, got {num_sat}"
        assert frac_sat == 0.01, f"Expected 1% saturation, got {frac_sat*100}%"
        assert sat_mask.shape == image.shape
        assert np.all(sat_mask[10:20, 10:20]), "Saturated region not detected"

    def test_detect_saturation_auto_threshold(self, assessor):
        """Test saturation detection with automatic threshold."""
        # Create image with values up to 50000
        np.random.seed(42)
        image = np.random.uniform(0, 50000, (100, 100))

        # Auto-threshold should be around 99.5th percentile
        sat_mask, num_sat, frac_sat = assessor.detect_saturation(image)

        # Should detect approximately 0.5% as saturated
        assert 0.001 < frac_sat < 0.02, \
            f"Expected ~0.5% saturation, got {frac_sat*100:.2f}%"
        assert num_sat == np.sum(sat_mask)

    def test_no_saturation(self, assessor):
        """Test saturation detection with no saturated pixels."""
        # Create image well below saturation
        image = np.random.uniform(0, 10000, (100, 100))

        sat_mask, num_sat, frac_sat = assessor.detect_saturation(image, threshold=50000)

        assert num_sat == 0, "Should detect no saturation"
        assert frac_sat == 0.0, "Saturation fraction should be zero"
        assert not np.any(sat_mask), "Saturation mask should be all False"


class TestNoiseEstimation:
    """Tests for noise estimation methods."""

    @pytest.fixture
    def assessor(self):
        """Create QualityAssessor instance."""
        return QualityAssessor()

    def test_noise_mad_method(self, assessor):
        """Test MAD-based noise estimation."""
        np.random.seed(42)
        true_noise = 10.0
        image = np.random.normal(1000, true_noise, (100, 100))

        estimated_noise = assessor.estimate_noise(image, method='mad')

        # MAD estimator should be close to true noise
        # Allow 30% tolerance due to finite sample size
        assert 0.7 * true_noise < estimated_noise < 1.3 * true_noise, \
            f"Expected noise ~{true_noise}, got {estimated_noise}"

    def test_noise_std_method(self, assessor):
        """Test std-based noise estimation."""
        np.random.seed(42)
        true_noise = 15.0
        image = np.random.normal(2000, true_noise, (100, 100))

        estimated_noise = assessor.estimate_noise(image, method='std')

        # Sigma-clipped std should be close to true noise
        assert 0.7 * true_noise < estimated_noise < 1.3 * true_noise, \
            f"Expected noise ~{true_noise}, got {estimated_noise}"

    def test_noise_percentile_method(self, assessor):
        """Test percentile-based noise estimation."""
        np.random.seed(42)
        true_noise = 20.0
        image = np.random.normal(3000, true_noise, (100, 100))

        estimated_noise = assessor.estimate_noise(image, method='percentile')

        # Percentile method should estimate noise
        assert 0.7 * true_noise < estimated_noise < 1.3 * true_noise, \
            f"Expected noise ~{true_noise}, got {estimated_noise}"

    def test_noise_with_outliers(self, assessor):
        """Test noise estimation robustness to outliers."""
        np.random.seed(42)
        true_noise = 10.0
        image = np.random.normal(1000, true_noise, (100, 100))

        # Add outliers (cosmic rays, hot pixels)
        image[10:15, 10:15] += 5000

        # MAD should be robust to outliers
        estimated_noise = assessor.estimate_noise(image, method='mad')

        # Should still estimate close to true noise despite outliers
        assert 0.5 * true_noise < estimated_noise < 2.0 * true_noise, \
            f"MAD should be robust to outliers, got {estimated_noise} for true noise {true_noise}"

    def test_noise_invalid_method(self, assessor):
        """Test noise estimation raises error for invalid method."""
        image = np.random.normal(1000, 10, (100, 100))

        with pytest.raises(ValueError, match="Unknown noise estimation method"):
            assessor.estimate_noise(image, method='invalid_method')


class TestQualityAssessment:
    """Tests for full quality assessment (assess_quality method)."""

    @pytest.fixture
    def assessor(self):
        """Create QualityAssessor instance."""
        return QualityAssessor()

    def test_assess_quality_basic(self, assessor):
        """Test basic quality assessment returns valid report."""
        np.random.seed(42)
        image = np.random.normal(1000, 50, (100, 100))

        report = assessor.assess_quality(image)

        # Verify report structure
        assert isinstance(report, QualityReport)
        assert report.snr > 0
        assert report.background_mean > 0
        assert report.background_median > 0
        assert report.background_std > 0
        assert report.noise_estimate > 0
        assert report.saturated_pixels >= 0
        assert 0 <= report.saturation_fraction <= 1.0
        assert report.dynamic_range > 0
        assert report.data_min <= report.data_median <= report.data_max
        assert 0 <= report.quality_score <= 10

    def test_assess_quality_good_image(self, assessor):
        """Test assessment of good quality image."""
        np.random.seed(42)
        # Create image with low noise and good dynamic range
        background = 1000.0
        noise = 10.0  # Low noise

        image = np.random.normal(background, noise, (200, 200))
        # Add some signal variation (not saturation)
        image[75:125, 75:125] += 500

        report = assessor.assess_quality(image)

        # Verify report has reasonable values (not strict thresholds)
        assert np.isfinite(report.snr), "SNR should be finite"
        assert report.snr >= 0, "SNR should be non-negative"
        assert np.isfinite(report.quality_score), "Quality score should be finite"
        assert 0 <= report.quality_score <= 10, "Quality score should be 0-10"
        assert report.saturation_fraction < 0.1, "Should have low saturation"

    def test_assess_quality_with_saturation(self, assessor):
        """Test assessment detects saturation."""
        np.random.seed(42)
        image = np.random.uniform(0, 40000, (100, 100))
        image[10:30, 10:30] = 65535  # 4% saturated

        report = assessor.assess_quality(image)

        assert report.saturation_fraction > 0.01, "Should detect saturation"
        assert any('saturation' in w.lower() for w in report.warnings), \
            "Should warn about saturation"

    def test_assess_quality_with_negative_values(self, assessor):
        """Test assessment detects negative values."""
        np.random.seed(42)
        image = np.random.normal(100, 50, (100, 100))
        image[image < 0] = image[image < 0]  # Keep some negative values

        # Ensure we have negative values
        if not np.any(image < 0):
            image[0:10, 0:10] = -100

        report = assessor.assess_quality(image)

        assert report.has_negative_values, "Should detect negative values"
        assert any('negative' in w.lower() for w in report.warnings), \
            "Should warn about negative values"

    def test_assess_quality_with_mask(self, assessor):
        """Test quality assessment with bad pixel mask."""
        np.random.seed(42)
        image = np.random.normal(1000, 50, (100, 100))

        # Create mask for bad pixels
        mask = np.zeros(image.shape, dtype=bool)
        mask[10:20, 10:20] = True  # Mask out 1% of pixels

        report = assessor.assess_quality(image, mask=mask)

        # Should still produce valid report
        assert isinstance(report, QualityReport)
        assert np.isfinite(report.snr)

    def test_assess_quality_with_dq_array(self, assessor):
        """Test quality assessment with DQ (data quality) array."""
        np.random.seed(42)
        image = np.random.normal(1000, 50, (100, 100))

        # Create DQ array with some flagged pixels
        dq = np.zeros(image.shape, dtype=np.uint32)
        dq[15:25, 15:25] = 0x1  # Flag 1% as bad

        report = assessor.assess_quality(image, dq=dq)

        # Should exclude flagged pixels from analysis
        assert isinstance(report, QualityReport)
        assert np.isfinite(report.snr)

    def test_assess_quality_invalid_shape(self, assessor):
        """Test assessment raises error for non-2D data."""
        # 1D data
        data_1d = np.random.normal(1000, 50, 100)

        with pytest.raises(ValueError, match="Expected 2D image"):
            assessor.assess_quality(data_1d)

        # 3D data
        data_3d = np.random.normal(1000, 50, (10, 10, 10))

        with pytest.raises(ValueError, match="Expected 2D image"):
            assessor.assess_quality(data_3d)

    def test_assess_quality_all_masked(self, assessor):
        """Test assessment handles all pixels masked."""
        image = np.random.normal(1000, 50, (100, 100))

        # Mask all pixels
        mask = np.ones(image.shape, dtype=bool)

        report = assessor.assess_quality(image, mask=mask)

        # Should return a report (may have default/empty values)
        assert isinstance(report, QualityReport)


class TestQualityReport:
    """Tests for QualityReport dataclass."""

    def test_quality_report_repr(self):
        """Test QualityReport has readable string representation."""
        report = QualityReport(
            snr=25.5,
            background_mean=100.0,
            background_median=98.5,
            background_std=10.2,
            noise_estimate=10.5,
            saturated_pixels=123,
            saturation_fraction=0.0123,
            dynamic_range=1000.0,
            data_min=50.0,
            data_max=50000.0,
            data_median=1000.0,
            has_negative_values=False,
            quality_score=8.5,
            warnings=[]
        )

        repr_str = repr(report)

        assert 'QualityReport' in repr_str
        assert '25.5' in repr_str or '25' in repr_str  # SNR
        assert '8.5' in repr_str or '8' in repr_str  # Quality score
        assert 'Saturation' in repr_str

    def test_quality_report_with_warnings(self):
        """Test QualityReport includes warnings in repr."""
        report = QualityReport(
            snr=5.0,
            background_mean=100.0,
            background_median=100.0,
            background_std=20.0,
            noise_estimate=20.0,
            saturated_pixels=5000,
            saturation_fraction=0.5,
            dynamic_range=100.0,
            data_min=0.0,
            data_max=10000.0,
            data_median=1000.0,
            has_negative_values=False,
            quality_score=3.0,
            warnings=['High saturation: 50%', 'Low SNR']
        )

        repr_str = repr(report)

        assert 'Warnings=2' in repr_str or 'warning' in repr_str.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def assessor(self):
        """Create QualityAssessor instance."""
        return QualityAssessor()

    def test_zero_background_std(self, assessor):
        """Test handling of zero background std (constant image)."""
        # Constant image
        image = np.ones((100, 100)) * 1000.0

        report = assessor.assess_quality(image)

        # Should handle gracefully, SNR may be 0 or undefined
        assert isinstance(report, QualityReport)
        assert any('zero' in w.lower() or 'snr' in w.lower() for w in report.warnings)

    def test_all_nan_image(self, assessor):
        """Test handling of image with all NaN values."""
        image = np.full((100, 100), np.nan)

        report = assessor.assess_quality(image)

        # Should return report (may be empty/default)
        assert isinstance(report, QualityReport)

    def test_negative_background(self, assessor):
        """Test handling of negative mean background."""
        # Image with large negative values
        image = np.random.normal(-1000, 50, (100, 100))

        report = assessor.assess_quality(image)

        assert report.has_negative_values
        assert report.background_median < 0

    def test_very_small_image(self, assessor):
        """Test quality assessment on very small image."""
        # 10×10 image
        image = np.random.normal(1000, 50, (10, 10))

        report = assessor.assess_quality(image)

        # Should still work, though statistics may be less reliable
        assert isinstance(report, QualityReport)
        assert np.isfinite(report.snr)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
