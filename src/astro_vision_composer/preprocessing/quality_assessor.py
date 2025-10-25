"""Quality assessment for astronomical images.

This module provides the QualityAssessor class for analyzing image quality
without AI - using statistical methods to calculate SNR, detect saturation,
estimate noise, and measure dynamic range.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from astropy.stats import sigma_clipped_stats, mad_std
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality assessment report for an astronomical image.

    Attributes:
        snr: Signal-to-noise ratio (median signal / background RMS)
        background_mean: Mean background level
        background_median: Median background level
        background_std: Standard deviation of background
        noise_estimate: Estimated noise level (MAD-based robust estimator)
        saturated_pixels: Number of saturated pixels
        saturation_fraction: Fraction of pixels that are saturated
        dynamic_range: Ratio of max to background level
        data_min: Minimum pixel value
        data_max: Maximum pixel value
        data_median: Median pixel value
        has_negative_values: Whether image contains negative pixels
        quality_score: Overall quality score (0-10)
        warnings: List of quality warnings
    """
    snr: float
    background_mean: float
    background_median: float
    background_std: float
    noise_estimate: float
    saturated_pixels: int
    saturation_fraction: float
    dynamic_range: float
    data_min: float
    data_max: float
    data_median: float
    has_negative_values: bool
    quality_score: float
    warnings: list

    def __repr__(self):
        """Pretty representation."""
        return (
            f"QualityReport(SNR={self.snr:.1f}, "
            f"Quality={self.quality_score:.1f}/10, "
            f"Saturation={self.saturation_fraction*100:.2f}%, "
            f"Warnings={len(self.warnings)})"
        )


class QualityAssessor:
    """Assess astronomical image quality using statistical methods.

    This class provides automated quality assessment without AI dependency,
    using robust statistical techniques to evaluate SNR, saturation,
    noise characteristics, and dynamic range.

    Examples:
        >>> from astropy.io import fits
        >>> assessor = QualityAssessor()
        >>> data = fits.getdata('image.fits')
        >>> report = assessor.assess_quality(data)
        >>> print(f"SNR: {report.snr:.1f}, Quality: {report.quality_score}/10")
        >>> if report.warnings:
        ...     print("Warnings:", report.warnings)
    """

    def __init__(
        self,
        saturation_threshold: Optional[float] = None,
        background_percentile: float = 25.0,
        sigma_clip: float = 3.0
    ):
        """Initialize quality assessor.

        Args:
            saturation_threshold: Pixel value threshold for saturation detection
                                 If None, will use 95th percentile heuristic
            background_percentile: Percentile to use for background estimation (default 25%)
            sigma_clip: Sigma threshold for sigma-clipped statistics (default 3.0)
        """
        self.saturation_threshold = saturation_threshold
        self.background_percentile = background_percentile
        self.sigma_clip = sigma_clip

    def assess_quality(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None
    ) -> QualityReport:
        """Assess the quality of an astronomical image.

        Args:
            data: 2D image array
            mask: Optional boolean mask (True = bad pixels to exclude)
            dq: Optional data quality bitmask array

        Returns:
            QualityReport object with assessment results
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {data.shape}")

        # Create combined mask
        combined_mask = self._create_mask(data, mask, dq)

        # Get valid pixels
        valid_data = data[~combined_mask] if combined_mask is not None else data.flatten()

        if len(valid_data) == 0:
            logger.warning("No valid pixels in image")
            return self._create_empty_report()

        warnings = []

        # Basic statistics
        data_min = float(np.min(valid_data))
        data_max = float(np.max(valid_data))
        data_median = float(np.median(valid_data))
        has_negative = data_min < 0

        if has_negative:
            warnings.append(f"Image contains negative values (min={data_min:.2e})")

        # Background statistics using sigma-clipped stats
        try:
            bg_mean, bg_median, bg_std = sigma_clipped_stats(
                valid_data,
                sigma=self.sigma_clip,
                maxiters=5
            )
        except Exception as e:
            logger.warning(f"Sigma-clipped stats failed: {e}, using simple stats")
            bg_mean = float(np.mean(valid_data))
            bg_median = float(np.median(valid_data))
            bg_std = float(np.std(valid_data))

        # Noise estimate using MAD (more robust than std)
        noise_estimate = mad_std(valid_data, ignore_nan=True)

        # SNR calculation
        signal_level = data_median - bg_median
        if bg_std > 0:
            snr = abs(signal_level) / bg_std
        else:
            snr = 0.0
            warnings.append("Background STD is zero - cannot calculate SNR")

        # Saturation detection
        if self.saturation_threshold is not None:
            sat_threshold = self.saturation_threshold
        else:
            # Heuristic: use 95th percentile or data_max - 0.01*range
            sat_threshold = np.percentile(valid_data, 95)

        saturated_mask = data >= sat_threshold
        saturated_pixels = int(np.sum(saturated_mask))
        saturation_fraction = saturated_pixels / data.size

        if saturation_fraction > 0.01:  # More than 1% saturated
            warnings.append(
                f"High saturation: {saturation_fraction*100:.2f}% of pixels "
                f"(threshold={sat_threshold:.2e})"
            )

        # Dynamic range
        if bg_median > 0:
            dynamic_range = data_max / bg_median
        else:
            dynamic_range = float('inf')
            warnings.append("Background median is zero or negative")

        # Overall quality score (0-10)
        quality_score = self._calculate_quality_score(
            snr=snr,
            saturation_fraction=saturation_fraction,
            dynamic_range=dynamic_range,
            noise_estimate=noise_estimate,
            background_std=bg_std
        )

        # Create report
        report = QualityReport(
            snr=float(snr),
            background_mean=float(bg_mean),
            background_median=float(bg_median),
            background_std=float(bg_std),
            noise_estimate=float(noise_estimate),
            saturated_pixels=saturated_pixels,
            saturation_fraction=float(saturation_fraction),
            dynamic_range=float(dynamic_range),
            data_min=data_min,
            data_max=data_max,
            data_median=data_median,
            has_negative_values=has_negative,
            quality_score=float(quality_score),
            warnings=warnings
        )

        logger.debug(f"Quality assessment complete: {report}")

        return report

    def calculate_snr(
        self,
        data: np.ndarray,
        source_region: Optional[Tuple[slice, slice]] = None,
        background_region: Optional[Tuple[slice, slice]] = None
    ) -> float:
        """Calculate SNR for a specific source region.

        Args:
            data: 2D image array
            source_region: Tuple of (y_slice, x_slice) for source
            background_region: Tuple of (y_slice, x_slice) for background

        Returns:
            Signal-to-noise ratio
        """
        if source_region is None:
            # Use central region as source
            center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
            box_size = min(50, data.shape[0] // 4, data.shape[1] // 4)
            source_region = (
                slice(center_y - box_size, center_y + box_size),
                slice(center_x - box_size, center_x + box_size)
            )

        if background_region is None:
            # Use outer edges as background
            border = 50
            background_region = (
                slice(0, border),
                slice(0, data.shape[1])
            )

        source_data = data[source_region]
        background_data = data[background_region]

        signal = np.median(source_data)
        bg_mean, bg_median, bg_std = sigma_clipped_stats(background_data, sigma=3.0)

        if bg_std > 0:
            snr = (signal - bg_median) / bg_std
        else:
            snr = 0.0

        return float(snr)

    def detect_saturation(
        self,
        data: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Detect saturated pixels in image.

        Args:
            data: 2D image array
            threshold: Saturation threshold (if None, use auto-detection)

        Returns:
            Tuple of (saturation_mask, num_saturated, fraction_saturated)
        """
        if threshold is None:
            threshold = np.percentile(data[np.isfinite(data)], 99.5)

        sat_mask = data >= threshold
        num_saturated = int(np.sum(sat_mask))
        fraction = num_saturated / data.size

        return sat_mask, num_saturated, fraction

    def estimate_noise(
        self,
        data: np.ndarray,
        method: str = 'mad'
    ) -> float:
        """Estimate noise level in image.

        Args:
            data: 2D image array
            method: Noise estimation method ('mad', 'std', 'percentile')

        Returns:
            Estimated noise level
        """
        valid_data = data[np.isfinite(data)]

        if method == 'mad':
            # Median Absolute Deviation (robust to outliers)
            noise = mad_std(valid_data, ignore_nan=True)
        elif method == 'std':
            # Standard deviation of sigma-clipped data
            _, _, noise = sigma_clipped_stats(valid_data, sigma=3.0)
        elif method == 'percentile':
            # Use 68.27th percentile (1-sigma for Gaussian)
            median = np.median(valid_data)
            deviations = np.abs(valid_data - median)
            noise = np.percentile(deviations, 68.27)
        else:
            raise ValueError(f"Unknown noise estimation method: {method}")

        return float(noise)

    def _create_mask(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray],
        dq: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Create combined mask from user mask and DQ array."""
        combined = None

        # Start with NaN/Inf mask
        combined = ~np.isfinite(data)

        # Add user mask
        if mask is not None:
            combined = combined | mask

        # Add DQ mask (any flag set = bad)
        if dq is not None:
            combined = combined | (dq != 0)

        return combined if np.any(combined) else None

    def _calculate_quality_score(
        self,
        snr: float,
        saturation_fraction: float,
        dynamic_range: float,
        noise_estimate: float,
        background_std: float
    ) -> float:
        """Calculate overall quality score from 0-10.

        Higher is better. Scoring criteria:
        - SNR contribution: 0-4 points
        - Saturation penalty: 0-2 points
        - Dynamic range contribution: 0-2 points
        - Noise quality: 0-2 points
        """
        score = 0.0

        # SNR contribution (0-4 points)
        if snr > 100:
            score += 4.0
        elif snr > 50:
            score += 3.5
        elif snr > 20:
            score += 3.0
        elif snr > 10:
            score += 2.5
        elif snr > 5:
            score += 2.0
        elif snr > 2:
            score += 1.0
        elif snr > 1:
            score += 0.5

        # Saturation penalty (0-2 points, more saturation = fewer points)
        if saturation_fraction < 0.001:  # < 0.1%
            score += 2.0
        elif saturation_fraction < 0.01:  # < 1%
            score += 1.5
        elif saturation_fraction < 0.05:  # < 5%
            score += 1.0
        elif saturation_fraction < 0.1:  # < 10%
            score += 0.5

        # Dynamic range contribution (0-2 points)
        if dynamic_range > 10000:
            score += 2.0
        elif dynamic_range > 1000:
            score += 1.5
        elif dynamic_range > 100:
            score += 1.0
        elif dynamic_range > 10:
            score += 0.5

        # Noise quality (0-2 points, lower noise_estimate/background_std is better)
        if background_std > 0:
            noise_ratio = noise_estimate / background_std
            if noise_ratio < 1.1:  # Very consistent
                score += 2.0
            elif noise_ratio < 1.3:
                score += 1.5
            elif noise_ratio < 1.5:
                score += 1.0
            elif noise_ratio < 2.0:
                score += 0.5

        return min(10.0, max(0.0, score))

    def _create_empty_report(self) -> QualityReport:
        """Create empty report for invalid data."""
        return QualityReport(
            snr=0.0,
            background_mean=0.0,
            background_median=0.0,
            background_std=0.0,
            noise_estimate=0.0,
            saturated_pixels=0,
            saturation_fraction=0.0,
            dynamic_range=0.0,
            data_min=0.0,
            data_max=0.0,
            data_median=0.0,
            has_negative_values=False,
            quality_score=0.0,
            warnings=["No valid pixels in image"]
        )
