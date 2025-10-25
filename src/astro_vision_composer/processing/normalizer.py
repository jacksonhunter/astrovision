"""Image normalization (scaling) for astronomical data.

This module provides the Normalizer class for scaling image data to a standard
range using various interval selection methods.
"""

from typing import Optional, Tuple, Literal
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import astropy visualization tools
try:
    from astropy.visualization import (
        MinMaxInterval, PercentileInterval, AsymmetricPercentileInterval,
        ZScaleInterval, ManualInterval
    )
    ASTROPY_VIZ_AVAILABLE = True
except ImportError:
    ASTROPY_VIZ_AVAILABLE = False
    logger.warning(
        "astropy.visualization not available. "
        "Install/update astropy: pip install astropy"
    )


IntervalMethod = Literal['minmax', 'percentile', 'zscale', 'manual']


class Normalizer:
    """Normalize astronomical image data to a standard range.

    This class provides methods to scale image data from arbitrary units
    (e.g., electrons/sec, ADU) to a normalized range suitable for stretching
    and display. It uses astropy.visualization interval classes.

    Common interval methods:
    - 'minmax': Simple min/max scaling (rarely used for astronomy)
    - 'percentile': Scale based on percentiles (e.g., 1% to 99%)
    - 'zscale': IRAF ZScale algorithm (good default for most images)
    - 'manual': User-specified min/max values

    Example:
        >>> import numpy as np
        >>> normalizer = Normalizer()
        >>> # Load some data
        >>> data = np.random.randn(1000, 1000) * 100 + 1000
        >>> # Normalize using ZScale
        >>> normalized = normalizer.normalize(data, method='zscale')
        >>> print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    """

    def __init__(self):
        """Initialize the Normalizer.

        Raises:
            ImportError: If astropy.visualization is not available
        """
        if not ASTROPY_VIZ_AVAILABLE:
            raise ImportError(
                "astropy.visualization is required. "
                "Install/update astropy: pip install astropy"
            )

    def normalize(
        self,
        data: np.ndarray,
        method: IntervalMethod = 'zscale',
        **kwargs
    ) -> np.ndarray:
        """Normalize image data to [0, 1] range.

        Args:
            data: Input image data
            method: Interval method to use
            **kwargs: Method-specific parameters (see individual methods)

        Returns:
            Normalized data array (values in [0, 1])

        Example:
            >>> # ZScale (default)
            >>> norm_data = normalizer.normalize(data, method='zscale')
            >>>
            >>> # Percentile with custom range
            >>> norm_data = normalizer.normalize(data, method='percentile',
            ...                                   lower=2, upper=98)
            >>>
            >>> # Manual interval
            >>> norm_data = normalizer.normalize(data, method='manual',
            ...                                   vmin=100, vmax=5000)
        """
        if data is None or data.size == 0:
            raise ValueError("data is empty or None")

        # Remove NaN/inf values for interval calculation
        finite_data = data[np.isfinite(data)]
        if finite_data.size == 0:
            logger.warning("No finite values in data, returning zeros")
            return np.zeros_like(data)

        # Select and apply interval method
        if method == 'minmax':
            interval = self._get_minmax_interval(**kwargs)
        elif method == 'percentile':
            interval = self._get_percentile_interval(**kwargs)
        elif method == 'zscale':
            interval = self._get_zscale_interval(**kwargs)
        elif method == 'manual':
            interval = self._get_manual_interval(**kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Must be one of: minmax, percentile, zscale, manual"
            )

        # Get the interval limits
        vmin, vmax = interval.get_limits(finite_data)

        logger.debug(
            f"Normalizing with {method}: vmin={vmin:.3e}, vmax={vmax:.3e}"
        )

        # Normalize data to [0, 1]
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)

        # Handle NaN/inf in original data
        normalized[~np.isfinite(data)] = 0

        return normalized

    def _get_minmax_interval(self) -> MinMaxInterval:
        """Get MinMaxInterval (simple min/max scaling)."""
        return MinMaxInterval()

    def _get_percentile_interval(
        self,
        lower: float = 1.0,
        upper: float = 99.0
    ) -> PercentileInterval:
        """Get PercentileInterval.

        Args:
            lower: Lower percentile (default: 1.0)
            upper: Upper percentile (default: 99.0)

        Returns:
            PercentileInterval object
        """
        if lower < 0 or upper > 100 or lower >= upper:
            raise ValueError(
                f"Invalid percentiles: lower={lower}, upper={upper}. "
                f"Must satisfy 0 <= lower < upper <= 100"
            )
        return PercentileInterval(percentile=upper, n_samples=10000)

    def _get_zscale_interval(
        self,
        nsamples: int = 10000,
        contrast: float = 0.25,
        max_reject: float = 0.5,
        min_npixels: int = 5,
        krej: float = 2.5,
        max_iterations: int = 5
    ) -> ZScaleInterval:
        """Get ZScaleInterval (IRAF ZScale algorithm).

        The ZScale algorithm is designed to display the bulk of the pixels
        while rejecting pixels that deviate from the linear fit.

        Args:
            nsamples: Number of samples to use
            contrast: Contrast parameter (default: 0.25)
            max_reject: Maximum fraction of pixels to reject (default: 0.5)
            min_npixels: Minimum number of pixels (default: 5)
            krej: Sigma-clipping threshold (default: 2.5)
            max_iterations: Maximum iterations (default: 5)

        Returns:
            ZScaleInterval object
        """
        return ZScaleInterval(
            nsamples=nsamples,
            contrast=contrast,
            max_reject=max_reject,
            min_npixels=min_npixels,
            krej=krej,
            max_iterations=max_iterations
        )

    def _get_manual_interval(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> ManualInterval:
        """Get ManualInterval (user-specified limits).

        Args:
            vmin: Minimum value (required)
            vmax: Maximum value (required)

        Returns:
            ManualInterval object

        Raises:
            ValueError: If vmin or vmax not specified
        """
        if vmin is None or vmax is None:
            raise ValueError("Manual interval requires both vmin and vmax")

        if vmin >= vmax:
            raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax})")

        return ManualInterval(vmin=vmin, vmax=vmax)

    def get_interval_limits(
        self,
        data: np.ndarray,
        method: IntervalMethod = 'zscale',
        **kwargs
    ) -> Tuple[float, float]:
        """Get interval limits without normalizing the data.

        Useful for inspecting what values will be used for normalization.

        Args:
            data: Input image data
            method: Interval method to use
            **kwargs: Method-specific parameters

        Returns:
            Tuple of (vmin, vmax) values

        Example:
            >>> vmin, vmax = normalizer.get_interval_limits(data, method='zscale')
            >>> print(f"ZScale interval: {vmin:.2f} to {vmax:.2f}")
        """
        if data is None or data.size == 0:
            raise ValueError("data is empty or None")

        # Remove NaN/inf
        finite_data = data[np.isfinite(data)]
        if finite_data.size == 0:
            logger.warning("No finite values in data")
            return (0.0, 0.0)

        # Get interval
        if method == 'minmax':
            interval = self._get_minmax_interval(**kwargs)
        elif method == 'percentile':
            interval = self._get_percentile_interval(**kwargs)
        elif method == 'zscale':
            interval = self._get_zscale_interval(**kwargs)
        elif method == 'manual':
            interval = self._get_manual_interval(**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'")

        vmin, vmax = interval.get_limits(finite_data)
        return (float(vmin), float(vmax))
