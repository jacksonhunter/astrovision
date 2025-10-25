"""Non-linear stretching transformations for astronomical images.

This module provides the Stretcher class for applying non-linear transformations
to normalized image data to enhance faint features and compress dynamic range.
"""

from typing import Literal
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import astropy visualization stretches
try:
    from astropy.visualization import (
        LinearStretch, SqrtStretch, SquaredStretch, LogStretch,
        AsinhStretch, SinhStretch, PowerStretch, PowerDistStretch,
        InvertedStretch, HistEqStretch, ContrastBiasStretch
    )
    ASTROPY_VIZ_AVAILABLE = True
except ImportError:
    ASTROPY_VIZ_AVAILABLE = False
    logger.warning(
        "astropy.visualization not available. "
        "Install/update astropy: pip install astropy"
    )


StretchMethod = Literal[
    'linear', 'sqrt', 'squared', 'log', 'asinh', 'sinh', 'power', 'histeq'
]


class Stretcher:
    """Apply non-linear stretches to normalized astronomical image data.

    Non-linear stretches compress the dynamic range of astronomical images,
    making both bright and faint features visible simultaneously. Different
    stretches are suitable for different types of images.

    Common stretches:
    - 'linear': No transformation (identity)
    - 'sqrt': Square root (mild stretch, good for most images)
    - 'log': Logarithmic (aggressive compression of bright features)
    - 'asinh': Inverse hyperbolic sine (excellent for wide dynamic range)
    - 'power': Power law (gamma correction, adjustable)
    - 'histeq': Histogram equalization (maximize local contrast)

    Note: Input data should be normalized to [0, 1] range first using Normalizer.

    Example:
        >>> from astro_vision_composer.processing import Normalizer, Stretcher
        >>> normalizer = Normalizer()
        >>> stretcher = Stretcher()
        >>>
        >>> # First normalize
        >>> normalized = normalizer.normalize(data, method='zscale')
        >>>
        >>> # Then stretch
        >>> stretched = stretcher.stretch(normalized, method='asinh', a=0.1)
    """

    def __init__(self):
        """Initialize the Stretcher.

        Raises:
            ImportError: If astropy.visualization is not available
        """
        if not ASTROPY_VIZ_AVAILABLE:
            raise ImportError(
                "astropy.visualization is required. "
                "Install/update astropy: pip install astropy"
            )

    def stretch(
        self,
        data: np.ndarray,
        method: StretchMethod = 'asinh',
        **kwargs
    ) -> np.ndarray:
        """Apply non-linear stretch to normalized data.

        Args:
            data: Normalized input data (should be in [0, 1] range)
            method: Stretch method to apply
            **kwargs: Method-specific parameters

        Returns:
            Stretched data array

        Example:
            >>> # Asinh stretch (default)
            >>> stretched = stretcher.stretch(data, method='asinh', a=0.1)
            >>>
            >>> # Log stretch
            >>> stretched = stretcher.stretch(data, method='log')
            >>>
            >>> # Power stretch (gamma correction)
            >>> stretched = stretcher.stretch(data, method='power', power=2.0)
        """
        if data is None or data.size == 0:
            raise ValueError("data is empty or None")

        # Get the stretch function
        if method == 'linear':
            stretch_fn = self._get_linear_stretch(**kwargs)
        elif method == 'sqrt':
            stretch_fn = self._get_sqrt_stretch(**kwargs)
        elif method == 'squared':
            stretch_fn = self._get_squared_stretch(**kwargs)
        elif method == 'log':
            stretch_fn = self._get_log_stretch(**kwargs)
        elif method == 'asinh':
            stretch_fn = self._get_asinh_stretch(**kwargs)
        elif method == 'sinh':
            stretch_fn = self._get_sinh_stretch(**kwargs)
        elif method == 'power':
            stretch_fn = self._get_power_stretch(**kwargs)
        elif method == 'histeq':
            stretch_fn = self._get_histeq_stretch(**kwargs)
        else:
            raise ValueError(
                f"Unknown stretch method '{method}'. "
                f"Must be one of: linear, sqrt, squared, log, asinh, sinh, power, histeq"
            )

        logger.debug(f"Applying {method} stretch")

        # Apply stretch
        stretched = stretch_fn(data)

        # Ensure output is finite and in [0, 1]
        stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
        stretched = np.clip(stretched, 0, 1)

        return stretched

    def _get_linear_stretch(self) -> LinearStretch:
        """Get LinearStretch (identity function)."""
        return LinearStretch()

    def _get_sqrt_stretch(self) -> SqrtStretch:
        """Get SqrtStretch (square root).

        Good for most astronomical images, provides mild compression
        of bright features while preserving faint details.
        """
        return SqrtStretch()

    def _get_squared_stretch(self) -> SquaredStretch:
        """Get SquaredStretch (inverse of sqrt).

        Enhances bright features while suppressing faint ones.
        Rarely used for visualization, mostly for special cases.
        """
        return SquaredStretch()

    def _get_log_stretch(self, a: float = 1000.0) -> LogStretch:
        """Get LogStretch (logarithmic).

        Aggressive compression of bright features. Good for images
        with extremely bright point sources (e.g., stars).

        Args:
            a: Scaling parameter (default: 1000.0)

        Returns:
            LogStretch object
        """
        return LogStretch(a=a)

    def _get_asinh_stretch(self, a: float = 0.1) -> AsinhStretch:
        """Get AsinhStretch (inverse hyperbolic sine).

        Excellent for images with wide dynamic range. Behaves linearly
        for faint features and logarithmically for bright features.
        This is often the best choice for astronomical composites.

        Args:
            a: Softening parameter (default: 0.1)
               - Smaller values: more aggressive stretch
               - Larger values: more linear behavior

        Returns:
            AsinhStretch object
        """
        if a <= 0:
            raise ValueError(f"Parameter 'a' must be positive, got {a}")
        return AsinhStretch(a=a)

    def _get_sinh_stretch(self, a: float = 0.33) -> SinhStretch:
        """Get SinhStretch (hyperbolic sine).

        Inverse of asinh. Enhances bright features.

        Args:
            a: Scaling parameter (default: 0.33)

        Returns:
            SinhStretch object
        """
        return SinhStretch(a=a)

    def _get_power_stretch(self, power: float = 2.0) -> PowerStretch:
        """Get PowerStretch (power law / gamma correction).

        Raises data to a power. Values > 1 darken the image (compress brights),
        values < 1 brighten it (enhance faint features).

        Args:
            power: Power exponent (default: 2.0)
                  - power > 1: darken/compress
                  - power < 1: brighten/enhance
                  - power = 1: no change

        Returns:
            PowerStretch object
        """
        if power <= 0:
            raise ValueError(f"Power must be positive, got {power}")
        return PowerStretch(a=power)

    def _get_histeq_stretch(self, nbins: int = 256) -> HistEqStretch:
        """Get HistEqStretch (histogram equalization).

        Maximizes local contrast by flattening the histogram.
        Can produce dramatic results but may introduce artifacts.

        Args:
            nbins: Number of histogram bins (default: 256)

        Returns:
            HistEqStretch object
        """
        return HistEqStretch(data=None)  # Will use passed data

    def apply_combined(
        self,
        data: np.ndarray,
        stretches: list
    ) -> np.ndarray:
        """Apply multiple stretches in sequence.

        Args:
            data: Input data
            stretches: List of (method, kwargs) tuples

        Returns:
            Data with all stretches applied

        Example:
            >>> # Apply sqrt then asinh
            >>> stretched = stretcher.apply_combined(
            ...     data,
            ...     [('sqrt', {}), ('asinh', {'a': 0.2})]
            ... )
        """
        result = data.copy()
        for method, kwargs in stretches:
            result = self.stretch(result, method=method, **kwargs)
        return result
