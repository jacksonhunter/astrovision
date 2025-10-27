"""Non-linear stretching transformations for astronomical images.

This module provides the Stretcher class for applying non-linear transformations
to normalized image data to enhance faint features and compress dynamic range.
"""

from __future__ import annotations
from typing import Literal, TYPE_CHECKING, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import astropy visualization stretches
from astropy.visualization import (
    LinearStretch, SqrtStretch, SquaredStretch, LogStretch,
    AsinhStretch, SinhStretch, PowerStretch, HistEqStretch,
    ContrastBiasStretch
)


StretchMethod = Literal[
    'linear', 'sqrt', 'squared', 'log', 'asinh', 'sinh', 'power', 'histeq', 'contrast_bias'
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
        """Initialize the Stretcher."""
        self._last_stretch_obj = None

    def stretch(
        self,
        data: np.ndarray,
        method: StretchMethod = 'asinh',
        **kwargs
    ) -> np.ndarray:
        """Apply non-linear stretch to normalized data.

        Raises:
            ImportError: If astropy.visualization is not available

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

        logger.debug(f"Applying {method} stretch")

        # HistEqStretch is special - it needs data at initialization
        # and we need to handle 2D data by flattening for both init and call
        if method == 'histeq':
            original_shape = data.shape
            data_flat = data.flatten()
            stretch_fn = self._get_histeq_stretch(data_flat, **kwargs)
            stretched_flat = stretch_fn(data_flat)
            stretched = stretched_flat.reshape(original_shape)
        else:
            # Get the stretch function for other methods
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
            elif method == 'contrast_bias':
                stretch_fn = self._get_contrast_bias_stretch(**kwargs)
            else:
                raise ValueError(
                    f"Unknown stretch method '{method}'. "
                    f"Must be one of: linear, sqrt, squared, log, asinh, sinh, power, histeq, contrast_bias"
                )

            # Apply stretch
            stretched = stretch_fn(data)

        # Store stretch object for Phase 3 use (can be pickled for workflow continuity)
        self._last_stretch_obj = stretch_fn

        # Ensure output is finite and in [0, 1]
        stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
        stretched = np.clip(stretched, 0, 1)

        return stretched

    def get_stretch_object(self):
        """Get the last used stretch object.

        Returns the astropy stretch object from the most recent stretch() call.
        This object can be pickled and passed to Phase 3 for workflow continuity.

        Returns:
            BaseStretch: The last used astropy stretch object, or None if no stretch applied yet

        Example:
            >>> stretcher = Stretcher()
            >>> stretched = stretcher.stretch(data, method='asinh', a=0.1)
            >>> stretch_obj = stretcher.get_stretch_object()
            >>> # Can pickle for Phase 3
            >>> import pickle
            >>> pickled = pickle.dumps(stretch_obj)
        """
        return self._last_stretch_obj

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

    def _get_histeq_stretch(self, data: np.ndarray) -> HistEqStretch:
        """Get HistEqStretch (histogram equalization).

        Maximizes local contrast by flattening the histogram.
        Can produce dramatic results but may introduce artifacts.

        Note: HistEqStretch requires data at initialization to compute histogram.
        Data should already be flattened to 1D.

        Args:
            data: 1D array of image data for computing histogram

        Returns:
            HistEqStretch object initialized with data
        """
        # HistEqStretch needs finite values only for histogram calculation
        finite_data = data[np.isfinite(data)]

        if finite_data.size == 0:
            logger.warning("No finite data for HistEq, using LinearStretch instead")
            return LinearStretch()

        # HistEqStretch expects data parameter only (values is optional and for output mapping)
        # Pass the finite 1D data
        return HistEqStretch(data=finite_data)

    def _get_contrast_bias_stretch(self, contrast: float = 1.0, bias: float = 0.5) -> ContrastBiasStretch:
        """Get ContrastBiasStretch (contrast and bias adjustment).

        Adjusts contrast and bias of the image. The stretch is given by:
        y = (x - bias) * contrast + 0.5

        Args:
            contrast: Contrast parameter (default: 1.0)
                     - contrast > 1: increase contrast
                     - contrast < 1: decrease contrast
                     - contrast = 1: no change
            bias: Bias parameter (default: 0.5, range [0, 1])
                  - bias > 0.5: shift towards darker values
                  - bias < 0.5: shift towards brighter values
                  - bias = 0.5: no shift

        Returns:
            ContrastBiasStretch object
        """
        if contrast <= 0:
            raise ValueError(f"Contrast must be positive, got {contrast}")
        if not (0 <= bias <= 1):
            raise ValueError(f"Bias must be in [0, 1] range, got {bias}")

        return ContrastBiasStretch(contrast=contrast, bias=bias)

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
