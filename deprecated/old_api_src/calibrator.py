"""Image calibration for astronomical data.

This module provides the Calibrator class for fundamental CCD/detector calibration
steps including bias, dark, and flat-field corrections, as well as background subtraction.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class Calibrator:
    """Apply calibration corrections to astronomical images.

    Handles standard CCD reduction steps:
    - Bias/overscan correction
    - Dark current subtraction
    - Flat-field correction
    - Background subtraction

    Note: Most modern space telescope data (JWST, HST) is delivered pre-calibrated.
    This class is primarily useful for ground-based data or custom reductions.

    Example:
        >>> calibrator = Calibrator()
        >>> # Basic calibration
        >>> calibrated = calibrator.calibrate(
        ...     science=raw_data,
        ...     bias=master_bias,
        ...     dark=master_dark,
        ...     flat=master_flat
        ... )
    """

    def __init__(self):
        """Initialize the Calibrator."""
        pass

    def calibrate(
        self,
        science: np.ndarray,
        bias: Optional[np.ndarray] = None,
        dark: Optional[np.ndarray] = None,
        flat: Optional[np.ndarray] = None,
        exposure_time: Optional[float] = None,
        dark_exposure_time: Optional[float] = None
    ) -> np.ndarray:
        """Apply full calibration pipeline.

        Applies corrections in the standard order:
        1. Bias subtraction
        2. Dark subtraction (scaled by exposure time)
        3. Flat-field correction

        Args:
            science: Raw science image
            bias: Master bias frame
            dark: Master dark frame
            flat: Master flat-field frame
            exposure_time: Science exposure time (seconds)
            dark_exposure_time: Dark frame exposure time (seconds)

        Returns:
            Calibrated image array

        Example:
            >>> calibrated = calibrator.calibrate(
            ...     science=raw,
            ...     bias=bias_frame,
            ...     dark=dark_frame,
            ...     flat=flat_frame,
            ...     exposure_time=300.0,
            ...     dark_exposure_time=300.0
            ... )
        """
        result = science.copy()

        # Step 1: Bias subtraction
        if bias is not None:
            result = self.subtract_bias(result, bias)
            logger.debug("Applied bias correction")

        # Step 2: Dark subtraction
        if dark is not None:
            result = self.subtract_dark(
                result, dark, exposure_time, dark_exposure_time
            )
            logger.debug("Applied dark correction")

        # Step 3: Flat-field correction
        if flat is not None:
            result = self.apply_flat(result, flat)
            logger.debug("Applied flat-field correction")

        logger.info("Full calibration pipeline complete")
        return result

    def subtract_bias(
        self,
        data: np.ndarray,
        bias: np.ndarray
    ) -> np.ndarray:
        """Subtract bias frame from data.

        Args:
            data: Input data array
            bias: Master bias frame

        Returns:
            Bias-subtracted data

        Raises:
            ValueError: If shapes don't match
        """
        if data.shape != bias.shape:
            raise ValueError(
                f"Data shape {data.shape} does not match bias shape {bias.shape}"
            )

        return data - bias

    def subtract_dark(
        self,
        data: np.ndarray,
        dark: np.ndarray,
        exposure_time: Optional[float] = None,
        dark_exposure_time: Optional[float] = None
    ) -> np.ndarray:
        """Subtract dark frame from data, scaling by exposure time.

        Args:
            data: Input data array
            dark: Master dark frame
            exposure_time: Science exposure time (seconds)
            dark_exposure_time: Dark frame exposure time (seconds)

        Returns:
            Dark-subtracted data

        Raises:
            ValueError: If shapes don't match
        """
        if data.shape != dark.shape:
            raise ValueError(
                f"Data shape {data.shape} does not match dark shape {dark.shape}"
            )

        # Scale dark if exposure times differ
        if exposure_time is not None and dark_exposure_time is not None:
            scale = exposure_time / dark_exposure_time
            dark_scaled = dark * scale
            logger.debug(
                f"Scaled dark frame: {exposure_time}s / {dark_exposure_time}s = {scale:.3f}"
            )
        else:
            dark_scaled = dark

        return data - dark_scaled

    def apply_flat(
        self,
        data: np.ndarray,
        flat: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Apply flat-field correction.

        Args:
            data: Input data array
            flat: Master flat-field frame
            normalize: Normalize flat to median value (default: True)

        Returns:
            Flat-fielded data

        Raises:
            ValueError: If shapes don't match or flat has zeros
        """
        if data.shape != flat.shape:
            raise ValueError(
                f"Data shape {data.shape} does not match flat shape {flat.shape}"
            )

        # Normalize flat to median
        if normalize:
            flat_median = np.median(flat[flat > 0])
            flat_norm = flat / flat_median
        else:
            flat_norm = flat

        # Check for zero/negative values
        if np.any(flat_norm <= 0):
            logger.warning(
                f"Flat-field contains {np.sum(flat_norm <= 0)} zero/negative pixels"
            )
            # Replace zeros with small positive value
            flat_norm = np.where(flat_norm > 0, flat_norm, 0.001)

        return data / flat_norm

    def subtract_background(
        self,
        data: np.ndarray,
        method: str = 'median',
        sigma: float = 3.0,
        iterations: int = 3,
        box_size: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """Estimate and subtract background from image.

        Args:
            data: Input data array
            method: Background estimation method ('median', 'mean', 'mode')
            sigma: Sigma-clipping threshold
            iterations: Number of sigma-clipping iterations
            box_size: Box size for local background (None = global)

        Returns:
            Tuple of (background-subtracted data, estimated background level)

        Example:
            >>> corrected, bg_level = calibrator.subtract_background(
            ...     data, method='median', sigma=3.0
            ... )
            >>> print(f"Background level: {bg_level:.2f}")
        """
        if box_size is not None:
            # Local background estimation (not yet implemented)
            logger.warning("Local background estimation not implemented, using global")

        # Global background estimation with sigma-clipping
        background = self._estimate_background_sigmaclip(
            data, method=method, sigma=sigma, iterations=iterations
        )

        logger.info(f"Estimated background: {background:.2f} (method={method})")

        return data - background, background

    def _estimate_background_sigmaclip(
        self,
        data: np.ndarray,
        method: str = 'median',
        sigma: float = 3.0,
        iterations: int = 3
    ) -> float:
        """Estimate background using sigma-clipped statistics.

        Args:
            data: Input data array
            method: Estimator ('median', 'mean', 'mode')
            sigma: Sigma-clipping threshold
            iterations: Number of iterations

        Returns:
            Estimated background value
        """
        # Start with all finite values
        clipped = data[np.isfinite(data)].copy()

        for i in range(iterations):
            median = np.median(clipped)
            std = np.std(clipped)

            # Sigma clip
            mask = np.abs(clipped - median) < sigma * std
            clipped = clipped[mask]

            logger.debug(
                f"Iteration {i+1}: {len(clipped)} pixels retained "
                f"(median={median:.2f}, std={std:.2f})"
            )

        # Final estimator
        if method == 'median':
            return np.median(clipped)
        elif method == 'mean':
            return np.mean(clipped)
        elif method == 'mode':
            # Approximate mode as 3*median - 2*mean
            return 3 * np.median(clipped) - 2 * np.mean(clipped)
        else:
            raise ValueError(f"Unknown method: {method}")

    def create_master_bias(
        self,
        bias_frames: list,
        method: str = 'median'
    ) -> np.ndarray:
        """Create master bias from multiple bias frames.

        Args:
            bias_frames: List of bias frame arrays
            method: Combination method ('median' or 'mean')

        Returns:
            Master bias frame

        Example:
            >>> bias_frames = [bias1, bias2, bias3]
            >>> master_bias = calibrator.create_master_bias(bias_frames)
        """
        if not bias_frames:
            raise ValueError("bias_frames list is empty")

        stack = np.array(bias_frames)

        if method == 'median':
            master = np.median(stack, axis=0)
        elif method == 'mean':
            master = np.mean(stack, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Created master bias from {len(bias_frames)} frames ({method})")
        return master

    def create_master_dark(
        self,
        dark_frames: list,
        method: str = 'median'
    ) -> np.ndarray:
        """Create master dark from multiple dark frames.

        Args:
            dark_frames: List of dark frame arrays
            method: Combination method ('median' or 'mean')

        Returns:
            Master dark frame
        """
        if not dark_frames:
            raise ValueError("dark_frames list is empty")

        stack = np.array(dark_frames)

        if method == 'median':
            master = np.median(stack, axis=0)
        elif method == 'mean':
            master = np.mean(stack, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Created master dark from {len(dark_frames)} frames ({method})")
        return master

    def create_master_flat(
        self,
        flat_frames: list,
        method: str = 'median',
        normalize: bool = True
    ) -> np.ndarray:
        """Create master flat from multiple flat-field frames.

        Args:
            flat_frames: List of flat-field frame arrays
            method: Combination method ('median' or 'mean')
            normalize: Normalize to median value

        Returns:
            Master flat-field frame
        """
        if not flat_frames:
            raise ValueError("flat_frames list is empty")

        stack = np.array(flat_frames)

        if method == 'median':
            master = np.median(stack, axis=0)
        elif method == 'mean':
            master = np.mean(stack, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        if normalize:
            master = master / np.median(master[master > 0])

        logger.info(
            f"Created master flat from {len(flat_frames)} frames "
            f"({method}, normalized={normalize})"
        )
        return master
