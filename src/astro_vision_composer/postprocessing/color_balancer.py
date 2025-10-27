"""Color balancing for RGB composite images.

This module provides the ColorBalancer class for adjusting color balance
in astronomical RGB composites.

.. warning::
    Color balance operations are NOT photometrically accurate and destroy
    scientific data integrity. Use only for aesthetic/presentation purposes.
    Most methods use naive RGB operations without proper color space conversion.
"""

from typing import Tuple, Optional
import numpy as np
import logging
import warnings

from ..utilities.decorators import experimental, deprecated

logger = logging.getLogger(__name__)


class ColorBalancer:
    """Adjust color balance in RGB composite images.

    Provides methods for:
    - Channel weight adjustment
    - White balance correction
    - Saturation adjustment
    - Color temperature adjustment

    Example:
        >>> balancer = ColorBalancer()
        >>> # Adjust channel weights
        >>> balanced = balancer.balance_channels(
        ...     rgb_data,
        ...     r_weight=1.0,
        ...     g_weight=1.1,
        ...     b_weight=0.9
        ... )
    """

    def __init__(self):
        """Initialize the ColorBalancer."""
        pass

    def balance_channels(
        self,
        rgb: np.ndarray,
        r_weight: float = 1.0,
        g_weight: float = 1.0,
        b_weight: float = 1.0
    ) -> np.ndarray:
        """Adjust individual channel weights.

        Args:
            rgb: RGB image array with shape (height, width, 3)
            r_weight: Red channel multiplier
            g_weight: Green channel multiplier
            b_weight: Blue channel multiplier

        Returns:
            Color-balanced RGB image

        Example:
            >>> # Boost green channel by 10%
            >>> balanced = balancer.balance_channels(
            ...     rgb, r_weight=1.0, g_weight=1.1, b_weight=1.0
            ... )
        """
        if len(rgb.shape) != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

        logger.info(f"Balancing channels: R={r_weight}, G={g_weight}, B={b_weight}")

        balanced = rgb.copy()
        balanced[:, :, 0] *= r_weight
        balanced[:, :, 1] *= g_weight
        balanced[:, :, 2] *= b_weight

        # Clip to valid range
        balanced = np.clip(balanced, 0, 1)

        return balanced

    @experimental(
        quality="LOW",
        warning=(
            "Uses simple channel scaling without proper color space conversion. "
            "Not photometrically accurate. Destroys color calibration. "
            "Consider using external tools (Photoshop, GIMP) for proper color correction."
        )
    )
    def white_balance(
        self,
        rgb: np.ndarray,
        reference_point: Optional[Tuple[int, int]] = None,
        target_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """Apply white balance correction.

        .. danger::
            **NAIVE IMPLEMENTATION - NOT PRODUCTION READY**

            This white balance implementation has serious limitations:

            * Simple RGB channel scaling without color space conversion
            * No perceptual color space support (LAB, LCh)
            * Destroys photometric color information
            * No validation against astronomical color indices

            **For scientific use:** DO NOT USE - preserves no photometric data

            **For aesthetic use:** Consider external tools like Photoshop or GIMP

        Adjusts colors so that a reference point (or the image average)
        becomes the target color (usually white/gray).

        Args:
            rgb: RGB image array
            reference_point: (y, x) coordinates of reference (None = use image average)
            target_color: Target RGB color (default: white)

        Returns:
            White-balanced RGB image

        Example:
            >>> # Balance using image average (USE WITH CAUTION)
            >>> balanced = balancer.white_balance(rgb)
            >>>
            >>> # Balance using specific point
            >>> balanced = balancer.white_balance(rgb, reference_point=(500, 500))
        """
        if len(rgb.shape) != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

        # Get reference color
        if reference_point is not None:
            y, x = reference_point
            ref_color = rgb[y, x, :]
            logger.info(f"White balance using point ({y}, {x}): {ref_color}")
        else:
            # Use image average
            ref_color = np.array([
                np.mean(rgb[:, :, 0]),
                np.mean(rgb[:, :, 1]),
                np.mean(rgb[:, :, 2])
            ])
            logger.info(f"White balance using image average: {ref_color}")

        # Calculate adjustment factors
        target = np.array(target_color)
        weights = target / (ref_color + 1e-10)

        logger.info(f"White balance weights: R={weights[0]:.3f}, G={weights[1]:.3f}, B={weights[2]:.3f}")

        # Apply weights
        return self.balance_channels(rgb, r_weight=weights[0], g_weight=weights[1], b_weight=weights[2])

    @experimental(
        quality="MEDIUM",
        warning=(
            "Saturation adjustment clips values rather than properly preserving luminance. "
            "May cause color shifts and loss of detail in highly saturated regions."
        )
    )
    def adjust_saturation(
        self,
        rgb: np.ndarray,
        factor: float = 1.0
    ) -> np.ndarray:
        """Adjust color saturation.

        .. warning::
            **BASIC IMPLEMENTATION**

            This saturation adjustment has limitations:

            * Clips values instead of proper luminance preservation
            * Uses RGB color space instead of HSL/HSV
            * May lose detail in highly saturated areas

            For better results, convert to HSL color space, adjust S channel,
            and convert back to RGB.

        Args:
            rgb: RGB image array
            factor: Saturation multiplier (>1 = more saturated, <1 = less saturated, 0 = grayscale)

        Returns:
            Saturation-adjusted RGB image

        Example:
            >>> # Increase saturation by 20%
            >>> saturated = balancer.adjust_saturation(rgb, factor=1.2)
            >>>
            >>> # Desaturate to 50%
            >>> desaturated = balancer.adjust_saturation(rgb, factor=0.5)
        """
        if len(rgb.shape) != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

        logger.info(f"Adjusting saturation: factor={factor}")

        # Calculate luminance (grayscale)
        luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        luminance = luminance[:, :, np.newaxis]  # Add channel dimension

        # Blend between grayscale and color
        adjusted = luminance + factor * (rgb - luminance)

        return np.clip(adjusted, 0, 1)

    @experimental(
        quality="LOW",
        warning=(
            "Uses ad-hoc RGB channel shifts without proper color temperature model. "
            "Not based on blackbody radiation or standard illuminants. "
            "Results are unpredictable and non-physical."
        )
    )
    def adjust_color_temperature(
        self,
        rgb: np.ndarray,
        temperature: float = 0.0
    ) -> np.ndarray:
        """Adjust color temperature.

        .. danger::
            **AD-HOC IMPLEMENTATION**

            This temperature adjustment is NOT based on proper color science:

            * No blackbody radiation model
            * No CIE standard illuminants (D65, D50, etc.)
            * Simple RGB channel multiplication (non-physical)
            * Arbitrary scaling factors (0.3) with no scientific basis

            **Recommended:** Use proper color management tools or
            implement true color temperature based on Planck's law.

        Args:
            rgb: RGB image array
            temperature: Temperature adjustment (-1.0 = cooler/blue, +1.0 = warmer/red)

        Returns:
            Temperature-adjusted RGB image

        Example:
            >>> # Make image warmer (more red) - USE WITH CAUTION
            >>> warmer = balancer.adjust_color_temperature(rgb, temperature=0.2)
            >>>
            >>> # Make image cooler (more blue)
            >>> cooler = balancer.adjust_color_temperature(rgb, temperature=-0.2)
        """
        if len(rgb.shape) != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

        logger.info(f"Adjusting color temperature: {temperature}")

        adjusted = rgb.copy()

        if temperature > 0:
            # Warmer: increase red, decrease blue
            adjusted[:, :, 0] *= (1.0 + temperature * 0.3)  # Boost red
            adjusted[:, :, 2] *= (1.0 - temperature * 0.3)  # Reduce blue
        elif temperature < 0:
            # Cooler: decrease red, increase blue
            adjusted[:, :, 0] *= (1.0 + temperature * 0.3)  # Reduce red
            adjusted[:, :, 2] *= (1.0 - temperature * 0.3)  # Boost blue

        return np.clip(adjusted, 0, 1)

    def apply_curves(
        self,
        rgb: np.ndarray,
        curve_func,
        per_channel: bool = False
    ) -> np.ndarray:
        """Apply custom curve adjustment.

        Args:
            rgb: RGB image array
            curve_func: Function that maps input values [0, 1] to output values [0, 1]
            per_channel: Apply same curve to each channel separately (vs. luminance only)

        Returns:
            Curve-adjusted RGB image

        Example:
            >>> # Apply gamma correction
            >>> gamma_curve = lambda x: np.power(x, 1/2.2)
            >>> adjusted = balancer.apply_curves(rgb, gamma_curve)
        """
        if per_channel:
            # Apply to each channel separately
            adjusted = np.zeros_like(rgb)
            for i in range(3):
                adjusted[:, :, i] = curve_func(rgb[:, :, i])
        else:
            # Apply to luminance only
            luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            adjusted_lum = curve_func(luminance)

            # Scale RGB by luminance change
            scale = adjusted_lum / (luminance + 1e-10)
            scale = scale[:, :, np.newaxis]
            adjusted = rgb * scale

        logger.info(f"Applied custom curve (per_channel={per_channel})")
        return np.clip(adjusted, 0, 1)

    def auto_balance(
        self,
        rgb: np.ndarray,
        method: str = 'gray_world'
    ) -> np.ndarray:
        """Automatically balance colors.

        Args:
            rgb: RGB image array
            method: Auto-balance method ('gray_world' or 'white_patch')

        Returns:
            Auto-balanced RGB image

        Example:
            >>> # Gray world assumption
            >>> balanced = balancer.auto_balance(rgb, method='gray_world')
        """
        if method == 'gray_world':
            # Assume average color should be gray
            return self.white_balance(rgb, reference_point=None)

        elif method == 'white_patch':
            # Assume brightest pixels should be white
            luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            threshold = np.percentile(luminance, 99)
            bright_mask = luminance > threshold

            if np.any(bright_mask):
                ref_color = np.array([
                    np.mean(rgb[:, :, 0][bright_mask]),
                    np.mean(rgb[:, :, 1][bright_mask]),
                    np.mean(rgb[:, :, 2][bright_mask])
                ])
                weights = 1.0 / (ref_color + 1e-10)
                # Normalize weights
                weights = weights / np.max(weights)

                logger.info(f"Auto-balance (white_patch): weights={weights}")
                return self.balance_channels(rgb, r_weight=weights[0], g_weight=weights[1], b_weight=weights[2])
            else:
                logger.warning("No bright pixels found for white_patch method")
                return rgb

        else:
            raise ValueError(f"Unknown method: {method}")
