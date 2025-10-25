"""RGB composite image generation.

This module provides the Compositor class for creating RGB composite images
from multi-wavelength astronomical data.
"""

from typing import Optional, Literal
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import astropy visualization for Lupton RGB
try:
    from astropy.visualization import make_lupton_rgb
    ASTROPY_VIZ_AVAILABLE = True
except ImportError:
    ASTROPY_VIZ_AVAILABLE = False
    logger.warning(
        "astropy.visualization not available. "
        "Install/update astropy: pip install astropy"
    )


CompositeMethod = Literal['lupton', 'simple']


class Compositor:
    """Create RGB composite images from multi-wavelength data.

    This class provides methods for combining three wavelength bands into
    an RGB color image using different algorithms optimized for astronomical
    data with wide dynamic range.

    Methods:
    - 'lupton': Lupton et al. algorithm (preserves color in bright regions)
    - 'simple': Simple independent channel scaling

    Example:
        >>> from astro_vision_composer.postprocessing import Compositor
        >>> compositor = Compositor()
        >>>
        >>> # Lupton RGB (recommended for most cases)
        >>> rgb = compositor.create_lupton_rgb(
        ...     r=i_band_data,
        ...     g=r_band_data,
        ...     b=g_band_data,
        ...     stretch=0.5,
        ...     Q=8
        ... )
    """

    def __init__(self):
        """Initialize the Compositor.

        Raises:
            ImportError: If astropy.visualization is not available
        """
        if not ASTROPY_VIZ_AVAILABLE:
            raise ImportError(
                "astropy.visualization is required. "
                "Install/update astropy: pip install astropy"
            )

    def create_lupton_rgb(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
        minimum: float = 0.0,
        stretch: float = 0.5,
        Q: float = 8.0,
        filename: Optional[str] = None
    ) -> np.ndarray:
        """Create RGB composite using Lupton et al. algorithm.

        The Lupton algorithm (SDSS) is designed for astronomical images with
        wide dynamic range. It preserves color information in bright regions
        while bringing out faint features.

        Args:
            r: Red channel data (longest wavelength)
            g: Green channel data (middle wavelength)
            b: Blue channel data (shortest wavelength)
            minimum: Black point (data values below this become black)
            stretch: Linear stretch parameter (higher = more aggressive)
            Q: Asinh softening parameter (higher = more linear)
            filename: Optional path to save directly

        Returns:
            RGB image array with shape (height, width, 3) and values in [0, 1]

        Notes:
            - Input data should be pre-normalized/stretched for best results
            - Default stretch=0.5, Q=8 work well for most cases
            - Increase Q for brighter images
            - Decrease stretch for more aggressive scaling

        Example:
            >>> # After normalization and stretching
            >>> rgb = compositor.create_lupton_rgb(
            ...     r=i_stretched, g=r_stretched, b=g_stretched,
            ...     stretch=0.5, Q=10
            ... )
        """
        if r.shape != g.shape or r.shape != b.shape:
            raise ValueError(
                f"All channels must have same shape. "
                f"Got r={r.shape}, g={g.shape}, b={b.shape}"
            )

        logger.info(
            f"Creating Lupton RGB: stretch={stretch}, Q={Q}, minimum={minimum}"
        )

        rgb = make_lupton_rgb(
            r, g, b,
            minimum=minimum,
            stretch=stretch,
            Q=Q,
            filename=filename
        )

        logger.debug(f"RGB composite created: shape={rgb.shape}, dtype={rgb.dtype}")

        return rgb

    def create_simple_rgb(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
        r_scale: float = 1.0,
        g_scale: float = 1.0,
        b_scale: float = 1.0
    ) -> np.ndarray:
        """Create RGB composite using simple independent channel scaling.

        Each channel is scaled independently and then stacked. This is simpler
        than Lupton but may not preserve color as well in bright regions.

        Args:
            r: Red channel data
            g: Green channel data
            b: Blue channel data
            r_scale: Multiplicative scale for red channel (default: 1.0)
            g_scale: Multiplicative scale for green channel (default: 1.0)
            b_scale: Multiplicative scale for blue channel (default: 1.0)

        Returns:
            RGB image array with shape (height, width, 3) and values in [0, 1]

        Example:
            >>> # Boost red channel by 20%
            >>> rgb = compositor.create_simple_rgb(
            ...     r=i_data, g=r_data, b=g_data,
            ...     r_scale=1.2
            ... )
        """
        if r.shape != g.shape or r.shape != b.shape:
            raise ValueError(
                f"All channels must have same shape. "
                f"Got r={r.shape}, g={g.shape}, b={b.shape}"
            )

        logger.info(
            f"Creating simple RGB: r_scale={r_scale}, "
            f"g_scale={g_scale}, b_scale={b_scale}"
        )

        # Scale channels
        r_scaled = np.clip(r * r_scale, 0, 1)
        g_scaled = np.clip(g * g_scale, 0, 1)
        b_scaled = np.clip(b * b_scale, 0, 1)

        # Stack into RGB
        rgb = np.dstack([r_scaled, g_scaled, b_scaled])

        logger.debug(f"RGB composite created: shape={rgb.shape}, dtype={rgb.dtype}")

        return rgb

    def create_narrowband_composite(
        self,
        ha: np.ndarray,
        oiii: np.ndarray,
        sii: np.ndarray,
        method: CompositeMethod = 'lupton',
        **kwargs
    ) -> np.ndarray:
        """Create narrowband composite (e.g., Hubble Palette).

        Common narrowband composite mapping:
        - H-alpha (656nm) → Red
        - OIII (501nm) → Green
        - SII (672nm) → Blue

        This is the "Hubble Palette" often used for emission nebulae.

        Args:
            ha: H-alpha data
            oiii: OIII data
            sii: SII data
            method: Composite method ('lupton' or 'simple')
            **kwargs: Additional arguments passed to composite method

        Returns:
            RGB composite array

        Example:
            >>> # Hubble Palette
            >>> rgb = compositor.create_narrowband_composite(
            ...     ha=ha_data, oiii=oiii_data, sii=sii_data,
            ...     method='lupton', stretch=0.5, Q=10
            ... )
        """
        logger.info("Creating narrowband composite (Hubble Palette)")

        if method == 'lupton':
            return self.create_lupton_rgb(
                r=ha, g=oiii, b=sii,
                **kwargs
            )
        elif method == 'simple':
            return self.create_simple_rgb(
                r=ha, g=oiii, b=sii,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def create_from_mapping(
        self,
        data: dict,
        mapping,
        method: CompositeMethod = 'lupton',
        **kwargs
    ) -> np.ndarray:
        """Create RGB composite using a ChannelMapping.

        Convenience method that extracts the correct channels based on
        a ChannelMapping object.

        Args:
            data: Dictionary mapping band names to data arrays
            mapping: ChannelMapping object (from ChannelMapper)
            method: Composite method ('lupton' or 'simple')
            **kwargs: Additional arguments for composite method

        Returns:
            RGB composite array

        Example:
            >>> from astro_vision_composer.postprocessing import ChannelMapper
            >>> mapper = ChannelMapper()
            >>> mapping = mapper.auto_map_by_wavelength({
            ...     'g': 481, 'r': 617, 'i': 752
            ... })
            >>> rgb = compositor.create_from_mapping(
            ...     data={'g': g_data, 'r': r_data, 'i': i_data},
            ...     mapping=mapping,
            ...     method='lupton'
            ... )
        """
        # Extract channels based on mapping
        r_data = data[mapping.red]
        g_data = data[mapping.green]
        b_data = data[mapping.blue]

        logger.info(
            f"Creating RGB from mapping: R={mapping.red}, "
            f"G={mapping.green}, B={mapping.blue}"
        )

        if method == 'lupton':
            return self.create_lupton_rgb(r_data, g_data, b_data, **kwargs)
        elif method == 'simple':
            return self.create_simple_rgb(r_data, g_data, b_data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
