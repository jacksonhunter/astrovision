"""Image reprojection and alignment.

This module provides the Reprojector class for aligning astronomical images
using WCS transformations via the reproject library.
"""

from typing import Optional, Tuple, Dict, Literal
import numpy as np
from astropy.wcs import WCS
import logging

logger = logging.getLogger(__name__)

# Import reproject functions
from reproject import reproject_interp, reproject_exact


ReprojectionMethod = Literal['interp', 'exact']


class Reprojector:
    """Reproject astronomical images to align them on a common pixel grid.

    This is a wrapper around the reproject library that provides a simplified
    interface for the most common reprojection tasks in astronomical image
    composition.

    The two main reprojection algorithms are:
    - 'interp': Fast interpolation-based (good for visualization)
    - 'exact': Flux-conserving (required for photometry)

    Example:
        >>> from astropy.io import fits
        >>> from astropy.wcs import WCS
        >>>
        >>> # Load target (reference) image
        >>> with fits.open('target.fits') as hdul:
        ...     target_wcs = WCS(hdul[0].header)
        ...     target_shape = hdul[0].data.shape
        >>>
        >>> # Load source image to reproject
        >>> with fits.open('source.fits') as hdul:
        ...     source_data = hdul[0].data
        ...     source_wcs = WCS(hdul[0].header)
        >>>
        >>> # Reproject source to target frame
        >>> reprojector = Reprojector(method='interp')
        >>> aligned_data, footprint = reprojector.reproject_to_target(
        ...     source_data, source_wcs, target_wcs, target_shape
        ... )
    """

    def __init__(
        self,
        method: ReprojectionMethod = 'interp',
        order: int = 1
    ):
        """Initialize the Reprojector.

        Args:
            method: Reprojection method ('interp' or 'exact')
            order: Interpolation order for 'interp' method (0=nearest, 1=bilinear, 2=bicubic)

        Raises:
            ImportError: If reproject library is not installed
        """
        if method not in ['interp', 'exact']:
            raise ValueError(f"method must be 'interp' or 'exact', got '{method}'")

        self.method = method
        self.order = order

    def reproject_to_target(
        self,
        source_data: np.ndarray,
        source_wcs: WCS,
        target_wcs: WCS,
        target_shape: Tuple[int, int],
        method: Optional[ReprojectionMethod] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reproject a source image to match a target WCS and shape.

        Args:
            source_data: Source image data array
            source_wcs: Source image WCS
            target_wcs: Target WCS to reproject onto
            target_shape: Target image shape (height, width)
            method: Override the default reprojection method

        Returns:
            Tuple of:
            - reprojected_data: Reprojected image array
            - footprint: Footprint array (1 where valid data, 0 where no coverage)

        Raises:
            ValueError: If source_data or WCS are invalid
        """
        if source_data is None or source_data.size == 0:
            raise ValueError("source_data is empty or None")

        if not source_wcs.has_celestial:
            raise ValueError("source_wcs has no celestial coordinates")

        if not target_wcs.has_celestial:
            raise ValueError("target_wcs has no celestial coordinates")

        # Use specified method or fall back to instance default
        reproject_method = method or self.method

        logger.info(
            f"Reprojecting image using '{reproject_method}' method "
            f"from shape {source_data.shape} to {target_shape}"
        )

        try:
            if reproject_method == 'interp':
                result, footprint = reproject_interp(
                    (source_data, source_wcs),
                    target_wcs,
                    shape_out=target_shape,
                    order=self.order
                )
            elif reproject_method == 'exact':
                result, footprint = reproject_exact(
                    (source_data, source_wcs),
                    target_wcs,
                    shape_out=target_shape
                )
            else:
                raise ValueError(f"Unknown reprojection method: {reproject_method}")

            logger.debug(f"Reprojection complete. Output shape: {result.shape}")
            return result, footprint

        except Exception as e:
            logger.error(f"Reprojection failed: {e}")
            raise

    def align_image_set(
        self,
        images: Dict[str, Tuple[np.ndarray, WCS]],
        reference: Optional[str] = None,
        method: Optional[ReprojectionMethod] = None
    ) -> Dict[str, np.ndarray]:
        """Align a set of images to a common WCS frame.

        Args:
            images: Dictionary mapping names to (data, wcs) tuples
            reference: Name of reference image (if None, uses first image)
            method: Reprojection method to use

        Returns:
            Dictionary mapping names to reprojected data arrays

        Example:
            >>> images = {
            ...     'g': (g_data, g_wcs),
            ...     'r': (r_data, r_wcs),
            ...     'i': (i_data, i_wcs)
            ... }
            >>> reprojector = Reprojector()
            >>> aligned = reprojector.align_image_set(images, reference='i')
        """
        if not images:
            raise ValueError("images dictionary is empty")

        # Determine reference image
        if reference is None:
            reference = list(images.keys())[0]
            logger.info(f"No reference specified, using '{reference}'")
        elif reference not in images:
            raise ValueError(f"Reference '{reference}' not found in images")

        # Get reference WCS and shape
        ref_data, ref_wcs = images[reference]
        target_shape = ref_data.shape
        target_wcs = ref_wcs

        logger.info(
            f"Aligning {len(images)} images to reference '{reference}' "
            f"(shape={target_shape})"
        )

        # Reproject all images
        aligned = {}
        for name, (data, wcs) in images.items():
            if name == reference:
                # Reference image doesn't need reprojection
                aligned[name] = data
                logger.debug(f"'{name}': reference image, no reprojection")
            else:
                logger.debug(f"'{name}': reprojecting...")
                reprojected, footprint = self.reproject_to_target(
                    data, wcs, target_wcs, target_shape, method=method
                )
                aligned[name] = reprojected

        logger.info("Image set alignment complete")
        return aligned

    def reproject_from_fits_data(
        self,
        source_fits_data,
        target_fits_data,
        method: Optional[ReprojectionMethod] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reproject using FITSData objects from preprocessing.

        Convenience method that extracts WCS and data from FITSData objects.

        Args:
            source_fits_data: Source FITSData object (from FITSLoader)
            target_fits_data: Target FITSData object
            method: Reprojection method

        Returns:
            Tuple of (reprojected_data, footprint)

        Example:
            >>> from astro_vision_composer.preprocessing import FITSLoader
            >>> loader = FITSLoader()
            >>> source = loader.load('source.fits')
            >>> target = loader.load('target.fits')
            >>> reprojector = Reprojector()
            >>> aligned, footprint = reprojector.reproject_from_fits_data(source, target)
        """
        if source_fits_data.wcs is None:
            raise ValueError("Source FITSData has no WCS")
        if target_fits_data.wcs is None:
            raise ValueError("Target FITSData has no WCS")

        return self.reproject_to_target(
            source_fits_data.science,
            source_fits_data.wcs,
            target_fits_data.wcs,
            target_fits_data.shape,
            method=method
        )
