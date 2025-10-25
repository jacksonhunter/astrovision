"""World Coordinate System (WCS) handling and validation.

This module provides the WCSHandler class for extracting, validating, and
working with WCS information from FITS headers and WCS objects.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging

logger = logging.getLogger(__name__)


@dataclass
class WCSInfo:
    """Container for WCS information and validation results.

    Attributes:
        wcs: The WCS object
        has_celestial: Whether WCS contains celestial coordinates
        has_spectral: Whether WCS contains spectral coordinates
        pixel_scale: Pixel scale in arcseconds per pixel (mean of both axes)
        pixel_scale_x: Pixel scale in X direction (arcsec/pixel)
        pixel_scale_y: Pixel scale in Y direction (arcsec/pixel)
        rotation_angle: Rotation angle in degrees
        projection: Projection type (e.g., 'TAN', 'SIN')
        reference_pixel: Reference pixel coordinates (CRPIX)
        reference_sky: Reference sky coordinates (CRVAL) as SkyCoord
        warnings: List of validation warnings
        is_valid: Whether the WCS is valid and usable
    """
    wcs: WCS
    has_celestial: bool
    has_spectral: bool
    pixel_scale: Optional[float] = None
    pixel_scale_x: Optional[float] = None
    pixel_scale_y: Optional[float] = None
    rotation_angle: Optional[float] = None
    projection: Optional[str] = None
    reference_pixel: Optional[Tuple[float, float]] = None
    reference_sky: Optional[SkyCoord] = None
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True

    def __repr__(self):
        """Pretty representation."""
        parts = []
        if self.projection:
            parts.append(f"Projection={self.projection}")
        if self.pixel_scale:
            parts.append(f"Scale={self.pixel_scale:.3f}\"/px")
        if self.rotation_angle is not None:
            parts.append(f"Rotation={self.rotation_angle:.1f}deg")
        status = "Valid" if self.is_valid else "Invalid"
        return f"WCSInfo({status}, {', '.join(parts)})"


class WCSHandler:
    """Handler for WCS extraction, validation, and coordinate conversion.

    This class provides utilities for working with World Coordinate Systems,
    including validation, coordinate conversion, and extracting useful properties.

    Note: This class works with astropy.wcs.WCS objects and does not replace
    astropy functionality. It provides analysis and validation on top of astropy.

    Example:
        >>> from astropy.io import fits
        >>> from astropy.wcs import WCS
        >>> handler = WCSHandler()
        >>> with fits.open('image.fits') as hdul:
        ...     wcs = WCS(hdul[0].header)
        ...     wcs_info = handler.validate(wcs)
        ...     if wcs_info.is_valid:
        ...         print(f"Pixel scale: {wcs_info.pixel_scale:.3f} arcsec/pixel")
    """

    def __init__(self):
        """Initialize the WCS handler."""
        pass

    def validate(self, wcs: WCS) -> WCSInfo:
        """Validate a WCS object and extract key information.

        Performs comprehensive validation and extracts useful properties
        such as pixel scale, rotation, and reference coordinates.

        Args:
            wcs: WCS object to validate

        Returns:
            WCSInfo dataclass containing validation results and extracted properties
        """
        info = WCSInfo(
            wcs=wcs,
            has_celestial=wcs.has_celestial,
            has_spectral=wcs.has_spectral
        )

        # Check for celestial coordinates
        if not wcs.has_celestial:
            info.warnings.append("WCS has no celestial coordinates")
            info.is_valid = False
            return info

        # Extract projection type
        try:
            # Get CTYPE for first axis
            if hasattr(wcs.wcs, 'ctype') and len(wcs.wcs.ctype) > 0:
                ctype = wcs.wcs.ctype[0]
                # Projection is typically last 3 characters (e.g., 'RA---TAN' -> 'TAN')
                if '-' in ctype:
                    info.projection = ctype.split('-')[-1]
        except Exception as e:
            logger.debug(f"Could not extract projection type: {e}")
            info.warnings.append("Could not determine projection type")

        # Extract reference pixel (CRPIX)
        try:
            if hasattr(wcs.wcs, 'crpix') and len(wcs.wcs.crpix) >= 2:
                crpix = wcs.wcs.crpix
                info.reference_pixel = (float(crpix[0]), float(crpix[1]))
        except Exception as e:
            logger.debug(f"Could not extract reference pixel: {e}")
            info.warnings.append("Could not extract reference pixel (CRPIX)")

        # Extract reference sky coordinates (CRVAL)
        try:
            if hasattr(wcs.wcs, 'crval') and len(wcs.wcs.crval) >= 2:
                crval = wcs.wcs.crval
                # Assume first two axes are RA/Dec
                info.reference_sky = SkyCoord(
                    ra=crval[0] * u.deg,
                    dec=crval[1] * u.deg,
                    frame='icrs'
                )
        except Exception as e:
            logger.debug(f"Could not extract reference sky coordinates: {e}")
            info.warnings.append("Could not extract reference sky coordinates (CRVAL)")

        # Calculate pixel scale
        try:
            pixel_scales = self._calculate_pixel_scale(wcs)
            if pixel_scales:
                info.pixel_scale_x, info.pixel_scale_y = pixel_scales
                info.pixel_scale = np.mean([info.pixel_scale_x, info.pixel_scale_y])

                # Check for large distortions (>10% difference between axes)
                if abs(info.pixel_scale_x - info.pixel_scale_y) / info.pixel_scale > 0.1:
                    info.warnings.append(
                        f"Significant pixel scale difference between axes: "
                        f"{info.pixel_scale_x:.3f} vs {info.pixel_scale_y:.3f} arcsec/pixel"
                    )
        except Exception as e:
            logger.debug(f"Could not calculate pixel scale: {e}")
            info.warnings.append("Could not calculate pixel scale")

        # Calculate rotation angle
        try:
            info.rotation_angle = self._calculate_rotation(wcs)
        except Exception as e:
            logger.debug(f"Could not calculate rotation angle: {e}")

        # Final validation
        if info.pixel_scale is None:
            info.warnings.append("Missing pixel scale - WCS may be incomplete")
            info.is_valid = False

        if info.reference_pixel is None or info.reference_sky is None:
            info.warnings.append("Missing reference coordinates - WCS may be incomplete")
            info.is_valid = False

        return info

    def _calculate_pixel_scale(self, wcs: WCS) -> Optional[Tuple[float, float]]:
        """Calculate pixel scale in arcseconds per pixel for both axes.

        Args:
            wcs: WCS object

        Returns:
            Tuple of (pixel_scale_x, pixel_scale_y) in arcsec/pixel, or None
        """
        try:
            # Try to use proj_plane_pixel_scales (most robust method)
            if hasattr(wcs, 'proj_plane_pixel_scales'):
                scales = wcs.proj_plane_pixel_scales()
                # Convert to arcseconds
                scale_x = abs(scales[0]) * 3600.0
                scale_y = abs(scales[1]) * 3600.0
                return (scale_x, scale_y)
        except:
            pass

        try:
            # Alternative: use CDELT keywords if available
            if hasattr(wcs.wcs, 'cdelt') and len(wcs.wcs.cdelt) >= 2:
                cdelt = wcs.wcs.cdelt
                scale_x = abs(cdelt[0]) * 3600.0
                scale_y = abs(cdelt[1]) * 3600.0
                return (scale_x, scale_y)
        except:
            pass

        return None

    def _calculate_rotation(self, wcs: WCS) -> Optional[float]:
        """Calculate rotation angle of the WCS in degrees.

        Args:
            wcs: WCS object

        Returns:
            Rotation angle in degrees (East of North), or None
        """
        try:
            # Get the CD matrix or PC matrix
            if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd.shape == (2, 2):
                cd = wcs.wcs.cd
                angle = np.rad2deg(np.arctan2(-cd[0, 1], cd[1, 1]))
                return angle

            # Try PC matrix with CDELT
            if hasattr(wcs.wcs, 'pc') and wcs.wcs.pc.shape == (2, 2):
                pc = wcs.wcs.pc
                angle = np.rad2deg(np.arctan2(-pc[0, 1], pc[1, 1]))
                return angle
        except Exception as e:
            logger.debug(f"Rotation calculation failed: {e}")

        return None

    def compare_wcs(self, wcs1: WCS, wcs2: WCS) -> dict:
        """Compare two WCS objects and report differences.

        Useful for determining if images can be aligned or need reprojection.

        Args:
            wcs1: First WCS object
            wcs2: Second WCS object

        Returns:
            Dictionary with comparison results including:
            - 'compatible': Whether WCS are compatible (same projection, similar scale)
            - 'pixel_scale_diff': Difference in pixel scales (%)
            - 'rotation_diff': Difference in rotation angles (degrees)
            - 'projection_match': Whether projections match
            - 'details': List of detailed comparison information
        """
        info1 = self.validate(wcs1)
        info2 = self.validate(wcs2)

        result = {
            'compatible': True,
            'pixel_scale_diff': None,
            'rotation_diff': None,
            'projection_match': None,
            'details': []
        }

        # Check if both are valid
        if not info1.is_valid or not info2.is_valid:
            result['compatible'] = False
            result['details'].append("One or both WCS are invalid")
            return result

        # Compare projections
        if info1.projection != info2.projection:
            result['projection_match'] = False
            result['compatible'] = False
            result['details'].append(
                f"Different projections: {info1.projection} vs {info2.projection}"
            )
        else:
            result['projection_match'] = True

        # Compare pixel scales
        if info1.pixel_scale and info2.pixel_scale:
            diff_pct = abs(info1.pixel_scale - info2.pixel_scale) / info1.pixel_scale * 100
            result['pixel_scale_diff'] = diff_pct

            if diff_pct > 5:  # More than 5% difference
                result['compatible'] = False
                result['details'].append(
                    f"Pixel scale difference: {diff_pct:.1f}% "
                    f"({info1.pixel_scale:.3f} vs {info2.pixel_scale:.3f} arcsec/pixel)"
                )

        # Compare rotations
        if info1.rotation_angle is not None and info2.rotation_angle is not None:
            diff_deg = abs(info1.rotation_angle - info2.rotation_angle)
            # Normalize to [0, 360]
            diff_deg = min(diff_deg, 360 - diff_deg)
            result['rotation_diff'] = diff_deg

            if diff_deg > 1:  # More than 1 degree difference
                result['details'].append(
                    f"Rotation difference: {diff_deg:.2f} degrees"
                )

        return result
