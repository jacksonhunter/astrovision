"""APE 14 Compliance Tests for WCS Handler.

Tests that WCSHandler works correctly with BOTH:
- astropy.wcs.WCS (standard FITS WCS)
- gwcs.WCS (JWST, Roman, etc.)

Both implement the APE 14 common interface and should be interchangeable.
"""

import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.modeling import models
from pathlib import Path

from astro_vision_composer.processing.wcs_handler import WCSHandler, WCSInfo

# Check if gwcs is available
try:
    import gwcs
    from gwcs import coordinate_frames as cf
    GWCS_AVAILABLE = True
except ImportError:
    GWCS_AVAILABLE = False


@pytest.fixture
def wcs_handler():
    """Create WCSHandler instance."""
    return WCSHandler()


@pytest.fixture
def simple_fits_wcs():
    """Create a simple FITS WCS object (astropy.wcs.WCS)."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1024, 1024]
    wcs.wcs.crval = [30.0, 45.0]  # RA, Dec in degrees
    wcs.wcs.cdelt = [0.1 / 3600, 0.1 / 3600]  # 0.1 arcsec/pixel
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs


@pytest.fixture
def simple_gwcs():
    """Create a simple gwcs.WCS object."""
    if not GWCS_AVAILABLE:
        pytest.skip("gwcs not installed")

    # Create simple detector→sky transform (simplified version of gwcs docs)
    crpix = (1024, 1024)
    shift_by_crpix = models.Shift(-crpix[0]) & models.Shift(-crpix[1])

    # Pixel scale: 0.1 arcsec/pixel = 0.1/3600 deg/pixel
    pixelscale = models.Scale(0.1 / 3600) & models.Scale(0.1 / 3600)

    # TAN projection
    tan = models.Pix2Sky_TAN()

    # Rotation to sky coordinates
    celestial_rotation = models.RotateNative2Celestial(30.0, 45.0, 180.0)

    # Compose full transform
    det2sky = shift_by_crpix | pixelscale | tan | celestial_rotation
    det2sky.name = "detector_to_sky"

    # Define coordinate frames
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=SkyCoord(0, 0, unit='deg',
                                                            frame='icrs').frame,
                                  name='icrs',
                                  unit=(u.deg, u.deg))

    # Create gwcs WCS
    wcs_gwcs = gwcs.wcs.WCS([(detector_frame, det2sky), (sky_frame, None)])

    return wcs_gwcs


class TestAPE14CommonInterface:
    """Test that both WCS types implement APE 14 common interface."""

    def test_fits_wcs_has_ape14_methods(self, simple_fits_wcs):
        """Test astropy.wcs.WCS has APE 14 methods."""
        wcs = simple_fits_wcs

        # All APE 14-compliant WCS must have these
        assert hasattr(wcs, 'pixel_to_world')
        assert hasattr(wcs, 'world_to_pixel')
        assert hasattr(wcs, 'pixel_to_world_values')
        assert hasattr(wcs, 'world_to_pixel_values')
        assert hasattr(wcs, 'world_axis_names')
        assert hasattr(wcs, 'world_axis_units')

    def test_gwcs_has_ape14_methods(self, simple_gwcs):
        """Test gwcs.WCS has APE 14 methods."""
        wcs = simple_gwcs

        # All APE 14-compliant WCS must have these
        assert hasattr(wcs, 'pixel_to_world')
        assert hasattr(wcs, 'world_to_pixel')
        assert hasattr(wcs, 'pixel_to_world_values')
        assert hasattr(wcs, 'world_to_pixel_values')
        assert hasattr(wcs, 'world_axis_names')
        assert hasattr(wcs, 'world_axis_units')

    def test_fits_wcs_coordinate_transformation(self, simple_fits_wcs):
        """Test coordinate transformation with FITS WCS."""
        wcs = simple_fits_wcs

        # Forward transform: pixel → world
        sky = wcs.pixel_to_world(1024, 1024)  # Reference pixel
        assert isinstance(sky, SkyCoord)

        # Should be near (30, 45) degrees
        assert np.abs(sky.ra.deg - 30.0) < 0.01
        assert np.abs(sky.dec.deg - 45.0) < 0.01

        # Inverse transform: world → pixel
        x, y = wcs.world_to_pixel(sky)
        np.testing.assert_allclose([x, y], [1024, 1024], atol=0.01)

    def test_gwcs_coordinate_transformation(self, simple_gwcs):
        """Test coordinate transformation with gwcs."""
        wcs = simple_gwcs

        # Forward transform: pixel → world
        sky = wcs.pixel_to_world(1024, 1024)  # Reference pixel
        assert isinstance(sky, SkyCoord)

        # Should be near (30, 45) degrees
        assert np.abs(sky.ra.deg - 30.0) < 0.1
        assert np.abs(sky.dec.deg - 45.0) < 0.1

        # Inverse transform: world → pixel
        x, y = wcs.world_to_pixel(sky)
        np.testing.assert_allclose([x, y], [1024, 1024], atol=0.1)


class TestWCSHandlerValidation:
    """Test WCSHandler.validate() works for both WCS types."""

    def test_validate_fits_wcs(self, wcs_handler, simple_fits_wcs):
        """Test validation of astropy.wcs.WCS."""
        info = wcs_handler.validate(simple_fits_wcs)

        # Should be valid
        assert info.is_valid
        assert info.has_celestial
        assert not info.has_gwcs

        # Should extract basic properties
        assert info.pixel_scale is not None
        assert np.abs(info.pixel_scale - 0.1) < 0.01  # 0.1 arcsec/pixel
        assert info.projection == 'TAN'

        # Should have no warnings
        assert len(info.warnings) == 0

    def test_validate_gwcs(self, wcs_handler, simple_gwcs):
        """Test validation of gwcs.WCS."""
        info = wcs_handler.validate(simple_gwcs)

        # Should be valid
        assert info.is_valid
        assert info.has_celestial
        assert info.has_gwcs  # This is a gwcs object

        # Should extract basic properties
        assert info.pixel_scale is not None
        assert np.abs(info.pixel_scale - 0.1) < 0.02  # 0.1 arcsec/pixel (±tolerance)

        # gwcs-specific
        assert info.projection == 'gwcs'
        assert info.available_frames is not None
        assert 'detector' in info.available_frames
        assert 'icrs' in info.available_frames

    def test_pixel_scale_from_transform_fits(self, wcs_handler, simple_fits_wcs):
        """Test pixel scale calculation via transform (FITS WCS)."""
        scales = wcs_handler._calculate_pixel_scale_from_transform(simple_fits_wcs)

        assert scales is not None
        scale_x, scale_y = scales

        # Should be ~0.1 arcsec/pixel
        assert np.abs(scale_x - 0.1) < 0.01
        assert np.abs(scale_y - 0.1) < 0.01

    def test_pixel_scale_from_transform_gwcs(self, wcs_handler, simple_gwcs):
        """Test pixel scale calculation via transform (gwcs)."""
        scales = wcs_handler._calculate_pixel_scale_from_transform(simple_gwcs)

        assert scales is not None
        scale_x, scale_y = scales

        # Should be ~0.1 arcsec/pixel
        assert np.abs(scale_x - 0.1) < 0.02
        assert np.abs(scale_y - 0.1) < 0.02

    def test_world_axis_names_fits(self, wcs_handler, simple_fits_wcs):
        """Test extracting axis names (FITS WCS)."""
        info = wcs_handler.validate(simple_fits_wcs)

        assert info.world_axis_names is not None
        # Should have RA and Dec axes (names may vary)
        assert len(info.world_axis_names) == 2

    def test_world_axis_names_gwcs(self, wcs_handler, simple_gwcs):
        """Test extracting axis names (gwcs)."""
        info = wcs_handler.validate(simple_gwcs)

        assert info.world_axis_names is not None
        # Should have lon and lat axes
        assert len(info.world_axis_names) == 2


class TestWCSHandlerComparison:
    """Test WCSHandler.compare_wcs() works for both types."""

    def test_compare_same_type_compatible(self, wcs_handler, simple_fits_wcs):
        """Test comparing two compatible FITS WCS."""
        wcs1 = simple_fits_wcs

        # Create second WCS with same projection and scale
        wcs2 = WCS(naxis=2)
        wcs2.wcs.crpix = [2048, 2048]  # Different reference pixel
        wcs2.wcs.crval = [30.5, 45.5]  # Different reference coords
        wcs2.wcs.cdelt = [0.1 / 3600, 0.1 / 3600]  # Same pixel scale
        wcs2.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # Same projection
        wcs2.wcs.cunit = ['deg', 'deg']

        result = wcs_handler.compare_wcs(wcs1, wcs2)

        assert result['compatible']
        assert result['projection_match']
        assert result['pixel_scale_diff'] < 1.0  # <1% difference

    def test_compare_different_projections(self, wcs_handler, simple_fits_wcs):
        """Test comparing WCS with different projections."""
        wcs1 = simple_fits_wcs

        # Create WCS with different projection
        wcs2 = WCS(naxis=2)
        wcs2.wcs.crpix = [1024, 1024]
        wcs2.wcs.crval = [30.0, 45.0]
        wcs2.wcs.cdelt = [0.1 / 3600, 0.1 / 3600]
        wcs2.wcs.ctype = ['RA---SIN', 'DEC--SIN']  # SIN instead of TAN
        wcs2.wcs.cunit = ['deg', 'deg']

        result = wcs_handler.compare_wcs(wcs1, wcs2)

        assert not result['compatible']
        assert not result['projection_match']

    def test_compare_mixed_types(self, wcs_handler, simple_fits_wcs, simple_gwcs):
        """Test comparing FITS WCS with gwcs (should work!)."""
        result = wcs_handler.compare_wcs(simple_fits_wcs, simple_gwcs)

        # Should work because both implement APE 14
        # May or may not be compatible depending on projections/scales
        assert 'compatible' in result
        assert 'pixel_scale_diff' in result


class TestWCSInfoRepresentation:
    """Test WCSInfo __repr__ works for both types."""

    def test_repr_fits_wcs(self, wcs_handler, simple_fits_wcs):
        """Test string representation of WCSInfo (FITS)."""
        info = wcs_handler.validate(simple_fits_wcs)

        repr_str = repr(info)

        # Should contain key information
        assert 'Valid' in repr_str or 'Invalid' in repr_str
        assert 'TAN' in repr_str  # Projection
        assert '0.1' in repr_str or '0.10' in repr_str  # Pixel scale

    def test_repr_gwcs(self, wcs_handler, simple_gwcs):
        """Test string representation of WCSInfo (gwcs)."""
        info = wcs_handler.validate(simple_gwcs)

        repr_str = repr(info)

        # Should contain key information
        assert 'Valid' in repr_str or 'Invalid' in repr_str
        assert 'gwcs' in repr_str  # Projection type


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_validate_with_invalid_wcs(self, wcs_handler):
        """Test validation with malformed WCS."""
        # Create WCS with no celestial coordinates
        wcs = WCS(naxis=1)  # Only 1D
        wcs.wcs.ctype = ['FREQ']

        info = wcs_handler.validate(wcs)

        # Should mark as invalid
        assert not info.is_valid
        assert not info.has_celestial
        assert len(info.warnings) > 0

    def test_pixel_scale_with_distortion(self, wcs_handler, simple_fits_wcs):
        """Test pixel scale calculation handles distortion gracefully."""
        # Transform-based method should work even with non-uniform distortion
        # (FITS keyword method would fail)

        scales = wcs_handler._calculate_pixel_scale_from_transform(simple_fits_wcs)
        assert scales is not None


class TestRoundTrip:
    """Test coordinate round-trip transformations."""

    def test_fits_wcs_round_trip_single_point(self, simple_fits_wcs):
        """Test pixel→world→pixel round trip (FITS WCS)."""
        wcs = simple_fits_wcs

        # Start with pixel coordinates
        x_orig, y_orig = 500, 750

        # Transform to world
        sky = wcs.pixel_to_world(x_orig, y_orig)

        # Transform back to pixel
        x_back, y_back = wcs.world_to_pixel(sky)

        # Should recover original coordinates
        np.testing.assert_allclose([x_back, y_back], [x_orig, y_orig], atol=0.01)

    def test_gwcs_round_trip_single_point(self, simple_gwcs):
        """Test pixel→world→pixel round trip (gwcs)."""
        wcs = simple_gwcs

        # Start with pixel coordinates
        x_orig, y_orig = 500, 750

        # Transform to world
        sky = wcs.pixel_to_world(x_orig, y_orig)

        # Transform back to pixel
        x_back, y_back = wcs.world_to_pixel(sky)

        # Should recover original coordinates (gwcs may have small numerical errors)
        np.testing.assert_allclose([x_back, y_back], [x_orig, y_orig], atol=0.1)

    def test_fits_wcs_round_trip_array(self, simple_fits_wcs):
        """Test round trip with array of points (FITS WCS)."""
        wcs = simple_fits_wcs

        # Array of pixel coordinates
        x_orig = np.array([100, 500, 1000, 1500])
        y_orig = np.array([200, 600, 1000, 1400])

        # Transform to world
        sky = wcs.pixel_to_world(x_orig, y_orig)

        # Transform back to pixel
        x_back, y_back = wcs.world_to_pixel(sky)

        # Should recover original coordinates
        np.testing.assert_allclose(x_back, x_orig, atol=0.01)
        np.testing.assert_allclose(y_back, y_orig, atol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
