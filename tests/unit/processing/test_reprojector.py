"""Unit tests for Reprojector class.

Tests image reprojection and alignment using synthetic data with known WCS.
"""

import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u

from astro_vision_composer.processing.reprojector import Reprojector


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_wcs_1():
    """Create a simple WCS (pixel scale 1 arcsec/pixel, center at RA=45, Dec=30)."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.5, 50.5]
    wcs.wcs.crval = [45.0, 30.0]
    wcs.wcs.cdelt = [1.0/3600, 1.0/3600]  # 1 arcsec/pixel
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs


@pytest.fixture
def simple_wcs_2():
    """Create a second WCS (pixel scale 0.5 arcsec/pixel, rotated, offset center)."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [100.5, 100.5]
    wcs.wcs.crval = [45.01, 30.01]  # Slightly offset
    wcs.wcs.cdelt = [0.5/3600, 0.5/3600]  # 0.5 arcsec/pixel
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs


@pytest.fixture
def rotated_wcs():
    """Create a WCS with 45-degree rotation."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.5, 50.5]
    wcs.wcs.crval = [45.0, 30.0]
    # CD matrix for 45-degree rotation
    angle = np.radians(45)
    scale = 1.0 / 3600  # 1 arcsec/pixel
    wcs.wcs.cd = np.array([
        [scale * np.cos(angle), -scale * np.sin(angle)],
        [scale * np.sin(angle), scale * np.cos(angle)]
    ])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs


@pytest.fixture
def gaussian_source_image():
    """Create a 100x100 image with Gaussian source at center."""
    y, x = np.ogrid[-50:50, -50:50]
    gaussian = np.exp(-(x**2 + y**2) / (2 * 10**2))
    return gaussian.astype(np.float32)


@pytest.fixture
def star_field_image():
    """Create image with multiple point sources."""
    image = np.zeros((100, 100), dtype=np.float32)
    # Add several "stars"
    stars = [(25, 25), (75, 75), (25, 75), (75, 25), (50, 50)]
    for y, x in stars:
        yy, xx = np.ogrid[max(0, y-5):min(100, y+5), max(0, x-5):min(100, x+5)]
        r = np.sqrt((xx - x)**2 + (yy - y)**2)
        image[yy, xx] += np.exp(-r**2 / 2)
    return image


# ============================================================================
# Test Reprojector Initialization
# ============================================================================

class TestReprojectorInit:
    """Test Reprojector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        reprojector = Reprojector()
        assert reprojector.method == 'interp'
        assert reprojector.order == 1

    def test_explicit_interp_method(self):
        """Test initialization with explicit 'interp' method."""
        reprojector = Reprojector(method='interp', order=2)
        assert reprojector.method == 'interp'
        assert reprojector.order == 2

    def test_exact_method(self):
        """Test initialization with 'exact' method."""
        reprojector = Reprojector(method='exact')
        assert reprojector.method == 'exact'

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be 'interp' or 'exact'"):
            Reprojector(method='invalid')


# ============================================================================
# Test reproject_to_target()
# ============================================================================

class TestReprojectToTarget:
    """Test reproject_to_target() method."""

    def test_basic_reprojection_interp(self, gaussian_source_image, simple_wcs_1, simple_wcs_2):
        """Test basic reprojection with 'interp' method."""
        reprojector = Reprojector(method='interp')

        result, footprint = reprojector.reproject_to_target(
            gaussian_source_image,
            simple_wcs_1,
            simple_wcs_2,
            target_shape=(200, 200)
        )

        # Verify output
        assert result is not None
        assert result.shape == (200, 200)
        assert footprint.shape == (200, 200)
        # NaNs are expected where there's no overlap
        assert np.any(np.isfinite(result))  # At least some valid pixels
        assert np.all(footprint >= 0)
        assert np.all(footprint <= 1)

    def test_basic_reprojection_exact(self, gaussian_source_image, simple_wcs_1, simple_wcs_2):
        """Test basic reprojection with 'exact' method."""
        reprojector = Reprojector(method='exact')

        result, footprint = reprojector.reproject_to_target(
            gaussian_source_image,
            simple_wcs_1,
            simple_wcs_2,
            target_shape=(200, 200)
        )

        # Verify output
        assert result is not None
        assert result.shape == (200, 200)
        assert footprint.shape == (200, 200)
        # NaNs are expected where there's no overlap
        assert np.any(np.isfinite(result))  # At least some valid pixels

    def test_method_override(self, gaussian_source_image, simple_wcs_1, simple_wcs_2):
        """Test method override in reproject_to_target()."""
        # Initialize with 'interp', but override with 'exact'
        reprojector = Reprojector(method='interp')

        result, footprint = reprojector.reproject_to_target(
            gaussian_source_image,
            simple_wcs_1,
            simple_wcs_2,
            target_shape=(100, 100),
            method='exact'
        )

        assert result is not None
        assert result.shape == (100, 100)

    def test_rotated_wcs_reprojection(self, gaussian_source_image, simple_wcs_1, rotated_wcs):
        """Test reprojection with rotated WCS."""
        reprojector = Reprojector(method='interp')

        result, footprint = reprojector.reproject_to_target(
            gaussian_source_image,
            simple_wcs_1,
            rotated_wcs,
            target_shape=(100, 100)
        )

        # Should still work
        assert result is not None
        assert result.shape == (100, 100)
        # Gaussian should still be visible but rotated (check using nanmax for NaN handling)
        assert np.nanmax(result) > 0

    def test_empty_source_raises(self, simple_wcs_1, simple_wcs_2):
        """Test that empty source data raises ValueError."""
        reprojector = Reprojector()
        empty_data = np.array([])

        with pytest.raises(ValueError, match="source_data is empty or None"):
            reprojector.reproject_to_target(
                empty_data,
                simple_wcs_1,
                simple_wcs_2,
                target_shape=(100, 100)
            )

    def test_none_source_raises(self, simple_wcs_1, simple_wcs_2):
        """Test that None source data raises ValueError."""
        reprojector = Reprojector()

        with pytest.raises(ValueError, match="source_data is empty or None"):
            reprojector.reproject_to_target(
                None,
                simple_wcs_1,
                simple_wcs_2,
                target_shape=(100, 100)
            )

    def test_no_celestial_source_wcs_raises(self, gaussian_source_image, simple_wcs_2):
        """Test that source WCS without celestial coordinates raises ValueError."""
        reprojector = Reprojector()

        # Create invalid WCS (no celestial)
        bad_wcs = WCS(naxis=2)
        bad_wcs.wcs.ctype = ['LINEAR', 'LINEAR']

        with pytest.raises(ValueError, match="source_wcs has no celestial coordinates"):
            reprojector.reproject_to_target(
                gaussian_source_image,
                bad_wcs,
                simple_wcs_2,
                target_shape=(100, 100)
            )

    def test_no_celestial_target_wcs_raises(self, gaussian_source_image, simple_wcs_1):
        """Test that target WCS without celestial coordinates raises ValueError."""
        reprojector = Reprojector()

        # Create invalid WCS (no celestial)
        bad_wcs = WCS(naxis=2)
        bad_wcs.wcs.ctype = ['LINEAR', 'LINEAR']

        with pytest.raises(ValueError, match="target_wcs has no celestial coordinates"):
            reprojector.reproject_to_target(
                gaussian_source_image,
                simple_wcs_1,
                bad_wcs,
                target_shape=(100, 100)
            )


# ============================================================================
# Test align_image_set()
# ============================================================================

class TestAlignImageSet:
    """Test align_image_set() method."""

    def test_align_three_images(self, gaussian_source_image, simple_wcs_1, simple_wcs_2, rotated_wcs):
        """Test aligning a set of 3 images."""
        reprojector = Reprojector(method='interp')

        # Create 3 images with different WCS
        images = {
            'image1': (gaussian_source_image, simple_wcs_1),
            'image2': (gaussian_source_image * 0.8, simple_wcs_2),
            'image3': (gaussian_source_image * 0.6, rotated_wcs)
        }

        # Align to image1
        aligned = reprojector.align_image_set(images, reference='image1')

        # Verify results
        assert len(aligned) == 3
        assert 'image1' in aligned
        assert 'image2' in aligned
        assert 'image3' in aligned

        # Reference image should be unchanged
        assert np.array_equal(aligned['image1'], gaussian_source_image)

        # Other images should be reprojected
        assert aligned['image2'].shape == gaussian_source_image.shape
        assert aligned['image3'].shape == gaussian_source_image.shape

    def test_align_auto_reference(self, gaussian_source_image, simple_wcs_1, simple_wcs_2):
        """Test align_image_set() with automatic reference selection."""
        reprojector = Reprojector()

        images = {
            'g': (gaussian_source_image, simple_wcs_1),
            'r': (gaussian_source_image * 0.9, simple_wcs_2)
        }

        # Don't specify reference (should use first: 'g')
        aligned = reprojector.align_image_set(images)

        assert len(aligned) == 2
        # Reference 'g' should be unchanged
        assert np.array_equal(aligned['g'], gaussian_source_image)
        # 'r' should be reprojected
        assert aligned['r'].shape == gaussian_source_image.shape

    def test_empty_images_dict_raises(self):
        """Test that empty images dict raises ValueError."""
        reprojector = Reprojector()

        with pytest.raises(ValueError, match="images dictionary is empty"):
            reprojector.align_image_set({})

    def test_invalid_reference_raises(self, gaussian_source_image, simple_wcs_1):
        """Test that invalid reference name raises ValueError."""
        reprojector = Reprojector()

        images = {
            'image1': (gaussian_source_image, simple_wcs_1)
        }

        with pytest.raises(ValueError, match="Reference 'invalid' not found"):
            reprojector.align_image_set(images, reference='invalid')

    def test_align_with_method_override(self, gaussian_source_image, simple_wcs_1, simple_wcs_2):
        """Test align_image_set() with method override."""
        reprojector = Reprojector(method='interp')

        images = {
            'a': (gaussian_source_image, simple_wcs_1),
            'b': (gaussian_source_image * 0.7, simple_wcs_2)
        }

        # Override with 'exact' method
        aligned = reprojector.align_image_set(images, reference='a', method='exact')

        assert len(aligned) == 2
        assert aligned['b'].shape == gaussian_source_image.shape


# ============================================================================
# Test reproject_from_fits_data()
# ============================================================================

class TestReprojectFromFITSData:
    """Test reproject_from_fits_data() convenience method."""

    def test_reproject_fits_data_objects(self, gaussian_source_image, simple_wcs_1, simple_wcs_2, tmp_path):
        """Test reprojection using FITSData-like objects."""
        from astro_vision_composer.preprocessing.fits_loader import FITSData
        from pathlib import Path

        reprojector = Reprojector()

        # Create temporary FITS files for FITSData objects
        source_file = tmp_path / "source.fits"
        target_file = tmp_path / "target.fits"

        # Create mock FITSData objects
        source_fits = FITSData(
            filepath=source_file,
            science=gaussian_source_image,
            header=fits.Header(),
            wcs=simple_wcs_1,
            error=None,
            dq=None,
            metadata=None,
            extension_name='SCI',
            extension_version=1
        )

        target_data = np.zeros((200, 200), dtype=np.float32)
        target_fits = FITSData(
            filepath=target_file,
            science=target_data,
            header=fits.Header(),
            wcs=simple_wcs_2,
            error=None,
            dq=None,
            metadata=None,
            extension_name='SCI',
            extension_version=1
        )

        # Reproject
        result, footprint = reprojector.reproject_from_fits_data(source_fits, target_fits)

        assert result is not None
        assert result.shape == (200, 200)
        assert footprint.shape == (200, 200)

    def test_source_no_wcs_raises(self, gaussian_source_image, simple_wcs_2, tmp_path):
        """Test that source FITSData without WCS raises ValueError."""
        from astro_vision_composer.preprocessing.fits_loader import FITSData

        reprojector = Reprojector()

        source_file = tmp_path / "source.fits"
        target_file = tmp_path / "target.fits"

        # Source without WCS
        source_fits = FITSData(
            filepath=source_file,
            science=gaussian_source_image,
            header=fits.Header(),
            wcs=None,  # No WCS
            error=None,
            dq=None,
            metadata=None,
            extension_name='SCI',
            extension_version=1
        )

        target_fits = FITSData(
            filepath=target_file,
            science=np.zeros((100, 100)),
            header=fits.Header(),
            wcs=simple_wcs_2,
            error=None,
            dq=None,
            metadata=None,
            extension_name='SCI',
            extension_version=1
        )

        with pytest.raises(ValueError, match="Source FITSData has no WCS"):
            reprojector.reproject_from_fits_data(source_fits, target_fits)

    def test_target_no_wcs_raises(self, gaussian_source_image, simple_wcs_1, tmp_path):
        """Test that target FITSData without WCS raises ValueError."""
        from astro_vision_composer.preprocessing.fits_loader import FITSData

        reprojector = Reprojector()

        source_file = tmp_path / "source.fits"
        target_file = tmp_path / "target.fits"

        source_fits = FITSData(
            filepath=source_file,
            science=gaussian_source_image,
            header=fits.Header(),
            wcs=simple_wcs_1,
            error=None,
            dq=None,
            metadata=None,
            extension_name='SCI',
            extension_version=1
        )

        # Target without WCS
        target_fits = FITSData(
            filepath=target_file,
            science=np.zeros((100, 100)),
            header=fits.Header(),
            wcs=None,  # No WCS
            error=None,
            dq=None,
            metadata=None,
            extension_name='SCI',
            extension_version=1
        )

        with pytest.raises(ValueError, match="Target FITSData has no WCS"):
            reprojector.reproject_from_fits_data(source_fits, target_fits)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestReprojectorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_identity_reprojection(self, gaussian_source_image, simple_wcs_1):
        """Test reprojection where source and target WCS are identical."""
        reprojector = Reprojector()

        result, footprint = reprojector.reproject_to_target(
            gaussian_source_image,
            simple_wcs_1,
            simple_wcs_1,
            target_shape=gaussian_source_image.shape
        )

        # Result should be very close to original
        assert result.shape == gaussian_source_image.shape
        # Footprint should be 1 everywhere (full coverage)
        assert np.all(footprint > 0.99)

    def test_different_pixel_scales(self, star_field_image, simple_wcs_1, simple_wcs_2):
        """Test reprojection with different pixel scales."""
        reprojector = Reprojector(method='interp')

        # simple_wcs_1 is 1 arcsec/pixel, simple_wcs_2 is 0.5 arcsec/pixel
        result, footprint = reprojector.reproject_to_target(
            star_field_image,
            simple_wcs_1,
            simple_wcs_2,
            target_shape=(200, 200)
        )

        # Should work, finer pixel scale means larger image
        assert result is not None
        assert result.shape == (200, 200)

    def test_interpolation_orders(self, gaussian_source_image, simple_wcs_1, simple_wcs_2):
        """Test different interpolation orders."""
        # Test order 0 (nearest neighbor)
        reprojector_0 = Reprojector(method='interp', order=0)
        result_0, _ = reprojector_0.reproject_to_target(
            gaussian_source_image, simple_wcs_1, simple_wcs_2, (100, 100)
        )

        # Test order 1 (bilinear)
        reprojector_1 = Reprojector(method='interp', order=1)
        result_1, _ = reprojector_1.reproject_to_target(
            gaussian_source_image, simple_wcs_1, simple_wcs_2, (100, 100)
        )

        # Test order 2 (bicubic)
        reprojector_2 = Reprojector(method='interp', order=2)
        result_2, _ = reprojector_2.reproject_to_target(
            gaussian_source_image, simple_wcs_1, simple_wcs_2, (100, 100)
        )

        # All should produce valid results
        assert result_0 is not None
        assert result_1 is not None
        assert result_2 is not None

        # Higher orders should be smoother (but we don't test this quantitatively)

    def test_small_target_shape(self, gaussian_source_image, simple_wcs_1, simple_wcs_2):
        """Test reprojection to very small target shape."""
        reprojector = Reprojector()

        result, footprint = reprojector.reproject_to_target(
            gaussian_source_image,
            simple_wcs_1,
            simple_wcs_2,
            target_shape=(10, 10)
        )

        assert result.shape == (10, 10)
        assert footprint.shape == (10, 10)

    def test_large_offset_reprojection(self, gaussian_source_image):
        """Test reprojection with large WCS offset (minimal overlap)."""
        reprojector = Reprojector()

        # Source WCS
        wcs_source = WCS(naxis=2)
        wcs_source.wcs.crpix = [50.5, 50.5]
        wcs_source.wcs.crval = [45.0, 30.0]
        wcs_source.wcs.cdelt = [1.0/3600, 1.0/3600]
        wcs_source.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        # Target WCS far away
        wcs_target = WCS(naxis=2)
        wcs_target.wcs.crpix = [50.5, 50.5]
        wcs_target.wcs.crval = [90.0, 60.0]  # Very different center
        wcs_target.wcs.cdelt = [1.0/3600, 1.0/3600]
        wcs_target.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        result, footprint = reprojector.reproject_to_target(
            gaussian_source_image,
            wcs_source,
            wcs_target,
            target_shape=(100, 100)
        )

        # Should produce result but with mostly zero footprint (no overlap)
        assert result.shape == (100, 100)
        assert np.sum(footprint) < 10  # Very little overlap
