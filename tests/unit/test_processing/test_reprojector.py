"""Unit tests for Reprojector class."""

import pytest
import numpy as np
from astropy.wcs import WCS


class TestReprojector:
    """Test Reprojector for image alignment."""

    def test_reproject_single_image(self, reprojector):
        """Test reprojecting a single image to target WCS."""
        # Create source image and WCS
        source_data = np.random.rand(100, 100)

        source_wcs = WCS(naxis=2)
        source_wcs.wcs.crpix = [50, 50]
        source_wcs.wcs.crval = [150.0, 2.5]
        source_wcs.wcs.cdelt = [0.0002778, 0.0002778]
        source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create target WCS (slightly offset)
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crpix = [50, 50]
        target_wcs.wcs.crval = [150.01, 2.51]  # Slightly offset
        target_wcs.wcs.cdelt = [0.0002778, 0.0002778]
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        reprojected, footprint = reprojector.reproject(
            source_data, source_wcs, target_wcs, shape_out=(100, 100)
        )

        assert reprojected is not None
        assert reprojected.shape == (100, 100)
        assert footprint is not None
        assert footprint.shape == (100, 100)

    def test_reproject_interpolation_method(self, reprojector):
        """Test reprojection with interpolation method."""
        source_data = np.random.rand(100, 100)

        source_wcs = WCS(naxis=2)
        source_wcs.wcs.crpix = [50, 50]
        source_wcs.wcs.crval = [150.0, 2.5]
        source_wcs.wcs.cdelt = [0.0003, 0.0003]
        source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crpix = [50, 50]
        target_wcs.wcs.crval = [150.0, 2.5]
        target_wcs.wcs.cdelt = [0.0002, 0.0002]  # Different pixel scale
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        reprojected, footprint = reprojector.reproject(
            source_data, source_wcs, target_wcs,
            shape_out=(150, 150), method='interp'
        )

        assert reprojected.shape == (150, 150)

    def test_align_image_set(self, reprojector):
        """Test aligning multiple images to common WCS."""
        # Create 3 images with slightly different WCS
        images = {}
        for i, offset in enumerate([0.0, 0.01, 0.02]):
            data = np.random.rand(100, 100)

            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [50, 50]
            wcs.wcs.crval = [150.0 + offset, 2.5 + offset]
            wcs.wcs.cdelt = [0.0002778, 0.0002778]
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

            images[f'band_{i}'] = (data, wcs)

        # Align to first image WCS
        aligned = reprojector.align_images(images, reference='band_0')

        assert len(aligned) == 3
        assert all(aligned[key].shape == (100, 100) for key in aligned)

    def test_footprint_calculation(self, reprojector):
        """Test footprint calculation during reprojection."""
        source_data = np.ones((50, 50))  # Uniform image

        source_wcs = WCS(naxis=2)
        source_wcs.wcs.crpix = [25, 25]
        source_wcs.wcs.crval = [150.0, 2.5]
        source_wcs.wcs.cdelt = [0.001, 0.001]
        source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crpix = [50, 50]
        target_wcs.wcs.crval = [150.0, 2.5]
        target_wcs.wcs.cdelt = [0.001, 0.001]
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        reprojected, footprint = reprojector.reproject(
            source_data, source_wcs, target_wcs, shape_out=(100, 100)
        )

        # Footprint should show where source data maps to
        assert footprint is not None
        assert 0 <= footprint.min() <= footprint.max() <= 1

    def test_flux_conservation(self, reprojector):
        """Test flux conservation during reprojection."""
        # Create image with known total flux
        source_data = np.ones((100, 100)) * 100

        source_wcs = WCS(naxis=2)
        source_wcs.wcs.crpix = [50, 50]
        source_wcs.wcs.crval = [150.0, 2.5]
        source_wcs.wcs.cdelt = [0.0002778, 0.0002778]
        source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Same WCS, so should preserve flux exactly
        reprojected, footprint = reprojector.reproject(
            source_data, source_wcs, source_wcs, shape_out=(100, 100)
        )

        # Total flux should be approximately preserved
        assert abs(np.nansum(reprojected) - np.sum(source_data)) < 100

    @pytest.mark.requires_noirlab_data
    def test_with_real_fits_data(self, reprojector, fits_loader, wcs_handler, edu008_data):
        """Test reprojection with real FITS data."""
        if len(edu008_data['fits_files']) < 2:
            pytest.skip("Need at least 2 FITS files")

        # Load two images
        fits1 = fits_loader.load(edu008_data['fits_files'][0])
        fits2 = fits_loader.load(edu008_data['fits_files'][1])

        wcs1 = wcs_handler.extract_wcs(fits1.header)
        wcs2 = wcs_handler.extract_wcs(fits2.header)

        if wcs1 is None or wcs2 is None or not wcs1.has_celestial:
            pytest.skip("FITS files don't have valid WCS")

        # Reproject second image to first image's WCS
        reprojected, footprint = reprojector.reproject(
            fits2.data, wcs2, wcs1, shape_out=fits1.data.shape
        )

        assert reprojected is not None
        assert reprojected.shape == fits1.data.shape

    def test_different_pixel_scales(self, reprojector):
        """Test reprojection between different pixel scales."""
        source_data = np.random.rand(100, 100)

        # 1 arcsec/pixel
        source_wcs = WCS(naxis=2)
        source_wcs.wcs.crpix = [50, 50]
        source_wcs.wcs.crval = [150.0, 2.5]
        source_wcs.wcs.cdelt = [0.0002778, 0.0002778]
        source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # 0.5 arcsec/pixel (finer)
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crpix = [100, 100]
        target_wcs.wcs.crval = [150.0, 2.5]
        target_wcs.wcs.cdelt = [0.0001389, 0.0001389]
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Output should be ~2x larger
        reprojected, footprint = reprojector.reproject(
            source_data, source_wcs, target_wcs, shape_out=(200, 200)
        )

        assert reprojected.shape == (200, 200)
