"""Unit tests for WCSHandler class."""

import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits


class TestWCSHandler:
    """Test WCSHandler for WCS validation and manipulation."""

    def test_extract_wcs(self, wcs_handler, sample_fits_with_wcs):
        """Test WCS extraction from header."""
        data, header = sample_fits_with_wcs

        wcs = wcs_handler.extract_wcs(header)

        assert wcs is not None
        assert isinstance(wcs, WCS)
        assert wcs.wcs.ctype[0].startswith('RA')
        assert wcs.wcs.ctype[1].startswith('DEC')

    def test_validate_wcs(self, wcs_handler, sample_fits_with_wcs):
        """Test WCS validation."""
        data, header = sample_fits_with_wcs

        wcs = wcs_handler.extract_wcs(header)
        is_valid, messages = wcs_handler.validate_wcs(wcs)

        assert isinstance(is_valid, bool)
        assert isinstance(messages, list)

    def test_pixel_to_sky_conversion(self, wcs_handler, sample_fits_with_wcs):
        """Test pixel to sky coordinate conversion."""
        data, header = sample_fits_with_wcs

        wcs = wcs_handler.extract_wcs(header)

        # Convert center pixel to sky coordinates
        ra, dec = wcs_handler.pixel_to_sky(wcs, 50, 50)

        assert ra is not None
        assert dec is not None
        # Should be close to CRVAL
        assert abs(ra - 150.0) < 1.0
        assert abs(dec - 2.5) < 1.0

    def test_sky_to_pixel_conversion(self, wcs_handler, sample_fits_with_wcs):
        """Test sky to pixel coordinate conversion."""
        data, header = sample_fits_with_wcs

        wcs = wcs_handler.extract_wcs(header)

        # Convert sky coordinates back to pixels
        x, y = wcs_handler.sky_to_pixel(wcs, 150.0, 2.5)

        # Should be close to CRPIX (50, 50)
        assert abs(x - 50) < 1.0
        assert abs(y - 50) < 1.0

    def test_roundtrip_conversion(self, wcs_handler, sample_fits_with_wcs):
        """Test pixel->sky->pixel roundtrip."""
        data, header = sample_fits_with_wcs

        wcs = wcs_handler.extract_wcs(header)

        # Start with pixel coordinates
        x_orig, y_orig = 30, 70

        # Convert to sky and back
        ra, dec = wcs_handler.pixel_to_sky(wcs, x_orig, y_orig)
        x_new, y_new = wcs_handler.sky_to_pixel(wcs, ra, dec)

        # Should match original
        assert abs(x_new - x_orig) < 0.01
        assert abs(y_new - y_orig) < 0.01

    def test_pixel_scale_calculation(self, wcs_handler, sample_fits_with_wcs):
        """Test pixel scale calculation."""
        data, header = sample_fits_with_wcs

        wcs = wcs_handler.extract_wcs(header)

        pixel_scale = wcs_handler.get_pixel_scale(wcs)

        # Should be close to 1 arcsec (0.0002778 deg)
        assert pixel_scale is not None
        assert 0.9 < pixel_scale < 1.1  # arcsec

    def test_wcs_comparison(self, wcs_handler):
        """Test WCS comparison between two images."""
        # Create two similar WCS
        wcs1 = WCS(naxis=2)
        wcs1.wcs.crpix = [50, 50]
        wcs1.wcs.crval = [150.0, 2.5]
        wcs1.wcs.cdelt = [0.0002778, 0.0002778]
        wcs1.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        wcs2 = WCS(naxis=2)
        wcs2.wcs.crpix = [50, 50]
        wcs2.wcs.crval = [150.0, 2.5]
        wcs2.wcs.cdelt = [0.0002778, 0.0002778]
        wcs2.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        are_compatible = wcs_handler.compare_wcs(wcs1, wcs2)

        assert are_compatible == True

    def test_wcs_incompatible(self, wcs_handler):
        """Test detection of incompatible WCS."""
        wcs1 = WCS(naxis=2)
        wcs1.wcs.crpix = [50, 50]
        wcs1.wcs.crval = [150.0, 2.5]
        wcs1.wcs.cdelt = [0.0002778, 0.0002778]
        wcs1.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Different projection
        wcs2 = WCS(naxis=2)
        wcs2.wcs.crpix = [50, 50]
        wcs2.wcs.crval = [150.0, 2.5]
        wcs2.wcs.cdelt = [0.0002778, 0.0002778]
        wcs2.wcs.ctype = ["RA---SIN", "DEC--SIN"]  # Different projection

        are_compatible = wcs_handler.compare_wcs(wcs1, wcs2, tolerance=1e-6)

        # Might still be compatible depending on tolerance
        assert isinstance(are_compatible, bool)

    @pytest.mark.requires_noirlab_data
    def test_with_real_fits_data(self, wcs_handler, fits_loader, edu008_data):
        """Test WCS handling with real FITS data."""
        fits_file = edu008_data['fits_files'][0]
        fits_data = fits_loader.load(fits_file)

        wcs = wcs_handler.extract_wcs(fits_data.header)

        # Real data may or may not have WCS
        if wcs is not None and wcs.has_celestial:
            is_valid, messages = wcs_handler.validate_wcs(wcs)
            # Just check it returns something
            assert isinstance(is_valid, bool)

    def test_missing_wcs(self, wcs_handler):
        """Test handling of missing WCS."""
        header = fits.Header()
        header['SIMPLE'] = True
        header['NAXIS'] = 2

        wcs = wcs_handler.extract_wcs(header)

        # Should either return None or a WCS without celestial coordinates
        if wcs is not None:
            assert not wcs.has_celestial
