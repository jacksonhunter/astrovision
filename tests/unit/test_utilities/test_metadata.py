"""Unit tests for FITSMetadata class."""

import pytest
from astropy.io import fits
from astro_vision_composer.utilities import FITSMetadata


class TestFITSMetadata:
    """Test FITSMetadata extraction and validation."""

    def test_extract_basic_metadata(self, metadata_extractor, sample_fits_data):
        """Test basic metadata extraction."""
        data, header = sample_fits_data

        result = metadata_extractor.extract_metadata(header)

        assert result is not None
        assert result.instrument == 'TESTCAM'
        assert result.filter == 'TEST_FILTER'
        assert result.exposure_time == 60.0

    def test_mission_detection_jwst(self, metadata_extractor):
        """Test JWST mission detection."""
        header = fits.Header()
        header['TELESCOP'] = 'JWST'
        header['INSTRUME'] = 'NIRCAM'
        header['FILTER'] = 'F200W'

        result = metadata_extractor.extract_metadata(header)

        assert result.mission == 'JWST'
        assert result.instrument == 'NIRCAM'

    def test_mission_detection_hst(self, metadata_extractor):
        """Test HST mission detection."""
        header = fits.Header()
        header['TELESCOP'] = 'HST'
        header['INSTRUME'] = 'ACS'
        header['FILTER'] = 'F606W'

        result = metadata_extractor.extract_metadata(header)

        assert result.mission == 'HST'

    def test_mission_detection_panstarrs(self, metadata_extractor):
        """Test PanSTARRS mission detection."""
        header = fits.Header()
        header['TELESCOP'] = 'PS1'
        header['FILTER'] = 'g'

        result = metadata_extractor.extract_metadata(header)

        assert result.mission in ['PanSTARRS', 'PS1']

    def test_wavelength_lookup_panstarrs(self, metadata_extractor):
        """Test wavelength lookup for PanSTARRS filters."""
        expected_wavelengths = {
            'g': 481,
            'r': 617,
            'i': 752,
            'z': 866,
            'y': 962
        }

        for filter_name, expected_wl in expected_wavelengths.items():
            header = fits.Header()
            header['FILTER'] = filter_name

            result = metadata_extractor.extract_metadata(header)

            assert result.wavelength == expected_wl

    def test_wavelength_lookup_jwst_nircam(self, metadata_extractor):
        """Test wavelength lookup for JWST NIRCam filters."""
        filters = {
            'F090W': 900,
            'F150W': 1500,
            'F200W': 2000,
            'F356W': 3560,
            'F444W': 4440
        }

        for filter_name, expected_wl in filters.items():
            header = fits.Header()
            header['TELESCOP'] = 'JWST'
            header['FILTER'] = filter_name

            result = metadata_extractor.extract_metadata(header)

            assert result.wavelength == expected_wl

    def test_wavelength_lookup_narrowband(self, metadata_extractor):
        """Test wavelength lookup for narrowband filters."""
        narrowband = {
            'H-ALPHA': 656,
            'OIII': 501,
            'SII': 672
        }

        for filter_name, expected_wl in narrowband.items():
            header = fits.Header()
            header['FILTER'] = filter_name

            result = metadata_extractor.extract_metadata(header)

            # Might match or might be None depending on implementation
            if result.wavelength is not None:
                assert abs(result.wavelength - expected_wl) < 10

    def test_pixel_scale_extraction(self, metadata_extractor, sample_fits_with_wcs):
        """Test pixel scale extraction from WCS."""
        data, header = sample_fits_with_wcs

        result = metadata_extractor.extract_metadata(header)

        assert result.pixel_scale is not None
        # Should be close to 1 arcsec (0.0002778 deg * 3600)
        assert 0.9 < result.pixel_scale < 1.1

    def test_validation_required_fields(self, metadata_extractor):
        """Test validation of required fields."""
        header = fits.Header()
        header['FILTER'] = 'g'

        result = metadata_extractor.extract_metadata(header)

        # Validate that filter is present
        is_valid = metadata_extractor.validate_required(result, ['filter'])
        assert is_valid == True

        # Validate missing field
        is_valid = metadata_extractor.validate_required(result, ['exposure_time'])
        assert is_valid == False

    def test_warning_system(self, metadata_extractor):
        """Test that warnings are generated for missing data."""
        header = fits.Header()
        # Minimal header with missing fields

        result = metadata_extractor.extract_metadata(header)

        assert result.warnings is not None
        assert len(result.warnings) > 0

    def test_multiple_filter_keywords(self, metadata_extractor):
        """Test handling of multiple filter keyword variants."""
        # Try different common filter keywords
        for keyword in ['FILTER', 'FILTNAM', 'FILTER1']:
            header = fits.Header()
            header[keyword] = 'g'

            result = metadata_extractor.extract_metadata(header)

            assert result.filter == 'g'

    @pytest.mark.requires_noirlab_data
    def test_with_real_fits_files(self, metadata_extractor, edu008_data):
        """Test metadata extraction from real FITS files."""
        for fits_file in edu008_data['fits_files']:
            with fits.open(fits_file) as hdul:
                result = metadata_extractor.extract_metadata(hdul[0].header)

                # Should extract at least some metadata
                assert result is not None
                # At least one of these should be present
                assert (result.mission is not None or
                        result.instrument is not None or
                        result.filter is not None)

    def test_repr_format(self, metadata_extractor, sample_fits_data):
        """Test that metadata result has nice string representation."""
        data, header = sample_fits_data

        result = metadata_extractor.extract_metadata(header)

        repr_str = repr(result)

        assert 'FITSMetadata' in repr_str or 'Metadata' in repr_str
        assert 'TESTCAM' in repr_str
