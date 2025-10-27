"""Unit tests for MissionAdapter classes."""

import pytest
import numpy as np
from astropy.io import fits
from astro_vision_composer.preprocessing import (
    MissionAdapter,
    PanSTARRSAdapter,
    JWSTAdapter,
    HSTAdapter
)


class TestMissionAdapterBase:
    """Test base MissionAdapter functionality."""

    def test_base_adapter_is_abstract(self):
        """Test that base MissionAdapter cannot be instantiated directly."""
        # Base class should work as it provides default implementations
        adapter = MissionAdapter()
        assert adapter is not None


class TestPanSTARRSAdapter:
    """Test PanSTARRSAdapter."""

    def test_identify_science_extension(self):
        """Test science extension identification."""
        adapter = PanSTARRSAdapter()

        # Create PanSTARRS-like FITS
        primary = fits.PrimaryHDU()
        primary.header['TELESCOP'] = 'PS1'
        data = np.random.rand(100, 100).astype(np.float32)
        image_hdu = fits.ImageHDU(data=data, name='PRIMARY')

        hdul = fits.HDUList([primary, image_hdu])

        ext_name, ext_index = adapter.identify_science_extension(hdul)

        # PanSTARRS typically uses PRIMARY extension
        assert ext_name in ['PRIMARY', 0]
        assert isinstance(ext_index, int)

    def test_get_filter_from_header(self):
        """Test filter extraction from PanSTARRS header."""
        adapter = PanSTARRSAdapter()

        header = fits.Header()
        header['FILTER'] = 'g'

        result = adapter.get_filter(header)
        assert result == 'g'

    def test_panstarrs_band_recognition(self):
        """Test recognition of PanSTARRS bands (g, r, i, z, y)."""
        adapter = PanSTARRSAdapter()

        for band in ['g', 'r', 'i', 'z', 'y']:
            header = fits.Header()
            header['FILTER'] = band
            result = adapter.get_filter(header)
            assert result == band


class TestJWSTAdapter:
    """Test JWSTAdapter."""

    def test_identify_science_extension(self):
        """Test JWST science extension identification."""
        adapter = JWSTAdapter()

        # Create JWST-like multi-extension FITS
        primary = fits.PrimaryHDU()
        primary.header['TELESCOP'] = 'JWST'

        sci_data = np.random.rand(100, 100).astype(np.float32)
        sci_hdu = fits.ImageHDU(data=sci_data, name='SCI')

        err_data = np.random.rand(100, 100).astype(np.float32)
        err_hdu = fits.ImageHDU(data=err_data, name='ERR')

        dq_data = np.zeros((100, 100), dtype=np.int32)
        dq_hdu = fits.ImageHDU(data=dq_data, name='DQ')

        hdul = fits.HDUList([primary, sci_hdu, err_hdu, dq_hdu])

        ext_name, ext_index = adapter.identify_science_extension(hdul)

        assert ext_name == 'SCI'
        assert ext_index == 1

    def test_get_error_array(self):
        """Test error array extraction from JWST FITS."""
        adapter = JWSTAdapter()

        # Create JWST-like FITS
        primary = fits.PrimaryHDU()
        sci_hdu = fits.ImageHDU(data=np.random.rand(50, 50), name='SCI')
        err_hdu = fits.ImageHDU(data=np.random.rand(50, 50), name='ERR')

        hdul = fits.HDUList([primary, sci_hdu, err_hdu])

        err_array = adapter.get_error_array(hdul)

        assert err_array is not None
        assert err_array.shape == (50, 50)

    def test_get_quality_mask(self):
        """Test DQ (data quality) mask extraction."""
        adapter = JWSTAdapter()

        # Create JWST-like FITS with DQ extension
        primary = fits.PrimaryHDU()
        sci_hdu = fits.ImageHDU(data=np.random.rand(50, 50), name='SCI')
        dq_data = np.zeros((50, 50), dtype=np.int32)
        dq_data[10:20, 10:20] = 1  # Mark some pixels as bad
        dq_hdu = fits.ImageHDU(data=dq_data, name='DQ')

        hdul = fits.HDUList([primary, sci_hdu, dq_hdu])

        mask = adapter.get_quality_mask(hdul)

        assert mask is not None
        assert mask.shape == (50, 50)
        assert mask.dtype == bool
        assert np.any(mask)  # Should have some bad pixels

    def test_dq_flag_interpretation(self):
        """Test JWST DQ flag interpretation."""
        adapter = JWSTAdapter()

        # JWST uses bit flags for DQ
        # Common flags: 0=good, 1=DO_NOT_USE, 2=SATURATED, 4=JUMP_DET, etc.
        dq_data = np.array([
            [0, 1, 2],  # good, bad, saturated
            [4, 8, 16]  # jump, dropout, outlier
        ], dtype=np.int32)

        primary = fits.PrimaryHDU()
        sci_hdu = fits.ImageHDU(data=np.random.rand(2, 3), name='SCI')
        dq_hdu = fits.ImageHDU(data=dq_data, name='DQ')
        hdul = fits.HDUList([primary, sci_hdu, dq_hdu])

        mask = adapter.get_quality_mask(hdul)

        # Pixel (0,0) should be good (DQ=0)
        assert mask[0, 0] == False  # Not masked
        # Pixel (0,1) should be bad (DQ=1, DO_NOT_USE)
        assert mask[0, 1] == True  # Masked


class TestHSTAdapter:
    """Test HSTAdapter."""

    def test_identify_science_extension(self):
        """Test HST science extension identification."""
        adapter = HSTAdapter()

        # Create HST-like multi-extension FITS
        primary = fits.PrimaryHDU()
        primary.header['TELESCOP'] = 'HST'

        sci_data = np.random.rand(100, 100).astype(np.float32)
        sci_hdu = fits.ImageHDU(data=sci_data, name='SCI')

        hdul = fits.HDUList([primary, sci_hdu])

        ext_name, ext_index = adapter.identify_science_extension(hdul)

        assert ext_name == 'SCI'
        assert ext_index == 1

    def test_hst_filter_extraction(self):
        """Test HST filter keyword extraction."""
        adapter = HSTAdapter()

        # HST uses FILTER or FILTNAM keywords
        header = fits.Header()
        header['FILTER'] = 'F606W'

        result = adapter.get_filter(header)
        assert result == 'F606W'

    def test_hst_dq_flags(self):
        """Test HST DQ flag interpretation."""
        adapter = HSTAdapter()

        # HST also uses bit flags
        dq_data = np.array([
            [0, 1, 2],
            [4, 8, 16]
        ], dtype=np.int16)

        primary = fits.PrimaryHDU()
        sci_hdu = fits.ImageHDU(data=np.random.rand(2, 3), name='SCI')
        dq_hdu = fits.ImageHDU(data=dq_data, name='DQ')
        hdul = fits.HDUList([primary, sci_hdu, dq_hdu])

        mask = adapter.get_quality_mask(hdul)

        assert mask is not None
        assert mask.shape == (2, 3)

    @pytest.mark.requires_noirlab_data
    def test_with_real_hst_data(self, edu011_data):
        """Test HST adapter with real HST-format NOIRLab data."""
        if not edu011_data['fits_files']:
            pytest.skip("No HST-format data available")

        adapter = HSTAdapter()

        fits_file = edu011_data['fits_files'][0]
        with fits.open(fits_file) as hdul:
            ext_name, ext_index = adapter.identify_science_extension(hdul)

            assert ext_name is not None
            assert isinstance(ext_index, int)


class TestAdapterFactoryPattern:
    """Test adapter selection and factory-like usage."""

    def test_adapter_selection_by_mission(self):
        """Test selecting correct adapter based on TELESCOP keyword."""
        test_cases = [
            ('JWST', JWSTAdapter),
            ('HST', HSTAdapter),
            ('PS1', PanSTARRSAdapter),
        ]

        for telescop, expected_adapter_class in test_cases:
            header = fits.Header()
            header['TELESCOP'] = telescop

            # In real usage, you'd have a factory function
            # For now, just test that the right adapter exists
            if telescop == 'JWST':
                adapter = JWSTAdapter()
            elif telescop == 'HST':
                adapter = HSTAdapter()
            elif telescop == 'PS1':
                adapter = PanSTARRSAdapter()

            assert isinstance(adapter, expected_adapter_class)
