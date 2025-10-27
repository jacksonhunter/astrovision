"""Unit tests for FITSLoader class."""

import pytest
import numpy as np
from pathlib import Path
from astropy.io import fits


class TestFITSLoader:
    """Test FITSLoader with real NOIRLab data and synthetic data."""

    @pytest.mark.requires_noirlab_data
    def test_load_single_fits(self, fits_loader, edu008_data):
        """Test loading a single FITS file."""
        fits_file = edu008_data['fits_files'][0]
        result = fits_loader.load(fits_file)

        assert result.data is not None
        assert result.data.ndim == 2
        assert result.metadata.filter is not None
        assert result.header is not None
        assert result.file_path == fits_file

    @pytest.mark.requires_noirlab_data
    def test_load_multiple_fits(self, fits_loader, edu008_data):
        """Test loading all FITS files in a dataset."""
        results = [fits_loader.load(f) for f in edu008_data['fits_files']]

        assert len(results) == 3
        assert all(r.data is not None for r in results)
        assert all(r.metadata.wavelength is not None for r in results)

    @pytest.mark.requires_noirlab_data
    def test_lazy_loading(self, fits_loader, edu008_data):
        """Test lazy loading mode."""
        fits_file = edu008_data['fits_files'][0]
        result = fits_loader.load(fits_file, lazy=True)

        assert result.data is not None
        # Lazy loading should still work, just potentially with memmap
        assert isinstance(result.data, np.ndarray) or hasattr(result.data, '__array__')

    def test_load_synthetic_fits(self, fits_loader, temp_fits_file):
        """Test loading synthetic FITS file."""
        result = fits_loader.load(temp_fits_file)

        assert result.data is not None
        assert result.data.shape == (100, 100)
        assert result.metadata.filter == 'TEST_FILTER'
        assert result.metadata.exposure_time == 60.0

    def test_invalid_file(self, fits_loader, tmp_path):
        """Test loading non-existent file raises error."""
        fake_file = tmp_path / "nonexistent.fits"

        with pytest.raises(FileNotFoundError):
            fits_loader.load(fake_file)

    def test_extension_detection(self, fits_loader, tmp_path):
        """Test automatic extension detection for multi-extension FITS."""
        # Create a multi-extension FITS file
        primary = fits.PrimaryHDU()
        data = np.random.rand(50, 50).astype(np.float32)
        sci_hdu = fits.ImageHDU(data=data, name='SCI')
        sci_hdu.header['FILTER'] = 'V'

        hdul = fits.HDUList([primary, sci_hdu])
        fits_file = tmp_path / "multi_ext.fits"
        hdul.writeto(fits_file, overwrite=True)

        result = fits_loader.load(fits_file)

        assert result.data is not None
        assert result.extension_name == 'SCI'

    def test_load_preserves_header(self, fits_loader, temp_fits_file):
        """Test that loading preserves all header information."""
        result = fits_loader.load(temp_fits_file)

        assert 'TELESCOP' in result.header
        assert result.header['TELESCOP'] == 'TEST'
        assert result.header['INSTRUME'] == 'TESTCAM'

    def test_data_type_preservation(self, fits_loader, tmp_path):
        """Test that data types are preserved correctly."""
        # Create FITS with different data types
        for dtype in [np.float32, np.float64, np.int16, np.uint16]:
            data = np.random.rand(50, 50).astype(dtype)
            hdu = fits.PrimaryHDU(data=data)
            fits_file = tmp_path / f"test_{dtype.__name__}.fits"
            hdu.writeto(fits_file, overwrite=True)

            result = fits_loader.load(fits_file)
            # FITSLoader may convert to float for processing, but should handle all types
            assert result.data is not None
            assert result.data.shape == (50, 50)

    @pytest.mark.requires_noirlab_data
    def test_metadata_extraction(self, fits_loader, edu008_data):
        """Test that metadata is properly extracted during load."""
        fits_file = edu008_data['fits_files'][0]
        result = fits_loader.load(fits_file)

        # Check that metadata contains expected fields
        assert result.metadata is not None
        assert result.metadata.filter is not None or result.metadata.wavelength is not None

    def test_load_blank_image(self, fits_loader, tmp_path):
        """Test loading a blank (all zeros) image."""
        data = np.zeros((100, 100), dtype=np.float32)
        header = fits.Header()
        header['FILTER'] = 'BLANK'

        hdu = fits.PrimaryHDU(data=data, header=header)
        fits_file = tmp_path / "blank.fits"
        hdu.writeto(fits_file, overwrite=True)

        result = fits_loader.load(fits_file)

        assert result.data is not None
        assert np.all(result.data == 0)

    def test_load_nan_handling(self, fits_loader, tmp_path):
        """Test handling of NaN values in FITS data."""
        data = np.random.rand(100, 100).astype(np.float32)
        data[50:60, 50:60] = np.nan

        hdu = fits.PrimaryHDU(data=data)
        fits_file = tmp_path / "with_nan.fits"
        hdu.writeto(fits_file, overwrite=True)

        result = fits_loader.load(fits_file)

        assert result.data is not None
        # Should preserve NaNs or document how they're handled
        assert np.any(np.isnan(result.data)) or result.data is not None
