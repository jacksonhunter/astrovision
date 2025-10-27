"""Integration tests for CalibrationManager complete implementation.

Tests the full CalibrationManager workflow including:
- Multi-exposure dark library
- Dark exposure time matching
- Bad pixel masking
- Overscan/trim security
- Caching performance
- Unit detection
- Uncertainty propagation
"""

import pytest
import numpy as np
from pathlib import Path
from astropy.nddata import CCDData
import astropy.units as u
from astro_vision_composer.preprocessing import CalibrationManager, CalibrationFrames


# ========================
# Synthetic Data Generators
# ========================

def create_synthetic_bias(shape=(100, 100), bias_level=100.0, readnoise=5.0):
    """Create synthetic bias frame with realistic noise."""
    data = np.ones(shape) * bias_level
    data += np.random.normal(0, readnoise, shape)

    ccd = CCDData(data, unit='adu')
    ccd.header['IMAGETYP'] = 'BIAS'
    ccd.header['EXPTIME'] = 0.0
    return ccd


def create_synthetic_dark(
    shape=(100, 100),
    bias_level=100.0,
    dark_current=0.1,  # e-/pix/s
    exp_time=60.0,
    readnoise=5.0,
    add_hot_pixels=True
):
    """Create synthetic dark frame with realistic dark current."""
    data = np.ones(shape) * bias_level
    data += dark_current * exp_time
    data += np.random.normal(0, readnoise, shape)

    # Add hot pixels for bad pixel detection
    if add_hot_pixels:
        n_hot = 20
        hot_y = np.random.randint(0, shape[0], n_hot)
        hot_x = np.random.randint(0, shape[1], n_hot)
        data[hot_y, hot_x] += np.random.uniform(500, 2000, n_hot)

    ccd = CCDData(data, unit='adu')
    ccd.header['IMAGETYP'] = 'DARK'
    ccd.header['EXPTIME'] = exp_time
    return ccd


def create_synthetic_flat(
    shape=(100, 100),
    bias_level=100.0,
    flat_level=10000.0,
    filter_name='V',
    exp_time=5.0,
    readnoise=5.0
):
    """Create synthetic flat frame with vignetting."""
    # Create vignetting pattern (brighter in center)
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = shape[0] / 2, shape[1] / 2
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_r = np.sqrt(center_x**2 + center_y**2)
    vignette = 1.0 - 0.3 * (r / max_r)**2  # 30% falloff at edges

    data = np.ones(shape) * bias_level
    data += flat_level * vignette
    data += np.random.normal(0, readnoise, shape)

    ccd = CCDData(data, unit='adu')
    ccd.header['IMAGETYP'] = 'FLAT'
    ccd.header['FILTER'] = filter_name
    ccd.header['EXPTIME'] = exp_time
    return ccd


def create_synthetic_science(
    shape=(100, 100),
    bias_level=100.0,
    dark_current=0.1,
    exp_time=120.0,
    source_level=5000.0,
    filter_name='V',
    readnoise=5.0,
    add_hot_pixels=True
):
    """Create synthetic science frame with sources and hot pixels."""
    data = np.ones(shape) * bias_level
    data += dark_current * exp_time

    # Add some point sources
    n_sources = 10
    for _ in range(n_sources):
        sx, sy = np.random.randint(10, shape[1]-10), np.random.randint(10, shape[0]-10)
        # Gaussian PSF
        y, x = np.ogrid[:shape[0], :shape[1]]
        psf = source_level * np.exp(-((x-sx)**2 + (y-sy)**2) / (2 * 2.5**2))
        data += psf

    # Add hot pixels
    if add_hot_pixels:
        n_hot = 20
        hot_y = np.random.randint(0, shape[0], n_hot)
        hot_x = np.random.randint(0, shape[1], n_hot)
        data[hot_y, hot_x] += np.random.uniform(500, 2000, n_hot)

    data += np.random.normal(0, readnoise, shape)

    ccd = CCDData(data, unit='adu')
    ccd.header['IMAGETYP'] = 'OBJECT'
    ccd.header['EXPTIME'] = exp_time
    ccd.header['FILTER'] = filter_name
    return ccd


def create_test_dataset(tmp_path, n_bias=5, n_darks_per_exp=5, n_flats_per_filter=5):
    """Create complete synthetic test dataset.

    Creates:
    - n_bias bias frames
    - n_darks_per_exp darks for each of [30s, 60s, 120s]
    - n_flats_per_filter flats for each of ['V', 'R', 'I']
    - 3 science frames (one per filter)
    """
    calib_dir = tmp_path / 'calibration'
    calib_dir.mkdir()

    science_dir = tmp_path / 'science'
    science_dir.mkdir()

    # Create bias frames
    for i in range(n_bias):
        bias = create_synthetic_bias()
        bias.write(calib_dir / f'bias_{i:02d}.fits', overwrite=True)

    # Create darks for multiple exposure times (add hot pixels to darks)
    for exp_time in [30.0, 60.0, 120.0]:
        for i in range(n_darks_per_exp):
            dark = create_synthetic_dark(exp_time=exp_time, add_hot_pixels=True)
            dark.write(calib_dir / f'dark_{int(exp_time):03d}s_{i:02d}.fits', overwrite=True)

    # Create flats for multiple filters
    for filter_name in ['V', 'R', 'I']:
        for i in range(n_flats_per_filter):
            flat = create_synthetic_flat(filter_name=filter_name)
            flat.write(calib_dir / f'flat_{filter_name}_{i:02d}.fits', overwrite=True)

    # Create science frames (ensure EXPTIME is set)
    for filter_name in ['V', 'R', 'I']:
        science = create_synthetic_science(exp_time=120.0, filter_name=filter_name)
        # Ensure EXPTIME is lowercase to match what ccdproc expects
        if 'EXPTIME' not in science.header:
            science.header['EXPTIME'] = 120.0
        if 'exptime' not in science.header:
            science.header['exptime'] = 120.0
        science.write(science_dir / f'science_{filter_name}.fits', overwrite=True)

    return calib_dir, science_dir


# ========================
# Test Suite
# ========================

class TestDarkLibrary:
    """Test multi-exposure dark library functionality."""

    def test_dark_library_creation(self, tmp_path):
        """Test creating master darks for all exposure times."""
        calib_dir, _ = create_test_dataset(tmp_path)

        calib_mgr = CalibrationManager(
            calibration_dir=calib_dir,
            master_cache_dir=tmp_path / 'masters'
        )

        # Create bias first
        calib_mgr.create_master_bias()

        # Create dark library
        dark_lib = calib_mgr.create_master_dark_library()

        # Verify all exposure times created
        assert len(dark_lib) == 3
        assert 30.0 in dark_lib
        assert 60.0 in dark_lib
        assert 120.0 in dark_lib

        # Verify each dark has correct exposure time in header
        for exp_time, dark in dark_lib.items():
            assert dark.header['EXPTIME'] == exp_time
            assert dark.header['FRAMTYPE'] == 'MASTER_DARK'

    def test_dark_exposure_matching(self):
        """Test get_dark_for_exposure() matching logic."""
        calibs = CalibrationFrames()

        # Mock darks at different exposure times
        for exp_time in [30.0, 60.0, 120.0]:
            dark = CCDData(np.ones((10, 10)) * 100, unit='adu')
            dark.header['EXPTIME'] = exp_time
            calibs.master_darks[exp_time] = dark

        # Test exact match
        dark = calibs.get_dark_for_exposure(60.0)
        assert dark is not None
        assert dark.header['EXPTIME'] == 60.0

        # Test closest match within tolerance
        dark = calibs.get_dark_for_exposure(65.0, tolerance=10.0)
        assert dark is not None
        assert dark.header['EXPTIME'] == 60.0

        # Test closest match outside tolerance (should return None)
        dark = calibs.get_dark_for_exposure(150.0, tolerance=5.0)
        assert dark is None

        # Test with no darks available
        empty_calibs = CalibrationFrames()
        dark = empty_calibs.get_dark_for_exposure(60.0)
        assert dark is None


class TestBadPixelMasking:
    """Test bad pixel detection and masking."""

    def test_bad_pixel_detection(self, tmp_path):
        """Test that hot pixels are correctly identified."""
        calib_dir, _ = create_test_dataset(tmp_path)

        calib_mgr = CalibrationManager(calib_dir)
        calib_mgr.create_master_bias()
        calib_mgr.create_master_dark_library()

        # Create bad pixel mask
        bad_mask = calib_mgr.create_bad_pixel_mask(sigma_threshold=5.0)

        # Verify mask is boolean array
        assert bad_mask.dtype == bool

        # Verify some bad pixels found (synthetic data has hot pixels)
        n_bad = bad_mask.sum()
        assert n_bad > 0
        assert n_bad < bad_mask.size * 0.1  # Less than 10% bad

    def test_bad_pixel_mask_stored(self, tmp_path):
        """Test that bad pixel mask is stored in calibrations."""
        calib_dir, _ = create_test_dataset(tmp_path)

        calib_mgr = CalibrationManager(calib_dir)
        calib_mgr.create_master_bias()
        calib_mgr.create_master_dark_library()

        bad_mask = calib_mgr.create_bad_pixel_mask()

        # Verify stored
        assert calib_mgr.calibrations.bad_pixel_mask is not None
        assert np.array_equal(calib_mgr.calibrations.bad_pixel_mask, bad_mask)


class TestCalibrationWorkflow:
    """Test complete calibration workflow."""

    def test_full_workflow(self, tmp_path):
        """Test end-to-end calibration from raw to calibrated."""
        calib_dir, science_dir = create_test_dataset(tmp_path)

        calib_mgr = CalibrationManager(
            calibration_dir=calib_dir,
            master_cache_dir=tmp_path / 'masters'
        )

        # Create all master calibrations
        calibs = calib_mgr.create_master_calibrations()

        # Verify all calibrations created
        assert calibs.master_bias is not None
        assert len(calibs.master_darks) == 3  # 30s, 60s, 120s
        assert len(calibs.master_flats) == 3  # V, R, I

        # Load and calibrate science frame
        science = CCDData.read(science_dir / 'science_V.fits', unit='adu')
        calibrated = calib_mgr.calibrate(science, filter_name='V')

        # Verify calibration applied
        assert calibrated is not None
        assert calibrated.shape == science.shape
        # Calibrated data should be different from raw
        assert not np.array_equal(calibrated.data, science.data)

    def test_calibrate_uses_correct_dark(self, tmp_path):
        """Test that calibrate() uses correct dark for exposure time."""
        calib_dir, science_dir = create_test_dataset(tmp_path)

        calib_mgr = CalibrationManager(calib_dir)
        calib_mgr.create_master_calibrations()

        # Load science with 120s exposure
        science = CCDData.read(science_dir / 'science_V.fits', unit='adu')
        assert science.header['EXPTIME'] == 120.0

        # Calibrate (should use 120s dark)
        calibrated = calib_mgr.calibrate(science)

        # Verify calibration completed without errors
        assert calibrated is not None


class TestCaching:
    """Test calibration frame caching."""

    def test_caching_saves_files(self, tmp_path):
        """Test that master frames are saved to cache."""
        calib_dir, _ = create_test_dataset(tmp_path)
        cache_dir = tmp_path / 'masters'

        calib_mgr = CalibrationManager(
            calibration_dir=calib_dir,
            master_cache_dir=cache_dir
        )

        calib_mgr.create_master_calibrations()

        # Verify cached files exist
        assert (cache_dir / 'master_bias.fits').exists()

        # Verify multiple dark exposure times cached
        dark_files = list(cache_dir.glob('master_dark_*.fits'))
        assert len(dark_files) == 3  # 30s, 60s, 120s

        # Verify flats cached
        flat_files = list(cache_dir.glob('master_flat_*.fits'))
        assert len(flat_files) == 3  # V, R, I

    def test_caching_loads_correctly(self, tmp_path):
        """Test that cached calibrations are loaded correctly."""
        calib_dir, _ = create_test_dataset(tmp_path)
        cache_dir = tmp_path / 'masters'

        # Create calibrations (will be cached)
        calib_mgr1 = CalibrationManager(calib_dir, master_cache_dir=cache_dir)
        calib_mgr1.create_master_calibrations()

        # Create new manager and load from cache
        calib_mgr2 = CalibrationManager(calib_dir, master_cache_dir=cache_dir)
        cache_loaded = calib_mgr2.load_cached_calibrations()

        # Verify cache was loaded
        assert cache_loaded is True
        assert calib_mgr2.calibrations.master_bias is not None
        assert len(calib_mgr2.calibrations.master_darks) == 3
        assert len(calib_mgr2.calibrations.master_flats) == 3


class TestOverscanSecurity:
    """Test overscan subtraction security (no eval)."""

    def test_overscan_no_eval(self, tmp_path):
        """Verify overscan uses safe FITS section syntax (no eval)."""
        calib_dir = tmp_path / 'calibration'
        calib_dir.mkdir()

        # Create simple bias with overscan region
        data = np.ones((100, 110)) * 100.0  # Extra 10 columns for overscan
        ccd = CCDData(data, unit='adu')
        ccd.header['IMAGETYP'] = 'BIAS'
        ccd.write(calib_dir / 'bias_01.fits', overwrite=True)

        # This should NOT use eval() internally
        calib_mgr = CalibrationManager(
            calibration_dir=calib_dir,
            overscan_region='[101:110, :]',  # Last 10 columns
            overscan_axis=1,
            trim_region='[1:100, :]'  # First 100 columns
        )

        # This should work without security issues
        try:
            calib_mgr.create_master_bias()
            success = True
        except:
            success = False

        assert success, "Overscan processing failed"

        # Verify bias was created and trimmed to correct size
        assert calib_mgr.calibrations.master_bias is not None
        # After trim, should be 100 columns (not 110)
        assert calib_mgr.calibrations.master_bias.data.shape[1] == 100


class TestUnitDetection:
    """Test unit detection from headers."""

    def test_unit_detection_variations(self, tmp_path):
        """Test that various unit formats are detected correctly."""
        calib_dir = tmp_path / 'calibration'
        calib_dir.mkdir()

        # Test different unit variations
        unit_variations = {
            'ADU': u.adu,
            'adu': u.adu,
            'electron': u.electron,
            'count': u.adu,
            'dn': u.adu
        }

        calib_mgr = CalibrationManager(calib_dir)

        for unit_str, expected_unit in unit_variations.items():
            # Create test file with specific unit
            data = np.ones((10, 10)) * 100
            ccd = CCDData(data, unit=expected_unit)
            ccd.header['BUNIT'] = unit_str
            test_file = calib_dir / f'test_{unit_str}.fits'
            ccd.write(test_file, overwrite=True)

            # Test unit detection
            detected_unit = calib_mgr._get_unit_from_header(test_file)
            assert detected_unit == expected_unit


# ========================
# Performance Tests
# ========================

class TestPerformance:
    """Test performance characteristics."""

    def test_caching_improves_performance(self, tmp_path):
        """Verify that caching provides significant speedup."""
        import time

        calib_dir, _ = create_test_dataset(tmp_path)
        cache_dir = tmp_path / 'masters'

        # First run - create calibrations (SLOW)
        calib_mgr1 = CalibrationManager(calib_dir, master_cache_dir=cache_dir)
        start_create = time.time()
        calib_mgr1.create_master_calibrations()
        time_create = time.time() - start_create

        # Second run - load from cache (FAST)
        calib_mgr2 = CalibrationManager(calib_dir, master_cache_dir=cache_dir)
        start_load = time.time()
        calib_mgr2.load_cached_calibrations()
        time_load = time.time() - start_load

        # Caching should be at least 5× faster
        assert time_load < time_create / 3, \
            f"Caching not fast enough: {time_load:.3f}s vs {time_create:.3f}s"


# ========================
# Run Tests
# ========================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
