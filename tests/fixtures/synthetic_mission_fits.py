"""Synthetic mission-specific FITS file generators for testing.

Creates minimal FITS files with correct mission-specific keywords and data
quality (DQ) arrays for testing mission adapters.

Author: VisionProject Testing Framework
Date: 2025-10-26
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from typing import Optional, Dict, Any


def create_jwst_fits(
    output_path: Path,
    instrument: str = 'NIRCAM',
    filter_name: str = 'F200W',
    detector: str = 'NRCA1',
    shape: tuple = (100, 100),
    add_dq: bool = True,
    seed: int = 42
) -> Path:
    """Create synthetic JWST FITS file with proper keywords and DQ array.

    Parameters
    ----------
    output_path : Path
        Output FITS file path
    instrument : str
        JWST instrument (NIRCAM, MIRI, NIRSPEC, etc.)
    filter_name : str
        Filter name (F090W, F200W, F444W, etc.)
    detector : str
        Detector name
    shape : tuple
        Image dimensions (height, width)
    add_dq : bool
        Add DQ (data quality) extension with JWST flags
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Path
        Path to created FITS file

    Notes
    -----
    JWST DQ flags (bit masks):
    - 0x1 (1): DO_NOT_USE (bad pixel)
    - 0x2 (2): SATURATED
    - 0x4 (4): JUMP_DET (cosmic ray)
    - 0x8 (8): DROPOUT (non-linear)
    - 0x10 (16): OUTLIER
    """
    np.random.seed(seed)

    # Create science data (realistic signal)
    science_data = np.random.poisson(1000, shape).astype(np.float32)
    science_data += np.random.normal(0, 5, shape).astype(np.float32)

    # PRIMARY HDU with JWST keywords
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['TELESCOP'] = 'JWST'
    primary_hdu.header['INSTRUME'] = instrument
    primary_hdu.header['DETECTOR'] = detector
    primary_hdu.header['FILTER'] = filter_name
    primary_hdu.header['EXPTIME'] = 1000.0
    primary_hdu.header['DATE-OBS'] = '2023-05-15'
    primary_hdu.header['ORIGIN'] = 'STScI'

    # SCI extension
    sci_hdu = fits.ImageHDU(science_data, name='SCI')
    sci_hdu.header['EXTNAME'] = 'SCI'
    sci_hdu.header['EXTVER'] = 1
    sci_hdu.header['BUNIT'] = 'MJy/sr'

    hdu_list = [primary_hdu, sci_hdu]

    # DQ extension (JWST-specific bit flags)
    if add_dq:
        dq_data = np.zeros(shape, dtype=np.uint32)

        # Add some realistic DQ flags
        # Dead pixels (DO_NOT_USE = 0x1)
        dead_pixels = np.random.random(shape) < 0.01
        dq_data[dead_pixels] |= 0x1

        # Saturated pixels (SATURATED = 0x2)
        saturated = science_data > 50000
        dq_data[saturated] |= 0x2

        # Cosmic rays (JUMP_DET = 0x4)
        cosmic_rays = np.random.random(shape) < 0.005
        dq_data[cosmic_rays] |= 0x4

        dq_hdu = fits.ImageHDU(dq_data, name='DQ')
        dq_hdu.header['EXTNAME'] = 'DQ'
        dq_hdu.header['EXTVER'] = 1
        hdu_list.append(dq_hdu)

    # Write FITS file
    hdulist = fits.HDUList(hdu_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdulist.writeto(output_path, overwrite=True)
    hdulist.close()

    return output_path


def create_hst_fits(
    output_path: Path,
    instrument: str = 'ACS',
    detector: str = 'WFC',
    filter_name: str = 'F606W',
    shape: tuple = (100, 100),
    add_dq: bool = True,
    seed: int = 42
) -> Path:
    """Create synthetic HST FITS file with proper keywords and DQ array.

    Parameters
    ----------
    output_path : Path
        Output FITS file path
    instrument : str
        HST instrument (ACS, WFC3, WFPC2, etc.)
    detector : str
        Detector name (WFC, UVIS, IR, etc.)
    filter_name : str
        Filter name (F435W, F606W, F814W, etc.)
    shape : tuple
        Image dimensions
    add_dq : bool
        Add DQ extension with HST flags
    seed : int
        Random seed

    Returns
    -------
    Path
        Path to created FITS file

    Notes
    -----
    HST DQ flags (bit masks):
    - 0x1 (1): Reed-Solomon decoding error
    - 0x2 (2): Data replaced by fill value
    - 0x4 (4): Bad detector pixel
    - 0x8 (8): Saturated pixel
    - 0x10 (16): Bad pixel in dark
    - 0x20 (32): Bad pixel in flat
    - 0x40 (64): Hot pixel
    - 0x80 (128): Warm pixel
    - 0x100 (256): Bad reference file
    """
    np.random.seed(seed)

    # Create science data
    science_data = np.random.poisson(2000, shape).astype(np.float32)
    science_data += np.random.normal(0, 10, shape).astype(np.float32)

    # PRIMARY HDU with HST keywords
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['TELESCOP'] = 'HST'
    primary_hdu.header['INSTRUME'] = instrument
    primary_hdu.header['DETECTOR'] = detector
    primary_hdu.header['FILTER'] = filter_name
    primary_hdu.header['EXPTIME'] = 600.0
    primary_hdu.header['DATE-OBS'] = '2022-08-10'
    primary_hdu.header['ORIGIN'] = 'STScI'

    # SCI extension
    sci_hdu = fits.ImageHDU(science_data, name='SCI')
    sci_hdu.header['EXTNAME'] = 'SCI'
    sci_hdu.header['EXTVER'] = 1
    sci_hdu.header['BUNIT'] = 'ELECTRONS/S'

    hdu_list = [primary_hdu, sci_hdu]

    # DQ extension (HST-specific flags)
    if add_dq:
        dq_data = np.zeros(shape, dtype=np.int16)

        # Bad detector pixels (0x4)
        bad_pixels = np.random.random(shape) < 0.02
        dq_data[bad_pixels] |= 0x4

        # Saturated pixels (0x8)
        saturated = science_data > 40000
        dq_data[saturated] |= 0x8

        # Hot pixels (0x40)
        hot_pixels = np.random.random(shape) < 0.01
        dq_data[hot_pixels] |= 0x40

        # Warm pixels (0x80)
        warm_pixels = np.random.random(shape) < 0.015
        dq_data[warm_pixels] |= 0x80

        dq_hdu = fits.ImageHDU(dq_data, name='DQ')
        dq_hdu.header['EXTNAME'] = 'DQ'
        dq_hdu.header['EXTVER'] = 1
        hdu_list.append(dq_hdu)

    # Write FITS file
    hdulist = fits.HDUList(hdu_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdulist.writeto(output_path, overwrite=True)
    hdulist.close()

    return output_path


def create_euclid_fits(
    output_path: Path,
    instrument: str = 'VIS',
    detector: str = 'CCD_1_1',
    filter_name: str = 'VIS',
    shape: tuple = (100, 100),
    seed: int = 42
) -> Path:
    """Create synthetic Euclid FITS file with proper keywords.

    Parameters
    ----------
    output_path : Path
        Output FITS file path
    instrument : str
        Euclid instrument (VIS or NISP)
    detector : str
        Detector quadrant (CCD_1_1, CCD_1_2, etc.)
    filter_name : str
        Filter name
    shape : tuple
        Image dimensions
    seed : int
        Random seed

    Returns
    -------
    Path
        Path to created FITS file
    """
    np.random.seed(seed)

    # Create science data
    science_data = np.random.poisson(1500, shape).astype(np.float32)
    science_data += np.random.normal(0, 8, shape).astype(np.float32)

    # PRIMARY HDU with Euclid keywords
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['TELESCOP'] = 'EUCLID'
    primary_hdu.header['INSTRUME'] = instrument
    primary_hdu.header['DETECTOR'] = detector
    primary_hdu.header['FILTER'] = filter_name
    primary_hdu.header['EXPTIME'] = 565.0
    primary_hdu.header['DATE-OBS'] = '2024-03-20'

    # SCI extension
    sci_hdu = fits.ImageHDU(science_data, name='SCI')
    sci_hdu.header['EXTNAME'] = 'SCI'
    sci_hdu.header['EXTVER'] = 1
    sci_hdu.header['BUNIT'] = 'electrons'

    # Write FITS file
    hdulist = fits.HDUList([primary_hdu, sci_hdu])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdulist.writeto(output_path, overwrite=True)
    hdulist.close()

    return output_path


def create_chandra_fits(
    output_path: Path,
    instrument: str = 'ACIS',
    detector: str = 'ACIS-I',
    shape: tuple = (100, 100),
    seed: int = 42
) -> Path:
    """Create synthetic Chandra FITS file (binned image, not event list).

    Parameters
    ----------
    output_path : Path
        Output FITS file path
    instrument : str
        Chandra instrument (ACIS or HRC)
    detector : str
        Detector name (ACIS-I, ACIS-S, HRC-I, HRC-S)
    shape : tuple
        Image dimensions
    seed : int
        Random seed

    Returns
    -------
    Path
        Path to created FITS file

    Notes
    -----
    This creates a binned image (not an event list). Event list testing
    is deferred to Phase 5 advanced tests.
    """
    np.random.seed(seed)

    # X-ray data (low counts, Poisson)
    science_data = np.random.poisson(10, shape).astype(np.float32)

    # PRIMARY HDU with Chandra keywords
    primary_hdu = fits.PrimaryHDU(science_data)
    primary_hdu.header['TELESCOP'] = 'CHANDRA'
    primary_hdu.header['INSTRUME'] = instrument
    primary_hdu.header['DETECTOR'] = detector
    primary_hdu.header['EXPTIME'] = 50000.0
    primary_hdu.header['DATE-OBS'] = '2023-11-05'
    primary_hdu.header['BUNIT'] = 'count'

    # Write FITS file
    hdulist = fits.HDUList([primary_hdu])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdulist.writeto(output_path, overwrite=True)
    hdulist.close()

    return output_path


def create_generic_fits(
    output_path: Path,
    telescope: str = 'Unknown',
    instrument: str = 'Camera',
    filter_name: str = 'R',
    shape: tuple = (100, 100),
    seed: int = 42
) -> Path:
    """Create generic FITS file without mission-specific keywords.

    Parameters
    ----------
    output_path : Path
        Output FITS file path
    telescope : str
        Telescope name
    instrument : str
        Instrument name
    filter_name : str
        Filter name
    shape : tuple
        Image dimensions
    seed : int
        Random seed

    Returns
    -------
    Path
        Path to created FITS file
    """
    np.random.seed(seed)

    # Create science data
    science_data = np.random.poisson(5000, shape).astype(np.float32)
    science_data += np.random.normal(0, 20, shape).astype(np.float32)

    # PRIMARY HDU with minimal keywords
    primary_hdu = fits.PrimaryHDU(science_data)
    primary_hdu.header['TELESCOP'] = telescope
    primary_hdu.header['INSTRUME'] = instrument
    primary_hdu.header['FILTER'] = filter_name
    primary_hdu.header['EXPTIME'] = 300.0
    primary_hdu.header['DATE-OBS'] = '2024-01-15'
    primary_hdu.header['BUNIT'] = 'ADU'

    # Write FITS file
    hdulist = fits.HDUList([primary_hdu])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdulist.writeto(output_path, overwrite=True)
    hdulist.close()

    return output_path


def create_multi_detector_fits(
    output_path: Path,
    telescope: str = 'EUCLID',
    n_detectors: int = 4,
    shape: tuple = (100, 100),
    seed: int = 42
) -> Path:
    """Create multi-detector FITS file (like Euclid multi-CCD mosaic).

    Parameters
    ----------
    output_path : Path
        Output FITS file path
    telescope : str
        Telescope name
    n_detectors : int
        Number of detector quadrants/CCDs
    shape : tuple
        Image dimensions per detector
    seed : int
        Random seed

    Returns
    -------
    Path
        Path to created FITS file
    """
    np.random.seed(seed)

    # PRIMARY HDU
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['TELESCOP'] = telescope
    primary_hdu.header['INSTRUME'] = 'VIS'
    primary_hdu.header['DATE-OBS'] = '2024-03-20'

    hdu_list = [primary_hdu]

    # Create multiple SCI extensions (one per detector)
    for i in range(n_detectors):
        science_data = np.random.poisson(1500 + i * 100, shape).astype(np.float32)
        science_data += np.random.normal(0, 8, shape).astype(np.float32)

        sci_hdu = fits.ImageHDU(science_data, name='SCI')
        sci_hdu.header['EXTNAME'] = 'SCI'
        sci_hdu.header['EXTVER'] = i + 1
        sci_hdu.header['DETECTOR'] = f'CCD_{i+1}'
        sci_hdu.header['BUNIT'] = 'electrons'

        hdu_list.append(sci_hdu)

    # Write FITS file
    hdulist = fits.HDUList(hdu_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdulist.writeto(output_path, overwrite=True)
    hdulist.close()

    return output_path


# DQ Flag Reference Dictionaries (for test validation)

JWST_DQ_FLAGS = {
    'DO_NOT_USE': 0x1,
    'SATURATED': 0x2,
    'JUMP_DET': 0x4,
    'DROPOUT': 0x8,
    'OUTLIER': 0x10,
    'PERSISTENCE': 0x20,
    'AD_FLOOR': 0x40,
    'RESERVED': 0x80,
    'UNRELIABLE_ERROR': 0x100,
    'NON_SCIENCE': 0x200,
    'DEAD': 0x400,
    'HOT': 0x800,
    'WARM': 0x1000,
    'LOW_QE': 0x2000,
    'RC': 0x4000,
    'TELEGRAPH': 0x8000,
    'NONLINEAR': 0x10000,
    'BAD_REF_PIXEL': 0x20000,
    'NO_FLAT_FIELD': 0x40000,
    'NO_GAIN_VALUE': 0x80000,
    'NO_LIN_CORR': 0x100000,
    'NO_SAT_CHECK': 0x200000,
    'UNRELIABLE_BIAS': 0x400000,
    'UNRELIABLE_DARK': 0x800000,
    'UNRELIABLE_SLOPE': 0x1000000,
    'UNRELIABLE_FLAT': 0x2000000
}

HST_DQ_FLAGS = {
    'RS_ERROR': 0x1,
    'DATA_REPLACED': 0x2,
    'BAD_DETECTOR_PIXEL': 0x4,
    'SATURATED': 0x8,
    'BAD_DARK': 0x10,
    'BAD_FLAT': 0x20,
    'HOT_PIXEL': 0x40,
    'WARM_PIXEL': 0x80,
    'BAD_REFERENCE': 0x100,
    'CHARGE_TRAP': 0x200,
    'A_TO_D_SATURATION': 0x400,
    'COSMIC_RAY': 0x1000,
    'UNSTABLE': 0x4000
}
