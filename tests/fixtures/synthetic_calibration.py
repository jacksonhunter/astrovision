"""
Synthetic calibration frame generators for testing.

Generates realistic bias, dark, flat, and science frames with known characteristics
for validating CalibrationManager functionality.
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
import astropy.units as u
from ccdproc import CCDData


def create_synthetic_bias(
    shape=(100, 100),
    bias_level=100.0,
    readnoise=5.0,
    seed=None
) -> CCDData:
    """Create synthetic bias frame.

    Parameters
    ----------
    shape : tuple
        Image dimensions (ny, nx)
    bias_level : float
        Mean bias level in ADU
    readnoise : float
        Read noise standard deviation in electrons
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    CCDData
        Synthetic bias frame with read noise
    """
    if seed is not None:
        np.random.seed(seed)

    # Bias is just readnoise pattern
    bias = bias_level + np.random.normal(0, readnoise, shape)

    # Create CCDData with metadata
    header = fits.Header()
    header['IMAGETYP'] = 'BIAS'
    header['EXPTIME'] = 0.0
    header['BUNIT'] = 'adu'
    header['RDNOISE'] = readnoise
    header['GAIN'] = 1.0
    header['BIASLEVL'] = bias_level

    return CCDData(bias, unit='adu', header=header)


def create_synthetic_dark(
    shape=(100, 100),
    exposure_time=300.0,
    bias_level=100.0,
    dark_current=0.1,  # electrons/sec/pixel
    readnoise=5.0,
    hot_pixel_fraction=0.001,
    hot_pixel_factor=10.0,
    seed=None
) -> CCDData:
    """Create synthetic dark frame.

    Parameters
    ----------
    shape : tuple
        Image dimensions (ny, nx)
    exposure_time : float
        Exposure time in seconds
    bias_level : float
        Bias offset in ADU
    dark_current : float
        Dark current rate in electrons/sec/pixel
    readnoise : float
        Read noise in electrons
    hot_pixel_fraction : float
        Fraction of hot pixels (0-1)
    hot_pixel_factor : float
        Hot pixel dark current multiplier
    seed : int, optional
        Random seed

    Returns
    -------
    CCDData
        Synthetic dark frame with bias, dark current, and hot pixels
    """
    if seed is not None:
        np.random.seed(seed)

    # Start with bias
    dark = bias_level + np.random.normal(0, readnoise, shape)

    # Add dark current (Poisson-distributed)
    dark_signal = np.random.poisson(dark_current * exposure_time, shape)
    dark += dark_signal

    # Add hot pixels
    n_hot = int(hot_pixel_fraction * np.prod(shape))
    hot_y = np.random.randint(0, shape[0], n_hot)
    hot_x = np.random.randint(0, shape[1], n_hot)
    dark[hot_y, hot_x] += np.random.poisson(
        dark_current * hot_pixel_factor * exposure_time,
        n_hot
    )

    # Create CCDData
    header = fits.Header()
    header['IMAGETYP'] = 'DARK'
    header['EXPTIME'] = exposure_time
    header['BUNIT'] = 'adu'
    header['RDNOISE'] = readnoise
    header['GAIN'] = 1.0
    header['DARKCURR'] = dark_current
    header['BIASLEVL'] = bias_level

    return CCDData(dark, unit='adu', header=header)


def create_synthetic_flat(
    shape=(100, 100),
    filter_name='V',
    bias_level=100.0,
    signal_level=30000.0,
    vignetting_strength=0.3,
    dust_spots=5,
    readnoise=5.0,
    seed=None
) -> CCDData:
    """Create synthetic flat field frame.

    Parameters
    ----------
    shape : tuple
        Image dimensions (ny, nx)
    filter_name : str
        Filter name
    bias_level : float
        Bias offset in ADU
    signal_level : float
        Mean signal level in ADU (peak, before vignetting)
    vignetting_strength : float
        Vignetting amount (0-1), 0=none, 1=severe
    dust_spots : int
        Number of dust spots to add
    readnoise : float
        Read noise in electrons
    seed : int, optional
        Random seed

    Returns
    -------
    CCDData
        Synthetic flat field with vignetting and dust
    """
    if seed is not None:
        np.random.seed(seed)

    ny, nx = shape

    # Create vignetting pattern (radial gradient)
    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny / 2, nx / 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_max = np.sqrt(cy**2 + cx**2)

    # Vignetting: 1.0 at center, reduces to (1 - strength) at corners
    vignetting = 1.0 - vignetting_strength * (r / r_max)**2

    # Start with uniform illumination
    flat = signal_level * vignetting

    # Add dust spots (small dark circles)
    for _ in range(dust_spots):
        spot_y = np.random.randint(ny // 4, 3 * ny // 4)
        spot_x = np.random.randint(nx // 4, 3 * nx // 4)
        spot_radius = np.random.uniform(2, 8)
        spot_strength = np.random.uniform(0.1, 0.3)

        dist = np.sqrt((y - spot_y)**2 + (x - spot_x)**2)
        mask = dist < spot_radius
        flat[mask] *= (1.0 - spot_strength)

    # Add Poisson noise (photon noise)
    flat = np.random.poisson(flat)

    # Add bias and readnoise
    flat = flat + bias_level + np.random.normal(0, readnoise, shape)

    # Create CCDData
    header = fits.Header()
    header['IMAGETYP'] = 'FLAT'
    header['FILTER'] = filter_name
    header['EXPTIME'] = 1.0  # Flats typically short exposure
    header['BUNIT'] = 'adu'
    header['RDNOISE'] = readnoise
    header['GAIN'] = 1.0
    header['BIASLEVL'] = bias_level

    return CCDData(flat, unit='adu', header=header)


def create_synthetic_science(
    shape=(100, 100),
    filter_name='V',
    exposure_time=300.0,
    n_stars=20,
    star_flux_range=(500, 5000),
    background=200.0,
    bias_level=100.0,
    dark_current=0.1,
    readnoise=5.0,
    cosmic_rays=10,
    seed=None
) -> CCDData:
    """Create synthetic science frame with stars and noise.

    Parameters
    ----------
    shape : tuple
        Image dimensions (ny, nx)
    filter_name : str
        Filter name
    exposure_time : float
        Exposure time in seconds
    n_stars : int
        Number of point sources
    star_flux_range : tuple
        (min, max) star flux in electrons/sec
    background : float
        Sky background in electrons/sec/pixel
    bias_level : float
        Bias offset in ADU
    dark_current : float
        Dark current in electrons/sec/pixel
    readnoise : float
        Read noise in electrons
    cosmic_rays : int
        Number of cosmic ray hits
    seed : int, optional
        Random seed

    Returns
    -------
    CCDData
        Synthetic science frame
    """
    if seed is not None:
        np.random.seed(seed)

    ny, nx = shape

    # Start with sky background (Poisson)
    science = np.random.poisson(background * exposure_time, shape).astype(float)

    # Add stars (Gaussian PSF)
    for _ in range(n_stars):
        star_y = np.random.uniform(10, ny - 10)
        star_x = np.random.uniform(10, nx - 10)
        star_flux = np.random.uniform(*star_flux_range) * exposure_time
        star_fwhm = np.random.uniform(2.0, 4.0)

        # Create Gaussian PSF
        y, x = np.ogrid[:ny, :nx]
        sigma = star_fwhm / 2.355
        psf = np.exp(-((x - star_x)**2 + (y - star_y)**2) / (2 * sigma**2))
        psf /= psf.sum()

        science += star_flux * psf

    # Add cosmic rays (single bright pixels)
    for _ in range(cosmic_rays):
        cr_y = np.random.randint(0, ny)
        cr_x = np.random.randint(0, nx)
        cr_flux = np.random.uniform(5000, 50000)
        science[cr_y, cr_x] += cr_flux

    # Add dark current
    science += np.random.poisson(dark_current * exposure_time, shape)

    # Add bias and read noise
    science = science + bias_level + np.random.normal(0, readnoise, shape)

    # Create CCDData
    header = fits.Header()
    header['IMAGETYP'] = 'OBJECT'
    header['FILTER'] = filter_name
    header['EXPTIME'] = exposure_time
    header['BUNIT'] = 'adu'
    header['RDNOISE'] = readnoise
    header['GAIN'] = 1.0
    header['BIASLEVL'] = bias_level
    header['DARKCURR'] = dark_current
    header['OBJECT'] = 'Test Target'

    return CCDData(science, unit='adu', header=header)


def create_test_dataset(
    output_dir: Path,
    n_bias=5,
    n_darks_per_exp=5,
    dark_exposures=[30.0, 60.0, 300.0],
    n_flats_per_filter=5,
    filters=['V', 'R', 'Ha'],
    n_science_per_filter=1,
    shape=(100, 100)
):
    """Create complete test calibration dataset.

    Generates bias, dark, flat, and science frames matching typical
    observatory structure.

    Parameters
    ----------
    output_dir : Path
        Directory to write FITS files
    n_bias : int
        Number of bias frames
    n_darks_per_exp : int
        Number of darks per exposure time
    dark_exposures : list
        Dark exposure times in seconds
    n_flats_per_filter : int
        Number of flats per filter
    filters : list
        Filter names
    n_science_per_filter : int
        Number of science frames per filter
    shape : tuple
        Image dimensions

    Returns
    -------
    dict
        Paths to generated files organized by type
    """
    output_dir = Path(output_dir)
    calib_dir = output_dir / 'calibration'
    science_dir = output_dir / 'science'

    calib_dir.mkdir(parents=True, exist_ok=True)
    science_dir.mkdir(parents=True, exist_ok=True)

    files = {'bias': [], 'dark': {}, 'flat': {}, 'science': {}}

    # Create bias frames
    for i in range(n_bias):
        bias = create_synthetic_bias(shape=shape, seed=1000 + i)
        filepath = calib_dir / f'bias_{i+1:03d}.fits'
        bias.write(filepath, overwrite=True)
        files['bias'].append(filepath)

    # Create dark frames (multiple exposure times)
    for exp_time in dark_exposures:
        files['dark'][exp_time] = []
        for i in range(n_darks_per_exp):
            dark = create_synthetic_dark(
                shape=shape,
                exposure_time=exp_time,
                seed=2000 + int(exp_time) * 10 + i
            )
            filepath = calib_dir / f'dark_{int(exp_time)}s_{i+1:03d}.fits'
            dark.write(filepath, overwrite=True)
            files['dark'][exp_time].append(filepath)

    # Create flat frames (per filter)
    for filt in filters:
        files['flat'][filt] = []
        for i in range(n_flats_per_filter):
            flat = create_synthetic_flat(
                shape=shape,
                filter_name=filt,
                seed=3000 + hash(filt) % 1000 + i
            )
            filepath = calib_dir / f'flat_{filt}_{i+1:03d}.fits'
            flat.write(filepath, overwrite=True)
            files['flat'][filt].append(filepath)

    # Create science frames (per filter)
    for filt in filters:
        files['science'][filt] = []
        for i in range(n_science_per_filter):
            science = create_synthetic_science(
                shape=shape,
                filter_name=filt,
                exposure_time=300.0,
                seed=4000 + hash(filt) % 1000 + i
            )
            filepath = science_dir / f'science_{filt}_{i+1:03d}.fits'
            science.write(filepath, overwrite=True)
            files['science'][filt].append(filepath)

    return files


if __name__ == '__main__':
    """Quick test of synthetic frame generation."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Creating test dataset in {tmpdir}")
        files = create_test_dataset(Path(tmpdir))

        print(f"\nGenerated files:")
        print(f"  Bias frames: {len(files['bias'])}")
        print(f"  Dark frames: {sum(len(v) for v in files['dark'].values())} "
              f"({len(files['dark'])} exposure times)")
        print(f"  Flat frames: {sum(len(v) for v in files['flat'].values())} "
              f"({len(files['flat'])} filters)")
        print(f"  Science frames: {sum(len(v) for v in files['science'].values())} "
              f"({len(files['science'])} filters)")
