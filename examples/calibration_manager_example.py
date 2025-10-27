"""Example usage of CalibrationManager for raw CCD data processing.

This example shows how to use the CalibrationManager to automatically
detect, combine, and apply calibration frames (bias, dark, flat) to
raw CCD data.

Requirements:
    - ccdproc >= 2.4.0
    - astropy >= 5.3.0
    - Raw calibration frames organized by image type (BIAS, DARK, FLAT)
"""

from pathlib import Path
from astro_vision_composer.preprocessing import CalibrationManager
from astro_vision_composer.pipeline import ProcessingPipeline
import ccdproc
import astropy.units as u


# =============================================================================
# Example 1: Basic Usage - Create Master Calibration Frames
# =============================================================================

def example_1_basic_usage():
    """Create master calibration frames from raw files."""
    print("=" * 70)
    print("Example 1: Creating Master Calibration Frames")
    print("=" * 70)

    # Initialize calibration manager
    # Point to directory containing raw calibration files
    calib_dir = Path('raw_data/calibration/')
    calib_mgr = CalibrationManager(calib_dir)

    # Create master bias
    print("\n1. Creating master bias...")
    master_bias = calib_mgr.create_master_bias(
        sigma_clip=True,
        clip_low=3.0,
        clip_high=3.0,
        method='average'
    )
    print(f"   Master bias shape: {master_bias.shape}")
    print(f"   Combined {master_bias.header['COMBINED']} frames")

    # Create master dark
    print("\n2. Creating master dark...")
    master_dark = calib_mgr.create_master_dark(
        exposure_time=300.0,  # Match dark exposure to science frames
        tolerance=1.0,
        subtract_bias=True,
        method='average'
    )
    print(f"   Master dark shape: {master_dark.shape}")

    # Create master flat for each filter
    print("\n3. Creating master flats...")
    for filter_name in ['V', 'R', 'I']:
        try:
            master_flat = calib_mgr.create_master_flat(
                filter_name=filter_name,
                subtract_bias=True,
                subtract_dark=False
            )
            print(f"   Master flat ({filter_name}): {master_flat.shape}")
        except ValueError as e:
            print(f"   Warning: {e}")

    print("\n   Master calibration frames created successfully!")


# =============================================================================
# Example 2: Automatic Calibration - One Command
# =============================================================================

def example_2_automatic():
    """Automatically create all master calibrations."""
    print("\n" + "=" * 70)
    print("Example 2: Automatic Master Calibration Creation")
    print("=" * 70)

    calib_dir = Path('raw_data/calibration/')
    calib_mgr = CalibrationManager(calib_dir)

    # Automatically detect filters and create all calibrations
    calibrations = calib_mgr.create_master_calibrations(
        bias=True,
        dark=True,
        flats=None  # Auto-detect filters from FLAT frames
    )

    print("\nCreated calibrations:")
    print(f"  - Master bias: {'Yes' if calibrations.master_bias else 'No'}")
    print(f"  - Master dark: {'Yes' if calibrations.master_dark else 'No'}")
    print(f"  - Master flats: {list(calibrations.master_flats.keys())}")


# =============================================================================
# Example 3: Calibrate Individual Science Frames
# =============================================================================

def example_3_calibrate_science():
    """Apply calibrations to individual science frames."""
    print("\n" + "=" * 70)
    print("Example 3: Calibrating Science Frames")
    print("=" * 70)

    # Setup
    calib_dir = Path('raw_data/calibration/')
    calib_mgr = CalibrationManager(calib_dir)
    calib_mgr.create_master_calibrations()

    # Load raw science frame
    raw_science = ccdproc.CCDData.read('raw_data/science_V.fits', unit='adu')
    print(f"\nRaw science frame: {raw_science.shape}")

    # Apply all calibrations
    calibrated = calib_mgr.calibrate(raw_science, filter_name='V')
    print(f"Calibrated frame: {calibrated.shape}")

    # Save calibrated frame
    calibrated.write('output/science_V_calibrated.fits', overwrite=True)
    print("Calibrated frame saved!")


# =============================================================================
# Example 4: Integrated Pipeline - End-to-End
# =============================================================================

def example_4_integrated_pipeline():
    """Process raw CCD data through complete pipeline."""
    print("\n" + "=" * 70)
    print("Example 4: Integrated Pipeline with Auto-Calibration")
    print("=" * 70)

    # Initialize pipeline with calibration directory
    pipeline = ProcessingPipeline(
        mode='scientific',
        calibration_dir='raw_data/calibration/'
    )

    # Process raw science frames to RGB composite
    # Pipeline automatically:
    # 1. Creates master calibration frames
    # 2. Applies calibrations to raw science data
    # 3. Processes to RGB composite
    rgb = pipeline.process_to_rgb(
        fits_files=[
            'raw_data/science_R.fits',
            'raw_data/science_G.fits',
            'raw_data/science_B.fits'
        ],
        auto_calibrate=True,  # Enable automatic calibration
        output_dir='output/calibrated/'
    )

    print(f"\nRGB composite shape: {rgb.shape}")
    print(f"Value range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print("\nPipeline complete! Check output/calibrated/ for results.")


# =============================================================================
# Example 5: Caching Master Calibrations
# =============================================================================

def example_5_caching():
    """Cache master calibrations for reuse."""
    print("\n" + "=" * 70)
    print("Example 5: Caching Master Calibrations")
    print("=" * 70)

    # Create calibration manager with cache directory
    calib_mgr = CalibrationManager(
        calibration_dir='raw_data/calibration/',
        master_cache_dir='cached_masters/'
    )

    # Create masters (will be cached to disk)
    print("\nCreating and caching master calibrations...")
    calib_mgr.create_master_calibrations()

    print("\nCached files:")
    cache_dir = Path('cached_masters/')
    for cached_file in cache_dir.glob('*.fits'):
        print(f"  - {cached_file.name}")

    # Later: Load from cache instead of recombining
    master_bias = ccdproc.CCDData.read('cached_masters/master_bias.fits')
    print(f"\nLoaded master bias from cache: {master_bias.shape}")


# =============================================================================
# Example 6: Advanced - Custom Combination Parameters
# =============================================================================

def example_6_advanced():
    """Use custom combination parameters for master frames."""
    print("\n" + "=" * 70)
    print("Example 6: Advanced Custom Parameters")
    print("=" * 70)

    calib_mgr = CalibrationManager('raw_data/calibration/')

    # Create master bias with custom clipping
    master_bias = calib_mgr.create_master_bias(
        sigma_clip=True,
        clip_low=5.0,  # More aggressive low clipping
        clip_high=2.5, # Less aggressive high clipping
        method='median'  # Use median instead of average
    )
    print(f"Master bias (median combined): {master_bias.shape}")

    # Create master dark with bias subtraction
    master_dark = calib_mgr.create_master_dark(
        exposure_time=600.0,  # 10-minute darks
        tolerance=5.0,  # Allow ±5s tolerance
        subtract_bias=True,
        sigma_clip=True,
        method='median'
    )
    print(f"Master dark (600s, bias-subtracted): {master_dark.shape}")

    # Create master flat with full calibration
    master_flat = calib_mgr.create_master_flat(
        filter_name='Ha',  # Narrowband filter
        subtract_bias=True,
        subtract_dark=True,  # Also subtract dark current
        sigma_clip=True,
        clip_low=3.0,
        clip_high=3.0
    )
    print(f"Master flat (Ha, fully calibrated): {master_flat.shape}")
    print(f"Flat normalization value: {master_flat.header['FLATNORM']}")


# =============================================================================
# Example 7: File Organization Best Practices
# =============================================================================

def example_7_file_organization():
    """Show recommended file organization for calibration files."""
    print("\n" + "=" * 70)
    print("Example 7: Calibration File Organization")
    print("=" * 70)

    print("""
Recommended directory structure:

raw_data/
├── calibration/          # All calibration frames here
│   ├── bias_001.fits
│   ├── bias_002.fits
│   ├── bias_003.fits
│   ├── dark_300s_001.fits
│   ├── dark_300s_002.fits
│   ├── flat_V_001.fits
│   ├── flat_V_002.fits
│   ├── flat_R_001.fits
│   └── flat_R_002.fits
└── science/              # Science frames
    ├── target_V.fits
    ├── target_R.fits
    └── target_B.fits

Required FITS header keywords:
- IMAGETYP: 'BIAS', 'DARK', 'FLAT', or 'OBJECT'
- EXPTIME: Exposure time in seconds (for darks)
- FILTER: Filter name (for flats and science)

Example header:
    IMAGETYP= 'FLAT'
    FILTER  = 'V'
    EXPTIME = 5.0
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\nCalibrationManager Examples")
    print("=" * 70)
    print("\nThese examples show how to use CalibrationManager")
    print("to process raw CCD data with automatic calibration.")
    print("\nNote: Requires ccdproc package (pip install ccdproc)")
    print("=" * 70)

    # Uncomment examples to run:

    # example_1_basic_usage()
    # example_2_automatic()
    # example_3_calibrate_science()
    # example_4_integrated_pipeline()
    # example_5_caching()
    # example_6_advanced()
    example_7_file_organization()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
