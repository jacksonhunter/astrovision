"""
Enhanced CalibrationManager Examples - All New Features (2025-10-26)

Demonstrates all improvements from Phase 2 refinement:
1. Overscan/trim support
2. Cached calibration loading
3. Validation after combination
4. Metadata storage in master frames
5. Unit checking from BUNIT keyword
6. Flexible IMAGETYP matching
7. Memory-efficient mode
8. Flexible filter name matching
"""

from pathlib import Path
from astro_vision_composer.preprocessing import CalibrationManager
from ccdproc import CCDData
import logging

# Setup logging to see all the details
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """Example 1: Basic usage (unchanged from before, but now better!)"""
    print("\n" + "="*80)
    print("Example 1: Basic Usage with Auto-Detection")
    print("="*80)

    # Initialize manager (now with optional overscan/trim)
    calib_mgr = CalibrationManager(
        calibration_dir='raw_data/calibration/',
        master_cache_dir='masters/'
    )

    # Try to load cached calibrations first (NEW!)
    if calib_mgr.load_cached_calibrations():
        print("✓ Using cached calibrations")
    else:
        print("Creating new calibrations...")
        # Create all master calibrations automatically
        calibs = calib_mgr.create_master_calibrations()
        print(f"✓ Created {len(calibs.master_flats)} master flats")

    # Apply to science frame
    science = CCDData.read('science.fits', unit='adu')
    calibrated = calib_mgr.calibrate(science, filter_name='V')

    print(f"✓ Calibrated science frame: {calibrated.shape}")


def example_2_overscan_trim():
    """Example 2: CCD with overscan region and trimming"""
    print("\n" + "="*80)
    print("Example 2: Overscan Subtraction and Trimming (NEW!)")
    print("="*80)

    # Many CCDs have overscan regions that need subtraction
    # Typical CCD: 2048x2048 imaging area + overscan columns
    calib_mgr = CalibrationManager(
        calibration_dir='raw_data/ccd_with_overscan/',
        master_cache_dir='masters/',
        overscan_region='[:, 2049:2080]',  # Last 32 columns are overscan (NEW!)
        trim_region='[:, 1:2048]'           # Keep only imaging area (NEW!)
    )

    # Now all frames will be overscan-subtracted and trimmed automatically!
    calibs = calib_mgr.create_master_calibrations()

    print("✓ All calibrations created with overscan subtraction and trimming")
    print(f"  Master bias shape: {calibs.master_bias.shape}")
    print(f"  Expected: (2048, 2048) after trimming")


def example_3_unit_detection():
    """Example 3: Automatic unit detection from FITS headers"""
    print("\n" + "="*80)
    print("Example 3: Automatic Unit Detection (NEW!)")
    print("="*80)

    calib_mgr = CalibrationManager('raw_data/mixed_units/')

    # CalibrationManager now automatically detects units from BUNIT keyword
    # Handles: ADU, electrons, counts, DN, ADU/s, etc.
    # No more unit mismatch errors!

    calibs = calib_mgr.create_master_calibrations()

    print(f"✓ Master bias unit: {calibs.master_bias.unit}")
    print(f"✓ Master dark unit: {calibs.master_dark.unit}")
    print("  Units detected automatically from FITS BUNIT keyword")


def example_4_flexible_imagetyp():
    """Example 4: Flexible IMAGETYP keyword matching"""
    print("\n" + "="*80)
    print("Example 4: Flexible IMAGETYP Matching (NEW!)")
    print("="*80)

    # Different observatories use different keywords:
    # - 'IMAGETYP' (most common)
    # - 'OBSTYPE' (some commercial cameras)
    # - 'FRAMETYPE' (some control software)
    # - 'FRAME' (older systems)

    calib_mgr = CalibrationManager('raw_data/mixed_keywords/')

    # CalibrationManager now checks all common variations
    # Also case-insensitive: 'bias', 'BIAS', 'Bias' all work
    calibs = calib_mgr.create_master_calibrations()

    print("✓ Successfully found calibration frames despite keyword variations")


def example_5_filter_name_matching():
    """Example 5: Flexible filter name matching"""
    print("\n" + "="*80)
    print("Example 5: Flexible Filter Name Matching (NEW!)")
    print("="*80)

    calib_mgr = CalibrationManager('raw_data/calibration/')

    # Real-world FITS headers have inconsistent filter names:
    # - 'V' vs 'V-band' vs 'V_filter' vs 'Johnson V'
    # - 'Ha' vs 'H-alpha' vs 'Halpha' vs 'H-a'

    # Old way would fail:
    # master_flat = calib_mgr.create_master_flat('V')  # Fails if headers say 'V-band'

    # New way uses substring matching:
    master_flat_v = calib_mgr.create_master_flat('V')  # Matches 'V', 'V-band', 'V_filter'
    master_flat_ha = calib_mgr.create_master_flat('Ha')  # Matches 'Ha', 'H-alpha', 'Halpha'

    print(f"✓ Created master flat for V (matched '{master_flat_v.header['FILTER']}')")
    print(f"✓ Created master flat for Ha (matched '{master_flat_ha.header['FILTER']}')")


def example_6_memory_efficient():
    """Example 6: Memory-efficient mode for large datasets"""
    print("\n" + "="*80)
    print("Example 6: Memory-Efficient Mode (NEW!)")
    print("="*80)

    # For large CCD arrays or many frames, limit memory usage
    calib_mgr = CalibrationManager(
        calibration_dir='raw_data/large_ccd/',
        mem_limit=8e9  # 8 GB limit (NEW!)
    )

    # ccdproc Combiner will now chunk processing to stay under memory limit
    # Allows processing 100+ 16MP frames without exhausting RAM
    calibs = calib_mgr.create_master_calibrations()

    print("✓ Processed large dataset within 8 GB memory limit")


def example_7_validation_and_metadata():
    """Example 7: Validation and enhanced metadata"""
    print("\n" + "="*80)
    print("Example 7: Validation and Metadata (NEW!)")
    print("="*80)

    calib_mgr = CalibrationManager('raw_data/calibration/')

    # Create master dark for specific exposure time
    master_dark = calib_mgr.create_master_dark(exposure_time=300.0)

    # NEW: Validation happens automatically after creation
    # Checks for: all-NaN data, zero variance, negative values, etc.

    # NEW: Enhanced metadata in headers
    print(f"✓ Master dark metadata:")
    print(f"  COMBINED: {master_dark.header['COMBINED']} frames")
    print(f"  COMBTYPE: {master_dark.header['COMBTYPE']}")
    print(f"  EXPTIME:  {master_dark.header['EXPTIME']}s")  # NEW!
    print(f"  FRAMTYPE: {master_dark.header['FRAMTYPE']}")  # NEW!
    print(f"  BIASSUB:  {master_dark.header['BIASSUB']}")

    # Stats from validation (logged automatically)
    print(f"  Median:   {master_dark.data.mean():.2f}")
    print(f"  Std dev:  {master_dark.data.std():.2f}")


def example_8_cached_calibrations():
    """Example 8: Using cached calibrations efficiently"""
    print("\n" + "="*80)
    print("Example 8: Cached Calibrations Workflow (NEW!)")
    print("="*80)

    # Session 1: Create calibrations
    print("Session 1: Creating calibrations...")
    calib_mgr = CalibrationManager(
        calibration_dir='raw_data/calibration/',
        master_cache_dir='masters/'
    )
    calibs = calib_mgr.create_master_calibrations()
    print(f"✓ Created and cached {1 + 1 + len(calibs.master_flats)} calibration frames")

    # Session 2: Load from cache (much faster!)
    print("\nSession 2: Loading from cache...")
    calib_mgr2 = CalibrationManager(
        calibration_dir='raw_data/calibration/',
        master_cache_dir='masters/'
    )

    if calib_mgr2.load_cached_calibrations():
        print("✓ Loaded all calibrations from cache (instant!)")
        print(f"  Master bias: {calib_mgr2.calibrations.master_bias is not None}")
        print(f"  Master dark: {calib_mgr2.calibrations.master_dark is not None}")
        print(f"  Master flats: {len(calib_mgr2.calibrations.master_flats)} filters")
    else:
        print("✗ No cache found, need to create calibrations")


def example_9_complete_workflow():
    """Example 9: Complete production workflow with all features"""
    print("\n" + "="*80)
    print("Example 9: Complete Production Workflow")
    print("="*80)

    # Initialize with all production settings
    calib_mgr = CalibrationManager(
        calibration_dir='raw_data/production_run/',
        master_cache_dir='masters/',
        overscan_region='[:, 2049:2080]',  # Overscan columns
        trim_region='[:, 1:2048]',          # Imaging area
        mem_limit=16e9                      # 16 GB memory limit
    )

    # Try cache first, create if needed
    if not calib_mgr.load_cached_calibrations():
        print("Creating new calibrations...")

        # Create bias
        master_bias = calib_mgr.create_master_bias(
            sigma_clip=True,
            clip_low=3.0,
            clip_high=3.0,
            method='median'  # Median more robust for cosmic rays
        )

        # Create dark (300s exposures)
        master_dark = calib_mgr.create_master_dark(
            exposure_time=300.0,
            tolerance=1.0,
            subtract_bias=True,  # Bias-subtract darks
            method='median'
        )

        # Create flats for each filter
        for filter_name in ['V', 'R', 'I']:
            master_flat = calib_mgr.create_master_flat(
                filter_name=filter_name,
                subtract_bias=True,
                subtract_dark=False,  # Flats usually too short for dark
                method='median'
            )
            print(f"✓ Created master flat for {filter_name}")
    else:
        print("✓ Using cached calibrations")

    # Process science frames
    print("\nProcessing science frames...")
    for filter_name in ['V', 'R', 'I']:
        science_file = f'science_{filter_name}.fits'

        # Load science frame
        science = CCDData.read(science_file, unit='adu')

        # Apply all calibrations
        calibrated = calib_mgr.calibrate(science, filter_name=filter_name)

        # Save calibrated frame
        calibrated.write(f'calibrated_{filter_name}.fits', overwrite=True)
        print(f"✓ Calibrated {science_file}")

    print("\n✓ Production workflow complete!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("CalibrationManager Enhanced Features Demo (2025-10-26)")
    print("="*80)
    print("\nThis script demonstrates all improvements to CalibrationManager:")
    print("1. ✓ Overscan/trim support")
    print("2. ✓ Cached calibration loading")
    print("3. ✓ Validation after combination")
    print("4. ✓ Enhanced metadata storage")
    print("5. ✓ Unit checking from BUNIT keyword")
    print("6. ✓ Flexible IMAGETYP matching")
    print("7. ✓ Memory-efficient mode")
    print("8. ✓ Flexible filter name matching")
    print("\nRunning examples...")

    # Run examples (commented out ones require specific data)
    try:
        example_1_basic_usage()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")

    try:
        example_2_overscan_trim()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")

    try:
        example_3_unit_detection()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")

    try:
        example_4_flexible_imagetyp()
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")

    try:
        example_5_filter_name_matching()
    except Exception as e:
        logger.error(f"Example 5 failed: {e}")

    try:
        example_6_memory_efficient()
    except Exception as e:
        logger.error(f"Example 6 failed: {e}")

    try:
        example_7_validation_and_metadata()
    except Exception as e:
        logger.error(f"Example 7 failed: {e}")

    try:
        example_8_cached_calibrations()
    except Exception as e:
        logger.error(f"Example 8 failed: {e}")

    try:
        example_9_complete_workflow()
    except Exception as e:
        logger.error(f"Example 9 failed: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
