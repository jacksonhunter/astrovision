"""WCS Handler Multi-Mission Example - APE 14 Compliance.

This example demonstrates the WCSHandler's ability to work with multiple
mission data formats using the APE 14 common interface:
- Standard FITS WCS (ground-based, PanSTARRS, etc.)
- JWST gwcs (from ASDF extension)
- HST with full distortion (via drizzlepac)
- Euclid multi-detector mosaics

All WCS types are interchangeable thanks to APE 14 compliance!

GETTING DATA FROM MAST:
=======================

For JWST data:
--------------
1. Go to https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
2. Search for your target (e.g., "SMACS 0723")
3. Filter by Mission: JWST
4. Filter by Product Type: "SCIENCE" or "CALIBRATED"
5. Download files ending in:
   - *_cal.fits  (calibrated 2D images - HAS gwcs in ASDF extension!)
   - *_i2d.fits  (resampled/combined images - also has gwcs)

   IMPORTANT: Avoid *_uncal.fits (uncalibrated, no WCS yet)

6. Recommended: Get 3-filter set for RGB (e.g., F090W, F200W, F444W for NIRCam)

For HST data:
-------------
1. Same MAST portal
2. Filter by Mission: HST
3. Filter by Instrument: ACS or WFC3 (have best WCS support)
4. Download files ending in:
   - *_flc.fits (ACS: CTE-corrected, calibrated)
   - *_flt.fits (WFC3: calibrated)

   These have WCS in headers but may need drizzlepac for full distortion.

For ground-based data:
---------------------
1. PanSTARRS: Download from https://ps1images.stsci.edu/
   - Files have standard FITS WCS with TPV distortion
2. Your own observatory: Any FITS with WCS keywords will work
   - Needs at least: CRPIX, CRVAL, CDELT or CD matrix, CTYPE

Example file structure:
-----------------------
data/
├── jwst/
│   ├── jw02736001001_02101_00001_nrca1_cal.fits  # F090W
│   ├── jw02736001001_02101_00001_nrca2_cal.fits  # F200W
│   └── jw02736001001_02101_00001_nrca3_cal.fits  # F444W
├── hst/
│   ├── j8xi01a4q_flc.fits  # ACS F606W
│   └── j8xi01a6q_flc.fits  # ACS F814W
└── panstarrs/
    └── skycell.1234.056.stk.fits
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import the WCS handler
from astro_vision_composer.processing import WCSHandler


def example1_basic_usage():
    """Example 1: Basic WCS loading and validation (works for any mission)."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic WCS Loading and Validation")
    print("="*70)

    handler = WCSHandler()

    # Example with a FITS file (replace with your actual file path)
    fits_file = Path("data/my_image.fits")

    # Skip if file doesn't exist (for documentation purposes)
    if not fits_file.exists():
        print(f"Skipping: {fits_file} not found (example only)")
        print("\nWhat this would do:")
        print("1. Auto-detect mission (JWST, HST, etc.)")
        print("2. Auto-detect WCS type (ASDF/gwcs vs FITS WCS)")
        print("3. Load WCS with appropriate method")
        print("4. Validate and extract properties")
        return

    # Load WCS (auto-detects format)
    wcs = handler.load_wcs(fits_file)

    # Validate WCS (works for both FITS WCS and gwcs!)
    info = handler.validate(wcs)

    # Print results
    print(f"\nWCS Info:")
    print(f"  Valid: {info.is_valid}")
    print(f"  Mission: {info.mission or 'Unknown'}")
    print(f"  Projection: {info.projection}")
    print(f"  Pixel scale: {info.pixel_scale:.3f} arcsec/pixel" if info.pixel_scale else "  Pixel scale: Unknown")
    print(f"  Has gwcs: {info.has_gwcs}")

    if info.has_gwcs:
        print(f"  Available frames: {info.available_frames}")
        print(f"  Axis names: {info.world_axis_names}")

    if info.warnings:
        print(f"  Warnings: {info.warnings}")

    # Use WCS for coordinate transformation (APE 14 interface)
    sky = wcs.pixel_to_world(1024, 1024)
    print(f"\n  Center coordinates: {sky}")


def example2_jwst_specific():
    """Example 2: JWST-specific features (gwcs) - Phase 3B Features!"""
    print("\n" + "="*70)
    print("EXAMPLE 2: JWST Data with gwcs (Phase 3B Features)")
    print("="*70)

    handler = WCSHandler()

    # JWST calibrated file (has ASDF extension with gwcs)
    jwst_file = Path("data/jwst/jw02736001001_02101_00001_nrcb1_cal.fits")

    if not jwst_file.exists():
        print(f"Skipping: {jwst_file} not found")
        print("\nWhat this would do:")
        print("1. Detect ASDF extension → load gwcs.WCS")
        print("2. Access intermediate coordinate frames")
        print("3. Extract sub-transforms between frames")
        print("4. Inspect full transformation pipeline")
        return

    # Load WCS (will auto-detect ASDF and load gwcs)
    wcs = handler.load_wcs(jwst_file)

    # Validate
    info = handler.validate(wcs)

    print(f"\nJWST WCS detected: {info.has_gwcs}")

    if info.has_gwcs:
        # Phase 3B Feature 1: Get available frames
        frames = handler.get_available_frames(wcs)
        print(f"\nAvailable coordinate frames:")
        for frame in frames:
            print(f"  - {frame}")

        print(f"\nTransformation pipeline: {' → '.join(frames)}")

        # Phase 3B Feature 2: Inspect pipeline
        pipeline_info = handler.inspect_pipeline(wcs)
        print(f"\nPipeline details:")
        print(f"  Type: {pipeline_info['type']}")
        print(f"  Input frame: {pipeline_info['input_frame']['name']}")
        print(f"  Output frame: {pipeline_info['output_frame']['name']}")

        if pipeline_info['steps']:
            print(f"\n  Transformation steps:")
            for i, step in enumerate(pipeline_info['steps'], 1):
                print(f"    {i}. {step['frame']} ({step['frame_type']})")

        # Phase 3B Feature 3: Extract intermediate transforms
        if len(frames) >= 2:
            print(f"\nExtracting distortion correction:")
            try:
                distortion = handler.get_transform(wcs, frames[0], frames[1])
                if distortion:
                    print(f"  ✓ Got transform: {frames[0]} → {frames[1]}")
                    print(f"  Example: (100, 200) → {distortion(100, 200)}")
            except Exception as e:
                print(f"  Could not extract transform: {e}")

        # Phase 3B Feature 4: Bounding box (if needed for IFU/MOS)
        print(f"\nBounding box support:")
        bbox = handler.get_bounding_box(wcs)
        if bbox:
            print(f"  Current: {bbox}")
        else:
            print(f"  Not set (can set with handler.set_bounding_box())")
            print(f"  Example: handler.set_bounding_box(wcs, ((0, 2048), (0, 2048)))")

        # Phase 3B Feature 5: Save WCS for caching
        print(f"\nWCS caching:")
        print(f"  Can save to ASDF: handler.save_wcs(wcs, 'cached_wcs.asdf')")
        print(f"  Load later: wcs = handler.load_wcs('cached_wcs.asdf')")


def example3_hst_distortion():
    """Example 3: HST with full distortion correction."""
    print("\n" + "="*70)
    print("EXAMPLE 3: HST Data with Full Distortion")
    print("="*70)

    # Check if drizzlepac is available
    try:
        from drizzlepac import stwcs
        has_drizzlepac = True
    except ImportError:
        has_drizzlepac = False
        print("drizzlepac not installed - will use standard WCS")
        print("Install with: pip install drizzlepac")
        print("(Requires CRDS environment: CRDS_PATH and CRDS_SERVER_URL)\n")

    handler = WCSHandler()

    hst_file = Path("data/hst/j8xi01a4q_flc.fits")

    if not hst_file.exists():
        print(f"Skipping: {hst_file} not found")
        print("\nWhat this would do:")
        if has_drizzlepac:
            print("1. Detect HST mission → use drizzlepac.stwcs")
            print("2. Load WCS with IDCTAB, D2IMFILE, NPOLFILE distortion")
            print("3. Get high-precision astrometry")
        else:
            print("1. Detect HST mission")
            print("2. Fall back to standard WCS (SIP distortion only)")
            print("3. Warning: Missing full distortion correction")
        return

    # Load WCS
    wcs = handler.load_wcs(hst_file)
    info = handler.validate(wcs)

    print(f"\nHST WCS loaded:")
    print(f"  Method: {info.wcs_origin}")
    print(f"  Distortion model: {info.distortion_model or 'none'}")

    if has_drizzlepac:
        print(f"  Full distortion correction applied ✓")
    else:
        print(f"  Limited distortion (SIP only) - install drizzlepac for full correction")


def example4_comparison():
    """Example 4: Comparing WCS from different missions (APE 14 interoperability)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Comparing WCS (APE 14 Interoperability)")
    print("="*70)

    handler = WCSHandler()

    # Create two simple WCS for demonstration
    # (Replace with actual file loading in real usage)

    # WCS 1: FITS WCS
    wcs1 = WCS(naxis=2)
    wcs1.wcs.crpix = [1024, 1024]
    wcs1.wcs.crval = [150.0, 2.5]
    wcs1.wcs.cdelt = [0.04 / 3600, 0.04 / 3600]  # 0.04 arcsec/pixel (HST)
    wcs1.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs1.wcs.cunit = ['deg', 'deg']

    # WCS 2: Similar WCS but different reference point
    wcs2 = WCS(naxis=2)
    wcs2.wcs.crpix = [2048, 2048]
    wcs2.wcs.crval = [150.1, 2.6]
    wcs2.wcs.cdelt = [0.04 / 3600, 0.04 / 3600]  # Same pixel scale
    wcs2.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs2.wcs.cunit = ['deg', 'deg']

    # Compare WCS (works regardless of WCS type!)
    result = handler.compare_wcs(wcs1, wcs2)

    print(f"\nComparison results:")
    print(f"  Compatible: {result['compatible']}")
    print(f"  Projection match: {result['projection_match']}")
    print(f"  Pixel scale difference: {result['pixel_scale_diff']:.2f}%")

    if result['rotation_diff'] is not None:
        print(f"  Rotation difference: {result['rotation_diff']:.2f} degrees")

    print(f"\n  Details: {result['details']}")

    if result['compatible']:
        print("\n  ✓ These images can be combined without reprojection!")
    else:
        print("\n  ⚠ These images need reprojection to align")


def example5_batch_processing():
    """Example 5: Batch process multiple files."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Batch Processing Multiple Files")
    print("="*70)

    handler = WCSHandler()

    # List of files to process (mix of missions)
    files = [
        "data/jwst/jw02736_f090w_cal.fits",
        "data/jwst/jw02736_f200w_cal.fits",
        "data/hst/hst_acs_f606w_flc.fits",
        "data/panstarrs/ps1_g.fits",
    ]

    print("Processing multiple files:\n")

    for fits_file in files:
        fits_path = Path(fits_file)

        if not fits_path.exists():
            print(f"[SKIP] {fits_path.name} (not found)")
            continue

        try:
            # Load and validate
            wcs = handler.load_wcs(fits_path)
            info = handler.validate(wcs)

            # Print summary
            status = "✓" if info.is_valid else "✗"
            wcs_type = "gwcs" if info.has_gwcs else "FITS"
            scale = f"{info.pixel_scale:.3f}\"" if info.pixel_scale else "N/A"

            print(f"[{status}] {fits_path.name}")
            print(f"    Type: {wcs_type}, Scale: {scale}, Projection: {info.projection}")

        except Exception as e:
            print(f"[✗] {fits_path.name}")
            print(f"    Error: {e}")

    print("\nAll files processed!")


def example6_coordinate_transformation():
    """Example 6: Coordinate transformations (APE 14 common interface)."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Coordinate Transformations (APE 14)")
    print("="*70)

    # Create simple WCS for demonstration
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1024, 1024]
    wcs.wcs.crval = [83.6333, -5.3911]  # Orion Nebula
    wcs.wcs.cdelt = [0.1 / 3600, 0.1 / 3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']

    print("\nAPE 14 transformations (work for ALL WCS types):\n")

    # 1. High-level: pixel → SkyCoord
    print("1. pixel_to_world (high-level):")
    sky = wcs.pixel_to_world(1024, 1024)
    print(f"   pixel(1024, 1024) → {sky}")

    # 2. Low-level: pixel → values
    print("\n2. pixel_to_world_values (low-level):")
    ra, dec = wcs.pixel_to_world_values(1024, 1024)
    print(f"   pixel(1024, 1024) → RA={ra:.6f}°, Dec={dec:.6f}°")

    # 3. Inverse: world → pixel
    print("\n3. world_to_pixel (inverse):")
    target = SkyCoord(83.6333*u.deg, -5.3911*u.deg)
    x, y = wcs.world_to_pixel(target)
    print(f"   {target} → pixel({x:.2f}, {y:.2f})")

    # 4. Array transformations
    print("\n4. Array transformations:")
    x_array = np.array([100, 500, 1000, 1500])
    y_array = np.array([200, 600, 1000, 1400])
    sky_array = wcs.pixel_to_world(x_array, y_array)
    print(f"   {len(sky_array)} positions transformed at once")
    print(f"   First: {sky_array[0]}")
    print(f"   Last: {sky_array[-1]}")

    # 5. Metadata
    print("\n5. WCS metadata (APE 14):")
    print(f"   Axis names: {wcs.world_axis_names}")
    print(f"   Axis units: {wcs.world_axis_units}")
    if hasattr(wcs, 'world_axis_physical_types'):
        print(f"   Physical types: {wcs.world_axis_physical_types}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("WCS HANDLER - MULTI-MISSION EXAMPLES")
    print("APE 14 Compliant - Works with JWST, HST, ground-based, etc.")
    print("="*70)

    # Run examples
    example1_basic_usage()
    example2_jwst_specific()
    example3_hst_distortion()
    example4_comparison()
    example5_batch_processing()
    example6_coordinate_transformation()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. WCSHandler auto-detects WCS format (ASDF/gwcs vs FITS)")
    print("2. All WCS types use the same APE 14 interface")
    print("3. Works with JWST (gwcs), HST (drizzlepac), ground-based (FITS)")
    print("4. Comparison works across different WCS types")
    print("\nFor real data:")
    print("- Download JWST *_cal.fits from MAST (has gwcs in ASDF)")
    print("- Download HST *_flc.fits or *_flt.fits")
    print("- Any ground-based FITS with WCS keywords works")


if __name__ == '__main__':
    main()
