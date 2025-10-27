"""Fetch JWST Data Using MissionControl for WCS Testing.

This script uses the MissionControl/UniversalLocation framework to download
real JWST calibrated data from MAST for testing the VisionProject WCS handler.

JWST data is perfect for testing because:
- Has gwcs in ASDF extension (tests Phase 3A+3B features)
- Multiple filters available (NIRCam: F090W, F200W, F444W for RGB)
- Well-calibrated with full distortion correction
- High-quality WCS for validation

Requirements:
- MissionControl project must be in parent directory
- MAST API access (automatic via astroquery)
"""

import sys
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord

# Add MissionControl to path
mission_control_path = Path(__file__).parent.parent.parent / "Mission Control"
if not mission_control_path.exists():
    print(f"ERROR: MissionControl not found at {mission_control_path}")
    print("Please ensure Mission Control project is at:")
    print("  C:\\Users\\jacks\\experiments\\PycharmProjects\\Mission Control")
    sys.exit(1)

# Add both tests directory (for src access) and direct src path
sys.path.insert(0, str(mission_control_path / "tests"))
sys.path.insert(0, str(mission_control_path))

# Import from MissionControl (now with correct path)
try:
    from src.services.imagery import MASTProvider
    from src.models.universal_location import UniversalLocation
    from src.utils.exceptions import NoDataError
except ImportError as e:
    print(f"ERROR: Could not import from MissionControl: {e}")
    print(f"Mission Control path: {mission_control_path}")
    print(f"Please ensure Mission Control src/ directory exists")
    sys.exit(1)

# Also add VisionProject to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astro_vision_composer.processing import WCSHandler


def fetch_jwst_target(
    target_name: str,
    ra: float,
    dec: float,
    fov_arcmin: float = 2.0,
    instrument: str = 'NIRCAM',
    output_dir: Path = None
):
    """Fetch JWST data for a specific target.

    Args:
        target_name: Human-readable target name (e.g., "SMACS 0723")
        ra: Right ascension in degrees
        dec: Declination in degrees
        fov_arcmin: Field of view in arcminutes (default: 2.0)
        instrument: JWST instrument (NIRCAM, NIRISS, MIRI, etc.)
        output_dir: Where to save downloaded files

    Returns:
        Tuple of (image, metadata) or (None, None) if no data
    """
    if output_dir is None:
        output_dir = Path("data/jwst")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Fetching JWST Data: {target_name}")
    print(f"{'='*70}")
    print(f"Coordinates: RA={ra:.4f}°, Dec={dec:.4f}°")
    print(f"Field of View: {fov_arcmin}' × {fov_arcmin}'")
    print(f"Instrument: {instrument}")
    print(f"Output: {output_dir.absolute()}")
    print()

    # Create UniversalLocation using from_astropy_frame() classmethod
    # UniversalLocation is a coordinate transformer, not a named object container
    # CRITICAL: Must use GCRS (not ICRS) because ICRS doesn't support obstime
    # GCRS = Geocentric Celestial Reference System (time-dependent, suitable for ITRS transforms)
    from astropy.time import Time
    from astropy.coordinates import GCRS

    gcrs_frame = GCRS(
        ra=ra*u.deg,
        dec=dec*u.deg,
        distance=1*u.kpc,
        obstime=Time.now()  # Required for terrestrial frame transformations
    )
    loc = UniversalLocation.from_astropy_frame(gcrs_frame)

    # Initialize MAST provider
    provider = MASTProvider()
    print(f"MAST Provider initialized")
    print(f"  Cache: {provider.cache_manager.cache_root}")
    print()

    # Fetch JWST image
    try:
        print(f"Querying MAST for {instrument} observations...")
        image, metadata = provider.get_jwst_image(
            loc,
            fov=(fov_arcmin*u.arcmin, fov_arcmin*u.arcmin),
            instrument=instrument,
            use_cache=True  # Use cache to avoid re-downloading
        )

        if image is not None:
            # Save image preview
            safe_name = target_name.lower().replace(' ', '_')
            image_path = output_dir / f"{safe_name}_{instrument.lower()}_preview.png"
            image.save(image_path)

            print(f"\n[OK] SUCCESS!")
            print(f"  Image saved: {image_path.name}")
            print(f"  Size: {image.size[0]}×{image.size[1]} pixels")
            print(f"  Source: {metadata.source}")
            print(f"  Instrument: {metadata.properties.get('instrument', 'N/A')}")
            print(f"  Filters: {metadata.properties.get('filters', 'N/A')}")
            print(f"  Exposure: {metadata.properties.get('exposure_time', 0):.1f}s")
            print(f"  Obs ID: {metadata.properties.get('obs_id', 'N/A')}")

            # Check for FITS file in cache
            obs_id = metadata.properties.get('obs_id')
            if obs_id:
                # FITS files are cached by MASTProvider
                fits_cache = provider.mast_staging_dir / instrument.lower()
                fits_files = list(fits_cache.glob(f"*{obs_id}*.fits"))

                if fits_files:
                    print(f"\n  FITS file(s) cached:")
                    for fits_file in fits_files[:3]:  # Show first 3
                        print(f"    {fits_file.name}")
                        # Check for gwcs
                        try:
                            from astropy.io import fits
                            with fits.open(fits_file) as hdul:
                                has_asdf = 'ASDF' in [hdu.name for hdu in hdul]
                                if has_asdf:
                                    print(f"      [OK] Has ASDF extension (gwcs present!)")
                        except Exception as e:
                            print(f"      Could not check ASDF: {e}")

            return image, metadata

        else:
            print(f"\n[ERROR] No image returned")
            return None, None

    except NoDataError as e:
        print(f"\n[ERROR] No data available: {e}")
        return None, None

    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_wcs_from_cached_fits():
    """Test WCS loading from cached JWST FITS files."""
    print(f"\n{'='*70}")
    print("Testing WCS Handler with Cached JWST Data")
    print(f"{'='*70}\n")

    # Initialize MAST provider to find cache location
    provider = MASTProvider()
    cache_dir = provider.mast_staging_dir

    print(f"Searching for cached JWST FITS files in:")
    print(f"  {cache_dir}\n")

    # Look for JWST files in MAST download structure
    jwst_cache = cache_dir / "mastDownload" / "JWST"

    # Find all FITS files (any calibration level)
    fits_files = []
    if jwst_cache.exists():
        # Search recursively for FITS files
        fits_files = list(jwst_cache.glob("**/*.fits"))

    if fits_files:
        print(f"Found {len(fits_files)} JWST FITS file(s)\n")

        # Test WCS from first file
        test_file = fits_files[0]
        print(f"Testing WCS from: {test_file.name}")
        print(f"Full path: {test_file}")

        try:
            handler = WCSHandler()

            # Load WCS (should auto-detect ASDF and use gwcs)
            wcs = handler.load_wcs(test_file)

            # Validate
            info = handler.validate(wcs)

            print(f"\n[OK] WCS loaded successfully!")
            print(f"  Type: {'gwcs' if info.has_gwcs else 'FITS WCS'}")
            print(f"  Valid: {info.is_valid}")
            print(f"  Projection: {info.projection}")
            print(f"  Pixel scale: {info.pixel_scale:.4f} arcsec/pixel" if info.pixel_scale else "  Pixel scale: N/A")

            if info.has_gwcs:
                print(f"\n  gwcs features:")
                print(f"    Available frames: {info.available_frames}")
                print(f"    Axis names: {info.world_axis_names}")

                # Test Phase 3B features
                print(f"\n  Testing Phase 3B features:")

                # Get available frames
                frames = handler.get_available_frames(wcs)
                print(f"    [OK] get_available_frames(): {frames}")

                # Inspect pipeline
                pipeline_info = handler.inspect_pipeline(wcs)
                print(f"    [OK] inspect_pipeline(): {pipeline_info['type']}")
                if pipeline_info['steps']:
                    print(f"      Pipeline has {len(pipeline_info['steps'])} steps")

                # Get transform (if multiple frames)
                if len(frames) >= 2:
                    try:
                        transform = handler.get_transform(wcs, frames[0], frames[1])
                        if transform:
                            print(f"    [OK] get_transform({frames[0]} -> {frames[1]}): Success")
                    except Exception as e:
                        print(f"    [ERROR] get_transform(): {e}")

                # Test bounding box
                bbox = handler.get_bounding_box(wcs)
                print(f"    [OK] get_bounding_box(): {bbox if bbox else 'Not set'}")

            print(f"\n{'='*70}")
            print("WCS Test PASSED! [OK]")
            print(f"{'='*70}\n")

            return True

        except Exception as e:
            print(f"\n[ERROR] WCS test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    else:
        print("No FITS files found in cache")
        print("Run fetch commands first to download JWST data")
        return False


def main():
    """Main demo - fetch JWST data and test WCS."""
    print("\n" + "="*70)
    print("JWST Data Fetching for WCS Testing (MissionControl)")
    print("="*70)
    print("\nThis script will:")
    print("  1. Download JWST NIRCam data from MAST")
    print("  2. Cache FITS files locally")
    print("  3. Test WCS handler with gwcs (Phase 3A+3B)")
    print()

    # Famous JWST targets
    targets = [
        {
            'name': 'SMACS 0723',
            'ra': 110.8375,
            'dec': -73.4540,
            'description': 'JWST First Deep Field'
        },
        {
            'name': 'Carina Nebula',
            'ra': 161.2667,
            'dec': -59.9167,
            'description': 'Cosmic Cliffs - JWST ERO'
        },
        {
            'name': 'Southern Ring Nebula',
            'ra': 151.0125,
            'dec': -40.4372,
            'description': 'NGC 3132 - JWST ERO'
        }
    ]

    print("Available targets:")
    for i, target in enumerate(targets, 1):
        print(f"  {i}. {target['name']} - {target['description']}")
    print()

    # Fetch first target as example
    target = targets[0]
    print(f"Fetching: {target['name']}")

    output_dir = Path("data/jwst")

    # Fetch JWST data
    image, metadata = fetch_jwst_target(
        target_name=target['name'],
        ra=target['ra'],
        dec=target['dec'],
        fov_arcmin=2.0,
        instrument='NIRCAM',
        output_dir=output_dir
    )

    if image is not None:
        print(f"\n[OK] Data fetched successfully!")
        print(f"\nNow testing WCS handler with cached FITS files...")

        # Test WCS
        success = test_wcs_from_cached_fits()

        if success:
            print("\n" + "="*70)
            print("COMPLETE! [OK]")
            print("="*70)
            print(f"\nJWST data downloaded and WCS tested successfully!")
            print(f"\nCached FITS files can be used for:")
            print(f"  - WCS handler testing (Phase 3A+3B)")
            print(f"  - RGB composite creation")
            print(f"  - Multi-band processing")
            print(f"\nTo fetch more targets, edit the targets list in this script")
            print(f"or call fetch_jwst_target() directly with custom coordinates.")
        else:
            print("\n[ERROR] WCS test failed - check cached files")

    else:
        print(f"\n[ERROR] Could not fetch JWST data")
        print(f"\nTroubleshooting:")
        print(f"  1. Check internet connection")
        print(f"  2. Verify MAST is accessible")
        print(f"  3. Try a different target (may have more observations)")
        print(f"  4. Increase FOV (fov_arcmin parameter)")


if __name__ == '__main__':
    main()

