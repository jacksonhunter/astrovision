"""Simple standalone test of WCS Handler APE 14 compliance.

Run with: python test_wcs_ape14_simple.py
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.modeling import models
from astro_vision_composer.processing.wcs_handler import WCSHandler

# Check for gwcs
try:
    import gwcs
    from gwcs import coordinate_frames as cf
    GWCS_AVAILABLE = True
    print("[OK] gwcs available")
except ImportError:
    GWCS_AVAILABLE = False
    print("[SKIP] gwcs NOT available (optional)")


def create_simple_fits_wcs():
    """Create a simple FITS WCS object."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1024, 1024]
    wcs.wcs.crval = [30.0, 45.0]  # RA, Dec in degrees
    wcs.wcs.cdelt = [0.1 / 3600, 0.1 / 3600]  # 0.1 arcsec/pixel
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']

    # Test that WCS actually works
    print(f"\nDEBUG: Testing WCS creation:")
    print(f"  WCS type: {type(wcs)}")
    print(f"  Has pixel_to_world_values: {hasattr(wcs, 'pixel_to_world_values')}")

    # Try calling it
    try:
        result = wcs.pixel_to_world_values(1024, 1024)
        print(f"  pixel_to_world_values(1024, 1024) = {result}")
        print(f"  Result type: {type(result)}")
    except Exception as e:
        print(f"  ERROR calling pixel_to_world_values: {e}")

    return wcs


def create_simple_gwcs():
    """Create a simple gwcs.WCS object."""
    if not GWCS_AVAILABLE:
        return None

    # Create simple detector→sky transform
    crpix = (1024, 1024)
    shift_by_crpix = models.Shift(-crpix[0]) & models.Shift(-crpix[1])

    # Pixel scale: 0.1 arcsec/pixel
    pixelscale = models.Scale(0.1 / 3600) & models.Scale(0.1 / 3600)

    # TAN projection
    tan = models.Pix2Sky_TAN()

    # Rotation to sky coordinates
    celestial_rotation = models.RotateNative2Celestial(30.0, 45.0, 180.0)

    # Compose full transform
    det2sky = shift_by_crpix | pixelscale | tan | celestial_rotation
    det2sky.name = "detector_to_sky"

    # Define coordinate frames
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=SkyCoord(0, 0, unit='deg',
                                                            frame='icrs').frame,
                                  name='icrs',
                                  unit=(u.deg, u.deg))

    # Create gwcs WCS
    wcs_gwcs = gwcs.wcs.WCS([(detector_frame, det2sky), (sky_frame, None)])

    return wcs_gwcs


def test_fits_wcs():
    """Test validation with standard FITS WCS."""
    print("\n" + "="*70)
    print("TEST 1: Standard FITS WCS (astropy.wcs.WCS)")
    print("="*70)

    handler = WCSHandler()
    wcs = create_simple_fits_wcs()

    # Test APE 14 methods exist
    print("\n1. Testing APE 14 interface:")
    methods = ['pixel_to_world', 'world_to_pixel', 'pixel_to_world_values',
               'world_to_pixel_values', 'world_axis_names', 'world_axis_units']
    for method in methods:
        has_method = hasattr(wcs, method)
        print(f"   - {method}: {'✓' if has_method else '✗'}")
        assert has_method, f"Missing {method}"

    # Test coordinate transformation
    print("\n2. Testing coordinate transformation:")
    sky = wcs.pixel_to_world(1024, 1024)
    print(f"   pixel_to_world(1024, 1024) = {sky}")
    assert isinstance(sky, SkyCoord)
    assert np.abs(sky.ra.deg - 30.0) < 0.01
    assert np.abs(sky.dec.deg - 45.0) < 0.01
    print("   ✓ Coordinate transformation works")

    # Test round-trip
    print("\n3. Testing round-trip:")
    x, y = wcs.world_to_pixel(sky)
    print(f"   world_to_pixel(sky) = ({x:.2f}, {y:.2f})")
    np.testing.assert_allclose([x, y], [1024, 1024], atol=0.01)
    print("   ✓ Round-trip successful")

    # Test validation
    print("\n4. Testing WCSHandler.validate():")
    info = handler.validate(wcs)
    print(f"   is_valid: {info.is_valid}")
    print(f"   has_celestial: {info.has_celestial}")
    print(f"   has_gwcs: {info.has_gwcs}")
    print(f"   projection: {info.projection}")
    pixel_scale_str = f"{info.pixel_scale:.3f}" if info.pixel_scale else "None"
    print(f"   pixel_scale: {pixel_scale_str} arcsec/pixel")
    print(f"   warnings: {info.warnings}")

    assert info.is_valid
    assert info.has_celestial
    assert not info.has_gwcs
    assert info.projection == 'TAN'
    assert np.abs(info.pixel_scale - 0.1) < 0.01
    print("   ✓ Validation successful")

    print("\n" + "="*70)
    print("FITS WCS TEST: ✓ PASSED")
    print("="*70)


def test_gwcs():
    """Test validation with gwcs."""
    if not GWCS_AVAILABLE:
        print("\n" + "="*70)
        print("TEST 2: gwcs.WCS - SKIPPED (gwcs not installed)")
        print("="*70)
        return

    print("\n" + "="*70)
    print("TEST 2: gwcs.WCS (JWST, Roman, etc.)")
    print("="*70)

    handler = WCSHandler()
    wcs = create_simple_gwcs()

    # Test APE 14 methods exist
    print("\n1. Testing APE 14 interface:")
    methods = ['pixel_to_world', 'world_to_pixel', 'pixel_to_world_values',
               'world_to_pixel_values', 'world_axis_names', 'world_axis_units']
    for method in methods:
        has_method = hasattr(wcs, method)
        print(f"   - {method}: {'✓' if has_method else '✗'}")
        assert has_method, f"Missing {method}"

    # Test gwcs-specific features
    print("\n2. Testing gwcs-specific features:")
    print(f"   available_frames: {wcs.available_frames}")
    assert hasattr(wcs, 'available_frames')
    assert 'detector' in wcs.available_frames
    assert 'icrs' in wcs.available_frames
    print("   ✓ Intermediate frames accessible")

    # Test coordinate transformation
    print("\n3. Testing coordinate transformation:")
    sky = wcs.pixel_to_world(1024, 1024)
    print(f"   pixel_to_world(1024, 1024) = {sky}")
    assert isinstance(sky, SkyCoord)
    assert np.abs(sky.ra.deg - 30.0) < 0.1
    assert np.abs(sky.dec.deg - 45.0) < 0.1
    print("   ✓ Coordinate transformation works")

    # Test round-trip
    print("\n4. Testing round-trip:")
    x, y = wcs.world_to_pixel(sky)
    print(f"   world_to_pixel(sky) = ({x:.2f}, {y:.2f})")
    np.testing.assert_allclose([x, y], [1024, 1024], atol=0.1)
    print("   ✓ Round-trip successful")

    # Test validation
    print("\n5. Testing WCSHandler.validate():")
    info = handler.validate(wcs)
    print(f"   is_valid: {info.is_valid}")
    print(f"   has_celestial: {info.has_celestial}")
    print(f"   has_gwcs: {info.has_gwcs}")
    print(f"   projection: {info.projection}")
    pixel_scale_str = f"{info.pixel_scale:.3f}" if info.pixel_scale else "None"
    print(f"   pixel_scale: {pixel_scale_str} arcsec/pixel")
    print(f"   available_frames: {info.available_frames}")
    print(f"   world_axis_names: {info.world_axis_names}")
    print(f"   warnings: {info.warnings}")

    assert info.is_valid
    assert info.has_celestial
    assert info.has_gwcs  # This IS a gwcs object
    assert info.projection == 'gwcs'
    assert np.abs(info.pixel_scale - 0.1) < 0.02
    assert info.available_frames is not None
    print("   ✓ Validation successful")

    print("\n" + "="*70)
    print("gwcs TEST: ✓ PASSED")
    print("="*70)


def test_comparison():
    """Test comparing WCS objects."""
    print("\n" + "="*70)
    print("TEST 3: Comparing WCS Objects (APE 14 Interoperability)")
    print("="*70)

    handler = WCSHandler()
    fits_wcs = create_simple_fits_wcs()

    # Create second FITS WCS with compatible properties
    fits_wcs2 = WCS(naxis=2)
    fits_wcs2.wcs.crpix = [2048, 2048]  # Different ref pixel
    fits_wcs2.wcs.crval = [30.5, 45.5]  # Different coords
    fits_wcs2.wcs.cdelt = [0.1 / 3600, 0.1 / 3600]  # Same scale
    fits_wcs2.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # Same projection
    fits_wcs2.wcs.cunit = ['deg', 'deg']

    print("\n1. Comparing two compatible FITS WCS:")
    result = handler.compare_wcs(fits_wcs, fits_wcs2)
    print(f"   compatible: {result['compatible']}")
    print(f"   projection_match: {result['projection_match']}")
    print(f"   pixel_scale_diff: {result['pixel_scale_diff']:.2f}%")

    assert result['compatible']
    assert result['projection_match']
    print("   ✓ Comparison works")

    if GWCS_AVAILABLE:
        print("\n2. Comparing FITS WCS with gwcs:")
        gwcs_wcs = create_simple_gwcs()
        result = handler.compare_wcs(fits_wcs, gwcs_wcs)
        print(f"   compatible: {result['compatible']}")
        print(f"   pixel_scale_diff: {result.get('pixel_scale_diff', 'N/A')}")
        print("   ✓ Mixed-type comparison works (APE 14 interoperability!)")

    print("\n" + "="*70)
    print("COMPARISON TEST: ✓ PASSED")
    print("="*70)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("WCS HANDLER - APE 14 COMPLIANCE TEST SUITE")
    print("="*70)

    try:
        test_fits_wcs()
        test_gwcs()
        test_comparison()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nWCSHandler successfully implements APE 14 common interface!")
        print("Works with:")
        print("  ✓ astropy.wcs.WCS (standard FITS WCS)")
        if GWCS_AVAILABLE:
            print("  ✓ gwcs.WCS (JWST, Roman, etc.)")
        else:
            print("  ○ gwcs.WCS (not tested - gwcs not installed)")

        return 0

    except Exception as e:
        print("\n" + "="*70)
        print(f"TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
