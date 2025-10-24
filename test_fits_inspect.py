"""Quick script to inspect any FITS file structure."""

import sys
from astropy.io import fits

if len(sys.argv) < 2:
    print("Usage: python test_m31_inspect.py <path_to_fits_file>")
    sys.exit(1)

fits_file = sys.argv[1]

print("="*80)
print(f"Inspecting: {fits_file}")
print("="*80)

# Open FITS file
with fits.open(fits_file) as hdul:
    # Print general info
    print("\nFITS Structure:")
    hdul.info()

    print("\n" + "="*80)
    print("Detailed Band Information:")
    print("="*80)

    # Inspect each HDU
    for i, hdu in enumerate(hdul):
        print(f"\nHDU {i}: {hdu.name}")
        print(f"  Type: {type(hdu).__name__}")

        if hdu.data is not None:
            print(f"  Shape: {hdu.data.shape}")
            print(f"  Data type: {hdu.data.dtype}")
            print(f"  Min value: {hdu.data.min():.2f}")
            print(f"  Max value: {hdu.data.max():.2f}")
            print(f"  Mean value: {hdu.data.mean():.2f}")

            # Check for relevant header keywords
            if 'FILTER' in hdu.header:
                print(f"  Filter: {hdu.header['FILTER']}")
            if 'WAVELENG' in hdu.header:
                print(f"  Wavelength: {hdu.header['WAVELENG']}")
        else:
            print("  No data")

        # Print first few header keywords
        print(f"  Header keywords: {len(hdu.header)} total")

print("\n" + "="*80)
print("Inspection complete!")
print("="*80)