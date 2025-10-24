"""Check FITS file metadata and header for source information."""

from astropy.io import fits
from pathlib import Path

fits_file = r"C:\Users\jacks\experiments\PycharmProjects\Mission Control\examples\output\m31_panstarrs_rgb.fits"

print("="*80)
print("M31 FITS FILE - METADATA & SOURCE INFORMATION")
print("="*80)

print(f"\nFile: {fits_file}")
print(f"File size: {Path(fits_file).stat().st_size / (1024*1024):.2f} MB")

# Open and examine header
with fits.open(fits_file) as hdul:
    hdu = hdul[0]

    print("\n" + "="*80)
    print("COMPLETE FITS HEADER")
    print("="*80)

    # Print all header cards
    for card in hdu.header.cards:
        print(f"{card.keyword:8s} = {card.value}")
        if card.comment:
            print(f"         / {card.comment}")

    print("\n" + "="*80)
    print("KEY METADATA FIELDS")
    print("="*80)

    # Look for common source/origin keywords
    source_keywords = [
        'ORIGIN', 'TELESCOP', 'INSTRUME', 'OBJECT', 'OBSERVER',
        'DATE-OBS', 'DATE', 'AUTHOR', 'CREATOR', 'COMMENT',
        'HISTORY', 'FILTER', 'SURVEY', 'PROGID', 'PROPOSID'
    ]

    found_any = False
    for keyword in source_keywords:
        if keyword in hdu.header:
            value = hdu.header[keyword]
            print(f"\n{keyword}: {value}")
            found_any = True

    if not found_any:
        print("\nNo standard source metadata found in header.")

    print("\n" + "="*80)
    print("DATA INFORMATION")
    print("="*80)

    print(f"\nShape: {hdu.data.shape}")
    print(f"Type: {hdu.data.dtype}")
    print(f"Min: {hdu.data.min()}")
    print(f"Max: {hdu.data.max()}")
    print(f"Mean: {hdu.data.mean():.2f}")

print("\n" + "="*80)
print("CHECKING PARENT DIRECTORY FOR CLUES")
print("="*80)

parent_dir = Path(fits_file).parent
print(f"\nDirectory: {parent_dir}")
print(f"\nFiles in same directory:")

for file in sorted(parent_dir.iterdir()):
    if file.is_file():
        size = file.stat().st_size / 1024
        if size < 1024:
            print(f"  {file.name:50s} {size:>8.1f} KB")
        else:
            print(f"  {file.name:50s} {size/1024:>8.1f} MB")

# Check if there's a README or info file
for info_file in ['README.md', 'README.txt', 'info.txt', 'source.txt']:
    info_path = parent_dir / info_file
    if info_path.exists():
        print(f"\n" + "="*80)
        print(f"FOUND: {info_file}")
        print("="*80)
        print(info_path.read_text())

print("\n" + "="*80)
print("Based on filename: 'm31_panstarrs_rgb.fits'")
print("="*80)
print("""
Analysis:
- m31 = Messier 31 (Andromeda Galaxy)
- panstarrs = Pan-STARRS survey (Panoramic Survey Telescope)
- rgb = RGB composite (already color-combined)

Likely Source: Pan-STARRS1 3π Survey
  - Website: https://panstarrs.stsci.edu/
  - Coverage: 3/4 of the sky
  - Filters: g, r, i, z, y (optical/near-IR)

This appears to be a pre-made RGB composite from Pan-STARRS data,
possibly downloaded from an archive or created from Pan-STARRS cutouts.
""")

print("\n" + "="*80)