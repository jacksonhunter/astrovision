"""Create composite from multiple single-band FITS files.

Usage:
    python test_multiband_composite.py band1.fits band2.fits band3.fits [output.png]
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import (
    FITSImageProcessor,
    CompositeImageGenerator,
    VisionGuidedCompositor
)

if len(sys.argv) < 4:
    print("Usage: python test_multiband_composite.py <band1.fits> <band2.fits> <band3.fits> [output.png]")
    print("\nExample:")
    print('  python test_multiband_composite.py g.fits r.fits i.fits composite.png')
    sys.exit(1)

# Parse arguments
fits_files = sys.argv[1:-1] if sys.argv[-1].endswith('.png') else sys.argv[1:]
output_file = sys.argv[-1] if sys.argv[-1].endswith('.png') else "multiband_composite.png"

print("="*80)
print("MULTI-BAND FITS COMPOSITE GENERATOR")
print("="*80)

print(f"\nInput bands: {len(fits_files)}")
for f in fits_files:
    print(f"  - {Path(f).name}")

print(f"\nOutput: {output_file}")

# Step 1: Load all bands from separate FITS files
print("\n" + "="*80)
print("Step 1: Loading bands from FITS files")
print("="*80)

all_bands = {}
all_metadata = {}

for fits_file in fits_files:
    print(f"\nLoading: {Path(fits_file).name}")

    try:
        with FITSImageProcessor(fits_file) as fits_proc:
            bands = fits_proc.extract_bands()
            metadata = fits_proc.get_band_info()

            # Get band name from each HDU
            for band_name, band_data in bands.items():
                # Create unique name: filename + HDU name
                file_stem = Path(fits_file).stem
                # Extract filter from filename (e.g., .stk.g. -> g)
                parts = file_stem.split('.')
                if 'stk' in parts:
                    idx = parts.index('stk')
                    if idx + 1 < len(parts):
                        filter_name = parts[idx + 1]
                        unique_name = f"{filter_name}_band"
                    else:
                        unique_name = f"{file_stem}_{band_name}"
                else:
                    unique_name = f"{file_stem}_{band_name}"

                print(f"  Found: {unique_name} - shape {band_data.shape}")

                # Normalize
                normalized = fits_proc.normalize_band(band_name, method='zscale')
                all_bands[unique_name] = normalized
                all_metadata[unique_name] = metadata[band_name]
                all_metadata[unique_name]['filter'] = filter_name if 'stk' in parts else 'unknown'

    except Exception as e:
        print(f"  ERROR loading {fits_file}: {e}")
        continue

print(f"\nTotal bands loaded: {len(all_bands)}")
for name in all_bands.keys():
    print(f"  - {name} (filter: {all_metadata[name].get('filter', 'unknown')})")

if len(all_bands) < 3:
    print(f"\nERROR: Need at least 3 bands for RGB composite, got {len(all_bands)}")
    sys.exit(1)

# Step 2: Determine band mapping
print("\n" + "="*80)
print("Step 2: Determining optimal band mapping")
print("="*80)

# Simple mapping based on filter names
band_names = list(all_bands.keys())

# Try to map by filter wavelength
filter_to_channel = {
    'g': 'blue',   # g-band (green) → blue channel
    'r': 'red',    # r-band → red channel
    'i': 'green',  # i-band (infrared) → green channel
    'z': 'red',    # z-band → red (alternative)
    'y': 'red',    # y-band → red (alternative)
}

rgb_mapping = {'red': None, 'green': None, 'blue': None}

# First pass: assign by filter name
for band_name in band_names:
    filter_name = all_metadata[band_name].get('filter', 'unknown')
    if filter_name in filter_to_channel:
        channel = filter_to_channel[filter_name]
        if rgb_mapping[channel] is None:
            rgb_mapping[channel] = band_name

# Fill in any missing channels
remaining = [b for b in band_names if b not in rgb_mapping.values()]
for channel in ['red', 'green', 'blue']:
    if rgb_mapping[channel] is None and remaining:
        rgb_mapping[channel] = remaining.pop(0)

print(f"\nAutomatic RGB mapping:")
print(f"  Red channel:   {rgb_mapping['red']}")
print(f"  Green channel: {rgb_mapping['green']}")
print(f"  Blue channel:  {rgb_mapping['blue']}")

# Step 3: Create composite
print("\n" + "="*80)
print("Step 3: Creating enhanced composite")
print("="*80)

generator = CompositeImageGenerator()

# Add all bands
for name, data in all_bands.items():
    generator.add_band(name, data)

print("\nApplying enhancements:")
print("  - Adaptive contrast (CLAHE)")
print("  - Detail enhancement")
print("  - Star highlighting")
print("  - Color balancing")

composite = generator.create_rgb_composite(
    r_band=rgb_mapping['red'],
    g_band=rgb_mapping['green'],
    b_band=rgb_mapping['blue'],
    enhance_contrast=True,
    enhance_details=True,
    enhance_stars=True,
    color_balance=True
)

# Step 4: Save
print("\n" + "="*80)
print("Step 4: Saving composite")
print("="*80)

generator.save_composite(composite, output_file, quality=95)

output_path = Path(output_file).resolve()
print(f"\n✓ Composite saved!")
print(f"  Location: {output_path}")
print(f"  Size: {composite.shape[1]}x{composite.shape[0]} pixels")
print(f"  File exists: {output_path.exists()}")

print("\n" + "="*80)
print("COMPOSITE GENERATION COMPLETE")
print("="*80)