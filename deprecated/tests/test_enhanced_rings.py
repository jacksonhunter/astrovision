"""Enhanced astronomical composite using the SAME method as working scripts.

This uses zscale normalization (like test_multiband_composite.py) instead of
Background2D, which was too aggressive for this data.

Usage:
    python test_enhanced_rings.py <g.fits> <r.fits> <i.fits> <output.png>
"""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import (
    FITSImageProcessor,
    CompositeImageGenerator
)

if len(sys.argv) < 5:
    print("Usage: python test_enhanced_rings.py <g.fits> <r.fits> <i.fits> <output.png>")
    sys.exit(1)

fits_files = sys.argv[1:4]
output_file = sys.argv[4]

print("="*80)
print("ENHANCED RINGS COMPOSITE - Using ZScale Normalization")
print("="*80)
print("\nUsing the SAME proven method as your working composites")
print("(zscale normalization, NOT Background2D)")

# Step 1: Load and normalize bands
print("\n" + "="*80)
print("Step 1: Loading bands with ZScale normalization")
print("="*80)

all_bands = {}
all_metadata = {}

for fits_file in fits_files:
    print(f"\nProcessing: {Path(fits_file).name}")

    with FITSImageProcessor(fits_file) as fits_proc:
        bands = fits_proc.extract_bands()
        metadata = fits_proc.get_band_info()

        for band_name, band_data in bands.items():
            file_stem = Path(fits_file).stem
            parts = file_stem.split('.')
            if 'stk' in parts:
                idx = parts.index('stk')
                filter_name = parts[idx + 1]
                unique_name = f"{filter_name}_band"
            else:
                unique_name = f"{file_stem}_{band_name}"

            print(f"  Band: {unique_name}")
            print(f"    Raw range: [{np.nanmin(band_data):.3e}, {np.nanmax(band_data):.3e}]")
            print(f"    Raw median: {np.nanmedian(band_data):.3e}")

            # Use ZScale normalization (same as your working scripts!)
            normalized = fits_proc.normalize_band(band_name, method='zscale')

            print(f"    Normalized median: {np.nanmedian(normalized):.4f}")
            print(f"    Normalized mean: {np.nanmean(normalized):.4f}")

            all_bands[unique_name] = normalized
            all_metadata[unique_name] = metadata[band_name]

print(f"\nLoaded bands: {list(all_bands.keys())}")

# Step 2: Map to RGB channels
mapping = {
    'red': 'r_band',
    'green': 'i_band',
    'blue': 'g_band'
}

print("\n" + "="*80)
print("Step 2: Creating composite with enhancements")
print("="*80)

generator = CompositeImageGenerator()

for name, data in all_bands.items():
    generator.add_band(name, data)

# Use the built-in RGB composite method (same as your working scripts)
print("\nApplying enhancements:")
print("  - Adaptive contrast (CLAHE)")
print("  - Detail enhancement (unsharp masking)")
print("  - Star highlighting")
print("  - Color balancing")

enhanced = generator.create_rgb_composite(
    r_band=mapping['red'],
    g_band=mapping['green'],
    b_band=mapping['blue'],
    enhance_contrast=True,
    enhance_details=True,
    enhance_stars=True,
    color_balance=True
)

# Step 3: Save
print("\n" + "="*80)
print("Step 3: Saving composite")
print("="*80)

generator.save_composite(enhanced, output_file, quality=95)

output_path = Path(output_file).resolve()
print(f"\n✓ Enhanced composite saved!")
print(f"  Location: {output_path}")

print(f"\nFinal image statistics:")
print(f"  Median: {np.nanmedian(enhanced):.4f}")
print(f"  Mean: {np.nanmean(enhanced):.4f}")
print(f"  Bright pixels (>0.5): {100*np.sum(enhanced > 0.5)/enhanced.size:.1f}%")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nThis uses the SAME normalization method as your working composites.")
print("No Background2D subtraction - just proven zscale normalization.")