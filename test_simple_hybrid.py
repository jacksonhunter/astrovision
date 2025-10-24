"""Simplified hybrid - skip the problematic LRGB step.

Just the essentials that work:
- zscale normalization (good signal for CLAHE)
- CLAHE contrast (what makes rings_composite.png look good)
- Star enhancement (bright stars)
- Luminance masking (dark backgrounds)

NO LRGB to avoid artifacts.

Usage:
    python test_simple_hybrid.py <g.fits> <r.fits> <i.fits> <output.png>
"""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import FITSImageProcessor, CompositeImageGenerator

if len(sys.argv) < 5:
    print("Usage: python test_simple_hybrid.py <g.fits> <r.fits> <i.fits> <output.png>")
    sys.exit(1)

fits_files = sys.argv[1:4]
output_file = sys.argv[4]

print("="*80)
print("SIMPLE HYBRID - NO LRGB ARTIFACTS")
print("="*80)
print("\nWhat works:")
print("  ✓ zscale normalization")
print("  ✓ CLAHE contrast (from rings_composite.png)")
print("  ✓ Unsharp masking")
print("  ✓ Star boost")
print("  ✓ Luminance masking (dark backgrounds)")
print("\nWhat we're skipping:")
print("  ✗ LRGB (causes artifacts)")

# Step 1: Load with zscale
print("\n" + "="*80)
print("Step 1: Load bands with zscale")
print("="*80)

all_bands = {}

for fits_file in fits_files:
    print(f"\nProcessing: {Path(fits_file).name}")

    with FITSImageProcessor(fits_file) as fits_proc:
        bands = fits_proc.extract_bands()

        for band_name, band_data in bands.items():
            file_stem = Path(fits_file).stem
            parts = file_stem.split('.')
            if 'stk' in parts:
                idx = parts.index('stk')
                filter_name = parts[idx + 1]
                unique_name = f"{filter_name}_band"
            else:
                unique_name = f"{file_stem}_{band_name}"

            normalized = fits_proc.normalize_band(band_name, method='zscale')
            print(f"  {unique_name}: median={np.nanmedian(normalized):.4f}")

            all_bands[unique_name] = normalized

# Map to RGB
mapping = {'red': 'r_band', 'green': 'i_band', 'blue': 'g_band'}

print("\n" + "="*80)
print("Step 2: Apply CLAHE + enhancements")
print("="*80)

generator = CompositeImageGenerator()
for name, data in all_bands.items():
    generator.add_band(name, data)

# Get channels
r = generator.bands[mapping['red']].copy()
g = generator.bands[mapping['green']].copy()
b = generator.bands[mapping['blue']].copy()

# CLAHE
print("  - CLAHE adaptive contrast")
from skimage import exposure

r_clahe = exposure.equalize_adapthist(r, clip_limit=0.03)
g_clahe = exposure.equalize_adapthist(g, clip_limit=0.03)
b_clahe = exposure.equalize_adapthist(b, clip_limit=0.03)

# Unsharp mask
print("  - Unsharp mask for detail")
from scipy import ndimage

r_detail = r_clahe + (r_clahe - ndimage.gaussian_filter(r_clahe, 2)) * 0.5
g_detail = g_clahe + (g_clahe - ndimage.gaussian_filter(g_clahe, 2)) * 0.5
b_detail = b_clahe + (b_clahe - ndimage.gaussian_filter(b_clahe, 2)) * 0.5

r_detail = np.clip(r_detail, 0, 1)
g_detail = np.clip(g_detail, 0, 1)
b_detail = np.clip(b_detail, 0, 1)

# Star boost
print("  - Star boost (99th percentile)")
r_thresh = np.percentile(r_detail, 99)
g_thresh = np.percentile(g_detail, 99)
b_thresh = np.percentile(b_detail, 99)

r_final = np.where(r_detail > r_thresh, np.clip(r_detail * 1.3, 0, 1), r_detail)
g_final = np.where(g_detail > g_thresh, np.clip(g_detail * 1.3, 0, 1), g_detail)
b_final = np.where(b_detail > b_thresh, np.clip(b_detail * 1.3, 0, 1), b_detail)

# Stack to RGB
rgb_enhanced = np.dstack([r_final, g_final, b_final])

print(f"  After enhancements median: {np.nanmedian(rgb_enhanced):.4f}")

# Step 3: Luminance masking
print("\n" + "="*80)
print("Step 3: Luminance masking (darkest 5% -> black)")
print("="*80)

# Calculate luminance
luminance = 0.2126 * rgb_enhanced[:, :, 0] + 0.7152 * rgb_enhanced[:, :, 1] + 0.0722 * rgb_enhanced[:, :, 2]

# Mask darkest 5%
lum_threshold = np.nanpercentile(luminance, 5)
print(f"  5th percentile threshold: {lum_threshold:.4f}")

dark_mask = luminance < lum_threshold
print(f"  Masking {np.sum(dark_mask)} pixels ({100*np.sum(dark_mask)/dark_mask.size:.1f}%)")

rgb_final = rgb_enhanced.copy()
rgb_final[dark_mask] = 0

print(f"  Final median: {np.nanmedian(rgb_final):.4f}")
print(f"  Final mean: {np.nanmean(rgb_final):.4f}")

# Save
print("\n" + "="*80)
print("Step 4: Save")
print("="*80)

generator.save_composite(rgb_final, output_file, quality=95)

print(f"\n✓ Saved: {Path(output_file).resolve()}")
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nThis should have:")
print("  ✓ Good CLAHE contrast (like rings_composite.png)")
print("  ✓ Dark backgrounds (5% masked to black)")
print("  ✓ Bright stars (99th percentile boost)")
print("  ✓ NO artifacts (no LRGB)")