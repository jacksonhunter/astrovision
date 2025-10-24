"""Test per-pixel luminance masking approach for dark backgrounds + bright stars.

Instead of background subtraction, we:
1. Calculate luminance from RGB
2. Mask pixels below 5th percentile to pure black
3. Apply LRGB with luminance layer for detail

Usage:
    python test_luminance_masking.py <g.fits> <r.fits> <i.fits> <output.png>
"""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import FITSImageProcessor, CompositeImageGenerator

if len(sys.argv) < 5:
    print("Usage: python test_luminance_masking.py <g.fits> <r.fits> <i.fits> <output.png>")
    sys.exit(1)

fits_files = sys.argv[1:4]
output_file = sys.argv[4]

print("="*80)
print("LUMINANCE MASKING APPROACH")
print("="*80)
print("\nCombining:")
print("  - Simple percentile normalization (no Background2D)")
print("  - Luminance-based 5th percentile masking for black backgrounds")
print("  - LRGB technique for sharp details")
print("  - Gamma stretch + star enhancement")

# Load bands with zscale (no background subtraction)
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

            # Use zscale (no background subtraction, just normalization)
            normalized = fits_proc.normalize_band(band_name, method='zscale')

            print(f"  {unique_name}: median={np.nanmedian(normalized):.4f}, mean={np.nanmean(normalized):.4f}")

            all_bands[unique_name] = normalized

# Map to RGB
mapping = {'red': 'r_band', 'green': 'i_band', 'blue': 'g_band'}

print("\n" + "="*80)
print("Step 2: Create initial RGB composite")
print("="*80)

generator = CompositeImageGenerator()
for name, data in all_bands.items():
    generator.add_band(name, data)

r = generator.bands[mapping['red']].copy()
g = generator.bands[mapping['green']].copy()
b = generator.bands[mapping['blue']].copy()

# Stack to RGB
rgb = np.dstack([r, g, b])

print(f"  Initial RGB median: {np.nanmedian(rgb):.4f}")

# Step 3: Calculate luminance
print("\n" + "="*80)
print("Step 3: Calculate luminance and create mask")
print("="*80)

# Luminance = weighted average (standard photometric weights)
luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

print(f"  Luminance median: {np.nanmedian(luminance):.4f}")
print(f"  Luminance mean: {np.nanmean(luminance):.4f}")

# Find 5th percentile threshold
lum_threshold = np.nanpercentile(luminance, 5)
print(f"  5th percentile threshold: {lum_threshold:.4f}")

# Create mask: pixels below threshold -> black
dark_mask = luminance < lum_threshold

print(f"  Pixels to mask (below 5th percentile): {np.sum(dark_mask)} ({100*np.sum(dark_mask)/dark_mask.size:.1f}%)")

# Apply mask to RGB
rgb_masked = rgb.copy()
rgb_masked[dark_mask] = 0  # Set dark pixels to pure black

print(f"  After masking median: {np.nanmedian(rgb_masked):.4f}")

# Step 4: Apply enhancements
print("\n" + "="*80)
print("Step 4: Apply enhancements")
print("="*80)

# Separate channels
r_masked = rgb_masked[:, :, 0]
g_masked = rgb_masked[:, :, 1]
b_masked = rgb_masked[:, :, 2]

# Gamma stretch to brighten mid-tones
print("  - Gamma stretch (0.6) to brighten")
r_gamma = np.clip(r_masked ** 0.6, 0, 1)
g_gamma = np.clip(g_masked ** 0.6, 0, 1)
b_gamma = np.clip(b_masked ** 0.6, 0, 1)

# Detail enhancement (unsharp mask)
print("  - Unsharp mask for detail")
from scipy import ndimage

r_detail = r_gamma + (r_gamma - ndimage.gaussian_filter(r_gamma, 2)) * 0.8
g_detail = g_gamma + (g_gamma - ndimage.gaussian_filter(g_gamma, 2)) * 0.8
b_detail = b_gamma + (b_gamma - ndimage.gaussian_filter(b_gamma, 2)) * 0.8

# Clip
r_detail = np.clip(r_detail, 0, 1)
g_detail = np.clip(g_detail, 0, 1)
b_detail = np.clip(b_detail, 0, 1)

# Star enhancement (brighten top 0.5%)
print("  - Star enhancement (99.5th percentile boost)")
r_thresh = np.percentile(r_detail, 99.5)
g_thresh = np.percentile(g_detail, 99.5)
b_thresh = np.percentile(b_detail, 99.5)

r_final = np.where(r_detail > r_thresh, np.clip(r_detail * 1.5, 0, 1), r_detail)
g_final = np.where(g_detail > g_thresh, np.clip(g_detail * 1.5, 0, 1), g_detail)
b_final = np.where(b_detail > b_thresh, np.clip(b_detail * 1.5, 0, 1), b_detail)

# Step 5: LRGB - Apply luminance layer
print("\n" + "="*80)
print("Step 5: Apply LRGB luminance layer")
print("="*80)

# Stack enhanced RGB
rgb_enhanced = np.dstack([r_final, g_final, b_final])

# Use original red channel as luminance (typically best SNR)
luminance_layer = r.copy()

# Apply gamma to luminance too
luminance_layer = np.clip(luminance_layer ** 0.6, 0, 1)

# Convert to HSV, replace V with luminance
from skimage import color

rgb_hsv = color.rgb2hsv(rgb_enhanced.astype(np.float32))

# Scale luminance to match RGB brightness
lum_scaled = luminance_layer / np.max(luminance_layer) if np.max(luminance_layer) > 0 else luminance_layer

# Replace value channel
rgb_hsv[:, :, 2] = lum_scaled.astype(np.float32)

# Convert back to RGB
rgb_lrgb = color.hsv2rgb(rgb_hsv).astype(np.float32)

print(f"  LRGB composite median: {np.nanmedian(rgb_lrgb):.4f}")
print(f"  LRGB composite mean: {np.nanmean(rgb_lrgb):.4f}")

# Save
print("\n" + "="*80)
print("Step 6: Save composite")
print("="*80)

generator.save_composite(rgb_lrgb, output_file, quality=95)

print(f"\n✓ Saved: {Path(output_file).resolve()}")
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nThis approach:")
print("  ✓ No Background2D - simple zscale normalization")
print("  ✓ Luminance-based masking - darkest 5% set to pure black")
print("  ✓ LRGB technique - sharp detail from luminance layer")
print("  ✓ Gamma + unsharp + star boost - bright stars, good contrast")