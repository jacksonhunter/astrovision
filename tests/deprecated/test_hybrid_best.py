"""HYBRID BEST-OF-EVERYTHING approach.

Combines:
- zscale normalization (preserves signal for CLAHE)
- CLAHE adaptive contrast (what makes rings_composite.png look good)
- Luminance masking (darkest 5% -> pure black backgrounds)
- LRGB technique (sharp detail from luminance layer)
- Star enhancement (bright white stars)

Usage:
    python test_hybrid_best.py <g.fits> <r.fits> <i.fits> <output.png>
"""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import FITSImageProcessor, CompositeImageGenerator

if len(sys.argv) < 5:
    print("Usage: python test_hybrid_best.py <g.fits> <r.fits> <i.fits> <output.png>")
    sys.exit(1)

fits_files = sys.argv[1:4]
output_file = sys.argv[4]

print("="*80)
print("HYBRID BEST-OF-EVERYTHING APPROACH")
print("="*80)
print("\nCombining:")
print("  ✓ zscale normalization (preserves signal)")
print("  ✓ CLAHE adaptive contrast (from rings_composite.png)")
print("  ✓ Luminance masking (dark backgrounds)")
print("  ✓ LRGB technique (sharp details)")
print("  ✓ Star enhancement (bright stars)")

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

            # zscale normalization (NOT Background2D)
            normalized = fits_proc.normalize_band(band_name, method='zscale')

            print(f"  {unique_name}: median={np.nanmedian(normalized):.4f}, mean={np.nanmean(normalized):.4f}")

            all_bands[unique_name] = normalized

# Map to RGB
mapping = {'red': 'r_band', 'green': 'i_band', 'blue': 'g_band'}

print("\n" + "="*80)
print("Step 2: Apply CLAHE + enhancements")
print("="*80)

generator = CompositeImageGenerator()
for name, data in all_bands.items():
    generator.add_band(name, data)

# Get individual channels
r = generator.bands[mapping['red']].copy()
g = generator.bands[mapping['green']].copy()
b = generator.bands[mapping['blue']].copy()

# Save original luminance BEFORE any processing (for LRGB later)
luminance_original = r.copy()  # Red channel typically has best SNR

# Apply CLAHE (this is what makes rings_composite look good)
print("  - CLAHE adaptive contrast")
from skimage import exposure

r_clahe = exposure.equalize_adapthist(r, clip_limit=0.03)
g_clahe = exposure.equalize_adapthist(g, clip_limit=0.03)
b_clahe = exposure.equalize_adapthist(b, clip_limit=0.03)

# Detail enhancement (unsharp mask)
print("  - Unsharp mask for detail")
from scipy import ndimage

r_detail = r_clahe + (r_clahe - ndimage.gaussian_filter(r_clahe, 2)) * 0.5
g_detail = g_clahe + (g_clahe - ndimage.gaussian_filter(g_clahe, 2)) * 0.5
b_detail = b_clahe + (b_clahe - ndimage.gaussian_filter(b_clahe, 2)) * 0.5

# Clip
r_detail = np.clip(r_detail, 0, 1)
g_detail = np.clip(g_detail, 0, 1)
b_detail = np.clip(b_detail, 0, 1)

# Star enhancement
print("  - Star boost (99th percentile)")
r_thresh = np.percentile(r_detail, 99)
g_thresh = np.percentile(g_detail, 99)
b_thresh = np.percentile(b_detail, 99)

r_stars = np.where(r_detail > r_thresh, np.clip(r_detail * 1.3, 0, 1), r_detail)
g_stars = np.where(g_detail > g_thresh, np.clip(g_detail * 1.3, 0, 1), g_detail)
b_stars = np.where(b_detail > b_thresh, np.clip(b_detail * 1.3, 0, 1), b_detail)

# Stack to RGB
rgb_enhanced = np.dstack([r_stars, g_stars, b_stars])

print(f"  After enhancements median: {np.nanmedian(rgb_enhanced):.4f}")

# Step 3: Luminance masking for dark backgrounds
print("\n" + "="*80)
print("Step 3: Luminance masking (darkest 5% -> black)")
print("="*80)

# Calculate luminance from enhanced RGB
luminance_calc = 0.2126 * rgb_enhanced[:, :, 0] + 0.7152 * rgb_enhanced[:, :, 1] + 0.0722 * rgb_enhanced[:, :, 2]

# Find 5th percentile threshold
lum_threshold = np.nanpercentile(luminance_calc, 5)
print(f"  5th percentile threshold: {lum_threshold:.4f}")

# Create mask
dark_mask = luminance_calc < lum_threshold
print(f"  Masking {np.sum(dark_mask)} pixels ({100*np.sum(dark_mask)/dark_mask.size:.1f}%)")

# Apply mask
rgb_masked = rgb_enhanced.copy()
rgb_masked[dark_mask] = 0  # Pure black

print(f"  After masking median: {np.nanmedian(rgb_masked):.4f}")
print(f"  After masking mean: {np.nanmean(rgb_masked):.4f}")
print(f"  Pixels > 0.1: {100*np.sum(rgb_masked > 0.1)/rgb_masked.size:.1f}%")

# Step 4: LRGB - Apply luminance layer for sharp detail
print("\n" + "="*80)
print("Step 4: Apply LRGB luminance layer (simplified)")
print("="*80)

# SIMPLIFIED LRGB: Blend original luminance with enhanced RGB
# Avoids HSV conversion issues that create NaN

# Prepare luminance from original red channel
lum_vmin = np.percentile(luminance_original, 0.5)
lum_vmax = np.percentile(luminance_original, 99.5)
luminance_normalized = np.clip(
    (luminance_original - lum_vmin) / (lum_vmax - lum_vmin + 1e-10),
    0, 1
)

# Apply CLAHE to luminance for detail
luminance_clahe = exposure.equalize_adapthist(luminance_normalized, clip_limit=0.03)

# Blend: 70% enhanced RGB color, 30% original luminance for sharpness
rgb_lrgb = rgb_masked.copy()

# Calculate current luminance of enhanced RGB
current_lum = 0.2126 * rgb_lrgb[:, :, 0] + 0.7152 * rgb_lrgb[:, :, 1] + 0.0722 * rgb_lrgb[:, :, 2]

# Scale factor to apply luminance detail
# Where current_lum is non-zero, adjust to match luminance_clahe
scale_factor = np.ones_like(current_lum)
non_zero_mask = current_lum > 0.01  # Only adjust non-black pixels
scale_factor[non_zero_mask] = luminance_clahe[non_zero_mask] / (current_lum[non_zero_mask] + 1e-10)

# Clamp scale factor to reasonable range
scale_factor = np.clip(scale_factor, 0.5, 2.0)

# Apply to all channels
for i in range(3):
    rgb_lrgb[:, :, i] = np.clip(rgb_lrgb[:, :, i] * scale_factor, 0, 1)

print(f"  Final LRGB median: {np.nanmedian(rgb_lrgb):.4f}")
print(f"  Final LRGB mean: {np.nanmean(rgb_lrgb):.4f}")

# Step 5: Save
print("\n" + "="*80)
print("Step 5: Save composite")
print("="*80)

generator.save_composite(rgb_lrgb, output_file, quality=95)

output_path = Path(output_file).resolve()
print(f"\n✓ Saved: {output_path}")

print("\n" + "="*80)
print("HYBRID COMPOSITE COMPLETE!")
print("="*80)
print("\nThis combines ALL the best elements:")
print("  ✓ zscale normalization (preserves signal for CLAHE)")
print("  ✓ CLAHE contrast (what makes rings_composite.png look good)")
print("  ✓ Unsharp masking (detail enhancement)")
print("  ✓ Star boost (99th percentile, bright white stars)")
print("  ✓ Luminance masking (darkest 5% -> pure black backgrounds)")
print("  ✓ LRGB technique (original luminance for maximum sharpness)")