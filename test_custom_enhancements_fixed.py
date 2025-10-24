"""Custom astronomical enhancements with Background2D subtraction - FIXED VERSION.

Usage:
    python test_custom_enhancements_fixed.py <g.fits> <r.fits> <i.fits> <output.png>
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

from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip

if len(sys.argv) < 5:
    print("Usage: python test_custom_enhancements_fixed.py <g.fits> <r.fits> <i.fits> <output.png>")
    sys.exit(1)

fits_files = sys.argv[1:4]
output_file = sys.argv[4]

print("="*80)
print("FIXED: CUSTOM ASTRONOMICAL ENHANCEMENTS WITH BACKGROUND2D")
print("="*80)
print("\nGoal: Professional background subtraction + Enhanced stars")

# Load bands
print("\n" + "="*80)
print("Step 1: Loading and Background Subtraction")
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

            print(f"  Loaded: {unique_name} - {band_data.shape}")

            # Background2D estimation
            print(f"  Estimating background with Background2D...")
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MedianBackground()

            try:
                bkg = Background2D(
                    band_data,
                    box_size=(50, 50),
                    filter_size=(3, 3),
                    sigma_clip=sigma_clip,
                    bkg_estimator=bkg_estimator
                )

                # Subtract background
                background_subtracted = band_data - bkg.background

                print(f"  Background median: {np.nanmedian(bkg.background):.2e}")
                print(f"  Background RMS: {bkg.background_rms_median:.2e}")

            except Exception as e:
                print(f"  Warning: Background2D failed ({e}), using simple median")
                bg_median = np.nanmedian(band_data)
                background_subtracted = band_data - bg_median

            # Clip negative values
            background_subtracted = np.clip(background_subtracted, 0, None)

            # FIXED: Use more aggressive percentile stretch for low-signal data
            # Use 0.5 percentile instead of 1st to avoid zeros
            p_low = np.nanpercentile(background_subtracted[background_subtracted > 0], 0.5)
            p_high = np.nanpercentile(background_subtracted, 99.8)

            # Ensure we have valid range
            if p_high - p_low < 1e-10:
                print(f"  Warning: Very low signal range, using simple scaling")
                p_low = 0
                p_high = np.nanpercentile(background_subtracted, 99.9)

            normalized = np.clip(
                (background_subtracted - p_low) / (p_high - p_low + 1e-10),
                0, 1
            )

            print(f"  Normalized: {p_low:.2e} - {p_high:.2e}")
            print(f"  Result median: {np.nanmedian(normalized):.4f}, mean: {np.nanmean(normalized):.4f}")

            all_bands[unique_name] = normalized
            all_metadata[unique_name] = metadata[band_name]

print(f"\nLoaded bands: {list(all_bands.keys())}")

# Map bands
mapping = {
    'red': 'r_band',
    'green': 'i_band',
    'blue': 'g_band'
}

print("\n" + "="*80)
print("Step 2: Applying Enhancements")
print("="*80)

generator = CompositeImageGenerator()

for name, data in all_bands.items():
    generator.add_band(name, data)

print("\n1. Moderate contrast enhancement")
for band_name in [mapping['red'], mapping['green'], mapping['blue']]:
    enhanced = generator.enhance_contrast(
        band_name,
        method='adaptive',
        clip_limit=0.02  # Slightly higher for better contrast
    )
    generator.bands[band_name] = enhanced

print("2. Sharp detail enhancement")
for band_name in [mapping['red'], mapping['green'], mapping['blue']]:
    enhanced = generator.enhance_detail(
        band_name,
        scale=2.0,    # Sharp but not overly aggressive
        radius=2
    )
    generator.bands[band_name] = enhanced

print("3. Star enhancement (brightest only)")
for band_name in [mapping['red'], mapping['green'], mapping['blue']]:
    enhanced = generator.enhance_stars(
        band_name,
        threshold_percentile=99.5,  # Top 0.5%
        boost=1.8
    )
    generator.bands[band_name] = enhanced

print("4. Creating composite")

# Get the enhanced bands
r = generator.bands[mapping['red']]
g = generator.bands[mapping['green']]
b = generator.bands[mapping['blue']]

# Stack
rgb = np.dstack([r, g, b])

# FIXED: Better color balance that handles low medians
print("5. Applying improved color balance")

# Use percentile-based color balance instead of median
# This avoids issues with very dark backgrounds
r_ref = np.nanpercentile(rgb[:, :, 0], 75)
g_ref = np.nanpercentile(rgb[:, :, 1], 75)
b_ref = np.nanpercentile(rgb[:, :, 2], 75)

avg_ref = (r_ref + g_ref + b_ref) / 3.0

if avg_ref > 0:
    r_scale = avg_ref / (r_ref + 1e-8)
    g_scale = avg_ref / (g_ref + 1e-8)
    b_scale = avg_ref / (b_ref + 1e-8)

    # Limit scaling to avoid extreme values
    r_scale = np.clip(r_scale, 0.5, 2.0)
    g_scale = np.clip(g_scale, 0.5, 2.0)
    b_scale = np.clip(b_scale, 0.5, 2.0)

    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * r_scale, 0, 1)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] * g_scale, 0, 1)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * b_scale, 0, 1)

    print(f"  Color balance scales: R={r_scale:.2f}, G={g_scale:.2f}, B={b_scale:.2f}")

print("6. Final adjustments")
# Apply gamma to brighten mid-tones while preserving blacks
rgb = np.clip(rgb ** 0.85, 0, 1)

# Optional: Slight contrast boost at the end
rgb_mean = np.mean(rgb)
if rgb_mean < 0.3:  # If still too dark
    print(f"  Image still dark (mean={rgb_mean:.3f}), applying brightness boost")
    rgb = np.clip(rgb * 1.3, 0, 1)

print("\n" + "="*80)
print("Step 3: Saving")
print("="*80)

generator.save_composite(rgb, output_file, quality=95)

output_path = Path(output_file).resolve()
print(f"\n✓ Background-subtracted composite saved!")
print(f"  Location: {output_path}")
print(f"\nEnhancements applied:")
print(f"  ✓ Background2D estimation and subtraction")
print(f"  ✓ Improved percentile-based normalization (0.5-99.8%)")
print(f"  ✓ Adaptive CLAHE contrast (clip=0.02)")
print(f"  ✓ Unsharp masking (scale=2.0)")
print(f"  ✓ Star boost (99.5th percentile, 1.8x)")
print(f"  ✓ Percentile-based color balancing (75th percentile)")
print(f"  ✓ Gamma adjustment (0.85)")

print("\n" + "="*80)
print("COMPLETE - Professional astronomical processing!")
print("="*80)