"""Custom astronomical enhancements with Background2D subtraction.

Usage:
    python test_custom_enhancements.py <g.fits> <r.fits> <i.fits> <output.png>
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
    print("Usage: python test_custom_enhancements.py <g.fits> <r.fits> <i.fits> <output.png>")
    sys.exit(1)

fits_files = sys.argv[1:4]
output_file = sys.argv[4]

print("="*80)
print("CUSTOM ASTRONOMICAL ENHANCEMENTS WITH BACKGROUND2D")
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
                    box_size=(50, 50),  # Box size for background estimation
                    filter_size=(3, 3),  # Median filter size
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

            # Normalize after background subtraction
            # Clip negative values and normalize
            background_subtracted = np.clip(background_subtracted, 0, None)

            # Use percentile-based normalization for better dynamic range
            p_low = np.nanpercentile(background_subtracted, 1)
            p_high = np.nanpercentile(background_subtracted, 99.5)

            normalized = np.clip(
                (background_subtracted - p_low) / (p_high - p_low),
                0, 1
            )

            print(f"  Normalized: {p_low:.2e} - {p_high:.2e}")

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
        clip_limit=0.015  # Lower for subtler contrast
    )
    generator.bands[band_name] = enhanced

print("2. Sharp detail enhancement")
for band_name in [mapping['red'], mapping['green'], mapping['blue']]:
    enhanced = generator.enhance_detail(
        band_name,
        scale=2.5,    # Very sharp
        radius=2      # Tight radius
    )
    generator.bands[band_name] = enhanced

print("3. Star enhancement (brightest only)")
for band_name in [mapping['red'], mapping['green'], mapping['blue']]:
    enhanced = generator.enhance_stars(
        band_name,
        threshold_percentile=99.7,  # Only very brightest
        boost=2.0                    # Strong boost
    )
    generator.bands[band_name] = enhanced

print("4. Creating composite")

# Get the enhanced bands
r = generator.bands[mapping['red']]
g = generator.bands[mapping['green']]
b = generator.bands[mapping['blue']]

# Stack
rgb = np.dstack([r, g, b])

# Color balance
rgb = generator.apply_color_balance(rgb)

print("5. Final adjustments")
# Slight mid-tone boost (background already subtracted)
rgb = np.clip(rgb ** 0.95, 0, 1)  # Gamma adjustment

print("\n" + "="*80)
print("Step 3: Saving")
print("="*80)

generator.save_composite(rgb, output_file, quality=95)

output_path = Path(output_file).resolve()
print(f"\n✓ Background-subtracted composite saved!")
print(f"  Location: {output_path}")
print(f"\nEnhancements applied:")
print(f"  ✓ Background2D estimation and subtraction")
print(f"  ✓ Percentile-based normalization (1-99.5%)")
print(f"  ✓ Adaptive CLAHE contrast (clip=0.015)")
print(f"  ✓ Aggressive unsharp masking (scale=2.5)")
print(f"  ✓ Brightest star boost (99.7th percentile, 2.0x)")
print(f"  ✓ Color balancing")
print(f"  ✓ Gamma adjustment (0.95)")

print("\n" + "="*80)
print("COMPLETE - Professional astronomical processing!")
print("="*80)
print("\nThis should have:")
print("  - Much blacker background (Background2D removed sky glow)")
print("  - Crisp, well-defined stars")
print("  - Enhanced nebulosity detail")
print("  - Proper astronomical appearance")