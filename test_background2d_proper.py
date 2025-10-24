"""Test Background2D with PROPER parameters for astronomical survey data.

Usage:
    python test_background2d_proper.py <g.fits> <r.fits> <i.fits> <output.png>
"""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import FITSImageProcessor, CompositeImageGenerator
from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS
from astropy.stats import SigmaClip

if len(sys.argv) < 5:
    print("Usage: python test_background2d_proper.py <g.fits> <r.fits> <i.fits> <output.png>")
    sys.exit(1)

fits_files = sys.argv[1:4]
output_file = sys.argv[4]

print("="*80)
print("BACKGROUND2D WITH PROPER ASTRONOMICAL PARAMETERS")
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

            print(f"\n  Band: {unique_name}")
            print(f"    Raw median: {np.nanmedian(band_data):.3e}")

            # Create mask for NaN values
            mask = np.isnan(band_data)
            print(f"    NaN pixels: {np.sum(mask)} ({100*np.sum(mask)/mask.size:.1f}%)")

            # Use PROPER Background2D parameters
            sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
            bkg_estimator = SExtractorBackground()  # DEFAULT, optimized for astronomy
            bkgrms_estimator = StdBackgroundRMS()

            bkg = Background2D(
                band_data,
                box_size=(128, 128),           # LARGER boxes for better background sampling
                mask=mask,                      # Explicitly mask NaNs
                filter_size=(3, 3),
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,    # SExtractor algorithm
                bkgrms_estimator=bkgrms_estimator,
                exclude_percentile=10.0,        # Default
                edge_method='pad'               # Default
            )

            print(f"    Background median: {np.nanmedian(bkg.background):.3e}")
            print(f"    Background RMS: {bkg.background_rms_median:.3e}")

            # Subtract background
            background_subtracted = band_data - bkg.background

            print(f"    After subtraction median: {np.nanmedian(background_subtracted):.3e}")
            neg_pct = 100*np.sum(background_subtracted < 0)/background_subtracted.size
            print(f"    Negative pixels: {neg_pct:.1f}%")

            # Clip negatives
            clipped = np.clip(background_subtracted, 0, None)

            # Normalize using percentiles
            p_low = np.nanpercentile(clipped[clipped > 0], 1)
            p_high = np.nanpercentile(clipped, 99.5)

            normalized = np.clip(
                (clipped - p_low) / (p_high - p_low + 1e-10),
                0, 1
            )

            print(f"    Normalized median: {np.nanmedian(normalized):.4f}")
            print(f"    Normalized mean: {np.nanmean(normalized):.4f}")

            all_bands[unique_name] = normalized

print(f"\n{'='*80}")
print("Creating composite")
print("="*80)

mapping = {'red': 'r_band', 'green': 'i_band', 'blue': 'g_band'}

generator = CompositeImageGenerator()
for name, data in all_bands.items():
    generator.add_band(name, data)

# Apply enhancements
print("\nEnhancements:")
print("  - Adaptive contrast (CLAHE)")
print("  - Detail enhancement")
print("  - Star boost")
print("  - Color balance")

# DON'T use create_rgb_composite - it applies CLAHE which kills low-median data
# Instead, manually build the composite with custom enhancements

r = generator.bands[mapping['red']].copy()
g = generator.bands[mapping['green']].copy()
b = generator.bands[mapping['blue']].copy()

print("  Manual enhancements (skip CLAHE, use custom contrast):")

# Apply gamma stretch INSTEAD of CLAHE for low-median data
print("    - Gamma stretch (0.5) to brighten")
r = np.clip(r ** 0.5, 0, 1)
g = np.clip(g ** 0.5, 0, 1)
b = np.clip(b ** 0.5, 0, 1)

# Detail enhancement
print("    - Detail enhancement (unsharp mask)")
from scipy import ndimage
r = r + (r - ndimage.gaussian_filter(r, 2)) * 0.8
g = g + (g - ndimage.gaussian_filter(g, 2)) * 0.8
b = b + (b - ndimage.gaussian_filter(b, 2)) * 0.8

# Star boost
print("    - Star boost")
r_thresh = np.percentile(r, 99)
g_thresh = np.percentile(g, 99)
b_thresh = np.percentile(b, 99)

r = np.where(r > r_thresh, np.clip(r * 1.5, 0, 1), r)
g = np.where(g > g_thresh, np.clip(g * 1.5, 0, 1), g)
b = np.where(b > b_thresh, np.clip(b * 1.5, 0, 1), b)

# Clip
r = np.clip(r, 0, 1)
g = np.clip(g, 0, 1)
b = np.clip(b, 0, 1)

# Stack
enhanced = np.dstack([r, g, b])

# Color balance
print("    - Color balance")
enhanced = generator.apply_color_balance(enhanced)

generator.save_composite(enhanced, output_file, quality=95)

print(f"\n✓ Saved: {Path(output_file).resolve()}")
print(f"\nFinal image stats:")
print(f"  Median: {np.nanmedian(enhanced):.4f}")
print(f"  Mean: {np.nanmean(enhanced):.4f}")