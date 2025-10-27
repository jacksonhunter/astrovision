"""Diagnose all three bands to check for differences."""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import FITSImageProcessor
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip

if len(sys.argv) < 4:
    print("Usage: python diagnose_all_bands.py <g.fits> <r.fits> <i.fits>")
    sys.exit(1)

fits_files = sys.argv[1:4]

print("="*80)
print("COMPARING ALL THREE BANDS")
print("="*80)

results = {}

for fits_file in fits_files:
    file_name = Path(fits_file).name

    # Extract filter name
    parts = Path(fits_file).stem.split('.')
    if 'stk' in parts:
        idx = parts.index('stk')
        filter_name = parts[idx + 1]
    else:
        filter_name = file_name

    print(f"\n{'='*80}")
    print(f"Processing: {filter_name.upper()} band - {file_name}")
    print(f"{'='*80}")

    with FITSImageProcessor(fits_file) as fits_proc:
        bands = fits_proc.extract_bands()

        for band_name, band_data in bands.items():
            # Raw data
            raw_median = np.nanmedian(band_data)
            raw_mean = np.nanmean(band_data)
            raw_min = np.nanmin(band_data)
            raw_max = np.nanmax(band_data)

            print(f"\nRAW DATA:")
            print(f"  Range: [{raw_min:.3e}, {raw_max:.3e}]")
            print(f"  Mean: {raw_mean:.3e}, Median: {raw_median:.3e}")

            # Background2D
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

                bkg_median = np.nanmedian(bkg.background)
                background_subtracted = band_data - bkg.background

                print(f"\nBACKGROUND SUBTRACTION:")
                print(f"  Background median: {bkg_median:.3e}")
                print(f"  After subtraction median: {np.nanmedian(background_subtracted):.3e}")
                print(f"  Negative pixels: {100*np.sum(background_subtracted < 0)/background_subtracted.size:.1f}%")

                # Clip and normalize
                clipped = np.clip(background_subtracted, 0, None)
                p_low = np.nanpercentile(clipped[clipped > 0], 0.5)
                p_high = np.nanpercentile(clipped, 99.8)

                normalized = np.clip(
                    (clipped - p_low) / (p_high - p_low + 1e-10),
                    0, 1
                )

                print(f"\nNORMALIZATION:")
                print(f"  Percentile range: [{p_low:.3e}, {p_high:.3e}]")
                print(f"  Normalized median: {np.nanmedian(normalized):.4f}")
                print(f"  Normalized mean: {np.nanmean(normalized):.4f}")
                print(f"  Pixels > 0.1: {100*np.sum(normalized > 0.1)/normalized.size:.1f}%")
                print(f"  Pixels > 0.5: {100*np.sum(normalized > 0.5)/normalized.size:.1f}%")

                # Store results
                results[filter_name] = {
                    'raw_median': raw_median,
                    'bkg_median': bkg_median,
                    'norm_median': np.nanmedian(normalized),
                    'norm_mean': np.nanmean(normalized),
                    'p_low': p_low,
                    'p_high': p_high
                }

            except Exception as e:
                print(f"  ERROR: {e}")

print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}\n")

print(f"{'Band':<8} {'Raw Med':<12} {'Bkg Med':<12} {'Norm Med':<12} {'Norm Mean':<12}")
print("-" * 60)
for filter_name, data in results.items():
    print(f"{filter_name:<8} {data['raw_median']:<12.3e} {data['bkg_median']:<12.3e} "
          f"{data['norm_median']:<12.4f} {data['norm_mean']:<12.4f}")

print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")

# Check if bands are similar
medians = [data['norm_median'] for data in results.values()]
means = [data['norm_mean'] for data in results.values()]

print(f"\nNormalized median range: {min(medians):.4f} - {max(medians):.4f}")
print(f"Normalized mean range: {min(means):.4f} - {max(means):.4f}")

if max(medians) - min(medians) > 0.05:
    print("\n⚠ WARNING: Large difference between band medians!")
    print("  This may cause color balance issues.")
else:
    print("\n✓ Band medians are similar - good for compositing")

if max(means) < 0.15:
    print("\n⚠ WARNING: All bands have low mean values!")
    print("  Image will be very dark - brightness boost recommended.")
elif max(means) < 0.3:
    print("\n⚠ Note: Bands have moderate-low mean values.")
    print("  Some brightness adjustment may be needed.")
else:
    print("\n✓ Band brightness looks good")