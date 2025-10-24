"""Diagnose background subtraction issue - check data values at each step."""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import FITSImageProcessor
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip

if len(sys.argv) < 2:
    print("Usage: python diagnose_background.py <fits_file>")
    sys.exit(1)

fits_file = sys.argv[1]

print("="*80)
print(f"DIAGNOSING: {Path(fits_file).name}")
print("="*80)

with FITSImageProcessor(fits_file) as fits_proc:
    bands = fits_proc.extract_bands()

    for band_name, band_data in bands.items():
        print(f"\n{'='*80}")
        print(f"Band: {band_name}")
        print(f"{'='*80}")

        print(f"\n1. RAW DATA:")
        print(f"   Shape: {band_data.shape}")
        print(f"   Min: {np.nanmin(band_data):.6e}")
        print(f"   Max: {np.nanmax(band_data):.6e}")
        print(f"   Mean: {np.nanmean(band_data):.6e}")
        print(f"   Median: {np.nanmedian(band_data):.6e}")
        print(f"   Std: {np.nanstd(band_data):.6e}")
        print(f"   NaN count: {np.sum(np.isnan(band_data))}")

        # Background2D estimation
        print(f"\n2. BACKGROUND2D ESTIMATION:")
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

            print(f"   Background min: {np.nanmin(bkg.background):.6e}")
            print(f"   Background max: {np.nanmax(bkg.background):.6e}")
            print(f"   Background median: {np.nanmedian(bkg.background):.6e}")
            print(f"   Background RMS median: {bkg.background_rms_median:.6e}")

            # Subtract background
            background_subtracted = band_data - bkg.background

            print(f"\n3. AFTER BACKGROUND SUBTRACTION:")
            print(f"   Min: {np.nanmin(background_subtracted):.6e}")
            print(f"   Max: {np.nanmax(background_subtracted):.6e}")
            print(f"   Mean: {np.nanmean(background_subtracted):.6e}")
            print(f"   Median: {np.nanmedian(background_subtracted):.6e}")
            print(f"   Negative values: {np.sum(background_subtracted < 0)} pixels ({100*np.sum(background_subtracted < 0)/background_subtracted.size:.2f}%)")
            print(f"   Zero/near-zero (< 1e-10): {np.sum(np.abs(background_subtracted) < 1e-10)} pixels")

            # Clip negative values
            clipped = np.clip(background_subtracted, 0, None)

            print(f"\n4. AFTER CLIPPING NEGATIVES:")
            print(f"   Min: {np.nanmin(clipped):.6e}")
            print(f"   Max: {np.nanmax(clipped):.6e}")
            print(f"   Non-zero pixels: {np.sum(clipped > 0)} ({100*np.sum(clipped > 0)/clipped.size:.2f}%)")

            # Percentile normalization
            p_low = np.nanpercentile(clipped, 1)
            p_high = np.nanpercentile(clipped, 99.5)

            print(f"\n5. NORMALIZATION PERCENTILES:")
            print(f"   1st percentile: {p_low:.6e}")
            print(f"   99.5th percentile: {p_high:.6e}")
            print(f"   Range: {p_high - p_low:.6e}")

            if p_high - p_low > 0:
                normalized = np.clip(
                    (clipped - p_low) / (p_high - p_low),
                    0, 1
                )

                print(f"\n6. AFTER NORMALIZATION:")
                print(f"   Min: {np.nanmin(normalized):.6f}")
                print(f"   Max: {np.nanmax(normalized):.6f}")
                print(f"   Mean: {np.nanmean(normalized):.6f}")
                print(f"   Median: {np.nanmedian(normalized):.6f}")
                print(f"   Non-zero pixels: {np.sum(normalized > 0)} ({100*np.sum(normalized > 0)/normalized.size:.2f}%)")
                print(f"   Pixels > 0.1: {np.sum(normalized > 0.1)} ({100*np.sum(normalized > 0.1)/normalized.size:.2f}%)")
                print(f"   Pixels > 0.5: {np.sum(normalized > 0.5)} ({100*np.sum(normalized > 0.5)/normalized.size:.2f}%)")
            else:
                print(f"\n   ERROR: Range is zero! Cannot normalize.")

        except Exception as e:
            print(f"   ERROR: Background2D failed: {e}")

print(f"\n{'='*80}")
print("DIAGNOSIS COMPLETE")
print("="*80)