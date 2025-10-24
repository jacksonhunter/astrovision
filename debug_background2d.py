"""Debug where the data goes to zero with Background2D.

Usage:
    python debug_background2d.py <fits_file>
"""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import FITSImageProcessor
from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS
from astropy.stats import SigmaClip

if len(sys.argv) < 2:
    print("Usage: python debug_background2d.py <fits_file>")
    sys.exit(1)

fits_file = sys.argv[1]

print("="*80)
print(f"DEBUGGING BACKGROUND2D - Step by step")
print(f"File: {Path(fits_file).name}")
print("="*80)

with FITSImageProcessor(fits_file) as fits_proc:
    bands = fits_proc.extract_bands()

    for band_name, band_data in bands.items():
        print(f"\nSTEP 1: RAW DATA")
        print(f"  Min: {np.nanmin(band_data):.6e}")
        print(f"  Max: {np.nanmax(band_data):.6e}")
        print(f"  Median: {np.nanmedian(band_data):.6e}")
        print(f"  Non-zero pixels: {np.sum(band_data != 0)} ({100*np.sum(band_data != 0)/band_data.size:.1f}%)")

        mask = np.isnan(band_data)

        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        bkg_estimator = SExtractorBackground()
        bkgrms_estimator = StdBackgroundRMS()

        print(f"\nSTEP 2: BACKGROUND2D ESTIMATION (box_size=128)")
        bkg = Background2D(
            band_data,
            box_size=(128, 128),
            mask=mask,
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            bkgrms_estimator=bkgrms_estimator,
            exclude_percentile=10.0,
            edge_method='pad'
        )

        print(f"  Background min: {np.nanmin(bkg.background):.6e}")
        print(f"  Background max: {np.nanmax(bkg.background):.6e}")
        print(f"  Background median: {np.nanmedian(bkg.background):.6e}")

        background_subtracted = band_data - bkg.background

        print(f"\nSTEP 3: AFTER BACKGROUND SUBTRACTION")
        print(f"  Min: {np.nanmin(background_subtracted):.6e}")
        print(f"  Max: {np.nanmax(background_subtracted):.6e}")
        print(f"  Median: {np.nanmedian(background_subtracted):.6e}")
        print(f"  Negative pixels: {np.sum(background_subtracted < 0)} ({100*np.sum(background_subtracted < 0)/background_subtracted.size:.1f}%)")
        print(f"  Zero pixels: {np.sum(background_subtracted == 0)} ({100*np.sum(background_subtracted == 0)/background_subtracted.size:.1f}%)")
        print(f"  Positive pixels: {np.sum(background_subtracted > 0)} ({100*np.sum(background_subtracted > 0)/background_subtracted.size:.1f}%)")

        clipped = np.clip(background_subtracted, 0, None)

        print(f"\nSTEP 4: AFTER CLIPPING NEGATIVES")
        print(f"  Min: {np.nanmin(clipped):.6e}")
        print(f"  Max: {np.nanmax(clipped):.6e}")
        print(f"  Median: {np.nanmedian(clipped):.6e}")
        print(f"  Zero pixels: {np.sum(clipped == 0)} ({100*np.sum(clipped == 0)/clipped.size:.1f}%)")
        print(f"  Non-zero pixels: {np.sum(clipped > 0)} ({100*np.sum(clipped > 0)/clipped.size:.1f}%)")

        if np.sum(clipped > 0) > 0:
            p_low = np.nanpercentile(clipped[clipped > 0], 1)
            p_high = np.nanpercentile(clipped, 99.5)

            print(f"\nSTEP 5: PERCENTILES FOR NORMALIZATION")
            print(f"  1st percentile (of non-zero): {p_low:.6e}")
            print(f"  99.5th percentile: {p_high:.6e}")
            print(f"  Range: {p_high - p_low:.6e}")

            if p_high - p_low > 0:
                normalized = np.clip(
                    (clipped - p_low) / (p_high - p_low + 1e-10),
                    0, 1
                )

                print(f"\nSTEP 6: AFTER NORMALIZATION")
                print(f"  Min: {np.nanmin(normalized):.6f}")
                print(f"  Max: {np.nanmax(normalized):.6f}")
                print(f"  Median: {np.nanmedian(normalized):.6f}")
                print(f"  Mean: {np.nanmean(normalized):.6f}")
                print(f"  Pixels > 0: {np.sum(normalized > 0)} ({100*np.sum(normalized > 0)/normalized.size:.1f}%)")
                print(f"  Pixels > 0.1: {np.sum(normalized > 0.1)} ({100*np.sum(normalized > 0.1)/normalized.size:.1f}%)")
                print(f"  Pixels > 0.5: {np.sum(normalized > 0.5)} ({100*np.sum(normalized > 0.5)/normalized.size:.1f}%)")
            else:
                print(f"\n  ERROR: Percentile range is ZERO - cannot normalize!")
        else:
            print(f"\n  ERROR: NO positive pixels after clipping!")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)