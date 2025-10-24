"""Debug the hybrid approach step-by-step."""

import sys
from pathlib import Path
import numpy as np
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import FITSImageProcessor

if len(sys.argv) < 2:
    print("Usage: python debug_hybrid.py <fits_file>")
    sys.exit(1)

fits_file = sys.argv[1]

print("="*80)
print("DEBUG HYBRID APPROACH")
print("="*80)

with FITSImageProcessor(fits_file) as fits_proc:
    bands = fits_proc.extract_bands()

    for band_name, band_data in bands.items():
        print(f"\nSTEP 1: ZSCALE NORMALIZATION")
        normalized = fits_proc.normalize_band(band_name, method='zscale')
        print(f"  Min: {np.nanmin(normalized):.6f}")
        print(f"  Max: {np.nanmax(normalized):.6f}")
        print(f"  Median: {np.nanmedian(normalized):.6f}")
        print(f"  Mean: {np.nanmean(normalized):.6f}")

        print(f"\nSTEP 2: AFTER CLAHE")
        from skimage import exposure
        clahe_result = exposure.equalize_adapthist(normalized, clip_limit=0.03)
        print(f"  Min: {np.nanmin(clahe_result):.6f}")
        print(f"  Max: {np.nanmax(clahe_result):.6f}")
        print(f"  Median: {np.nanmedian(clahe_result):.6f}")
        print(f"  Mean: {np.nanmean(clahe_result):.6f}")

        print(f"\nSTEP 3: LUMINANCE MASKING")
        # Simulate RGB luminance (just using this channel 3 times)
        luminance = clahe_result  # In real case, weighted average of R,G,B

        lum_threshold = np.nanpercentile(luminance, 5)
        print(f"  5th percentile threshold: {lum_threshold:.6f}")

        dark_mask = luminance < lum_threshold
        print(f"  Pixels to mask: {np.sum(dark_mask)} ({100*np.sum(dark_mask)/dark_mask.size:.1f}%)")

        masked = clahe_result.copy()
        masked[dark_mask] = 0

        print(f"\nSTEP 4: AFTER MASKING")
        print(f"  Min: {np.nanmin(masked):.6f}")
        print(f"  Max: {np.nanmax(masked):.6f}")
        print(f"  Median: {np.nanmedian(masked):.6f}")
        print(f"  Mean: {np.nanmean(masked):.6f}")
        print(f"  Pixels > 0.1: {100*np.sum(masked > 0.1)/masked.size:.1f}%")
        print(f"  Pixels > 0.5: {100*np.sum(masked > 0.5)/masked.size:.1f}%")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)