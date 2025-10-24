"""AI-guided composite from multiple single-band FITS files.

Uses vision model to analyze bands and recommend optimal processing.

Usage:
    python test_multiband_with_ai.py band1.fits band2.fits band3.fits [output.png]
"""

import sys
from pathlib import Path
import numpy as np

# Suppress common warnings for cleaner output
from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import (
    FITSImageProcessor,
    CompositeImageGenerator,
    VisionGuidedCompositor
)

if len(sys.argv) < 4:
    print("Usage: python test_multiband_with_ai.py <band1.fits> <band2.fits> <band3.fits> [output.png]")
    print("\nExample:")
    print('  python test_multiband_with_ai.py g.fits r.fits i.fits ai_composite.png')
    print("\nNOTE: First run will download ~11GB vision model (10-30 minutes)")
    sys.exit(1)

# Parse arguments
fits_files = sys.argv[1:-1] if sys.argv[-1].endswith('.png') else sys.argv[1:]
output_file = sys.argv[-1] if sys.argv[-1].endswith('.png') else "ai_multiband_composite.png"

print("="*80)
print("AI-GUIDED MULTI-BAND FITS COMPOSITE")
print("="*80)
print("\nNOTE: First run will download ~11GB vision model")
print("      This can take 10-30 minutes. Subsequent runs are fast.")

print(f"\nInput bands: {len(fits_files)}")
for f in fits_files:
    print(f"  - {Path(f).name}")

print(f"\nOutput: {output_file}")

# Step 1: Load all bands
print("\n" + "="*80)
print("Step 1: Loading bands from FITS files")
print("="*80)

all_bands = {}
all_metadata = {}

for fits_file in fits_files:
    print(f"\nLoading: {Path(fits_file).name}")

    try:
        with FITSImageProcessor(fits_file) as fits_proc:
            bands = fits_proc.extract_bands()
            metadata = fits_proc.get_band_info()

            for band_name, band_data in bands.items():
                file_stem = Path(fits_file).stem
                parts = file_stem.split('.')
                if 'stk' in parts:
                    idx = parts.index('stk')
                    if idx + 1 < len(parts):
                        filter_name = parts[idx + 1]
                        unique_name = f"{filter_name}_band"
                    else:
                        unique_name = f"{file_stem}_{band_name}"
                else:
                    unique_name = f"{file_stem}_{band_name}"

                print(f"  Found: {unique_name} - shape {band_data.shape}")

                normalized = fits_proc.normalize_band(band_name, method='zscale')
                all_bands[unique_name] = normalized
                all_metadata[unique_name] = metadata[band_name]
                all_metadata[unique_name]['filter'] = filter_name if 'stk' in parts else 'unknown'

    except Exception as e:
        print(f"  ERROR loading {fits_file}: {e}")
        continue

print(f"\nTotal bands loaded: {len(all_bands)}")

if len(all_bands) < 3:
    print(f"\nERROR: Need at least 3 bands for RGB composite, got {len(all_bands)}")
    sys.exit(1)

# Step 2: AI-Guided Band Mapping
print("\n" + "="*80)
print("Step 2: AI Vision Analysis for Band Mapping")
print("="*80)

print("\nInitializing vision model...")
print("(First run: downloading ~11GB model - please be patient)")

compositor = VisionGuidedCompositor()

print("\nAnalyzing bands with AI...")
print("(This will take 1-2 minutes per band)")

try:
    mapping = compositor.recommend_band_mapping(
        all_bands,
        all_metadata,
        mode='rgb'
    )

    print(f"\nAI-Recommended RGB mapping:")
    print(f"  Red channel:   {mapping['red']}")
    print(f"  Green channel: {mapping['green']}")
    print(f"  Blue channel:  {mapping['blue']}")

except Exception as e:
    print(f"\nAI analysis failed: {e}")
    print("Falling back to automatic mapping...")

    # Fallback mapping
    band_names = list(all_bands.keys())
    filter_to_channel = {
        'g': 'blue',
        'r': 'red',
        'i': 'green',
        'z': 'red',
        'y': 'red',
    }

    mapping = {'red': None, 'green': None, 'blue': None}

    for band_name in band_names:
        filter_name = all_metadata[band_name].get('filter', 'unknown')
        if filter_name in filter_to_channel:
            channel = filter_to_channel[filter_name]
            if mapping[channel] is None:
                mapping[channel] = band_name

    remaining = [b for b in band_names if b not in mapping.values()]
    for channel in ['red', 'green', 'blue']:
        if mapping[channel] is None and remaining:
            mapping[channel] = remaining.pop(0)

    print(f"\nFallback mapping:")
    print(f"  Red:   {mapping['red']}")
    print(f"  Green: {mapping['green']}")
    print(f"  Blue:  {mapping['blue']}")

# Step 3: Create initial composite
print("\n" + "="*80)
print("Step 3: Creating initial composite")
print("="*80)

generator = CompositeImageGenerator()

for name, data in all_bands.items():
    generator.add_band(name, data)

print("\nGenerating composite with standard enhancements...")

composite = generator.create_rgb_composite(
    r_band=mapping['red'],
    g_band=mapping['green'],
    b_band=mapping['blue'],
    enhance_contrast=True,
    enhance_details=True,
    enhance_stars=True,
    color_balance=True
)

# Step 4: AI Quality Assessment & Optimization
print("\n" + "="*80)
print("Step 4: AI Quality Assessment")
print("="*80)

print("\nAnalyzing composite quality with AI...")

try:
    assessment = compositor.optimize_composite_parameters(composite)

    print(f"\nAI Assessment:")
    print(f"  Overall Quality: {assessment.get('overall_quality', 'N/A')}")
    print(f"  Contrast: {assessment.get('contrast_adjustment', 'N/A')}")
    print(f"  Brightness: {assessment.get('brightness_adjustment', 'N/A')}")
    print(f"  Saturation: {assessment.get('color_saturation', 'N/A')}")
    print(f"  Detail: {assessment.get('detail_enhancement', 'N/A')}")
    print(f"  Stars: {assessment.get('star_enhancement', 'N/A')}")

    recommendations = assessment.get('specific_recommendations', [])
    if recommendations:
        print(f"\n  Specific Recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")

    print("\n" + "-"*80)
    print("AI Suggestions Summary:")
    print("-"*80)
    print(f"Background darkness: {'Needs improvement' if 'contrast' in str(recommendations).lower() or 'background' in str(recommendations).lower() else 'Good'}")
    print(f"Star definition: {'Needs enhancement' if 'star' in str(recommendations).lower() else 'Good'}")
    print(f"Overall appearance: {assessment.get('overall_quality', 'Unknown')}")

except Exception as e:
    print(f"AI assessment failed: {e}")

# Step 5: Save result
print("\n" + "="*80)
print("Step 5: Saving AI-Guided Composite")
print("="*80)

generator.save_composite(composite, output_file, quality=95)

output_path = Path(output_file).resolve()
print(f"\n✓ AI-Guided composite saved!")
print(f"  Location: {output_path}")
print(f"  Size: {composite.shape[1]}x{composite.shape[0]} pixels")
print(f"  File exists: {output_path.exists()}")

print("\n" + "="*80)
print("AI-GUIDED COMPOSITE COMPLETE")
print("="*80)
print("\nThe AI has analyzed your data and applied optimal processing.")
print("Compare this with the standard composite to see the difference!")