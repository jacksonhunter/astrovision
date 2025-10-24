"""Test script with AI-guided band mapping for M31 FITS file.

This version uses the vision model to analyze bands and recommend
optimal color mapping. First run will download the ~11GB model.

Run this in PyCharm after installing dependencies with:
    pip install -e .
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import (
    FITSImageProcessor,
    CompositeImageGenerator,
    VisionGuidedCompositor
)

# Input/output paths
FITS_FILE = r"C:\Users\jacks\experiments\PycharmProjects\Mission Control\examples\output\m31_panstarrs_rgb.fits"
OUTPUT_FILE = "m31_ai_guided_composite.png"

print("="*80)
print("M31 ANDROMEDA - AI-GUIDED COMPOSITE GENERATOR")
print("="*80)
print("\nNOTE: First run will download ~11GB vision model (10-30 minutes)")
print("Subsequent runs will use cached model and be much faster.")

# Step 1: Load FITS file
print("\n" + "="*80)
print("Step 1: Loading FITS file...")
print("="*80)

with FITSImageProcessor(FITS_FILE) as fits_proc:
    # Extract all bands
    bands = fits_proc.extract_bands()
    metadata = fits_proc.get_band_info()

    print(f"\nFound {len(bands)} bands: {list(bands.keys())}")

    # Normalize all bands
    print("\nNormalizing bands using ZScale...")
    normalized = {}
    for name in bands.keys():
        normalized[name] = fits_proc.normalize_band(name, method='zscale')
        print(f"  ✓ {name}")

    # Step 2: AI Analysis
    print("\n" + "="*80)
    print("Step 2: AI Vision Analysis of Bands")
    print("="*80)

    print("\nInitializing vision model...")
    print("(This may take a while on first run - downloading model)")

    compositor = VisionGuidedCompositor()

    print("\nAnalyzing bands with AI...")
    mapping = compositor.recommend_band_mapping(
        normalized,
        metadata,
        mode='rgb'
    )

    print(f"\nAI-Recommended RGB mapping:")
    print(f"  Red channel:   {mapping['red']}")
    print(f"  Green channel: {mapping['green']}")
    print(f"  Blue channel:  {mapping['blue']}")

    # Step 3: Create composite
    print("\n" + "="*80)
    print("Step 3: Creating Enhanced Composite")
    print("="*80)

    generator = CompositeImageGenerator()

    # Add all bands
    for name, data in normalized.items():
        generator.add_band(name, data)

    # Create RGB composite with all enhancements
    print("\nApplying enhancements:")
    print("  ✓ Adaptive contrast (CLAHE)")
    print("  ✓ Detail enhancement")
    print("  ✓ Star highlighting")
    print("  ✓ Color balancing")

    composite = generator.create_rgb_composite(
        r_band=mapping['red'],
        g_band=mapping['green'],
        b_band=mapping['blue'],
        enhance_contrast=True,
        enhance_details=True,
        enhance_stars=True,
        color_balance=True
    )

    # Step 4: AI Quality Assessment
    print("\n" + "="*80)
    print("Step 4: AI Quality Assessment")
    print("="*80)

    print("\nAnalyzing composite quality...")
    try:
        optimization = compositor.optimize_composite_parameters(composite)

        print(f"\nAI Assessment:")
        print(f"  Overall Quality: {optimization.get('overall_quality', 'N/A')}")
        print(f"  Contrast: {optimization.get('contrast_adjustment', 'N/A')}")
        print(f"  Brightness: {optimization.get('brightness_adjustment', 'N/A')}")
        print(f"  Saturation: {optimization.get('color_saturation', 'N/A')}")

        recommendations = optimization.get('specific_recommendations', [])
        if recommendations:
            print(f"\n  Recommendations:")
            for rec in recommendations:
                print(f"    - {rec}")
    except Exception as e:
        print(f"AI assessment skipped: {e}")

    # Step 5: Save result
    print("\n" + "="*80)
    print("Step 5: Saving Composite")
    print("="*80)

    generator.save_composite(composite, OUTPUT_FILE, quality=95)

    print(f"\n✓ Composite saved to: {OUTPUT_FILE}")
    print(f"  Image shape: {composite.shape}")
    print(f"  Image size: {composite.shape[0]}x{composite.shape[1]} pixels")

print("\n" + "="*80)
print("SUCCESS! AI-Guided Composite Complete")
print("="*80)
print(f"\nOutput: {OUTPUT_FILE}")
print("\nCompare this with the automatic version (test_m31_composite.py)")
print("to see the difference AI guidance makes!")