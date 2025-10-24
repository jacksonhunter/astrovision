"""Test script to create a composite from M31 FITS file.

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
OUTPUT_FILE = "m31_composite_output.png"

print("="*80)
print("M31 ANDROMEDA GALAXY - COMPOSITE GENERATOR")
print("="*80)

# Step 1: Load FITS file
print("\nStep 1: Loading FITS file...")
with FITSImageProcessor(FITS_FILE) as fits_proc:
    # Extract all bands
    bands = fits_proc.extract_bands()
    print(f"Found {len(bands)} bands: {list(bands.keys())}")

    # Get metadata
    metadata = fits_proc.get_band_info()
    for name, info in metadata.items():
        print(f"\n{name}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Filter: {info['filter']}")
        print(f"  Mean: {info['mean']:.2e}, Std: {info['std']:.2e}")

    # Normalize all bands
    print("\nNormalizing bands...")
    normalized = {}
    for name in bands.keys():
        normalized[name] = fits_proc.normalize_band(name, method='zscale')
        print(f"  Normalized {name}")

    # Step 2: Choose mapping strategy
    print("\n" + "="*80)
    print("Step 2: Determining band mapping")
    print("="*80)

    # For M31 PanSTARRS RGB, we likely have 3 bands already
    # Let's use simple automatic mapping first
    band_names = list(normalized.keys())

    if len(band_names) >= 3:
        # Assume RGB ordering or use first 3 bands
        mapping = {
            'red': band_names[0],
            'green': band_names[1],
            'blue': band_names[2]
        }
        print(f"\nAutomatic RGB mapping:")
        print(f"  Red:   {mapping['red']}")
        print(f"  Green: {mapping['green']}")
        print(f"  Blue:  {mapping['blue']}")
    else:
        print(f"Warning: Only {len(band_names)} bands found, need at least 3")
        sys.exit(1)

    # Step 3: Create composite
    print("\n" + "="*80)
    print("Step 3: Creating enhanced composite")
    print("="*80)

    generator = CompositeImageGenerator()

    # Add all bands
    for name, data in normalized.items():
        generator.add_band(name, data)

    # Create RGB composite with all enhancements
    print("\nGenerating RGB composite with enhancements:")
    print("  - Adaptive contrast enhancement (CLAHE)")
    print("  - Detail enhancement (unsharp masking)")
    print("  - Star enhancement")
    print("  - Color balancing")

    composite = generator.create_rgb_composite(
        r_band=mapping['red'],
        g_band=mapping['green'],
        b_band=mapping['blue'],
        enhance_contrast=True,
        enhance_details=True,
        enhance_stars=True,
        color_balance=True
    )

    # Step 4: Save result
    print("\n" + "="*80)
    print("Step 4: Saving composite")
    print("="*80)

    generator.save_composite(composite, OUTPUT_FILE, quality=95)

    print(f"\nComposite saved to: {OUTPUT_FILE}")
    print(f"Image shape: {composite.shape}")

print("\n" + "="*80)
print("SUCCESS! Composite generation complete.")
print("="*80)
print(f"\nOutput file: {OUTPUT_FILE}")
print("Open the image to see your beautiful M31 composite!")