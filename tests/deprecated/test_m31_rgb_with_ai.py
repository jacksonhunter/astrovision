"""AI-guided enhancement of M31 RGB composite FITS file.

Uses vision model to analyze the composite and suggest optimizations.
"""

import numpy as np
from astropy.io import fits
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import CompositeImageGenerator, VisionGuidedCompositor

# Input/output paths
FITS_FILE = r"C:\Users\jacks\experiments\PycharmProjects\Mission Control\examples\output\m31_panstarrs_rgb.fits"
OUTPUT_FILE = "m31_ai_optimized.png"

print("="*80)
print("M31 ANDROMEDA - AI-GUIDED ENHANCEMENT")
print("="*80)
print("\nNOTE: First run will download ~11GB vision model")
print("      This can take 10-30 minutes depending on connection.")

# Step 1: Load RGB data
print("\n" + "="*80)
print("Step 1: Loading RGB FITS data")
print("="*80)

with fits.open(FITS_FILE) as hdul:
    rgb_data = hdul[0].data
    print(f"  Loaded: {rgb_data.shape}")

# Handle dimension order
if rgb_data.shape[0] == 3:
    rgb_data = np.moveaxis(rgb_data, 0, -1)
    print(f"  Transposed to: {rgb_data.shape}")

# Normalize
if rgb_data.dtype == np.uint8:
    rgb_normalized = rgb_data.astype(np.float32) / 255.0
else:
    rgb_normalized = rgb_data.astype(np.float32)
    rgb_normalized = (rgb_normalized - rgb_normalized.min()) / (rgb_normalized.max() - rgb_normalized.min())

# Step 2: Extract channels
print("\n" + "="*80)
print("Step 2: Extracting RGB channels")
print("="*80)

r_channel = rgb_normalized[:, :, 0]
g_channel = rgb_normalized[:, :, 1]
b_channel = rgb_normalized[:, :, 2]

print(f"  Red:   {r_channel.shape}, mean={r_channel.mean():.3f}")
print(f"  Green: {g_channel.shape}, mean={g_channel.mean():.3f}")
print(f"  Blue:  {b_channel.shape}, mean={b_channel.mean():.3f}")

# Step 3: Get AI baseline assessment
print("\n" + "="*80)
print("Step 3: AI Assessment of Original Image")
print("="*80)

print("\nInitializing AI vision model...")
print("(First run: downloading model - please wait...)")

compositor = VisionGuidedCompositor()

print("\nAnalyzing original composite quality...")
try:
    original_assessment = compositor.optimize_composite_parameters(rgb_normalized)

    print(f"\nOriginal Image Assessment:")
    print(f"  Overall Quality: {original_assessment.get('overall_quality', 'N/A')}")
    print(f"  Contrast: {original_assessment.get('contrast_adjustment', 'N/A')}")
    print(f"  Brightness: {original_assessment.get('brightness_adjustment', 'N/A')}")
    print(f"  Saturation: {original_assessment.get('color_saturation', 'N/A')}")
    print(f"  Detail Enhancement: {original_assessment.get('detail_enhancement', 'N/A')}")
    print(f"  Star Enhancement: {original_assessment.get('star_enhancement', 'N/A')}")

    recommendations = original_assessment.get('specific_recommendations', [])
    if recommendations:
        print(f"\n  AI Recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")

    # Decide on enhancements based on AI
    should_enhance_contrast = original_assessment.get('contrast_adjustment') in ['increase', 'more']
    should_enhance_detail = original_assessment.get('detail_enhancement') in ['more', 'increase']
    should_enhance_stars = original_assessment.get('star_enhancement') in ['more', 'increase']

except Exception as e:
    print(f"  AI assessment failed: {e}")
    print("  Falling back to default enhancements")
    should_enhance_contrast = True
    should_enhance_detail = True
    should_enhance_stars = True

# Step 4: Apply AI-recommended enhancements
print("\n" + "="*80)
print("Step 4: Applying AI-Recommended Enhancements")
print("="*80)

generator = CompositeImageGenerator()
generator.add_band('red', r_channel)
generator.add_band('green', g_channel)
generator.add_band('blue', b_channel)

print(f"\nEnhancements to apply:")
print(f"  Contrast enhancement: {should_enhance_contrast}")
print(f"  Detail enhancement: {should_enhance_detail}")
print(f"  Star enhancement: {should_enhance_stars}")
print(f"  Color balancing: True")

enhanced = generator.create_rgb_composite(
    r_band='red',
    g_band='green',
    b_band='blue',
    enhance_contrast=should_enhance_contrast,
    enhance_details=should_enhance_detail,
    enhance_stars=should_enhance_stars,
    color_balance=True
)

# Step 5: AI assessment of enhanced version
print("\n" + "="*80)
print("Step 5: AI Assessment of Enhanced Image")
print("="*80)

print("\nAnalyzing enhanced composite quality...")
try:
    enhanced_assessment = compositor.optimize_composite_parameters(enhanced)

    print(f"\nEnhanced Image Assessment:")
    print(f"  Overall Quality: {enhanced_assessment.get('overall_quality', 'N/A')}")
    print(f"  Contrast: {enhanced_assessment.get('contrast_adjustment', 'N/A')}")
    print(f"  Brightness: {enhanced_assessment.get('brightness_adjustment', 'N/A')}")
    print(f"  Saturation: {enhanced_assessment.get('color_saturation', 'N/A')}")

    recommendations = enhanced_assessment.get('specific_recommendations', [])
    if recommendations:
        print(f"\n  Further Suggestions:")
        for rec in recommendations:
            print(f"    - {rec}")
except Exception as e:
    print(f"  AI assessment failed: {e}")

# Step 6: Save result
print("\n" + "="*80)
print("Step 6: Saving AI-Optimized Composite")
print("="*80)

generator.save_composite(enhanced, OUTPUT_FILE, quality=95)
print(f"\n✓ Saved: {OUTPUT_FILE}")
print(f"  Size: {enhanced.shape[1]}x{enhanced.shape[0]} pixels")

print("\n" + "="*80)
print("AI-GUIDED ENHANCEMENT COMPLETE")
print("="*80)
print(f"\nOutput: {OUTPUT_FILE}")
print("\nThe AI analyzed your image and applied targeted enhancements")
print("based on what it detected in the galaxy structure!")