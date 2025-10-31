"""Handle M31 FITS file that's already an RGB composite.

This FITS file contains a single HDU with RGB data (H, W, 3),
not separate band HDUs. We'll extract the RGB channels and
apply enhancements.
"""

import numpy as np
from astropy.io import fits
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import CompositeImageGenerator

# Input/output paths
FITS_FILE = r"C:\Users\jacks\experiments\PycharmProjects\Mission Control\examples\output\m31_panstarrs_rgb.fits"
OUTPUT_SIMPLE = "m31_simple_extract.png"
OUTPUT_ENHANCED = "m31_enhanced.png"

print("="*80)
print("M31 ANDROMEDA - RGB FITS ENHANCEMENT")
print("="*80)

# Step 1: Load RGB FITS data
print("\nStep 1: Loading RGB FITS data...")
with fits.open(FITS_FILE) as hdul:
    # Data is already RGB in shape (H, W, 3)
    rgb_data = hdul[0].data
    print(f"  Loaded data shape: {rgb_data.shape}")
    print(f"  Data type: {rgb_data.dtype}")
    print(f"  Value range: {rgb_data.min()} - {rgb_data.max()}")

# Step 2: Convert to correct format
print("\nStep 2: Converting to normalized float format...")

# FITS convention: (H, W, C) but could be (C, H, W)
# Let's check which dimension is 3
if rgb_data.shape[0] == 3:
    # (C, H, W) -> (H, W, C)
    rgb_data = np.moveaxis(rgb_data, 0, -1)
    print(f"  Transposed to: {rgb_data.shape}")
elif rgb_data.shape[-1] == 3:
    # Already (H, W, C)
    print(f"  Already in correct format: {rgb_data.shape}")
else:
    print(f"  ERROR: Cannot find RGB dimension in shape {rgb_data.shape}")
    sys.exit(1)

# Normalize to 0-1 range
if rgb_data.dtype == np.uint8:
    rgb_normalized = rgb_data.astype(np.float32) / 255.0
else:
    rgb_normalized = rgb_data.astype(np.float32)
    rgb_normalized = (rgb_normalized - rgb_normalized.min()) / (rgb_normalized.max() - rgb_normalized.min())

print(f"  Normalized to 0-1 range")

# Step 3: Save simple extraction
print("\nStep 3: Saving simple extraction...")
rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
img = Image.fromarray(rgb_uint8, mode='RGB')
img.save(OUTPUT_SIMPLE)
print(f"  Saved: {OUTPUT_SIMPLE}")

# Step 4: Extract individual channels for enhancement
print("\nStep 4: Extracting RGB channels...")
r_channel = rgb_normalized[:, :, 0]
g_channel = rgb_normalized[:, :, 1]
b_channel = rgb_normalized[:, :, 2]

print(f"  Red channel shape: {r_channel.shape}")
print(f"  Green channel shape: {g_channel.shape}")
print(f"  Blue channel shape: {b_channel.shape}")

# Step 5: Apply enhancements
print("\nStep 5: Applying enhancements...")
print("  - Adaptive contrast enhancement (CLAHE)")
print("  - Detail enhancement (unsharp masking)")
print("  - Star highlighting")
print("  - Color balancing")

generator = CompositeImageGenerator()

# Add each channel as a "band"
generator.add_band('red', r_channel)
generator.add_band('green', g_channel)
generator.add_band('blue', b_channel)

# Create enhanced composite
enhanced = generator.create_rgb_composite(
    r_band='red',
    g_band='green',
    b_band='blue',
    enhance_contrast=True,
    enhance_details=True,
    enhance_stars=True,
    color_balance=True
)

# Step 6: Save enhanced version
print("\nStep 6: Saving enhanced composite...")
generator.save_composite(enhanced, OUTPUT_ENHANCED, quality=95)
print(f"  Saved: {OUTPUT_ENHANCED}")

# Show comparison stats
print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)

print(f"\nOriginal image:")
print(f"  Size: {rgb_data.shape[1]}x{rgb_data.shape[0]} pixels")
print(f"  Channels: RGB")
print(f"  File: {OUTPUT_SIMPLE}")

print(f"\nEnhanced image:")
print(f"  Size: {enhanced.shape[1]}x{enhanced.shape[0]} pixels")
print(f"  Enhancements: Contrast + Detail + Stars + Color Balance")
print(f"  File: {OUTPUT_ENHANCED}")

print("\n" + "="*80)
print("Compare the two images to see the enhancement effect!")
print("="*80)