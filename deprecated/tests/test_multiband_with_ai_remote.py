"""AI-guided composite using REMOTE transformers server via tunnel.

Instead of loading 11GB model locally, uses the remote server.

Prerequisites:
    1. Server running: python scripts/transformers_pipeline_server.py --port 5000
    2. SSH tunnel active: ssh -L 5000:localhost:5000 jakko@192.168.50.194

Usage:
    python test_multiband_with_ai_remote.py band1.fits band2.fits band3.fits [output.png]
"""

import sys
import io
import base64
from pathlib import Path
import numpy as np
import requests
from PIL import Image

from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from astro_vision_composer import (
    FITSImageProcessor,
    CompositeImageGenerator
)

# Server URL (via SSH tunnel)
SERVER_URL = "http://localhost:5000"


def analyze_image_remote(image_array, prompt):
    """Send image to remote server for analysis.

    Args:
        image_array: numpy array (H, W, 3) in 0-1 range
        prompt: text prompt for the model

    Returns:
        Generated text from model
    """
    # Convert numpy array to PNG bytes
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8, mode='RGB')

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()

    # Send to server
    image_b64 = base64.b64encode(image_bytes).decode()

    response = requests.post(
        f"{SERVER_URL}/generate",
        json={
            "image_base64": image_b64,
            "prompt": prompt,
            "max_tokens": 300
        },
        headers={"Content-Type": "application/json"},
        timeout=120
    )

    response.raise_for_status()
    result = response.json()
    return result["text"]


def check_server():
    """Check if remote server is accessible."""
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=5)
        info = response.json()
        print(f"✓ Remote server connected: {info['model']}")
        print(f"  Status: {info['status']}")
        print(f"  Model loaded: {info['pipeline_loaded']}")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print()
        print("Make sure:")
        print("  1. Server running: python scripts/transformers_pipeline_server.py --port 5000")
        print("  2. SSH tunnel active: ssh -L 5000:localhost:5000 jakko@192.168.50.194")
        return False


if len(sys.argv) < 4:
    print("Usage: python test_multiband_with_ai_remote.py <band1.fits> <band2.fits> <band3.fits> [output.png]")
    print("\nExample:")
    print('  python test_multiband_with_ai_remote.py g.fits r.fits i.fits ai_composite.png')
    sys.exit(1)

# Parse arguments
fits_files = sys.argv[1:-1] if sys.argv[-1].endswith('.png') else sys.argv[1:]
output_file = sys.argv[-1] if sys.argv[-1].endswith('.png') else "ai_remote_composite.png"

print("="*80)
print("AI-GUIDED MULTI-BAND FITS COMPOSITE (REMOTE)")
print("="*80)
print("\nUsing REMOTE transformers server via SSH tunnel")
print(f"Server: {SERVER_URL}")

# Check server connectivity
print("\n" + "="*80)
print("Checking remote server connection")
print("="*80)
if not check_server():
    sys.exit(1)

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

# Step 2: Create quick preview for AI analysis
print("\n" + "="*80)
print("Step 2: Creating preview for AI band analysis")
print("="*80)

# Use simple mapping for preview
band_names = list(all_bands.keys())
preview_mapping = {
    'red': band_names[0] if len(band_names) > 0 else None,
    'green': band_names[1] if len(band_names) > 1 else None,
    'blue': band_names[2] if len(band_names) > 2 else None,
}

generator = CompositeImageGenerator()
for name, data in all_bands.items():
    generator.add_band(name, data)

# Create simple preview composite
r = all_bands[preview_mapping['red']]
g = all_bands[preview_mapping['green']]
b = all_bands[preview_mapping['blue']]
preview_rgb = np.dstack([r, g, b])

print("\nSending preview to AI for band analysis...")
print("(This takes ~30-60s on first request)")

try:
    prompt = f"""Analyze this astronomical image composite made from {len(all_bands)} bands.
Available bands: {', '.join(all_bands.keys())}
Current mapping: R={preview_mapping['red']}, G={preview_mapping['green']}, B={preview_mapping['blue']}

Recommend the optimal RGB channel mapping based on:
1. Wavelength order (longer wavelengths -> red channel)
2. Visual appearance and contrast
3. Scientific accuracy

Provide your recommendation in this format:
Red channel: [band_name]
Green channel: [band_name]
Blue channel: [band_name]"""

    ai_response = analyze_image_remote(preview_rgb, prompt)

    print("\nAI Analysis:")
    print("-" * 80)
    print(ai_response)
    print("-" * 80)

    # Try to parse mapping from response (simple parsing)
    mapping = preview_mapping.copy()  # Default to preview if parsing fails

    for line in ai_response.split('\n'):
        line_lower = line.lower()
        if 'red channel' in line_lower:
            for band in all_bands.keys():
                if band.lower() in line_lower:
                    mapping['red'] = band
                    break
        elif 'green channel' in line_lower:
            for band in all_bands.keys():
                if band.lower() in line_lower:
                    mapping['green'] = band
                    break
        elif 'blue channel' in line_lower:
            for band in all_bands.keys():
                if band.lower() in line_lower:
                    mapping['blue'] = band
                    break

    print(f"\nParsed mapping:")
    print(f"  Red:   {mapping['red']}")
    print(f"  Green: {mapping['green']}")
    print(f"  Blue:  {mapping['blue']}")

except Exception as e:
    print(f"AI analysis failed: {e}")
    print("Using default mapping...")
    mapping = preview_mapping

# Step 3: Create final composite with AI-recommended mapping
print("\n" + "="*80)
print("Step 3: Creating final composite")
print("="*80)

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

# Step 4: AI Quality Assessment
print("\n" + "="*80)
print("Step 4: AI Quality Assessment")
print("="*80)

print("\nSending final composite to AI for quality assessment...")

try:
    assessment_prompt = """Analyze this processed astronomical image composite and evaluate:

1. Background darkness - is the sky properly dark/black?
2. Star definition - are stars sharp and well-defined?
3. Contrast - is there good separation between features?
4. Color balance - do the colors look natural for astronomy?
5. Noise/artifacts - any visible processing artifacts?
6. Overall quality - rate 1-10

Provide specific recommendations for improvement."""

    quality_analysis = analyze_image_remote(composite, assessment_prompt)

    print("\nAI Quality Assessment:")
    print("=" * 80)
    print(quality_analysis)
    print("=" * 80)

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

print("\n" + "="*80)
print("AI-GUIDED COMPOSITE COMPLETE (REMOTE)")
print("="*80)
print("\nThe AI on the remote server has analyzed your data!")