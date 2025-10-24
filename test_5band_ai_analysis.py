"""Comprehensive AI analysis of all 5 PanSTARRS bands (g, r, i, z, y).

Gets AI feedback on:
- Color mapping (which bands -> RGB channels)
- Zscale normalization quality
- CLAHE parameters
- Star enhancement
- Luminance masking
- Overall processing recommendations

Usage:
    python test_5band_ai_analysis.py <g_dir> <r_dir> <i_dir> <z_dir> <y_dir> [output.png]

Example:
    python test_5band_ai_analysis.py \
        "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.g" \
        "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.r" \
        "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.i" \
        "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.z" \
        "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.y" \
        ai_5band_analysis.png
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

from astro_vision_composer import FITSImageProcessor, CompositeImageGenerator
from skimage import exposure
from scipy import ndimage

SERVER_URL = "http://localhost:5000"

# PanSTARRS wavelengths for reference
BAND_INFO = {
    'g': {'wavelength': 481, 'name': 'Green (g-band)', 'typical_rgb': 'blue'},
    'r': {'wavelength': 617, 'name': 'Red (r-band)', 'typical_rgb': 'green'},
    'i': {'wavelength': 752, 'name': 'Near-IR (i-band)', 'typical_rgb': 'red'},
    'z': {'wavelength': 866, 'name': 'IR (z-band)', 'typical_rgb': 'red'},
    'y': {'wavelength': 962, 'name': 'Far-IR (y-band)', 'typical_rgb': 'red'},
}


def analyze_image_remote(image_array, prompt, max_tokens=500):
    """Send image to remote AI for analysis."""
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8, mode='RGB')

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()

    image_b64 = base64.b64encode(image_bytes).decode()

    response = requests.post(
        f"{SERVER_URL}/generate",
        json={
            "image_base64": image_b64,
            "prompt": prompt,
            "max_tokens": max_tokens
        },
        headers={"Content-Type": "application/json"},
        timeout=120
    )

    response.raise_for_status()
    return response.json()["text"]


def check_server():
    """Check remote server connectivity."""
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=5)
        info = response.json()
        print(f"✓ Remote AI server connected")
        print(f"  Model: {info['model']}")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("\nMake sure:")
        print("  1. Server running: python scripts/transformers_pipeline_server.py --port 5000")
        print("  2. SSH tunnel active: ssh -L 5000:localhost:5000 jakko@192.168.50.194")
        return False


if len(sys.argv) < 6:
    print("Usage: python test_5band_ai_analysis.py <g_dir> <r_dir> <i_dir> <z_dir> <y_dir> [output.png]")
    print("\nProvide paths to the 5 band directories (each containing .stk.X.unconv.fits file)")
    sys.exit(1)

band_dirs = sys.argv[1:6]
output_file = sys.argv[6] if len(sys.argv) > 6 else "ai_5band_analysis.png"

print("="*80)
print("5-BAND AI-GUIDED ANALYSIS (PanSTARRS g, r, i, z, y)")
print("="*80)
print("\nRemote AI will analyze:")
print("  ✓ Individual band characteristics")
print("  ✓ Optimal RGB color mapping")
print("  ✓ Zscale normalization quality")
print("  ✓ Processing parameters (CLAHE, stars, masking)")

# Check server
print("\n" + "="*80)
print("Checking remote AI server")
print("="*80)
if not check_server():
    sys.exit(1)

# Load all 5 bands
print("\n" + "="*80)
print("Step 1: Loading all 5 PanSTARRS bands")
print("="*80)

all_bands = {}
band_names_ordered = ['g', 'r', 'i', 'z', 'y']

for band_char, band_dir in zip(band_names_ordered, band_dirs):
    band_path = Path(band_dir)

    # Find the .fits file in the directory
    fits_files = list(band_path.glob("*.fits"))
    if not fits_files:
        print(f"✗ No FITS file found in {band_dir}")
        sys.exit(1)

    fits_file = fits_files[0]

    print(f"\n{band_char.upper()}-band ({BAND_INFO[band_char]['wavelength']} nm):")
    print(f"  File: {fits_file.name}")

    with FITSImageProcessor(str(fits_file)) as fits_proc:
        bands = fits_proc.extract_bands()

        for band_name, band_data in bands.items():
            # Normalize with zscale
            normalized = fits_proc.normalize_band(band_name, method='zscale')

            print(f"  Raw: min={np.nanmin(band_data):.2e}, max={np.nanmax(band_data):.2e}, median={np.nanmedian(band_data):.2e}")
            print(f"  Zscale: median={np.nanmedian(normalized):.4f}, mean={np.nanmean(normalized):.4f}")

            all_bands[f"{band_char}_band"] = normalized

print(f"\n✓ Loaded {len(all_bands)} bands: {', '.join(all_bands.keys())}")

# Step 2: AI analysis of individual bands
print("\n" + "="*80)
print("Step 2: AI Analysis of Individual Bands")
print("="*80)

print("\nSending each band to AI for analysis...")
print("(This will take 2-3 minutes total)")

individual_analyses = {}

for band_char in band_names_ordered:
    band_key = f"{band_char}_band"
    band_data = all_bands[band_key]

    # Create grayscale preview
    preview_rgb = np.dstack([band_data, band_data, band_data])

    prompt = f"""Analyze this {BAND_INFO[band_char]['name']} astronomical image band ({BAND_INFO[band_char]['wavelength']} nm wavelength).

Evaluate:
1. Data quality - noise level, dynamic range, saturation
2. Feature visibility - what astronomical features are visible?
3. Brightness distribution - histogram characteristics
4. Suitability for RGB composite

Provide concise technical assessment (2-3 sentences)."""

    print(f"\n{band_char.upper()}-band analysis...")
    try:
        analysis = analyze_image_remote(preview_rgb, prompt, max_tokens=200)
        individual_analyses[band_char] = analysis
        print(f"  {analysis[:100]}...")
    except Exception as e:
        print(f"  Error: {e}")
        individual_analyses[band_char] = "Analysis failed"

# Step 3: AI recommendation for RGB mapping
print("\n" + "="*80)
print("Step 3: AI Recommendation for RGB Color Mapping")
print("="*80)

# Create simple composite for AI to see all bands
print("\nCreating multi-band preview composite...")

# Default mapping for preview
preview_composite = np.dstack([
    all_bands['i_band'],  # R
    all_bands['r_band'],  # G
    all_bands['g_band']   # B
])

wavelength_info = "\n".join([f"{k}: {v['wavelength']}nm" for k, v in BAND_INFO.items()])

prompt = f"""You are analyzing a multi-band astronomical image with 5 available bands:

{wavelength_info}

Individual band analyses:
{chr(10).join([f"{k.upper()}: {v}" for k, v in individual_analyses.items()])}

Recommend the optimal RGB channel mapping following these principles:
1. Longer wavelengths -> redder channels (wavelength order)
2. Best signal-to-noise -> luminance contribution
3. Maximize color separation for scientific visualization

Provide your recommendation in this EXACT format:
RED: [band letter]
GREEN: [band letter]
BLUE: [band letter]

Then explain your reasoning (2-3 sentences)."""

print("Asking AI for optimal RGB mapping...")
try:
    mapping_response = analyze_image_remote(preview_composite, prompt, max_tokens=300)
    print("\nAI Response:")
    print("="*80)
    print(mapping_response)
    print("="*80)

    # Parse mapping
    mapping = {}
    for line in mapping_response.split('\n'):
        if 'RED:' in line.upper():
            for band in band_names_ordered:
                if band in line.lower():
                    mapping['red'] = f"{band}_band"
                    break
        elif 'GREEN:' in line.upper():
            for band in band_names_ordered:
                if band in line.lower():
                    mapping['green'] = f"{band}_band"
                    break
        elif 'BLUE:' in line.upper():
            for band in band_names_ordered:
                if band in line.lower():
                    mapping['blue'] = f"{band}_band"
                    break

    # Fallback if parsing failed
    if len(mapping) != 3:
        print("\nCouldn't parse AI mapping, using standard i-r-g mapping")
        mapping = {'red': 'i_band', 'green': 'r_band', 'blue': 'g_band'}

except Exception as e:
    print(f"AI mapping failed: {e}")
    mapping = {'red': 'i_band', 'green': 'r_band', 'blue': 'g_band'}

print(f"\nFinal RGB Mapping:")
print(f"  RED:   {mapping['red']}")
print(f"  GREEN: {mapping['green']}")
print(f"  BLUE:  {mapping['blue']}")

# Step 4: Create composite with full processing pipeline
print("\n" + "="*80)
print("Step 4: Creating Composite with Full Processing Pipeline")
print("="*80)

generator = CompositeImageGenerator()
for name, data in all_bands.items():
    generator.add_band(name, data)

# Get channels
r = all_bands[mapping['red']].copy()
g = all_bands[mapping['green']].copy()
b = all_bands[mapping['blue']].copy()

print("\nApplying processing pipeline:")
print("  1. CLAHE contrast (clip_limit=0.03)")
r_clahe = exposure.equalize_adapthist(r, clip_limit=0.03)
g_clahe = exposure.equalize_adapthist(g, clip_limit=0.03)
b_clahe = exposure.equalize_adapthist(b, clip_limit=0.03)

print("  2. Unsharp mask detail enhancement")
r_detail = r_clahe + (r_clahe - ndimage.gaussian_filter(r_clahe, 2)) * 0.5
g_detail = g_clahe + (g_clahe - ndimage.gaussian_filter(g_clahe, 2)) * 0.5
b_detail = b_clahe + (b_clahe - ndimage.gaussian_filter(b_clahe, 2)) * 0.5

r_detail = np.clip(r_detail, 0, 1)
g_detail = np.clip(g_detail, 0, 1)
b_detail = np.clip(b_detail, 0, 1)

print("  3. Star boost (99th percentile)")
r_thresh = np.percentile(r_detail, 99)
g_thresh = np.percentile(g_detail, 99)
b_thresh = np.percentile(b_detail, 99)

r_final = np.where(r_detail > r_thresh, np.clip(r_detail * 1.3, 0, 1), r_detail)
g_final = np.where(g_detail > g_thresh, np.clip(g_detail * 1.3, 0, 1), g_detail)
b_final = np.where(b_detail > b_thresh, np.clip(b_detail * 1.3, 0, 1), b_detail)

rgb_enhanced = np.dstack([r_final, g_final, b_final])

print("  4. Luminance masking (darkest 5%)")
luminance = 0.2126 * rgb_enhanced[:, :, 0] + 0.7152 * rgb_enhanced[:, :, 1] + 0.0722 * rgb_enhanced[:, :, 2]
lum_threshold = np.nanpercentile(luminance, 5)
dark_mask = luminance < lum_threshold

rgb_final = rgb_enhanced.copy()
rgb_final[dark_mask] = 0

print(f"\n  Result: median={np.nanmedian(rgb_final):.4f}, mean={np.nanmean(rgb_final):.4f}")

# Step 5: Comprehensive AI quality assessment
print("\n" + "="*80)
print("Step 5: Comprehensive AI Quality Assessment")
print("="*80)

print("\nSending final composite to AI for detailed evaluation...")

assessment_prompt = f"""Analyze this processed astronomical composite image. Evaluate each processing step:

Processing pipeline applied:
1. Zscale normalization - adaptive histogram scaling
2. CLAHE (clip_limit=0.03) - adaptive contrast enhancement
3. Unsharp masking (sigma=2, amount=0.5) - detail enhancement
4. Star boost (99th percentile, 1.3x) - brighten brightest stars
5. Luminance masking (5th percentile -> black) - darken background

For each step, assess:
- Effectiveness (working well / needs adjustment)
- Recommended parameter changes
- Visual artifacts or issues

Also evaluate:
- RGB color mapping quality
- Overall scientific/aesthetic balance
- Specific recommendations for improvement

Provide structured feedback with specific parameter suggestions."""

try:
    assessment = analyze_image_remote(rgb_final, assessment_prompt, max_tokens=600)

    print("\n" + "="*80)
    print("AI COMPREHENSIVE ASSESSMENT")
    print("="*80)
    print(assessment)
    print("="*80)

except Exception as e:
    print(f"AI assessment failed: {e}")

# Step 6: Save
print("\n" + "="*80)
print("Step 6: Saving Composite")
print("="*80)

generator.save_composite(rgb_final, output_file, quality=95)

output_path = Path(output_file).resolve()
print(f"\n✓ Saved: {output_path}")

print("\n" + "="*80)
print("5-BAND AI ANALYSIS COMPLETE")
print("="*80)
print(f"\nBands used: {', '.join(all_bands.keys())}")
print(f"RGB Mapping: R={mapping['red']}, G={mapping['green']}, B={mapping['blue']}")
print(f"Output: {output_path.name}")