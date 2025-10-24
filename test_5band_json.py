"""5-band analysis with STRUCTURED JSON responses from AI.

Much cleaner than parsing free text!

Usage:
    python test_5band_json.py <g_dir> <r_dir> <i_dir> <z_dir> <y_dir> [output.png]
"""

import sys
import json
from pathlib import Path
import numpy as np

from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from astro_vision_composer import FITSImageProcessor, CompositeImageGenerator
from astro_ai_client import AstroAIClient
from skimage import exposure
from scipy import ndimage

# Band metadata
BANDS = {
    'g': {'wavelength': 481, 'name': 'Green'},
    'r': {'wavelength': 617, 'name': 'Red'},
    'i': {'wavelength': 752, 'name': 'Near-IR'},
    'z': {'wavelength': 866, 'name': 'IR'},
    'y': {'wavelength': 962, 'name': 'Far-IR'},
}

if len(sys.argv) < 6:
    print("Usage: python test_5band_json.py <g_dir> <r_dir> <i_dir> <z_dir> <y_dir> [output.png]")
    sys.exit(1)

band_dirs = sys.argv[1:6]
output_file = sys.argv[6] if len(sys.argv) > 6 else "ai_5band_json.png"

print("="*80)
print("5-BAND AI ANALYSIS WITH STRUCTURED JSON")
print("="*80)

# Initialize AI client
client = AstroAIClient()

print("\nChecking AI server...")
if not client.check_server():
    print("✗ Server not accessible")
    print("\nMake sure:")
    print("  1. Server: python scripts/transformers_pipeline_server.py --port 5000")
    print("  2. Tunnel: ssh -L 5000:localhost:5000 jakko@192.168.50.194")
    sys.exit(1)
print("✓ AI server connected")

# Step 1: Load bands
print("\n" + "="*80)
print("Step 1: Loading 5 bands")
print("="*80)

all_bands = {}
band_order = ['g', 'r', 'i', 'z', 'y']

for band_char, band_dir in zip(band_order, band_dirs):
    fits_file = list(Path(band_dir).glob("*.fits"))[0]

    print(f"\n{band_char.upper()}-band ({BANDS[band_char]['wavelength']}nm)...")

    with FITSImageProcessor(str(fits_file)) as fits_proc:
        bands = fits_proc.extract_bands()
        for band_name, band_data in bands.items():
            normalized = fits_proc.normalize_band(band_name, method='zscale')
            all_bands[band_char] = normalized
            print(f"  ✓ Loaded: median={np.nanmedian(normalized):.4f}")

# Step 2: AI analysis of each band (with JSON!)
print("\n" + "="*80)
print("Step 2: AI Analysis of Each Band (JSON responses)")
print("="*80)

band_analyses = {}

for band_char in band_order:
    print(f"\nAnalyzing {band_char}-band...")

    result = client.analyze_band(
        all_bands[band_char],
        BANDS[band_char]['name'],
        BANDS[band_char]['wavelength']
    )

    if result.get('json'):
        band_analyses[band_char] = result
        data = result['json']
        print(f"  Quality: {data.get('quality', 'N/A')}/10")
        print(f"  Noise: {data.get('noise_level', 'N/A')}")
        print(f"  Best channel: {data.get('best_rgb_channel', 'N/A')}")
        print(f"  Features: {', '.join(data.get('features_visible', []))}")
    else:
        print(f"  ✗ Failed to get JSON: {result.get('parse_error')}")
        band_analyses[band_char] = result

# Step 3: AI RGB mapping recommendation (with JSON!)
print("\n" + "="*80)
print("Step 3: AI RGB Mapping Recommendation (JSON)")
print("="*80)

# Create preview
preview = np.dstack([all_bands['i'], all_bands['r'], all_bands['g']])

print("\nAsking AI for optimal RGB mapping...")
mapping_result = client.recommend_rgb_mapping(band_analyses, preview)

if mapping_result.get('json'):
    mapping_data = mapping_result['json']
    mapping = {
        'red': mapping_data.get('red', 'i'),
        'green': mapping_data.get('green', 'r'),
        'blue': mapping_data.get('blue', 'g')
    }

    print(f"\n✓ AI Recommendation:")
    print(f"  RED:   {mapping['red']}-band")
    print(f"  GREEN: {mapping['green']}-band")
    print(f"  BLUE:  {mapping['blue']}-band")
    print(f"\n  Reasoning: {mapping_data.get('reasoning', 'N/A')}")
else:
    print(f"  ✗ Failed to parse JSON, using default i-r-g")
    mapping = {'red': 'i', 'green': 'r', 'blue': 'g'}

# Step 4: Create composite
print("\n" + "="*80)
print("Step 4: Creating Composite")
print("="*80)

r = all_bands[mapping['red']].copy()
g = all_bands[mapping['green']].copy()
b = all_bands[mapping['blue']].copy()

processing_steps = {
    "zscale": "Adaptive histogram normalization",
    "clahe": "clip_limit=0.03",
    "unsharp_mask": "sigma=2, amount=0.5",
    "star_enhancement": "99th percentile, boost=1.3x",
    "luminance_masking": "5th percentile -> black"
}

print("\nApplying processing pipeline...")
print("  1. CLAHE...")
r = exposure.equalize_adapthist(r, clip_limit=0.03)
g = exposure.equalize_adapthist(g, clip_limit=0.03)
b = exposure.equalize_adapthist(b, clip_limit=0.03)

print("  2. Unsharp mask...")
r = r + (r - ndimage.gaussian_filter(r, 2)) * 0.5
g = g + (g - ndimage.gaussian_filter(g, 2)) * 0.5
b = b + (b - ndimage.gaussian_filter(b, 2)) * 0.5
r, g, b = np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)

print("  3. Star boost...")
r = np.where(r > np.percentile(r, 99), np.clip(r * 1.3, 0, 1), r)
g = np.where(g > np.percentile(g, 99), np.clip(g * 1.3, 0, 1), g)
b = np.where(b > np.percentile(b, 99), np.clip(b * 1.3, 0, 1), b)

rgb = np.dstack([r, g, b])

print("  4. Luminance masking...")
lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
rgb[lum < np.nanpercentile(lum, 5)] = 0

# Step 5: AI quality assessment (with JSON!)
print("\n" + "="*80)
print("Step 5: AI Quality Assessment (JSON)")
print("="*80)

print("\nSending composite to AI for evaluation...")
assessment = client.assess_processing(rgb, processing_steps)

if assessment.get('json'):
    data = assessment['json']

    print("\n" + "="*80)
    print("STRUCTURED AI ASSESSMENT")
    print("="*80)

    print(f"\nOverall Quality: {data.get('overall_quality', 'N/A')}/10")

    print("\nProcessing Steps Evaluation:")
    for step in ['zscale', 'clahe', 'unsharp_mask', 'star_enhancement', 'luminance_masking']:
        if step in data:
            step_data = data[step]
            effective = "✓" if step_data.get('effective') else "✗"
            print(f"  {effective} {step}:")
            for k, v in step_data.items():
                if k != 'effective':
                    print(f"      {k}: {v}")

    if data.get('major_issues'):
        print(f"\nMajor Issues:")
        for issue in data['major_issues']:
            print(f"  - {issue}")

    if data.get('recommendations'):
        print(f"\nRecommendations:")
        for rec in data['recommendations']:
            print(f"  - {rec}")

    print("\n" + "="*80)

    # Save JSON results
    json_file = Path(output_file).with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump({
            'band_analyses': band_analyses,
            'rgb_mapping': mapping_result,
            'quality_assessment': assessment
        }, f, indent=2)

    print(f"\n✓ JSON results saved: {json_file}")

else:
    print(f"  ✗ Failed to get JSON assessment")

# Step 6: Save composite
print("\n" + "="*80)
print("Step 6: Saving Composite")
print("="*80)

generator = CompositeImageGenerator()
generator.save_composite(rgb, output_file, quality=95)

print(f"\n✓ Saved: {Path(output_file).resolve()}")

print("\n" + "="*80)
print("COMPLETE - Structured JSON Analysis")
print("="*80)
print(f"\nOutputs:")
print(f"  Image: {output_file}")
print(f"  JSON:  {json_file}")