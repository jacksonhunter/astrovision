"""Test 5-band analysis with proper few-shot JSON learning.

Uses context_building strategy:
- Bands 1-2: Explicit JSON format
- Bands 3-5: Vague prompts (rely on examples)
"""

import sys
from pathlib import Path
import numpy as np

from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from astro_vision_composer import FITSImageProcessor
from astro_ai_client_v2 import AstroAIClientV2

# Band metadata
BAND_INFO = {
    'g': {'wavelength': 481, 'name': 'Green'},
    'r': {'wavelength': 617, 'name': 'Red'},
    'i': {'wavelength': 752, 'name': 'Near-IR'},
    'z': {'wavelength': 866, 'name': 'IR'},
    'y': {'wavelength': 962, 'name': 'Far-IR'},
}

if len(sys.argv) < 6:
    print("Usage: python test_5band_v2.py <g_dir> <r_dir> <i_dir> <z_dir> <y_dir>")
    sys.exit(1)

band_dirs = sys.argv[1:6]

print("="*80)
print("5-BAND ANALYSIS WITH PROPER FEW-SHOT LEARNING")
print("="*80)

# Initialize client
client = AstroAIClientV2()

print("\nChecking AI server...")
if not client.check_server():
    print("✗ Server not accessible")
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

    print(f"\n{band_char.upper()}-band ({BAND_INFO[band_char]['wavelength']}nm)...")

    with FITSImageProcessor(str(fits_file)) as fits_proc:
        bands = fits_proc.extract_bands()
        for band_name, band_data in bands.items():
            normalized = fits_proc.normalize_band(band_name, method='zscale')
            all_bands[band_char] = normalized
            print(f"  ✓ Loaded: median={np.nanmedian(normalized):.4f}")

# Step 2: AI analysis with few-shot learning
print("\n" + "="*80)
print("Step 2: AI Analysis (Context Building Strategy)")
print("="*80)

print("\nFew-shot pattern:")
print("  Band 1-2: EXPLICIT JSON format")
print("  Band 3-5: VAGUE prompts (rely on examples)\n")

print("Analyzing all 5 bands...")

analyses = client.analyze_bands(all_bands, BAND_INFO)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

all_success = True

for analysis in analyses:
    band = analysis['band']
    print(f"\n{band.upper()}-band:")

    if analysis['success']:
        data = analysis['json']
        print(f"  ✓ JSON parsed successfully")
        print(f"    Quality: {data.get('quality', 'N/A')}/10")
        print(f"    Noise: {data.get('noise_level', 'N/A')}")
        print(f"    Brightness: {data.get('brightness', 'N/A')}")
        print(f"    Features: {', '.join(data.get('features_visible', []))}")
        if data.get('notes'):
            print(f"    Notes: {data.get('notes')[:60]}...")
    else:
        print(f"  ✗ Failed to parse JSON")
        print(f"    Raw: {analysis['raw_text'][:100]}...")
        all_success = False

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

# Check for variation in responses
if all_success:
    qualities = [a['json'].get('quality') for a in analyses if a['success']]
    noises = [a['json'].get('noise_level') for a in analyses if a['success']]
    brightnesses = [a['json'].get('brightness') for a in analyses if a['success']]

    print(f"\n✓ All {len(analyses)} bands parsed successfully")
    print(f"\nQuality scores: {qualities}")
    print(f"Noise levels: {noises}")
    print(f"Brightness: {brightnesses}")

    # Check for variation (not all identical)
    if len(set(qualities)) > 1 or len(set(noises)) > 1 or len(set(brightnesses)) > 1:
        print("\n🎉 SUCCESS!")
        print("  ✓ AI analyzing actual image content")
        print("  ✓ Not just copying examples")
        print("  ✓ Variation in responses detected")
    else:
        print("\n⚠ WARNING:")
        print("  All responses identical - AI may be copying pattern without analyzing")
else:
    print(f"\n✗ FAILED: {sum(not a['success'] for a in analyses)} bands failed to parse")

print("\n" + "="*80)
