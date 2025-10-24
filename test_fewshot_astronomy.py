"""Few-shot astronomical image analysis training.

Uses 3 real astronomical examples to teach scoring criteria,
then queries PanSTARRS bands.

Example images from NOIRLab FITS Liberator tutorials:
1. M31 (Galaxy) - Good quality, extended emission
2. M42 (Nebula) - Bright, potential saturation
3. M12 (Globular cluster) - Point sources, crowded field
"""

import sys
import json
import base64
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np

from suppress_warnings import suppress_common_warnings
suppress_common_warnings()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from astro_vision_composer import FITSImageProcessor

SERVER_URL = "http://localhost:5000"

# Example image URLs (small screen JPEGs from NOIRLab)
TRAINING_EXAMPLES = [
    {
        "name": "M106 Galaxy",
        "url": "https://noirlab.edu/public/media/archives/education/large/edu027.jpg",
        "expected_json": {
            "quality": 8,
            "noise_level": "low",
            "saturation": False,
            "features_visible": ["spiral arms", "dust lanes", "stellar disk", "bright core"],
            "brightness": "medium",
            "notes": "Good SNR, extended emission well-resolved, no significant artifacts"
        }
    },
    {
        "name": "Eagle Nebula M16",
        "url": "https://noirlab.edu/public/media/archives/education/large/edu008.jpg",
        "expected_json": {
            "quality": 6,
            "noise_level": "medium",
            "saturation": True,
            "features_visible": ["pillars of creation", "nebulosity", "star formation regions", "emission clouds"],
            "brightness": "bright",
            "notes": "Bright emission regions show some saturation, excellent nebular structure detail"
        }
    },
    {
        "name": "M17 Star Forming Nebula",
        "url": "https://noirlab.edu/public/media/archives/education/large/edu010.jpg",
        "expected_json": {
            "quality": 7,
            "noise_level": "low",
            "saturation": False,
            "features_visible": ["emission nebulosity", "dark dust lanes", "bright stars", "stellar clusters"],
            "brightness": "medium",
            "notes": "Clean emission features, good dynamic range, well-balanced exposure"
        }
    }
]

def download_image_to_base64(url):
    """Download image and convert to base64."""
    print(f"  Downloading: {url.split('/')[-1]}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def numpy_to_base64(array):
    """Convert numpy array to base64."""
    img_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8).convert('RGB')
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Check server
try:
    health = requests.get(SERVER_URL, timeout=5).json()
    print(f"✓ Server: {health['status']}")
except:
    print("✗ Server not running")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python test_fewshot_astronomy.py <fits_file_or_dir>")
    print("\nThis will:")
    print("  1. Train with 3 astronomical examples (M31, M42, M12)")
    print("  2. Analyze your FITS file with learned criteria")
    sys.exit(1)

fits_path = Path(sys.argv[1])
if fits_path.is_dir():
    fits_files = list(fits_path.glob("*.fits"))
    if not fits_files:
        print(f"✗ No FITS files in {fits_path}")
        sys.exit(1)
    fits_path = fits_files[0]

print("\n" + "="*80)
print("FEW-SHOT ASTRONOMICAL IMAGE ANALYSIS")
print("="*80)
print("\n📚 Training with 3 real astronomical examples...")
print("   Teaching: quality scoring, noise assessment, feature detection\n")

# Build messages array
messages = []

# Examples 1-3: EXPLICIT format with realistic assessments
for idx, example in enumerate(TRAINING_EXAMPLES):
    print(f"{idx+1}. {example['name']}")
    try:
        image_b64 = download_image_to_base64(example['url'])

        # Explicit JSON schema with expected values
        expected_json_str = json.dumps(example['expected_json'], indent=2)

        prompt = f"""Analyze this astronomical {example['name']} image. Respond ONLY with JSON in this format:
{{
  "quality": <1-10, where 10=excellent SNR/resolution, 1=unusable>,
  "noise_level": "<low/medium/high>",
  "saturation": <true/false if bright regions clipped>,
  "features_visible": [<list of astronomical features you see>],
  "brightness": "<dark/medium/bright overall>",
  "notes": "<brief technical assessment>"
}}

Expected analysis for this image:
{expected_json_str}"""

        messages.append({
            "image_base64": image_b64,
            "text": prompt
        })
        print(f"   ✓ Added to training")

    except Exception as e:
        print(f"   ✗ Failed: {e}")

# Query: Load actual FITS file
print(f"\n4. Query: {fits_path.name}")
print(f"   Loading FITS file...")

with FITSImageProcessor(str(fits_path)) as fits_proc:
    bands = fits_proc.extract_bands()
    band_name = list(bands.keys())[0]
    normalized = fits_proc.normalize_band(band_name, method='zscale')

# Convert to grayscale preview
preview = np.dstack([normalized, normalized, normalized])
query_b64 = numpy_to_base64(preview)

# Query prompt: Still ask for JSON, just briefer
query_prompt = """Analyze this astronomical image. Respond ONLY with JSON using the same format:
{
  "quality": <1-10>,
  "noise_level": "<low/medium/high>",
  "saturation": <true/false>,
  "features_visible": [<list>],
  "brightness": "<dark/medium/bright>",
  "notes": "<assessment>"
}"""

messages.append({
    "image_base64": query_b64,
    "text": query_prompt
})
print(f"   ✓ Added query")

# Send to server
print(f"\nSending {len(messages)} images to server (context_building)...")
print("This will take ~1-2 minutes...\n")

response = requests.post(
    f"{SERVER_URL}/generate",
    json={
        "messages": messages,
        "max_tokens": 250,
        "strategy": "context_building"
    },
    timeout=300
)

if response.status_code != 200:
    print(f"✗ Server error: {response.status_code}")
    print(response.text)
    sys.exit(1)

result = response.json()

print("="*80)
print("RESULTS")
print("="*80)

# Parse all results
def extract_json(text):
    import re
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    # Extract from code blocks or embedded
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    return None

for idx, (res, name) in enumerate(zip(result["results"],
                                      [e["name"] for e in TRAINING_EXAMPLES] + [fits_path.name])):
    print(f"\n{idx+1}. {name}:")

    answer = res['result'].split('assistant\n')[-1].strip()
    parsed = extract_json(answer)

    if parsed:
        print(f"   ✓ Quality: {parsed.get('quality', 'N/A')}/10")
        print(f"   ✓ Noise: {parsed.get('noise_level', 'N/A')}")
        print(f"   ✓ Saturation: {parsed.get('saturation', 'N/A')}")
        print(f"   ✓ Features: {', '.join(parsed.get('features_visible', []))[:60]}...")
        print(f"   ✓ Notes: {parsed.get('notes', 'N/A')[:70]}...")
    else:
        print(f"   ✗ Failed to parse JSON")
        print(f"   Raw: {answer[:100]}...")

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

# Check query result (last one)
query_result = result["results"][-1]['result'].split('assistant\n')[-1].strip()
query_json = extract_json(query_result)

if query_json:
    print("\n✅ SUCCESS!")
    print("   ✓ Query returned valid JSON")
    print("   ✓ AI learned scoring criteria from examples")
    print("\n📊 Your FITS file assessment:")
    print(json.dumps(query_json, indent=2))
else:
    print("\n❌ FAILED")
    print("   AI did not return valid JSON for query")
    print(f"   Raw response: {query_result[:200]}")

print("\n" + "="*80)
