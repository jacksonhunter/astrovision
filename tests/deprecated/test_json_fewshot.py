"""Test few-shot JSON responses using upgraded server API.

Uses context_building strategy to teach JSON format with examples.
"""

import sys
import json
import base64
import requests
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

SERVER_URL = "http://localhost:5000"

def numpy_to_base64(array):
    """Convert numpy array to base64 PNG."""
    img_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8).convert('RGB')
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Check server
try:
    health = requests.get(SERVER_URL, timeout=5).json()
    print(f"✓ Server: {health['status']}")
    print(f"✓ Model: {health['model']}")
except:
    print("✗ Server not running on localhost:5000")
    sys.exit(1)

print("\n" + "="*80)
print("Testing Few-Shot JSON Learning with Context Building")
print("="*80)

# Create test images
print("\nCreating test images...")
black_img = np.zeros((100, 100, 3))
white_img = np.ones((100, 100, 3))
red_img = np.zeros((100, 100, 3))
red_img[:, :, 0] = 1.0
green_img = np.zeros((100, 100, 3))
green_img[:, :, 1] = 1.0

# Build messages with few-shot examples
messages = [
    # Example 1: Black
    {
        "image_base64": numpy_to_base64(black_img),
        "text": "Analyze this image. Respond with JSON: {\"color\": \"<color>\", \"brightness\": \"<level>\"}"
    },
    # Example 2: White
    {
        "image_base64": numpy_to_base64(white_img),
        "text": "Analyze this image. Respond with JSON: {\"color\": \"<color>\", \"brightness\": \"<level>\"}"
    },
    # Query: Red (this is what we actually want analyzed)
    {
        "image_base64": numpy_to_base64(red_img),
        "text": "Analyze this image. Respond with JSON: {\"color\": \"<color>\", \"brightness\": \"<level>\"}"
    }
]

print("\nSending 3 images with context_building strategy...")
print("  1. Black (example)")
print("  2. White (example)")
print("  3. Red (query)\n")

response = requests.post(
    f"{SERVER_URL}/generate",
    json={
        "messages": messages,
        "max_tokens": 100,
        "strategy": "context_building"
    },
    timeout=120
)

if response.status_code != 200:
    print(f"✗ Error {response.status_code}: {response.text}")
    sys.exit(1)

result = response.json()

print(f"Strategy: {result['strategy']}")
print(f"Processed: {result['num_images']} images\n")

print("="*80)
print("RESULTS")
print("="*80)

colors = ["BLACK", "WHITE", "RED"]
for idx, (color_name, res) in enumerate(zip(colors, result["results"])):
    print(f"\n{idx+1}. {color_name} image:")

    # Extract answer (after "assistant\n")
    answer = res['result'].split('assistant\n')[-1].strip()
    print(f"   Raw: {answer[:100]}")

    # Try to parse JSON
    try:
        # Try direct parse
        parsed = json.loads(answer)
        print(f"   ✓ Parsed JSON: {parsed}")
    except:
        # Try to extract JSON from text
        import re
        json_match = re.search(r'\{.*?\}', answer, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                print(f"   ✓ Extracted JSON: {parsed}")
            except:
                print(f"   ✗ Failed to parse JSON")
                parsed = None
        else:
            print(f"   ✗ No JSON found")
            parsed = None

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

# Check if RED was correctly identified
if len(result["results"]) >= 3:
    red_answer = result["results"][2]['result'].split('assistant\n')[-1].strip()

    # Try to extract JSON
    try:
        parsed = json.loads(red_answer)
    except:
        import re
        json_match = re.search(r'\{.*?\}', red_answer, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except:
                parsed = None
        else:
            parsed = None

    if parsed and 'red' in str(parsed.get('color', '')).lower():
        print("\n✓✓ SUCCESS!")
        print("   - AI responded in JSON format")
        print("   - AI correctly identified RED color")
        print("   - Few-shot learning with context_building works!")
    elif parsed:
        print(f"\n✓ Partial success:")
        print(f"   - Got JSON format")
        print(f"   ✗ Wrong color: {parsed.get('color')} (expected 'red')")
    else:
        print("\n✗ Failed:")
        print("   - Could not parse JSON from response")
else:
    print("\n✗ Incomplete results")
