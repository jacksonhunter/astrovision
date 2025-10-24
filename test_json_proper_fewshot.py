"""Proper few-shot JSON learning using the Golden Rule.

Examples: Explicit format instructions
Query: Vague prompt (relies on learned pattern)
"""

import sys
import json
import base64
import requests
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image

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
    print("✗ Server not running")
    sys.exit(1)

print("\n" + "="*80)
print("PROPER FEW-SHOT JSON LEARNING")
print("="*80)
print("\nGolden Rule: 'Give explicit examples, ask vague questions'\n")

# Create test images
black_img = np.zeros((100, 100, 3))
white_img = np.ones((100, 100, 3))
red_img = np.zeros((100, 100, 3))
red_img[:, :, 0] = 1.0

# Build messages following the pattern
messages = [
    # Example 1: EXPLICIT JSON instruction
    {
        "image_base64": numpy_to_base64(black_img),
        "text": 'Analyze this calibration image. Respond in JSON format: {"color": "<dominant_color>", "brightness": "<dark/medium/bright>", "description": "<one sentence>"}'
    },
    # Example 2: REINFORCE the pattern
    {
        "image_base64": numpy_to_base64(white_img),
        "text": 'Analyze this reference image. Respond in JSON format: {"color": "<dominant_color>", "brightness": "<dark/medium/bright>", "description": "<one sentence>"}'
    },
    # Query: VAGUE (relies on examples)
    {
        "image_base64": numpy_to_base64(red_img),
        "text": "Analyze this image in the same JSON format"
    }
]

print("Strategy: context_building")
print("\nMessages:")
print("  1. Black image: EXPLICIT format instruction")
print("  2. White image: REINFORCE format")
print("  3. Red image: VAGUE query ('same JSON format')\n")

print("Sending to server...")
response = requests.post(
    f"{SERVER_URL}/generate",
    json={
        "messages": messages,
        "max_tokens": 150,
        "strategy": "context_building"
    },
    timeout=120
)

if response.status_code != 200:
    print(f"✗ Error: {response.text}")
    sys.exit(1)

result = response.json()

print("\n" + "="*80)
print("RESULTS")
print("="*80)

def extract_json(text):
    """Extract JSON from response."""
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Look for JSON object
    import re
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    return None

images = ["BLACK", "WHITE", "RED"]
expected_colors = ["black", "white", "red"]

all_correct = True

for idx, (img_name, expected_color, res) in enumerate(zip(images, expected_colors, result["results"])):
    print(f"\n{idx+1}. {img_name} image:")

    # Extract answer
    answer = res['result'].split('assistant\n')[-1].strip()
    print(f"   Raw: {answer}")

    # Parse JSON
    parsed = extract_json(answer)
    if parsed:
        print(f"   ✓ JSON: {parsed}")

        # Check color
        actual_color = str(parsed.get('color', '')).lower()
        if expected_color in actual_color:
            print(f"   ✓✓ Color correct: '{actual_color}'")
        else:
            print(f"   ✗ Color wrong: '{actual_color}' (expected '{expected_color}')")
            all_correct = False
    else:
        print(f"   ✗ Failed to parse JSON")
        all_correct = False

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if all_correct:
    print("\n🎉 SUCCESS!")
    print("  ✓ All images returned valid JSON")
    print("  ✓ All colors correctly identified")
    print("  ✓ Few-shot learning works!")
    print("\nReady to apply to astronomical data!")
else:
    print("\n❌ PARTIAL SUCCESS")
    print("  Check results above for issues")

print("\n" + "="*80)
