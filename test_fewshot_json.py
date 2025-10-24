"""Test few-shot prompting to get JSON responses.

Show AI examples with images, then ask it to analyze a new one.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from astro_ai_client import AstroAIClient

client = AstroAIClient()

if not client.check_server():
    print("Server not running")
    sys.exit(1)

print("Testing FEW-SHOT prompting for JSON responses")
print("="*60)

# Create test images
black_img = np.zeros((100, 100, 3))
white_img = np.ones((100, 100, 3))
red_img = np.zeros((100, 100, 3))
red_img[:, :, 0] = 1.0

print("\nSending few-shot prompt with image examples...")
print("-"*60)

# Build a few-shot prompt as a single text (since our server takes single prompt)
# We'll describe the examples in text since our current API doesn't support multi-turn with images

prompt = """I will show you astronomical images and you will analyze them in JSON format.

Example 1: A black calibration frame
Your response: ```json
{
  "dominant_color": "black",
  "brightness": "dark",
  "description": "Completely dark calibration frame"
}
```

Example 2: A bright white reference frame
Your response: ```json
{
  "dominant_color": "white",
  "brightness": "bright",
  "description": "Bright white reference frame"
}
```

Now analyze THIS image and respond in the same JSON format:"""

result = client.analyze(red_img, prompt, max_tokens=200, expect_json=True)

print("\nRED image result:")
if result.get('json'):
    print(f"  ✓ Parsed JSON:")
    for key, val in result['json'].items():
        print(f"    {key}: {val}")
else:
    print(f"  ✗ Failed to parse")
    print(f"  Raw text: {result['text'][:200]}")

# Now test if it learned - send another without examples
print("\n" + "-"*60)
print("Testing if pattern learned (no examples this time)...")

simple_prompt = "Analyze this image in JSON format with dominant_color, brightness, and description fields:"

result2 = client.analyze(white_img, simple_prompt, max_tokens=200, expect_json=True)

print("\nWHITE image result (no examples):")
if result2.get('json'):
    print(f"  ✓ Parsed JSON:")
    for key, val in result2['json'].items():
        print(f"    {key}: {val}")
else:
    print(f"  ✗ Failed to parse")
    print(f"  Raw text: {result2['text'][:200]}")

print("\n" + "="*60)
print("CONCLUSION:")
if result.get('json') and result['json'].get('dominant_color', '').lower() == 'red':
    print("  ✓ Few-shot prompting works!")
    print("  ✓ AI correctly identified red color in JSON format")
else:
    print("  ✗ Still having issues with JSON or color identification")
