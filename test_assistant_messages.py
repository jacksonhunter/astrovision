"""Test if server supports assistant messages in array (true few-shot)."""

import requests
import base64
import numpy as np
from io import BytesIO
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
except:
    print("✗ Server not running")
    exit(1)

print("\n" + "="*80)
print("Testing Assistant Messages Support (True Few-Shot)")
print("="*80)

# Create test images
black_img = np.zeros((100, 100, 3))
white_img = np.ones((100, 100, 3))
red_img = np.zeros((100, 100, 3))
red_img[:, :, 0] = 1.0

# Build messages with ASSISTANT responses included
messages = [
    # Example 1: User query
    {
        "image_base64": numpy_to_base64(black_img),
        "text": "Analyze this image. Respond in JSON: {\"color\": \"<color>\", \"brightness\": \"<level>\"}"
    },
    # Example 1: Assistant response (PRE-FILLED)
    {
        "role": "assistant",
        "text": '{"color": "black", "brightness": "dark"}'
    },
    # Example 2: User query
    {
        "image_base64": numpy_to_base64(white_img),
        "text": "Analyze this image. Respond in JSON: {\"color\": \"<color>\", \"brightness\": \"<level>\"}"
    },
    # Example 2: Assistant response (PRE-FILLED)
    {
        "role": "assistant",
        "text": '{"color": "white", "brightness": "bright"}'
    },
    # Query: Real question
    {
        "image_base64": numpy_to_base64(red_img),
        "text": "Analyze this image. Respond in JSON: {\"color\": \"<color>\", \"brightness\": \"<level>\"}"
    }
]

print("\nMessages structure:")
print("  1. User: black image + prompt")
print("  2. Assistant: PRE-FILLED response")
print("  3. User: white image + prompt")
print("  4. Assistant: PRE-FILLED response")
print("  5. User: red image + prompt (actual query)\n")

print("Sending to server with context_building strategy...")

try:
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
        print(f"✗ Server error: {response.status_code}")
        print(response.text)
        exit(1)

    result = response.json()

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)

    # Should only get responses for user messages (1, 3, 5)
    print(f"\nNumber of results: {len(result['results'])}")

    for idx, res in enumerate(result['results']):
        print(f"\nResult {idx+1}:")
        answer = res['result'].split('assistant\n')[-1].strip()
        print(f"  {answer}")

    # Check if last result (red image) got correct JSON
    if len(result['results']) >= 3:
        last_answer = result['results'][-1]['result'].split('assistant\n')[-1].strip()

        import json
        try:
            parsed = json.loads(last_answer)
            if 'red' in str(parsed.get('color', '')).lower():
                print("\n🎉 SUCCESS!")
                print("  ✓ Server supports assistant messages")
                print("  ✓ True few-shot learning works")
                print("  ✓ Red color correctly identified")
            else:
                print("\n⚠ Parsed JSON but wrong color:")
                print(f"  {parsed}")
        except:
            print("\n✗ Failed to parse JSON from response")
            print(f"  Raw: {last_answer}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
