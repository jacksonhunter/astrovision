"""Test if AI can actually SEE different images.

Simple test: Send completely different images and see if we get different responses.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from astro_ai_client import AstroAIClient

client = AstroAIClient()

if not client.check_server():
    print("Server not running")
    sys.exit(1)

print("Testing AI vision with 3 different images:")
print("="*60)

# Test 1: All black
black = np.zeros((100, 100, 3))
print("\n1. All BLACK image:")
result1 = client.analyze(
    black,
    "Describe what you see in this image. What color is it?",
    max_tokens=100,
    expect_json=False
)
print(f"   Response: {result1['text'][:200]}")

# Test 2: All white
white = np.ones((100, 100, 3))
print("\n2. All WHITE image:")
result2 = client.analyze(
    white,
    "Describe what you see in this image. What color is it?",
    max_tokens=100,
    expect_json=False
)
print(f"   Response: {result2['text'][:200]}")

# Test 3: Red square
red = np.zeros((100, 100, 3))
red[:, :, 0] = 1.0  # Red channel only
print("\n3. All RED image:")
result3 = client.analyze(
    red,
    "Describe what you see in this image. What color is it?",
    max_tokens=100,
    expect_json=False
)
print(f"   Response: {result3['text'][:200]}")

print("\n" + "="*60)
print("VERDICT:")
if result1['text'] == result2['text'] == result3['text']:
    print("✗ FAIL - All responses identical (AI is blind!)")
elif "black" in result1['text'].lower() and "white" in result2['text'].lower():
    print("✓ PASS - AI can see different images")
else:
    print("? UNCLEAR - Responses differ but not conclusive")
