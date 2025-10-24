# """Test if CLAHE is killing the low-median data."""

import numpy as np
from skimage import exposure

# Simulate data EXACTLY like we have: median ~0.004, mean ~0.07
np.random.seed(42)

# Create synthetic data matching our ACTUAL characteristics
# 48% zeros, rest exponentially distributed
data = np.random.exponential(scale=0.02, size=(1000, 1000))
# Make 48% of pixels zero (like after clipping negatives)
zero_mask = np.random.random(size=(1000, 1000)) < 0.48
data[zero_mask] = 0
data = np.clip(data, 0, 1)

print("BEFORE CLAHE:")
print(f"  Min: {np.min(data):.6f}")
print(f"  Max: {np.max(data):.6f}")
print(f"  Median: {np.median(data):.6f}")
print(f"  Mean: {np.mean(data):.6f}")
print(f"  Pixels > 0.1: {100*np.sum(data > 0.1)/data.size:.1f}%")
print(f"  Pixels > 0.5: {100*np.sum(data > 0.5)/data.size:.1f}%")

# Apply CLAHE like create_rgb_composite does
clahe_result = exposure.equalize_adapthist(data, clip_limit=0.03)

print("\nAFTER CLAHE (clip_limit=0.03):")
print(f"  Min: {np.min(clahe_result):.6f}")
print(f"  Max: {np.max(clahe_result):.6f}")
print(f"  Median: {np.median(clahe_result):.6f}")
print(f"  Mean: {np.mean(clahe_result):.6f}")
print(f"  Pixels > 0.1: {100*np.sum(clahe_result > 0.1)/clahe_result.size:.1f}%")
print(f"  Pixels > 0.5: {100*np.sum(clahe_result > 0.5)/clahe_result.size:.1f}%")

print("\n" + "="*80)
print("DIAGNOSIS:")
if np.median(clahe_result) < 0.0001:
    print("✗ CLAHE is KILLING the data with low medians!")
else:
    print("✓ CLAHE seems OK - problem is elsewhere")