"""Check available system memory for vision model loading."""

import psutil
import sys

print("="*80)
print("SYSTEM MEMORY CHECK")
print("="*80)

# Get memory info
mem = psutil.virtual_memory()

print(f"\nTotal RAM: {mem.total / (1024**3):.2f} GB")
print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
print(f"Used RAM: {mem.used / (1024**3):.2f} GB")
print(f"Free RAM: {mem.free / (1024**3):.2f} GB")
print(f"Usage: {mem.percent}%")

print("\n" + "="*80)
print("MODEL REQUIREMENTS")
print("="*80)

model_size_gb = 7.91  # Actual model files downloaded
model_ram_needed = 10  # Estimated RAM needed for 4-bit quantized

print(f"\nModel files on disk: ~{model_size_gb:.2f} GB")
print(f"Estimated RAM needed: ~{model_ram_needed} GB")
print(f"  (Model weights + overhead for 4-bit quantization)")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if mem.available / (1024**3) >= model_ram_needed:
    print(f"\n✓ You have {mem.available / (1024**3):.2f} GB available")
    print(f"  Model needs ~{model_ram_needed} GB")
    print("  Should work!")
elif mem.available / (1024**3) >= model_ram_needed * 0.8:
    print(f"\n⚠ Borderline: {mem.available / (1024**3):.2f} GB available")
    print(f"  Model needs ~{model_ram_needed} GB")
    print("  Might work if you close some apps")
else:
    print(f"\n✗ Insufficient: {mem.available / (1024**3):.2f} GB available")
    print(f"  Model needs ~{model_ram_needed} GB")
    print(f"  Need to free up {model_ram_needed - mem.available / (1024**3):.2f} GB")

print("\n" + "="*80)
print("TOP MEMORY CONSUMERS")
print("="*80)

# Get top processes by memory
processes = []
for proc in psutil.process_iter(['name', 'memory_info']):
    try:
        processes.append((proc.info['name'], proc.info['memory_info'].rss / (1024**3)))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

# Sort by memory usage
processes.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 processes by RAM usage:")
for i, (name, mem_gb) in enumerate(processes[:10], 1):
    print(f"{i:2}. {name:30} {mem_gb:>6.2f} GB")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if mem.available / (1024**3) < model_ram_needed:
    print("\nTo free up RAM, consider closing:")
    print("  - Web browsers (Chrome/Firefox/Edge)")
    print("  - PyCharm (if not needed)")
    print("  - Other development tools")
    print("  - Background apps")
    print(f"\nTarget: Free up {model_ram_needed - mem.available / (1024**3):.2f} GB more")
else:
    print("\nYou have enough RAM. The error might be due to:")
    print("  1. bitsandbytes CPU offload not enabled")
    print("  2. Model loading configuration issue")
    print("  3. Temporary spike in memory usage")
    print("\nTry closing some apps anyway to ensure clean run.")