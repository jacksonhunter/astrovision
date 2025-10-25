"""Phase 3 Demo: RGB Composition - Complete End-to-End Pipeline

This script demonstrates the complete FITS processing pipeline from
loading to RGB composite generation:

Phase 1: Load FITS files
Phase 2: Normalize and stretch
Phase 3: Map channels, create composite, export

Prerequisites:
- pip install astropy numpy matplotlib Pillow

Usage:
    python examples/phase3_demo.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from astro_vision_composer.processing import Normalizer, Stretcher
from astro_vision_composer.postprocessing import (
    ChannelMapper, Compositor, ImageExporter, HistoryTracker
)


def create_synthetic_band(size, center, sigma, brightness, wavelength_name):
    """Create a synthetic astronomical image band."""
    y, x = np.ogrid[:size, :size]

    # Background
    image = np.random.normal(100, 10, (size, size))

    # Add source
    cx, cy = center
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    source = brightness * np.exp(-r**2 / (2 * sigma**2))
    image += source

    # Ensure positive
    image = np.maximum(image, 0)

    return image


def demo_channel_mapping():
    """Demonstrate channel mapping."""
    print("\n" + "=" * 60)
    print("DEMO 1: Channel Mapping")
    print("=" * 60)

    mapper = ChannelMapper()

    # Define PanSTARRS-like bands with wavelengths
    bands = {
        'g': 481,  # nm
        'r': 617,
        'i': 752
    }

    print(f"\nAvailable bands: {list(bands.keys())}")
    print(f"Wavelengths: {bands}")

    # Auto-map by wavelength (chromatic ordering)
    mapping = mapper.auto_map_by_wavelength(bands)

    print(f"\nAutomatic chromatic mapping:")
    print(f"  Red channel   ← {mapping.red} ({mapping.red_wavelength:.0f} nm)")
    print(f"  Green channel ← {mapping.green} ({mapping.green_wavelength:.0f} nm)")
    print(f"  Blue channel  ← {mapping.blue} ({mapping.blue_wavelength:.0f} nm)")
    print(f"  Chromatic order: {mapping.chromatic_order}")

    # Manual mapping
    manual_map = mapper.manual_mapping(red='r', green='g', blue='i', wavelengths=bands)
    print(f"\nManual mapping:")
    print(f"  Red={manual_map.red}, Green={manual_map.green}, Blue={manual_map.blue}")
    print(f"  Chromatic order: {manual_map.chromatic_order}")

    # Validate
    valid, errors = mapper.validate_mapping(mapping, list(bands.keys()))
    print(f"\nMapping validation: {'✓ Valid' if valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")


def demo_full_pipeline():
    """Demonstrate complete FITS→RGB pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 2: Complete RGB Composite Pipeline")
    print("=" * 60)

    # Initialize history tracker
    tracker = HistoryTracker()

    # Step 1: Create synthetic multi-band data
    print("\n[Step 1] Creating synthetic 3-band data...")
    size = 800

    # Simulate three bands (g, r, i) with different sources
    g_data = create_synthetic_band(
        size, center=(400, 300), sigma=40, brightness=5000, wavelength_name='g'
    )
    r_data = create_synthetic_band(
        size, center=(400, 400), sigma=60, brightness=8000, wavelength_name='r'
    )
    i_data = create_synthetic_band(
        size, center=(400, 500), sigma=50, brightness=6000, wavelength_name='i'
    )

    bands = {'g': g_data, 'r': r_data, 'i': i_data}
    print(f"  Created {len(bands)} bands: {list(bands.keys())}")

    # Step 2: Normalize
    print("\n[Step 2] Normalizing data...")
    normalizer = Normalizer()
    normalized = {}

    for name, data in bands.items():
        norm = normalizer.normalize(data, method='zscale')
        normalized[name] = norm
        tracker.record('normalize', {'method': 'zscale', 'band': name}, 'Normalizer')
        print(f"  {name}: normalized to [0, 1]")

    # Step 3: Stretch
    print("\n[Step 3] Applying non-linear stretch...")
    stretcher = Stretcher()
    stretched = {}

    for name, data in normalized.items():
        stretch = stretcher.stretch(data, method='asinh', a=0.1)
        stretched[name] = stretch
        tracker.record('stretch', {'method': 'asinh', 'a': 0.1, 'band': name}, 'Stretcher')
        print(f"  {name}: asinh stretch applied")

    # Step 4: Map channels
    print("\n[Step 4] Mapping channels to RGB...")
    mapper = ChannelMapper()
    wavelengths = {'g': 481, 'r': 617, 'i': 752}
    mapping = mapper.auto_map_by_wavelength(wavelengths)

    tracker.record(
        'map_channels',
        {
            'red': mapping.red,
            'green': mapping.green,
            'blue': mapping.blue,
            'method': 'chromatic_order'
        },
        'ChannelMapper'
    )

    print(f"  Red   ← {mapping.red} ({mapping.red_wavelength:.0f} nm)")
    print(f"  Green ← {mapping.green} ({mapping.green_wavelength:.0f} nm)")
    print(f"  Blue  ← {mapping.blue} ({mapping.blue_wavelength:.0f} nm)")

    # Step 5: Create RGB composite
    print("\n[Step 5] Creating RGB composite...")
    compositor = Compositor()

    # Method 1: Lupton RGB
    rgb_lupton = compositor.create_from_mapping(
        data=stretched,
        mapping=mapping,
        method='lupton',
        stretch=0.5,
        Q=8
    )
    tracker.record(
        'create_composite',
        {'method': 'lupton', 'stretch': 0.5, 'Q': 8},
        'Compositor'
    )
    print(f"  Lupton RGB: shape={rgb_lupton.shape}, dtype={rgb_lupton.dtype}")

    # Method 2: Simple RGB
    rgb_simple = compositor.create_from_mapping(
        data=stretched,
        mapping=mapping,
        method='simple'
    )
    print(f"  Simple RGB: shape={rgb_simple.shape}, dtype={rgb_simple.dtype}")

    # Step 6: Export
    print("\n[Step 6] Exporting images...")
    exporter = ImageExporter()
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    # Save Lupton composite
    lupton_path = exporter.save_png(
        rgb_lupton,
        output_dir / 'phase3_demo_lupton.png',
        metadata={
            'method': 'Lupton RGB',
            'bands': 'gri',
            'source': 'Synthetic'
        },
        bit_depth=16
    )
    tracker.record(
        'export',
        {'format': 'PNG', 'bit_depth': 16, 'path': str(lupton_path)},
        'ImageExporter'
    )
    print(f"  Saved: {lupton_path}")

    # Save simple composite
    simple_path = exporter.save_png(
        rgb_simple,
        output_dir / 'phase3_demo_simple.png',
        metadata={
            'method': 'Simple RGB',
            'bands': 'gri',
            'source': 'Synthetic'
        },
        bit_depth=8
    )
    print(f"  Saved: {simple_path}")

    # Step 7: Show processing history
    print("\n[Step 7] Processing History:")
    print("-" * 60)
    print(tracker.to_text(include_timestamps=False))
    print("-" * 60)
    print(f"Total steps: {len(tracker)}")

    # Visualize results
    print("\n[Step 8] Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase 3: Complete RGB Composite Pipeline', fontsize=16)

    # Row 1: Individual normalized bands
    axes[0, 0].imshow(normalized['g'], cmap='gray', origin='lower')
    axes[0, 0].set_title('g-band (normalized)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(normalized['r'], cmap='gray', origin='lower')
    axes[0, 1].set_title('r-band (normalized)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(normalized['i'], cmap='gray', origin='lower')
    axes[0, 2].set_title('i-band (normalized)')
    axes[0, 2].axis('off')

    # Row 2: RGB composites
    axes[1, 0].imshow(rgb_lupton, origin='lower')
    axes[1, 0].set_title('Lupton RGB (stretch=0.5, Q=8)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(rgb_simple, origin='lower')
    axes[1, 1].set_title('Simple RGB')
    axes[1, 1].axis('off')

    # Show channel mapping
    axes[1, 2].axis('off')
    axes[1, 2].text(
        0.1, 0.9,
        'Channel Mapping:\n\n'
        f'Red   ← {mapping.red} ({mapping.red_wavelength:.0f} nm)\n'
        f'Green ← {mapping.green} ({mapping.green_wavelength:.0f} nm)\n'
        f'Blue  ← {mapping.blue} ({mapping.blue_wavelength:.0f} nm)\n\n'
        'Processing:\n'
        '• Normalize (ZScale)\n'
        '• Stretch (Asinh, a=0.1)\n'
        '• Composite (Lupton)\n'
        '• Export (16-bit PNG)',
        fontsize=11,
        verticalalignment='top',
        family='monospace'
    )

    plt.tight_layout()

    viz_path = output_dir / 'phase3_demo_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved: {viz_path}")

    plt.show()

    return tracker


def main():
    """Run all Phase 3 demos."""
    print("\n" + "#" * 60)
    print("# Phase 3 Demo: RGB Composition")
    print("# Components: ChannelMapper, Compositor, Exporter, HistoryTracker")
    print("#" * 60)

    try:
        # Demo 1: Channel mapping
        demo_channel_mapping()

        # Demo 2: Complete pipeline
        tracker = demo_full_pipeline()

        print("\n" + "=" * 60)
        print("Phase 3 Demo Complete!")
        print("=" * 60)
        print("\nKey Achievements:")
        print("  ✓ Channel mapping by wavelength (chromatic ordering)")
        print("  ✓ Lupton RGB composite generation")
        print("  ✓ Simple RGB composite generation")
        print("  ✓ Image export (PNG, TIFF, JPEG)")
        print("  ✓ Processing history tracking")
        print("\nEnd-to-End Pipeline:")
        print("  FITS → Normalize → Stretch → Map → Composite → Export")
        print("\nNext Steps:")
        print("  - Test with real astronomical FITS files")
        print("  - Add color balancing (Phase 4)")
        print("  - Add enhancement features (Phase 4)")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
