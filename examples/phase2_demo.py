"""Phase 2 Demo: Image Processing - Alignment, Normalization, and Stretching

This script demonstrates the Phase 2 FITS processing capabilities:
- WCS validation and comparison
- Image reprojection and alignment
- Data normalization (interval selection)
- Non-linear stretching

Prerequisites:
- pip install astropy numpy reproject matplotlib

Usage:
    python examples/phase2_demo.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from astro_vision_composer.preprocessing import FITSLoader
from astro_vision_composer.processing import (
    WCSHandler, Reprojector, Normalizer, Stretcher
)


def demo_wcs_validation():
    """Demonstrate WCS validation and comparison."""
    print("\n" + "=" * 60)
    print("DEMO 1: WCS Validation and Comparison")
    print("=" * 60)

    # Create synthetic WCS for demonstration
    from astropy.wcs import WCS
    from astropy.io import fits

    # Create a simple WCS header
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 2048
    header['NAXIS2'] = 2048
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRPIX1'] = 1024.0
    header['CRPIX2'] = 1024.0
    header['CRVAL1'] = 150.0  # RA in degrees
    header['CRVAL2'] = 2.5     # Dec in degrees
    header['CDELT1'] = -0.0001  # degrees per pixel
    header['CDELT2'] = 0.0001
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'

    wcs1 = WCS(header)

    # Validate WCS
    handler = WCSHandler()
    wcs_info = handler.validate(wcs1)

    print(f"\nWCS Validation Result:")
    print(f"  Valid: {wcs_info.is_valid}")
    print(f"  Projection: {wcs_info.projection}")
    print(f"  Pixel Scale: {wcs_info.pixel_scale:.3f} arcsec/pixel")
    print(f"  Reference Pixel: {wcs_info.reference_pixel}")
    print(f"  Reference Sky: {wcs_info.reference_sky}")

    if wcs_info.warnings:
        print(f"  Warnings: {', '.join(wcs_info.warnings)}")

    # Create a second WCS with slight differences
    header2 = header.copy()
    header2['CDELT1'] = -0.00012  # Different pixel scale
    header2['CRVAL1'] = 150.1     # Different center
    wcs2 = WCS(header2)

    # Compare WCS
    comparison = handler.compare_wcs(wcs1, wcs2)
    print(f"\nWCS Comparison:")
    print(f"  Compatible: {comparison['compatible']}")
    print(f"  Projection Match: {comparison['projection_match']}")
    if comparison['pixel_scale_diff']:
        print(f"  Pixel Scale Difference: {comparison['pixel_scale_diff']:.1f}%")
    if comparison['details']:
        for detail in comparison['details']:
            print(f"  - {detail}")


def demo_normalization_stretching():
    """Demonstrate normalization and stretching on synthetic data."""
    print("\n" + "=" * 60)
    print("DEMO 2: Normalization and Stretching")
    print("=" * 60)

    # Create synthetic astronomical image
    # Simulate: bright star + faint extended emission + noise
    size = 1000
    y, x = np.ogrid[:size, :size]

    # Background
    image = np.random.normal(100, 10, (size, size))

    # Bright star at center
    star_y, star_x = size // 2, size // 2
    r_star = np.sqrt((x - star_x)**2 + (y - star_y)**2)
    star = 50000 * np.exp(-r_star**2 / (20**2))
    image += star

    # Faint extended emission (galaxy-like)
    galaxy_y, galaxy_x = size // 2 + 200, size // 2 - 150
    r_galaxy = np.sqrt((x - galaxy_x)**2 + (y - galaxy_y)**2)
    galaxy = 800 * np.exp(-r_galaxy**2 / (100**2))
    image += galaxy

    print(f"\nSynthetic Image Statistics:")
    print(f"  Shape: {image.shape}")
    print(f"  Min: {image.min():.1f}")
    print(f"  Max: {image.max():.1f}")
    print(f"  Mean: {image.mean():.1f}")
    print(f"  Median: {np.median(image):.1f}")

    # Initialize normalizer and stretcher
    normalizer = Normalizer()
    stretcher = Stretcher()

    # Test different normalization methods
    print(f"\nNormalization Methods:")

    methods = ['minmax', 'percentile', 'zscale']
    normalized_images = {}

    for method in methods:
        if method == 'percentile':
            norm = normalizer.normalize(image, method=method, lower=1, upper=99)
        else:
            norm = normalizer.normalize(image, method=method)

        vmin, vmax = normalizer.get_interval_limits(image, method=method)
        normalized_images[method] = norm

        print(f"  {method:12s}: vmin={vmin:8.1f}, vmax={vmax:8.1f}")

    # Test different stretch methods
    print(f"\nStretch Methods (using zscale normalization):")

    stretch_methods = ['linear', 'sqrt', 'log', 'asinh']
    stretched_images = {}

    normalized = normalized_images['zscale']

    for stretch_method in stretch_methods:
        if stretch_method == 'log':
            stretched = stretcher.stretch(normalized, method=stretch_method, a=1000)
        elif stretch_method == 'asinh':
            stretched = stretcher.stretch(normalized, method=stretch_method, a=0.1)
        else:
            stretched = stretcher.stretch(normalized, method=stretch_method)

        stretched_images[stretch_method] = stretched
        print(f"  {stretch_method:12s}: min={stretched.min():.3f}, max={stretched.max():.3f}")

    # Visualize results
    print(f"\nCreating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase 2: Normalization and Stretching Demo', fontsize=16)

    # Row 1: Original + normalized versions
    axes[0, 0].imshow(image, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original (raw values)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(normalized_images['percentile'], cmap='gray', origin='lower')
    axes[0, 1].set_title('Normalized (percentile 1-99)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(normalized_images['zscale'], cmap='gray', origin='lower')
    axes[0, 2].set_title('Normalized (zscale)')
    axes[0, 2].axis('off')

    # Row 2: Stretched versions
    axes[1, 0].imshow(stretched_images['linear'], cmap='gray', origin='lower')
    axes[1, 0].set_title('Linear stretch')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(stretched_images['sqrt'], cmap='gray', origin='lower')
    axes[1, 1].set_title('Sqrt stretch')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(stretched_images['asinh'], cmap='gray', origin='lower')
    axes[1, 2].set_title('Asinh stretch (a=0.1)')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent.parent / 'outputs' / 'phase2_demo_stretch.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


def demo_reprojection():
    """Demonstrate image reprojection (conceptual - requires real FITS files)."""
    print("\n" + "=" * 60)
    print("DEMO 3: Image Reprojection (Conceptual)")
    print("=" * 60)

    print("\nReprojector Usage Example:")
    print("""
    # With real FITS files:
    from astro_vision_composer.preprocessing import FITSLoader
    from astro_vision_composer.processing import Reprojector

    # Load images
    loader = FITSLoader()
    target = loader.load('reference.fits')
    source = loader.load('image_to_align.fits')

    # Reproject
    reprojector = Reprojector(method='interp')
    aligned, footprint = reprojector.reproject_from_fits_data(source, target)

    print(f"Aligned shape: {aligned.shape}")
    print(f"Target shape: {target.shape}")
    """)

    print("\nFor multiple images:")
    print("""
    # Align a set of images
    images = {
        'g': (g_data, g_wcs),
        'r': (r_data, r_wcs),
        'i': (i_data, i_wcs)
    }

    reprojector = Reprojector(method='interp')
    aligned = reprojector.align_image_set(images, reference='i')

    # Now all images are on the same pixel grid
    """)


def main():
    """Run all Phase 2 demos."""
    print("\n" + "#" * 60)
    print("# Phase 2 Demo: Image Processing")
    print("# Components: WCSHandler, Reprojector, Normalizer, Stretcher")
    print("#" * 60)

    try:
        # Demo 1: WCS validation
        demo_wcs_validation()

        # Demo 2: Normalization and stretching (with visualization)
        demo_normalization_stretching()

        # Demo 3: Reprojection (conceptual)
        demo_reprojection()

        print("\n" + "=" * 60)
        print("Phase 2 Demo Complete!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  - WCSHandler validates and compares WCS information")
        print("  - Normalizer scales data using various interval methods")
        print("  - Stretcher applies non-linear transformations")
        print("  - Reprojector aligns images to common pixel grid")
        print("\nNext Steps:")
        print("  - Phase 3: Channel mapping and RGB composition")
        print("  - Test with real astronomical FITS files")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
