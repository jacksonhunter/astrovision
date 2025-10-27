"""Comprehensive example of Phase 4: Narrowband Palette Support.

This example demonstrates:
1. Narrowband palette mapping (Hubble, HOO, natural, custom)
2. Multi-band selection strategies (max-span, PCA, science-driven)
3. Per-channel intervals and stretches for narrowband imaging
4. Advanced false-color RGB composition

Requirements:
- astropy >= 5.3.0
- numpy >= 1.24.0
- scikit-learn >= 1.2.0 (for PCA selection, optional)
"""

import sys
from pathlib import Path

# Add src to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
from astropy.visualization import (
    ZScaleInterval, PercentileInterval, AsymmetricPercentileInterval,
    AsinhStretch, LogStretch, LinearStretch
)

from astro_vision_composer.pipeline import ProcessingPipeline
from astro_vision_composer.postprocessing import (
    ChannelMapper, PaletteMapper, BandSelector, SelectionStrategy
)


def example_1_hubble_palette():
    """Example 1: Use Hubble palette (SHO) for narrowband imaging.

    The Hubble palette assigns:
    - S-II (672nm) → Red
    - H-α (656nm) → Green
    - O-III (501nm) → Blue

    This creates dramatic false-color images emphasizing different emission regions.
    """
    print("\n" + "="*70)
    print("Example 1: Hubble Palette (SHO) for Narrowband Imaging")
    print("="*70)

    # Initialize channel mapper
    mapper = ChannelMapper()

    # List available narrowband palettes
    print("\nAvailable narrowband palettes:")
    for name, desc in mapper.list_narrowband_palettes().items():
        print(f"  - {name}: {desc}")

    # Map narrowband bands using Hubble palette
    bands = ['sii', 'ha', 'oiii']
    mapping = mapper.map_narrowband(bands, palette='hubble')
    print(f"\nHubble palette mapping: {mapping}")

    # Use with pipeline - per-channel intervals for different emission line intensities
    pipeline = ProcessingPipeline(mode='manual')

    # Narrowband data often needs different normalization per channel
    # H-alpha is usually strongest, OIII weaker, SII weakest
    intervals = [
        PercentileInterval(99.5),  # SII (weakest, brightest percentile)
        PercentileInterval(99.0),  # Ha (strongest, lower percentile OK)
        PercentileInterval(99.8),  # OIII (medium, very bright percentile)
    ]

    stretches = [
        AsinhStretch(a=0.1),  # SII - aggressive stretch
        AsinhStretch(a=0.5),  # Ha - moderate stretch
        AsinhStretch(a=0.2),  # OIII - aggressive stretch
    ]

    # Process with per-channel normalization
    # (Uncomment when you have actual narrowband FITS files)
    # rgb = pipeline.process_to_rgb(
    #     fits_files=['sii.fits', 'ha.fits', 'oiii.fits'],
    #     interval=intervals,
    #     stretch=stretches,
    #     compositor='simple'
    # )

    print("\nPer-channel intervals configured:")
    for i, (band, interval, stretch) in enumerate(zip(bands, intervals, stretches)):
        print(f"  {band}: {type(interval).__name__} + {type(stretch).__name__}")


def example_2_hoo_bicolor():
    """Example 2: HOO bicolor palette for emission nebulae.

    The HOO palette is popular for emission nebulae:
    - H-α → Red
    - O-III → Cyan (synthesized from green + blue channels)

    This creates teal/red images with good contrast.
    """
    print("\n" + "="*70)
    print("Example 2: HOO Bicolor Palette")
    print("="*70)

    mapper = ChannelMapper()

    # Map with HOO palette
    bands = ['ha', 'oiii']
    mapping = mapper.map_narrowband(bands, palette='hoo')
    print(f"\nHOO palette mapping: {mapping}")
    print("Note: OIII assigned to both green and blue to create cyan")

    # When using HOO, OIII appears in both G and B channels
    print("\nChannel assignments:")
    print(f"  Red:   {mapping.red}")
    print(f"  Green: {mapping.green}")
    print(f"  Blue:  {mapping.blue}")


def example_3_custom_palette():
    """Example 3: Create a custom false-color palette.

    You can define your own band-to-color mappings for artistic
    or scientific visualization.
    """
    print("\n" + "="*70)
    print("Example 3: Custom False-Color Palette")
    print("="*70)

    mapper = ChannelMapper()

    # Custom palette: Assign your own colors
    custom_palette = {
        'ha': 'yellow',    # H-alpha to yellow (red + green)
        'oiii': 'cyan',    # OIII to cyan (green + blue)
        'sii': 'magenta'   # SII to magenta (red + blue)
    }

    mapping = mapper.map_narrowband(
        bands=['ha', 'oiii', 'sii'],
        palette=custom_palette
    )

    print(f"\nCustom palette mapping: {mapping}")

    # Another example: Emphasize specific features
    scientific_palette = {
        'ha': 'red',      # Ionized hydrogen regions
        'oiii': 'green',  # Doubly ionized oxygen (hotter regions)
        'sii': 'blue'     # Ionized sulfur (shock regions)
    }

    mapping2 = mapper.map_narrowband(
        bands=['ha', 'oiii', 'sii'],
        palette=scientific_palette
    )

    print(f"Scientific palette mapping: {mapping2}")


def example_4_multi_band_selection_max_span():
    """Example 4: Select 3 bands from 10 using maximum wavelength span.

    When you have many filters, max-span selects the shortest, longest,
    and middle wavelength bands to maximize color separation.
    """
    print("\n" + "="*70)
    print("Example 4: Multi-Band Selection (Max Wavelength Span)")
    print("="*70)

    mapper = ChannelMapper()

    # JWST NIRCam has many filters - select best 3 for RGB
    bands = ['f070w', 'f090w', 'f115w', 'f150w', 'f200w',
             'f277w', 'f356w', 'f410m', 'f444w', 'f480m']

    wavelengths = {
        'f070w': 704,    # 0.7 μm
        'f090w': 901,    # 0.9 μm
        'f115w': 1154,   # 1.15 μm
        'f150w': 1501,   # 1.5 μm
        'f200w': 1989,   # 2.0 μm
        'f277w': 2762,   # 2.8 μm
        'f356w': 3568,   # 3.6 μm
        'f410m': 4082,   # 4.1 μm
        'f444w': 4421,   # 4.4 μm
        'f480m': 4834    # 4.8 μm
    }

    # Select 3 bands with maximum wavelength coverage
    mapping = mapper.select_and_map(
        bands=bands,
        wavelengths=wavelengths,
        strategy='max_span'
    )

    print(f"\nMax-span selection from {len(bands)} bands: {mapping}")
    print(f"  Red:   {mapping.red} @ {wavelengths[mapping.red]} nm")
    print(f"  Green: {mapping.green} @ {wavelengths[mapping.green]} nm")
    print(f"  Blue:  {mapping.blue} @ {wavelengths[mapping.blue]} nm")

    span = wavelengths[mapping.red] - wavelengths[mapping.blue]
    print(f"  Wavelength span: {span:.0f} nm")


def example_5_multi_band_selection_pca():
    """Example 5: Select 3 bands using PCA (most informative combination).

    PCA finds the 3 bands that capture the most variance in the data,
    which often corresponds to the most visually interesting combination.

    Note: Requires scikit-learn
    """
    print("\n" + "="*70)
    print("Example 5: Multi-Band Selection (PCA)")
    print("="*70)

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("\nSkipping PCA example - scikit-learn not installed")
        print("Install with: pip install scikit-learn")
        return

    mapper = ChannelMapper()
    selector = BandSelector()

    # Simulate multi-band data (in practice, load actual FITS files)
    bands = ['f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
    wavelengths = {
        'f090w': 901,
        'f150w': 1501,
        'f200w': 1989,
        'f277w': 2762,
        'f356w': 3568,
        'f444w': 4421
    }

    # Create synthetic data for demonstration
    # In real usage, this would be your actual image data
    np.random.seed(42)
    data = {
        band: np.random.rand(100, 100) * (i + 1)  # Vary intensity by wavelength
        for i, band in enumerate(bands)
    }

    # Select using PCA
    selection = selector.select_bands(
        bands=bands,
        wavelengths=wavelengths,
        strategy='pca',
        data=data
    )

    print(f"\nPCA selection: {selection}")
    print(f"Variance explained: {selection.metadata['variance_explained']:.1%}")


def example_6_science_driven_selection():
    """Example 6: Select bands based on science goals.

    Pre-defined selections optimize for specific astrophysical features.
    """
    print("\n" + "="*70)
    print("Example 6: Science-Driven Band Selection")
    print("="*70)

    selector = BandSelector()

    # List available science goals
    print("\nAvailable science goals:")
    for goal, desc in selector.list_science_goals().items():
        print(f"  - {goal}: {desc}")

    # Select for star formation
    bands = ['f070w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
    wavelengths = {
        'f070w': 704,
        'f150w': 1501,
        'f200w': 1989,
        'f277w': 2762,
        'f356w': 3568,
        'f444w': 4421
    }

    selection = selector.select_bands(
        bands=bands,
        wavelengths=wavelengths,
        strategy='science',
        science_goal='star_formation'
    )

    print(f"\nStar formation optimized selection: {selection}")


def example_7_per_channel_normalization():
    """Example 7: Advanced per-channel normalization with pipeline.

    Demonstrates using the pipeline's per-channel interval/stretch support
    for narrowband imaging where different emission lines have vastly
    different intensities.
    """
    print("\n" + "="*70)
    print("Example 7: Per-Channel Normalization in Pipeline")
    print("="*70)

    # Different normalization strategies for each narrowband filter
    # This is crucial for narrowband where Ha is often 10x brighter than SII

    intervals = [
        AsymmetricPercentileInterval(1, 99.5),  # SII - brightest 1-99.5%
        PercentileInterval(98.0),                # Ha - top 98%
        AsymmetricPercentileInterval(0.5, 99.8), # OIII - brightest 0.5-99.8%
    ]

    stretches = [
        AsinhStretch(a=0.05),  # SII - very aggressive stretch
        AsinhStretch(a=0.5),   # Ha - moderate stretch
        LogStretch(a=1000),    # OIII - log stretch for faint features
    ]

    print("\nPer-channel normalization configuration:")
    bands = ['sii', 'ha', 'oiii']
    for band, interval, stretch in zip(bands, intervals, stretches):
        print(f"  {band}:")
        print(f"    Interval: {type(interval).__name__}")
        if hasattr(interval, 'percentile'):
            print(f"      Percentile: {interval.percentile}%")
        elif hasattr(interval, 'lower_percentile'):
            print(f"      Range: {interval.lower_percentile}% - {interval.upper_percentile}%")
        print(f"    Stretch: {type(stretch).__name__}")
        if hasattr(stretch, 'a'):
            print(f"      Parameter a: {stretch.a}")

    print("\nUsage with pipeline:")
    print("""
    pipeline = ProcessingPipeline(mode='manual')
    rgb = pipeline.process_to_rgb(
        fits_files=['sii.fits', 'ha.fits', 'oiii.fits'],
        interval=intervals,  # Pass list of interval objects
        stretch=stretches,   # Pass list of stretch objects
        compositor='simple'
    )
    """)


def example_8_complete_narrowband_workflow():
    """Example 8: Complete narrowband imaging workflow.

    Combines palette mapping, per-channel normalization, and RGB composition
    for a production-quality narrowband image.
    """
    print("\n" + "="*70)
    print("Example 8: Complete Narrowband Workflow")
    print("="*70)

    print("\n1. Map bands to RGB using Hubble palette")
    mapper = ChannelMapper()
    mapping = mapper.map_narrowband(['sii', 'ha', 'oiii'], palette='hubble')
    print(f"   {mapping}")

    print("\n2. Configure per-channel normalization")
    print("   (Ha is brightest, OIII medium, SII faintest)")

    intervals = [
        PercentileInterval(99.5),  # SII - use bright 99.5%
        PercentileInterval(98.0),  # Ha - use bright 98%
        PercentileInterval(99.8),  # OIII - use bright 99.8%
    ]

    stretches = [
        AsinhStretch(a=0.1),   # SII - aggressive
        AsinhStretch(a=0.5),   # Ha - moderate
        AsinhStretch(a=0.2),   # OIII - aggressive
    ]

    print("\n3. Process with pipeline")
    print("""
    pipeline = ProcessingPipeline(mode='manual')

    # Method 1: Using process_to_rgb with lists
    rgb = pipeline.process_to_rgb(
        fits_files=['sii.fits', 'ha.fits', 'oiii.fits'],
        interval=intervals,
        stretch=stretches,
        compositor='simple',
        output_dir='output/hubble_palette'
    )

    # Method 2: Using process_with_arrays
    rgb = pipeline.process_with_arrays(
        fits_files=['sii.fits', 'ha.fits', 'oiii.fits'],
        intervals=intervals,
        stretches=stretches,
        compositor='simple'
    )

    # Method 3: Using full ImageNormalize objects
    from astropy.visualization import ImageNormalize

    normalizations = [
        ImageNormalize(interval=intervals[0], stretch=stretches[0]),
        ImageNormalize(interval=intervals[1], stretch=stretches[1]),
        ImageNormalize(interval=intervals[2], stretch=stretches[2])
    ]

    rgb = pipeline.process_with_normalizations(
        fits_files=['sii.fits', 'ha.fits', 'oiii.fits'],
        normalizations=normalizations,
        compositor='simple'
    )
    """)

    print("\n4. Result: Beautiful false-color narrowband image!")
    print("   Red channel shows sulfur emission (shock regions)")
    print("   Green channel shows hydrogen emission (H-II regions)")
    print("   Blue channel shows oxygen emission (hottest regions)")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Phase 4: Narrowband Palette Support - Comprehensive Examples")
    print("="*70)

    print("\nThis demo showcases the new narrowband imaging capabilities:")
    print("  ✓ Narrowband palette mapping (Hubble, HOO, natural, custom)")
    print("  ✓ Multi-band selection (>3 filters) with max-span, PCA, science goals")
    print("  ✓ Per-channel intervals and stretches")
    print("  ✓ Advanced false-color RGB composition")

    example_1_hubble_palette()
    example_2_hoo_bicolor()
    example_3_custom_palette()
    example_4_multi_band_selection_max_span()
    example_5_multi_band_selection_pca()
    example_6_science_driven_selection()
    example_7_per_channel_normalization()
    example_8_complete_narrowband_workflow()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Try with your own narrowband FITS files")
    print("  2. Experiment with different palettes and parameters")
    print("  3. Combine with color balance adjustments for final images")
    print("  4. See CLAUDE.md for more details on Phase 4 features")


if __name__ == '__main__':
    main()
