"""Phase 1 Demo: FITS Loading, Metadata, and Quality Assessment.

This script demonstrates the Phase 1 preprocessing classes:
- FITSMetadata: Extract mission-aware metadata
- FITSLoader: Load FITS files with automatic extension detection
- MissionAdapter: Handle mission-specific FITS conventions
- QualityAssessor: Analyze image quality without AI

Example usage:
    python examples/phase1_demo.py path/to/fits/file.fits
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astro_vision_composer.preprocessing import (
    FITSLoader,
    QualityAssessor,
    get_mission_adapter
)
from astro_vision_composer.utilities import FITSMetadata

def main():
    if len(sys.argv) < 2:
        print("Usage: python phase1_demo.py <fits_file>")
        print("\nThis demo will:")
        print("  1. Extract metadata from the FITS file")
        print("  2. Load the science data")
        print("  3. Assess image quality")
        print("  4. Display results\n")
        sys.exit(1)

    fits_file = Path(sys.argv[1])

    if not fits_file.exists():
        print(f"Error: File not found: {fits_file}")
        sys.exit(1)

    print("="*80)
    print("PHASE 1 DEMO: FITS Processing Suite")
    print("="*80)
    print(f"\nFile: {fits_file.name}")

    # Step 1: Extract metadata (fast, no data loading)
    print("\n" + "-"*80)
    print("Step 1: Extracting Metadata")
    print("-"*80)

    metadata_extractor = FITSMetadata()
    loader = FITSLoader()

    # Quick metadata extraction (no data loading)
    metadata = loader.get_metadata(fits_file)

    print(f"\n{metadata}")
    print(f"\n  Mission: {metadata.mission}")
    print(f"  Instrument: {metadata.instrument}")
    print(f"  Filter: {metadata.filter_name}")
    print(f"  Wavelength: {metadata.wavelength} nm" if metadata.wavelength else "  Wavelength: Unknown")
    print(f"  Exposure Time: {metadata.exposure_time}s" if metadata.exposure_time else "  Exposure Time: Unknown")
    print(f"  Pixel Scale: {metadata.pixel_scale:.3f} arcsec/pixel" if metadata.pixel_scale else "  Pixel Scale: Unknown")

    if metadata.warnings:
        print(f"\n  ⚠ Warnings:")
        for warning in metadata.warnings:
            print(f"    - {warning}")

    # Step 2: List file structure
    print("\n" + "-"*80)
    print("Step 2: FITS File Structure")
    print("-"*80)

    extensions = loader.list_extensions(fits_file)
    print(f"\nFound {len(extensions)} HDU(s):\n")
    for ext in extensions:
        shape_str = str(ext['shape']) if ext['shape'] else "No data"
        print(f"  [{ext['index']}] {ext['name']:20s} (ver={ext['ver']}) - {ext['type']:15s} - {shape_str}")

    # Step 3: Load data with mission adapter (if applicable)
    print("\n" + "-"*80)
    print("Step 3: Loading Science Data")
    print("-"*80)

    fits_data = loader.load(fits_file, load_error=True, load_dq=True)

    print(f"\n{fits_data}")
    print(f"\n  Science array: {fits_data.shape}, dtype={fits_data.dtype}")
    print(f"  Has error array: {fits_data.error is not None}")
    print(f"  Has DQ array: {fits_data.dq is not None}")
    print(f"  Has WCS: {fits_data.wcs is not None and fits_data.wcs.has_celestial}")

    # Show basic statistics
    import numpy as np
    print(f"\n  Data range: [{np.min(fits_data.science):.2e}, {np.max(fits_data.science):.2e}]")
    print(f"  Data median: {np.median(fits_data.science):.2e}")

    # Step 4: Mission adapter demo (if recognized mission)
    if metadata.mission:
        print("\n" + "-"*80)
        print("Step 4: Mission-Specific Adapter")
        print("-"*80)

        try:
            adapter = get_mission_adapter(metadata.mission)
            print(f"\n  Using: {adapter.__class__.__name__}")

            # Show what the adapter detected
            from astropy.io import fits as astropy_fits
            with astropy_fits.open(fits_file) as hdul:
                ext_info = adapter.get_all_extensions_info(hdul)

            print(f"\n  Science extension: {ext_info['science']['name']} (ver={ext_info['science']['version']})")
            if ext_info['error']:
                print(f"  Error extension: {ext_info['error']['name']} (ver={ext_info['error']['version']})")
            else:
                print(f"  Error extension: Not found")
            if ext_info['quality']:
                print(f"  Quality extension: {ext_info['quality']['name']} (ver={ext_info['quality']['version']})")
            else:
                print(f"  Quality extension: Not found")

        except ValueError as e:
            print(f"\n  ⚠ {e}")
            print("  Using generic FITS loading")

    # Step 5: Quality assessment
    print("\n" + "-"*80)
    print("Step 5: Quality Assessment (No AI!)")
    print("-"*80)

    assessor = QualityAssessor()
    quality_report = assessor.assess_quality(
        fits_data.science,
        dq=fits_data.dq
    )

    print(f"\n{quality_report}")
    print(f"\n  Overall Quality Score: {quality_report.quality_score:.1f}/10")
    print(f"  Signal-to-Noise Ratio: {quality_report.snr:.2f}")
    print(f"  Background Level: {quality_report.background_median:.2e}")
    print(f"  Background STD: {quality_report.background_std:.2e}")
    print(f"  Noise Estimate (MAD): {quality_report.noise_estimate:.2e}")
    print(f"  Dynamic Range: {quality_report.dynamic_range:.1f}x")
    print(f"  Saturated Pixels: {quality_report.saturated_pixels:,} ({quality_report.saturation_fraction*100:.2f}%)")

    if quality_report.has_negative_values:
        print(f"  ⚠ Contains negative values (min={quality_report.data_min:.2e})")

    if quality_report.warnings:
        print(f"\n  ⚠ Quality Warnings:")
        for warning in quality_report.warnings:
            print(f"    - {warning}")

    # Interpretation
    print("\n  Quality Interpretation:")
    if quality_report.quality_score >= 8:
        print("    ✓ Excellent quality image")
    elif quality_report.quality_score >= 6:
        print("    ✓ Good quality image")
    elif quality_report.quality_score >= 4:
        print("    ~ Acceptable quality image")
    else:
        print("    ✗ Poor quality image - check warnings")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n  File: {fits_file.name}")
    print(f"  Mission: {metadata.mission or 'Unknown'}")
    print(f"  Filter: {metadata.filter_name or 'Unknown'}")
    print(f"  Shape: {fits_data.shape}")
    print(f"  Quality: {quality_report.quality_score:.1f}/10")
    print(f"  SNR: {quality_report.snr:.1f}")
    print(f"  Saturation: {quality_report.saturation_fraction*100:.2f}%")
    print(f"\n  Phase 1 classes demonstrated:")
    print(f"    ✓ FITSMetadata - Mission-aware metadata extraction")
    print(f"    ✓ FITSLoader - Intelligent FITS loading")
    print(f"    ✓ MissionAdapter - Mission-specific conventions")
    print(f"    ✓ QualityAssessor - Statistical quality analysis")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
