"""Phase 1 Demo: Batch FITS Processing

Process all FITS files in a directory:
- Extract metadata
- Load and validate data
- Assess quality
- Save processed data and reports

Usage:
    python examples/phase1_demo.py <input_dir>

Example:
    python examples/phase1_demo.py examples/data/NOIRLab/examples/edu008/data

Output structure:
    <input_dir>/output/phase1/
        ├── <file1>_processed.fits
        ├── <file1>_metadata.json
        ├── <file1>_quality.json
        ├── <file2>_processed.fits
        └── ...
        └── phase1_summary.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from astropy.io import fits as astropy_fits

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from astro_vision_composer.preprocessing import FITSLoader, QualityAssessor
from astro_vision_composer.utilities import FITSMetadata


def find_fits_files(input_dir):
    """Find FITS files in directory, excluding output folder."""
    fits_patterns = ['*.fits', '*.fit', '*.fts']
    fits_files = []

    for pattern in fits_patterns:
        for file in input_dir.rglob(pattern):
            # Skip files in output directory
            if 'output' not in file.parts:
                fits_files.append(file)

    return sorted(fits_files)


def process_single_fits(fits_path, output_dir, loader, assessor):
    """Process a single FITS file through Phase 1."""

    print(f"\n{'='*80}")
    print(f"Processing: {fits_path.name}")
    print(f"{'='*80}")

    # Create output filename base (remove .fits extension)
    output_base = output_dir / fits_path.stem

    results = {
        'input_file': str(fits_path),
        'timestamp': datetime.now().isoformat(),
        'phase': 'phase1'
    }

    try:
        # Step 1: Extract metadata
        print("\n[1/4] Extracting metadata...")
        metadata = loader.get_metadata(fits_path)

        print(f"  Mission: {metadata.mission or 'Unknown'}")
        print(f"  Filter: {metadata.filter_name or 'Unknown'}")
        print(f"  Wavelength: {metadata.wavelength} nm" if metadata.wavelength else "  Wavelength: Unknown")

        # Save metadata to JSON
        metadata_file = output_base.with_name(f"{output_base.name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump({
                'mission': metadata.mission,
                'instrument': metadata.instrument,
                'filter': metadata.filter_name,
                'wavelength': metadata.wavelength,
                'exposure_time': metadata.exposure_time,
                'pixel_scale': metadata.pixel_scale,
                'warnings': metadata.warnings
            }, f, indent=2)

        results['metadata'] = str(metadata_file)

        # Step 2: Load FITS data
        print("\n[2/4] Loading FITS data...")
        fits_data = loader.load(fits_path, load_error=True, load_dq=True)

        print(f"  Shape: {fits_data.shape}")
        print(f"  Has error: {fits_data.error is not None}")
        print(f"  Has DQ: {fits_data.dq is not None}")
        print(f"  Has WCS: {fits_data.wcs is not None and fits_data.wcs.has_celestial}")

        # Step 3: Quality assessment
        print("\n[3/4] Assessing quality...")
        quality = assessor.assess_quality(fits_data.science, dq=fits_data.dq)

        print(f"  Quality Score: {quality.quality_score:.1f}/10")
        print(f"  SNR: {quality.snr:.1f}")
        print(f"  Background: {quality.background_median:.2e}")
        print(f"  Saturation: {quality.saturation_fraction*100:.2f}%")

        # Save quality report
        quality_file = output_base.with_name(f"{output_base.name}_quality.json")
        with open(quality_file, 'w') as f:
            json.dump({
                'quality_score': float(quality.quality_score),
                'snr': float(quality.snr),
                'background_median': float(quality.background_median),
                'background_std': float(quality.background_std),
                'noise_estimate': float(quality.noise_estimate),
                'dynamic_range': float(quality.dynamic_range),
                'saturation_fraction': float(quality.saturation_fraction),
                'saturated_pixels': int(quality.saturated_pixels),
                'has_negative_values': quality.has_negative_values,
                'warnings': quality.warnings
            }, f, indent=2)

        results['quality'] = str(quality_file)
        results['quality_score'] = float(quality.quality_score)
        results['snr'] = float(quality.snr)

        # Step 4: Save processed FITS (for Phase 2)
        print("\n[4/4] Saving processed FITS...")
        output_fits = output_base.with_name(f"{output_base.name}_processed.fits")

        # Create HDU list with science, error, DQ
        hdu_list = [astropy_fits.PrimaryHDU(fits_data.science, header=fits_data.header)]

        if fits_data.error is not None:
            hdu_list.append(astropy_fits.ImageHDU(fits_data.error, name='ERR'))

        if fits_data.dq is not None:
            hdu_list.append(astropy_fits.ImageHDU(fits_data.dq, name='DQ'))

        hdul = astropy_fits.HDUList(hdu_list)
        hdul.writeto(output_fits, overwrite=True)
        hdul.close()

        print(f"  ✓ Saved: {output_fits.name}")

        results['processed_fits'] = str(output_fits)
        results['status'] = 'success'

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        results['status'] = 'failed'
        results['error'] = str(e)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase1_demo.py <input_dir>")
        print("\nExample:")
        print("  python phase1_demo.py examples/data/NOIRLab/examples/edu008/data")
        print("\nThis will process all FITS files found in the directory and save:")
        print("  - Metadata JSON files")
        print("  - Quality assessment JSON files")
        print("  - Processed FITS files (for Phase 2)")
        print("  - Summary report")
        sys.exit(1)

    input_dir = Path(sys.argv[1])

    if not input_dir.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"ERROR: Not a directory: {input_dir}")
        sys.exit(1)

    # Find all FITS files
    print(f"{'='*80}")
    print(f"PHASE 1: FITS Preprocessing")
    print(f"{'='*80}")
    print(f"\nInput directory: {input_dir}")
    print("\nSearching for FITS files...")

    fits_files = find_fits_files(input_dir)

    if not fits_files:
        print(f"\n✗ No FITS files found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(fits_files)} FITS file(s):")
    for f in fits_files:
        print(f"  - {f.relative_to(input_dir)}")

    # Create output directory
    output_dir = input_dir / "output" / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Initialize components
    loader = FITSLoader()
    assessor = QualityAssessor()

    # Process each file
    all_results = []

    for fits_file in fits_files:
        result = process_single_fits(fits_file, output_dir, loader, assessor)
        all_results.append(result)

    # Create summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] == 'failed']

    print(f"\nProcessed: {len(fits_files)} files")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")

    if successful:
        avg_quality = np.mean([r['quality_score'] for r in successful])
        avg_snr = np.mean([r['snr'] for r in successful])
        print(f"\nAverage Quality: {avg_quality:.1f}/10")
        print(f"Average SNR: {avg_snr:.1f}")

    # Save summary
    summary_file = output_dir / "phase1_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'total_files': len(fits_files),
        'successful': len(successful),
        'failed': len(failed),
        'results': all_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved: {summary_file}")
    print(f"\nOutput files in: {output_dir}/")
    print(f"{'='*80}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
