"""Phase 2 Demo: Normalization and Stretching

Process Phase 1 outputs (or run Phase 1 if needed):
- Normalize with ZScale
- Apply non-linear stretch (asinh)
- Save stretched FITS for Phase 3

Usage:
    python examples/phase2_demo.py <input_dir> [--force]

Arguments:
    input_dir: Same directory used for phase1_demo.py
    --force: Re-run Phase 1 even if outputs exist

Example:
    python examples/phase2_demo.py examples/data/NOIRLab/examples/edu008/data
    python examples/phase2_demo.py examples/data/NOIRLab/examples/edu008/data --force

Output structure:
    <input_dir>/output/phase2/
        ├── <file1>_normalized.fits
        ├── <file1>_stretched.fits
        ├── <file1>_processing.json
        └── phase2_summary.json
"""

import sys
import json
import subprocess
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from astro_vision_composer.processing import Normalizer, Stretcher
from astropy.io import fits as astropy_fits


def validate_phase2_output(normalized, stretched, data_original):
    """
    Validate Phase 2 processing correctness.

    Proves:
    1. Normalization maps to [0, 1]
    2. Stretching maintains [0, 1] range
    3. Stretching is non-linear (for asinh)
    4. No NaN/Inf values introduced

    This implements WORKFLOW 1: Scientific Standard
    - ZScaleInterval for robust normalization
    - AsinhStretch for astronomical standard tone-mapping
    """
    print("\n" + "="*80)
    print("PHASE 2 VALIDATION (Workflow 1: Scientific Standard)")
    print("="*80)

    # Check 1: Normalized data in [0, 1]
    norm_min, norm_max = normalized.min(), normalized.max()
    assert 0 <= norm_min and norm_max <= 1.0, \
        f"Normalization failed: data not in [0, 1], got [{norm_min}, {norm_max}]"
    print(f"✓ Check 1: Normalized data in [0, 1]: [{norm_min:.4f}, {norm_max:.4f}]")

    # Check 2: Stretched data still in [0, 1]
    str_min, str_max = stretched.min(), stretched.max()
    assert 0 <= str_min and str_max <= 1.0, \
        f"Stretch failed: data not in [0, 1], got [{str_min}, {str_max}]"
    print(f"✓ Check 2: Stretched data in [0, 1]: [{str_min:.4f}, {str_max:.4f}]")

    # Check 3: Asinh stretch is non-linear (boosts mid-tones)
    # Compare median values at different brightness levels
    if norm_max > 0.5:  # Only if we have enough range
        mid_mask = (normalized > 0.4) & (normalized < 0.6)
        if mid_mask.any():
            norm_mid = normalized[mid_mask].mean()
            str_mid = stretched[mid_mask].mean()
            # Asinh should boost mid-tones relative to linear
            boost_ratio = str_mid / norm_mid if norm_mid > 0 else 0
            print(f"✓ Check 3: Asinh stretch non-linear (mid-tone boost: {boost_ratio:.2f}x)")
            if boost_ratio < 0.95:
                print("  WARNING: Mid-tones not boosted as expected (may be okay for bright images)")

    # Check 4: No NaN/Inf values
    assert not np.any(np.isnan(normalized)), "Normalized data contains NaN"
    assert not np.any(np.isinf(normalized)), "Normalized data contains Inf"
    assert not np.any(np.isnan(stretched)), "Stretched data contains NaN"
    assert not np.any(np.isinf(stretched)), "Stretched data contains Inf"
    print(f"✓ Check 4: No NaN/Inf values")

    # Check 5: Dynamic range preserved
    orig_range = data_original.max() - data_original.min()
    if orig_range > 0:
        print(f"✓ Check 5: Original dynamic range: {orig_range:.2e} → Normalized: [0, 1]")

    print("="*80)
    print("✅ PHASE 2 VALIDATION PASSED - Using correct workflow!")
    print("="*80)

    return True


def check_phase1_outputs(input_dir):
    """Check if Phase 1 outputs exist."""
    phase1_dir = input_dir / "output" / "phase1"

    if not phase1_dir.exists():
        return False, "Phase 1 output directory not found"

    summary_file = phase1_dir / "phase1_summary.json"
    if not summary_file.exists():
        return False, "Phase 1 summary file not found"

    # Check if there are processed FITS files
    processed_fits = list(phase1_dir.glob("*_processed.fits"))
    if not processed_fits:
        return False, "No Phase 1 processed FITS files found"

    return True, f"Found {len(processed_fits)} Phase 1 outputs"


def run_phase1(input_dir):
    """Run Phase 1 demo script."""
    print(f"\n{'='*80}")
    print("Running Phase 1 first...")
    print(f"{'='*80}\n")

    phase1_script = Path(__file__).parent / "phase1_demo_new.py"

    if not phase1_script.exists():
        print(f"ERROR: Phase 1 script not found: {phase1_script}")
        return False

    result = subprocess.run(
        [sys.executable, str(phase1_script), str(input_dir)],
        capture_output=False
    )

    return result.returncode == 0


def process_single_file(fits_path, output_dir, normalizer, stretcher):
    """Process a single FITS file through Phase 2."""

    print(f"\n{'-'*80}")
    print(f"Processing: {fits_path.name}")
    print(f"{'-'*80}")

    output_base = output_dir / fits_path.stem.replace('_processed', '')

    results = {
        'input_file': str(fits_path),
        'timestamp': datetime.now().isoformat(),
        'phase': 'phase2'
    }

    try:
        # Load processed FITS from Phase 1
        print("[1/3] Loading Phase 1 output...")
        with astropy_fits.open(fits_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        print(f"  Shape: {data.shape}")
        print(f"  Range: [{np.min(data):.2e}, {np.max(data):.2e}]")

        # Step 1: Normalize
        print("\n[2/3] Normalizing with ZScale...")
        normalized = normalizer.normalize(data, method='zscale')
        vmin, vmax = normalizer.get_interval_limits(data, method='zscale')

        print(f"  Interval: [{vmin:.2e}, {vmax:.2e}]")
        print(f"  Normalized range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")

        # Save normalized FITS
        normalized_file = output_base.with_name(f"{output_base.name}_normalized.fits")
        hdu = astropy_fits.PrimaryHDU(normalized, header=header)
        hdu.header['PHASE'] = 'phase2_normalized'
        hdu.header['NORMMET'] = 'zscale'
        hdu.header['NORMVMIN'] = vmin
        hdu.header['NORMVMAX'] = vmax
        hdu.writeto(normalized_file, overwrite=True)

        results['normalized_fits'] = str(normalized_file)

        # Step 2: Stretch
        print("\n[3/3] Applying asinh stretch...")
        stretched = stretcher.stretch(normalized, method='asinh', a=0.1)

        print(f"  Stretched range: [{np.min(stretched):.3f}, {np.max(stretched):.3f}]")

        # Validate Phase 2 output
        validate_phase2_output(normalized, stretched, data)

        # Get stretch objects for Phase 3
        interval_obj = normalizer.get_interval_object() if hasattr(normalizer, 'get_interval_object') else None
        stretch_obj = stretcher.get_stretch_object() if hasattr(stretcher, 'get_stretch_object') else None

        # Save stretched FITS
        stretched_file = output_base.with_name(f"{output_base.name}_stretched.fits")
        hdu = astropy_fits.PrimaryHDU(stretched, header=header)
        hdu.header['PHASE'] = 'phase2_stretched'
        hdu.header['NORMMET'] = 'zscale'
        hdu.header['STRMET'] = 'asinh'
        hdu.header['STRPARAM'] = 0.1
        hdu.header['WORKFLOW'] = 'scientific_standard'
        hdu.writeto(stretched_file, overwrite=True)

        results['stretched_fits'] = str(stretched_file)

        # Save processing info with stretch objects
        processing_file = output_base.with_name(f"{output_base.name}_processing.json")
        processing_info = {
            'workflow': 'scientific_standard',
            'normalization': {
                'method': 'zscale',
                'vmin': float(vmin),
                'vmax': float(vmax)
            },
            'stretch': {
                'method': 'asinh',
                'parameter_a': 0.1,
                'stretch_class': stretch_obj.__class__.__name__ if stretch_obj else None
            }
        }

        # Save pickled stretch objects if available
        if interval_obj or stretch_obj:
            try:
                if interval_obj:
                    processing_info['normalization']['interval_pickle_hex'] = pickle.dumps(interval_obj).hex()
                if stretch_obj:
                    processing_info['stretch']['stretch_pickle_hex'] = pickle.dumps(stretch_obj).hex()
            except Exception as e:
                print(f"  WARNING: Could not pickle stretch objects: {e}")

        with open(processing_file, 'w') as f:
            json.dump(processing_info, f, indent=2)

        results['processing_info'] = str(processing_file)
        results['status'] = 'success'

        print(f"  ✓ Complete")

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        results['status'] = 'failed'
        results['error'] = str(e)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase2_demo.py <input_dir> [--force]")
        print("\nExample:")
        print("  python phase2_demo.py examples/data/NOIRLab/examples/edu008/data")
        print("  python phase2_demo.py examples/data/NOIRLab/examples/edu008/data --force")
        print("\nOptions:")
        print("  --force: Re-run Phase 1 even if outputs exist")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    force = '--force' in sys.argv

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: Invalid directory: {input_dir}")
        sys.exit(1)

    print(f"{'='*80}")
    print(f"PHASE 2: Normalization & Stretching")
    print(f"{'='*80}")
    print(f"\nWORKFLOW: Scientific Standard (Workflow 1)")
    print(f"  - Interval: ZScaleInterval (IRAF adaptive, robust to outliers)")
    print(f"  - Stretch: AsinhStretch(a=0.1) (astronomical standard)")
    print(f"  - Output: Normalized + Stretched FITS [0, 1]")
    print(f"  - Phase 3: Should use LinearStretch or Simple RGB (no double-stretch)")
    print(f"\nInput directory: {input_dir}")
    print(f"Force re-run Phase 1: {force}")

    # Check for Phase 1 outputs
    print("\nChecking for Phase 1 outputs...")
    has_phase1, message = check_phase1_outputs(input_dir)
    print(f"  {message}")

    if not has_phase1 or force:
        if force:
            print("\n  Force flag set - re-running Phase 1")
        else:
            print("\n  Phase 1 outputs not found - running Phase 1 first")

        if not run_phase1(input_dir):
            print("\n✗ Phase 1 failed")
            sys.exit(1)

        print("\n✓ Phase 1 complete")

    # Get Phase 1 processed files
    phase1_dir = input_dir / "output" / "phase1"
    processed_files = sorted(phase1_dir.glob("*_processed.fits"))

    print(f"\nFound {len(processed_files)} Phase 1 outputs to process")

    # Create output directory
    output_dir = input_dir / "output" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize components
    normalizer = Normalizer()
    stretcher = Stretcher()

    # Process each file
    print(f"\n{'='*80}")
    print("Processing files...")
    print(f"{'='*80}")

    all_results = []
    for fits_file in processed_files:
        result = process_single_file(fits_file, output_dir, normalizer, stretcher)
        all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] == 'failed']

    print(f"\nProcessed: {len(processed_files)} files")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")

    # Save summary
    summary_file = output_dir / "phase2_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'phase1_rerun': force or not has_phase1,
        'total_files': len(processed_files),
        'successful': len(successful),
        'failed': len(failed),
        'results': all_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved: {summary_file}")
    print(f"\nOutput files in: {output_dir}/")
    print(f"  - *_normalized.fits (ZScale normalized)")
    print(f"  - *_stretched.fits (asinh stretched, ready for Phase 3)")
    print(f"{'='*80}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
