"""Pipeline Demo: Unified workflow using ProcessingPipeline

Demonstrates the ProcessingPipeline class which implements all validated workflows:
- Scientific Standard
- SDSS/Lupton Method
- Aesthetic/Publication
- Narrowband Science (if applicable)
- Custom

By default, runs ALL workflows to demonstrate differences.
Use --mode to test a single workflow.

Usage:
    python examples/pipeline_demo.py <input_dir> [--mode MODE]

Arguments:
    input_dir: Directory containing FITS files (e.g., 502nmos.fits, 656nmos.fits, 673nmos.fits)
    --mode: Run single workflow (scientific, sdss, aesthetic, narrowband, custom) [default: all]

Examples:
    # Run all workflows (default)
    python examples/pipeline_demo.py examples/data/NOIRLab/examples/edu008/data

    # Run single workflow
    python examples/pipeline_demo.py examples/data/NOIRLab/examples/edu008/data --mode sdss
    python examples/pipeline_demo.py examples/data/NOIRLab/examples/edu008/data --mode aesthetic

Output structure (all modes create separate files):
    <input_dir>/output/pipeline/
        ├── rgb_composite_scientific.png/tif
        ├── rgb_composite_sdss.png/tif
        ├── rgb_composite_aesthetic.png/tif
        ├── rgb_composite_narrowband.png/tif (if applicable)
        ├── rgb_composite_custom.png/tif
        ├── processing_history_*.txt
        └── workflow_metadata_*.json
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from astro_vision_composer import ProcessingPipeline


def run_single_workflow(mode: str, fits_files, output_dir, descriptions):
    """Run a single workflow mode.

    Args:
        mode: Workflow mode name
        fits_files: List of FITS file paths
        output_dir: Output directory path
        descriptions: Dictionary of workflow descriptions

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"WORKFLOW {mode.upper()}")
    print(f"{'='*80}")
    print(descriptions[mode])
    print(f"{'='*80}")

    # Initialize pipeline
    print(f"\nInitializing {mode} pipeline...")
    pipeline = ProcessingPipeline(mode=mode)

    # Process to RGB
    print("Processing...")
    try:
        rgb = pipeline.process_to_rgb(
            fits_files=fits_files,
            output_dir=output_dir
        )

        print(f"\n[SUCCESS] {mode.upper()} workflow complete!")
        print(f"   Shape: {rgb.shape}, Dtype: {rgb.dtype}, Range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print(f"   Output: rgb_composite_{mode}.png/tif")

        return True

    except Exception as e:
        print(f"\n[FAILED] {mode.upper()} workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Demo: Test all workflows or run a single workflow"
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help="Input directory containing FITS files"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['scientific', 'sdss', 'aesthetic', 'narrowband', 'custom', 'all'],
        default='all',
        help="Workflow mode (default: all - runs all workflows)"
    )

    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    mode = args.mode

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: Invalid directory: {input_dir}")
        sys.exit(1)

    # Determine which modes to run
    if mode == 'all':
        modes_to_run = ['scientific', 'sdss', 'aesthetic', 'custom']
        print("="*80)
        print("PIPELINE DEMO: ALL WORKFLOWS")
        print("="*80)
        print(f"\nWill test {len(modes_to_run)} workflows:")
        for m in modes_to_run:
            print(f"  - {m}")
    else:
        modes_to_run = [mode]
        print("="*80)
        print(f"PIPELINE DEMO: {mode.upper()} WORKFLOW")
        print("="*80)

    print(f"\nInput directory: {input_dir}")

    # Find FITS files
    print("\nSearching for FITS files...")
    fits_files = []

    # Look in subdirectories (NOIRLab structure)
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            for fits_file in subdir.glob("*.fits"):
                if not fits_file.name.endswith(('_processed.fits', '_normalized.fits', '_stretched.fits')):
                    fits_files.append(fits_file)

    # Also look in root directory
    for fits_file in input_dir.glob("*.fits"):
        if not fits_file.name.endswith(('_processed.fits', '_normalized.fits', '_stretched.fits')):
            if fits_file not in fits_files:
                fits_files.append(fits_file)

    if len(fits_files) < 3:
        print(f"\nERROR: Need at least 3 FITS files for RGB, found {len(fits_files)}")
        print("Files found:")
        for f in fits_files:
            print(f"  - {f}")
        sys.exit(1)

    print(f"\nFound {len(fits_files)} FITS file(s):")
    for f in fits_files:
        print(f"  - {f.name}")

    # Create output directory
    output_dir = input_dir / "output" / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Workflow descriptions
    descriptions = {
        'scientific': (
            "ZScaleInterval + AsinhStretch + Lupton(LinearStretch)\n"
            "  Goal: Preserve photometry, minimize artifacts, reproducible"
        ),
        'sdss': (
            "ZScaleInterval only + Lupton(LuptonAsinhZscaleStretch)\n"
            "  Goal: Follow SDSS algorithm, automatic parameter tuning"
        ),
        'aesthetic': (
            "AsymmetricPercentileInterval(2, 99.5) + HistEqStretch + Lupton\n"
            "  Goal: Maximum visual impact, dramatic presentation"
        ),
        'narrowband': (
            "Per-channel optimization + Simple RGB\n"
            "  Goal: False-color composition (e.g., Hubble Palette)"
        ),
        'custom': (
            "User-defined parameters\n"
            "  Goal: Full control over every parameter"
        )
    }

    # Run workflows
    results = {}
    for workflow_mode in modes_to_run:
        success = run_single_workflow(
            mode=workflow_mode,
            fits_files=fits_files,
            output_dir=output_dir,
            descriptions=descriptions
        )
        results[workflow_mode] = success

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)

    for workflow_mode, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {workflow_mode.upper():<15} {status}")

    print(f"\nTotal: {success_count}/{total_count} workflows completed successfully")

    if success_count > 0:
        print(f"\nOutput files in: {output_dir}/")
        for workflow_mode, success in results.items():
            if success:
                print(f"  - rgb_composite_{workflow_mode}.png/tif")
                print(f"  - processing_history_{workflow_mode}.txt")
                print(f"  - workflow_metadata_{workflow_mode}.json")

    print("="*80)

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
