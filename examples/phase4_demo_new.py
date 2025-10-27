"""Phase 4 Demo: Advanced Image Enhancement & Pickle Workflow Test

This demo has two main purposes:
1. TEST PICKLE WORKFLOW: Load pickled stretch/interval objects from Phase 2,
   demonstrate they can be unpickled and reused
2. DEMONSTRATE PHASE 4 FEATURES: Apply advanced enhancements (CLAHE, unsharp masking,
   color balancing, preview generation)

Usage:
    python examples/phase4_demo_new.py <input_dir> [--force]

Arguments:
    input_dir: Same directory used for phase1/phase2/phase3 demos
    --force: Re-run Phase 3 if needed, overwrite existing Phase 4 outputs

Example:
    python examples/phase4_demo_new.py examples/data/NOIRLab/examples/edu008/data
    python examples/phase4_demo_new.py examples/data/NOIRLab/examples/edu008/data --force

Output structure:
    <input_dir>/output/phase4/
        ├── rgb_enhanced_clahe.png
        ├── rgb_enhanced_unsharp.png
        ├── rgb_color_balanced.png
        ├── preview_thumbnail.png
        ├── preview_medium.png
        ├── pickle_test_results.json
        └── phase4_summary.json
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from astro_vision_composer.processing import Normalizer, Stretcher, Enhancer
from astro_vision_composer.postprocessing import (
    ColorBalancer, PreviewGenerator, ImageExporter
)
from astropy.io import fits as astropy_fits
import subprocess


def check_phase3_outputs(input_dir):
    """Check if Phase 3 outputs exist."""
    phase3_dir = input_dir / "output" / "phase3"

    if not phase3_dir.exists():
        return False, "Phase 3 output directory not found"

    # Check for RGB composite
    rgb_files = list(phase3_dir.glob("rgb_composite_*.png"))
    if not rgb_files:
        return False, "No RGB composite found in Phase 3 output"

    return True, f"Found {len(rgb_files)} RGB composite(s)"


def check_phase4_outputs(output_dir):
    """Check if Phase 4 outputs already exist."""
    if not output_dir.exists():
        return False, "Phase 4 output directory does not exist"

    # Check for key output files
    expected_files = [
        "rgb_enhanced_clahe.png",
        "rgb_enhanced_unsharp.png",
        "rgb_color_balanced.png",
        "phase4_summary.json"
    ]

    existing = [f for f in expected_files if (output_dir / f).exists()]

    if len(existing) == len(expected_files):
        return True, "All Phase 4 outputs exist"
    elif len(existing) > 0:
        return True, f"Some Phase 4 outputs exist ({len(existing)}/{len(expected_files)})"
    else:
        return False, "No Phase 4 outputs found"


def run_phase3(input_dir, force=False):
    """Run Phase 3 demo script."""
    print("\n" + "="*80)
    print("Running Phase 3 first...")
    print("="*80 + "\n")

    phase3_script = Path(__file__).parent / "phase3_demo_new.py"
    if not phase3_script.exists():
        print(f"ERROR: Phase 3 script not found: {phase3_script}")
        return False

    args = [sys.executable, str(phase3_script), str(input_dir)]
    if force:
        args.append('--force')

    result = subprocess.run(args, capture_output=False)
    return result.returncode == 0


def test_pickle_workflow(phase2_dir):
    """Test that pickled objects from Phase 2 can be restored and used.

    This validates the complete workflow continuity system.
    """
    print("\n" + "="*80)
    print("PICKLE WORKFLOW TEST")
    print("="*80)
    print("\nPurpose: Verify that stretch/interval objects can be serialized,")
    print("         stored in JSON, and later restored for reuse in other phases.")

    # Find processing metadata files
    processing_files = list(phase2_dir.glob("*_processing.json"))

    if not processing_files:
        print("\n⚠️  No Phase 2 processing metadata found - cannot test pickle workflow")
        return None

    print(f"\nFound {len(processing_files)} processing metadata file(s)")

    # Load first metadata file
    metadata_file = processing_files[0]
    print(f"\nLoading: {metadata_file.name}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    results = {
        'metadata_file': str(metadata_file),
        'workflow': metadata.get('workflow', 'unknown'),
        'tests': []
    }

    # Test 1: Unpickle interval object
    print("\n" + "-"*80)
    print("TEST 1: Unpickle Interval Object")
    print("-"*80)

    if 'interval_pickle_hex' in metadata.get('normalization', {}):
        try:
            interval_hex = metadata['normalization']['interval_pickle_hex']
            print(f"Found pickled interval: {len(interval_hex)} hex chars")

            # Unpickle
            interval_obj = pickle.loads(bytes.fromhex(interval_hex))
            print(f"✓ Unpickled: {type(interval_obj).__name__}")

            # Test it works - load a FITS file and apply
            stretched_files = list(phase2_dir.glob("*_stretched.fits"))
            if stretched_files:
                test_file = phase2_dir.parent / 'phase1' / stretched_files[0].name.replace('_stretched', '_processed')
                if test_file.exists():
                    with astropy_fits.open(test_file) as hdul:
                        test_data = hdul[0].data

                    vmin, vmax = interval_obj.get_limits(test_data)
                    print(f"✓ Applied to test data: interval=[{vmin:.2e}, {vmax:.2e}]")

                    results['tests'].append({
                        'test': 'interval_unpickle',
                        'status': 'passed',
                        'object_type': type(interval_obj).__name__,
                        'vmin': float(vmin),
                        'vmax': float(vmax)
                    })
                    print("\n✅ Interval unpickle test PASSED")
                else:
                    print(f"⚠️  Test data file not found: {test_file}")
            else:
                print("⚠️  No stretched files found for testing")

        except Exception as e:
            print(f"\n❌ Interval unpickle test FAILED: {e}")
            results['tests'].append({
                'test': 'interval_unpickle',
                'status': 'failed',
                'error': str(e)
            })
    else:
        print("⚠️  No pickled interval object found in metadata")
        print("   (This is expected if Phase 2 was run before pickle support was added)")
        results['tests'].append({
            'test': 'interval_unpickle',
            'status': 'skipped',
            'reason': 'No pickled object in metadata'
        })

    # Test 2: Unpickle stretch object
    print("\n" + "-"*80)
    print("TEST 2: Unpickle Stretch Object")
    print("-"*80)

    if 'stretch_pickle_hex' in metadata.get('stretch', {}):
        try:
            stretch_hex = metadata['stretch']['stretch_pickle_hex']
            print(f"Found pickled stretch: {len(stretch_hex)} hex chars")

            # Unpickle
            stretch_obj = pickle.loads(bytes.fromhex(stretch_hex))
            print(f"✓ Unpickled: {type(stretch_obj).__name__}")

            # Test it works
            test_input = np.linspace(0, 1, 10)
            test_output = stretch_obj(test_input)
            print(f"✓ Applied to test data: [0.0, 0.1, ..., 1.0]")
            print(f"  → [{test_output[0]:.3f}, {test_output[1]:.3f}, ..., {test_output[-1]:.3f}]")

            results['tests'].append({
                'test': 'stretch_unpickle',
                'status': 'passed',
                'object_type': type(stretch_obj).__name__,
                'test_input': test_input.tolist(),
                'test_output': test_output.tolist()
            })
            print("\n✅ Stretch unpickle test PASSED")

        except Exception as e:
            print(f"\n❌ Stretch unpickle test FAILED: {e}")
            results['tests'].append({
                'test': 'stretch_unpickle',
                'status': 'failed',
                'error': str(e)
            })
    else:
        print("⚠️  No pickled stretch object found in metadata")
        print("   (This is expected if Phase 2 was run before pickle support was added)")
        results['tests'].append({
            'test': 'stretch_unpickle',
            'status': 'skipped',
            'reason': 'No pickled object in metadata'
        })

    # Summary
    print("\n" + "="*80)
    passed = sum(1 for t in results['tests'] if t['status'] == 'passed')
    failed = sum(1 for t in results['tests'] if t['status'] == 'failed')
    skipped = sum(1 for t in results['tests'] if t['status'] == 'skipped')

    if failed > 0:
        print(f"❌ PICKLE WORKFLOW TEST: {failed} failed, {passed} passed, {skipped} skipped")
    elif passed > 0:
        print(f"✅ PICKLE WORKFLOW TEST: All {passed} tests passed! ({skipped} skipped)")
    else:
        print(f"⚠️  PICKLE WORKFLOW TEST: All tests skipped (re-run Phase 2 to enable)")
    print("="*80)

    return results


def apply_enhancements(rgb, output_dir):
    """Apply Phase 4 enhancements to RGB composite.

    Demonstrates:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Unsharp masking
    - Color balancing
    """
    print("\n" + "="*80)
    print("PHASE 4: ADVANCED IMAGE ENHANCEMENTS")
    print("="*80)

    enhancer = Enhancer()
    color_balancer = ColorBalancer()
    exporter = ImageExporter()

    results = []

    # Enhancement 1: CLAHE
    print("\n[1/3] Applying CLAHE enhancement...")
    print("  Purpose: Enhance local contrast while limiting noise amplification")

    try:
        rgb_clahe = enhancer.apply_clahe(
            rgb,
            clip_limit=0.03,
            kernel_size=128  # Changed from tile_grid_size
        )

        clahe_file = output_dir / "rgb_enhanced_clahe.png"
        exporter.save_png(rgb_clahe, clahe_file, bit_depth=8)
        print(f"  ✓ Saved: {clahe_file.name}")

        results.append({
            'enhancement': 'clahe',
            'status': 'success',
            'file': str(clahe_file),
            'parameters': {'clip_limit': 0.03, 'kernel_size': 128}
        })
    except Exception as e:
        print(f"  ✗ CLAHE failed: {e}")
        results.append({'enhancement': 'clahe', 'status': 'failed', 'error': str(e)})

    # Enhancement 2: Unsharp masking
    print("\n[2/3] Applying unsharp masking...")
    print("  Purpose: Sharpen fine details and enhance edges")

    try:
        rgb_unsharp = enhancer.unsharp_mask(
            rgb,
            sigma=2.0,      # Changed from radius
            strength=1.5    # Changed from amount
        )

        unsharp_file = output_dir / "rgb_enhanced_unsharp.png"
        exporter.save_png(rgb_unsharp, unsharp_file, bit_depth=8)
        print(f"  ✓ Saved: {unsharp_file.name}")

        results.append({
            'enhancement': 'unsharp_mask',
            'status': 'success',
            'file': str(unsharp_file),
            'parameters': {'sigma': 2.0, 'strength': 1.5}
        })
    except Exception as e:
        print(f"  ✗ Unsharp masking failed: {e}")
        results.append({'enhancement': 'unsharp_mask', 'status': 'failed', 'error': str(e)})

    # Enhancement 3: Color balancing
    print("\n[3/3] Applying color balancing...")
    print("  Purpose: Adjust white balance and saturation for aesthetic appeal")

    try:
        rgb_balanced = color_balancer.white_balance(rgb)  # Changed from adjust_white_balance
        rgb_balanced = color_balancer.adjust_saturation(rgb_balanced, factor=1.2)

        balanced_file = output_dir / "rgb_color_balanced.png"
        exporter.save_png(rgb_balanced, balanced_file, bit_depth=8)
        print(f"  ✓ Saved: {balanced_file.name}")

        results.append({
            'enhancement': 'color_balance',
            'status': 'success',
            'file': str(balanced_file),
            'parameters': {'white_balance': 'gray_world', 'saturation': 1.2}
        })
    except Exception as e:
        print(f"  ✗ Color balancing failed: {e}")
        results.append({'enhancement': 'color_balance', 'status': 'failed', 'error': str(e)})

    return results


def generate_previews(rgb, output_dir):
    """Generate preview images at different resolutions."""
    print("\n" + "="*80)
    print("PREVIEW GENERATION")
    print("="*80)

    preview_gen = PreviewGenerator()
    results = []

    # Thumbnail
    print("\n[1/2] Generating thumbnail (256px)...")
    try:
        thumb = preview_gen.create_thumbnail(rgb, max_size=256)  # Changed from size
        thumb_file = output_dir / "preview_thumbnail.png"
        # Use ImageExporter instead of preview_gen.save_preview
        from astro_vision_composer.postprocessing import ImageExporter
        img_exporter = ImageExporter()
        img_exporter.save_png(thumb, thumb_file, bit_depth=8)
        print(f"  ✓ Saved: {thumb_file.name} ({thumb.shape[1]}x{thumb.shape[0]})")

        results.append({
            'preview': 'thumbnail',
            'status': 'success',
            'file': str(thumb_file),
            'size': [thumb.shape[1], thumb.shape[0]]
        })
    except Exception as e:
        print(f"  ✗ Thumbnail generation failed: {e}")
        results.append({'preview': 'thumbnail', 'status': 'failed', 'error': str(e)})

    # Medium preview
    print("\n[2/2] Generating medium preview (800px)...")
    try:
        medium = preview_gen.generate_preview(rgb, target_size=(800, 800))  # Changed from create_preview
        medium_file = output_dir / "preview_medium.png"
        # Use ImageExporter
        img_exporter.save_png(medium, medium_file, bit_depth=8)
        print(f"  ✓ Saved: {medium_file.name} ({medium.shape[1]}x{medium.shape[0]})")

        results.append({
            'preview': 'medium',
            'status': 'success',
            'file': str(medium_file),
            'size': [medium.shape[1], medium.shape[0]]
        })
    except Exception as e:
        print(f"  ✗ Medium preview generation failed: {e}")
        results.append({'preview': 'medium', 'status': 'failed', 'error': str(e)})

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4: Advanced enhancements and pickle workflow test"
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help="Input directory (same as Phase 1/2/3)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Re-run Phase 3 if needed, overwrite existing Phase 4 outputs"
    )

    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    force = args.force

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: Invalid directory: {input_dir}")
        sys.exit(1)

    print("="*80)
    print("PHASE 4: Advanced Enhancements & Pickle Workflow Test")
    print("="*80)
    print(f"\nInput directory: {input_dir}")
    print(f"Force re-run Phase 3: {force}")

    # Create output directory
    output_dir = input_dir / "output" / "phase4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for Phase 2 outputs (needed for pickle test)
    phase2_dir = input_dir / "output" / "phase2"
    if not phase2_dir.exists():
        print("\n⚠️  WARNING: Phase 2 output directory not found")
        print("   Pickle workflow test will be skipped")
        print("   (Phase 2 not required for enhancements)")

    # Check for existing Phase 4 outputs
    print("\nChecking for Phase 4 outputs...")
    has_phase4, message = check_phase4_outputs(output_dir)
    print(f"  {message}")

    if has_phase4 and not force:
        print("\n  Phase 4 outputs already exist. Use --force to regenerate.")
        print("  Exiting without processing.")
        return 0

    if force and has_phase4:
        print("  Force flag set - will overwrite existing outputs")

    # Check for Phase 3 outputs
    print("\nChecking for Phase 3 outputs...")
    has_phase3, message = check_phase3_outputs(input_dir)
    print(f"  {message}")

    if not has_phase3:
        if force:
            print("\n  Force flag set - re-running Phase 3")
            if not run_phase3(input_dir, force=True):
                print("\nERROR: Phase 3 failed")
                sys.exit(1)
            print("\n✓ Phase 3 complete")
        else:
            print("\nERROR: Phase 3 outputs not found")
            print("Please run Phase 3 first: python examples/phase3_demo_new.py <input_dir>")
            print("Or use --force to run it automatically")
            sys.exit(1)

    phase3_dir = input_dir / "output" / "phase3"
    print(f"Output directory: {output_dir}")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'tests': []
    }

    # Test 1: Pickle workflow (auto-skips if Phase 2 doesn't exist)
    if phase2_dir.exists():
        pickle_results = test_pickle_workflow(phase2_dir)
        if pickle_results:
            summary['pickle_test'] = pickle_results

            # Save pickle test results
            pickle_file = output_dir / "pickle_test_results.json"
            with open(pickle_file, 'w') as f:
                json.dump(pickle_results, f, indent=2)
            print(f"\n✓ Pickle test results saved: {pickle_file}")
    else:
        print("\n⚠️  Skipping pickle workflow test (Phase 2 outputs not found)")

    # Load Phase 3 RGB composite
    print("\n" + "="*80)
    print("Loading Phase 3 RGB composite...")
    print("="*80)

    # Try to find RGB composite (prefer lupton, fall back to simple)
    rgb_files = list(phase3_dir.glob("rgb_composite_*.png"))
    if not rgb_files:
        print("\nERROR: No RGB composite found in Phase 3 output")
        sys.exit(1)

    # Prefer lupton over simple
    rgb_file = None
    for f in rgb_files:
        if 'lupton' in f.name:
            rgb_file = f
            break
    if not rgb_file:
        rgb_file = rgb_files[0]

    print(f"\nLoading: {rgb_file.name}")

    # Load RGB image
    try:
        from PIL import Image
        rgb_pil = Image.open(rgb_file)
        rgb = np.array(rgb_pil).astype(np.float64) / 255.0
        print(f"✓ Loaded RGB: shape={rgb.shape}, dtype={rgb.dtype}")
        print(f"  Range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    except Exception as e:
        print(f"\nERROR: Failed to load RGB composite: {e}")
        sys.exit(1)

    # Apply enhancements
    enhancement_results = apply_enhancements(rgb, output_dir)
    summary['enhancements'] = enhancement_results

    # Generate previews
    preview_results = generate_previews(rgb, output_dir)
    summary['previews'] = preview_results

    # Save summary
    summary_file = output_dir / "phase4_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    print("\n" + "="*80)
    print("PHASE 4 COMPLETE")
    print("="*80)

    success_count = sum(
        1 for r in enhancement_results + preview_results
        if r.get('status') == 'success'
    )
    total_count = len(enhancement_results) + len(preview_results)

    print(f"\nProcessing complete: {success_count}/{total_count} operations successful")
    print(f"\nOutput files in: {output_dir}/")
    print("  - rgb_enhanced_clahe.png (CLAHE enhancement)")
    print("  - rgb_enhanced_unsharp.png (unsharp masking)")
    print("  - rgb_color_balanced.png (color balanced)")
    print("  - preview_thumbnail.png (256px thumbnail)")
    print("  - preview_medium.png (800px preview)")
    if phase2_dir.exists():
        print("  - pickle_test_results.json (pickle workflow validation)")
    print(f"  - phase4_summary.json (complete summary)")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
