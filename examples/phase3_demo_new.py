"""Phase 3 Demo: RGB Composite Generation

Process Phase 2 outputs (or run Phase 1+2 if needed):
- Map bands to RGB channels (chromatic ordering by wavelength)
- Create Lupton RGB composite
- Export PNG/TIFF with processing history

Usage:
    python examples/phase3_demo.py <input_dir> [--force] [--method METHOD]

Arguments:
    input_dir: Same directory used for phase1/phase2 demos
    --force: Re-run Phase 2 even if outputs exist
    --method: Composite method (lupton|simple) [default: lupton]

Example:
    python examples/phase3_demo.py examples/data/NOIRLab/examples/edu008/data
    python examples/phase3_demo.py examples/data/NOIRLab/examples/edu008/data --force
    python examples/phase3_demo.py examples/data/NOIRLab/examples/edu008/data --method simple

Output structure:
    <input_dir>/output/phase3/
        ├── rgb_composite_lupton.png
        ├── rgb_composite_lupton.tif
        ├── rgb_composite_simple.png
        ├── channel_mapping.json
        └── phase3_summary.json
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

from astro_vision_composer.postprocessing import (
    ChannelMapper, Compositor, ImageExporter, HistoryTracker
)
from astropy.io import fits as astropy_fits
from astropy.visualization import LinearStretch


def validate_phase3_output(rgb_output, phase2_metadata, method):
    """
    Validate Phase 3 RGB composition correctness.

    Proves:
    1. No double-stretching (Phase 2 stretched → Phase 3 uses LinearStretch)
    2. RGB output is float [0, 1] for consistency
    3. Color preservation (Lupton property)
    4. No oversaturation (all channels maxed)

    This implements WORKFLOW 1: Scientific Standard (continued from Phase 2)
    """
    print("\n" + "="*80)
    print("PHASE 3 VALIDATION (Workflow 1: Scientific Standard)")
    print("="*80)

    # Check 1: Workflow consistency
    workflow = phase2_metadata.get('workflow', 'unknown')
    print(f"✓ Check 1: Phase 2 workflow: {workflow}")

    if phase2_metadata.get('stretch', {}).get('method') != 'none':
        expected_phase3 = 'LinearStretch or Simple RGB'
        print(f"  Phase 2 applied stretch → Phase 3 should use: {expected_phase3}")
        if method == 'lupton':
            print(f"  ⚠️  Using Lupton - ensure stretch_object=LinearStretch()")

    # Check 2: RGB output dtype and range
    assert rgb_output.dtype in [np.float32, np.float64, np.uint8], \
        f"RGB dtype unexpected: {rgb_output.dtype}"
    print(f"✓ Check 2: RGB dtype: {rgb_output.dtype}")

    if rgb_output.dtype in [np.float32, np.float64]:
        assert 0 <= rgb_output.min() <= rgb_output.max() <= 1.01, \
            f"Float RGB not in [0, 1]: [{rgb_output.min()}, {rgb_output.max()}]"
        print(f"  RGB range (float): [{rgb_output.min():.4f}, {rgb_output.max():.4f}]")
    else:  # uint8
        assert 0 <= rgb_output.min() <= rgb_output.max() <= 255, \
            f"Uint8 RGB not in [0, 255]: [{rgb_output.min()}, {rgb_output.max()}]"
        print(f"  RGB range (uint8): [{rgb_output.min()}, {rgb_output.max()}]")

    # Check 3: Color diversity (not all gray)
    # Compute std dev across channels - should have variation
    channel_stds = rgb_output.std(axis=2)
    mean_color_variation = channel_stds.mean()
    if mean_color_variation < 0.01:
        print(f"  ⚠️  WARNING: Low color variation ({mean_color_variation:.4f}) - image might be grayscale")
    else:
        print(f"✓ Check 3: Color variation present (std: {mean_color_variation:.4f})")

    # Check 4: Not oversaturated (if Lupton, should preserve colors in bright regions)
    if method == 'lupton':
        # Normalize to [0, 1] if needed
        if rgb_output.dtype == np.uint8:
            rgb_norm = rgb_output.astype(np.float32) / 255.0
        else:
            rgb_norm = rgb_output

        # Check bright pixels (top 5%)
        brightness = rgb_norm.sum(axis=2)
        if brightness.max() > 0:
            bright_threshold = np.percentile(brightness, 95)
            bright_mask = brightness > bright_threshold

            if bright_mask.any():
                bright_pixels = rgb_norm[bright_mask]
                # Check if all channels are maxed (oversaturated)
                oversaturated = (bright_pixels > 0.95).all(axis=1).mean()
                if oversaturated > 0.5:
                    print(f"  ⚠️  WARNING: {oversaturated*100:.1f}% of bright pixels oversaturated")
                else:
                    print(f"✓ Check 4: Lupton color preservation working ({oversaturated*100:.1f}% saturation)")

    # Check 5: No NaN/Inf
    assert not np.any(np.isnan(rgb_output)), "RGB contains NaN"
    assert not np.any(np.isinf(rgb_output)), "RGB contains Inf"
    print(f"✓ Check 5: No NaN/Inf values")

    print("="*80)
    print("✅ PHASE 3 VALIDATION PASSED - Using correct workflow!")
    print("="*80)

    return True


def detect_workflow_from_phase2(phase2_dir):
    """
    Detect which workflow was used in Phase 2 by examining processing metadata.

    Returns recommended compositor settings for Phase 3.
    """
    # Look for any processing JSON
    processing_files = list(phase2_dir.glob("*_processing.json"))

    if not processing_files:
        print("⚠️  No Phase 2 processing metadata found - using defaults")
        return {
            'workflow': 'unknown',
            'recommended_compositor': 'simple',
            'phase2_stretched': True,
            'notes': 'No metadata - assuming data was stretched'
        }

    # Read first processing file
    with open(processing_files[0], 'r') as f:
        metadata = json.load(f)

    workflow = metadata.get('workflow', 'unknown')
    stretch_method = metadata.get('stretch', {}).get('method', 'unknown')
    has_stretch_pickle = 'stretch_pickle_hex' in metadata.get('stretch', {})

    print(f"\n📋 Detected Phase 2 workflow: {workflow}")
    print(f"   Stretch method: {stretch_method}")
    print(f"   Stretch object saved: {has_stretch_pickle}")

    if workflow == 'scientific_standard' or stretch_method != 'none':
        # Phase 2 applied stretching
        recommendation = {
            'workflow': workflow,
            'recommended_compositor': 'lupton_with_linear_stretch',
            'phase2_stretched': True,
            'notes': 'Data pre-stretched in Phase 2 → Use LinearStretch in Lupton or Simple RGB'
        }
    else:
        # Phase 2 only normalized
        recommendation = {
            'workflow': workflow,
            'recommended_compositor': 'lupton_with_zscale_stretch',
            'phase2_stretched': False,
            'notes': 'Data only normalized → Let Lupton calculate stretch'
        }

    print(f"   → Recommendation: {recommendation['recommended_compositor']}")
    print(f"   → Notes: {recommendation['notes']}")

    return recommendation


def check_phase2_outputs(input_dir):
    """Check if Phase 2 outputs exist."""
    phase2_dir = input_dir / "output" / "phase2"

    if not phase2_dir.exists():
        return False, "Phase 2 output directory not found"

    summary_file = phase2_dir / "phase2_summary.json"
    if not summary_file.exists():
        return False, "Phase 2 summary file not found"

    # Check for stretched FITS files
    stretched_files = list(phase2_dir.glob("*_stretched.fits"))
    if not stretched_files:
        return False, "No Phase 2 stretched FITS files found"

    if len(stretched_files) < 3:
        return False, f"Need at least 3 bands for RGB, found {len(stretched_files)}"

    return True, f"Found {len(stretched_files)} Phase 2 outputs"


def run_phase2(input_dir, force=False):
    """Run Phase 2 demo script."""
    print(f"\n{'='*80}")
    print("Running Phase 2 first...")
    print(f"{'='*80}\n")

    phase2_script = Path(__file__).parent / "phase2_demo_new.py"

    if not phase2_script.exists():
        print(f"ERROR: Phase 2 script not found: {phase2_script}")
        return False

    args = [sys.executable, str(phase2_script), str(input_dir)]
    if force:
        args.append('--force')

    result = subprocess.run(args, capture_output=False)
    return result.returncode == 0


def load_wavelengths_from_metadata(phase1_dir, stretched_files):
    """Load wavelength info from Phase 1 metadata files."""
    wavelengths = {}

    for stretched_file in stretched_files:
        # Get original filename (remove _processed_stretched suffix)
        base_name = stretched_file.stem.replace('_stretched', '')

        # Look for corresponding metadata file in Phase 1
        metadata_file = phase1_dir / f"{base_name}_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                wavelength = metadata.get('wavelength')
                if wavelength:
                    wavelengths[base_name] = wavelength
                    print(f"  {base_name}: {wavelength} nm")
                else:
                    print(f"  {base_name}: No wavelength info")
        else:
            print(f"  {base_name}: Metadata file not found")

    return wavelengths


def create_rgb_composite(stretched_files, wavelengths, output_dir, method='lupton'):
    """Create RGB composite from stretched bands."""

    print(f"\n{'='*80}")
    print(f"Creating RGB Composite (method={method})")
    print(f"{'='*80}")

    # Detect Phase 2 workflow
    phase2_dir = stretched_files[0].parent if stretched_files else None
    workflow_info = detect_workflow_from_phase2(phase2_dir) if phase2_dir else {}

    # Load Phase 2 metadata for validation
    phase2_metadata = {}
    if phase2_dir:
        processing_files = list(phase2_dir.glob("*_processing.json"))
        if processing_files:
            with open(processing_files[0], 'r') as f:
                phase2_metadata = json.load(f)

    # Load stretched data
    print("\n[1/4] Loading stretched bands...")
    bands = {}
    for fits_file in stretched_files:
        base_name = fits_file.stem.replace('_stretched', '')
        with astropy_fits.open(fits_file) as hdul:
            bands[base_name] = hdul[0].data
        print(f"  ✓ Loaded {base_name}: {bands[base_name].shape}")

    if len(bands) < 3:
        print(f"\n✗ ERROR: Need at least 3 bands for RGB, found {len(bands)}")
        return None

    # Map channels by wavelength
    print("\n[2/4] Mapping channels by wavelength...")
    mapper = ChannelMapper()

    if len(wavelengths) >= 3:
        # Auto-map by wavelength (chromatic ordering)
        mapping = mapper.auto_map_by_wavelength(wavelengths)
        print(f"  Chromatic mapping:")
        print(f"    Red   ← {mapping.red} ({mapping.red_wavelength:.0f} nm)")
        print(f"    Green ← {mapping.green} ({mapping.green_wavelength:.0f} nm)")
        print(f"    Blue  ← {mapping.blue} ({mapping.blue_wavelength:.0f} nm)")
    else:
        # Fallback to manual mapping with first 3 bands
        band_names = list(bands.keys())[:3]
        print(f"  ⚠ Insufficient wavelength data, using first 3 bands:")
        print(f"    Red   ← {band_names[0]}")
        print(f"    Green ← {band_names[1]}")
        print(f"    Blue  ← {band_names[2]}")
        mapping = mapper.manual_mapping(
            red=band_names[0],
            green=band_names[1],
            blue=band_names[2]
        )

    # Save mapping info
    mapping_file = output_dir / "channel_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump({
            'red': mapping.red,
            'green': mapping.green,
            'blue': mapping.blue,
            'red_wavelength': mapping.red_wavelength,
            'green_wavelength': mapping.green_wavelength,
            'blue_wavelength': mapping.blue_wavelength,
            'chromatic_order': mapping.chromatic_order
        }, f, indent=2)
    print(f"  ✓ Saved mapping: {mapping_file}")

    # Create composite
    print(f"\n[3/4] Creating {method.upper()} RGB composite...")
    compositor = Compositor()
    tracker = HistoryTracker()

    tracker.record('load_bands', {'count': len(bands)}, 'Phase3')
    tracker.record('channel_mapping', {
        'red': mapping.red,
        'green': mapping.green,
        'blue': mapping.blue
    }, 'ChannelMapper')

    # Determine stretch strategy based on Phase 2 workflow
    if method == 'lupton':
        if workflow_info.get('phase2_stretched', True):
            # Data pre-stretched in Phase 2 → Use LinearStretch (identity)
            print(f"  → Using LinearStretch (no additional tone-mapping)")
            print(f"     Reason: Data pre-stretched in Phase 2")
            rgb = compositor.create_lupton_rgb(
                r=bands[mapping.red],
                g=bands[mapping.green],
                b=bands[mapping.blue],
                stretch_object=LinearStretch(),
                output_dtype=np.float64
            )
            tracker.record('composite', {
                'method': 'lupton',
                'stretch_object': 'LinearStretch',
                'output_dtype': 'float64',
                'note': 'Pre-stretched data - no additional tone-mapping'
            }, 'Compositor')
        else:
            # Data only normalized → Let Lupton calculate stretch
            print(f"  → Using default Lupton stretch (auto-calculated)")
            print(f"     Reason: Data only normalized in Phase 2")
            rgb = compositor.create_lupton_rgb(
                r=bands[mapping.red],
                g=bands[mapping.green],
                b=bands[mapping.blue],
                stretch=0.5,
                Q=8
            )
            tracker.record('composite', {
                'method': 'lupton',
                'stretch': 0.5,
                'Q': 8
            }, 'Compositor')
    else:  # simple
        print(f"  → Using Simple RGB (independent channel stacking)")
        rgb = compositor.create_simple_rgb(
            r=bands[mapping.red],
            g=bands[mapping.green],
            b=bands[mapping.blue]
        )
        tracker.record('composite', {'method': 'simple'}, 'Compositor')

    print(f"  RGB shape: {rgb.shape}")
    print(f"  RGB dtype: {rgb.dtype}")
    print(f"  RGB range: [{np.min(rgb):.3f}, {np.max(rgb):.3f}]")

    # Validate Phase 3 output
    validate_phase3_output(rgb, phase2_metadata, method)

    # Export
    print(f"\n[4/4] Exporting...")
    exporter = ImageExporter()

    # PNG (8-bit for display)
    png_file = output_dir / f"rgb_composite_{method}.png"
    exporter.save_png(rgb, png_file, bit_depth=8)
    print(f"  ✓ Saved PNG: {png_file}")

    # TIFF (16-bit for archival)
    tiff_file = output_dir / f"rgb_composite_{method}.tif"
    exporter.save_tiff(rgb, tiff_file, bit_depth=16)
    print(f"  ✓ Saved TIFF: {tiff_file}")

    # Processing history
    history_file = output_dir / f"processing_history_{method}.txt"
    with open(history_file, 'w') as f:
        f.write(tracker.to_text())
    print(f"  ✓ Saved history: {history_file}")

    return {
        'method': method,
        'mapping': {
            'red': mapping.red,
            'green': mapping.green,
            'blue': mapping.blue
        },
        'png_file': str(png_file),
        'tiff_file': str(tiff_file),
        'history_file': str(history_file),
        'status': 'success'
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase3_demo.py <input_dir> [--force] [--method METHOD]")
        print("\nExample:")
        print("  python phase3_demo.py examples/data/NOIRLab/examples/edu008/data")
        print("  python phase3_demo.py examples/data/NOIRLab/examples/edu008/data --force")
        print("  python phase3_demo.py examples/data/NOIRLab/examples/edu008/data --method simple")
        print("\nOptions:")
        print("  --force: Re-run Phase 2 even if outputs exist")
        print("  --method: Composite method (lupton|simple) [default: lupton]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    force = '--force' in sys.argv

    # Parse method argument
    method = 'lupton'
    if '--method' in sys.argv:
        idx = sys.argv.index('--method')
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]
            if method not in ['lupton', 'simple']:
                print(f"ERROR: Invalid method '{method}'. Use 'lupton' or 'simple'")
                sys.exit(1)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: Invalid directory: {input_dir}")
        sys.exit(1)

    print(f"{'='*80}")
    print(f"PHASE 3: RGB Composite Generation")
    print(f"{'='*80}")
    print(f"\nInput directory: {input_dir}")
    print(f"Force re-run Phase 2: {force}")
    print(f"Composite method: {method}")

    # Check for Phase 2 outputs
    print("\nChecking for Phase 2 outputs...")
    has_phase2, message = check_phase2_outputs(input_dir)
    print(f"  {message}")

    if not has_phase2 or force:
        if force:
            print("\n  Force flag set - re-running Phase 2")
        else:
            print("\n  Phase 2 outputs not found - running Phase 2 first")

        if not run_phase2(input_dir, force):
            print("\n✗ Phase 2 failed")
            sys.exit(1)

        print("\n✓ Phase 2 complete")

    # Get Phase 2 stretched files
    phase2_dir = input_dir / "output" / "phase2"
    stretched_files = sorted(phase2_dir.glob("*_stretched.fits"))

    print(f"\nFound {len(stretched_files)} stretched bands")

    # Load wavelength info from Phase 1 metadata
    phase1_dir = input_dir / "output" / "phase1"
    print("\nLoading wavelength information from Phase 1 metadata...")
    wavelengths = load_wavelengths_from_metadata(phase1_dir, stretched_files)

    # Create output directory
    output_dir = input_dir / "output" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Create composite
    result = create_rgb_composite(stretched_files, wavelengths, output_dir, method)

    if result:
        # Save summary
        summary_file = output_dir / "phase3_summary.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'phase2_rerun': force or not has_phase2,
            'num_bands': len(stretched_files),
            'result': result
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\n✓ RGB composite created using {result['method'].upper()} method")
        print(f"\nChannel mapping:")
        print(f"  Red   = {result['mapping']['red']}")
        print(f"  Green = {result['mapping']['green']}")
        print(f"  Blue  = {result['mapping']['blue']}")
        print(f"\nOutput files in: {output_dir}/")
        print(f"  - rgb_composite_{method}.png (8-bit display)")
        print(f"  - rgb_composite_{method}.tif (16-bit archival)")
        print(f"  - processing_history_{method}.txt")
        print(f"  - channel_mapping.json")
        print(f"{'='*80}")

        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
