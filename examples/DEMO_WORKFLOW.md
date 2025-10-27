# Demo Workflow: Phase 1-3 Pipeline

This demonstrates the complete FITS processing pipeline from raw data to RGB composites using **real astronomical data**.

## Quick Start

```bash
# Use NOIRLab Eagle Nebula dataset (edu008)
INPUT_DIR="examples/data/NOIRLab/examples/edu008/data"

# Run complete pipeline (Phase 1 → 2 → 3)
python examples/phase1_demo_new.py $INPUT_DIR
python examples/phase2_demo_new.py $INPUT_DIR
python examples/phase3_demo_new.py $INPUT_DIR

# View results
ls $INPUT_DIR/output/phase3/rgb_composite_lupton.png
```

## Pipeline Overview

```
Phase 1: Preprocessing
  Input:  *.fits (raw FITS files)
  Output: *_processed.fits + metadata + quality reports

Phase 2: Normalization & Stretching
  Input:  Phase 1 outputs (auto-runs Phase 1 if missing)
  Output: *_normalized.fits + *_stretched.fits

Phase 3: RGB Composition
  Input:  Phase 2 outputs (auto-runs Phase 2 if missing)
  Output: rgb_composite.png + .tif + processing history
```

## Detailed Usage

### Phase 1: Preprocessing

**Purpose:** Load FITS files, extract metadata, assess quality

```bash
python examples/phase1_demo_new.py <input_dir>
```

**What it does:**
- Finds all `.fits` files in `<input_dir>` (recursive search)
- Extracts mission-aware metadata (mission, filter, wavelength, etc.)
- Loads science data with automatic extension detection
- Runs quality assessment (SNR, saturation, dynamic range)
- Saves processed FITS files for Phase 2

**Output structure:**
```
<input_dir>/output/phase1/
├── 502nmos_metadata.json          # Metadata for each file
├── 502nmos_quality.json           # Quality report
├── 502nmos_processed.fits         # Processed data
├── 656nmos_metadata.json
├── 656nmos_quality.json
├── 656nmos_processed.fits
├── 673nmos_metadata.json
├── 673nmos_quality.json
├── 673nmos_processed.fits
└── phase1_summary.json            # Overall summary
```

**Example output:**
```
Processing: 502nmos.fits
  Mission: HST
  Filter: F502N
  Wavelength: 502 nm
  Quality Score: 8.5/10
  SNR: 45.2
  Saturation: 0.3%
```

---

### Phase 2: Normalization & Stretching

**Purpose:** Normalize and apply non-linear stretch for visualization

```bash
python examples/phase2_demo_new.py <input_dir> [--force]
```

**Arguments:**
- `<input_dir>`: Same directory used in Phase 1
- `--force`: Re-run Phase 1 even if outputs exist

**Dependency handling:**
- Checks if `<input_dir>/output/phase1/` exists
- If missing or `--force`: runs Phase 1 automatically
- If exists: uses Phase 1 outputs directly

**What it does:**
- Loads Phase 1 processed FITS files
- Applies ZScale normalization (astronomical standard)
- Applies asinh stretch (good for wide dynamic range)
- Saves normalized and stretched FITS files

**Output structure:**
```
<input_dir>/output/phase2/
├── 502nmos_normalized.fits        # ZScale normalized (0-1 range)
├── 502nmos_stretched.fits         # Asinh stretched
├── 502nmos_processing.json        # Processing parameters
├── 656nmos_normalized.fits
├── 656nmos_stretched.fits
├── 656nmos_processing.json
├── 673nmos_normalized.fits
├── 673nmos_stretched.fits
├── 673nmos_processing.json
└── phase2_summary.json
```

**Example output:**
```
[2/3] Normalizing with ZScale...
  Interval: [1.23e+02, 4.56e+03]
  Normalized range: [0.000, 1.000]

[3/3] Applying asinh stretch...
  Stretched range: [0.000, 0.987]
```

---

### Phase 3: RGB Composite Generation

**Purpose:** Create RGB composites from multiple bands

```bash
python examples/phase3_demo_new.py <input_dir> [--force] [--method METHOD]
```

**Arguments:**
- `<input_dir>`: Same directory used in Phase 1/2
- `--force`: Re-run Phase 2 even if outputs exist
- `--method`: Composite method (`lupton` or `simple`) [default: `lupton`]

**Dependency handling:**
- Checks if `<input_dir>/output/phase2/` exists with ≥3 bands
- If missing or `--force`: runs Phase 2 automatically (which may run Phase 1)
- If exists: uses Phase 2 outputs directly

**What it does:**
- Loads stretched FITS files from Phase 2
- Loads wavelength info from Phase 1 metadata
- Auto-maps bands to RGB channels (chromatic ordering: blue←short λ, red←long λ)
- Creates RGB composite using Lupton or simple algorithm
- Exports PNG (8-bit), TIFF (16-bit), and processing history

**Output structure:**
```
<input_dir>/output/phase3/
├── rgb_composite_lupton.png       # 8-bit PNG for display
├── rgb_composite_lupton.tif       # 16-bit TIFF for archival
├── rgb_composite_simple.png       # If --method simple used
├── processing_history_lupton.txt  # Full processing history
├── channel_mapping.json           # RGB channel assignments
└── phase3_summary.json
```

**Example output:**
```
[2/4] Mapping channels by wavelength...
  Chromatic mapping:
    Red   ← sii (673 nm)
    Green ← ha (656 nm)
    Blue  ← oiii (502 nm)

[3/4] Creating LUPTON RGB composite...
  RGB shape: (1024, 1024, 3)
  RGB range: [0.000, 0.995]
```

---

## Advanced Usage

### Force Re-processing

Re-run entire pipeline from scratch:
```bash
python examples/phase3_demo_new.py $INPUT_DIR --force
```

This will:
1. Delete existing Phase 3 outputs
2. Re-run Phase 2 (which re-runs Phase 1)
3. Create fresh RGB composite

### Different Composite Methods

**Lupton RGB** (default): SDSS algorithm, preserves color in bright regions
```bash
python examples/phase3_demo_new.py $INPUT_DIR --method lupton
```

**Simple RGB**: Independent channel scaling
```bash
python examples/phase3_demo_new.py $INPUT_DIR --method simple
```

### Incremental Processing

Process only new files:
1. Add new FITS files to `<input_dir>`
2. Run Phase 1 (only new files processed)
3. Run Phase 2 (picks up new Phase 1 outputs)
4. Run Phase 3 (creates new composite with all bands)

---

## Example Datasets

### NOIRLab edu008 - Eagle Nebula
```bash
INPUT_DIR="examples/data/NOIRLab/examples/edu008/data"
python examples/phase3_demo_new.py $INPUT_DIR
```

**3 HST WFPC2 narrowband filters:**
- 502nm [OIII] (oxygen emission)
- 656nm H-alpha (hydrogen emission)
- 673nm [SII] (sulfur emission)

**Expected output:**
- Beautiful narrowband composite (Hubble Palette style)
- Blue = oxygen, Green = hydrogen, Red = sulfur

### NOIRLab edu010 - M17 Nebula
```bash
INPUT_DIR="examples/data/NOIRLab/examples/edu010/data/edu010"
python examples/phase3_demo_new.py $INPUT_DIR
```

**3 HST narrowband filters (same as edu008)**

---

## Output Directory Structure

Complete output after running all 3 phases:

```
examples/data/NOIRLab/examples/edu008/data/output/
├── phase1/
│   ├── 502nmos_metadata.json
│   ├── 502nmos_quality.json
│   ├── 502nmos_processed.fits
│   ├── ... (656nmos, 673nmos)
│   └── phase1_summary.json
├── phase2/
│   ├── 502nmos_normalized.fits
│   ├── 502nmos_stretched.fits
│   ├── 502nmos_processing.json
│   ├── ... (656nmos, 673nmos)
│   └── phase2_summary.json
└── phase3/
    ├── rgb_composite_lupton.png      ← FINAL OUTPUT
    ├── rgb_composite_lupton.tif
    ├── processing_history_lupton.txt
    ├── channel_mapping.json
    └── phase3_summary.json
```

---

## Classes Demonstrated

**Phase 1:**
- `FITSLoader` - Intelligent FITS loading with automatic extension detection
- `FITSMetadata` - Mission-aware metadata extraction
- `MissionAdapter` - Mission-specific FITS conventions (HST, JWST, PanSTARRS, etc.)
- `QualityAssessor` - Statistical quality analysis (no AI!)

**Phase 2:**
- `Normalizer` - Data normalization (ZScale, Percentile, MinMax)
- `Stretcher` - Non-linear visualization stretches (asinh, log, sqrt, power)

**Phase 3:**
- `ChannelMapper` - RGB channel assignment by wavelength (chromatic ordering)
- `Compositor` - RGB composition algorithms (Lupton, simple)
- `ImageExporter` - Multi-format export (PNG, TIFF, JPEG) with metadata
- `HistoryTracker` - Processing history tracking for reproducibility

---

## Troubleshooting

### "No FITS files found"
- Check that FITS files exist in `<input_dir>`
- Script searches recursively for `*.fits`, `*.fit`, `*.fts`

### "Need at least 3 bands for RGB"
- RGB composites require ≥3 bands
- Single-band datasets can use Phase 1+2 only

### "Phase X output directory not found"
- Let the script auto-run previous phases
- Or use `--force` to re-run from scratch

### "Module not found" errors
- Ensure you're in the VisionProject root directory
- Required packages: `astropy`, `numpy`, `scipy`, `matplotlib`, `Pillow`

---

## Next Steps

After Phase 3, you can:
1. **Enhance** with Phase 4 (CLAHE, unsharp masking, color balancing)
2. **Compare** Lupton vs. Simple methods
3. **Process** other NOIRLab datasets (edu010, edu011, etc.)
4. **Customize** processing parameters in the scripts
5. **Build** ProcessingPipeline to automate this workflow

---

## Comparison: Old vs New Demos

| Old Demos | New Demos |
|-----------|-----------|
| Single FITS file | Batch processing (all files in directory) |
| Synthetic test data | Real astronomical data (NOIRLab) |
| Manual execution | Auto-dependency checking |
| No incremental processing | Reuses previous phase outputs |
| No output organization | Organized output structure |

**Migration:** Replace old demos with new versions once tested.
