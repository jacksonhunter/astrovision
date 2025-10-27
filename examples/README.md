# Astro Vision Composer - Examples

This directory contains comprehensive examples demonstrating all features of the Astro Vision Composer pipeline.

**Updated:** 2025-10-26 (Phase 2 Complete - Production Grade)

---

## Quick Start

**Most users should start here:**

1. **`pipeline_demo.py`** - Basic 3-band RGB composite workflow
   - Simplest example showing end-to-end processing
   - Pre-calibrated FITS → RGB image in ~10 lines

2. **`calibration_manager_example.py`** - Raw CCD data calibration
   - Complete bias/dark/flat workflow
   - Essential for ground-based observatory data
   - Shows all core calibration features

3. **`calibration_manager_enhanced_example.py`** ⭐ **NEW!**
   - **9 comprehensive scenarios** showing all production features
   - Overscan/trim support
   - Cached calibration loading
   - Memory-efficient mode
   - Flexible keyword matching
   - **Recommended for production workflows**

---

## Examples by Category

### 🚀 Getting Started (Pre-Calibrated Data)

These examples work with pre-processed FITS files (JWST `_cal.fits`, HST `_flc.fits`, ground-based calibrated frames).

#### **`pipeline_demo.py`**
**What it demonstrates:**
- Basic 3-band RGB workflow
- Automatic normalization and stretching
- Default "scientific" mode

**Code:**
```python
from astro_vision_composer import ProcessingPipeline

pipeline = ProcessingPipeline(mode='scientific')
rgb = pipeline.process_to_rgb(
    fits_files=['r_band.fits', 'g_band.fits', 'b_band.fits'],
    output_path='output.png'
)
```

**Best for:** Quick start, testing, pre-calibrated space telescope data

---

#### **`phase1_demo_new.py`** / **`phase1_demo.py`**
**What it demonstrates:**
- Manual workflow mode (Phase 1 feature)
- Per-band normalization control
- Custom interval/stretch combinations

**Code:**
```python
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch

pipeline = ProcessingPipeline(mode='manual')

normalizations = [
    ImageNormalize(interval=ZScaleInterval(), stretch=AsinhStretch(a=0.1)),
    ImageNormalize(interval=PercentileInterval(99), stretch=LogStretch()),
    ImageNormalize(interval=MinMaxInterval(), stretch=SqrtStretch())
]

rgb = pipeline.process_with_normalizations(
    fits_files=['ha.fits', 'oiii.fits', 'sii.fits'],
    normalizations=normalizations
)
```

**Best for:** Narrowband imaging, advanced users, iterative refinement

---

### 🔬 Raw CCD Data (Ground-Based Observatories)

These examples show how to process completely raw CCD data from bias/dark/flat through to final RGB.

#### **`calibration_manager_example.py`**
**What it demonstrates:**
- Core calibration workflow (Phase 2 initial implementation)
- Auto-detection of bias, dark, flat frames
- Combining with sigma clipping
- Applying calibrations to science frames

**Example scenarios:**
1. Basic auto-detection
2. Manual calibration creation
3. Filter-specific flats
4. Batch processing multiple science frames
5. Integration with ProcessingPipeline
6. Saving/loading calibration frames
7. Complete production workflow

**Best for:** Learning calibration basics, simple CCD setups

---

#### **`calibration_manager_enhanced_example.py`** ⭐ **RECOMMENDED**
**What it demonstrates:**
- **All Phase 2 refinement features** (production-grade)
- Overscan subtraction and trimming
- Cached calibration loading (speed!)
- Unit detection from FITS headers
- Flexible IMAGETYP matching (OBSTYPE, FRAMETYPE, etc.)
- Flexible filter name matching ('V' matches 'V-band', 'V_filter')
- Memory-efficient mode for large datasets
- Enhanced metadata and validation

**9 Example Scenarios:**

1. **Basic Usage** - Auto-detection with optional caching
   ```python
   calib_mgr = CalibrationManager('raw_data/calibration/', master_cache_dir='masters/')
   if calib_mgr.load_cached_calibrations():
       print("Using cached calibrations")
   else:
       calib_mgr.create_master_calibrations()
   ```

2. **Overscan/Trim** - CCD with overscan regions
   ```python
   calib_mgr = CalibrationManager(
       'raw_data/ccd_with_overscan/',
       overscan_region='[:, 2049:2080]',  # Last 32 columns
       trim_region='[:, 1:2048]'           # Imaging area only
   )
   ```

3. **Unit Detection** - Automatic BUNIT handling
   - Handles ADU, electrons, counts, DN, ADU/s automatically
   - No more unit mismatch errors!

4. **Flexible IMAGETYP** - Works with different observatories
   - Checks: IMAGETYP, OBSTYPE, FRAMETYPE, FRAME, IMAGECAT
   - Case-insensitive: 'bias', 'BIAS', 'Bias' all work

5. **Filter Matching** - Handles naming variations
   - 'V' matches 'V-band', 'V_filter', 'Johnson V'
   - 'Ha' matches 'H-alpha', 'Halpha', 'H-a'

6. **Memory-Efficient** - Large CCD datasets
   ```python
   calib_mgr = CalibrationManager(
       'raw_data/large_ccd/',
       mem_limit=8e9  # 8 GB limit
   )
   # Process 100+ frames without exhausting RAM
   ```

7. **Validation & Metadata** - Quality checks
   - Automatic NaN detection
   - Zero variance warnings
   - Enhanced metadata (EXPTIME, FRAMTYPE, etc.)

8. **Cached Calibrations** - Efficient workflow
   - Session 1: Create and cache
   - Session 2: Load from cache (instant!)

9. **Complete Production Workflow** - All features together
   - Overscan/trim + caching + memory-efficient + validation
   - Ready for real observatory use

**Best for:** Production deployments, diverse observatories, large datasets

---

### 🎨 Advanced Processing (Phase 2-4 Demos)

#### **`phase2_demo_new.py`** / **`phase2_demo.py`**
**What it demonstrates:**
- Advanced normalization strategies
- Multiple compositor modes (simple, Lupton)
- Custom stretching combinations

**Best for:** Comparing different processing approaches

---

#### **`phase3_demo_new.py`** / **`phase3_demo.py`**
**What it demonstrates:**
- WCS handling and reprojection
- Multi-mission data alignment
- Footprint handling

**Best for:** Combining data from different instruments/telescopes

---

#### **`phase4_demo_new.py`**
**What it demonstrates:**
- Advanced color compositing
- Narrowband palette selection (future)
- HDR tone mapping (future)

**Best for:** Publication-quality images, narrowband imaging

---

## Example Data

### **`examples/data/`**
Contains sample FITS files for testing examples. If missing, examples will attempt to download NOIRLab data.

**Structure:**
```
examples/data/
├── raw_ccd/               # Raw CCD frames with calibrations
│   ├── calibration/
│   │   ├── bias_*.fits
│   │   ├── dark_*.fits
│   │   └── flat_*.fits
│   └── science_*.fits
├── calibrated/            # Pre-calibrated frames
│   ├── r_band.fits
│   ├── g_band.fits
│   └── b_band.fits
└── narrowband/            # Narrowband imaging
    ├── ha.fits
    ├── oiii.fits
    └── sii.fits
```

---

## Running Examples

### Prerequisites

```bash
# Core dependencies
pip install numpy astropy scipy

# For calibration examples
pip install ccdproc

# For WCS/reprojection examples (Phase 3+)
pip install reproject

# Optional: For JWST data
pip install stdatamodels

# Optional: For HST data
pip install drizzlepac
```

### Running an Example

```bash
# Navigate to examples directory
cd examples/

# Run any example
python pipeline_demo.py
python calibration_manager_enhanced_example.py
```

**Note:** Examples may require downloading sample data on first run.

---

## Example Workflows by Use Case

### **Use Case 1: Space Telescope Data (Pre-Calibrated)**
**Files you have:** JWST `_cal.fits`, HST `_flc.fits`, or similar pre-processed FITS

**Examples to use:**
1. Start with: `pipeline_demo.py`
2. Customize with: `phase1_demo_new.py` (manual workflow)
3. Advanced processing: `phase2_demo_new.py`, `phase3_demo_new.py`

**Typical workflow:**
```python
# Simple case
pipeline = ProcessingPipeline(mode='sdss')
rgb = pipeline.process_to_rgb(['f090w_cal.fits', 'f200w_cal.fits', 'f444w_cal.fits'])

# Advanced case (manual control)
pipeline = ProcessingPipeline(mode='manual')
normalizations = [...]  # Custom per-band
rgb = pipeline.process_with_normalizations(files, normalizations)
```

---

### **Use Case 2: Ground-Based Raw CCD Data**
**Files you have:** Raw FITS from CCD camera with separate bias/dark/flat frames

**Examples to use:**
1. Start with: `calibration_manager_example.py` (learn basics)
2. Production use: `calibration_manager_enhanced_example.py` (all features)
3. Integration: Example 9 in enhanced example (complete workflow)

**Typical workflow:**
```python
# Setup calibration manager
calib_mgr = CalibrationManager(
    calibration_dir='raw_data/calibration/',
    master_cache_dir='masters/',
    overscan_region='[:, 2049:2080]' if needed,
    trim_region='[:, 1:2048]' if needed
)

# Try cache, create if needed
if not calib_mgr.load_cached_calibrations():
    calib_mgr.create_master_calibrations()

# Calibrate science frames
for filter_name in ['V', 'R', 'I']:
    science = CCDData.read(f'science_{filter_name}.fits', unit='adu')
    calibrated = calib_mgr.calibrate(science, filter_name=filter_name)
    calibrated.write(f'calibrated_{filter_name}.fits')

# Create RGB
pipeline = ProcessingPipeline(mode='scientific')
rgb = pipeline.process_to_rgb([
    'calibrated_R.fits',
    'calibrated_V.fits',  # Maps to green
    'calibrated_B.fits'   # Would need B filter or use V
])
```

---

### **Use Case 3: Narrowband Imaging (Ha/OIII/SII)**
**Files you have:** Narrowband filter images from ground-based telescope

**Examples to use:**
1. Calibration: `calibration_manager_enhanced_example.py` (Example 5 for filter matching)
2. Manual processing: `phase1_demo_new.py` (per-band control essential!)
3. Advanced: `phase4_demo_new.py` (palette selection - future feature)

**Typical workflow:**
```python
# Calibrate narrowband frames
calib_mgr = CalibrationManager('raw_data/calibration/')
calib_mgr.create_master_calibrations()

for filter_name in ['Ha', 'OIII', 'SII']:
    science = CCDData.read(f'science_{filter_name}.fits', unit='adu')
    calibrated = calib_mgr.calibrate(science, filter_name=filter_name)
    calibrated.write(f'calibrated_{filter_name}.fits')

# Process with manual control (each band needs different treatment!)
pipeline = ProcessingPipeline(mode='manual')
normalizations = [
    ImageNormalize(interval=ZScaleInterval(), stretch=AsinhStretch(a=0.05)),  # Ha
    ImageNormalize(interval=PercentileInterval(99), stretch=AsinhStretch(a=0.1)),  # OIII
    ImageNormalize(interval=PercentileInterval(98), stretch=AsinhStretch(a=0.15))  # SII
]

# Hubble palette: SII→R, Ha→G, OIII→B
rgb = pipeline.process_with_normalizations(
    ['calibrated_SII.fits', 'calibrated_Ha.fits', 'calibrated_OIII.fits'],
    normalizations=normalizations
)
```

---

### **Use Case 4: Large Dataset / Memory-Constrained**
**Challenge:** 100+ bias frames, 4K×4K CCD, limited RAM

**Example to use:** `calibration_manager_enhanced_example.py` (Example 6)

**Typical workflow:**
```python
calib_mgr = CalibrationManager(
    'raw_data/large_dataset/',
    master_cache_dir='masters/',
    mem_limit=8e9  # 8 GB memory limit
)

# ccdproc will automatically chunk processing
calib_mgr.create_master_calibrations()
# Process stays under 8 GB even with 100+ frames!
```

---

## Troubleshooting Examples

### Example fails with "No bias frames found"
**Cause:** IMAGETYP keyword not matching

**Solution:** CalibrationManager now checks multiple keywords automatically (IMAGETYP, OBSTYPE, FRAMETYPE). If still failing, check your FITS headers:
```python
from astropy.io import fits
with fits.open('bias_001.fits') as hdul:
    print(hdul[0].header)  # Look for frame type keyword
```

### Example fails with "No flat frames found for filter V"
**Cause:** Filter name mismatch (header says 'V-band', you specified 'V')

**Solution:** Use enhanced CalibrationManager - it does flexible matching automatically:
```python
# These all match now:
calib_mgr.create_master_flat('V')  # Matches 'V', 'V-band', 'V_filter', 'Johnson V'
calib_mgr.create_master_flat('Ha')  # Matches 'Ha', 'H-alpha', 'Halpha'
```

### Memory errors during combination
**Solution:** Use mem_limit parameter:
```python
calib_mgr = CalibrationManager(
    'raw_data/',
    mem_limit=4e9  # 4 GB limit
)
```

### Unit mismatch errors (ADU vs electrons)
**Solution:** Enhanced CalibrationManager detects units automatically from BUNIT:
```python
# No manual unit specification needed anymore!
calib_mgr = CalibrationManager('raw_data/')  # Auto-detects from headers
```

---

## Feature Roadmap

### ✅ Currently Available (Phase 0-2 Complete)
- Basic RGB workflow
- Manual per-band control
- Raw CCD calibration (bias/dark/flat)
- Overscan/trim support
- Cached calibrations
- Memory-efficient mode
- Flexible keyword/filter matching
- Validation and quality checks

### ⏳ Coming Soon (Phase 3)
- Multi-mission WCS support (JWST gwcs, HST drizzlepac)
- Advanced reprojection
- Automatic reference frame selection

### ⏳ Future (Phase 4)
- Narrowband palette presets (Hubble, HOO, natural)
- Custom palette definition
- Advanced color balancing

---

## Contributing Examples

If you've created a useful workflow, consider contributing it as an example!

**Requirements:**
1. Well-commented code
2. Docstring explaining use case
3. Sample data or instructions to download
4. Add entry to this README

**Submit via:** GitHub pull request or issue

---

## Getting Help

**Example not working?**
1. Check prerequisites are installed
2. Review troubleshooting section above
3. Check CLAUDE.md for known issues
4. Open GitHub issue with:
   - Example name
   - Error message
   - Python/package versions
   - Sample FITS header (if relevant)

**Need a new example?**
Open a GitHub issue describing your use case!

---

## Version History

**2025-10-26 - Phase 2 Complete (Production Grade)**
- Added `calibration_manager_enhanced_example.py` with 9 scenarios
- Updated all calibration examples with new features
- Added troubleshooting section
- Comprehensive use case workflows

**2025-10-26 - Phase 1 Complete**
- Added manual workflow examples
- Phase demos updated with ImageNormalize
- Fixed example data paths

**Earlier:**
- Initial example suite
- Basic pipeline demos

---

**Last Updated:** 2025-10-26
**Examples Count:** 12 files, 9 comprehensive scenarios in enhanced example
**Production Ready:** ✅ Calibration workflows tested with real observatory data
