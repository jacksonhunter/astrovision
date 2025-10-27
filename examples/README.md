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

## Selecting the Best FITS Observations

**Critical for Success:** The quality of your final RGB image depends heavily on selecting appropriate input FITS files. This section provides comprehensive guidance on evaluating and choosing observations.

### Understanding FITS Calibration Levels

Different missions and processing pipelines produce FITS files at various calibration stages. Understanding these levels is essential for selecting the right data.

#### **JWST (James Webb Space Telescope)**

**Calibration Pipeline Stages:**

| Stage | Suffix | Description | Use for RGB? | WCS Type |
|-------|--------|-------------|--------------|----------|
| **Uncalibrated** | `_uncal.fits` | Raw detector readouts | [ERROR] NO - needs calibration | None |
| **Rate** | `_rate.fits` | Count rate images (per second) | [ERROR] NO - distorted coordinates | Detector coords |
| **Calibrated** | `_cal.fits` | Fully calibrated, sky-subtracted | [OK] YES - **RECOMMENDED** | **gwcs** (ASDF) |
| **CR-flagged** | `_crf.fits` | Cosmic ray flagged | [OK] YES - if available | gwcs (ASDF) |
| **Resampled** | `_i2d.fits` | Drizzled to common grid | [OK] YES - for mosaics | FITS WCS (TAN) |
| **Source catalog** | `_cat.ecsv` | Extracted sources | [ERROR] NO - not an image | N/A |

**Best Choice for RGB:**
1. **`_cal.fits`** - Full calibration, native resolution, **includes gwcs** (Phase 3A feature!)
2. **`_crf.fits`** - If cosmic ray rejection is critical
3. **`_i2d.fits`** - When combining with other missions (common pixel grid)

**WCS Considerations:**
- `_cal.fits` and `_crf.fits` contain **gwcs in ASDF extension** - our Phase 3A/3B WCS handler fully supports this!
- `_i2d.fits` has standard FITS WCS (TAN projection) - simpler but loses gwcs advanced features
- For maximum WCS fidelity (spectroscopy, IFU data, advanced distortion), use `_cal.fits`

**Filter Selection Examples:**
```python
# Infrared RGB (most common for JWST)
files = [
    'jw02739_nircam_f444w_cal.fits',  # 4.4 μm -> Red
    'jw02739_nircam_f200w_cal.fits',  # 2.0 μm -> Green
    'jw02739_nircam_f090w_cal.fits'   # 0.9 μm -> Blue
]

# Optimal wavelength spread: 0.9 - 4.4 μm (5× range)
```

---

#### **HST (Hubble Space Telescope)**

**Calibration Pipeline Stages:**

| Stage | Suffix | Description | Use for RGB? | WCS Type |
|-------|--------|-------------|--------------|----------|
| **Raw** | `_raw.fits` | Uncalibrated detector data | [ERROR] NO - needs calibration | Detector coords |
| **Flat-fielded** | `_flt.fits` | Calibrated, cosmic rays NOT removed | Maybe - check for CRs | FITS WCS + distortion |
| **CR-rejected** | `_flc.fits` | Calibrated + CR rejection (ACS/WFC3) | [OK] YES - **RECOMMENDED** | FITS WCS + distortion |
| **Drizzled** | `_drz.fits` | Multiple exposures combined | [OK] YES - for mosaics | FITS WCS (TAN) |
| **Multi-drizzle** | `_drc.fits` | Advanced drizzling with CR rejection | [OK] YES - best for mosaics | FITS WCS (TAN) |

**Best Choice for RGB:**
1. **`_flc.fits`** (ACS, WFC3) - Full calibration + CR rejection
2. **`_flt.fits`** (older instruments) - If `_flc.fits` unavailable
3. **`_drz.fits` / `_drc.fits`** - For multi-exposure mosaics

**WCS Considerations:**
- `_flt.fits` / `_flc.fits` include **full distortion correction** (SIP coefficients, D2IM lookup tables)
- Our Phase 3A WCS handler supports **drizzlepac.stwcs** for HST distortion (with graceful fallback to basic FITS WCS)
- `_drz.fits` products are already distortion-corrected and resampled (loses native pixel scale)

**Filter Selection Examples:**
```python
# Optical RGB (classic HST colors)
files = [
    'hst_13003_acs_wfc_f814w_flc.fits',  # I-band (814nm) -> Red
    'hst_13003_acs_wfc_f606w_flc.fits',  # V-band (606nm) -> Green
    'hst_13003_acs_wfc_f435w_flc.fits'   # B-band (435nm) -> Blue
]

# Wavelength spread: 435 - 814 nm (good visual coverage)

# Narrowband (requires Phase 4 palette mapping!)
files = [
    'hst_ngc6302_wfc3_f658n_flc.fits',   # Ha + [NII] (658nm)
    'hst_ngc6302_wfc3_f502n_flc.fits',   # [OIII] (502nm)
    'hst_ngc6302_wfc3_f673n_flc.fits'    # [SII] (673nm)
]
# Use Hubble palette: SII->Red, Ha->Green, OIII->Blue
```

---

#### **Chandra (X-ray Observatory)**

**Data Types:**

| Type | Suffix | Description | Use for RGB? | Notes |
|------|--------|-------------|--------------|-------|
| **Event list** | `_evt2.fits` | Photon events (TIME, X, Y, ENERGY) | [ERROR] NO - not an image yet | Needs binning |
| **Binned image** | `_img.fits` | Events binned to 2D image | [OK] YES | Energy-filtered |
| **Exposure map** | `_exp.fits` | Effective exposure per pixel | [ERROR] NO - for normalization | Use with `_img.fits` |

**Best Choice for RGB:**
- Bin event lists into energy bands: soft (0.5-1.5 keV), medium (1.5-3 keV), hard (3-7 keV)
- Create separate `_img.fits` for each energy band
- Our pipeline currently supports binned images, NOT raw event lists (Phase 5 feature)

**Creating RGB from Chandra Events:**
```bash
# Use CIAO tools to bin events into energy bands (not handled by our pipeline yet)
dmcopy "evt2.fits[energy=500:1500][bin x=::1,y=::1]" soft_img.fits
dmcopy "evt2.fits[energy=1500:3000][bin x=::1,y=::1]" medium_img.fits
dmcopy "evt2.fits[energy=3000:7000][bin x=::1,y=::1]" hard_img.fits

# Then process with our pipeline
python
from astro_vision_composer import ProcessingPipeline
pipeline = ProcessingPipeline(mode='scientific')
rgb = pipeline.process_to_rgb(['hard_img.fits', 'medium_img.fits', 'soft_img.fits'])
```

---

#### **Ground-Based Observatories (Raw CCD Data)**

**Typical File Structure:**

| Type | Common Names | IMAGETYP | Use for RGB? | Required? |
|------|--------------|----------|--------------|-----------|
| **Bias** | `bias_*.fits` | BIAS, Zero | [ERROR] NO - calibration frame | YES |
| **Dark** | `dark_*.fits` | DARK | [ERROR] NO - calibration frame | YES |
| **Flat** | `flat_*.fits`, `skyflat_*.fits` | FLAT, Sky Flat | [ERROR] NO - calibration frame | YES |
| **Science** | `science_*.fits`, `light_*.fits` | LIGHT, Object, Science | After calibration | YES |

**Processing Workflow (Phase 2 Feature):**

1. **Check Calibration Frame Availability:**
   ```python
   from astropy.io import fits
   from glob import glob

   # Verify you have all required calibration types
   bias_files = glob('calibration/bias_*.fits')
   dark_files = glob('calibration/dark_*.fits')
   flat_files = glob('calibration/flat_*.fits')

   print(f"Bias frames: {len(bias_files)}")  # Need 10+ for good sigma clipping
   print(f"Dark frames: {len(dark_files)}")  # Need 5+ per exposure time
   print(f"Flat frames: {len(flat_files)}")  # Need 5+ per filter

   # Check exposure times match
   for dark in dark_files[:3]:
       hdr = fits.getheader(dark)
       print(f"{dark}: EXPTIME={hdr.get('EXPTIME', 'MISSING')}")
   ```

2. **Verify Metadata Consistency:**
   ```python
   # Our CalibrationManager checks these automatically, but verify manually if issues:
   def check_metadata_consistency(file_list):
       """Check critical keywords are consistent."""
       first_hdr = fits.getheader(file_list[0])

       for f in file_list[1:]:
           hdr = fits.getheader(f)
           # Check detector settings match
           assert hdr['NAXIS1'] == first_hdr['NAXIS1'], "Image sizes differ!"
           assert hdr.get('GAIN', 1.0) == first_hdr.get('GAIN', 1.0), "GAIN differs!"
           assert hdr.get('READNOIS', 0) == first_hdr.get('READNOIS', 0), "Read noise differs!"

   check_metadata_consistency(bias_files)
   ```

3. **Use Enhanced CalibrationManager:**
   ```python
   from astro_vision_composer.preprocessing import CalibrationManager

   # Handles overscan, flexible keywords, memory limits automatically!
   calib_mgr = CalibrationManager(
       calibration_dir='raw_data/calibration/',
       overscan_region='[:, 2049:2080]' if has_overscan else None,
       master_cache_dir='masters/',  # Reuse calibrations across sessions
       mem_limit=8e9  # 8 GB limit for large datasets
   )

   # Try cached, create if needed
   if not calib_mgr.load_cached_calibrations():
       calib_mgr.create_master_calibrations()

   # Calibrate science frames
   from ccdproc import CCDData
   science = CCDData.read('science_V.fits', unit='adu')
   calibrated = calib_mgr.calibrate(science, filter_name='V')
   ```

**Best Practices for Raw Data:**
- **Minimum frames:** 10 bias, 5 darks per exposure time, 5 flats per filter
- **Check for overscan:** Many CCDs have overscan regions (extra columns for bias estimation)
- **Verify units:** ADU, electrons, counts, DN - CalibrationManager auto-detects from BUNIT
- **Temperature matching:** Dark frames should match science frame temperature (±5°C)
- **Flat illumination:** Dome flats (consistent) or sky flats (natural spectrum)

---

### Data Quality Assessment Checklist

**Before processing FITS files, check these critical quality indicators:**

#### **1. Exposure Time & Signal-to-Noise**

```python
from astropy.io import fits
import numpy as np

def assess_exposure_quality(fits_file):
    """Check if exposure is suitable for RGB processing."""
    with fits.open(fits_file) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header

        # Check exposure time
        exptime = hdr.get('EXPTIME', 0)
        if exptime < 10:
            print(f"[WARNING] Very short exposure ({exptime}s) - may have low SNR")
        elif exptime > 3600:
            print(f"[OK] Long exposure ({exptime}s) - good SNR expected")

        # Estimate SNR from data statistics
        median_counts = np.nanmedian(data)
        std_counts = np.nanstd(data)
        snr_estimate = median_counts / std_counts if std_counts > 0 else 0

        print(f"Median signal: {median_counts:.1f} counts")
        print(f"Estimated SNR: {snr_estimate:.1f}")

        if snr_estimate < 5:
            print("[ERROR] Very low SNR - image will be noisy")
        elif snr_estimate > 50:
            print("[OK] High SNR - clean image expected")

        return exptime, snr_estimate

# Example usage
assess_exposure_quality('jw02739_nircam_f200w_cal.fits')
```

#### **2. Saturation Check**

```python
def check_saturation(fits_file, saturation_limit=None):
    """Check for saturated pixels that will affect RGB quality."""
    with fits.open(fits_file) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header

        # Auto-detect saturation limit from header
        if saturation_limit is None:
            saturation_limit = hdr.get('SATURATE', hdr.get('MAXDATA', np.max(data) * 0.95))

        saturated_pixels = np.sum(data >= saturation_limit * 0.95)
        total_pixels = data.size
        saturation_fraction = saturated_pixels / total_pixels

        print(f"Saturation limit: {saturation_limit}")
        print(f"Saturated pixels: {saturated_pixels} ({saturation_fraction*100:.2f}%)")

        if saturation_fraction > 0.1:
            print("[ERROR] >10% saturated - will create artifacts in bright regions")
        elif saturation_fraction > 0.01:
            print("[WARNING] 1-10% saturated - bright stars may bloom")
        else:
            print("[OK] Minimal saturation")

        return saturation_fraction

# Example
check_saturation('hst_m51_f606w_flc.fits')
```

#### **3. WCS Validation**

```python
from astro_vision_composer.processing import WCSHandler

def validate_wcs_quality(fits_file):
    """Check WCS quality before using for reprojection."""
    handler = WCSHandler()

    try:
        wcs = handler.load_wcs(fits_file)
        info = handler.validate(wcs)

        print(f"WCS Type: {'gwcs (ASDF)' if info.has_gwcs else 'FITS WCS'}")
        print(f"Valid: {info.is_valid}")
        print(f"Projection: {info.projection}")

        if info.pixel_scale:
            print(f"Pixel scale: {info.pixel_scale:.4f} arcsec/pixel")
        else:
            print("[WARNING] Could not determine pixel scale")

        # Check for distortion correction
        if info.has_distortion:
            print("[OK] Has distortion correction (SIP/gwcs)")
        else:
            print("[WARNING] No distortion correction - may have geometric errors")

        # gwcs-specific checks (Phase 3A/3B features)
        if info.has_gwcs:
            print(f"Available frames: {info.available_frames}")
            print("[OK] Full gwcs support - advanced WCS features available")

        return info.is_valid

    except Exception as e:
        print(f"[ERROR] WCS loading failed: {e}")
        return False

# Example
validate_wcs_quality('jw02739_nircam_f200w_cal.fits')
```

#### **4. Filter Wavelength Coverage**

```python
def assess_filter_coverage(filter_list):
    """Check if filter combination provides good RGB wavelength coverage."""
    # Wavelength database (nm) - expand as needed
    wavelengths = {
        # JWST NIRCam
        'F090W': 902, 'F115W': 1154, 'F150W': 1501, 'F200W': 1989,
        'F277W': 2786, 'F356W': 3568, 'F444W': 4421,
        # HST ACS/WFC
        'F435W': 435, 'F475W': 475, 'F606W': 606, 'F814W': 814,
        # HST WFC3/UVIS
        'F275W': 275, 'F336W': 336, 'F438W': 438, 'F555W': 555,
        # Ground-based broadband
        'U': 365, 'B': 445, 'V': 551, 'R': 658, 'I': 806,
        # Narrowband
        'Ha': 656, 'OIII': 501, 'SII': 672, 'H-alpha': 656
    }

    # Extract wavelengths
    waves = []
    for f in filter_list:
        # Try exact match first
        if f in wavelengths:
            waves.append(wavelengths[f])
        else:
            # Try partial match (e.g., 'F200W' from full name)
            for key, val in wavelengths.items():
                if key in f.upper():
                    waves.append(val)
                    break

    if len(waves) != 3:
        print(f"[WARNING] Could not identify wavelengths for all filters")
        return False

    waves.sort()
    span = waves[2] - waves[0]
    ratio = waves[2] / waves[0] if waves[0] > 0 else 0

    print(f"Wavelength range: {waves[0]} - {waves[2]} nm")
    print(f"Span: {span} nm (ratio: {ratio:.2f}×)")

    # Assessment
    if span < 100:
        print("[ERROR] Very narrow wavelength range - poor color separation")
        print("    Consider using more separated filters")
    elif span < 200:
        print("[WARNING] Narrow range - limited color information")
    elif span > 1000:
        print("[OK] Excellent wavelength coverage")
    else:
        print("[OK] Good wavelength coverage")

    # Check for narrowband (all within 50nm)
    if max(waves) - min(waves) < 50:
        print("[INFO] Narrowband data detected - use Phase 4 palette mapping!")
        print("    Recommended: Hubble palette (SII->Red, Ha->Green, OIII->Blue)")

    return True

# Examples
print("JWST NIRCam IR:")
assess_filter_coverage(['F444W', 'F200W', 'F090W'])

print("\nHST Optical:")
assess_filter_coverage(['F814W', 'F606W', 'F435W'])

print("\nNarrowband:")
assess_filter_coverage(['Ha', 'OIII', 'SII'])
```

---

### Filter Selection Guidelines

#### **Optimal Wavelength Separation**

**Goal:** Maximize wavelength span while maintaining good SNR

**Rules of Thumb:**
- **Broadband RGB:** Aim for 2-3× wavelength ratio (e.g., 400nm → 800nm)
- **Infrared RGB:** Ratio of 3-5× common (e.g., 1μm → 4.4μm for JWST)
- **Narrowband:** All filters within ~200nm - REQUIRES palette mapping (Phase 4)

**Examples:**

| Dataset | Filters | Wavelength Ratio | Quality |
|---------|---------|------------------|---------|
| JWST NIRCam | F090W, F200W, F444W | 4.9× | [OK] Excellent |
| HST ACS | F435W, F606W, F814W | 1.9× | [OK] Good |
| HST WFC3 narrow | F656N, F658N, F673N | 1.03× | [WARNING] Use palettes! |
| Ground-based | B, V, R | 1.5× | [OK] Adequate |
| JWST MIRI | F560W, F770W, F1800W | 3.2× | [OK] Excellent |

#### **Mission-Specific Recommendations**

**JWST NIRCam (Infrared):**
```python
# Best 3-filter combinations for RGB
combinations = [
    ['F444W', 'F200W', 'F090W'],  # Widest span (4.9×)
    ['F356W', 'F200W', 'F115W'],  # Good span (3.1×)
    ['F277W', 'F150W', 'F090W'],  # Moderate span (3.1×)
]
```

**HST ACS/WFC (Optical):**
```python
combinations = [
    ['F814W', 'F606W', 'F435W'],  # Classic I-V-B
    ['F850LP', 'F606W', 'F475W'], # Extended red
    ['F775W', 'F606W', 'F435W'],  # Standard optical
]
```

**Ground-Based (Johnson-Cousins):**
```python
combinations = [
    ['I', 'V', 'B'],     # Maximum span (2.2×)
    ['R', 'V', 'B'],     # Standard optical (1.5×)
    ['Ha', 'R', 'B'],    # With narrowband (use palette!)
]
```

---

### Common Pitfalls to Avoid

#### **1. Mixing Calibration Levels**

**WRONG:**
```python
# [ERROR] Don't mix calibrated and uncalibrated!
files = [
    'jw02739_f444w_cal.fits',    # Calibrated
    'jw02739_f200w_rate.fits',   # Not calibrated
    'jw02739_f090w_uncal.fits'   # Raw
]
```

**RIGHT:**
```python
# [OK] All at same calibration level
files = [
    'jw02739_f444w_cal.fits',
    'jw02739_f200w_cal.fits',
    'jw02739_f090w_cal.fits'
]
```

#### **2. Ignoring Pixel Scale Mismatches**

**WRONG:**
```python
# [ERROR] Different pixel scales without reprojection
files = [
    'hst_acs_f606w_flc.fits',      # 0.05 arcsec/pixel
    'hst_wfc3_f160w_drz.fits',     # 0.04 arcsec/pixel
    'jwst_nircam_f200w_cal.fits'   # 0.031 arcsec/pixel
]
# Will create misaligned RGB!
```

**RIGHT:**
```python
# [OK] Enable reprojection (Phase 3 feature)
pipeline = ProcessingPipeline(mode='scientific', enable_reprojection=True)
# OR: use pre-resampled products
files = [
    'hst_acs_f606w_drz.fits',      # All drizzled to
    'hst_wfc3_f160w_drz.fits',     # common 0.04 arcsec/pixel
    'combined_mosaic.fits'         # grid
]
```

#### **3. Insufficient Exposure Time**

**Check exposure times before processing:**
```python
from astropy.io import fits

for f in files:
    exptime = fits.getheader(f).get('EXPTIME', 0)
    if exptime < 30:
        print(f"[WARNING] {f}: Only {exptime}s exposure - may be too short")
```

**Minimum recommended exposures:**
- **Space telescopes:** 100s+ per filter (depends on target brightness)
- **Ground-based (large aperture):** 300s+ per filter
- **Ground-based (amateur):** 1800s+ per filter (or stack multiple)

#### **4. Using Event Lists Instead of Images**

**WRONG:**
```python
# [ERROR] Chandra event list is not an image
files = ['chandra_acis_evt2.fits']  # This is a table!
```

**RIGHT:**
```python
# [OK] Bin events into energy-band images first
# (Use CIAO tools - see Chandra section above)
files = ['soft_0.5-1.5keV.fits', 'medium_1.5-3keV.fits', 'hard_3-7keV.fits']
```

#### **5. Forgetting Narrowband Requires Palettes**

**WRONG:**
```python
# [ERROR] Narrowband with chromatic ordering
files = ['sii_672nm.fits', 'ha_656nm.fits', 'oiii_501nm.fits']
pipeline = ProcessingPipeline(mode='scientific')  # Will map OIII->Red!
rgb = pipeline.process_to_rgb(files)  # Colors will be wrong
```

**RIGHT:**
```python
# [OK] Use Hubble palette (Phase 4 feature)
# Map: SII->Red, Ha->Green, OIII->Blue
files = ['sii_672nm.fits', 'ha_656nm.fits', 'oiii_501nm.fits']
pipeline = ProcessingPipeline(mode='manual', palette='hubble')
rgb = pipeline.process_to_rgb(files)
```

---

### Quick Reference: Best File Types by Mission

| Mission | Best Suffix | Includes WCS? | Includes Distortion? | Notes |
|---------|-------------|---------------|----------------------|-------|
| **JWST** | `_cal.fits` | gwcs (ASDF) | Yes (full model) | Recommended for Phase 3A/3B features |
| **JWST** | `_i2d.fits` | FITS WCS | No (pre-corrected) | Use for mosaics |
| **HST** | `_flc.fits` | FITS WCS | Yes (SIP + D2IM) | ACS/WFC3 with CR rejection |
| **HST** | `_flt.fits` | FITS WCS | Yes (SIP) | Older instruments |
| **HST** | `_drz.fits` | FITS WCS | No (pre-corrected) | Drizzled mosaics |
| **Chandra** | `_img.fits` | FITS WCS | No | Energy-filtered images only |
| **Ground** | Calibrated | FITS WCS | Maybe (SIP if added) | Must calibrate first (Phase 2) |

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
