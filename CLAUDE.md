# Astronomical FITS Processing Pipeline - Status & Implementation Plan
**Updated:** 2025-10-26  
**Status:** Core complete, needs testing, workflow refinement, and preprocessing integration

---

## Executive Summary

**Current State:** All 16 core processing components implemented (~5,200 LOC). The pipeline can process pre-calibrated FITS files to RGB export. However: (1) implementation deviates from astropy best practices, (2) test coverage is 0%, (3) CLAHE/color-correction are low quality and need tagging, (4) raw data preprocessing (ccdproc integration) is incomplete, (5) manual workflow mode is missing, (6) our test examples are too simple (nearly perfect FITS files).

**Mission:** Build production-grade pipeline for processing real-world FITS files (JWST, HST, Chandra, Euclid, PanSTARRS, ground-based raw data) into presentation-quality RGB images using astropy's ecosystem (astropy.visualization, ccdproc, reproject, gwcs).

---

## Critical Design Requirements

### Manual Workflow Mode (HIGH PRIORITY - NOT IMPLEMENTED)

**Requirement:** Users must be able to provide arrays of processing classes with or without parameters for maximum flexibility.

**Current Implementation:** Pipeline modes are hardcoded with automatic selection of intervals/stretches. No way to specify custom processing chains.

**Required Design:**
```python
# Manual workflow with explicit processing chain
from astropy.visualization import (
    ZScaleInterval, AsymmetricPercentileInterval,
    AsinhStretch, LogStretch, HistEqStretch
)

pipeline = ProcessingPipeline(mode='manual')

# Define per-band normalization strategies
normalizations = [
    ImageNormalize(interval=ZScaleInterval(), stretch=AsinhStretch(a=0.1)),
    ImageNormalize(interval=PercentileInterval(99.0), stretch=LogStretch(a=1000)),
    ImageNormalize(interval=AsymmetricPercentileInterval(2, 99.5), stretch=SqrtStretch())
]

# Or provide arrays of interval/stretch objects separately
intervals = [ZScaleInterval(), PercentileInterval(99.5), PercentileInterval(98.0)]
stretches = [AsinhStretch(), LinearStretch(), LinearStretch()]

rgb = pipeline.process_to_rgb(
    fits_files=['ha.fits', 'oiii.fits', 'sii.fits'],
    normalizations=normalizations,  # Option 1: Full ImageNormalize objects
    # OR
    intervals=intervals,            # Option 2: Separate arrays
    stretches=stretches,
    # Override default Lupton compositor
    compositor='simple',  # or provide custom Compositor instance
    output_dir='output/'
)
```

**Benefits:**
- Complete user control over each band's processing
- Essential for narrowband imaging (Ha/OIII/SII need different treatments)
- Supports experimentation and iterative refinement
- Matches real-world workflows from experienced imagers

**Implementation Status:** ❌ NOT IMPLEMENTED - Add to Phase 1

---

### Quality Warnings & Component Tagging

**CRITICAL:** The following components are LOW QUALITY and must not be deployed without fixing or removing:

#### 🚨 CLAHE Implementation (`enhancer.py` lines 37-92)
**Status:** QUESTIONABLE QUALITY - NEEDS REVIEW

**Issues:**
- Uses `skimage.exposure.equalize_adapthist()` with default parameters
- No validation that CLAHE is appropriate for astronomical data
- Kernel size auto-calculation is simplistic (max_dim // 8)
- May over-enhance noise in low-SNR regions
- No masking for stars vs. nebulosity (treats all equally)

**Tags Required:**
```python
@deprecated(reason="CLAHE implementation needs astronomical-specific tuning", version="0.2.0")
@experimental(quality="LOW", warning="May over-enhance noise. Use with caution.")
```

**Recommended Action:**
1. Add extensive warnings in docstring
2. Implement star masking to protect point sources
3. Add SNR-based adaptive clipping
4. Test on real astronomical data (not just synthetic)
5. Or: Remove entirely until proper implementation exists

#### 🚨 Color Balance/White Balance (`color_balancer.py`)
**Status:** NAIVE IMPLEMENTATION - NOT PRODUCTION READY

**Issues:**
- `white_balance()` uses simple channel scaling without color space conversion
- No support for perceptual color spaces (LAB, LCh)
- Temperature adjustment (lines 155-200) uses ad-hoc RGB shifts
- No validation against photometric color ratios
- Saturation adjustment clips rather than preserving luminance

**Tags Required:**
```python
@experimental(quality="LOW", warning="Naive RGB operations. Not photometrically accurate.")
@warning("Color balance destroys photometric information. Use only for aesthetic output.")
```

**Recommended Action:**
1. Add prominent warnings that these operations are non-photometric
2. Implement proper color space conversions (RGB → LAB → RGB)
3. Or: Remove and document that users should use external tools (Photoshop, GIMP)

#### 🟡 Unsharp Masking (`enhancer.py` lines 94-150)
**Status:** ACCEPTABLE BUT LIMITED

**Issues:**
- Basic Gaussian blur unsharp mask
- No luminance masking (affects all pixels equally)
- Single-scale only (no multi-scale sharpening)

**Action:** Add warning that this is basic implementation. Document limitations.

---

## Preprocessing: The Missing Foundation

### Current Gap: No Raw Data Support

**Problem:** Our current pipeline assumes pre-calibrated FITS files (JWST `_cal.fits`, HST `_flc.fits`). We cannot process raw ground-based or amateur data that requires:
- Bias/overscan correction
- Dark current subtraction  
- Flat-field correction
- Bad pixel masking
- Cosmic ray rejection

**Impact:** ~60% of potential users (ground-based observers, amateur astrophotographers, researchers doing custom reductions) cannot use the pipeline.

### Solution: Integrate `ccdproc`

**`ccdproc`** is the Astropy-affiliated package specifically designed for fundamental CCD calibration. It provides:

#### The CCDData Object
Extension of numpy arrays with:
- **Units** (`astropy.units`) - prevents unit mismatch errors (seconds vs. milliseconds)
- **Metadata** (FITS headers) - automatically propagated through operations
- **Uncertainty** - proper error propagation through all calibration steps
- **Masking** - bad pixel tracking

**Example:**
```python
from ccdproc import CCDData
import astropy.units as u

# Load raw image with units and uncertainty tracking
ccd = CCDData.read('raw_science.fits', unit='adu')
print(f"Exposure time: {ccd.header['EXPTIME']} {ccd.unit}")
```

#### Master Calibration Frame Generation

**Problem:** Single calibration frames contain random noise and cosmic rays.  
**Solution:** Combine multiple frames with outlier rejection.

```python
from ccdproc import Combiner, ImageFileCollection
import numpy as np

# Find all bias frames in directory
ic = ImageFileCollection('raw_data/', keywords='*')
bias_files = ic.files_filtered(imagetyp='BIAS', include_path=True)

# Load as CCDData objects
bias_list = [CCDData.read(f, unit='adu') for f in bias_files]

# Combine with sigma clipping to reject cosmic rays
combiner = Combiner(bias_list)
combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.median)
master_bias = combiner.average_combine()  # or median_combine()

# Result has proper uncertainty propagation
print(f"Master bias uncertainty: {master_bias.uncertainty}")
```

**Key Features:**
- `sigma_clipping()` - Remove cosmic rays and hot pixels (3-sigma median clipping)
- `clip_extrema()` - IRAF-style min/max rejection
- `scaling` - Normalize flats to common level before combining
- `mem_limit` - Process huge datasets by chunking (don't exhaust RAM)

#### Calibration Pipeline

**Standard CCD reduction sequence:**

```python
import ccdproc as ccdp

# Method 1: Manual step-by-step (maximum control)
processed = raw_science
processed = ccdp.subtract_overscan(processed, overscan=raw_science[:, 2048:])
processed = ccdp.trim_image(processed[:, :2048])
processed = ccdp.subtract_bias(processed, master_bias)
processed = ccdp.subtract_dark(processed, master_dark, 
                                exposure_time=300*u.second,
                                exposure_unit=u.second,
                                scale=True)  # Auto-scale for exposure time
processed = ccdp.flat_correct(processed, master_flat)

# Method 2: One-step pipeline (recommended for production)
calibrated = ccdp.ccd_process(
    raw_science,
    oscan=raw_science[:, 2048:],
    trim='[:, :2048]',
    master_bias=master_bias,
    dark_frame=master_dark,
    master_flat=master_flat,
    exposure_key='EXPTIME',
    exposure_unit=u.second,
    dark_scale=True
)
```

**Critical ccdproc Features:**
- **Unit safety:** Dark scaling requires exposure times in same units. ccdproc enforces `astropy.units`, preventing 1000× errors from seconds/milliseconds mismatch.
- **Uncertainty propagation:** Every operation updates uncertainty array using proper error propagation rules.
- **FITS provenance:** Operations automatically append HISTORY cards to header.

#### Auto-Detection of Calibration Files

**Required Feature:** Pipeline should automatically find and combine calibration files.

```python
class CalibrationManager:
    """Automatically find and combine calibration frames."""
    
    def __init__(self, data_directory):
        self.ic = ImageFileCollection(data_directory, keywords='*')
    
    def create_master_bias(self) -> CCDData:
        """Find all bias frames and combine."""
        bias_files = self.ic.files_filtered(imagetyp='BIAS')
        if len(bias_files) == 0:
            raise ValueError("No bias frames found")
        
        bias_list = [CCDData.read(f, unit='adu') for f in bias_files]
        combiner = Combiner(bias_list)
        combiner.sigma_clipping(low_thresh=3, high_thresh=3)
        return combiner.average_combine()
    
    def create_master_dark(self, exposure_time: float) -> CCDData:
        """Find matching dark frames and combine."""
        # Find darks with matching exposure time (±1 second tolerance)
        dark_files = self.ic.files_filtered(
            imagetyp='DARK',
            exptime=f'{exposure_time}±1'
        )
        # ... combine similar to bias
    
    def create_master_flat(self, filter_name: str) -> CCDData:
        """Find matching flat frames for specific filter."""
        flat_files = self.ic.files_filtered(
            imagetyp='FLAT',
            filter=filter_name
        )
        
        flat_list = [CCDData.read(f, unit='adu') for f in flat_files]
        
        # CRITICAL: Scale flats to common level before combining
        def inv_median(arr):
            return 1.0 / np.ma.median(arr)
        
        combiner = Combiner(flat_list)
        combiner.scaling = inv_median  # Normalize each flat to median=1.0
        combiner.sigma_clipping(low_thresh=3, high_thresh=3)
        master_flat = combiner.median_combine()
        
        # Normalize final flat to 1.0
        master_flat = master_flat.divide(np.ma.median(master_flat.data)*master_flat.unit)
        return master_flat
```

**Integration into Pipeline:**
```python
pipeline = ProcessingPipeline(mode='scientific', calibration_dir='raw_data/')

# Pipeline auto-detects and combines calibration frames
# Then processes science frames
rgb = pipeline.process_to_rgb(
    fits_files=['science_r.fits', 'science_g.fits', 'science_b.fits'],
    auto_calibrate=True  # Use calibration files from calibration_dir
)
```

---

## Multi-Mission WCS & Distortion Correction

### The Real-World Challenge

**Problem:** Our test examples (NOIRLab FITS) are nearly perfect:
- Clean WCS headers with SIP distortion coefficients
- Pre-aligned or single-instrument
- No complex MEF structures
- Standard TAN projection

**Reality:** Real multi-mission data is messy:
- HST: Multi-layer distortion (polynomial + lookup tables + filter-dependent)
- JWST: WCS embedded in binary ASDF extension, not readable by standard tools
- Chandra: Time-dependent WCS (dither pattern), requires aspect solution file
- Euclid: 36-CCD mosaic, each with independent WCS solution
- Ground-based: May have NO valid WCS, requires astrometry.net or custom solutions

### Mission-Specific WCS Architectures

#### HST (Hubble Space Telescope)
**WCS Model:** Layered polynomial + lookup table corrections

**Challenges:**
1. **IDCTAB reference files** - Fourth-order polynomial distortion coefficients stored externally
2. **D2IMFILE** - 2D lookup table for detector manufacturing defects (WFC3/UVIS)
3. **NPOLFILE** - Filter-dependent distortion corrections
4. **Multiple layers** must be applied in sequence

**Solution:** Use `drizzlepac.stwcs`
```python
from drizzlepac import stwcs

# Standard astropy.wcs CANNOT read distortion from reference files
# Must use drizzlepac's HSTWCS class
hst_wcs = stwcs.wcsutil.HSTWCS('hst_acs_flt.fits', ext=('SCI', 1))

# This WCS object includes ALL distortion layers
# Now high-precision astrometry is possible
```

**Critical:** Requires CRDS (Calibration Reference Data System) environment:
```bash
export CRDS_PATH="/path/to/crds_cache"
export CRDS_SERVER_URL="https://hst-crds.stsci.edu"
```

#### JWST (James Webb Space Telescope)
**WCS Model:** Generalized WCS (gwcs) pipeline in ASDF format

**Paradigm Shift:** WCS is not parameters in FITS header, but an executable pipeline of transformations:
```
Detector → Local Frame → Telescope → V2V3 → World
    ↓           ↓             ↓         ↓        ↓
 (pixels)   (distortion)  (pointing) (optics) (RA/Dec)
```

**Challenge:** Standard FITS readers cannot access ASDF extension.

**Solution:** Use `stdatamodels`
```python
from stdatamodels.jwst import datamodels

# Must use JWST-specific datamodel, not astropy.io.fits
with datamodels.open('jwst_nircam_cal.fits') as model:
    # WCS is in model.meta.wcs, not header
    wcs = model.meta.wcs
    
    # Inspect the pipeline
    print(wcs.available_frames)  # Shows transformation stages
    # ['detector', 'v2v3', 'world']
    
    # Transform pixel to world
    ra, dec = wcs(1024, 1024)
```

**Critical:** JWST WCS includes:
- Geometric distortion polynomials
- Velocity aberration corrections
- Telescope pointing models
- All serialized as astropy.modeling pipeline

#### Euclid Space Telescope
**WCS Model:** Independent solutions per detector quadrant

**Challenge:** Single exposure spans 36 CCDs (VIS) or 16 detectors (NISP). Cannot have one WCS for entire image.

**Structure:** Multi-extension FITS where each SCI extension is a detector quadrant with its own WCS.

```python
from astropy.io import fits
from astropy.wcs import WCS

with fits.open('euclid_vis_cal.fits') as hdul:
    # Each detector quadrant is separate HDU
    for hdu in hdul:
        if 'SCI' in hdu.name:
            # Independent WCS per quadrant
            wcs = WCS(hdu.header)
            print(f"{hdu.name} (quad {hdu.ver}): "
                  f"center = {wcs.pixel_to_world(1024, 1024)}")
```

**Implication:** Mosaic algorithms must treat each quadrant independently, then stitch.

#### Chandra X-ray Observatory
**WCS Model:** Time-resolved aspect solution (dynamic WCS)

**Challenge:** Data is event list (photon table), not image. Telescope dithers during observation, so same star hits different pixels at different times.

**Required Files:**
- `*_evt2.fits` - Event list with (TIME, CHIPX, CHIPY, ENERGY)
- `*_asol1.fits` - Aspect solution with (TIME, RA, DEC, ROLL)

**Process:** Must interpolate aspect solution to event times, then bin into image.

```python
from ciao_contrib.runtool import dmcoords, dmcopy

# Apply aspect solution to event file
dmcoords(infile='evt2.fits', asolfile='asol1.fits', ...)

# Bin events into image with energy filter (0.5-7 keV)
dmcopy('evt2.fits[energy=500:7000][bin x=::1,y=::1]', 'image.fits')
```

**Our Current Support:** ❌ NOT IMPLEMENTED  
**Required:** `EventBinner` class (deferred to Phase 5)

### Reprojection: The Non-Negotiable Alignment Step

**Problem:** Cannot create RGB composite from images with different:
- Pixel scales (arcsec/pixel)
- Image orientations (rotation)
- Map projections (TAN vs. SIN vs. ARC)
- Image sizes

**Solution:** Reproject all images to a common WCS grid.

#### The `reproject` Library

**Two Algorithm Families:**

**1. Interpolation-based (fast, approximate)**
```python
from reproject import reproject_interp

reprojected, footprint = reproject_interp(
    input_data=(source_data, source_wcs),
    output_projection=target_wcs,
    shape_out=target_shape,
    order=1  # 0=nearest, 1=bilinear, 2=bicubic
)
```

**Pros:** Fast, good for visualization  
**Cons:** Does NOT conserve flux (photometry invalid)  
**Use case:** RGB composites, quick-look images

**2. Exact/adaptive (slow, flux-conserving)**
```python
from reproject import reproject_exact, reproject_adaptive

# For high-accuracy reprojection
reprojected, footprint = reproject_exact(
    input_data=(source_data, source_wcs),
    output_projection=target_wcs,
    shape_out=target_shape
)
```

**Pros:** Conserves flux, photometrically accurate  
**Cons:** Slow (100× slower than interp)  
**Use case:** Photometry, scientific analysis

**3. Adaptive (compromise)**
```python
# Adaptively switches between exact and interpolation based on distortion
reprojected, footprint = reproject_adaptive(
    input_data=(source_data, source_wcs),
    output_projection=target_wcs,
    shape_out=target_shape
)
```

#### Choosing Target Frame

**Critical Decision:** Which image should be the reference frame?

**Best Practice:**
1. **Highest resolution** - Don't degrade data
2. **Largest field of view** - Don't crop data
3. **Best WCS quality** - Most reliable distortion correction

```python
def select_reference_frame(images_with_wcs):
    """Select optimal reference frame for reprojection."""
    best_score = -np.inf
    best_idx = 0
    
    for i, (data, wcs, header) in enumerate(images_with_wcs):
        # Score based on resolution (arcsec/pixel)
        pixel_scale = np.abs(wcs.wcs.cdelt[0]) * 3600  # deg to arcsec
        resolution_score = 1.0 / pixel_scale  # Higher score = finer pixels
        
        # Score based on field of view
        fov_score = data.shape[0] * data.shape[1]
        
        # Score based on WCS quality (has distortion?)
        wcs_score = 2.0 if wcs.sip is not None else 1.0
        
        total_score = resolution_score * fov_score * wcs_score
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
    
    return best_idx
```

#### Handling Reprojection Artifacts

**Common Issues:**

**1. NaN pixels** - Areas with no input data coverage
```python
# Fill NaNs with zeros (background)
reprojected = np.nan_to_num(reprojected, nan=0.0)

# Or: use footprint to mask invalid regions
valid_mask = footprint > 0.5  # footprint=1 where valid
```

**2. Edge artifacts** - Interpolation at boundaries
```python
# Shrink footprint slightly to exclude edge pixels
from scipy import ndimage
valid_mask = ndimage.binary_erosion(footprint > 0.5, iterations=2)
reprojected[~valid_mask] = 0
```

**3. Flux errors near poles** - Projection distortion
```python
# Use adaptive reprojection near celestial poles
if np.abs(target_wcs.wcs.crval[1]) > 80:  # Dec > ±80 degrees
    reprojected, footprint = reproject_adaptive(...)
```

### Our Current Implementation Status

**Reprojector class (reprojector.py):**
- ✅ Wraps `reproject_interp` and `reproject_exact`
- ✅ Basic error handling
- ❌ No automatic reference frame selection
- ❌ No artifact handling (NaN filling, edge cropping)
- ❌ No WCS validation before reprojection
- ❌ No support for Chandra time-dependent WCS
- ❌ No support for JWST gwcs (assumes FITS WCS)

**Required Improvements:**
1. Add `select_reference_frame()` method
2. Implement artifact mitigation
3. Add WCS validation and quality scoring
4. Support gwcs for JWST data
5. Add special handling for Euclid multi-detector mosaics

---

## Palette Selection: The Under-Discussed Challenge

### The Problem

**Our test data (NOIRLab):** 3 broadband optical filters covering 400-900nm. Chromatic ordering is obvious:
- Shortest wavelength (blue filter) → Blue channel
- Middle wavelength (green filter) → Green channel  
- Longest wavelength (red filter) → Red channel

**Result:** Natural-looking color images with minimal tuning.

**Real-world scenarios:**

1. **Narrowband imaging (Ha/OIII/SII):** All three wavelengths are in the red part of spectrum (620-680nm). No "natural" color exists.

2. **Infrared imaging (JWST):** Observing at 1-5 μm (invisible to human eye). Must map to visible colors arbitrarily.

3. **Multi-spectral (10+ bands):** How to select 3 bands from 10 available filters?

4. **Mixed spectral coverage:** Combining UV, optical, infrared in single image.

### Palette Strategies

#### 1. Chromatic Ordering (Current Implementation)
**What it does:** Maps longest→Red, middle→Green, shortest→Blue

**Good for:**
- Optical broadband (ugriz, BVRI)
- When wavelength span covers most of visible spectrum

**Fails for:**
- All-infrared data (no wavelength corresponds to visible colors)
- Narrowband data (wavelengths too close together)

```python
# Current ChannelMapper
mapper = ChannelMapper()
wavelengths = {'f090w': 0.9, 'f200w': 2.0, 'f444w': 4.4}  # μm
mapping = mapper.auto_map_by_wavelength(wavelengths)
# Result: f444w→Red, f200w→Green, f090w→Blue
# But all three are infrared! No correspondence to "real" color
```

#### 2. False-Color Palettes (Narrowband)
**Strategy:** Assign visible colors to enhance specific emission lines

**Standard Palettes:**

**Hubble Palette (SHO):**
```
S (SII 672nm) → Red
H (Ha 656nm) → Green
O (OIII 500nm) → Blue
```

**HOO Bicolor:**
```
H (Ha) → Red
O (OIII) → Cyan (synthesized from Green + Blue)
```

**Implementation:**
```python
class PaletteMapper:
    """Map narrowband channels to perceptual colors."""
    
    PALETTES = {
        'hubble': {'sii': 'red', 'ha': 'green', 'oiii': 'blue'},
        'hoo': {'ha': 'red', 'oiii': 'cyan'},  # Needs special handling
        'natural': {'ha': 'red', 'oiii': 'cyan', 'sii': 'red'}  # Mixed
    }
    
    def map_narrowband(self, bands: Dict[str, np.ndarray], palette='hubble'):
        """Map narrowband data to RGB using named palette."""
        p = self.PALETTES[palette]
        
        if palette == 'hoo':
            # Special case: synthesize cyan from OIII
            r = bands['ha']
            g = bands['oiii']
            b = bands['oiii']  # Cyan = G + B
        else:
            r = bands[list(p.keys())[p[list(p.keys())[0]] == 'red'][0]]
            g = bands[list(p.keys())[p[list(p.keys())[0]] == 'green'][0]]
            b = bands[list(p.keys())[p[list(p.keys())[0]] == 'blue'][0]]
        
        return r, g, b
```

#### 3. Representative Color (Infrared)
**Strategy:** Map wavelengths to perceptual equivalents

**JWST NIRCam Example:**
```
F090W (0.9 μm) → Blue   (representative of short-wave IR)
F200W (2.0 μm) → Green  (mid-wave IR)
F444W (4.4 μm) → Red    (long-wave IR)
```

**Rationale:** Preserve relative wavelength ordering, map to visible spectrum for intuitive interpretation.

#### 4. Multi-Band Selection (>3 filters)
**Problem:** Have 10 filters, need to pick 3 for RGB.

**Strategies:**

**a) Maximum wavelength span:**
```python
def select_max_span(wavelengths: Dict[str, float]) -> Tuple[str, str, str]:
    """Select 3 filters with maximum wavelength separation."""
    sorted_bands = sorted(wavelengths.items(), key=lambda x: x[1])
    return sorted_bands[0][0], sorted_bands[len(sorted_bands)//2][0], sorted_bands[-1][0]
```

**b) Principal component selection:**
```python
from sklearn.decomposition import PCA

def select_pca_bands(images: Dict[str, np.ndarray]) -> Tuple[str, str, str]:
    """Select 3 bands that capture most variance."""
    # Stack images into data matrix
    data = np.stack([img.flatten() for img in images.values()], axis=1)
    
    # PCA to find 3 most important components
    pca = PCA(n_components=3)
    pca.fit(data)
    
    # Select bands closest to principal components
    # (Implementation details omitted)
```

**c) Science-driven selection:**
```python
# Example: Emphasize specific features
selection = {
    'star_formation': ('uv', 'ha', 'ir'),  # Young stars, ionized gas, dust
    'stellar_population': ('u', 'g', 'i'),  # Age/metallicity tracers
    'agn_activity': ('xray', 'ir', 'radio')  # Multi-wavelength AGN
}
```

### Required Implementation

**Current:** `ChannelMapper` only does chromatic ordering  
**Needed:** Support for:
- Named palette selection (Hubble, HOO, natural, etc.)
- Custom palette definition
- Multi-band selection strategies
- Wavelength-to-perceptual-color mapping for IR/UV

```python
class AdvancedChannelMapper:
    def map_channels(
        self,
        bands: Dict[str, np.ndarray],
        wavelengths: Dict[str, float],
        palette: Union[str, Dict[str, str]] = 'chromatic',
        selection_strategy: str = 'max_span'
    ) -> ChannelMapping:
        """
        Map channels with flexible strategies.
        
        palette options:
        - 'chromatic': Wavelength ordering (current default)
        - 'hubble': SII→R, Ha→G, OIII→B
        - 'hoo': Ha→R, OIII→Cyan
        - Custom dict: {'band_name': 'red'|'green'|'blue'|'cyan'}
        
        selection_strategy (if len(bands) > 3):
        - 'max_span': Maximum wavelength separation
        - 'pca': Principal component analysis
        - 'manual': User provides list of 3 band names
        """
```

---

## Code Status Assessment

### What's Implemented
**All 16 core components functional:**

**Tier 1 - Preprocessing (5/5 production-ready)**
- `fits_loader.py` (320 lines) - Multi-extension FITS loading
- `mission_adapters.py` (630 lines) - JWST, HST, Chandra adapters
- `quality_assessor.py` (340 lines) - SNR, saturation, noise metrics
- `calibrator.py` (320 lines) - Bias, dark, flat correction
- `metadata.py` (260 lines) - Mission-aware metadata extraction

**Tier 2 - Processing (5/5 production-ready)**
- `wcs_handler.py` (305 lines) - WCS validation and extraction
- `reprojector.py` (250 lines) - Image alignment via reproject
- `normalizer.py` (260 lines) - Interval-based normalization
- `stretcher.py` (260 lines) - Non-linear stretching
- `enhancer.py` (280 lines) - CLAHE, unsharp mask

**Tier 3 - Postprocessing (6/6 production-ready)**
- `channel_mapper.py` (200 lines) - Chromatic wavelength mapping
- `compositor.py` (305 lines) - Lupton & simple RGB
- `color_balancer.py` (280 lines) - White balance, saturation
- `exporter.py` (240 lines) - PNG/TIFF/JPEG export
- `preview.py` (220 lines) - Quick previews
- `history_tracker.py` (200 lines) - Processing provenance

**Utilities (1/1 production-ready, 1/1 needs refinement)**
- `pipeline.py` (458 lines) - End-to-end orchestration [NEEDS REFINEMENT]
- `metadata.py` (260 lines) - Shared metadata utilities

### Critical Gaps

**1. Zero Test Coverage**
- No unit tests exist
- No integration tests
- No validation against known outputs
- Impact: Cannot verify correctness, refactoring is risky

**2. Suboptimal Astropy Integration**
Current implementation doesn't fully leverage astropy.visualization best practices:

**Current approach (pipeline.py lines 214-255):**
```python
# Phase 2: Manual normalization then stretching
normalized = self.normalizer.normalize(data, interval=interval_obj)
stretched = self.stretcher.stretch(normalized, method='asinh')
```

**Astropy best practice:**
```python
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch

# Combined normalization + stretch in single object
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
# Use directly with matplotlib or extract normalized data
normalized_data = norm(data)
```

**Benefits of astropy approach:**
- Single normalization object encapsulates both operations
- Direct matplotlib compatibility
- Proper vmin/vmax handling
- Better stretch composition
- Matches documentation examples exactly

**3. Workflow Inconsistencies**
Current `pipeline.py` has conflicting design patterns:
- "Scientific" mode: Normalizes, stretches, then uses `LinearStretch()` in Lupton (redundant)
- "SDSS" mode: Only normalizes, lets Lupton auto-calculate stretch (correct)
- "Aesthetic" mode: Uses histogram equalization then Lupton (questionable)

**4. Missing Validation Framework**
- No `ValidationReport` class (designed but not implemented)
- No automated quality checks
- No processing metrics collection

---

## Critical Issues to Fix

### Issue 1: Normalization/Stretch Architecture
**Problem:** Current two-step approach (Normalizer → Stretcher) doesn't align with astropy's `ImageNormalize` pattern.

**Root Cause:** `Normalizer` and `Stretcher` were designed as separate classes, but astropy treats them as composed transformations.

**Fix Required:**
Refactor to use astropy's `ImageNormalize` directly:
```python
# Instead of:
normalized = normalizer.normalize(data, interval=ZScaleInterval())
stretched = stretcher.stretch(normalized, method='asinh')

# Do:
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
processed = norm(data)  # Returns [0,1] normalized+stretched array
```

**Impact:**
- Lines 183-274 of `pipeline.py` need rewrite
- `normalizer.py` and `stretcher.py` can be deprecated or refactored as thin wrappers
- Simplifies workflow logic significantly

### Issue 2: Lupton RGB Workflow Confusion
**Problem:** Conflicting stretch object handling in `_compose_rgb()` (lines 276-353).

**Current behavior:**
- Scientific mode: Pre-stretches data, then passes `LinearStretch()` to Lupton (correct but unclear)
- SDSS mode: Only normalizes, lets Lupton auto-calculate via `LuptonAsinhZscaleStretch` (correct)
- Logic spread across multiple private methods with auto-detection

**Correct Implementation:**
Per astropy docs, there are exactly 3 valid Lupton workflows:

**Workflow A: Pre-stretched data**
```python
# Phase 2: Normalize AND stretch
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
stretched = norm(data)  # [0,1] range

# Phase 3: Lupton with identity stretch
from astropy.visualization import LinearStretch
rgb = make_lupton_rgb(r, g, b, stretch_object=LinearStretch())
```

**Workflow B: SDSS auto-calculated**
```python
# Phase 2: Only normalize to physical units
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
normalized = norm(data)

# Phase 3: Let Lupton calculate stretch from data
from astropy.visualization import LuptonAsinhZscaleStretch
stretch = LuptonAsinhZscaleStretch(r_data, Q=8)
rgb = make_lupton_rgb(r, g, b, stretch_object=stretch)
```

**Workflow C: Manual Lupton stretch**
```python
# Phase 2: Only normalize
normalized = ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())(data)

# Phase 3: Explicit Lupton stretch parameters
from astropy.visualization import LuptonAsinhStretch
stretch = LuptonAsinhStretch(stretch=5, Q=8)
rgb = make_lupton_rgb(r, g, b, stretch_object=stretch)
```

**Fix Required:**
- Simplify `_compose_rgb()` to explicitly implement these 3 workflows
- Remove auto-detection logic (lines 388-404)
- Make workflow selection explicit in mode definitions

### Issue 3: Missing Quality Framework
**Problem:** `QualityAssessor` exists but isn't integrated into pipeline.

**Required:**
- Add quality checks after each major phase
- Collect metrics (SNR, saturation %, noise level, dynamic range)
- Generate processing report
- Abort or warn on quality failures

---

## Implementation Plan (REVISED)

### Phase 0: Quality & Safety (1-2 days)
**Priority: CRITICAL - Must be done before any other work**

**0.1 Tag low-quality components**
- Add `@deprecated` and `@experimental` decorators to CLAHE, color balance
- Update docstrings with prominent warnings
- Create QUALITY.md documenting known issues

**0.2 Disable dangerous defaults**
- Remove CLAHE/color-balance from default workflows
- Require explicit opt-in for experimental features
- Add runtime warnings when experimental code is used

### Phase 1: Fix Core Architecture (3-4 days)
**Priority: CRITICAL - Foundation for everything else**

**1.1 Refactor normalization/stretching (1.5 days)**
- Replace manual Normalizer/Stretcher usage with `ImageNormalize`
- Update `pipeline.py` Phase 2 (lines 183-274)
- Maintain backward compatibility by keeping wrappers
- Update all docstrings and examples

**1.2 Implement manual workflow mode (1.5 days)**
```python
# New API design
pipeline = ProcessingPipeline(mode='manual')

# Option 1: Full ImageNormalize objects
normalizations = [
    ImageNormalize(data1, interval=ZScaleInterval(), stretch=AsinhStretch()),
    ImageNormalize(data2, interval=PercentileInterval(99), stretch=LogStretch()),
    ImageNormalize(data3, interval=MinMaxInterval(), stretch=LinearStretch())
]

rgb = pipeline.process_with_normalizations(
    fits_files=['f1.fits', 'f2.fits', 'f3.fits'],
    normalizations=normalizations
)

# Option 2: Separate interval/stretch arrays
intervals = [ZScaleInterval(), PercentileInterval(99), MinMaxInterval()]
stretches = [AsinhStretch(), LogStretch(), LinearStretch()]

rgb = pipeline.process_with_arrays(
    fits_files=['f1.fits', 'f2.fits', 'f3.fits'],
    intervals=intervals,
    stretches=stretches
)
```

**1.3 Simplify Lupton workflows (1 day)**
- Rewrite `_compose_rgb()` to explicitly implement 3 canonical workflows
- Remove auto-detection code
- Update mode definitions

### Phase 2: Preprocessing Integration (4-5 days)
**Priority: HIGH - Enables raw data processing**

**2.1 Implement CalibrationManager (2 days)**
- Auto-detect calibration files (bias, dark, flat)
- Combine using ccdproc.Combiner with sigma clipping
- Handle exposure time matching
- Filter-specific flat selection

**2.2 Integrate ccdproc into pipeline (2 days)**
```python
class ProcessingPipeline:
    def __init__(self, calibration_dir=None):
        if calibration_dir:
            self.calib_mgr = CalibrationManager(calibration_dir)
    
    def process_to_rgb(self, fits_files, auto_calibrate=True):
        if auto_calibrate and self.calib_mgr:
            # Auto-find and apply calibration
            calibrated_files = []
            for f in fits_files:
                ccd = CCDData.read(f, unit='adu')
                
                # Get matching calibration frames
                master_bias = self.calib_mgr.get_master_bias()
                master_dark = self.calib_mgr.get_master_dark(ccd.header['EXPTIME'])
                master_flat = self.calib_mgr.get_master_flat(ccd.header['FILTER'])
                
                # Apply calibration
                calibrated = ccdp.ccd_process(
                    ccd,
                    master_bias=master_bias,
                    dark_frame=master_dark,
                    master_flat=master_flat,
                    dark_scale=True
                )
                calibrated_files.append(calibrated)
            
            # Continue with normal pipeline...
```

**2.3 Add background subtraction (0.5 day)**
- Integrate photutils.Background2D
- Support multiple estimation methods
- Sky subtraction for each band

**2.4 Test with real ground-based data (0.5 day)**
- Download raw CCD data from amateur observatory
- Process from raw → calibrated → RGB
- Validate against manual reduction

### Phase 3: Multi-Mission WCS Support (3-4 days)
**Priority: HIGH - Currently broken for JWST, Chandra, Euclid**

**3.1 Enhance WCS validation (1 day)**
- Add WCS quality scoring
- Detect projection types, distortion models
- Validate WCS before reprojection
- Support gwcs (JWST) via duck-typing

**3.2 Mission-specific WCS handling (1.5 days)**
```python
class WCSHandler:
    def load_wcs(self, fits_file, mission=None):
        """Load WCS with mission-specific handling."""
        if mission == 'JWST' or self._is_jwst_file(fits_file):
            # Use stdatamodels for ASDF/gwcs
            from stdatamodels.jwst import datamodels
            with datamodels.open(fits_file) as model:
                return model.meta.wcs
        
        elif mission == 'HST' or self._is_hst_file(fits_file):
            # Use drizzlepac for full distortion correction
            from drizzlepac import stwcs
            return stwcs.wcsutil.HSTWCS(fits_file, ext=('SCI', 1))
        
        elif mission == 'Euclid' or self._is_euclid_file(fits_file):
            # Return dict of WCS per detector quadrant
            with fits.open(fits_file) as hdul:
                return {hdu.name: WCS(hdu.header) 
                        for hdu in hdul if 'SCI' in hdu.name}
        
        else:
            # Standard FITS WCS
            return WCS(fits.getheader(fits_file))
```

**3.3 Enhance Reprojector (1 day)**
- Implement `select_reference_frame()`
- Add artifact mitigation (NaN fill, edge crop)
- Support gwcs inputs
- Handle Euclid multi-detector case

**3.4 Test with multi-mission data (0.5 day)**
- JWST: NIRCam 3-filter
- HST: ACS with full distortion
- Euclid: VIS mosaic
- Validate alignment quality

### Phase 4: Advanced Palette Selection (2-3 days)
**Priority: MEDIUM - Current implementation only works for simple cases**

**4.1 Implement AdvancedChannelMapper (1.5 days)**
```python
class AdvancedChannelMapper:
    NARROWBAND_PALETTES = {
        'hubble': {'sii': 'red', 'ha': 'green', 'oiii': 'blue'},
        'hoo': {'ha': 'red', 'oiii': 'cyan'},
        'natural': {'ha': 'red', 'oiii': 'teal', 'sii': 'amber'}
    }
    
    def map_channels(self, bands, palette='chromatic', **kwargs):
        if palette == 'chromatic':
            return self._chromatic_mapping(bands)
        elif palette in self.NARROWBAND_PALETTES:
            return self._narrowband_mapping(bands, palette)
        elif isinstance(palette, dict):
            return self._custom_mapping(bands, palette)
        else:
            raise ValueError(f"Unknown palette: {palette}")
```

**4.2 Add multi-band selection (1 day)**
- Implement max-span strategy
- Implement PCA-based selection
- Add science-driven presets

**4.3 Test with narrowband data (0.5 day)**
- Ha/OIII/SII triplet
- Validate Hubble palette matches published images
- Test with 10-band dataset

### Phase 5: Testing Framework (4-5 days)
**Priority: CRITICAL - Cannot ship without tests**

**5.1 Unit tests (2.5 days)**
Target: 70% coverage of core functions
```
tests/
├── preprocessing/
│   ├── test_fits_loader.py       # FITS loading, MEF handling
│   ├── test_mission_adapters.py  # DQ flag interpretation
│   ├── test_quality_assessor.py  # SNR calculation
│   ├── test_calibrator.py        # Bias/dark/flat math
│   └── test_calibration_manager.py  # Auto-detection, ccdproc integration
├── processing/
│   ├── test_wcs_handler.py       # WCS validation, mission-specific
│   ├── test_reprojector.py       # Alignment correctness, artifact handling
│   └── test_normalization.py     # ImageNormalize integration
├── postprocessing/
│   ├── test_channel_mapper.py    # Chromatic ordering, palettes
│   ├── test_compositor.py        # Lupton RGB correctness
│   └── test_exporter.py          # File format handling
├── integration/
│   ├── test_raw_to_rgb.py        # End-to-end raw CCD data
│   ├── test_multi_mission.py    # JWST, HST, Euclid workflows
│   └── test_narrowband.py       # Ha/OIII/SII palette mapping
└── test_pipeline.py              # Pipeline API, manual workflows
```

**5.2 Validation datasets (1.5 days)**
Download and package test FITS files:
- **Ground-based raw:** Bias/dark/flat + science frames
- **JWST:** 3-filter NIRCam observation (F090W, F200W, F444W) with ASDF
- **HST:** 3-filter ACS observation (F435W, F606W, F814W) with drizzlepac reference files
- **Narrowband:** Ha, OIII, SII from ground-based telescope
- **Euclid:** VIS multi-detector mosaic (if public data available)
- Known-good outputs for regression testing

**5.3 CI/CD setup (0.5 day)**
- GitHub Actions for automated testing
- Test against numpy 1.24-2.0, astropy 5.3-7.1, Python 3.9-3.12
- Coverage reporting to codecov
- Automated warnings for low-quality components

**5.4 Integration test scenarios (0.5 day)**
```python
def test_end_to_end_raw_ccd():
    """Test complete pipeline from raw CCD to RGB."""
    pipeline = ProcessingPipeline(
        mode='scientific',
        calibration_dir='tests/data/raw_ccd/calibration/'
    )
    
    rgb = pipeline.process_to_rgb(
        fits_files=[
            'tests/data/raw_ccd/science_r.fits',
            'tests/data/raw_ccd/science_g.fits',
            'tests/data/raw_ccd/science_b.fits'
        ],
        auto_calibrate=True
    )
    
    # Validate output
    assert rgb.shape == (2048, 2048, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1
    assert not np.any(np.isnan(rgb))

def test_jwst_gwcs_workflow():
    """Test JWST data with gwcs."""
    pipeline = ProcessingPipeline(mode='sdss')
    
    # Should handle gwcs automatically
    rgb = pipeline.process_to_rgb(
        fits_files=[
            'tests/data/jwst/nircam_f090w_cal.fits',
            'tests/data/jwst/nircam_f200w_cal.fits',
            'tests/data/jwst/nircam_f444w_cal.fits'
        ]
    )
    
    # Validate gwcs was used
    assert rgb is not None

def test_narrowband_hubble_palette():
    """Test Hubble palette for narrowband."""
    pipeline = ProcessingPipeline(mode='manual')
    
    rgb = pipeline.process_to_rgb(
        fits_files=['ha.fits', 'oiii.fits', 'sii.fits'],
        palette='hubble',  # SII→R, Ha→G, OIII→B
        intervals='auto',
        stretches='asinh'
    )
    
    # Validate color assignment
    # (Implementation-specific checks)
```

### Phase 6: Quality & Validation (2-3 days)
**Priority: HIGH - Required for production use**

**6.1 Implement ValidationReport class (1 day)**
```python
class ValidationReport:
    """Comprehensive quality and processing metrics."""
    
    def generate_report(self, phase1_data, phase2_data, rgb_output):
        """Generate full processing report.
        
        Includes:
        - Per-band SNR, saturation %, noise characteristics
        - WCS quality scores (distortion model, astrometric accuracy)
        - Calibration applied (if any): bias/dark/flat statistics
        - Reprojection quality: footprint coverage, alignment error
        - Normalization/stretch parameters used
        - RGB color distribution histogram
        - Processing time per stage
        - Warnings/errors encountered
        - Output file metadata
        """
        report = {
            'bands': self._assess_bands(phase1_data),
            'wcs': self._assess_wcs(phase1_data),
            'calibration': self._assess_calibration(phase1_data),
            'alignment': self._assess_alignment(phase2_data),
            'processing': self._assess_processing(phase2_data),
            'rgb': self._assess_rgb(rgb_output),
            'warnings': self._collect_warnings()
        }
        return report
```

**6.2 Integrate quality checks (1 day)**
- Add validation hooks in `pipeline.py`:
  - Post-load: Check for valid data, WCS, metadata
  - Post-calibration: Verify no NaNs, reasonable ranges, proper units
  - Post-alignment: Check reprojection quality, footprint coverage
  - Post-stretch: Verify [0,1] range, detect clipping
  - Post-composite: RGB validity, color balance metrics

**6.3 Automated reporting (0.5 day)**
- HTML report generation with matplotlib plots
- Processing log export (JSON, CSV)
- Quality metrics dashboard

### Phase 7: Advanced Features (3-4 days)
**Priority: MEDIUM - Nice to have, not blocking**

**7.1 Cosmic ray rejection (1 day)**
Implement or integrate cosmic ray detection:
- Option A: Use `astroscrappy` library (recommended, LACosmicimplementation)
- Add to preprocessing pipeline before calibration
- Support both single-frame (for space data) and multi-frame (for ground data)

```python
class CosmicRayRejecter:
    """Detect and remove cosmic rays."""
    
    def reject_single(self, data, gain=1.0, readnoise=5.0):
        """Remove cosmic rays from single frame using LAcosmic."""
        import astroscrappy
        mask, cleaned = astroscrappy.detect_cosmics(
            data, 
            gain=gain,
            readnoise=readnoise,
            sigclip=4.5,
            sigfrac=0.3
        )
        return cleaned, mask
    
    def reject_multi(self, ccd_list):
        """Remove cosmic rays by combining multiple frames."""
        # Use ccdproc.Combiner with sigma clipping
        combiner = ccdp.Combiner(ccd_list)
        combiner.sigma_clipping(low_thresh=3, high_thresh=3)
        return combiner.median_combine()
```

**7.2 Advanced compositing (1 day)**
Implement additional RGB methods:
- `make_rgb()` with per-channel intervals/stretches
- HDR tone mapping for extreme dynamic range
- Multi-scale sharpening (wavelet-based)
- Luminance-chrominance separation

**7.3 Metadata preservation (1 day)**
Enhance FITS header and EXIF handling:
- Copy relevant FITS keywords to output images
- Embed processing history in PNG/TIFF metadata
- Generate sidecar files with full provenance
- Support FITS header inheritance through pipeline

**7.4 Performance optimization (0.5 day)**
- Memory-efficient processing for large files (>4GB)
- Lazy loading and chunked operations
- Parallel processing for multi-band operations
- Dask integration for distributed processing

### Phase 8: Documentation & Examples (3-4 days)
**Priority: MEDIUM - Required for user adoption**

**8.1 Tutorial notebooks (2 days)**
Create comprehensive Jupyter notebooks:
- Tutorial 1: Basic 3-band RGB composite (pre-calibrated JWST)
- Tutorial 2: Ground-based raw data processing (bias/dark/flat)
- Tutorial 3: Narrowband imaging with palette selection (Ha/OIII/SII)
- Tutorial 4: Multi-mission comparison (HST vs JWST alignment)
- Tutorial 5: Manual workflow for advanced users
- Tutorial 6: Batch processing multiple fields

**8.2 API documentation (1 day)**
- Sphinx documentation build
- Auto-generated API reference from docstrings
- Workflow diagrams (mermaid.js)
- Troubleshooting guide
- Performance tuning guide
- Mission-specific guides (JWST, HST, Euclid, etc.)

**8.3 Example gallery (0.5 day)**
- 15-20 example outputs with code
- Before/after comparisons
- Parameter effect demonstrations
- Real astronomical targets (M31, M51, Orion, etc.)

---

## Updated Dependencies & Requirements

### Core Dependencies (must have)
```
numpy >= 1.24.0
astropy >= 5.3.0
scipy >= 1.10.0
ccdproc >= 2.4.0        # NEW: CCD calibration
```

### Mission-Specific (optional, graceful degradation)
```
reproject >= 0.13.0       # Image alignment
drizzlepac >= 3.5.0       # HST distortion correction
stdatamodels >= 1.5.0     # JWST gwcs support
ciao-contrib >= 4.15      # Chandra aspect solutions (if processing X-ray)
```

### Processing & Enhancement (optional)
```
scikit-image >= 0.20.0    # CLAHE, morphology (⚠️ experimental)
matplotlib >= 3.7.0       # Plotting, colormaps
Pillow >= 9.5.0          # Image export
photutils >= 1.8.0       # Background estimation
astroscrappy >= 1.1.0    # Cosmic ray rejection
scikit-learn >= 1.2.0    # PCA band selection
```

### Testing Dependencies
```
pytest >= 7.4.0
pytest-cov >= 4.1.0
pytest-xdist >= 3.3.0    # Parallel test execution
```

---

## Key References (Expanded)

### Astropy Documentation
- **Image stretching:** https://docs.astropy.org/en/stable/visualization/normalization.html
- **RGB images:** https://docs.astropy.org/en/stable/visualization/rgb.html
- **make_lupton_rgb API:** https://docs.astropy.org/en/stable/api/astropy.visualization.make_lupton_rgb.html
- **ImageNormalize API:** https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.ImageNormalize.html
- **ccdproc documentation:** https://ccdproc.readthedocs.io/
- **reproject documentation:** https://reproject.readthedocs.io/

### Mission-Specific Documentation
- **JWST gwcs:** https://gwcs.readthedocs.io/
- **HST Data Handbook:** https://hst-docs.stsci.edu/
- **drizzlepac:** https://drizzlepac.readthedocs.io/
- **Euclid Data Processing:** https://euclid.esac.esa.int/
- **Chandra CIAO:** https://cxc.cfa.harvard.edu/ciao/

### Scientific Papers
- **Lupton et al. 2004 (SDSS RGB):** https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L
- **Calabretta & Greisen 2002 (FITS WCS):** https://ui.adsabs.harvard.edu/abs/2002A%26A...395.1077C
- **van Dokkum 2001 (LAcosmic):** https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V

### Project Documentation
- `/mnt/project/Guide_to_Professional_Astronomical_Image_Processing.md` - Comprehensive guide
- `/mnt/project/Guide_to_Astronomical_Image_Processing_and_Visualization_with_Astropy.md` - Astropy-focused
- `/mnt/project/Guide_to_Multi-Mission_Distortion_Correction_in_Python.md` - WCS and distortion
- `/mnt/project/Creating_color_RGB_images___Astropy_v7_1_1.md` - RGB compositing reference

---

## Action Items (Priority Order - REVISED)

### Week 1: Safety & Foundation
1. ✅ Audit current code against astropy best practices (DONE)
2. 🚨 **Tag low-quality components** (CLAHE, color balance) - CRITICAL
3. Implement manual workflow mode API design
4. Refactor `pipeline.py` Phase 2 to use `ImageNormalize`
5. Simplify `_compose_rgb()` to 3 explicit workflows

### Week 2: Preprocessing Integration
6. Implement CalibrationManager class with ccdproc
7. Add auto-detection of calibration files
8. Integrate ccdproc.ccd_process into pipeline
9. Test with real ground-based raw data
10. Add background subtraction (photutils.Background2D)

### Week 3: Multi-Mission Support
11. Enhance WCSHandler for mission-specific handling (JWST gwcs, HST drizzlepac, Euclid multi-detector)
12. Improve Reprojector (reference frame selection, artifact mitigation)
13. Implement AdvancedChannelMapper with narrowband palettes
14. Test with JWST, HST, narrowband data

### Week 4: Testing
15. Set up pytest framework
16. Write unit tests for core functions (target: 70% coverage)
17. Download validation datasets (JWST, HST, raw CCD, narrowband)
18. Create known-good test outputs
19. Set up CI/CD pipeline

### Week 5: Quality & Documentation
20. Implement ValidationReport class
21. Integrate quality checks into pipeline
22. Write 6 tutorial notebooks
23. Build Sphinx documentation
24. Create example gallery
25. Write troubleshooting guide

---

## Metrics & Success Criteria (Updated)

### Code Quality
- [ ] Test coverage ≥ 70% (currently 0%)
- [ ] All workflows validated against known outputs
- [ ] Low-quality components tagged with warnings
- [ ] No pylint warnings > 8/10 rating
- [ ] Type hints on public APIs

### Feature Completeness
- [ ] Manual workflow mode implemented and tested
- [ ] ccdproc integration complete (raw data → calibrated)
- [ ] Multi-mission WCS support (JWST gwcs, HST drizzlepac, Euclid)
- [ ] Narrowband palette support (Hubble, HOO, custom)
- [ ] Cosmic ray rejection integrated

### Performance
- [ ] Process 3-band RGB in < 30s (4K × 4K images, consumer laptop)
- [ ] Memory usage < 2× input file size
- [ ] Support files up to 10GB (chunked processing)

### Usability
- [ ] 3-line quick start example works
- [ ] Manual workflow example documented
- [ ] All 3 Lupton workflows documented with examples
- [ ] Error messages are actionable
- [ ] Processing progress visible

### Compatibility
- [ ] Raw CCD data (bias/dark/flat calibration)
- [ ] JWST (gwcs, ASDF extensions)
- [ ] HST (drizzlepac distortion correction)
- [ ] Chandra (X-ray event lists) - Phase 5
- [ ] Euclid (multi-detector mosaics)
- [ ] Ground-based (standard FITS WCS)
- [ ] Narrowband (Ha/OIII/SII with custom palettes)

---

## Critical Reminders

**Design Philosophy:**
- Follow astropy conventions, don't reinvent
- Explicit is better than implicit (workflow selection, palette choice)
- Fail fast with clear error messages
- Tag experimental/low-quality code prominently
- Optimize for the 80% use case, support edge cases

**What We Got Wrong:**
- Assumed pre-calibrated data (ignores raw CCD processing)
- Only tested with "perfect" FITS files (simple WCS, no distortion)
- Hardcoded workflows (no manual control)
- Missed narrowband/false-color use cases
- CLAHE/color-balance without validation

**What We're Fixing:**
- ccdproc integration for raw data
- Mission-specific WCS handling (gwcs, drizzlepac)
- Manual workflow mode
- Narrowband palette support
- Quality tagging for experimental features

**Non-Goals:**
- Not replacing SAOImage DS9 or other interactive viewers
- Not a photometry pipeline (use photutils)
- Not for time-series or spectroscopy
- Not for image deconvolution or PSF fitting
- Not attempting "auto-magic" processing (explicit > implicit)

---

**Last Updated:** 2025-10-26  
**Reviewed Against:** Astropy v7.1.1, ccdproc v2.4, reproject v0.13 documentation  
**Test Coverage:** 0% (target: 70%)  
**Production Ready:** Core yes (with warnings), preprocessing/testing/docs no
