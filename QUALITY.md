# Code Quality and Known Issues

**Last Updated:** 2025-10-26
**Status:** Critical quality issues documented - use with caution

---

## Executive Summary

This document tracks known quality issues, experimental features, and components requiring validation in the Astronomical FITS Processing Pipeline. Several components are marked as **LOW QUALITY** and should not be used in production without significant improvements.

---

## Critical Quality Issues

### 🚨 LOW QUALITY Components

These components have serious issues and should be avoided or used with extreme caution:

#### 1. CLAHE Implementation (`processing/enhancer.py::apply_clahe`)
**Quality:** LOW
**Status:** Experimental, Not Recommended

**Issues:**
- Uses `skimage.exposure.equalize_adapthist()` with default parameters not optimized for astronomy
- No star masking - treats point sources same as extended emission
- May severely over-enhance noise in low-SNR regions
- Kernel size auto-calculation is overly simplistic (max_dim // 8)
- No SNR-based adaptive clipping
- No validation on real astronomical data

**Impact:**
- Can destroy photometric information
- Creates artifacts around bright stars
- Over-enhances noise making images unusable

**Recommended Alternatives:**
- Use `unsharp_mask()` for controlled sharpening
- Apply CLAHE in external tools (PixInsight, GIMP, Photoshop) with proper masking
- Implement custom local contrast with star protection

**Required Fixes:**
1. Implement star detection and masking
2. Add SNR-based adaptive parameters
3. Validate against real astronomical data
4. Add multi-scale approach

---

#### 2. White Balance (`postprocessing/color_balancer.py::white_balance`)
**Quality:** LOW
**Status:** Naive Implementation

**Issues:**
- Simple RGB channel scaling without color space conversion
- No support for perceptual color spaces (LAB, LCh)
- Destroys photometric color information
- No validation against astronomical color indices
- No CIE standard illuminant support

**Impact:**
- Completely destroys photometric accuracy
- Results are not scientifically valid
- Cannot be used for any quantitative analysis

**Recommended Alternatives:**
- For scientific use: DO NOT USE
- For aesthetic use: Use external tools (Photoshop, GIMP, RawTherapee)
- Implement proper color space conversions if needed

**Required Fixes:**
1. Implement LAB color space conversion
2. Add CIE standard illuminant support
3. Preserve photometric ratios option
4. Add prominent warnings about data destruction

---

#### 3. Color Temperature Adjustment (`postprocessing/color_balancer.py::adjust_color_temperature`)
**Quality:** LOW
**Status:** Ad-hoc Implementation

**Issues:**
- Uses arbitrary RGB shifts (factor of 0.3) with no scientific basis
- Not based on blackbody radiation physics
- No CIE standard illuminants (D65, D50)
- Results are unpredictable and non-physical

**Impact:**
- Produces non-physical color shifts
- Cannot simulate real temperature changes
- Results vary wildly with input data

**Recommended Alternatives:**
- Implement proper blackbody color temperature model
- Use Planck's law for temperature-to-RGB conversion
- Use external color management tools

**Required Fixes:**
1. Implement proper color temperature based on Planck's law
2. Add CIE illuminant models
3. Support correlated color temperature (CCT)
4. Validate against standard color targets

---

### 🟡 MEDIUM QUALITY Components

These components work but have limitations:

#### 1. Unsharp Mask (`processing/enhancer.py::unsharp_mask`)
**Quality:** ACCEPTABLE
**Status:** Basic but Reliable

**Limitations:**
- Single-scale only (no multi-scale sharpening)
- No luminance masking
- May enhance noise in low-SNR regions
- Affects all pixels equally

**Suitable For:**
- Basic detail enhancement
- Most astronomical use cases
- Alternative to CLAHE

**Improvements Needed:**
- Multi-scale wavelet sharpening
- Luminance-based masking
- Noise-aware sharpening

---

#### 2. Saturation Adjustment (`postprocessing/color_balancer.py::adjust_saturation`)
**Quality:** MEDIUM
**Status:** Basic Implementation

**Limitations:**
- Clips values instead of preserving luminance
- Uses RGB space instead of HSL/HSV
- May lose detail in highly saturated areas

**Suitable For:**
- Minor saturation adjustments
- Non-critical color enhancement

**Improvements Needed:**
- HSL/HSV color space conversion
- Proper luminance preservation
- Gamut mapping for out-of-range colors

---

## Architectural Issues

### 1. Normalization/Stretch Architecture
**Problem:** Current two-step approach doesn't align with astropy's `ImageNormalize` pattern

**Current:**
```python
normalized = normalizer.normalize(data, interval=ZScaleInterval())
stretched = stretcher.stretch(normalized, method='asinh')
```

**Should Be:**
```python
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
processed = norm(data)
```

**Impact:** Inconsistent with astropy best practices, harder to maintain

---

### 2. Missing Manual Workflow Mode
**Problem:** No way to specify custom processing chains per band

**Required:** Users need ability to provide arrays of intervals/stretches for fine control

**Impact:** Cannot handle complex multi-band scenarios (narrowband, multi-mission)

---

### 3. Lupton RGB Workflow Confusion
**Problem:** Conflicting stretch object handling with auto-detection logic

**Impact:** Unpredictable results, difficult to debug, doesn't follow astropy patterns

---

## Validation Requirements

### Components Requiring Real-Data Validation

1. **CLAHE** - Must test on:
   - High dynamic range nebulae
   - Dense star fields
   - Low SNR regions
   - Mixed emission/stellar fields

2. **Color Balance** - Must test on:
   - Calibrated photometric data
   - Known color standards
   - Multi-filter datasets
   - Various telescope/camera combinations

3. **Reprojection** - Must test on:
   - JWST gwcs data
   - HST with full distortion
   - Multi-detector mosaics
   - High-distortion wide-field images

---

## Quality Assurance Checklist

### Before Using in Production

- [ ] All experimental decorators have been reviewed
- [ ] Runtime warnings are displayed to users
- [ ] Alternative methods are documented
- [ ] Validation on real astronomical data complete
- [ ] Test coverage > 70%
- [ ] Performance benchmarks established
- [ ] Memory usage profiled
- [ ] Error handling comprehensive

### For Each Release

- [ ] Review and update this document
- [ ] Check for new quality issues
- [ ] Update decorator warnings
- [ ] Validate against latest astropy versions
- [ ] Review user feedback on experimental features

---

## Recommended Usage Guidelines

### Safe to Use
✅ `fits_loader` - Production ready
✅ `mission_adapters` - Well tested
✅ `quality_assessor` - Reliable metrics
✅ `calibrator` - Standard operations
✅ `reprojector` - Basic functionality works
✅ `normalizer` - When used with ImageNormalize
✅ `stretcher` - When used with ImageNormalize
✅ `channel_mapper` - Chromatic ordering only
✅ `compositor` - Lupton and simple RGB
✅ `exporter` - All formats supported
✅ `preview` - Quick look generation

### Use with Caution
⚠️ `unsharp_mask` - Basic but acceptable
⚠️ `adjust_saturation` - Limited implementation
⚠️ `enhance_stars` - Needs validation
⚠️ `local_contrast_enhancement` - Test first

### Do Not Use (Production)
❌ `apply_clahe` - Severe quality issues
❌ `white_balance` - Destroys photometric data
❌ `adjust_color_temperature` - Non-physical
❌ `auto_balance` - Naive implementation

---

## Improvement Roadmap

### Phase 1: Critical Fixes (Immediate)
- [x] Add warning decorators to low-quality components
- [x] Document known issues (this document)
- [x] Fix pytest collection errors and install pytest-cov
- [x] Update tests/README.md with accurate documentation
- [ ] Disable dangerous defaults in pipeline
- [ ] Add runtime warnings for experimental features

### Phase 2: Core Refactoring (1 week)
- [ ] Refactor to use ImageNormalize properly
- [ ] Implement manual workflow mode
- [ ] Simplify Lupton RGB workflows
- [ ] Add calibration manager

### Phase 3: Quality Improvements (2 weeks)
- [ ] Improve or remove CLAHE
- [ ] Implement proper color space conversions
- [ ] Add multi-scale sharpening
- [ ] Implement proper color temperature

### Phase 4: Validation (1 week)
- [ ] Test with real astronomical data
- [ ] Validate against published results
- [ ] Performance benchmarking
- [ ] Memory profiling

---

## Contact

For questions about quality issues or to report new problems:
- GitHub Issues: [Create Issue](https://github.com/[repo]/issues)
- Documentation: See `/docs/` directory
- Test Data: Available in `/tests/data/`

---

**Remember:** This is a scientific pipeline. When in doubt, preserve the data integrity over aesthetic improvements. Most "enhancement" operations destroy photometric accuracy and should only be used for visualization, never for analysis.