# VisionProject - Complete Workflow Guide

**Author:** Claude + User
**Date:** 2025-10-26
**Astropy Version:** 7.1.1

---

## Table of Contents

1. [Overview](#overview)
2. [Astropy Visualization Components](#astropy-visualization-components)
3. [Seven Standard Workflows](#seven-standard-workflows)
4. [Component Reference](#component-reference)
5. [Workflow Decision Tree](#workflow-decision-tree)
6. [Implementation Examples](#implementation-examples)

---

## Overview

This guide documents the complete astronomical image processing pipeline using astropy.visualization. The pipeline has three phases:

- **Phase 1:** Calibration (bias, dark, flat, background subtraction)
- **Phase 2:** Normalization + Stretching (interval selection + tone-mapping)
- **Phase 3:** RGB Composition (combining multiple wavelengths)

**Key Principle:** Avoid double-processing! If Phase 2 applies stretching, Phase 3 should NOT apply additional tone-mapping.

---

## Astropy Visualization Components

### **INTERVAL CLASSES (Normalization: arbitrary units → [0, 1])**

Intervals define the **black point (vmin) and white point (vmax)** for normalization:
```
normalized = (data - vmin) / (vmax - vmin)
```

| Class | Parameters | Use Case | Scientific | Aesthetic |
|-------|-----------|----------|------------|-----------|
| `MinMaxInterval()` | None | Use literal min/max | ❌ Outlier-sensitive | ❌ Poor contrast |
| `ManualInterval(vmin, vmax)` | vmin, vmax | User-specified bounds | ✅ Calibrated standards | ✅ Precise control |
| `PercentileInterval(percentile, n_samples)` | percentile (e.g., 95), n_samples | Symmetric clipping | ✅ Robust to outliers | ✅ Adjustable contrast |
| `AsymmetricPercentileInterval(lower, upper, n_samples)` | lower_percentile, upper_percentile | Different clipping at ends | ✅ Asymmetric data | ✅ Nebulae, galaxies |
| `ZScaleInterval(contrast, max_reject, krej, ...)` | contrast=0.25, max_reject=0.5, krej=2.5 | IRAF adaptive algorithm | ✅✅✅ **GOLD STANDARD** | ✅ Good default |

**Best Practices:**
- **Science:** Use `ZScaleInterval` (robust, reproducible)
- **Aesthetic:** Use `AsymmetricPercentileInterval(2, 99.5)` for drama
- **Interactive:** Use `PercentileInterval` with adjustable percentile
- **Calibrated Data:** Use `ManualInterval` when you know exact values

---

### **STRETCH CLASSES (Transform: [0, 1] → [0, 1])**

Stretches apply **non-linear tone-mapping** to enhance faint features:

#### **Basic Stretches**

| Class | Parameters | Formula | Use Case | Scientific | Aesthetic |
|-------|-----------|---------|----------|------------|-----------|
| `LinearStretch(slope, intercept)` | slope=1.0, intercept=0.0 | y = slope×x + intercept | Identity or custom linear | ✅ When data pre-stretched | ✅ Fine-tuning |
| `SqrtStretch()` | None | y = √x | Mild compression | ✅ Most images | ✅ Good default |
| `SquaredStretch()` | None | y = x² | Inverse of sqrt | ❌ Suppresses faint | ✅ Enhance bright |

#### **Logarithmic Family**

| Class | Parameters | Formula | Use Case | Scientific | Aesthetic |
|-------|-----------|---------|----------|------------|-----------|
| `LogStretch(a)` | a=1000 | y = log(ax+1)/log(a+1) | Aggressive bright compression | ✅ Point sources | ✅ Stars, quasars |
| `AsinhStretch(a)` | a=0.1 | y = asinh(x/a)/asinh(1/a) | Linear→faint, log→bright | ✅✅✅ **ASTRONOMICAL STANDARD** | ✅✅✅ Wide dynamic range |
| `SinhStretch(a)` | a=0.33 | y = sinh(ax)/sinh(a) | Inverse of asinh | ❌ Rarely used | ❌ Enhances bright only |

#### **Power Law**

| Class | Parameters | Formula | Use Case | Scientific | Aesthetic |
|-------|-----------|---------|----------|------------|-----------|
| `PowerStretch(a)` | a (power) | y = x^a | Gamma correction | ✅ a>1 darken, a<1 brighten | ✅ Adjustable tone curve |
| `PowerDistStretch(a)` | a=1000 | Alternative power formula | Niche cases | ✅ Experimental | ✅ Alternative to Power |

#### **Advanced Stretches**

| Class | Parameters | Formula | Use Case | Scientific | Aesthetic |
|-------|-----------|---------|----------|------------|-----------|
| `ContrastBiasStretch(contrast, bias)` | contrast (multiplier), bias (center) | y = (x-bias)×contrast + 0.5 | Interactive tuning | ✅ Refinement | ✅✅✅ Like Photoshop Levels |
| `HistEqStretch(data, values)` | data (for histogram), values=1000 | Histogram equalization | Maximize local contrast | ⚠️ Can introduce artifacts | ✅✅ Dramatic results |
| `CompositeStretch(stretch_1, stretch_2)` | Two BaseStretch objects | Applies stretch_1 then stretch_2 | Custom pipelines | ✅ Advanced users | ✅ Hybrid approaches |

#### **Lupton-Specific (for RGB Composition)**

| Class | Parameters | Formula | Use Case | Scientific | Aesthetic |
|-------|-----------|---------|----------|------------|-----------|
| `LuptonAsinhStretch(stretch, Q)` | stretch=5, Q=8 | Modified asinh for RGB | Manual control over Lupton | ✅ Precise tuning | ✅ Match specific look |
| `LuptonAsinhZscaleStretch(image, Q, pedestal)` | image (to analyze), Q=8 | Auto-calculates stretch via ZScale | Automatic optimization | ✅✅✅ **SDSS METHOD** | ✅ No manual tuning |

**Best Practices:**
- **Science:** `AsinhStretch(a=0.1)` is the astronomical standard
- **Aesthetic:** `HistEqStretch` for maximum impact (watch for artifacts)
- **Interactive:** `ContrastBiasStretch` for fine-tuning
- **Pre-stretched data in Phase 3:** Use `LinearStretch()` (identity) to avoid double-processing

---

### **COMPOSITOR METHODS (RGB Creation)**

| Method | Algorithm | Input Expectation | Output | Best For |
|--------|-----------|-------------------|--------|----------|
| `make_lupton_rgb(r, g, b, stretch, Q)` | Lupton et al. 2004 - preserves color in bright regions | Normalized [0, 1] (NOT pre-stretched) | uint8 [0-255] (default) | ✅ SDSS-style, auto-stretch |
| `make_lupton_rgb(r, g, b, stretch_object=LinearStretch())` | Lupton with custom stretch | Pre-stretched [0, 1] | float64/uint8 (configurable) | ✅ Color preservation + manual stretch control |
| `make_rgb(r, g, b, interval, stretch)` | Independent channel processing | Any range (interval applied) | uint8 [0-255] (default) | ✅ Per-channel control |
| `create_simple_rgb(r, g, b)` | Simple channel stacking | Pre-stretched [0, 1] | float [0, 1] | ✅ Pre-stretched data, no additional processing |
| `create_narrowband_composite(ha, oiii, sii)` | False-color mapping (Hubble Palette) | Pre-processed [0, 1] | float [0, 1] | ✅ Emission nebulae |

**Key Differences:**
- **Lupton:** Interconnected scaling - preserves color ratios in bright regions
- **Simple/make_rgb:** Independent channels - more prone to color saturation
- **Narrowband:** False-color mapping (not true RGB)

---

## Seven Standard Workflows

### **WORKFLOW 1: SCIENTIFIC STANDARD**

**Goal:** Preserve photometry, minimize artifacts, reproducible

**When:** Publication-quality science images, photometric analysis

```
Phase 1: Calibration (bias, dark, flat, background subtraction)
         ↓
Phase 2: ZScaleInterval + AsinhStretch(a=0.1)
         Output: *_stretched.fits [0, 1]
         Save: interval_object, stretch_object (pickled)
         ↓
Phase 3: Compositor.create_lupton_rgb(
             r, g, b,
             stretch_object=LinearStretch(),  # No additional stretching
             output_dtype=np.float64
         )
         OR
         Compositor.create_simple_rgb(r, g, b)  # Pure stacking
```

**Components Used:**
- **Interval:** `ZScaleInterval()` - robust, outlier-resistant
- **Stretch:** `AsinhStretch(a=0.1)` - astronomical standard
- **Compositor:** `create_lupton_rgb` with `LinearStretch()` OR `create_simple_rgb`

**Why:**
- ZScale: IRAF standard, sigma-clips outliers
- Asinh: Preserves faint features while compressing bright
- LinearStretch in Lupton: Get color preservation WITHOUT double-stretching
- Simple RGB: If Lupton's color algorithm not needed

---

### **WORKFLOW 2: SDSS/LUPTON METHOD**

**Goal:** Follow published SDSS algorithm, automatic parameter tuning

**When:** Want Lupton's color preservation with automatic optimization

```
Phase 1: Calibration
         ↓
Phase 2: ZScaleInterval ONLY (no stretching!)
         Output: *_normalized.fits [0, 1]
         Save: interval_object
         ↓
Phase 3: Compositor.create_lupton_rgb(
             r, g, b,
             stretch_object=LuptonAsinhZscaleStretch(r, Q=8),
             output_dtype=np.float64
         )
```

**Components Used:**
- **Interval:** `ZScaleInterval()` - normalization only
- **Stretch:** NONE in Phase 2 (Lupton handles it)
- **Compositor:** `create_lupton_rgb` with `LuptonAsinhZscaleStretch`

**Why:**
- Single tone-mapping step (no double-processing)
- Lupton analyzes data and auto-calculates optimal stretch parameter
- Follows Lupton et al. 2004 methodology exactly

**Critical:** Phase 2 must NOT apply stretching!

---

### **WORKFLOW 3: AESTHETIC/PUBLICATION**

**Goal:** Maximum visual impact, dramatic presentation

**When:** Public outreach, press releases, desktop wallpapers

```
Phase 1: Calibration
         ↓
Phase 2: AsymmetricPercentileInterval(2, 99.5) + HistEqStretch(data)
         Output: *_stretched.fits [0, 1]
         Save: interval_object, stretch_object
         ↓
Phase 3: Compositor.create_lupton_rgb(
             r, g, b,
             stretch_object=LinearStretch(),
             output_dtype=np.float64
         )
         +
         ColorBalancer.adjust(rgb, saturation=1.2, white_point='auto')
```

**Components Used:**
- **Interval:** `AsymmetricPercentileInterval(2, 99.5)` - aggressive clipping
- **Stretch:** `HistEqStretch(data)` - maximize contrast
- **Compositor:** `create_lupton_rgb` with `LinearStretch()`
- **Post-processing:** `ColorBalancer` for saturation boost

**Why:**
- Tight percentiles: Clip extremes for dramatic range
- HistEq: Flatten histogram, maximize local contrast
- Lupton: Preserve color in bright regions (prevents oversaturation)
- Color balance: Final aesthetic refinement

**Warning:** Can introduce artifacts, not suitable for photometry

---

### **WORKFLOW 4: NARROWBAND SCIENCE**

**Goal:** False-color composition (e.g., Hubble Palette: SHO)

**When:** Emission nebulae with H-alpha, OIII, SII filters

```
Phase 1: Calibration
         ↓
Phase 2: Per-channel optimization (different intervals/stretches)
         H-alpha (→Red):   ZScaleInterval + AsinhStretch(a=0.05)  [often brightest]
         OIII (→Green):    ZScaleInterval + AsinhStretch(a=0.1)
         SII (→Blue):      PercentileInterval(99) + AsinhStretch(a=0.15)  [often faintest]
         Output: 3× *_stretched.fits [0, 1]
         ↓
Phase 3: Compositor.create_narrowband_composite(
             ha=ha_stretched,
             oiii=oiii_stretched,
             sii=sii_stretched,
             method='simple'  # Already stretched, no additional processing
         )
```

**Components Used:**
- **Interval:** Per-channel (ZScale or Percentile depending on brightness)
- **Stretch:** Per-channel `AsinhStretch` with different parameters
- **Compositor:** `create_narrowband_composite` with `method='simple'`

**Why:**
- Different emission lines have vastly different intensities
- Per-channel optimization essential for balanced composition
- Simple composition (data already tone-mapped)
- False-color mapping reveals physical processes

**Note:** This is NOT true-color imaging!

---

### **WORKFLOW 5: QUICK PREVIEW**

**Goal:** Fast iteration, interactive parameter exploration

**When:** Testing parameters, exploring data, quick visualizations

```
Phase 1: Calibration (or even skip - use raw data)
         ↓
Phase 3: make_rgb(r, g, b,
             interval=PercentileInterval(95),
             stretch=AsinhStretch(a=0.1)
         )
         # Skips Phase 2 entirely!
```

**Components Used:**
- **All-in-one:** `make_rgb` combines interval + stretch + composition

**Why:**
- No intermediate files
- Fast iteration over parameters
- Good for exploring what interval/stretch works best
- Single function call

**Limitation:** Independent channel processing (no Lupton color preservation)

---

### **WORKFLOW 6: MANUAL/CUSTOM**

**Goal:** Full control over every parameter

**When:** Specific artistic vision, unusual data, research experiments

```
Phase 1: Calibration
         ↓
Phase 2: Custom interval + Custom stretch (user's choice)
         Example 1: PercentileInterval(98) + CompositeStretch(
                        SqrtStretch() + AsinhStretch(a=0.05)
                    )
         Example 2: ManualInterval(100, 5000) + ContrastBiasStretch(
                        contrast=1.5, bias=0.3
                    )
         Output: *_stretched.fits [0, 1]
         ↓
Phase 3: Any compositor method
         User decides: Lupton vs Simple vs Custom
```

**Components Used:**
- **Any interval**
- **Any stretch** (including `CompositeStretch` for chaining)
- **Any compositor**

**Why:**
- Maximum flexibility
- Experimental pipelines
- Artistic control
- Research-specific needs

**Examples:**
- Combine sqrt + asinh for hybrid tone curve
- Use ContrastBiasStretch for interactive refinement
- Manual intervals for calibrated data

---

### **WORKFLOW 7: QUICK LUPTON (Phase 1 → Phase 3)**

**Goal:** Skip Phase 2, let Lupton do everything

**When:** Trust Lupton's auto-optimization, want fastest path to RGB

```
Phase 1: Calibration
         ↓
Phase 3: make_lupton_rgb(r, g, b,
             interval=ZScaleInterval(),  # Optional pre-normalization
             stretch_object=LuptonAsinhZscaleStretch([r, g, b], Q=8),
             output_dtype=np.float64
         )
         # Lupton handles normalization + stretching + composition
```

**OR even simpler:**

```
Phase 1: Calibration
         ↓
Phase 3: make_lupton_rgb(r_raw, g_raw, b_raw,
             stretch=0.5,  # Auto-defaults
             Q=8,
             output_dtype=np.float64
         )
         # Lupton auto-normalizes + stretches + composes
```

**Components Used:**
- **Built-in:** `make_lupton_rgb` with default parameters

**Why:**
- Fastest workflow
- Lupton's algorithm handles everything
- Good default results
- No intermediate files

**Limitation:** Less control over individual channels

---

## Component Reference

### **Complete Interval List**

```python
from astropy.visualization import (
    MinMaxInterval,           # Simple min/max
    ManualInterval,           # User-specified bounds
    PercentileInterval,       # Symmetric clipping
    AsymmetricPercentileInterval,  # Asymmetric clipping
    ZScaleInterval           # IRAF adaptive (GOLD STANDARD)
)

# Examples:
interval1 = ZScaleInterval(contrast=0.25, max_reject=0.5, krej=2.5)
interval2 = AsymmetricPercentileInterval(lower_percentile=1, upper_percentile=99)
interval3 = ManualInterval(vmin=100, vmax=5000)
```

### **Complete Stretch List**

```python
from astropy.visualization import (
    # Basic
    LinearStretch,            # Identity or custom linear
    SqrtStretch,              # Mild compression
    SquaredStretch,           # Inverse of sqrt

    # Logarithmic family
    LogStretch,               # Aggressive bright compression
    AsinhStretch,             # ASTRONOMICAL STANDARD
    SinhStretch,              # Inverse of asinh

    # Power law
    PowerStretch,             # Gamma correction
    PowerDistStretch,         # Alternative power

    # Advanced
    ContrastBiasStretch,      # Interactive tuning
    HistEqStretch,            # Histogram equalization
    CompositeStretch,         # Chain multiple stretches

    # Lupton-specific
    LuptonAsinhStretch,       # Manual Lupton params
    LuptonAsinhZscaleStretch  # Auto-calculate stretch
)

# Examples:
stretch1 = AsinhStretch(a=0.1)  # Standard astronomical
stretch2 = HistEqStretch(data, values=1000)  # Dramatic
stretch3 = CompositeStretch(SqrtStretch(), AsinhStretch(a=0.05))  # Hybrid
stretch4 = LuptonAsinhZscaleStretch(image_r, Q=8)  # Auto-optimized
```

### **Complete Compositor List**

```python
from astropy.visualization import make_lupton_rgb, make_rgb
from astro_vision_composer.postprocessing import Compositor

compositor = Compositor()

# Method 1: Lupton RGB (color preservation)
rgb1 = make_lupton_rgb(r, g, b, stretch=0.5, Q=8)
rgb2 = compositor.create_lupton_rgb(r, g, b, stretch_object=LinearStretch())

# Method 2: Simple RGB (independent channels)
rgb3 = compositor.create_simple_rgb(r, g, b)

# Method 3: make_rgb (flexible, independent)
rgb4 = make_rgb(r, g, b, interval=PercentileInterval(95), stretch=AsinhStretch())

# Method 4: Narrowband (false-color)
rgb5 = compositor.create_narrowband_composite(ha, oiii, sii, method='lupton')
```

---

## Workflow Decision Tree

```
START: What is your goal?
│
├─ SCIENTIFIC PUBLICATION?
│  └─ Use: Workflow 1 (Scientific Standard)
│     ZScale + Asinh + Lupton(LinearStretch) or Simple RGB
│
├─ FOLLOW SDSS METHOD?
│  └─ Use: Workflow 2 (SDSS/Lupton)
│     ZScale only + Lupton(LuptonAsinhZscaleStretch)
│
├─ MAXIMUM VISUAL IMPACT?
│  └─ Use: Workflow 3 (Aesthetic)
│     AsymmetricPercentile + HistEq + Lupton + ColorBalance
│
├─ NARROWBAND EMISSION LINES?
│  └─ Use: Workflow 4 (Narrowband)
│     Per-channel ZScale/Percentile + Per-channel Asinh + Narrowband compositor
│
├─ QUICK PREVIEW/TESTING?
│  └─ Use: Workflow 5 (Quick Preview)
│     make_rgb(interval, stretch) - all in one
│
├─ FASTEST PATH TO RGB?
│  └─ Use: Workflow 7 (Quick Lupton)
│     make_lupton_rgb with defaults - Phase 1 → Phase 3
│
└─ NEED CUSTOM CONTROL?
   └─ Use: Workflow 6 (Manual)
      Choose your own interval, stretch, compositor
```

---

## Implementation Examples

### **Example 1: Scientific Standard (Workflow 1)**

```python
from astropy.visualization import ZScaleInterval, AsinhStretch, LinearStretch
from astro_vision_composer.processing import Normalizer, Stretcher
from astro_vision_composer.postprocessing import Compositor

# Phase 2: Normalize + Stretch
normalizer = Normalizer()
stretcher = Stretcher()

normalized = normalizer.normalize(data, method='zscale')
stretched = stretcher.stretch(normalized, method='asinh', a=0.1)

# Save stretch object for Phase 3
stretch_obj = stretcher.get_stretch_object()

# Phase 3: Lupton with LinearStretch (no double-processing)
compositor = Compositor()
rgb = compositor.create_lupton_rgb(
    r=r_stretched,
    g=g_stretched,
    b=b_stretched,
    stretch_object=LinearStretch(),  # Identity - no additional stretch
    output_dtype=np.float64
)
```

### **Example 2: SDSS Method (Workflow 2)**

```python
from astropy.visualization import ZScaleInterval, LuptonAsinhZscaleStretch
from astro_vision_composer.processing import Normalizer

# Phase 2: Normalize ONLY (no stretching)
normalizer = Normalizer()
r_norm = normalizer.normalize(r_data, method='zscale')
g_norm = normalizer.normalize(g_data, method='zscale')
b_norm = normalizer.normalize(b_data, method='zscale')

# Phase 3: Lupton with auto-calculated stretch
stretch_obj = LuptonAsinhZscaleStretch(r_norm, Q=8)
rgb = compositor.create_lupton_rgb(
    r=r_norm,
    g=g_norm,
    b=b_norm,
    stretch_object=stretch_obj,
    output_dtype=np.float64
)
```

### **Example 3: Quick Lupton (Workflow 7)**

```python
from astropy.visualization import make_lupton_rgb

# Skip Phase 2 entirely - Lupton does everything
rgb = make_lupton_rgb(
    r_calibrated, g_calibrated, b_calibrated,
    stretch=0.5,
    Q=8,
    output_dtype=np.float64
)
```

### **Example 4: Manual/Custom (Workflow 6)**

```python
from astropy.visualization import (
    AsymmetricPercentileInterval, CompositeStretch,
    SqrtStretch, AsinhStretch, ContrastBiasStretch
)

# Phase 2: Custom interval + Composite stretch
interval = AsymmetricPercentileInterval(2, 99.5)
vmin, vmax = interval.get_limits(data)
normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)

# Chain sqrt then asinh
hybrid_stretch = CompositeStretch(
    SqrtStretch(),
    AsinhStretch(a=0.05)
)
stretched = hybrid_stretch(normalized)

# Fine-tune with contrast/bias
final_stretch = ContrastBiasStretch(contrast=1.2, bias=0.4)
final = final_stretch(stretched)

# Phase 3: Simple composition (already heavily processed)
rgb = compositor.create_simple_rgb(r_final, g_final, b_final)
```

---

## Validation Checklist

After processing, validate your results:

### **Phase 2 Validation:**
- [ ] Normalized data in [0, 1] range
- [ ] Stretched data in [0, 1] range
- [ ] Stretch is non-linear (for asinh/log/etc)
- [ ] No NaN or Inf values
- [ ] Histogram shows expected distribution

### **Phase 3 Validation:**
- [ ] RGB output dtype is consistent (float64 or uint8)
- [ ] RGB values in expected range ([0, 1] for float, [0, 255] for uint8)
- [ ] No double-stretching artifacts (check if using Lupton after Phase 2 stretch)
- [ ] Color preservation in bright regions (if using Lupton)
- [ ] Channels not all saturated (colorful, not white/gray)

### **Workflow Validation:**
- [ ] If Phase 2 stretched, Phase 3 uses LinearStretch or Simple RGB
- [ ] If Phase 2 only normalized, Phase 3 can use LuptonAsinhZscaleStretch
- [ ] Stretch objects saved and reloadable (pickle test)
- [ ] Processing history recorded for reproducibility

---

## Common Pitfalls

### **Pitfall 1: Double-Stretching**
**Problem:** Applying asinh in Phase 2, then Lupton's asinh in Phase 3
**Solution:** Use `LinearStretch()` in Phase 3 if data already stretched

### **Pitfall 2: Wrong Data Range**
**Problem:** Feeding uint8 [0-255] to functions expecting float [0-1]
**Solution:** Use `_normalize_to_float()` helper in Exporter

### **Pitfall 3: Saturated Colors**
**Problem:** All RGB channels hit 1.0 (white/gray instead of colorful)
**Solution:** Use gentler stretch parameters or Lupton's color preservation

### **Pitfall 4: Lost Faint Features**
**Problem:** Tight percentile intervals clip faint nebulosity
**Solution:** Use ZScaleInterval or wider percentiles (90-95%)

### **Pitfall 5: Artifacts from HistEq**
**Problem:** Blocky artifacts or unnatural gradients
**Solution:** Use HistEq cautiously, or combine with smoother stretch

---

## References

- **Lupton et al. (2004):** "Preparing Red-Green-Blue Images from CCD Data" (PASP 116:133)
- **SDSS SkyServer:** Color image generation methodology
- **Astropy Documentation:** https://docs.astropy.org/en/stable/visualization/
- **IRAF ZScale:** Nurit et al. 1996 (ZScale algorithm paper)

---

**END OF WORKFLOW GUIDE**
