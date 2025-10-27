# VisionProject - AI-Guided Astronomical Image Processing

**Project Status:** DUAL TRACK DEVELOPMENT
**Date:** 2025-10-25
**Track 1:** AI Vision (Paused - awaiting better model support)
**Track 2:** FITS Processing Suite (Active - **PHASE 3 COMPLETE + WORKING DEMOS**)
**Production Readiness:** 60% (Core pipeline functional, demo scripts working, needs ProcessingPipeline API & tests)

---

## Executive Summary

**Goal:** Use local vision AI to analyze astronomical FITS images and provide structured feedback for processing decisions (quality scores, RGB mapping, stretch parameters).

**Current Reality:** We can get JSON responses, but the AI often copies example patterns rather than independently analyzing images. This is a **learning project** about prompt engineering for vision models, not yet a production tool.

---

## What We've Learned

### ✅ Confirmed Working

1. **Basic AI Vision:** Model CAN see and distinguish different images
   - Test: Black/white/red squares → Correctly identified all colors
   - Conclusion: The model has functional vision capabilities

2. **JSON Format Training:** Can teach JSON response format via few-shot learning
   - 3 examples with explicit schema → Query returns valid JSON
   - Server auto-extracts JSON from responses

3. **Context Building Strategy:** Server supports accumulating context across images
   - `strategy: "context_building"` passes previous examples to model
   - Each image sees what came before

### ❌ Current Limitations

1. **Template Copying vs. Analysis**
   - AI learns the JSON structure but often copies content from examples
   - Example: Showed M106 (galaxy), M16 (nebula), M17 (nebula), then query
   - Result: M17 and query both got M106's assessment ("spiral arms, dust lanes")
   - **Not analyzing independently**

2. **Prompt Engineering Challenge**
   - Initial attempt: Gave example values in schema → AI copied exactly
   - Second attempt: Few-shot with real images → AI copies last similar example
   - **Need better prompting strategy**

3. **No True Multi-Turn Support**
   - Server doesn't accept `{"role": "assistant", "text": "..."}` messages
   - Can't pre-fill expected responses like true few-shot examples show
   - Limited to user messages with images

---

## Project Evolution

### Phase 1: False Start (2025-10-23)
**Claimed:** "Baseline established, 100% JSON parsing, 8/10 quality"
**Reality:** AI was copying example template, all 5 bands got identical scores
**Lesson:** JSON format ≠ meaningful analysis

### Phase 2: Understanding the Problem (2025-10-24)
**Discovery:** Structured prompts constrained AI too much
**Test:** Simple colored squares with few-shot → **SUCCESS** (correctly identified colors)
**Insight:** Few-shot CAN work, but needs proper examples

### Phase 3: Real Astronomical Training (Current)
**Approach:** 3 real NOIRLab images (M106, Eagle Nebula, M17) as training
**Result:** JSON format learned, but content still copies from examples
**Status:** Need to refine prompting strategy

---

## Technical Architecture

### Components

**1. Server** (Remote, via SSH tunnel)
- `http://localhost:5000` (tunneled to 192.168.50.194)
- Model: `unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit`
- Supports: `sequential` and `context_building` strategies
- Auto-extracts JSON from responses

**2. FITS Processing** (`src/astro_vision_composer/`)
- `fits_processor.py` - Load/normalize FITS data
- `composite_generator.py` - Multi-band composition with enhancement
- `vision_compositor.py` - AI integration (needs work)

**3. Test Scripts** (Root directory)
- `test_ai_vision.py` - Basic vision test (black/white/red)
- `test_json_proper_fewshot.py` - JSON training with colored squares
- `test_fewshot_astronomy.py` - **Current best attempt** - Real astronomical examples
- `test_5band_v2.py` - 5-band analysis (not yet working properly)

---

## Current Best Practice

### What Works for JSON Format

```python
# 3 examples with EXPLICIT schema
messages = [
    {
        "image_base64": example1_b64,
        "text": 'Analyze. Respond ONLY with JSON: {"color": "<color>", "brightness": "<level>"}'
    },
    {
        "image_base64": example2_b64,
        "text": 'Analyze. Respond ONLY with JSON: {"color": "<color>", "brightness": "<level>"}'
    },
    {
        "image_base64": example3_b64,
        "text": 'Analyze. Respond ONLY with JSON: {"color": "<color>", "brightness": "<level>"}'
    },
    # Query: Still explicit about JSON
    {
        "image_base64": query_b64,
        "text": 'Analyze in JSON with same fields'
    }
]

response = requests.post(
    "http://localhost:5000/generate",
    json={"messages": messages, "strategy": "context_building", "max_tokens": 200}
)
```

**Key Insights:**
- **Always** ask for JSON, even in query
- 3+ examples to establish pattern
- Use `context_building` strategy
- Keep prompts consistent across examples

### What Doesn't Work

1. **Vague query prompts** - "analyze this" without "respond in JSON"
2. **Example values in schema** - `"quality": 7` → AI copies "7" for everything
3. **Too few examples** - 1-2 examples = coincidence, not pattern
4. **Mixed prompt styles** - Inconsistent format confuses learning

---

## Known Issues

### Issue 1: Template Copying
**Symptom:** Different images get same assessment
**Cause:** AI pattern-matches to most similar training example
**Status:** Investigating better prompting strategies

### Issue 2: Server Limitations
**Symptom:** Can't pre-fill assistant responses
**Impact:** Limited few-shot capabilities vs. true multi-turn dialogue
**Workaround:** Describe expected responses in user prompt text

### Issue 3: No Variation Detection
**Symptom:** Can't easily tell if AI is analyzing or copying
**Need:** Validation tests with known-different images

---

## Realistic Goals

### Short-term (This Week)
- [ ] Find prompting strategy that prevents template copying
- [ ] Test with known-different astronomical features (galaxy vs. nebula vs. stars)
- [ ] Validate AI can distinguish these independently

### Medium-term (This Month)
- [ ] Reliable JSON responses for single-band analysis
- [ ] Quality scoring that reflects actual image characteristics
- [ ] RGB mapping recommendation based on wavelength

### Long-term (Aspirational)
- [ ] Batch processing of multi-band FITS data
- [ ] Parameter suggestions for stretch/enhancement
- [ ] Automated composite generation pipeline

---

## FITS Processing Suite (New Architecture)

**Status:** Phase 1 Implementation Started (2025-10-24)
**Foundation:** Based on comprehensive FITS.md guide covering JWST, HST, Chandra, Euclid, PanSTARRS

### Architecture Overview

A professional-grade FITS processing toolkit **independent of AI vision models**, designed to handle end-to-end astronomical image processing from raw FITS files to publication-quality composites.

**Design Principles:**
- ✅ **No AI Dependency** - All classes work independently
- ✅ **Mission-Aware** - Adapters handle observatory-specific conventions
- ✅ **Composable** - Each class does one thing well, can be chained
- ✅ **Reproducible** - HistoryTracker records all operations
- ✅ **Quality-First** - Assessment and validation at every stage

### Three-Tier Pipeline

#### **TIER 1: Preprocessing** (Data Acquisition & Calibration)

**FITSLoader** ✅ PLANNED
- Load FITS files with mission-aware logic
- Handle multi-extension FITS (MEF): SCI, ERR, DQ arrays
- Lazy loading for memory efficiency
- Methods: `load()`, `get_hdu()`, `get_metadata()`, `list_extensions()`

**MissionAdapter** (Abstract + Concrete) ✅ PLANNED
- Base: `MissionAdapter` (abstract)
- Concrete: `JWSTAdapter`, `HSTAdapter`, `ChandraAdapter`, `EuclidAdapter`, `PanSTARRSAdapter`
- Map mission-specific conventions to unified interface
- Methods: `identify_science_extension()`, `get_error_array()`, `get_quality_mask()`

**EventBinner** ⏸️ PHASE 5 (Chandra X-ray)
- Convert event lists to images
- Energy filtering (soft/medium/hard: 0.5-1.2, 1.2-2.0, 2.0-7.0 keV)
- 2D histogram binning
- Methods: `bin_events()`, `filter_by_energy()`, `create_band_images()`

**Calibrator** ⏸️ PHASE 4
- Bias/overscan correction, dark subtraction, flat-fielding
- Background estimation/subtraction (sigma-clipped stats)
- Methods: `apply_bias()`, `apply_dark()`, `apply_flat()`, `subtract_background()`

**QualityAssessor** ✅ PLANNED - Phase 1
- SNR calculation, saturation detection, noise estimation
- Dynamic range measurement
- Methods: `calculate_snr()`, `detect_saturation()`, `estimate_noise()`, `assess_quality()`
- Returns: Structured quality report

**FITSMetadata** ✅ **IMPLEMENTED** (2025-10-24)
- Extract key metadata: filter, wavelength, exposure, instrument
- Mission-aware keyword parsing (JWST, HST, Chandra, Euclid, PanSTARRS, GALEX)
- Validate required keywords
- Methods: `extract_metadata()`, `validate_required()`, `get_filter()`, `get_wavelength()`
- **Status:** Fully functional with dataclass result, warning system, pixel scale extraction

#### **TIER 2: Processing** (Alignment & Enhancement)

**WCSHandler** ✅ PLANNED - Phase 2
- Extract/validate WCS from headers
- Pixel ↔ sky coordinate conversion
- Methods: `extract_wcs()`, `validate()`, `pixel_to_sky()`, `sky_to_pixel()`

**Reprojector** ✅ PLANNED - Phase 2
- Wrapper around `reproject` library
- Interpolation and exact flux-conserving methods
- Align multiple images to target WCS
- Methods: `reproject_to_target()`, `align_image_set()`, `choose_reference_frame()`

**Normalizer** ✅ PLANNED - Phase 2
- Interval selection: MinMax, Percentile, ZScale, Manual
- Methods: `apply_zscale()`, `apply_percentile()`, `apply_manual()`, `get_interval()`

**Stretcher** ✅ PLANNED - Phase 2
- Non-linear transformations: Linear, Sqrt, Log, Asinh, Power
- Per-channel or unified stretching
- Methods: `apply_linear()`, `apply_asinh()`, `apply_log()`, `create_stretch()`

**Enhancer** ⏸️ PHASE 4
- CLAHE, unsharp masking, star highlighting, luminance masking
- Methods: `apply_clahe()`, `unsharp_mask()`, `enhance_stars()`, `luminance_mask()`

**CosmicRayRejecter** ⏸️ PHASE 5
- Identify cosmic ray hits, create rejection masks
- Methods: `detect_cosmic_rays()`, `create_mask()`, `clean_image()`

#### **TIER 3: Postprocessing** (Composition & Output)

**ChannelMapper** ✅ PLANNED - Phase 3
- Assign bands to RGB channels
- Chromatic ordering (wavelength → color)
- Methods: `auto_map_by_wavelength()`, `custom_mapping()`, `validate_mapping()`

**Compositor** ✅ PLANNED - Phase 3
- Lupton algorithm (preserve color in bright regions)
- Simple RGB (independent channel scaling)
- Methods: `create_lupton_rgb()`, `create_simple_rgb()`, `create_narrowband()`

**ColorBalancer** ⏸️ PHASE 4
- Channel weight adjustment, white balance
- Methods: `balance_channels()`, `adjust_white_point()`, `scale_channel()`

**HistoryTracker** ✅ PLANNED - Phase 3
- Record all processing steps for reproducibility
- FITS header-compatible history
- Methods: `record_step()`, `get_history()`, `export_to_fits_header()`

**ImageExporter** ✅ PLANNED - Phase 3
- Save PNG/TIFF/FITS with proper metadata
- Methods: `save_png()`, `save_fits()`, `save_with_metadata()`

**PreviewGenerator** ⏸️ PHASE 4
- Quick low-resolution previews, thumbnails
- Methods: `generate_preview()`, `create_thumbnail()`, `quick_display()`

### File Organization

```
src/astro_vision_composer/
├── preprocessing/
│   ├── fits_loader.py          [PLANNED - Phase 1]
│   ├── mission_adapters.py     [PLANNED - Phase 1]
│   ├── event_binner.py         [PHASE 5]
│   ├── calibrator.py           [PHASE 4]
│   └── quality_assessor.py     [PLANNED - Phase 1]
├── processing/
│   ├── wcs_handler.py          [✅ IMPLEMENTED]
│   ├── reprojector.py          [✅ IMPLEMENTED]
│   ├── normalizer.py           [✅ IMPLEMENTED]
│   ├── stretcher.py            [✅ IMPLEMENTED]
│   ├── enhancer.py             [PHASE 4]
│   └── cosmic_ray.py           [PHASE 5]
├── postprocessing/
│   ├── channel_mapper.py       [PLANNED - Phase 3]
│   ├── compositor.py           [PLANNED - Phase 3]
│   ├── color_balancer.py       [PHASE 4]
│   ├── history_tracker.py      [PLANNED - Phase 3]
│   ├── exporter.py             [PLANNED - Phase 3]
│   └── preview.py              [PHASE 4]
└── utilities/
    ├── metadata.py             [✅ IMPLEMENTED]
    ├── pipeline.py             [PHASE 4]
    ├── optimizer.py            [PHASE 5]
    └── validation.py           [PHASE 4]
```

### Implementation Schedule

**Phase 1 - Core Preprocessing** (Week 1) - ✅ **COMPLETE**
- [x] FITSMetadata ✅ Mission-aware metadata extraction
- [x] FITSLoader ✅ Intelligent FITS loading with lazy/memmap support
- [x] MissionAdapter ✅ Abstract base class
- [x] PanSTARRSAdapter ✅ PanSTARRS-specific adapter
- [x] JWSTAdapter ✅ JWST-specific adapter with DQ flag interpretation
- [x] HSTAdapter ✅ HST-specific adapter with DQ flag interpretation
- [x] QualityAssessor ✅ Statistical quality analysis (SNR, saturation, noise)
- [x] Demo script ✅ Phase 1 demonstration example
- [ ] Unit tests for Phase 1 (deferred to Phase 2)

**Phase 2 - Processing Essentials** (Week 2) - ✅ **COMPLETE**
- [x] WCSHandler ✅ WCS validation and comparison
- [x] Reprojector ✅ Image alignment wrapper (reproject library)
- [x] Normalizer ✅ Interval selection (MinMax, Percentile, ZScale, Manual)
- [x] Stretcher ✅ Non-linear transforms (Linear, Sqrt, Log, Asinh, Power, HistEq)
- [x] Demo script ✅ Phase 2 demonstration with visualization
- [ ] Unit tests for Phase 2 (deferred to Phase 3)

**Phase 3 - Composition** (Week 3) - ✅ **COMPLETE**
- [x] ChannelMapper ✅ RGB channel assignment by wavelength (chromatic ordering)
- [x] Compositor ✅ Lupton RGB and simple RGB composition
- [x] ImageExporter ✅ Save PNG/TIFF/JPEG with metadata
- [x] HistoryTracker ✅ Record processing steps for reproducibility
- [x] Demo script ✅ Phase 3 demonstration with end-to-end pipeline
- [ ] Unit tests for Phase 3 (deferred to Phase 4)

**Phase 4 - Advanced Features** (Week 4) - ⚠️ **PARTIALLY COMPLETE** (3/5 components)
- [x] Calibrator ✅ Bias/dark/flat corrections, background subtraction
- [x] Enhancer ✅ CLAHE, unsharp masking, star highlighting
- [x] ColorBalancer ✅ White balance, saturation, color temperature
- [x] PreviewGenerator ✅ Thumbnails, multi-resolution previews
- [ ] ProcessingPipeline ⚠️ **DESIGNED** (see DESIGN_ProcessingPipeline.md) - Not yet implemented
- [ ] ValidationReport ⚠️ **DESIGNED** (see DESIGN_ValidationReport.md) - Not yet implemented

**Phase 5 - Specialized** (Optional)
- [ ] EventBinner (Chandra)
- [ ] CosmicRayRejecter
- [ ] ParameterOptimizer

### FITSMetadata Class (Implemented)

**Features:**
- **Mission Detection:** Identifies JWST, HST, Chandra, Euclid, PanSTARRS, GALEX from headers
- **Filter Extraction:** Handles FILTER, FILTNAM, FILTER1 keywords
- **Wavelength Lookup:** Database of standard filters (optical, NIR, MIRI, narrowband)
  - PanSTARRS: g(481nm), r(617nm), i(752nm), z(866nm), y(962nm)
  - JWST NIRCam: F070W-F444W
  - JWST MIRI: F560W-F2100W
  - HST: F435W, F555W, F606W, F814W
  - Narrowband: H-alpha(656nm), OIII(501nm), SII(672nm)
- **WCS Pixel Scale:** Extracts from CD matrix or CDELT keywords
- **Validation:** Warning system for missing critical data
- **Dataclass Result:** Clean, typed metadata with pretty repr

**Methods:**
```python
extract_metadata(header, extension_name=None) -> FITSMetadataResult
validate_required(result, required_fields) -> bool
```

**Example Usage:**
```python
from astropy.io import fits
from astro_vision_composer.utilities import FITSMetadata

metadata = FITSMetadata()
with fits.open('panstarrs_g.fits') as hdul:
    result = metadata.extract_metadata(hdul[0].header)
    print(result)  # FITSMetadata(Mission=PanSTARRS, Filter=g, λ=481nm, ...)

    if result.warnings:
        print("Warnings:", result.warnings)
```

### Example Workflow (After Full Implementation)

```python
from astro_vision_composer.preprocessing import FITSLoader, QualityAssessor
from astro_vision_composer.processing import Reprojector, Normalizer, Stretcher
from astro_vision_composer.postprocessing import ChannelMapper, Compositor, ImageExporter

# Load 3 PanSTARRS bands
loader = FITSLoader(mission="PanSTARRS")
bands = {
    'g': loader.load('g_band.fits'),
    'r': loader.load('r_band.fits'),
    'i': loader.load('i_band.fits')
}

# Quality assessment (no AI needed!)
qa = QualityAssessor()
for name, data in bands.items():
    report = qa.assess_quality(data)
    print(f"{name}: SNR={report.snr:.1f}, Saturation={report.saturated}")

# Align to common WCS
reprojector = Reprojector(method='interp')
aligned = reprojector.align_image_set(bands, reference='i')

# Normalize and stretch
normalizer = Normalizer()
stretcher = Stretcher()
for name, img in aligned.items():
    aligned[name] = stretcher.apply_asinh(
        normalizer.apply_zscale(img), a=0.1
    )

# Auto-map by wavelength: g→Blue, r→Green, i→Red
mapper = ChannelMapper()
mapping = mapper.auto_map_by_wavelength(['i', 'r', 'g'])

# Create Lupton composite
compositor = Compositor()
rgb = compositor.create_lupton_rgb(
    r=aligned[mapping['red']],
    g=aligned[mapping['green']],
    b=aligned[mapping['blue']],
    stretch=0.5, Q=8
)

# Export with processing history
exporter = ImageExporter()
exporter.save_png(rgb, 'composite.png', history=True)
```

### Current Progress Summary

**Phase 1 Complete (2025-10-24):**
- ✅ Architecture design (21 classes, 3 tiers)
- ✅ Folder structure created
- ✅ FITSMetadata - Mission-aware metadata extraction (JWST, HST, PanSTARRS, Chandra, Euclid, GALEX)
- ✅ FITSLoader - Intelligent FITS loading with lazy/memmap support, auto extension detection
- ✅ MissionAdapter - Abstract base + 3 concrete adapters (PanSTARRS, JWST, HST)
- ✅ QualityAssessor - Statistical quality analysis (SNR, saturation, noise, dynamic range)
- ✅ Demo script - Complete Phase 1 demonstration (examples/phase1_demo.py)

**Phase 1 Deliverables:**
- **4 Core Classes:** FITSMetadata, FITSLoader, MissionAdapter family (3 adapters), QualityAssessor
- **3 Dataclasses:** FITSData, FITSMetadataResult, QualityReport
- **~1000 lines of production code** with comprehensive docstrings
- **No AI dependency** - Pure statistical/astronomical analysis

**Phase 2 Complete (2025-10-25):**
- ✅ WCSHandler - WCS validation, comparison, pixel scale calculation
- ✅ Reprojector - Image alignment via reproject library (interp/exact methods)
- ✅ Normalizer - Interval selection (MinMax, Percentile, ZScale, Manual)
- ✅ Stretcher - Non-linear transforms (Linear, Sqrt, Log, Asinh, Power, HistEq)
- ✅ Demo script - Phase 2 demonstration with visualization (examples/phase2_demo.py)

**Phase 2 Deliverables:**
- **4 Core Classes:** WCSHandler, Reprojector, Normalizer, Stretcher
- **2 Dataclasses:** WCSInfo
- **~1200 lines of production code** with comprehensive docstrings
- **Astropy-based** - Uses astropy.wcs, astropy.visualization, reproject library
- **Working pipeline:** Load → Validate WCS → Align → Normalize → Stretch

**Phase 3 Complete (2025-10-25):**
- ✅ ChannelMapper - RGB channel assignment by wavelength (chromatic ordering)
- ✅ Compositor - Lupton RGB and simple RGB compositing
- ✅ ImageExporter - Save PNG/TIFF/JPEG with metadata
- ✅ HistoryTracker - Record processing steps for reproducibility
- ✅ Demo script - End-to-end pipeline demonstration (examples/phase3_demo.py)

**Phase 3 Deliverables:**
- **4 Core Classes:** ChannelMapper, Compositor, ImageExporter, HistoryTracker
- **2 Dataclasses:** ChannelMapping, ProcessingStep
- **~1000 lines of production code** with comprehensive docstrings
- **Complete RGB pipeline:** FITS → Normalize → Stretch → Map → Composite → Export

**Key Achievement:** COMPLETE END-TO-END PIPELINE! Can now process FITS files to publication-ready RGB composites!

**Phase 4 Complete (2025-10-25):**
- ✅ Calibrator - CCD reduction (bias, dark, flat, background)
- ✅ Enhancer - Advanced image enhancement (CLAHE, unsharp masking)
- ✅ ColorBalancer - Color adjustment (white balance, saturation, temperature)
- ✅ PreviewGenerator - Quick previews and thumbnails
- ⚠️ ProcessingPipeline - Designed but not yet implemented (see DESIGN_ProcessingPipeline.md)
- ⚠️ ValidationReport - Designed but not yet implemented (see DESIGN_ValidationReport.md)

**Phase 4 Deliverables:**
- **4 Core Classes:** Calibrator, Enhancer, ColorBalancer, PreviewGenerator
- **~1100 lines of production code** with comprehensive docstrings
- **Complete feature set:** Calibration → Enhancement → Color adjustment → Export

---

## Project Metrics & Status (2025-10-25)

### Codebase Statistics:
- **Total Lines of Code:** ~5,200 (production code only, deprecated code moved)
- **Active Production Code:** 5,200 lines (Phases 1-3 complete)
- **Components Implemented:** 16/21 (76%)
- **Test Coverage:** 0% (no unit tests yet - critical gap)
- **Demo Scripts:** 3 NEW batch-processing demos (working end-to-end)

### Implementation by Tier:
- **Tier 1 (Preprocessing):** 5/6 components (83%) - Missing: EventBinner (Phase 5)
- **Tier 2 (Processing):** 5/6 components (83%) - Missing: CosmicRayRejecter (Phase 5)
- **Tier 3 (Postprocessing):** 6/6 components (100%) - ✅ **COMPLETE**
- **Utilities:** 2/4 components (50%) - Missing: ProcessingPipeline, ValidationReport

### Production Readiness Assessment:
- ✅ **Core Functionality:** 100% (Full pipeline works, demos proven)
- ❌ **Testing:** 0% (No unit tests - CRITICAL)
- ✅ **Documentation:** 90% (Comprehensive docstrings)
- ⚠️ **Usability:** 70% (Demo scripts work, but need ProcessingPipeline API)
- ❌ **Quality Assurance:** 10% (No ValidationReport, no tests)

**Overall Production Readiness: 60%**

### What Works Right Now:

**Option 1: Demo Scripts (RECOMMENDED for users)**
```bash
# Complete end-to-end pipeline with NOIRLab Eagle Nebula data
python examples/phase3_demo_new.py examples/data/NOIRLab/examples/edu008/data --force

# Output: RGB composite PNG + TIFF in examples/data/.../output/phase3/
# Automatically runs Phase 1 & 2 if needed
```

**Option 2: Manual Python API**
```python
# Complete workflow (manual chaining):
from astro_vision_composer.preprocessing import FITSLoader, QualityAssessor
from astro_vision_composer.processing import Normalizer, Stretcher
from astro_vision_composer.postprocessing import ChannelMapper, Compositor, ImageExporter

# Load → Assess → Normalize → Stretch → Map → Composite → Export
# ALL of this works! Just requires manual chaining.
```

### What's Missing for Production:

1. **ProcessingPipeline** (DESIGNED) - High-level convenience API
   - Single-call workflows: `pipeline.create_rgb_composite(files, output)`
   - Builder pattern: `pipeline.load().normalize().stretch().export()`
   - See: [DESIGN_ProcessingPipeline.md](DESIGN_ProcessingPipeline.md) (500-600 lines estimated)
   - **Status:** Fully designed, 0 lines implemented

2. **ValidationReport** (DESIGNED) - Quality assurance system
   - Aggregate quality checks across all bands
   - WCS validation and alignment checks
   - Exportable reports (JSON, HTML, text)
   - See: [DESIGN_ValidationReport.md](DESIGN_ValidationReport.md) (400-500 lines estimated)
   - **Status:** Fully designed, 0 lines implemented

3. **Unit Tests** - Critical for production confidence
   - No test coverage currently (0%)
   - Estimated 1000+ lines for core coverage
   - Deferred across all phases
   - **Status:** Not started

### Critical Next Steps (Priority Order):

1. **CRITICAL:** Write unit tests (~5 days) - Zero test coverage is unacceptable
2. **HIGH:** Implement ProcessingPipeline (~3 days) - Convenience API to replace demos
3. **HIGH:** Implement ValidationReport (~2 days) - Quality assurance
4. **MEDIUM:** Test with multiple NOIRLab datasets (~2 days) - We have 20 datasets!
5. **LOW:** Add Phase 4 demo script (~1 day) - Show Calibrator/Enhancer/ColorBalancer
6. **LOW:** Implement Phase 5 optional components (EventBinner, CosmicRayRejecter)

### Files & Organization:
```
src/astro_vision_composer/
├── preprocessing/          [✅ COMPLETE] 5/6 components (1,870 lines)
│   ├── fits_loader.py          [✅ IMPLEMENTED] ~320 lines
│   ├── mission_adapters.py     [✅ IMPLEMENTED] ~630 lines (3 adapters)
│   ├── quality_assessor.py     [✅ IMPLEMENTED] ~340 lines
│   ├── calibrator.py           [✅ IMPLEMENTED] ~320 lines
│   └── event_binner.py         [❌ NOT IMPL] Deferred to Phase 5
├── processing/             [✅ COMPLETE] 5/6 components (1,355 lines)
│   ├── wcs_handler.py          [✅ IMPLEMENTED] ~305 lines
│   ├── reprojector.py          [✅ IMPLEMENTED] ~250 lines
│   ├── normalizer.py           [✅ IMPLEMENTED] ~260 lines
│   ├── stretcher.py            [✅ IMPLEMENTED] ~260 lines
│   ├── enhancer.py             [✅ IMPLEMENTED] ~280 lines
│   └── cosmic_ray.py           [❌ NOT IMPL] Deferred to Phase 5
├── postprocessing/         [✅ COMPLETE] 6/6 components (1,440 lines)
│   ├── channel_mapper.py       [✅ IMPLEMENTED] ~200 lines
│   ├── compositor.py           [✅ IMPLEMENTED] ~300 lines
│   ├── history_tracker.py      [✅ IMPLEMENTED] ~200 lines
│   ├── exporter.py             [✅ IMPLEMENTED] ~240 lines
│   ├── color_balancer.py       [✅ IMPLEMENTED] ~280 lines
│   └── preview.py              [✅ IMPLEMENTED] ~220 lines
└── utilities/              [⚠️ PARTIAL] 2/4 components (260 lines + designs)
    ├── metadata.py             [✅ IMPLEMENTED] ~260 lines
    ├── pipeline.py             [⚠️ DESIGNED] See DESIGN_ProcessingPipeline.md
    └── validation.py           [⚠️ DESIGNED] See DESIGN_ValidationReport.md

deprecated/ (candidates for removal):
├── composite_generator.py  [358 lines] - Superseded by Compositor
├── fits_processor.py       [212 lines] - Superseded by FITSLoader + Normalizer
└── vision_compositor.py    [373 lines] - AI vision attempt (paused)
```

### Dependencies Health:
- ✅ All dependencies actively maintained
- ✅ Graceful degradation for optional deps
- ✅ Clear error messages when deps missing

**Core:** numpy, astropy, scipy
**Optional:** reproject, scikit-image, matplotlib, Pillow

---

## Future Work: Mission-Specific WCS Characteristics

**Deferred to Post-Phase 3:**

During Phase 2 planning, we identified the need for mission-specific WCS quirk knowledge (geometric distortion, astrometric quality, etc.). This is separate from DQ flag interpretation (already handled by MissionAdapter).

**Implementation Strategy (decided 2025-10-25):**

**Option A (Near-term):** Header-driven WCS characteristics
- Extract distortion model info from headers (SIP keywords, TPV projection, etc.)
- No assumptions about what "should" be there
- Let real data teach us the patterns

**Option B (Medium-term):** Empirical testing with real data
- Download sample FITS from each mission (JWST, HST, Chandra, PanSTARRS, Euclid)
- Examine actual WCS headers and quality
- Document observed patterns and quirks
- Build knowledge base organically

**Option C (Long-term):** Curated from official documentation
- Research STScI/CXC/ESA instrument handbooks
- Document authoritative WCS specifications
- Implement based on official specs

**Proposed Addition to MissionAdapter:**
```python
def get_wcs_characteristics(self, header, instrument=None) -> Dict:
    """Get mission-specific WCS quality characteristics.

    Returns:
        {
            'distortion_model': 'SIP'|'TPV'|'polynomial'|'none'|'unknown',
            'distortion_magnitude': 'low'|'moderate'|'high'|'unknown',
            'astrometric_quality': 'excellent'|'good'|'fair'|'poor'|'unknown',
            'notes': List[str],
            'warnings': List[str]
        }
    """
```

**Use Case:** Reprojector can choose algorithm based on distortion level:
- Low distortion → `reproject_interp` (fast)
- High distortion → `reproject_exact` (flux-conserving, accurate)

**Status:** Documented for future implementation, not blocking Phase 3.

---

## Test Image Sources

### Working URLs
- **NOIRLab FITS Liberator:** `https://noirlab.edu/public/media/archives/education/large/eduXXX.jpg`
  - edu027: M106 Galaxy
  - edu008: Eagle Nebula (M16)
  - edu010: M17 Star Forming Nebula
  - Pattern: Replace XXX with 3-digit number

### User-Agent Required
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
```

---

## Development Environment

### Server Access
```bash
# Start SSH tunnel
ssh -L 5000:localhost:5000 jakko@192.168.50.194

# On remote server
python scripts/transformers_pipeline_server.py --port 5000
```

### Test Run
```bash
python test_fewshot_astronomy.py "path/to/fits/file"
```

---

## Critical Reflections

### What We Got Wrong
1. **Premature celebration:** Assumed JSON format = working analysis
2. **Didn't validate:** Should have checked if responses varied with input
3. **Template problem:** Giving example values caused copying behavior

### What We've Learned
1. **Vision works:** Model can see, basic tests prove it
2. **Format teaching works:** JSON structure is teachable
3. **Content is hard:** Teaching WHAT to analyze vs. HOW to format is the challenge

### Next Steps
1. **Validate variation:** Test with obviously different images (empty vs. crowded field)
2. **Minimize copying:** Remove similar features from training examples
3. **Explicit comparisons:** "This is different from previous images because..."

---

## File Reference

### Core Code
- `src/astro_vision_composer/` - Main package
- `scripts/astro_ai_client_v2.py` - Updated client (uses new server API)
- `suppress_warnings.py` - FITS warning suppression

### Test Scripts
- `test_ai_vision.py` - **Proof AI can see** (black/white/red test)
- `test_json_proper_fewshot.py` - **Proof JSON teaching works** (colored squares)
- `test_fewshot_astronomy.py` - **Current focus** (real astronomical training)
- `test_5band_v2.py` - Multi-band analysis (needs work)

### Deprecated/Old
- `test_5band_ai_analysis.py` - Original (template copying issue)
- `test_5band_json.py` - First JSON attempt (same issue)
- `scripts/astro_ai_client.py` - Old single-image API

---

## Honest Assessment

**We are NOT production-ready.** This is a research/learning project exploring:
- How to use vision models for scientific image analysis
- Prompt engineering for structured outputs
- Few-shot learning limitations and capabilities

**What works:** Basic image analysis, JSON format learning, server communication
**What doesn't:** Independent analysis of each image, avoiding template copying
**What's needed:** Better prompting strategy, validation tests, possibly server improvements

---

## Commit History

### 2025-10-24: Honest reassessment
- Identified template copying problem
- Validated basic AI vision works
- Tested few-shot JSON training
- Documented current limitations
- Reset expectations to realistic goals

### 2025-10-23: Initial (false) baseline
- Created infrastructure
- Got JSON responses (but copying templates)
- Documented as "working" (premature)

---

*This project is about learning and experimentation. We're building understanding, not claiming solved problems.*
