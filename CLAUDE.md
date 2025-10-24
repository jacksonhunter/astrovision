# VisionProject - AI-Guided Astronomical Image Processing

**Project Status:** BASELINE ESTABLISHED
**Date:** 2025-10-23
**Baseline Script:** `test_5band_ai_analysis.py`

## Project Overview

AI-guided astronomical image composition from multi-band FITS data using local vision models for intelligent processing decisions. This project enables planetarium-quality image creation from scientific data with structured AI feedback on all processing parameters.

---

## BASELINE: 5-Band AI Analysis Pipeline

### Current Capabilities ✅

**Script:** `test_5band_ai_analysis.py`

#### 1. Multi-Band FITS Processing
- Loads all 5 PanSTARRS bands (g, r, i, z, y) from MAST archive
- Handles compressed FITS with WCS metadata
- Zscale normalization with proper dynamic range handling
- Suppresses common FITS warnings (deprecated PC keywords)

#### 2. Structured AI Analysis
- **Individual Band Assessment** (per band):
  - Quality score (0-10)
  - Noise level (low/medium/high)
  - Dynamic range evaluation
  - Visible features (stars, nebulosity, galaxies)
  - Brightness distribution characteristics
  - Best RGB channel recommendation

- **RGB Mapping Recommendation**:
  - Physics-informed wavelength ordering (longer → red, shorter → blue)
  - Band assignment with reasoning
  - Current result: i(752nm)→Red, r(617nm)→Green, g(481nm)→Blue

- **Quality Assessment** (final composite):
  - Evaluation of each processing step (zscale, CLAHE, unsharp mask, star enhancement, luminance masking)
  - Effective/not effective flags
  - Specific parameter suggestions (clip_limit, sigma, amount, threshold, percentile)
  - Overall quality score
  - Recommendations for improvement

#### 3. Processing Pipeline
```
Raw FITS → Zscale → CLAHE → Unsharp Mask → Star Boost → Luminance Masking → Composite
```

**Parameters (AI-validated):**
- Zscale: Adaptive histogram normalization ✓ Working well
- CLAHE: clip_limit=0.03 ✓ Good contrast
- Unsharp mask: sigma=2, amount=0.5 ✓ Effective
- Star enhancement: 99th percentile, 1.3x boost ✓ Effective
- Luminance masking: 5th percentile → black ✓ Effective

#### 4. Dual Output Format
- **Visual:** PNG composite (95% quality JPEG compression)
- **Data:** Structured JSON with:
  - All band analyses (raw text + parsed JSON)
  - RGB mapping decision + reasoning
  - Quality assessment with parameter suggestions
  - Model attribution (Llama-3.2-11B-Vision-Instruct)

### Success Metrics

**Latest Run (2025-10-23):**
- Bands processed: 5/5 ✅
- AI responses: 7/7 (5 bands + 1 mapping + 1 assessment) ✅
- JSON parsing: 100% success rate ✅
- Overall quality: 8/10 ✅
- Processing time: ~3 minutes total
- Major issues: None

### Architecture

#### Components
1. **FITSImageProcessor** (`src/astro_vision_composer/fits_processor.py`)
   - FITS loading with WCS support
   - Band extraction and normalization
   - Metadata extraction

2. **CompositeImageGenerator** (`src/astro_vision_composer/composite_generator.py`)
   - Multi-band composition
   - Enhancement pipeline (CLAHE, unsharp, star boost, masking)
   - Image output with quality control

3. **Remote AI Server** (`scripts/transformers_pipeline_server.py`)
   - HTTP API on localhost:5000
   - Llama-3.2-11B-Vision model
   - Base64 image handling
   - Structured JSON prompts

#### Data Flow
```
FITS Files → FITSImageProcessor → Normalized Bands
                                        ↓
                                   AI Analysis (per band)
                                        ↓
                                   RGB Mapping (AI recommendation)
                                        ↓
                           CompositeImageGenerator → Processing Pipeline
                                        ↓
                              AI Quality Assessment
                                        ↓
                            PNG Output + JSON Results
```

---

## Project Goals

### Completed ✅
- [x] FITS loading and band extraction
- [x] Zscale normalization
- [x] Multi-band composition
- [x] Processing pipeline (CLAHE, unsharp, star boost, masking)
- [x] Remote AI integration via HTTP
- [x] **Structured JSON responses (MAJOR ACHIEVEMENT)**
- [x] Individual band analysis
- [x] RGB mapping recommendation
- [x] Quality assessment with parameter feedback
- [x] Dual output (image + data)
- [x] 5-band workflow (g, r, i, z, y)

### In Progress 🔄
- [ ] Parameter optimization loop (use AI suggestions to refine processing)
- [ ] Batch processing of multiple targets
- [ ] Alternative band combinations (z, y utilization)

### Future Goals 🎯
- [ ] Integration into main `astro_vision_composer` module
- [ ] Automated parameter tuning based on AI feedback
- [ ] Multi-target comparison and ranking
- [ ] CMYK composite support with AI guidance
- [ ] Processing history tracking (JSON log of all runs)
- [ ] Web interface for batch processing
- [ ] Real-time processing preview

---

## Known Issues & Limitations

### Current Limitations
1. **Unused bands:** z (866nm) and y (962nm) bands not utilized in RGB composite
   - Opportunity: Could use for luminance or alternative mappings
   - AI consistently recommends i-r-g mapping for RGB

2. **Processing time:** ~3 minutes for full pipeline
   - 5 bands × ~30s each for loading/normalization
   - 3 AI calls × ~20s each
   - Processing pipeline ~30s
   - **Acceptable for single-target processing**

3. **AI server dependency:** Requires separate server process
   - Need SSH tunnel: `ssh -L 5000:localhost:5000 jakko@192.168.50.194`
   - Server startup: `python scripts/transformers_pipeline_server.py --port 5000`
   - **Could benefit from automatic server check/restart**

4. **FITS warnings:** Deprecation warnings for PC matrix keywords
   - Suppressed via `suppress_warnings.py`
   - Not a functional issue (WCS still loads correctly)

### Non-Issues (Confirmed Working)
- ✅ JSON parsing reliability (100% success rate)
- ✅ Band alignment (all 6284×6320, same WCS)
- ✅ Memory handling (processed without issues)
- ✅ AI response quality (consistent, relevant feedback)

---

## Critical Success Factors

### What Made This Work

1. **Structured JSON Prompts:**
   - Provide exact example JSON in prompt
   - Instruct "respond ONLY with valid JSON"
   - Include expected schema/format
   - AI follows format consistently

2. **Physics-Informed Guidance:**
   - Wavelength information in prompts
   - Wavelength-to-color mapping principle
   - AI uses domain knowledge correctly

3. **Modular Processing:**
   - Separate band loading from composition
   - Independent processing steps
   - Each step can be AI-evaluated

4. **Dual Output:**
   - Visual for human assessment
   - JSON for programmatic analysis
   - Enables feedback loops and comparison

---

## Quick Start

### Prerequisites
```bash
# 1. Start AI server (on remote or local)
python scripts/transformers_pipeline_server.py --port 5000

# 2. SSH tunnel if remote
ssh -L 5000:localhost:5000 jakko@192.168.50.194
```

### Run Baseline Script
```bash
python test_5band_ai_analysis.py \
  "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.g" \
  "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.r" \
  "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.i" \
  "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.z" \
  "C:\Users\jacks\.mission_control\cache\mast_staging\mastDownload\PS1\rings.v3.skycell.2159.034.stk.y" \
  output.png
```

### Expected Output
- **Image:** `output.png` (or specified filename)
- **JSON:** `5_band_ai.json` (hardcoded in current script)
- **Console:** Detailed processing log with AI assessments

---

## File Reference

### Core Scripts
- `test_5band_ai_analysis.py` - **BASELINE** - Full 5-band pipeline with AI guidance
- `scripts/transformers_pipeline_server.py` - Remote AI server
- `suppress_warnings.py` - FITS warning suppression

### Core Modules
- `src/astro_vision_composer/fits_processor.py` - FITS handling
- `src/astro_vision_composer/composite_generator.py` - Image composition

### Output Examples
- `5_band_ai.png` - Latest composite image
- `5_band_ai.json` - Latest structured analysis

### Documentation
- `README.md` - Project overview and API documentation
- `SETUP.md` - Installation instructions
- `CLAUDE.md` - **This file** - Project status and baseline

---

## Next Steps (Recommended Priority)

### Immediate (This Week)
1. **Batch processing script** - Process multiple targets from MAST
2. **Parameter optimization loop** - Use AI suggestions to refine and reprocess
3. **Alternative band combinations** - Test z/y bands in different roles

### Short-term (This Month)
4. **Integration into main module** - Move logic into `VisionGuidedCompositor`
5. **Processing history** - JSON log of all runs for comparison
6. **Automated server management** - Check/start server automatically

### Long-term
7. **Web interface** - Browser-based batch processing
8. **Multi-object comparison** - Rank multiple targets by quality
9. **Real-time preview** - Adjust parameters with live feedback

---

## Changelog

### 2025-10-23 - BASELINE ESTABLISHED
- ✅ Created `test_5band_ai_analysis.py` as reference implementation
- ✅ Successfully processed all 5 PanSTARRS bands (g, r, i, z, y)
- ✅ Achieved 100% JSON parsing success rate
- ✅ AI quality assessment: 8/10 overall quality
- ✅ All processing steps validated as effective by AI
- ✅ Dual output format (PNG + JSON) working
- ✅ Documented baseline in CLAUDE.md

---

## Critical Insights

### JSON Prompt Engineering
**The key to reliable AI responses:**

```python
prompt = """Respond ONLY with valid JSON in this exact format:
{
  "quality": 7,
  "noise_level": "low",
  "features_visible": ["stars", "nebulosity"]
}

Do not include any text outside the JSON object."""
```

**Results:** 100% parsing success vs previous ~60% with freeform text

### Physics-Informed AI
**The AI understands wavelength-to-color mapping:**
- Correctly maps longer wavelengths (i-band 752nm) → red channel
- Shorter wavelengths (g-band 481nm) → blue channel
- Provides reasoning based on physics, not just aesthetics

### Processing Pipeline Validation
**All steps confirmed effective:**
- Zscale: "working well"
- CLAHE (0.03): "good contrast"
- Unsharp mask (σ=2, amount=0.5): "effective"
- Star boost (99th percentile, 1.3x): "effective"
- Luminance masking (5th percentile): "effective"

**No parameter changes recommended** - current pipeline is well-tuned

---

## Contact & Attribution

**Model:** unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit
**Data:** Pan-STARRS1 via MAST Archive
**Processing:** Astropy, scikit-image, Pillow
**Baseline Established:** 2025-10-23

---

*This baseline represents a working, validated AI-guided astronomical image processing pipeline with structured feedback and physics-informed decision making.*
