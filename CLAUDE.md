# VisionProject - AI-Guided Astronomical Image Processing

**Project Status:** IN DEVELOPMENT - Learning Phase
**Date:** 2025-10-24
**Current Focus:** Understanding few-shot JSON training for astronomical image analysis

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
