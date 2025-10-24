# Test Scripts

Organized test scripts for the AstroVision project.

## Directory Structure

### `validation/`
**Tests that validate AI capabilities work**

- `test_ai_vision.py` - Proves AI can see different images (black/white/red test)
- `test_json_proper_fewshot.py` - Proves JSON format training works (colored squares)
- `test_assistant_messages.py` - Tests server capabilities (assistant message support)

These tests confirm basic functionality and serve as regression tests.

### `astronomical/`
**Current working tests with real astronomical data**

- `test_fewshot_astronomy.py` - **PRIMARY TEST** - Few-shot training with NOIRLab images
- `test_5band_v2.py` - Multi-band PanSTARRS analysis (work in progress)

These are the active development tests we're improving.

### `deprecated/`
**Old tests that revealed problems**

Tests that helped us discover issues (template copying, example value copying, etc.). Kept for historical reference and to document what doesn't work.

## Running Tests

```bash
# Validation tests (should always pass)
python tests/validation/test_ai_vision.py
python tests/validation/test_json_proper_fewshot.py

# Current astronomical tests
python tests/astronomical/test_fewshot_astronomy.py "path/to/fits/file"
```

## Requirements

- AI server running on localhost:5000 (via SSH tunnel)
- FITS files for astronomical tests
- See main README.md for setup instructions
