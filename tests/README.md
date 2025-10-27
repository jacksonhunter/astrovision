# Test Suite for astro_vision_composer

Comprehensive pytest test suite for the VisionProject astronomical image processing pipeline.

## Overview

- **Total Test Files:** 20+ test files
- **Test Coverage Target:** 80%+
- **Test Categories:** Unit, Integration, Regression
- **Real Data Testing:** Uses NOIRLab FITS datasets

## Installation

### Install Test Dependencies

The tests are configured to automatically find the source code (via [conftest.py](conftest.py:6-10)), so you only need to install test dependencies:

```bash
pip install pytest pytest-cov
```

**Note:** The package does NOT need to be installed in editable mode (`pip install -e .`) for tests to work. The test configuration handles imports automatically.

## Running Tests

### Basic Usage

```bash
# Run all tests (219 tests currently)
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/astro_vision_composer --cov-report=html
# Coverage report will be in htmlcov/index.html

# Run specific category
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests
pytest -m integration                 # All tests marked as integration
pytest -m regression                  # Regression tests

# Skip slow tests
pytest -m "not slow"

# Run specific test file
pytest tests/unit/test_postprocessing/test_compositor.py

# Run specific test function
pytest tests/unit/test_postprocessing/test_compositor.py::test_lupton_rgb_basic
```

## Current Status

**✅ All 219 tests collect successfully with zero errors!**

### Test Skipping

Some tests will automatically skip if:

1. **NOIRLab data not found**: Real astronomical test data not downloaded
   - Tests will skip gracefully (expected behavior)
   - Download NOIRLab examples to enable these tests

2. **Slow tests**: Marked with `@pytest.mark.slow`
   - Skip with: `pytest -m "not slow"`

### Excluded Directories

The following directory is excluded from test collection (configured in [pytest.ini](../pytest.ini)):
- `tests/deprecated/` - Old/unmaintained experimental scripts

## Quality Assessment Scoring System

Tests can record quality scores (SNR, saturation, etc.) without affecting pass/fail. This provides diagnostic information without making tests brittle.

### Usage:

```python
def test_something(record_quality_score):
    # ... test code ...
    record_quality_score('snr', 42.5, {'band': 'g'})
```

After tests, you'll see aggregate quality statistics:
```
========== Quality Assessment Scores (Passed Tests) ==========
  snr: avg=42.50, min=15.20, max=87.30, count=45
  saturation: avg=0.02, min=0.00, max=0.15, count=45
```

### How It Works

The quality scoring system uses pytest hooks ([conftest.py:309-406](conftest.py#L309)):
- `pytest_runtest_makereport` - Collects scores from passed tests
- `pytest_terminal_summary` - Displays aggregated statistics
- Scores are stored in `config._quality_scores` during test run
- Only displayed for tests that passed (doesn't affect pass/fail)

## Test Coverage

**Current Status:** 219 tests, ~45% passing (118 failed, 98 passed, 3 skipped)
**Coverage Target:** 80%+

### Generate Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src/astro_vision_composer --cov-report=html

# Open the report (Windows)
start htmlcov\index.html

# Generate terminal report with missing lines
pytest --cov=src/astro_vision_composer --cov-report=term-missing
```

### Test Suite Structure

```
tests/
├── conftest.py              # 443 lines - Fixtures, hooks, quality scoring
├── unit/ (16 files)         # ~185 tests - Component isolation
│   ├── test_preprocessing/  # FITSLoader, QualityAssessor, Calibrator, MissionAdapters
│   ├── test_processing/     # WCSHandler, Reprojector, Normalizer, Stretcher, Enhancer
│   └── test_postprocessing/ # ChannelMapper, Compositor, ColorBalancer, etc.
├── integration/ (4 files)   # ~25 tests - Multi-component workflows
│   ├── test_phase1_preprocessing.py
│   ├── test_phase2_processing.py
│   ├── test_phase3_postprocessing.py
│   └── test_end_to_end_pipeline.py
└── regression/ (1 file)     # ~15 tests - Real NOIRLab datasets
    └── test_noirlab_datasets.py
```

### Component Coverage

| Component | Unit Tests | Integration | Status |
|-----------|------------|-------------|--------|
| FITSLoader | 12+ | ✅ | Mostly passing |
| MissionAdapter | 15+ | ✅ | API fixes needed |
| QualityAssessor | 13+ | ✅ | Mostly passing |
| Calibrator | 10+ | ✅ | Mostly passing |
| WCSHandler | 10+ | ✅ | API fixes needed |
| Reprojector | 8+ | ✅ | API fixes needed |
| Normalizer | 12+ | ✅ | Mostly passing |
| Stretcher | 13+ | ✅ | Mostly passing |
| Enhancer | 10+ | ✅ | API fixes needed |
| ChannelMapper | 10+ | ✅ | API fixes needed |
| Compositor | 12+ | ✅ | Mostly passing |
| ColorBalancer | 10+ | ✅ | API fixes needed |
| PreviewGenerator | 10+ | ✅ | API fixes needed |
| HistoryTracker | 10+ | ✅ | API fixes needed |
| ImageExporter | 13+ | ✅ | API fixes needed |
| FITSMetadata | 11+ | ✅ | API fixes needed |

## Recent API Fixes (2025-10-26)

The following API mismatches were fixed in tests to match actual implementations:

1. **FITSData attribute:** `fits_data.data` → `fits_data.science`
2. **HistoryTracker signature:** `record(op, params)` → `record(op, params, component)`
3. **ChannelMapper method:** `map_by_wavelength()` → `auto_map_by_wavelength()`
4. **FITSMetadata attribute:** `.filter` → `.filter_name`
5. **ImageExporter method:** `.save()` → `.auto_save()`
6. **QualityReport attribute:** `.saturated_fraction` → `.saturation_fraction`
7. **PreviewGenerator method:** `.create_multiple_previews()` → `.generate_preview()`
8. **Glob patterns:** Fixed duplicate paths in regression tests

**Result:** Reduced failures from 91 to ~60 remaining issues.

## Troubleshooting

### "ModuleNotFoundError: No module named 'astro_vision_composer'"

This should NOT happen with the current test configuration. If you see this error:

1. Check that `conftest.py` has the sys.path modification at the top
2. Verify you're running pytest from the project root directory
3. As a fallback, install in editable mode: `pip install -e .`

### "pytest: error: unrecognized arguments: --cov"

pytest-cov is not installed:
```bash
pip install pytest-cov
```

### Tests skip due to missing NOIRLab data

This is expected. Tests will automatically skip if test data is not available. To run these tests:

1. Download NOIRLab example datasets
2. Place in `examples/data/NOIRLab/examples/`
3. Tests will auto-detect and run

### All tests fail with import errors

Check your Python path and ensure you're in the correct conda environment:
```bash
# Activate environment
conda activate visionproject

# Verify environment
python -c "import sys; print(sys.executable)"
```
