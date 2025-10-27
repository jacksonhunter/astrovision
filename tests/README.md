# Test Suite for astro_vision_composer

Comprehensive pytest test suite for the VisionProject astronomical image processing pipeline.

## Overview

- **Total Test Files:** 20+ test files
- **Test Coverage Target:** 80%+
- **Test Categories:** Unit, Integration, Regression
- **Real Data Testing:** Uses NOIRLab FITS datasets

## Installation

Install test dependencies:

```bash
pip install -e ".[test]"
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/astro_vision_composer --cov-report=html

# Run specific category
pytest tests/unit/          # Unit tests only
pytest -m integration       # Integration tests
pytest -m regression        # Regression tests

# Skip slow tests
pytest -m "not slow"
```

## Quality Assessment Scoring System

Tests can record quality scores (SNR, saturation, etc.) without affecting pass/fail.

### Usage:

```python
def test_something(record_quality_score):
    # ... test code ...
    record_quality_score('snr', 42.5, {'band': 'g'})
```

After tests, see aggregate quality statistics in the summary!

See full documentation in PYTEST_STRATEGY.md
