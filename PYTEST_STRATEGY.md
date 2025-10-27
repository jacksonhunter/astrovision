# Pytest Strategy for VisionProject

## Overview

Comprehensive testing strategy using real NOIRLab FITS data to test all components of the `astro_vision_composer` package. Tests are designed to handle edge cases like varying numbers of FITS files, blank images, and different observatory formats.

---

## Test Data: NOIRLab Examples

**Location:** `examples/data/NOIRLab/examples/`

**Available Datasets:** ~20 astronomical datasets (edu008-edu025+)

**Characteristics:**
- **edu008**: 3 FITS files (502nm, 656nm, 673nm) - Eagle Nebula ✅ Primary test dataset
- **edu010**: 3 FITS files (502nm, 656nm, 673nm) - M17 Nebula
- **edu011**: 3 FITS files (F606W, F656N, F673N) - HST format
- **edu012**: 2 FITS files (F555W, F814W) - ⚠️ Only 2 bands
- **edu019**: 4 FITS files (336nm, 555nm, 656nm, 814nm) - ⚠️ More than 3 bands
- **edu020**: 2 FITS files (B, V) - Ground-based broadband

**Edge Cases to Handle:**
1. **Exactly 3 FITS files** - Standard RGB workflow
2. **< 3 FITS files** - Can't create RGB (use grayscale or skip)
3. **> 3 FITS files** - Must select best 3 bands for RGB
4. **Blank/corrupted FITS** - Graceful error handling
5. **Different WCS** - Requires reprojection
6. **Different pixel scales** - Requires resampling
7. **Different dimensions** - Requires cropping/padding

---

## Test Structure

```
tests/
├── conftest.py                          # Pytest fixtures and configuration
├── test_data/                           # Symlink or copy of NOIRLab data
├── unit/                                # Unit tests (single class/method)
│   ├── test_preprocessing/
│   │   ├── test_fits_loader.py
│   │   ├── test_mission_adapters.py
│   │   ├── test_quality_assessor.py
│   │   └── test_calibrator.py
│   ├── test_processing/
│   │   ├── test_wcs_handler.py
│   │   ├── test_reprojector.py
│   │   ├── test_normalizer.py
│   │   ├── test_stretcher.py
│   │   └── test_enhancer.py
│   ├── test_postprocessing/
│   │   ├── test_channel_mapper.py
│   │   ├── test_compositor.py
│   │   ├── test_color_balancer.py
│   │   ├── test_exporter.py
│   │   ├── test_history_tracker.py
│   │   └── test_preview.py
│   └── test_utilities/
│       ├── test_metadata.py
│       └── test_pipeline.py
├── integration/                         # Integration tests (multi-class)
│   ├── test_phase1_preprocessing.py
│   ├── test_phase2_processing.py
│   ├── test_phase3_postprocessing.py
│   └── test_end_to_end_pipeline.py
└── regression/                          # Regression tests (known outputs)
    └── test_noirlab_datasets.py
```

---

## Pytest Fixtures (`conftest.py`)

### Core Fixtures

```python
import pytest
from pathlib import Path
import numpy as np
from astropy.io import fits

# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def noirlab_data_dir():
    """Root directory for NOIRLab test data."""
    return Path(__file__).parent.parent / "examples" / "data" / "NOIRLab" / "examples"

@pytest.fixture(scope="session")
def edu008_data(noirlab_data_dir):
    """Eagle Nebula dataset (3 FITS files) - Primary test dataset."""
    data_dir = noirlab_data_dir / "edu008" / "data"
    fits_files = sorted(data_dir.glob("*/*.fits"))
    # Filter out processed files from output directories
    fits_files = [f for f in fits_files if "output" not in str(f)]
    return {
        'dir': data_dir,
        'fits_files': fits_files[:3],  # Take first 3
        'num_bands': 3,
        'expected_wavelengths': [502, 656, 673]  # nm
    }

@pytest.fixture(scope="session")
def edu012_data(noirlab_data_dir):
    """Dataset with only 2 FITS files (edge case)."""
    data_dir = noirlab_data_dir / "edu012" / "data"
    fits_files = sorted(data_dir.glob("edu012/*/*.fits"))
    return {
        'dir': data_dir,
        'fits_files': fits_files,
        'num_bands': 2
    }

@pytest.fixture(scope="session")
def edu019_data(noirlab_data_dir):
    """Dataset with 4 FITS files (edge case)."""
    data_dir = noirlab_data_dir / "edu019" / "data"
    fits_files = sorted(data_dir.glob("edu019/*/*.fits"))
    return {
        'dir': data_dir,
        'fits_files': fits_files,
        'num_bands': 4
    }

@pytest.fixture(scope="function")
def sample_fits_data():
    """Generate synthetic FITS data for unit tests."""
    data = np.random.rand(100, 100).astype(np.float32)
    header = fits.Header()
    header['TELESCOP'] = 'TEST'
    header['INSTRUME'] = 'TESTCAM'
    header['FILTER'] = 'TEST_FILTER'
    header['EXPTIME'] = 60.0
    return data, header

# ============================================================================
# Component Fixtures
# ============================================================================

@pytest.fixture
def fits_loader():
    """FITSLoader instance."""
    from astro_vision_composer.preprocessing import FITSLoader
    return FITSLoader()

@pytest.fixture
def quality_assessor():
    """QualityAssessor instance."""
    from astro_vision_composer.preprocessing import QualityAssessor
    return QualityAssessor()

@pytest.fixture
def normalizer():
    """Normalizer instance."""
    from astro_vision_composer.processing import Normalizer
    return Normalizer()

@pytest.fixture
def stretcher():
    """Stretcher instance."""
    from astro_vision_composer.processing import Stretcher
    return Stretcher()

@pytest.fixture
def enhancer():
    """Enhancer instance."""
    from astro_vision_composer.processing import Enhancer
    return Enhancer()

@pytest.fixture
def channel_mapper():
    """ChannelMapper instance."""
    from astro_vision_composer.postprocessing import ChannelMapper
    return ChannelMapper()

@pytest.fixture
def compositor():
    """Compositor instance."""
    from astro_vision_composer.postprocessing import Compositor
    return Compositor()

@pytest.fixture
def color_balancer():
    """ColorBalancer instance."""
    from astro_vision_composer.postprocessing import ColorBalancer
    return ColorBalancer()

@pytest.fixture
def preview_generator():
    """PreviewGenerator instance."""
    from astro_vision_composer.postprocessing import PreviewGenerator
    return PreviewGenerator()

@pytest.fixture
def pipeline():
    """ProcessingPipeline instance."""
    from astro_vision_composer import ProcessingPipeline
    return ProcessingPipeline()

# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
```

---

## Example Unit Tests

### `test_fits_loader.py`

```python
import pytest
import numpy as np

class TestFITSLoader:
    """Test FITSLoader with real NOIRLab data."""

    def test_load_single_fits(self, fits_loader, edu008_data):
        """Test loading a single FITS file."""
        fits_file = edu008_data['fits_files'][0]
        result = fits_loader.load(fits_file)

        assert result.data is not None
        assert result.data.ndim == 2
        assert result.metadata.filter is not None
        assert result.header is not None

    def test_load_multiple_fits(self, fits_loader, edu008_data):
        """Test loading all FITS files in a dataset."""
        results = [fits_loader.load(f) for f in edu008_data['fits_files']]

        assert len(results) == 3
        assert all(r.data is not None for r in results)
        assert all(r.metadata.wavelength is not None for r in results)

    def test_lazy_loading(self, fits_loader, edu008_data):
        """Test lazy loading mode."""
        fits_file = edu008_data['fits_files'][0]
        result = fits_loader.load(fits_file, lazy=True)

        assert result.data is not None
        # Lazy loading should still work, just potentially with memmap

    def test_invalid_file(self, fits_loader, tmp_path):
        """Test loading non-existent file raises error."""
        fake_file = tmp_path / "nonexistent.fits"

        with pytest.raises(FileNotFoundError):
            fits_loader.load(fake_file)

    def test_hst_format(self, fits_loader, edu011_data):
        """Test loading HST format FITS (multi-extension)."""
        fits_file = edu011_data['fits_files'][0]
        result = fits_loader.load(fits_file)

        # HST files should be correctly handled
        assert result.data is not None
        assert result.metadata.mission in ['HST', 'Hubble', None]
```

### `test_stretcher.py`

```python
class TestStretcher:
    """Test Stretcher with all methods including new additions."""

    def test_all_stretch_methods(self, stretcher):
        """Test all available stretch methods."""
        data = np.random.rand(100, 100)

        methods = ['linear', 'sqrt', 'squared', 'log', 'asinh', 'sinh',
                   'power', 'histeq', 'contrast_bias']

        for method in methods:
            if method == 'power':
                result = stretcher.stretch(data, method=method, power=2.0)
            elif method == 'histeq':
                result = stretcher.stretch(data, method=method)
            elif method == 'contrast_bias':
                result = stretcher.stretch(data, method=method, contrast=1.5, bias=0.5)
            elif method in ('asinh', 'sinh'):
                result = stretcher.stretch(data, method=method, a=0.1)
            else:
                result = stretcher.stretch(data, method=method)

            assert result is not None
            assert result.shape == data.shape
            assert np.all(np.isfinite(result))
            assert result.min() >= 0 and result.max() <= 1

    def test_histeq_with_2d_data(self, stretcher):
        """Test HistEq stretch handles 2D data correctly."""
        data = np.random.rand(200, 200)
        result = stretcher.stretch(data, method='histeq')

        assert result.shape == data.shape
        assert np.all(np.isfinite(result))
```

---

## Integration Tests

### `test_end_to_end_pipeline.py`

```python
import pytest
from pathlib import Path

class TestEndToEndPipeline:
    """Test complete pipeline with various datasets."""

    @pytest.mark.parametrize("dataset_fixture", [
        "edu008_data",  # 3 bands
        "edu010_data",  # 3 bands
    ])
    def test_3band_rgb_pipeline(self, dataset_fixture, pipeline, temp_output_dir, request):
        """Test end-to-end pipeline with 3-band datasets."""
        dataset = request.getfixturevalue(dataset_fixture)
        fits_files = dataset['fits_files']

        # Run pipeline
        rgb = pipeline.process_to_rgb(
            fits_files=fits_files,
            mode='scientific',
            output_dir=temp_output_dir
        )

        # Verify output
        assert rgb is not None
        assert rgb.shape == (rgb.shape[0], rgb.shape[1], 3)
        assert rgb.min() >= 0 and rgb.max() <= 1

        # Check output files were created
        assert (temp_output_dir / "rgb_composite_scientific.png").exists()

    def test_2band_dataset_handles_gracefully(self, edu012_data, pipeline, temp_output_dir):
        """Test pipeline handles 2-band datasets (can't make RGB)."""
        fits_files = edu012_data['fits_files']

        # Should raise error or handle gracefully
        with pytest.raises(ValueError, match="Need at least 3"):
            pipeline.process_to_rgb(
                fits_files=fits_files,
                mode='scientific',
                output_dir=temp_output_dir
            )

    def test_4band_dataset_selects_best_3(self, edu019_data, pipeline, temp_output_dir):
        """Test pipeline selects best 3 bands from 4-band dataset."""
        fits_files = edu019_data['fits_files']

        # Should automatically select best 3 bands
        rgb = pipeline.process_to_rgb(
            fits_files=fits_files,
            mode='scientific',
            output_dir=temp_output_dir
        )

        assert rgb is not None
        assert rgb.shape[2] == 3
```

---

## Parametrized Tests for All Datasets

### `test_noirlab_datasets.py`

```python
import pytest
from pathlib import Path

# Discover all NOIRLab datasets
def discover_noirlab_datasets(noirlab_dir):
    """Discover all available NOIRLab datasets."""
    datasets = []
    for edu_dir in sorted(noirlab_dir.glob("edu*")):
        data_dir = edu_dir / "data"
        if data_dir.exists():
            fits_files = list(data_dir.glob("**/*.fits"))
            # Filter out output directories
            fits_files = [f for f in fits_files if "output" not in str(f)]
            if fits_files:
                datasets.append({
                    'name': edu_dir.name,
                    'dir': data_dir,
                    'fits_files': fits_files,
                    'num_bands': len(fits_files)
                })
    return datasets

@pytest.fixture(scope="session")
def all_noirlab_datasets(noirlab_data_dir):
    """All available NOIRLab datasets."""
    return discover_noirlab_datasets(noirlab_data_dir)

class TestAllNOIRLabDatasets:
    """Regression tests across all NOIRLab datasets."""

    def test_all_datasets_load(self, all_noirlab_datasets, fits_loader):
        """Test that all datasets can be loaded."""
        for dataset in all_noirlab_datasets:
            for fits_file in dataset['fits_files']:
                try:
                    result = fits_loader.load(fits_file)
                    assert result.data is not None
                except Exception as e:
                    pytest.fail(f"Failed to load {dataset['name']}/{fits_file.name}: {e}")

    def test_metadata_extraction(self, all_noirlab_datasets, fits_loader):
        """Test metadata extraction for all datasets."""
        from astro_vision_composer.utilities import FITSMetadata

        metadata_extractor = FITSMetadata()

        for dataset in all_noirlab_datasets:
            for fits_file in dataset['fits_files']:
                result = fits_loader.load(fits_file)
                metadata = metadata_extractor.extract_metadata(result.header)

                # Should extract at least some metadata
                assert metadata.mission is not None or metadata.instrument is not None
```

---

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/astro_vision_composer --cov-report=html

# Run specific test file
pytest tests/unit/test_preprocessing/test_fits_loader.py

# Run specific test
pytest tests/unit/test_preprocessing/test_fits_loader.py::TestFITSLoader::test_load_single_fits

# Run tests matching pattern
pytest -k "test_histeq"

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Verbose output
pytest -v
```

### Test Markers

```python
# In conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "regression: marks tests as regression tests")

# In test files
@pytest.mark.slow
def test_large_dataset():
    ...

@pytest.mark.integration
def test_full_pipeline():
    ...
```

```bash
# Run only unit tests
pytest tests/unit/

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

---

## Continuous Integration (CI)

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[test]"

    - name: Run tests
      run: |
        pytest --cov=src/astro_vision_composer --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Test Coverage Goals

### Phase-wise Coverage Targets

- **Phase 1 (Preprocessing):** 80% coverage
- **Phase 2 (Processing):** 85% coverage
- **Phase 3 (Postprocessing):** 85% coverage
- **Phase 4 (Advanced Features):** 75% coverage

### Overall Target: 80% code coverage

---

## Summary

This pytest strategy provides:

✅ **Comprehensive fixture system** using real NOIRLab data
✅ **Unit tests** for every class and method
✅ **Integration tests** for multi-class workflows
✅ **Regression tests** across all available datasets
✅ **Edge case handling** (2 bands, 4+ bands, blank data)
✅ **Parametrized tests** to run same test across multiple datasets
✅ **CI/CD ready** with GitHub Actions integration
✅ **Coverage tracking** with detailed reports

**Estimated Test Count:** 200-300 tests covering all components
**Est. Time to Write:** 5-7 days for comprehensive coverage
**Est. Test Runtime:** < 5 minutes for full suite
