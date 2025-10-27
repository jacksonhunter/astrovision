"""Pytest configuration and fixtures for astro_vision_composer tests."""

import pytest
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def noirlab_data_dir():
    """Root directory for NOIRLab test data."""
    base_dir = Path(__file__).parent.parent / "examples" / "data" / "NOIRLab" / "examples"
    if not base_dir.exists():
        pytest.skip(f"NOIRLab test data not found at {base_dir}")
    return base_dir


@pytest.fixture(scope="session")
def edu008_data(noirlab_data_dir):
    """Eagle Nebula dataset (3 FITS files) - Primary test dataset."""
    data_dir = noirlab_data_dir / "edu008" / "data"
    if not data_dir.exists():
        pytest.skip(f"edu008 dataset not found at {data_dir}")

    fits_files = sorted(data_dir.glob("edu008/*/*.fits"))
    # Filter out processed files from output directories
    fits_files = [f for f in fits_files if "output" not in str(f)]

    if len(fits_files) < 3:
        pytest.skip(f"edu008 dataset needs at least 3 FITS files, found {len(fits_files)}")

    return {
        'dir': data_dir,
        'fits_files': fits_files[:3],  # Take first 3
        'num_bands': 3,
        'expected_wavelengths': [502, 656, 673]  # nm
    }


@pytest.fixture(scope="session")
def edu010_data(noirlab_data_dir):
    """M17 Nebula dataset (3 FITS files)."""
    data_dir = noirlab_data_dir / "edu010" / "data"
    if not data_dir.exists():
        pytest.skip(f"edu010 dataset not found at {data_dir}")

    fits_files = sorted(data_dir.glob("edu010/*/*.fits"))
    fits_files = [f for f in fits_files if "output" not in str(f)]

    return {
        'dir': data_dir,
        'fits_files': fits_files,
        'num_bands': len(fits_files)
    }


@pytest.fixture(scope="session")
def edu011_data(noirlab_data_dir):
    """HST format dataset."""
    data_dir = noirlab_data_dir / "edu011" / "data"
    if not data_dir.exists():
        pytest.skip(f"edu011 dataset not found at {data_dir}")

    fits_files = sorted(data_dir.glob("edu011/*/*.fits"))
    fits_files = [f for f in fits_files if "output" not in str(f)]

    return {
        'dir': data_dir,
        'fits_files': fits_files,
        'num_bands': len(fits_files)
    }


@pytest.fixture(scope="session")
def edu012_data(noirlab_data_dir):
    """Dataset with only 2 FITS files (edge case)."""
    data_dir = noirlab_data_dir / "edu012" / "data"
    if not data_dir.exists():
        pytest.skip(f"edu012 dataset not found at {data_dir}")

    fits_files = sorted(data_dir.glob("edu012/*/*.fits"))
    fits_files = [f for f in fits_files if "output" not in str(f)]

    return {
        'dir': data_dir,
        'fits_files': fits_files,
        'num_bands': 2
    }


@pytest.fixture(scope="session")
def edu019_data(noirlab_data_dir):
    """Dataset with 4 FITS files (edge case)."""
    data_dir = noirlab_data_dir / "edu019" / "data"
    if not data_dir.exists():
        pytest.skip(f"edu019 dataset not found at {data_dir}")

    fits_files = sorted(data_dir.glob("edu019/*/*.fits"))
    fits_files = [f for f in fits_files if "output" not in str(f)]

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
    header['NAXIS'] = 2
    header['NAXIS1'] = 100
    header['NAXIS2'] = 100
    return data, header


@pytest.fixture(scope="function")
def sample_fits_with_wcs():
    """Generate synthetic FITS data with WCS."""
    data = np.random.rand(100, 100).astype(np.float32)

    # Create a simple WCS
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50, 50]
    wcs.wcs.crval = [150.0, 2.5]  # RA, Dec in degrees
    wcs.wcs.cdelt = [0.0002778, 0.0002778]  # 1 arcsec/pixel
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    header = wcs.to_header()
    header['TELESCOP'] = 'TEST'
    header['INSTRUME'] = 'TESTCAM'
    header['FILTER'] = 'TEST_FILTER'
    header['EXPTIME'] = 60.0

    return data, header


@pytest.fixture(scope="function")
def sample_rgb_data():
    """Generate synthetic RGB image data for compositor tests."""
    r = np.random.rand(100, 100).astype(np.float32)
    g = np.random.rand(100, 100).astype(np.float32)
    b = np.random.rand(100, 100).astype(np.float32)
    return r, g, b


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
def calibrator():
    """Calibrator instance."""
    from astro_vision_composer.preprocessing import Calibrator
    return Calibrator()


@pytest.fixture
def wcs_handler():
    """WCSHandler instance."""
    from astro_vision_composer.processing import WCSHandler
    return WCSHandler()


@pytest.fixture
def reprojector():
    """Reprojector instance."""
    from astro_vision_composer.processing import Reprojector
    return Reprojector()


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
def history_tracker():
    """HistoryTracker instance."""
    from astro_vision_composer.postprocessing import HistoryTracker
    return HistoryTracker()


@pytest.fixture
def image_exporter():
    """ImageExporter instance."""
    from astro_vision_composer.postprocessing import ImageExporter
    return ImageExporter()


@pytest.fixture
def metadata_extractor():
    """FITSMetadata instance."""
    from astro_vision_composer.utilities import FITSMetadata
    return FITSMetadata()


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def temp_fits_file(tmp_path, sample_fits_data):
    """Create a temporary FITS file for testing."""
    data, header = sample_fits_data
    fits_file = tmp_path / "test.fits"
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fits_file, overwrite=True)
    return fits_file


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "regression: marks tests as regression tests")
    config.addinivalue_line("markers", "requires_noirlab_data: marks tests that require NOIRLab test data")

    # Initialize quality scores dictionary
    config._quality_scores = []


# ============================================================================
# Quality Assessment Scoring System (for passed tests)
# ============================================================================

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to collect quality scores from passed tests.

    This hook runs after each test and collects quality assessment scores
    for diagnostic purposes. Quality scores do NOT affect pass/fail status.
    """
    outcome = yield
    report = outcome.get_result()

    # Only process tests that passed
    if report.when == "call" and report.outcome == "passed":
        # Check if test generated any quality scores
        if hasattr(item, '_quality_scores'):
            for score in item._quality_scores:
                item.config._quality_scores.append({
                    'test': item.nodeid,
                    'score_type': score['type'],
                    'score_value': score['value'],
                    'metadata': score.get('metadata', {})
                })


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Display quality assessment summary after test run.

    Shows aggregate quality scores from all passed tests.
    """
    if not hasattr(config, '_quality_scores') or not config._quality_scores:
        return

    scores = config._quality_scores

    terminalreporter.section("Quality Assessment Scores (Passed Tests)")

    # Group scores by type
    score_types = {}
    for score in scores:
        score_type = score['score_type']
        if score_type not in score_types:
            score_types[score_type] = []
        score_types[score_type].append(score['score_value'])

    # Display aggregate statistics
    for score_type, values in score_types.items():
        if not values:
            continue

        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)

        terminalreporter.write_line(
            f"  {score_type}: avg={avg:.2f}, min={min_val:.2f}, max={max_val:.2f}, count={len(values)}"
        )

    # Display top 5 and bottom 5 scores if we have SNR data
    snr_scores = [(s['test'], s['score_value']) for s in scores if s['score_type'] == 'snr']
    if snr_scores:
        terminalreporter.write_line("")
        terminalreporter.write_line("  Top 5 SNR Scores:")
        for test, snr in sorted(snr_scores, key=lambda x: x[1], reverse=True)[:5]:
            terminalreporter.write_line(f"    {snr:.2f} - {test}")

        terminalreporter.write_line("")
        terminalreporter.write_line("  Bottom 5 SNR Scores:")
        for test, snr in sorted(snr_scores, key=lambda x: x[1])[:5]:
            terminalreporter.write_line(f"    {snr:.2f} - {test}")


@pytest.fixture
def record_quality_score(request):
    """Fixture to record quality scores for a test.

    Usage in tests:
        def test_something(record_quality_score):
            # ... test code ...
            record_quality_score('snr', 42.5, {'band': 'g'})
    """
    def _record(score_type, value, metadata=None):
        """Record a quality score.

        Args:
            score_type: Type of score (e.g., 'snr', 'saturation', 'dynamic_range')
            value: Numeric score value
            metadata: Optional dict with additional context
        """
        if not hasattr(request.node, '_quality_scores'):
            request.node._quality_scores = []

        request.node._quality_scores.append({
            'type': score_type,
            'value': value,
            'metadata': metadata or {}
        })

    return _record


# ============================================================================
# Utility Functions for Tests
# ============================================================================

def discover_noirlab_datasets(noirlab_dir):
    """Discover all available NOIRLab datasets.

    Args:
        noirlab_dir: Path to NOIRLab examples directory

    Returns:
        List of dataset dictionaries with metadata
    """
    datasets = []
    for edu_dir in sorted(noirlab_dir.glob("edu*")):
        data_dir = edu_dir / "data"
        if data_dir.exists():
            # Look for FITS files in subdirectories
            fits_files = list(data_dir.glob(f"{edu_dir.name}/*/*.fits"))
            # Filter out output directories
            fits_files = [f for f in fits_files if "output" not in str(f)]
            if fits_files:
                datasets.append({
                    'name': edu_dir.name,
                    'dir': data_dir,
                    'fits_files': sorted(fits_files),
                    'num_bands': len(fits_files)
                })
    return datasets


@pytest.fixture(scope="session")
def all_noirlab_datasets(noirlab_data_dir):
    """All available NOIRLab datasets."""
    return discover_noirlab_datasets(noirlab_data_dir)
