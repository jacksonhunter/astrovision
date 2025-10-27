"""Integration tests for Phase 1: Preprocessing pipeline."""

import pytest


@pytest.mark.integration
@pytest.mark.requires_noirlab_data
class TestPhase1Preprocessing:
    """Test Phase 1 preprocessing workflow."""

    def test_load_and_assess_quality(self, fits_loader, quality_assessor, edu008_data):
        """Test loading FITS and assessing quality."""
        fits_file = edu008_data['fits_files'][0]

        # Load
        fits_data = fits_loader.load(fits_file)
        assert fits_data.data is not None

        # Assess quality
        quality = quality_assessor.assess_quality(fits_data.data)
        assert quality.snr > 0
        assert quality.dynamic_range > 0

    def test_load_multiple_bands(self, fits_loader, quality_assessor, edu008_data):
        """Test loading and assessing all bands in dataset."""
        results = []

        for fits_file in edu008_data['fits_files']:
            fits_data = fits_loader.load(fits_file)
            quality = quality_assessor.assess_quality(fits_data.data)

            results.append({
                'file': fits_file.name,
                'filter': fits_data.metadata.filter,
                'wavelength': fits_data.metadata.wavelength,
                'snr': quality.snr,
                'saturation': quality.saturated_fraction
            })

        # All bands should load and have quality metrics
        assert len(results) == len(edu008_data['fits_files'])
        assert all(r['snr'] > 0 for r in results)

    def test_metadata_extraction_workflow(self, fits_loader, metadata_extractor, edu008_data):
        """Test metadata extraction workflow."""
        fits_file = edu008_data['fits_files'][0]

        # Load FITS
        fits_data = fits_loader.load(fits_file)

        # Extract metadata
        metadata = metadata_extractor.extract_metadata(fits_data.header)

        # Verify metadata is useful
        assert metadata.filter is not None or metadata.wavelength is not None

    def test_calibration_workflow(self, fits_loader, calibrator, edu008_data):
        """Test calibration workflow (background subtraction)."""
        fits_file = edu008_data['fits_files'][0]

        # Load
        fits_data = fits_loader.load(fits_file)

        # Calibrate (background subtraction)
        calibrated = calibrator.subtract_background(fits_data.data)

        assert calibrated is not None
        assert calibrated.shape == fits_data.data.shape
        # Background should be removed
        assert calibrated.median() < fits_data.data.median()

    def test_full_phase1_pipeline(self, fits_loader, quality_assessor,
                                   calibrator, edu008_data):
        """Test complete Phase 1 pipeline."""
        results = []

        for fits_file in edu008_data['fits_files']:
            # Load
            fits_data = fits_loader.load(fits_file)

            # Assess quality
            quality_before = quality_assessor.assess_quality(fits_data.data)

            # Calibrate
            calibrated = calibrator.subtract_background(fits_data.data)

            # Assess quality after calibration
            quality_after = quality_assessor.assess_quality(calibrated)

            results.append({
                'file': fits_file.name,
                'snr_before': quality_before.snr,
                'snr_after': quality_after.snr,
                'calibrated': calibrated
            })

        # All files should complete pipeline
        assert len(results) == len(edu008_data['fits_files'])
        assert all(r['calibrated'] is not None for r in results)
