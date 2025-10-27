"""Integration tests for refactored ProcessingPipeline class.

Tests the new ImageNormalize integration, manual workflow mode,
and safety features added on 2025-10-26.
"""

import pytest
import numpy as np
from pathlib import Path
from astropy.visualization import (
    ImageNormalize, ZScaleInterval, PercentileInterval, MinMaxInterval,
    AsinhStretch, LogStretch, LinearStretch, SqrtStretch
)

from astro_vision_composer.pipeline import ProcessingPipeline


@pytest.mark.integration
@pytest.mark.requires_noirlab_data
class TestProcessingPipelineRefactored:
    """Test refactored ProcessingPipeline with ImageNormalize integration."""

    def test_scientific_workflow_with_image_normalize(self, edu008_data, temp_output_dir):
        """Test scientific workflow uses ImageNormalize correctly."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        # Initialize pipeline with scientific mode
        pipeline = ProcessingPipeline(mode='scientific')

        # Process files
        rgb = pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            output_dir=temp_output_dir
        )

        # Verify output
        assert rgb is not None
        assert rgb.shape[-1] == 3  # RGB channels
        assert 0 <= rgb.min() <= rgb.max() <= 1  # Valid range
        assert not np.any(np.isnan(rgb))  # No NaNs

        # Verify ImageNormalize was used in Phase 2
        assert pipeline.phase2_data is not None
        for band_name, band_data in pipeline.phase2_data.items():
            assert 'norm_object' in band_data
            assert band_data['norm_object'] is not None
            # Should have interval and stretch
            assert band_data['interval_object'] is not None
            assert band_data['stretch_object'] is not None

        # Verify outputs were created
        assert (temp_output_dir / f"rgb_composite_scientific.png").exists()
        assert (temp_output_dir / f"rgb_composite_scientific.tif").exists()

    def test_sdss_workflow_with_lupton_auto(self, edu008_data, temp_output_dir):
        """Test SDSS workflow uses Lupton auto-calculated stretch."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        pipeline = ProcessingPipeline(mode='sdss')
        rgb = pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            output_dir=temp_output_dir
        )

        assert rgb is not None
        assert rgb.shape[-1] == 3

        # Verify SDSS workflow - Phase 2 should have None stretch (Lupton handles it)
        for band_data in pipeline.phase2_data.values():
            # SDSS mode normalizes only, no stretch in Phase 2
            assert band_data['stretch_object'] is None or isinstance(band_data['stretch_object'], LinearStretch)

    def test_aesthetic_workflow(self, edu008_data, temp_output_dir):
        """Test aesthetic workflow with HistEq stretch."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        pipeline = ProcessingPipeline(mode='aesthetic')
        rgb = pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            output_dir=temp_output_dir
        )

        assert rgb is not None
        assert 0 <= rgb.min() <= rgb.max() <= 1

    def test_manual_workflow_with_normalizations(self, edu008_data, temp_output_dir):
        """Test manual workflow with explicit ImageNormalize per band."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        pipeline = ProcessingPipeline(mode='manual')

        # Create custom normalizations for each band
        # Simulate narrowband processing: Ha, OIII, SII
        normalizations = [
            ImageNormalize(interval=ZScaleInterval(), stretch=AsinhStretch(a=0.1)),      # Ha - asinh
            ImageNormalize(interval=PercentileInterval(99.5), stretch=LogStretch(a=1000)),  # OIII - log
            ImageNormalize(interval=PercentileInterval(98.0), stretch=SqrtStretch())     # SII - sqrt
        ]

        rgb = pipeline.process_with_normalizations(
            fits_files=edu008_data['fits_files'][:3],
            normalizations=normalizations,
            compositor='simple',
            output_dir=temp_output_dir
        )

        # Verify output
        assert rgb is not None
        assert rgb.shape[-1] == 3
        assert 0 <= rgb.min() <= rgb.max() <= 1

        # Verify custom normalizations were applied
        bands = list(pipeline.phase2_data.keys())
        assert len(bands) == 3

        # Check each band got its custom normalization
        for i, band_name in enumerate(bands):
            band_data = pipeline.phase2_data[band_name]
            assert band_data['norm_object'] == normalizations[i]

    def test_manual_workflow_with_arrays(self, edu008_data, temp_output_dir):
        """Test manual workflow with separate interval/stretch arrays."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        pipeline = ProcessingPipeline(mode='manual')

        # Separate arrays
        intervals = [
            ZScaleInterval(),
            PercentileInterval(99.5),
            MinMaxInterval()
        ]
        stretches = [
            AsinhStretch(a=0.1),
            LinearStretch(),
            SqrtStretch()
        ]

        rgb = pipeline.process_with_arrays(
            fits_files=edu008_data['fits_files'][:3],
            intervals=intervals,
            stretches=stretches,
            compositor='simple',
            output_dir=temp_output_dir
        )

        assert rgb is not None
        assert rgb.shape[-1] == 3
        assert 0 <= rgb.min() <= rgb.max() <= 1

    def test_explicit_lupton_workflows(self, edu008_data):
        """Test 3 explicit Lupton workflow types."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        # Workflow A: Pre-stretched
        pipeline = ProcessingPipeline(mode='scientific')
        rgb_pre = pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            lupton_workflow='pre_stretched'
        )
        assert rgb_pre is not None

        # Workflow B: SDSS auto
        pipeline = ProcessingPipeline(mode='sdss')
        rgb_auto = pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            lupton_workflow='sdss_auto'
        )
        assert rgb_auto is not None

        # Workflow C: Manual parameters
        pipeline = ProcessingPipeline(mode='scientific')
        rgb_manual = pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            lupton_workflow='manual'
        )
        assert rgb_manual is not None

        # All should produce valid RGB
        for rgb in [rgb_pre, rgb_auto, rgb_manual]:
            assert 0 <= rgb.min() <= rgb.max() <= 1

    def test_experimental_features_blocked_by_default(self):
        """Test that experimental features are blocked by default."""
        pipeline = ProcessingPipeline(mode='scientific')

        # Should not have experimental enabled
        assert pipeline.enable_experimental is False

        # Trying to use experimental method should raise error
        test_data = np.random.rand(100, 100)
        with pytest.raises(RuntimeError, match="experimental.*disabled"):
            pipeline.apply_enhancement(test_data, method='clahe')

    def test_experimental_features_with_permission(self):
        """Test experimental features work when explicitly enabled."""
        # Enable experimental features
        pipeline = ProcessingPipeline(mode='scientific', enable_experimental=True)

        assert pipeline.enable_experimental is True

        # Should be able to use CLAHE now (with warning)
        test_data = np.random.rand(100, 100)
        with pytest.warns(UserWarning, match="CLAHE.*low-quality"):
            enhanced = pipeline.apply_enhancement(test_data, method='clahe', clip_limit=0.01)

        assert enhanced is not None
        assert enhanced.shape == test_data.shape

    def test_safe_enhancement_always_works(self):
        """Test that safe enhancements work without experimental flag."""
        pipeline = ProcessingPipeline(mode='scientific')

        test_data = np.random.rand(100, 100)

        # Unsharp mask is safe
        enhanced = pipeline.apply_enhancement(test_data, method='unsharp_mask', sigma=2.0)
        assert enhanced is not None
        assert enhanced.shape == test_data.shape

    def test_custom_interval_stretch_parameters(self, edu008_data):
        """Test custom interval and stretch parameters."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        pipeline = ProcessingPipeline(mode='custom')

        # Use custom parameters
        rgb = pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            interval='percentile',  # String spec
            stretch='sqrt'          # String spec
        )

        assert rgb is not None
        assert 0 <= rgb.min() <= rgb.max() <= 1

    def test_processing_history_tracking(self, edu008_data, temp_output_dir):
        """Test that processing history is correctly tracked."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        pipeline = ProcessingPipeline(mode='scientific')
        pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            output_dir=temp_output_dir
        )

        # Check history
        history = pipeline.history.get_history()
        assert len(history) > 0

        # Should have load, normalize_stretch, and compose_rgb steps
        operations = [step.operation for step in history]
        assert 'load_fits' in operations
        assert 'normalize_stretch' in operations
        assert 'compose_rgb' in operations

    def test_invalid_inputs(self):
        """Test proper error handling for invalid inputs."""
        pipeline = ProcessingPipeline(mode='scientific')

        # Too few files
        with pytest.raises(ValueError, match="Need at least 3"):
            pipeline.process_to_rgb(fits_files=['file1.fits', 'file2.fits'])

        # Mismatched normalizations
        with pytest.raises(ValueError, match="must match"):
            pipeline.process_with_normalizations(
                fits_files=['a.fits', 'b.fits', 'c.fits'],
                normalizations=[ImageNormalize()]  # Only 1, need 3
            )

        # Mismatched interval/stretch arrays
        with pytest.raises(ValueError, match="same length"):
            pipeline.process_with_arrays(
                fits_files=['a.fits', 'b.fits', 'c.fits'],
                intervals=[ZScaleInterval()],  # Only 1
                stretches=[AsinhStretch(), LinearStretch()]  # 2 - mismatch
            )


@pytest.mark.integration
@pytest.mark.requires_noirlab_data
class TestBackwardCompatibility:
    """Ensure refactored pipeline maintains backward compatibility."""

    def test_original_workflow_still_works(self, edu008_data, temp_output_dir):
        """Test that code using old API still works."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        # Old-style usage (should still work)
        pipeline = ProcessingPipeline(mode='scientific')
        rgb = pipeline.process_to_rgb(
            fits_files=edu008_data['fits_files'][:3],
            output_dir=temp_output_dir
        )

        assert rgb is not None
        assert rgb.shape[-1] == 3

    def test_all_modes_still_work(self, edu008_data):
        """Test all workflow modes still function."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        modes = ['scientific', 'sdss', 'aesthetic', 'custom']

        for mode in modes:
            pipeline = ProcessingPipeline(mode=mode)
            rgb = pipeline.process_to_rgb(
                fits_files=edu008_data['fits_files'][:3]
            )
            assert rgb is not None, f"Mode {mode} failed"
            assert rgb.shape[-1] == 3, f"Mode {mode} produced wrong shape"
