"""Integration tests for Phase 2: Processing pipeline."""

import pytest


@pytest.mark.integration
@pytest.mark.requires_noirlab_data
class TestPhase2Processing:
    """Test Phase 2 processing workflow."""

    def test_normalize_and_stretch(self, fits_loader, normalizer, stretcher, edu008_data):
        """Test normalization and stretching workflow."""
        fits_file = edu008_data['fits_files'][0]

        # Load
        fits_data = fits_loader.load(fits_file)

        # Normalize
        normalized = normalizer.normalize(fits_data.data, method='zscale')

        assert 0 <= normalized.min() <= normalized.max() <= 1

        # Stretch
        stretched = stretcher.stretch(normalized, method='asinh', a=0.1)

        assert 0 <= stretched.min() <= stretched.max() <= 1

    def test_alignment_workflow(self, fits_loader, wcs_handler, reprojector, edu008_data):
        """Test WCS extraction and alignment workflow."""
        if len(edu008_data['fits_files']) < 2:
            pytest.skip("Need at least 2 FITS files")

        # Load two images
        fits1 = fits_loader.load(edu008_data['fits_files'][0])
        fits2 = fits_loader.load(edu008_data['fits_files'][1])

        # Extract WCS
        wcs1 = wcs_handler.extract_wcs(fits1.header)
        wcs2 = wcs_handler.extract_wcs(fits2.header)

        if wcs1 is None or wcs2 is None:
            pytest.skip("FITS files don't have WCS")

        if not wcs1.has_celestial:
            pytest.skip("WCS doesn't have celestial coordinates")

        # Reproject second image to first
        reprojected, footprint = reprojector.reproject(
            fits2.data, wcs2, wcs1, shape_out=fits1.data.shape
        )

        assert reprojected is not None
        assert reprojected.shape == fits1.data.shape

    def test_full_phase2_pipeline(self, fits_loader, normalizer, stretcher,
                                   enhancer, edu008_data):
        """Test complete Phase 2 pipeline."""
        results = []

        for fits_file in edu008_data['fits_files']:
            # Load
            fits_data = fits_loader.load(fits_file)

            # Normalize
            normalized = normalizer.normalize(fits_data.data, method='zscale')

            # Stretch
            stretched = stretcher.stretch(normalized, method='asinh', a=0.1)

            # Enhance
            enhanced = enhancer.apply_clahe(stretched, clip_limit=2.0)

            results.append({
                'file': fits_file.name,
                'normalized': normalized,
                'stretched': stretched,
                'enhanced': enhanced
            })

        # All files should complete pipeline
        assert len(results) == len(edu008_data['fits_files'])
        assert all(r['enhanced'] is not None for r in results)
        assert all(0 <= r['enhanced'].min() <= r['enhanced'].max() <= 1 for r in results)
