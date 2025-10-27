"""End-to-end integration tests for complete pipeline."""

import pytest


@pytest.mark.integration
@pytest.mark.requires_noirlab_data
@pytest.mark.slow
class TestEndToEndPipeline:
    """Test complete end-to-end processing pipeline."""

    def test_3band_rgb_pipeline(self, fits_loader, quality_assessor, normalizer,
                                 stretcher, channel_mapper, compositor,
                                 history_tracker, image_exporter,
                                 edu008_data, temp_output_dir):
        """Test complete pipeline from FITS to RGB PNG."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        history_tracker.record('load', {'dataset': 'edu008'})

        # Phase 1: Load and assess
        channels = {}
        for i, fits_file in enumerate(edu008_data['fits_files'][:3]):
            # Load
            fits_data = fits_loader.load(fits_file)
            history_tracker.record('load_band', {'file': fits_file.name})

            # Assess quality
            quality = quality_assessor.assess_quality(fits_data.data)
            history_tracker.record('quality_assessment', {
                'band': i,
                'snr': quality.snr,
                'saturation': quality.saturated_fraction
            })

            band_name = fits_data.metadata.filter or f'band_{i}'
            channels[band_name] = {
                'data': fits_data.data,
                'wavelength': fits_data.metadata.wavelength or 500 + i*100
            }

        # Phase 2: Normalize and stretch
        for name in channels:
            normalized = normalizer.normalize(channels[name]['data'], method='zscale')
            history_tracker.record('normalize', {'band': name, 'method': 'zscale'})

            stretched = stretcher.stretch(normalized, method='asinh', a=0.1)
            history_tracker.record('stretch', {'band': name, 'method': 'asinh'})

            channels[name]['data'] = stretched

        # Phase 3: Map and composite
        band_info = {name: {'wavelength': channels[name]['wavelength']}
                     for name in channels}
        mapping = channel_mapper.map_by_wavelength(band_info)
        history_tracker.record('channel_mapping', {'mapping': mapping})

        rgb = compositor.create_lupton_rgb(
            channels[mapping['red']]['data'],
            channels[mapping['green']]['data'],
            channels[mapping['blue']]['data'],
            stretch=0.5, Q=8
        )
        history_tracker.record('composite', {'algorithm': 'lupton', 'stretch': 0.5, 'Q': 8})

        # Export
        output_file = temp_output_dir / "end_to_end_composite.png"
        image_exporter.save(rgb, output_file, history=history_tracker.get_history())
        history_tracker.record('export', {'file': str(output_file)})

        # Verify
        assert output_file.exists()
        assert output_file.stat().st_size > 1000

        # Verify history
        history = history_tracker.get_history()
        assert len(history) > 10  # Multiple steps
        assert history[0]['operation'] == 'load'
        assert history[-1]['operation'] == 'export'

    def test_4band_dataset_pipeline(self, fits_loader, normalizer, stretcher,
                                     channel_mapper, compositor, edu019_data,
                                     temp_output_dir):
        """Test pipeline with 4-band dataset (selects best 3)."""
        if len(edu019_data['fits_files']) < 4:
            pytest.skip("Need 4-band dataset")

        # Load and process all 4 bands
        channels = {}
        for i, fits_file in enumerate(edu019_data['fits_files'][:4]):
            fits_data = fits_loader.load(fits_file)
            normalized = normalizer.normalize(fits_data.data, method='zscale')
            stretched = stretcher.stretch(normalized, method='asinh', a=0.1)

            band_name = fits_data.metadata.filter or f'band_{i}'
            channels[band_name] = {
                'data': stretched,
                'wavelength': fits_data.metadata.wavelength or 400 + i*150
            }

        # Map (should select best 3)
        band_info = {name: {'wavelength': channels[name]['wavelength']}
                     for name in channels}
        mapping = channel_mapper.map_by_wavelength(band_info, select_best=3)

        # Should have selected exactly 3 bands
        assert len(set(mapping.values())) == 3

        # Composite
        rgb = compositor.create_lupton_rgb(
            channels[mapping['red']]['data'],
            channels[mapping['green']]['data'],
            channels[mapping['blue']]['data']
        )

        assert rgb is not None

    def test_pipeline_with_enhancement(self, fits_loader, normalizer, stretcher,
                                        enhancer, compositor, channel_mapper,
                                        color_balancer, edu008_data):
        """Test pipeline with advanced enhancement."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        # Load and process with enhancement
        channels = {}
        for i, fits_file in enumerate(edu008_data['fits_files'][:3]):
            fits_data = fits_loader.load(fits_file)
            normalized = normalizer.normalize(fits_data.data, method='zscale')
            stretched = stretcher.stretch(normalized, method='asinh', a=0.1)

            # Enhance
            enhanced = enhancer.apply_clahe(stretched, clip_limit=2.0)

            band_name = f'band_{i}'
            channels[band_name] = {
                'data': enhanced,
                'wavelength': 500 + i*100
            }

        # Map and composite
        band_info = {name: {'wavelength': channels[name]['wavelength']}
                     for name in channels}
        mapping = channel_mapper.map_by_wavelength(band_info)

        rgb = compositor.create_lupton_rgb(
            channels[mapping['red']]['data'],
            channels[mapping['green']]['data'],
            channels[mapping['blue']]['data']
        )

        # Color balance
        final = color_balancer.white_balance(rgb)

        assert final is not None
        assert 0 <= final.min() <= final.max() <= 1
