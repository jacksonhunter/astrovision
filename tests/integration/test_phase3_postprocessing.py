"""Integration tests for Phase 3: Postprocessing pipeline."""

import pytest


@pytest.mark.integration
@pytest.mark.requires_noirlab_data
class TestPhase3Postprocessing:
    """Test Phase 3 postprocessing workflow."""

    def test_channel_mapping_and_composition(self, fits_loader, normalizer, stretcher,
                                              channel_mapper, compositor, edu008_data):
        """Test channel mapping and RGB composition workflow."""
        if len(edu008_data['fits_files']) < 3:
            pytest.skip("Need at least 3 FITS files")

        # Load and process 3 bands
        channels = {}
        for i, fits_file in enumerate(edu008_data['fits_files'][:3]):
            fits_data = fits_loader.load(fits_file)
            normalized = normalizer.normalize(fits_data.science, method='zscale')
            stretched = stretcher.stretch(normalized, method='asinh', a=0.1)

            band_name = fits_data.metadata.filter_name or f'band_{i}'
            channels[band_name] = {
                'data': stretched,
                'wavelength': fits_data.metadata.wavelength or 500 + i*100
            }

        # Map channels to RGB
        band_names = list(channels.keys())
        band_info = {name: {'wavelength': channels[name]['wavelength']}
                     for name in band_names}

        mapping = channel_mapper.auto_map_by_wavelength(band_info)

        # Create composite
        rgb = compositor.create_lupton_rgb(
            channels[mapping['red']]['data'],
            channels[mapping['green']]['data'],
            channels[mapping['blue']]['data'],
            stretch=0.5, Q=8
        )

        assert rgb is not None
        assert rgb.shape[2] == 3
        assert 0 <= rgb.min() <= rgb.max() <= 1

    def test_composition_and_export(self, compositor, image_exporter,
                                     history_tracker, temp_output_dir, sample_rgb_data):
        """Test composition, history tracking, and export workflow."""
        r, g, b = sample_rgb_data

        # Track history
        history_tracker.record('normalize', {'method': 'zscale'}, 'Normalizer')
        history_tracker.record('stretch', {'method': 'asinh'}, 'Stretcher')
        history_tracker.record('composite', {'algorithm': 'lupton'}, 'Compositor')

        # Composite
        rgb = compositor.create_lupton_rgb(r, g, b, stretch=0.5, Q=8)

        # Export with history
        output_file = temp_output_dir / "composite.png"
        image_exporter.auto_save(rgb, output_file, history=history_tracker.get_history())

        assert output_file.exists()

    def test_color_balancing_workflow(self, compositor, color_balancer,
                                       sample_rgb_data):
        """Test composition and color balancing workflow."""
        r, g, b = sample_rgb_data

        # Composite
        rgb = compositor.create_simple_rgb(r, g, b)

        # Balance
        balanced = color_balancer.white_balance(rgb)

        # Adjust saturation
        final = color_balancer.adjust_saturation(balanced, factor=1.2)

        assert final is not None
        assert 0 <= final.min() <= final.max() <= 1

    def test_preview_generation_workflow(self, compositor, preview_generator,
                                          sample_rgb_data, temp_output_dir):
        """Test composition and preview generation workflow."""
        r, g, b = sample_rgb_data

        # Composite
        rgb = compositor.create_lupton_rgb(r, g, b)

        # Generate preview
        preview = preview_generator.generate_preview(rgb, target_size=(512, 512))

        assert preview is not None
        assert preview.shape[0] <= 512 and preview.shape[1] <= 512

        # Save preview
        preview_file = temp_output_dir / "preview.png"
        preview_generator.save_thumbnail(rgb, preview_file, size=(256, 256))

        assert preview_file.exists()
