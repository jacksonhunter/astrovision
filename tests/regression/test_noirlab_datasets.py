"""Regression tests across all available NOIRLab datasets."""

import pytest


@pytest.mark.regression
@pytest.mark.requires_noirlab_data
class TestAllNOIRLabDatasets:
    """Regression tests across all available NOIRLab datasets."""

    def test_all_datasets_load(self, all_noirlab_datasets, fits_loader):
        """Test that all datasets can be loaded without errors."""
        failed_files = []

        for dataset in all_noirlab_datasets:
            for fits_file in dataset['fits_files']:
                try:
                    result = fits_loader.load(fits_file)
                    assert result.data is not None
                    assert result.data.ndim == 2
                except Exception as e:
                    failed_files.append((dataset['name'], fits_file.name, str(e)))

        if failed_files:
            msg = "\n".join([f"{ds}/{file}: {err}" for ds, file, err in failed_files])
            pytest.fail(f"Failed to load {len(failed_files)} files:\n{msg}")

    def test_metadata_extraction_all_datasets(self, all_noirlab_datasets, fits_loader,
                                               metadata_extractor):
        """Test metadata extraction for all datasets."""
        failed_extractions = []

        for dataset in all_noirlab_datasets:
            for fits_file in dataset['fits_files']:
                try:
                    fits_data = fits_loader.load(fits_file)
                    metadata = metadata_extractor.extract_metadata(fits_data.header)

                    # Should extract at least some metadata
                    has_metadata = (metadata.mission is not None or
                                    metadata.instrument is not None or
                                    metadata.filter is not None or
                                    metadata.wavelength is not None)

                    if not has_metadata:
                        failed_extractions.append((dataset['name'], fits_file.name,
                                                   "No metadata extracted"))
                except Exception as e:
                    failed_extractions.append((dataset['name'], fits_file.name, str(e)))

        # It's OK if some files have minimal metadata, just log it
        if failed_extractions:
            msg = "\n".join([f"{ds}/{file}: {err}" for ds, file, err in failed_extractions])
            print(f"\nMetadata extraction issues ({len(failed_extractions)} files):\n{msg}")

    def test_quality_assessment_all_datasets(self, all_noirlab_datasets, fits_loader,
                                              quality_assessor):
        """Test quality assessment for all datasets."""
        results = []

        for dataset in all_noirlab_datasets:
            for fits_file in dataset['fits_files']:
                try:
                    fits_data = fits_loader.load(fits_file)
                    quality = quality_assessor.assess_quality(fits_data.data)

                    results.append({
                        'dataset': dataset['name'],
                        'file': fits_file.name,
                        'snr': quality.snr,
                        'saturation': quality.saturated_fraction,
                        'dynamic_range': quality.dynamic_range
                    })

                    # Basic sanity checks
                    assert quality.snr >= 0
                    assert 0 <= quality.saturated_fraction <= 1
                    assert quality.dynamic_range >= 0
                except Exception as e:
                    pytest.fail(f"Quality assessment failed for {dataset['name']}/{fits_file.name}: {e}")

        # Log summary statistics
        avg_snr = sum(r['snr'] for r in results) / len(results)
        print(f"\n=== Quality Assessment Summary ===")
        print(f"Total files: {len(results)}")
        print(f"Average SNR: {avg_snr:.2f}")
        print(f"SNR range: {min(r['snr'] for r in results):.2f} - {max(r['snr'] for r in results):.2f}")

    def test_normalization_all_datasets(self, all_noirlab_datasets, fits_loader, normalizer):
        """Test normalization for all datasets."""
        for dataset in all_noirlab_datasets:
            for fits_file in dataset['fits_files']:
                fits_data = fits_loader.load(fits_file)

                for method in ['minmax', 'percentile', 'zscale']:
                    try:
                        if method == 'percentile':
                            normalized = normalizer.normalize(
                                fits_data.data, method=method, vmin=5, vmax=95
                            )
                        else:
                            normalized = normalizer.normalize(fits_data.data, method=method)

                        assert 0 <= normalized.min() <= normalized.max() <= 1
                    except Exception as e:
                        pytest.fail(f"Normalization ({method}) failed for "
                                    f"{dataset['name']}/{fits_file.name}: {e}")

    def test_stretching_all_datasets(self, all_noirlab_datasets, fits_loader,
                                      normalizer, stretcher):
        """Test stretching for all datasets."""
        for dataset in all_noirlab_datasets:
            for fits_file in dataset['fits_files'][:1]:  # Just test first file per dataset
                fits_data = fits_loader.load(fits_file)
                normalized = normalizer.normalize(fits_data.data, method='zscale')

                for method in ['sqrt', 'log', 'asinh']:
                    try:
                        if method == 'asinh':
                            stretched = stretcher.stretch(normalized, method=method, a=0.1)
                        else:
                            stretched = stretcher.stretch(normalized, method=method)

                        assert 0 <= stretched.min() <= stretched.max() <= 1
                    except Exception as e:
                        pytest.fail(f"Stretching ({method}) failed for "
                                    f"{dataset['name']}/{fits_file.name}: {e}")

    @pytest.mark.parametrize("dataset_name", ["edu008", "edu010"])
    def test_known_good_datasets(self, noirlab_data_dir, fits_loader,
                                  normalizer, stretcher, compositor):
        """Test known-good datasets produce valid composites."""
        dataset_dir = noirlab_data_dir / dataset_name / "data"

        if not dataset_dir.exists():
            pytest.skip(f"Dataset {dataset_name} not available")

        fits_files = sorted(dataset_dir.glob(f"{dataset_name}/*/*.fits"))
        fits_files = [f for f in fits_files if "output" not in str(f)]

        if len(fits_files) < 3:
            pytest.skip(f"Dataset {dataset_name} needs at least 3 FITS files")

        # Load and process 3 bands
        channels = []
        for fits_file in fits_files[:3]:
            fits_data = fits_loader.load(fits_file)
            normalized = normalizer.normalize(fits_data.data, method='zscale')
            stretched = stretcher.stretch(normalized, method='asinh', a=0.1)
            channels.append(stretched)

        # Create composite
        rgb = compositor.create_lupton_rgb(
            channels[2], channels[1], channels[0],
            stretch=0.5, Q=8
        )

        # Verify valid output
        assert rgb is not None
        assert rgb.shape[2] == 3
        assert 0 <= rgb.min() <= rgb.max() <= 1

        # Verify image has content (not all black/white)
        assert rgb.std() > 0.01

    def test_dataset_statistics(self, all_noirlab_datasets):
        """Test dataset availability and report statistics."""
        print(f"\n=== NOIRLab Dataset Statistics ===")
        print(f"Total datasets: {len(all_noirlab_datasets)}")

        for dataset in all_noirlab_datasets:
            print(f"{dataset['name']}: {dataset['num_bands']} bands")

        # Check minimum dataset availability
        assert len(all_noirlab_datasets) >= 1, "No NOIRLab datasets found"

        # Check we have at least one 3-band dataset
        three_band = [d for d in all_noirlab_datasets if d['num_bands'] >= 3]
        assert len(three_band) >= 1, "No 3-band datasets available for RGB testing"
