"""Unit tests for Phase 4: Narrowband palette mapping and band selection.

Tests PaletteMapper, BandSelector, and enhanced ChannelMapper functionality.
"""

import pytest
import numpy as np

from astro_vision_composer.postprocessing import (
    ChannelMapper, ChannelMapping,
    PaletteMapper, PaletteMapping, ColorChannel,
    BandSelector, BandSelection, SelectionStrategy
)


class TestPaletteMapper:
    """Test PaletteMapper class for narrowband imaging."""

    def test_hubble_palette_mapping(self):
        """Test Hubble palette (SHO) mapping."""
        mapper = PaletteMapper()
        bands = ['sii', 'ha', 'oiii']

        mapping = mapper.map_narrowband(bands, palette='hubble')

        assert isinstance(mapping, PaletteMapping)
        assert mapping.red_band == 'sii'
        assert mapping.green_band == 'ha'
        assert mapping.blue_band == 'oiii'
        assert mapping.is_false_color is True

    def test_hoo_bicolor_palette(self):
        """Test HOO bicolor palette."""
        mapper = PaletteMapper()
        bands = ['ha', 'oiii']

        mapping = mapper.map_narrowband(bands, palette='hoo')

        assert isinstance(mapping, PaletteMapping)
        assert mapping.red_band == 'ha'
        assert mapping.green_band == 'oiii'  # Cyan synthesized
        assert mapping.blue_band == 'oiii'   # Cyan synthesized
        assert 'cyan' in mapping.synthetic_channels
        assert mapping.synthetic_channels['cyan'] == 'oiii'

    def test_case_insensitive_band_matching(self):
        """Test that band matching is case-insensitive."""
        mapper = PaletteMapper()

        # Test various case combinations
        test_cases = [
            ['SII', 'HA', 'OIII'],
            ['Sii', 'Ha', 'Oiii'],
            ['s2', 'halpha', 'o3'],
            ['sulfur', 'hydrogen', 'oxygen']
        ]

        for bands in test_cases:
            mapping = mapper.map_narrowband(bands, palette='hubble')
            assert mapping.red_band is not None
            assert mapping.green_band is not None
            assert mapping.blue_band is not None

    def test_custom_palette_dict(self):
        """Test custom palette definition."""
        mapper = PaletteMapper()

        custom_palette = {
            'ha': 'red',
            'oiii': 'green',
            'sii': 'blue'
        }

        mapping = mapper.map_narrowband(
            bands=['ha', 'oiii', 'sii'],
            palette=custom_palette
        )

        assert mapping.red_band == 'ha'
        assert mapping.green_band == 'oiii'
        assert mapping.blue_band == 'sii'
        assert mapping.palette_name == 'custom'

    def test_custom_palette_with_cyan(self):
        """Test custom palette with synthetic cyan channel."""
        mapper = PaletteMapper()

        custom_palette = {
            'ha': 'red',
            'oiii': 'cyan'  # Should create cyan from oiii
        }

        mapping = mapper.map_narrowband(
            bands=['ha', 'oiii'],
            palette=custom_palette
        )

        assert mapping.red_band == 'ha'
        assert mapping.green_band == 'oiii'
        assert mapping.blue_band == 'oiii'
        assert 'cyan' in mapping.synthetic_channels

    def test_wavelength_mapped_palette(self):
        """Test automatic wavelength-based mapping."""
        mapper = PaletteMapper()

        bands = ['ha', 'oiii', 'sii']
        wavelengths = {
            'ha': 656.3,
            'oiii': 500.7,
            'sii': 672.4
        }

        mapping = mapper.map_narrowband(
            bands=bands,
            palette='mapped',
            wavelengths=wavelengths
        )

        # Should be chromatic order: longest → red, shortest → blue
        assert mapping.red_band == 'sii'  # 672.4 nm (longest)
        assert mapping.blue_band == 'oiii'  # 500.7 nm (shortest)
        assert mapping.green_band == 'ha'  # 656.3 nm (middle)
        assert mapping.is_false_color is False

    def test_unknown_palette_raises_error(self):
        """Test that unknown palette names raise ValueError."""
        mapper = PaletteMapper()

        with pytest.raises(ValueError, match="Unknown palette"):
            mapper.map_narrowband(['ha', 'oiii'], palette='nonexistent')

    def test_list_palettes(self):
        """Test listing available palettes."""
        mapper = PaletteMapper()
        palettes = mapper.list_palettes()

        assert isinstance(palettes, dict)
        assert 'hubble' in palettes
        assert 'hoo' in palettes
        assert 'natural' in palettes
        assert 'mapped' in palettes


class TestBandSelector:
    """Test BandSelector for multi-band selection."""

    def test_max_span_selection(self):
        """Test maximum wavelength span selection."""
        selector = BandSelector()

        bands = ['f070w', 'f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
        wavelengths = {
            'f070w': 704,
            'f090w': 901,
            'f150w': 1501,
            'f200w': 1989,
            'f277w': 2762,
            'f356w': 3568,
            'f444w': 4421
        }

        selection = selector.select_bands(
            bands=bands,
            wavelengths=wavelengths,
            strategy='max_span'
        )

        assert isinstance(selection, BandSelection)
        assert selection.red_band == 'f444w'  # Longest wavelength
        assert selection.blue_band == 'f070w'  # Shortest wavelength
        assert selection.green_band in bands  # Middle wavelength
        assert selection.strategy == SelectionStrategy.MAX_SPAN

        # Check metadata
        assert 'wavelength_span_nm' in selection.metadata
        assert selection.metadata['wavelength_span_nm'] > 3000  # >3000 nm span

    def test_exactly_three_bands(self):
        """Test that exactly 3 bands skips selection."""
        selector = BandSelector()

        bands = ['r', 'g', 'b']
        wavelengths = {'r': 650, 'g': 550, 'b': 450}

        selection = selector.select_bands(
            bands=bands,
            wavelengths=wavelengths,
            strategy='max_span'
        )

        # Should return all 3 bands in wavelength order
        assert selection.red_band == 'r'
        assert selection.green_band == 'g'
        assert selection.blue_band == 'b'
        assert selection.strategy == SelectionStrategy.MANUAL

    def test_fewer_than_three_bands_raises_error(self):
        """Test that <3 bands raises ValueError."""
        selector = BandSelector()

        with pytest.raises(ValueError, match="Need at least 3 bands"):
            selector.select_bands(
                bands=['r', 'g'],
                wavelengths={'r': 650, 'g': 550},
                strategy='max_span'
            )

    def test_pca_selection_requires_data(self):
        """Test that PCA strategy requires data parameter."""
        selector = BandSelector()

        bands = ['r', 'g', 'b', 'i']
        wavelengths = {'r': 650, 'g': 550, 'b': 450, 'i': 800}

        with pytest.raises(ValueError, match="PCA strategy requires data"):
            selector.select_bands(
                bands=bands,
                wavelengths=wavelengths,
                strategy='pca'
            )

    def test_pca_selection_with_data(self):
        """Test PCA selection with synthetic data."""
        pytest.importorskip('sklearn')  # Skip if scikit-learn not installed

        selector = BandSelector()

        bands = ['r', 'g', 'b', 'i', 'z']
        wavelengths = {'r': 650, 'g': 550, 'b': 450, 'i': 800, 'z': 900}

        # Create synthetic data with some structure
        np.random.seed(42)
        data = {}
        for i, band in enumerate(bands):
            # Create correlated data with some variation
            base = np.random.rand(50, 50)
            data[band] = base + np.random.rand(50, 50) * (i + 1) * 0.1

        selection = selector.select_bands(
            bands=bands,
            wavelengths=wavelengths,
            strategy='pca',
            data=data
        )

        assert isinstance(selection, BandSelection)
        assert selection.strategy == SelectionStrategy.PCA
        assert 'variance_explained' in selection.metadata
        assert 0 < selection.metadata['variance_explained'] <= 1.0

    def test_science_driven_selection(self):
        """Test science-driven band selection."""
        selector = BandSelector()

        # Optical bands suitable for stellar population
        bands = ['u', 'g', 'r', 'i', 'z']
        wavelengths = {'u': 354, 'g': 477, 'r': 623, 'i': 762, 'z': 913}

        selection = selector.select_bands(
            bands=bands,
            wavelengths=wavelengths,
            strategy='science',
            science_goal='stellar_population'
        )

        assert isinstance(selection, BandSelection)
        assert selection.strategy == SelectionStrategy.SCIENCE
        assert 'science_goal' in selection.metadata
        assert selection.metadata['science_goal'] == 'stellar_population'

    def test_list_science_goals(self):
        """Test listing available science goals."""
        selector = BandSelector()
        goals = selector.list_science_goals()

        assert isinstance(goals, dict)
        assert 'star_formation' in goals
        assert 'stellar_population' in goals


class TestChannelMapperIntegration:
    """Test integrated ChannelMapper with palette and selection."""

    def test_map_narrowband_integration(self):
        """Test ChannelMapper.map_narrowband method."""
        mapper = ChannelMapper()

        bands = ['sii', 'ha', 'oiii']
        mapping = mapper.map_narrowband(bands, palette='hubble')

        # Should return ChannelMapping (converted from PaletteMapping)
        assert isinstance(mapping, ChannelMapping)
        assert mapping.red == 'sii'
        assert mapping.green == 'ha'
        assert mapping.blue == 'oiii'

    def test_select_and_map_integration(self):
        """Test ChannelMapper.select_and_map method."""
        mapper = ChannelMapper()

        bands = ['f070w', 'f150w', 'f200w', 'f356w', 'f444w']
        wavelengths = {
            'f070w': 704,
            'f150w': 1501,
            'f200w': 1989,
            'f356w': 3568,
            'f444w': 4421
        }

        mapping = mapper.select_and_map(
            bands=bands,
            wavelengths=wavelengths,
            strategy='max_span'
        )

        assert isinstance(mapping, ChannelMapping)
        assert mapping.red is not None
        assert mapping.green is not None
        assert mapping.blue is not None

    def test_list_narrowband_palettes(self):
        """Test listing palettes through ChannelMapper."""
        mapper = ChannelMapper()
        palettes = mapper.list_narrowband_palettes()

        assert isinstance(palettes, dict)
        assert len(palettes) >= 3  # At least hubble, hoo, natural

    def test_list_science_goals_through_mapper(self):
        """Test listing science goals through ChannelMapper."""
        mapper = ChannelMapper()
        goals = mapper.list_science_goals()

        assert isinstance(goals, dict)
        assert len(goals) >= 3  # At least a few science goals


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
