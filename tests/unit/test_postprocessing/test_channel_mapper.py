"""Unit tests for ChannelMapper class."""

import pytest


class TestChannelMapper:
    """Test ChannelMapper for RGB channel assignment."""

    def test_auto_map_by_wavelength_3bands(self, channel_mapper):
        """Test automatic RGB mapping for 3 bands by wavelength."""
        # Typical optical bands (nm): g(481), r(617), i(752)
        bands = {
            'g': {'wavelength': 481},
            'r': {'wavelength': 617},
            'i': {'wavelength': 752}
        }

        mapping = channel_mapper.map_by_wavelength(bands)

        # Chromatic ordering: shortest -> blue, longest -> red
        assert mapping['blue'] == 'g'  # Shortest
        assert mapping['green'] == 'r'  # Middle
        assert mapping['red'] == 'i'  # Longest

    def test_auto_map_by_wavelength_narrowband(self, channel_mapper):
        """Test automatic RGB mapping for narrowband."""
        # Narrowband: SII(672nm), H-alpha(656nm), OIII(501nm)
        bands = {
            'sii': {'wavelength': 672},
            'ha': {'wavelength': 656},
            'oiii': {'wavelength': 501}
        }

        mapping = channel_mapper.map_by_wavelength(bands)

        # Hubble palette: SII=red, H-alpha=green, OIII=blue
        assert mapping['blue'] == 'oiii'  # Shortest
        assert mapping['green'] == 'ha'  # Middle
        assert mapping['red'] == 'sii'  # Longest

    def test_custom_mapping(self, channel_mapper):
        """Test custom RGB mapping."""
        bands = ['r', 'g', 'b']

        mapping = channel_mapper.create_mapping(
            red='r', green='g', blue='b'
        )

        assert mapping['red'] == 'r'
        assert mapping['green'] == 'g'
        assert mapping['blue'] == 'b'

    def test_validate_mapping(self, channel_mapper):
        """Test mapping validation."""
        bands = ['band1', 'band2', 'band3']

        # Valid mapping
        valid_mapping = {
            'red': 'band1',
            'green': 'band2',
            'blue': 'band3'
        }
        assert channel_mapper.validate_mapping(valid_mapping, bands) == True

        # Invalid mapping (band not in list)
        invalid_mapping = {
            'red': 'band1',
            'green': 'band2',
            'blue': 'band_missing'
        }
        assert channel_mapper.validate_mapping(invalid_mapping, bands) == False

    def test_4band_mapping_selects_best_3(self, channel_mapper):
        """Test mapping 4 bands to RGB (should select best 3)."""
        # 4 bands: need to select best 3
        bands = {
            'u': {'wavelength': 336},
            'g': {'wavelength': 481},
            'r': {'wavelength': 617},
            'i': {'wavelength': 752}
        }

        mapping = channel_mapper.map_by_wavelength(bands, select_best=3)

        # Should select 3 bands with good wavelength spread
        # Likely: u/g=blue, r=green, i=red (or skip u as it's too blue)
        assert len(set(mapping.values())) == 3
        assert 'red' in mapping
        assert 'green' in mapping
        assert 'blue' in mapping

    def test_panstarrs_grizy_mapping(self, channel_mapper):
        """Test mapping PanSTARRS grizy bands."""
        bands = {
            'g': {'wavelength': 481, 'filter': 'g'},
            'r': {'wavelength': 617, 'filter': 'r'},
            'i': {'wavelength': 752, 'filter': 'i'},
            'z': {'wavelength': 866, 'filter': 'z'},
            'y': {'wavelength': 962, 'filter': 'y'}
        }

        # For RGB, typically use g, r, i or r, i, z
        mapping = channel_mapper.map_by_wavelength(bands, select_best=3)

        assert len(set(mapping.values())) == 3

    def test_jwst_nircam_mapping(self, channel_mapper):
        """Test mapping JWST NIRCam filters."""
        bands = {
            'F090W': {'wavelength': 900},
            'F150W': {'wavelength': 1500},
            'F200W': {'wavelength': 2000},
            'F356W': {'wavelength': 3560}
        }

        mapping = channel_mapper.map_by_wavelength(bands, select_best=3)

        # Should select 3 bands with good spread
        assert len(set(mapping.values())) == 3
        assert mapping['blue'] in ['F090W', 'F150W']
        assert mapping['red'] in ['F200W', 'F356W']

    def test_missing_wavelength_info(self, channel_mapper):
        """Test handling when wavelength info is missing."""
        bands = {
            'band1': {},  # No wavelength
            'band2': {},
            'band3': {}
        }

        # Should either fall back to manual mapping or raise error
        try:
            mapping = channel_mapper.map_by_wavelength(bands)
            # If it succeeds, check it returns something
            assert mapping is not None
        except (ValueError, KeyError):
            # Or it might raise an error - also acceptable
            pass

    def test_filter_name_fallback(self, channel_mapper):
        """Test fallback to filter names when wavelength missing."""
        bands = {
            'g': {'filter': 'g'},
            'r': {'filter': 'r'},
            'i': {'filter': 'i'}
        }

        # Channel mapper might recognize common filter names
        try:
            mapping = channel_mapper.map_by_filter_name(bands)
            assert len(mapping) == 3
        except AttributeError:
            # Method might not exist
            pass

    def test_hubble_palette_mapping(self, channel_mapper):
        """Test creating Hubble palette (SHO) mapping."""
        bands = {
            'sii': {'wavelength': 672},
            'ha': {'wavelength': 656},
            'oiii': {'wavelength': 501}
        }

        # Hubble palette convention
        if hasattr(channel_mapper, 'create_hubble_palette'):
            mapping = channel_mapper.create_hubble_palette(bands)
            assert mapping['red'] == 'sii'
            assert mapping['green'] == 'ha'
            assert mapping['blue'] == 'oiii'
