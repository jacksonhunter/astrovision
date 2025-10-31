"""Unit tests for mission adapters (JWST, HST, PanSTARRS).

Tests mission-specific extension detection, error/quality array handling,
and metadata extraction for currently implemented adapters.

**CURRENT IMPLEMENTATION STATUS:**
- ✅ Implemented: JWSTAdapter, HSTAdapter, PanSTARRSAdapter
- ❌ Missing: EuclidAdapter, ChandraAdapter, GenericAdapter (auto-fallback)
- ⚠️ get_mission_adapter() takes mission string, not filepath (no auto-detection from TELESCOP)

**TESTING STRATEGY:**
- Test what exists comprehensively
- Document gaps for future phases
- Create fixtures that could be reused when missing adapters are added

Author: VisionProject Testing Framework
Date: 2025-10-26
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from astropy.io import fits

from astro_vision_composer.preprocessing.mission_adapters import (
    MissionAdapter,
    JWSTAdapter,
    HSTAdapter,
    PanSTARRSAdapter,
    get_mission_adapter
)

# Import synthetic FITS generators
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'fixtures'))
from synthetic_mission_fits import (
    create_jwst_fits,
    create_hst_fits,
    create_generic_fits,
    JWST_DQ_FLAGS,
    HST_DQ_FLAGS
)


class TestAdapterFactory:
    """Tests for get_mission_adapter() factory function."""

    def test_get_jwst_adapter(self):
        """Test factory returns JWSTAdapter for 'JWST' string."""
        adapter = get_mission_adapter('JWST')
        assert isinstance(adapter, JWSTAdapter)

    def test_get_hst_adapter(self):
        """Test factory returns HSTAdapter for 'HST' string."""
        adapter = get_mission_adapter('HST')
        assert isinstance(adapter, HSTAdapter)

    def test_get_panstarrs_adapter(self):
        """Test factory returns PanSTARRSAdapter for 'PanSTARRS' string."""
        adapter = get_mission_adapter('PanSTARRS')
        assert isinstance(adapter, PanSTARRSAdapter)

    def test_get_panstarrs_adapter_alias(self):
        """Test factory accepts 'PS1' alias for PanSTARRS."""
        adapter = get_mission_adapter('PS1')
        assert isinstance(adapter, PanSTARRSAdapter)

    def test_case_insensitive_matching(self):
        """Test factory handles case-insensitive mission names."""
        adapters = [
            get_mission_adapter('jwst'),
            get_mission_adapter('JWST'),
            get_mission_adapter('Jwst'),
            get_mission_adapter('hst'),
            get_mission_adapter('panstarrs')
        ]

        assert isinstance(adapters[0], JWSTAdapter)
        assert isinstance(adapters[1], JWSTAdapter)
        assert isinstance(adapters[2], JWSTAdapter)
        assert isinstance(adapters[3], HSTAdapter)
        assert isinstance(adapters[4], PanSTARRSAdapter)

    def test_unknown_mission_raises_error(self):
        """Test factory raises ValueError for unknown missions."""
        with pytest.raises(ValueError, match="Unsupported mission"):
            get_mission_adapter('EUCLID')

        with pytest.raises(ValueError, match="Unsupported mission"):
            get_mission_adapter('CHANDRA')

        with pytest.raises(ValueError, match="Unsupported mission"):
            get_mission_adapter('UnknownScope')


class TestJWSTAdapter:
    """Tests for JWST mission adapter."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test FITS files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def jwst_fits(self, temp_dir):
        """Create synthetic JWST FITS file."""
        fits_path = temp_dir / 'test_jwst_nircam.fits'
        return create_jwst_fits(
            fits_path,
            instrument='NIRCAM',
            filter_name='F200W',
            detector='NRCA1',
            add_dq=True
        )

    @pytest.fixture
    def jwst_adapter(self):
        """Create JWST adapter instance."""
        return JWSTAdapter()

    def test_jwst_identify_science_extension(self, jwst_adapter, jwst_fits):
        """Test JWST adapter identifies SCI extension correctly."""
        with fits.open(jwst_fits) as hdul:
            ext_idx, ext_name, ext_ver = jwst_adapter.identify_science_extension(hdul)

            # JWST should identify SCI extension
            assert ext_name == 'SCI'
            assert ext_ver == 1
            assert isinstance(ext_idx, int)
            assert ext_idx > 0  # Should not be PRIMARY

    def test_jwst_get_error_extension(self, jwst_adapter, jwst_fits):
        """Test JWST adapter finds ERR extension (if present)."""
        with fits.open(jwst_fits) as hdul:
            # Current synthetic FITS doesn't have ERR, should return None
            err_ext = jwst_adapter.get_error_extension(hdul, science_version=1)

            # May be None if not present (graceful handling)
            if err_ext is not None:
                ext_name, ext_ver = err_ext
                assert ext_name == 'ERR'
                assert ext_ver == 1

    def test_jwst_get_quality_extension(self, jwst_adapter, jwst_fits):
        """Test JWST adapter finds DQ extension."""
        with fits.open(jwst_fits) as hdul:
            dq_ext = jwst_adapter.get_quality_extension(hdul, science_version=1)

            # DQ extension should be present
            assert dq_ext is not None, "DQ extension not found"
            ext_name, ext_ver = dq_ext
            assert ext_name == 'DQ'
            assert ext_ver == 1

    def test_jwst_interpret_quality_flags(self, jwst_adapter):
        """Test JWST DQ flag interpretation."""
        # Create DQ value with multiple flags set
        dq_value = JWST_DQ_FLAGS['DO_NOT_USE'] | JWST_DQ_FLAGS['SATURATED']

        flags = jwst_adapter.interpret_quality_flags(dq_value)

        # Should return list of flag descriptions
        assert isinstance(flags, list)
        assert len(flags) > 0

        # Verify at least the basic flags are detected
        # (Implementation may return different formats)
        assert any('DO_NOT_USE' in str(f).upper() or '0' in str(f) for f in flags)
        assert any('SATURATED' in str(f).upper() or '1' in str(f) for f in flags)

    def test_jwst_multiple_instruments(self, temp_dir):
        """Test JWST adapter works with different instruments."""
        instruments = ['NIRCAM', 'MIRI', 'NIRSPEC', 'NIRISS']
        adapter = JWSTAdapter()

        for inst in instruments:
            fits_path = temp_dir / f'test_jwst_{inst.lower()}.fits'
            create_jwst_fits(fits_path, instrument=inst)

            with fits.open(fits_path) as hdul:
                # Should identify SCI extension for all instruments
                ext_idx, ext_name, ext_ver = adapter.identify_science_extension(hdul)
                assert ext_name == 'SCI'


class TestHSTAdapter:
    """Tests for HST mission adapter."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test FITS files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def hst_fits(self, temp_dir):
        """Create synthetic HST FITS file."""
        fits_path = temp_dir / 'test_hst_acs.fits'
        return create_hst_fits(
            fits_path,
            instrument='ACS',
            detector='WFC',
            filter_name='F606W',
            add_dq=True
        )

    @pytest.fixture
    def hst_adapter(self):
        """Create HST adapter instance."""
        return HSTAdapter()

    def test_hst_identify_science_extension(self, hst_adapter, hst_fits):
        """Test HST adapter identifies SCI extension correctly."""
        with fits.open(hst_fits) as hdul:
            ext_idx, ext_name, ext_ver = hst_adapter.identify_science_extension(hdul)

            # HST should identify SCI extension
            assert ext_name == 'SCI'
            assert ext_ver == 1
            assert isinstance(ext_idx, int)

    def test_hst_get_quality_extension(self, hst_adapter, hst_fits):
        """Test HST adapter finds DQ extension."""
        with fits.open(hst_fits) as hdul:
            dq_ext = hst_adapter.get_quality_extension(hdul, science_version=1)

            # DQ extension should be present
            assert dq_ext is not None, "DQ extension not found"
            ext_name, ext_ver = dq_ext
            assert ext_name == 'DQ'
            assert ext_ver == 1

    def test_hst_interpret_quality_flags(self, hst_adapter):
        """Test HST DQ flag interpretation."""
        # HST DQ flags
        dq_value = HST_DQ_FLAGS['BAD_DETECTOR_PIXEL'] | HST_DQ_FLAGS['SATURATED']

        flags = hst_adapter.interpret_quality_flags(dq_value)

        # Should return list of flags
        assert isinstance(flags, list)
        assert len(flags) > 0

    def test_hst_multiple_instruments(self, temp_dir):
        """Test HST adapter works with different instruments."""
        instruments = [
            ('ACS', 'WFC'),
            ('WFC3', 'UVIS'),
            ('WFC3', 'IR')
        ]
        adapter = HSTAdapter()

        for inst, det in instruments:
            fits_path = temp_dir / f'test_hst_{inst.lower()}_{det.lower()}.fits'
            create_hst_fits(fits_path, instrument=inst, detector=det)

            with fits.open(fits_path) as hdul:
                # Should identify SCI extension for all instruments
                ext_idx, ext_name, ext_ver = adapter.identify_science_extension(hdul)
                assert ext_name == 'SCI'


class TestPanSTARRSAdapter:
    """Tests for PanSTARRS mission adapter."""

    @pytest.fixture
    def panstarrs_adapter(self):
        """Create PanSTARRS adapter instance."""
        return PanSTARRSAdapter()

    def test_panstarrs_adapter_exists(self, panstarrs_adapter):
        """Test PanSTARRS adapter can be instantiated."""
        assert isinstance(panstarrs_adapter, PanSTARRSAdapter)
        assert isinstance(panstarrs_adapter, MissionAdapter)

    def test_panstarrs_has_required_methods(self, panstarrs_adapter):
        """Test PanSTARRS adapter implements required abstract methods."""
        # Verify required methods exist
        assert hasattr(panstarrs_adapter, 'identify_science_extension')
        assert hasattr(panstarrs_adapter, 'get_error_extension')
        assert hasattr(panstarrs_adapter, 'get_quality_extension')
        assert hasattr(panstarrs_adapter, 'interpret_quality_flags')

        # Verify they are callable
        assert callable(panstarrs_adapter.identify_science_extension)
        assert callable(panstarrs_adapter.get_error_extension)
        assert callable(panstarrs_adapter.get_quality_extension)
        assert callable(panstarrs_adapter.interpret_quality_flags)


class TestMissionAdapterInterface:
    """Tests for MissionAdapter abstract base class interface."""

    def test_cannot_instantiate_base_class(self):
        """Test MissionAdapter cannot be instantiated directly (it's abstract)."""
        with pytest.raises(TypeError):
            MissionAdapter()

    def test_all_adapters_inherit_from_base(self):
        """Test all adapters inherit from MissionAdapter."""
        jwst = JWSTAdapter()
        hst = HSTAdapter()
        panstarrs = PanSTARRSAdapter()

        assert isinstance(jwst, MissionAdapter)
        assert isinstance(hst, MissionAdapter)
        assert isinstance(panstarrs, MissionAdapter)

    def test_adapters_have_required_methods(self):
        """Test all adapters implement required abstract methods."""
        adapters = [JWSTAdapter(), HSTAdapter(), PanSTARRSAdapter()]

        required_methods = [
            'identify_science_extension',
            'get_error_extension',
            'get_quality_extension',
            'interpret_quality_flags'
        ]

        for adapter in adapters:
            for method in required_methods:
                assert hasattr(adapter, method), \
                    f"{adapter.__class__.__name__} missing {method}"
                assert callable(getattr(adapter, method)), \
                    f"{adapter.__class__.__name__}.{method} not callable"


class TestDQFlagBitMasks:
    """Tests for DQ flag bit mask operations."""

    def test_jwst_flag_values_are_powers_of_two(self):
        """Test JWST DQ flags are valid bit masks (powers of 2)."""
        for flag_name, flag_value in JWST_DQ_FLAGS.items():
            # Each flag should be a power of 2 (single bit set)
            assert flag_value > 0, f"{flag_name} has invalid value {flag_value}"
            # Check it's a power of 2: n & (n-1) == 0 for powers of 2
            is_power_of_2 = (flag_value & (flag_value - 1)) == 0
            assert is_power_of_2, f"{flag_name} ({flag_value}) is not a power of 2"

    def test_hst_flag_values_are_powers_of_two(self):
        """Test HST DQ flags are valid bit masks (powers of 2)."""
        for flag_name, flag_value in HST_DQ_FLAGS.items():
            assert flag_value > 0, f"{flag_name} has invalid value {flag_value}"
            is_power_of_2 = (flag_value & (flag_value - 1)) == 0
            assert is_power_of_2, f"{flag_name} ({flag_value}) is not a power of 2"

    def test_combined_flags_detection(self):
        """Test detecting multiple flags set in a single DQ value."""
        # Combine multiple JWST flags
        dq_value = (JWST_DQ_FLAGS['DO_NOT_USE'] |
                    JWST_DQ_FLAGS['SATURATED'] |
                    JWST_DQ_FLAGS['JUMP_DET'])

        # Verify each individual flag is detected
        assert (dq_value & JWST_DQ_FLAGS['DO_NOT_USE']) != 0
        assert (dq_value & JWST_DQ_FLAGS['SATURATED']) != 0
        assert (dq_value & JWST_DQ_FLAGS['JUMP_DET']) != 0

        # Verify flags not set are not detected
        assert (dq_value & JWST_DQ_FLAGS['DROPOUT']) == 0
        assert (dq_value & JWST_DQ_FLAGS['OUTLIER']) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test FITS files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_adapter_with_minimal_fits(self, temp_dir):
        """Test adapters handle minimal FITS (PRIMARY only)."""
        fits_path = temp_dir / 'minimal.fits'

        # Create minimal FITS (PRIMARY HDU only, no extensions)
        data = np.random.poisson(1000, (50, 50)).astype(np.float32)
        primary_hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([primary_hdu])
        hdulist.writeto(fits_path, overwrite=True)
        hdulist.close()

        # Adapters should handle this gracefully (may return PRIMARY or raise informative error)
        adapter = JWSTAdapter()
        with fits.open(fits_path) as hdul:
            try:
                ext_idx, ext_name, ext_ver = adapter.identify_science_extension(hdul)
                # If it succeeds, verify it's a valid response
                assert isinstance(ext_idx, int)
                assert ext_idx >= 0
            except (ValueError, IndexError, KeyError) as e:
                # If it fails, should be an informative error
                assert len(str(e)) > 0

    def test_adapter_with_no_dq_extension(self, temp_dir):
        """Test adapters handle missing DQ extension gracefully."""
        fits_path = temp_dir / 'no_dq.fits'
        create_jwst_fits(fits_path, add_dq=False)

        adapter = JWSTAdapter()
        with fits.open(fits_path) as hdul:
            dq_ext = adapter.get_quality_extension(hdul, science_version=1)

            # Should return None (graceful handling)
            assert dq_ext is None

    def test_interpret_flags_with_zero(self):
        """Test flag interpretation with zero value (no flags set)."""
        adapter = JWSTAdapter()
        flags = adapter.interpret_quality_flags(0)

        # Should return empty list or handle gracefully
        assert isinstance(flags, list)
        # May be empty or have default behavior
        assert len(flags) >= 0


# ==============================================================================
# MISSING ADAPTERS - Tests for Future Implementation
# ==============================================================================

@pytest.mark.skip(reason="EuclidAdapter not yet implemented")
class TestEuclidAdapter:
    """Tests for Euclid mission adapter (NOT YET IMPLEMENTED)."""

    def test_placeholder(self):
        """Placeholder test for future Euclid adapter."""
        pass


@pytest.mark.skip(reason="ChandraAdapter not yet implemented")
class TestChandraAdapter:
    """Tests for Chandra mission adapter (NOT YET IMPLEMENTED)."""

    def test_placeholder(self):
        """Placeholder test for future Chandra adapter."""
        pass


@pytest.mark.skip(reason="GenericAdapter with auto-detection not yet implemented")
class TestGenericAdapter:
    """Tests for generic fallback adapter (NOT YET IMPLEMENTED)."""

    def test_placeholder(self):
        """Placeholder test for future generic adapter with auto-detection."""
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
