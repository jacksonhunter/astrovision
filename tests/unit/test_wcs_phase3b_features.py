"""Phase 3B WCS Features Tests - Intermediate Frames, Bounding Boxes, Saving.

Tests the advanced gwcs features added in Phase 3B:
- get_available_frames()
- get_transform()
- inspect_pipeline()
- set_bounding_box() / get_bounding_box()
- save_wcs()
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.modeling import models

from astro_vision_composer.processing.wcs_handler import WCSHandler

# Check if gwcs is available
try:
    import gwcs
    from gwcs import coordinate_frames as cf
    GWCS_AVAILABLE = True
except ImportError:
    GWCS_AVAILABLE = False


@pytest.fixture
def wcs_handler():
    """Create WCSHandler instance."""
    return WCSHandler()


@pytest.fixture
def simple_fits_wcs():
    """Create a simple FITS WCS object."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1024, 1024]
    wcs.wcs.crval = [30.0, 45.0]
    wcs.wcs.cdelt = [0.1 / 3600, 0.1 / 3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs


@pytest.fixture
def simple_gwcs():
    """Create a simple gwcs.WCS object with multiple frames."""
    if not GWCS_AVAILABLE:
        pytest.skip("gwcs not installed")

    # Create detector→sky transform with intermediate frame
    crpix = (1024, 1024)

    # Step 1: Shift to origin
    shift_by_crpix = models.Shift(-crpix[0]) & models.Shift(-crpix[1])
    shift_by_crpix.name = "center_shift"

    # Step 2: Distortion (simplified)
    distortion = models.Scale(1.0) & models.Scale(1.0)
    distortion.name = "distortion"

    # Step 3: Pixel scale
    pixelscale = models.Scale(0.1 / 3600) & models.Scale(0.1 / 3600)
    pixelscale.name = "pixel_scale"

    # Step 4: Projection
    tan = models.Pix2Sky_TAN()
    tan.name = "tan_projection"

    # Step 5: Sky rotation
    celestial_rotation = models.RotateNative2Celestial(30.0, 45.0, 180.0)
    celestial_rotation.name = "sky_rotation"

    # Compose: detector → undistorted → v2v3 → world
    det2undist = shift_by_crpix | distortion
    det2undist.name = "detector_to_undistorted"

    undist2v2v3 = pixelscale
    undist2v2v3.name = "undistorted_to_v2v3"

    v2v32world = tan | celestial_rotation
    v2v32world.name = "v2v3_to_world"

    # Define frames
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
    undistorted_frame = cf.Frame2D(name="undistorted", axes_names=("x", "y"), unit=(u.pix, u.pix))
    v2v3_frame = cf.Frame2D(name="v2v3", axes_names=("v2", "v3"), unit=(u.deg, u.deg))
    world_frame = cf.CelestialFrame(
        reference_frame=SkyCoord(0, 0, unit='deg', frame='icrs').frame,
        name='world',
        unit=(u.deg, u.deg)
    )

    # Create gwcs pipeline with intermediate frames
    wcs_gwcs = gwcs.wcs.WCS([
        (detector_frame, det2undist),
        (undistorted_frame, undist2v2v3),
        (v2v3_frame, v2v32world),
        (world_frame, None)
    ])

    return wcs_gwcs


class TestIntermediateFrames:
    """Test intermediate frame access (Phase 3B)."""

    def test_get_available_frames_gwcs(self, wcs_handler, simple_gwcs):
        """Test get_available_frames() with gwcs."""
        frames = wcs_handler.get_available_frames(simple_gwcs)

        assert frames is not None
        assert isinstance(frames, list)
        assert 'detector' in frames
        assert 'world' in frames
        # Should have intermediate frames
        assert len(frames) >= 3

    def test_get_available_frames_fits(self, wcs_handler, simple_fits_wcs):
        """Test get_available_frames() with FITS WCS (returns None)."""
        frames = wcs_handler.get_available_frames(simple_fits_wcs)

        assert frames is None  # FITS WCS has no intermediate frames

    def test_get_transform_gwcs(self, wcs_handler, simple_gwcs):
        """Test get_transform() between frames."""
        # Get transform from detector to undistorted
        transform = wcs_handler.get_transform(simple_gwcs, 'detector', 'undistorted')

        assert transform is not None

        # Test the transform
        x, y = 100, 200
        x_undist, y_undist = transform(x, y)

        # Should apply transformation
        assert x_undist != x or y_undist != y

    def test_get_transform_full_chain(self, wcs_handler, simple_gwcs):
        """Test get_transform() for full chain (detector→world)."""
        # Get full transform
        full_transform = wcs_handler.get_transform(simple_gwcs, 'detector', 'world')

        assert full_transform is not None

        # Transform some pixels
        ra, dec = full_transform(1024, 1024)

        # Should be near reference coordinates
        assert np.abs(ra - 30.0) < 0.1
        assert np.abs(dec - 45.0) < 0.1

    def test_get_transform_fits(self, wcs_handler, simple_fits_wcs):
        """Test get_transform() with FITS WCS (returns None)."""
        transform = wcs_handler.get_transform(simple_fits_wcs, 'any', 'frame')

        assert transform is None  # FITS WCS doesn't support sub-transforms

    def test_inspect_pipeline_gwcs(self, wcs_handler, simple_gwcs):
        """Test inspect_pipeline() with gwcs."""
        info = wcs_handler.inspect_pipeline(simple_gwcs)

        assert info['type'] == 'gwcs'
        assert info['has_pipeline'] is True
        assert info['frames'] is not None
        assert len(info['frames']) >= 3

        # Check steps
        assert info['steps'] is not None
        assert len(info['steps']) >= 3

        # Check frames have names
        for step in info['steps']:
            assert 'frame' in step
            assert 'transform' in step

        # Check input/output frames
        assert info['input_frame'] is not None
        assert info['input_frame']['name'] == 'detector'
        assert info['output_frame'] is not None
        assert info['output_frame']['name'] == 'world'

    def test_inspect_pipeline_fits(self, wcs_handler, simple_fits_wcs):
        """Test inspect_pipeline() with FITS WCS."""
        info = wcs_handler.inspect_pipeline(simple_fits_wcs)

        assert info['type'] == 'standard'
        assert info['has_pipeline'] is False
        assert info['frames'] is None
        assert info['steps'] is None

        # Should still have basic frame info
        assert info['input_frame'] is not None
        assert info['input_frame']['name'] == 'pixel'
        assert info['output_frame'] is not None


class TestBoundingBox:
    """Test bounding box support (Phase 3B)."""

    def test_set_get_bounding_box_gwcs(self, wcs_handler, simple_gwcs):
        """Test set/get bounding box with gwcs."""
        # Set bounding box
        bbox = ((0, 2048), (0, 1024))
        wcs_handler.set_bounding_box(simple_gwcs, bbox)

        # Get it back
        bbox_retrieved = wcs_handler.get_bounding_box(simple_gwcs)

        assert bbox_retrieved is not None
        assert bbox_retrieved == bbox

    def test_bounding_box_enforcement_gwcs(self, wcs_handler, simple_gwcs):
        """Test that bounding box actually restricts transformations."""
        # Set bounding box
        bbox = ((0, 2048), (0, 1024))
        wcs_handler.set_bounding_box(simple_gwcs, bbox)

        # Transform within bounds (should work)
        sky_inside = simple_gwcs.pixel_to_world(100, 200)
        assert isinstance(sky_inside, SkyCoord)
        assert not np.isnan(sky_inside.ra.deg)

        # Transform outside bounds (should return NaN)
        # Note: behavior depends on gwcs version and with_bounding_box flag
        sky_outside = simple_gwcs.pixel_to_world(3000, 500, with_bounding_box=True)
        # gwcs returns NaN for out-of-bounds
        assert np.isnan(sky_outside.ra.deg) or np.isnan(sky_outside.dec.deg)

    def test_bounding_box_fits_warning(self, wcs_handler, simple_fits_wcs):
        """Test that setting bounding box on FITS WCS issues warning."""
        bbox = ((0, 2048), (0, 1024))

        # Should warn (FITS WCS doesn't support bounding boxes)
        with pytest.warns(UserWarning):
            wcs_handler.set_bounding_box(simple_fits_wcs, bbox)

        # Get should return None
        bbox_retrieved = wcs_handler.get_bounding_box(simple_fits_wcs)
        assert bbox_retrieved is None


class TestWCSSaving:
    """Test WCS saving/loading (Phase 3B)."""

    def test_save_load_gwcs(self, wcs_handler, simple_gwcs):
        """Test saving and loading gwcs."""
        # Check if asdf is available
        try:
            import asdf
        except ImportError:
            pytest.skip("asdf not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_wcs.asdf"

            # Save WCS
            wcs_handler.save_wcs(simple_gwcs, output_file)

            # Check file was created
            assert output_file.exists()

            # Load it back
            wcs_loaded = wcs_handler.load_wcs(output_file)

            # Verify it's the same type
            assert hasattr(wcs_loaded, 'available_frames')

            # Verify transformation works the same
            sky_orig = simple_gwcs.pixel_to_world(1024, 1024)
            sky_loaded = wcs_loaded.pixel_to_world(1024, 1024)

            # Should be very close
            assert np.abs(sky_orig.ra.deg - sky_loaded.ra.deg) < 0.001
            assert np.abs(sky_orig.dec.deg - sky_loaded.dec.deg) < 0.001

    def test_save_overwrite_protection(self, wcs_handler, simple_gwcs):
        """Test that save_wcs() protects against overwriting."""
        try:
            import asdf
        except ImportError:
            pytest.skip("asdf not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_wcs.asdf"

            # Save once
            wcs_handler.save_wcs(simple_gwcs, output_file)

            # Try to save again without overwrite=True
            with pytest.raises(FileExistsError):
                wcs_handler.save_wcs(simple_gwcs, output_file, overwrite=False)

            # Should work with overwrite=True
            wcs_handler.save_wcs(simple_gwcs, output_file, overwrite=True)

    def test_save_fits_wcs(self, wcs_handler, simple_fits_wcs):
        """Test saving FITS WCS (should work via ASDF)."""
        try:
            import asdf
        except ImportError:
            pytest.skip("asdf not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "fits_wcs.asdf"

            # Save FITS WCS
            wcs_handler.save_wcs(simple_fits_wcs, output_file)

            # Should succeed
            assert output_file.exists()


class TestIntegration:
    """Integration tests combining multiple Phase 3B features."""

    def test_full_gwcs_workflow(self, wcs_handler, simple_gwcs):
        """Test complete workflow: inspect → extract → use."""
        # 1. Inspect pipeline
        pipeline_info = wcs_handler.inspect_pipeline(simple_gwcs)
        assert pipeline_info['type'] == 'gwcs'

        # 2. Get available frames
        frames = wcs_handler.get_available_frames(simple_gwcs)
        assert 'detector' in frames
        assert 'undistorted' in frames

        # 3. Extract distortion correction
        distortion = wcs_handler.get_transform(simple_gwcs, 'detector', 'undistorted')
        assert distortion is not None

        # 4. Apply distortion to some pixels
        x_dist, y_dist = distortion(100, 200)
        assert x_dist is not None

        # 5. Set bounding box
        bbox = ((0, 2048), (0, 1024))
        wcs_handler.set_bounding_box(simple_gwcs, bbox)

        # 6. Verify bounding box
        bbox_check = wcs_handler.get_bounding_box(simple_gwcs)
        assert bbox_check == bbox

    def test_save_with_bounding_box(self, wcs_handler, simple_gwcs):
        """Test that bounding box is preserved when saving."""
        try:
            import asdf
        except ImportError:
            pytest.skip("asdf not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "wcs_with_bbox.asdf"

            # Set bounding box
            bbox = ((0, 2048), (0, 1024))
            wcs_handler.set_bounding_box(simple_gwcs, bbox)

            # Save
            wcs_handler.save_wcs(simple_gwcs, output_file)

            # Load back
            wcs_loaded = wcs_handler.load_wcs(output_file)

            # Bounding box should be preserved
            bbox_loaded = wcs_handler.get_bounding_box(wcs_loaded)
            assert bbox_loaded == bbox


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
