#!/usr/bin/env python3
"""Astronomical image processing pipeline server.

Runs the hybrid processing pipeline (zscale + CLAHE + masking) via HTTP.
Access via SSH tunnel from your Windows laptop.

Usage:
    # On remote server (after copying VisionProject files)
    cd ~/experiments/VisionProject
    python scripts/astro_pipeline_server.py --port 5001

    # On laptop (SSH tunnel)
    ssh -L 5001:localhost:5001 jakko@192.168.50.194

    # Process FITS files from laptop
    python scripts/astro_pipeline_client.py --fits g.fits r.fits i.fits --output composite.png
"""

import argparse
import base64
import io
import logging
import sys
from pathlib import Path

from flask import Flask, request, jsonify, send_file
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astro_vision_composer import FITSImageProcessor, CompositeImageGenerator
from skimage import exposure
from scipy import ndimage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)


def process_fits_hybrid(g_data, r_data, i_data):
    """Process 3 FITS files with hybrid pipeline.

    Args:
        g_data: bytes of g-band FITS file
        r_data: bytes of r-band FITS file
        i_data: bytes of i-band FITS file

    Returns:
        PNG image bytes
    """
    logger.info("Starting hybrid pipeline processing...")

    # Save temporary FITS files
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    try:
        g_path = temp_dir / "g.fits"
        r_path = temp_dir / "r.fits"
        i_path = temp_dir / "i.fits"

        g_path.write_bytes(g_data)
        r_path.write_bytes(r_data)
        i_path.write_bytes(i_data)

        # Load bands with zscale
        logger.info("Loading FITS files with zscale...")
        all_bands = {}

        for name, path in [('g_band', g_path), ('r_band', r_path), ('i_band', i_path)]:
            with FITSImageProcessor(str(path)) as fits_proc:
                bands = fits_proc.extract_bands()
                for band_name, band_data in bands.items():
                    normalized = fits_proc.normalize_band(band_name, method='zscale')
                    all_bands[name] = normalized
                    logger.info(f"  {name}: median={np.nanmedian(normalized):.4f}")

        # Map to RGB
        mapping = {'red': 'r_band', 'green': 'i_band', 'blue': 'g_band'}

        generator = CompositeImageGenerator()
        for name, data in all_bands.items():
            generator.add_band(name, data)

        # Get channels
        r = generator.bands[mapping['red']].copy()
        g = generator.bands[mapping['green']].copy()
        b = generator.bands[mapping['blue']].copy()

        # Apply CLAHE
        logger.info("Applying CLAHE contrast enhancement...")
        r_clahe = exposure.equalize_adapthist(r, clip_limit=0.03)
        g_clahe = exposure.equalize_adapthist(g, clip_limit=0.03)
        b_clahe = exposure.equalize_adapthist(b, clip_limit=0.03)

        # Unsharp mask
        logger.info("Applying unsharp mask for detail...")
        r_detail = r_clahe + (r_clahe - ndimage.gaussian_filter(r_clahe, 2)) * 0.5
        g_detail = g_clahe + (g_clahe - ndimage.gaussian_filter(g_clahe, 2)) * 0.5
        b_detail = b_clahe + (b_clahe - ndimage.gaussian_filter(b_clahe, 2)) * 0.5

        r_detail = np.clip(r_detail, 0, 1)
        g_detail = np.clip(g_detail, 0, 1)
        b_detail = np.clip(b_detail, 0, 1)

        # Star boost
        logger.info("Boosting bright stars...")
        r_thresh = np.percentile(r_detail, 99)
        g_thresh = np.percentile(g_detail, 99)
        b_thresh = np.percentile(b_detail, 99)

        r_final = np.where(r_detail > r_thresh, np.clip(r_detail * 1.3, 0, 1), r_detail)
        g_final = np.where(g_detail > g_thresh, np.clip(g_detail * 1.3, 0, 1), g_detail)
        b_final = np.where(b_detail > b_thresh, np.clip(b_detail * 1.3, 0, 1), b_detail)

        # Stack to RGB
        rgb_enhanced = np.dstack([r_final, g_final, b_final])

        # Luminance masking
        logger.info("Applying luminance masking (darkest 5% -> black)...")
        luminance = 0.2126 * rgb_enhanced[:, :, 0] + 0.7152 * rgb_enhanced[:, :, 1] + 0.0722 * rgb_enhanced[:, :, 2]
        lum_threshold = np.nanpercentile(luminance, 5)
        dark_mask = luminance < lum_threshold

        rgb_final = rgb_enhanced.copy()
        rgb_final[dark_mask] = 0

        logger.info(f"Final median: {np.nanmedian(rgb_final):.4f}")

        # Save to PNG bytes
        logger.info("Encoding PNG...")
        output_path = temp_dir / "output.png"
        generator.save_composite(rgb_final, str(output_path), quality=95)

        png_bytes = output_path.read_bytes()

        return png_bytes

    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir)


@app.route("/")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "pipeline": "astro_hybrid",
        "accepts": ["g.fits", "r.fits", "i.fits"],
        "returns": "PNG composite image"
    })


@app.route("/process", methods=["POST"])
def process():
    """Process FITS files with hybrid pipeline.

    Accepts multipart/form-data with 3 FITS files:
        - g_band: g-band FITS file
        - r_band: r-band FITS file
        - i_band: i-band FITS file

    Returns:
        PNG image file
    """
    try:
        # Validate files
        required_files = ['g_band', 'r_band', 'i_band']
        for file_key in required_files:
            if file_key not in request.files:
                return jsonify({"error": f"Missing {file_key} file"}), 400

        # Read FITS data
        g_data = request.files['g_band'].read()
        r_data = request.files['r_band'].read()
        i_data = request.files['i_band'].read()

        logger.info(f"Received FITS files: g={len(g_data)/1024/1024:.1f}MB, "
                   f"r={len(r_data)/1024/1024:.1f}MB, i={len(i_data)/1024/1024:.1f}MB")

        # Process
        png_bytes = process_fits_hybrid(g_data, r_data, i_data)

        logger.info(f"✓ Processing complete, returning {len(png_bytes)/1024:.1f}KB PNG")

        # Return PNG
        return send_file(
            io.BytesIO(png_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name='composite.png'
        )

    except Exception as e:
        logger.error(f"Error processing FITS files: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/info")
def info():
    """Get pipeline information."""
    return jsonify({
        "pipeline": "astro_hybrid",
        "stages": [
            "zscale normalization",
            "CLAHE contrast enhancement",
            "Unsharp masking",
            "Star boost (99th percentile)",
            "Luminance masking (5% -> black)"
        ],
        "input_format": "FITS",
        "output_format": "PNG",
        "channels": {"red": "r_band", "green": "i_band", "blue": "g_band"}
    })


def main():
    parser = argparse.ArgumentParser(description="Astro Pipeline Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Astronomical Image Processing Pipeline Server")
    logger.info("="*80)
    logger.info("Pipeline: Hybrid (zscale + CLAHE + masking)")
    logger.info(f"Binding: {args.host}:{args.port}")
    logger.info("")
    logger.info("SSH Tunnel from laptop:")
    logger.info(f"  ssh -L {args.port}:localhost:{args.port} jakko@192.168.50.194")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  GET  /       - Health check")
    logger.info("  GET  /info   - Pipeline info")
    logger.info("  POST /process - Process FITS files")
    logger.info("="*80)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True  # Allow concurrent requests
    )


if __name__ == "__main__":
    main()