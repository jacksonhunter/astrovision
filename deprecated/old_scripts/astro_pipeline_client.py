#!/usr/bin/env python3
"""Client for astro pipeline server.

Send FITS files to remote server via SSH tunnel, get processed PNG back.

Usage:
    # First, establish SSH tunnel (in separate terminal):
    ssh -L 5001:localhost:5001 jakko@192.168.50.194

    # Then process FITS files:
    python scripts/astro_pipeline_client.py \
        --fits "Mission Control/examples/output/raw_fits/panstarrs/rings.v3.skycell.2159.034.stk.g.unconv.fits" \
               "Mission Control/examples/output/raw_fits/panstarrs/rings.v3.skycell.2159.034.stk.r.unconv.fits" \
               "Mission Control/examples/output/raw_fits/panstarrs/rings.v3.skycell.2159.034.stk.i.unconv.fits" \
        --output remote_composite.png
"""

import argparse
import sys
from pathlib import Path

import requests

def main():
    parser = argparse.ArgumentParser(description="Astro Pipeline Client")
    parser.add_argument(
        "--fits",
        nargs=3,
        required=True,
        metavar=("G_FITS", "R_FITS", "I_FITS"),
        help="3 FITS files (g, r, i bands) in order"
    )
    parser.add_argument(
        "--output",
        default="remote_composite.png",
        help="Output PNG file path"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:5001",
        help="Server URL (should be tunneled via SSH)"
    )

    args = parser.parse_args()

    # Validate FITS files exist
    fits_files = [Path(f) for f in args.fits]
    for fits_file in fits_files:
        if not fits_file.exists():
            print(f"Error: FITS file not found: {fits_file}")
            sys.exit(1)

    print("="*80)
    print("Astro Pipeline Client")
    print("="*80)
    print(f"Server: {args.server}")
    print(f"Output: {args.output}")
    print()

    # Check server health
    print("Checking server...")
    try:
        response = requests.get(f"{args.server}/", timeout=5)
        info = response.json()
        print(f"✓ Server online: {info.get('pipeline', 'unknown')}")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print()
        print("Make sure SSH tunnel is running:")
        print("  ssh -L 5001:localhost:5001 jakko@192.168.50.194")
        sys.exit(1)

    # Upload FITS files
    print()
    print("Uploading FITS files...")
    for i, fits_file in enumerate(fits_files):
        size_mb = fits_file.stat().st_size / 1024 / 1024
        print(f"  {['g', 'r', 'i'][i]}-band: {fits_file.name} ({size_mb:.1f} MB)")

    try:
        files = {
            'g_band': ('g.fits', fits_files[0].open('rb'), 'application/octet-stream'),
            'r_band': ('r.fits', fits_files[1].open('rb'), 'application/octet-stream'),
            'i_band': ('i.fits', fits_files[2].open('rb'), 'application/octet-stream'),
        }

        print()
        print("Processing on remote server...")
        print("(This may take 30-60 seconds)")

        response = requests.post(
            f"{args.server}/process",
            files=files,
            timeout=300  # 5 minutes
        )

        if response.status_code == 200:
            # Save PNG
            output_path = Path(args.output)
            output_path.write_bytes(response.content)

            output_size_kb = len(response.content) / 1024
            print(f"✓ Success! Saved {output_size_kb:.1f} KB to: {output_path.resolve()}")
        else:
            print(f"✗ Server error: {response.status_code}")
            print(response.text)
            sys.exit(1)

    except requests.exceptions.Timeout:
        print("✗ Request timed out (server may be busy)")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    finally:
        # Close file handles
        for file_tuple in files.values():
            file_tuple[1].close()

    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()