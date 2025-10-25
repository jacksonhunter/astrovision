#!/usr/bin/env python3
"""
Download NOIRLab FITS Liberator datasets

Downloads FITS files and supporting data from NOIRLab's educational archive.
Based on the dataset_metadata.json in this directory.

Usage:
    python download_noirlab_datasets.py [dataset_ids...]

Examples:
    python download_noirlab_datasets.py edu008         # Download Eagle Nebula only
    python download_noirlab_datasets.py edu008 edu010  # Download multiple
    python download_noirlab_datasets.py --all          # Download all 18 datasets
    python download_noirlab_datasets.py --small        # Download only small datasets (<50MB)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import zipfile
import shutil

# User-Agent to avoid 403 errors
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Small datasets for quick testing
SMALL_DATASETS = ['edu008', 'edu010', 'edu011', 'edu012', 'edu013', 'edu014',
                  'edu015', 'edu016', 'edu017', 'edu018', 'edu021', 'edu025']

# Large datasets (>100MB)
LARGE_DATASETS = ['edu020', 'edu022', 'edu023', 'edu024', 'edu027']


def fetch_url(url):
    """Fetch URL with proper headers"""
    req = Request(url, headers=HEADERS)
    with urlopen(req) as response:
        return response.read()


def download_file(url, dest_path):
    """Download file with progress indication"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading: {url}")
    req = Request(url, headers=HEADERS)

    with urlopen(req) as response:
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            downloaded = 0
            chunk_size = 8192

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"    Progress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB)", end='\r')

    print(f"\n    Saved: {dest_path}")
    return dest_path


def extract_zip(zip_path, extract_to):
    """Extract ZIP file"""
    print(f"  Extracting: {zip_path.name}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"    Extracted to: {extract_to}")


def scrape_dataset_page(url):
    """
    Scrape NOIRLab dataset page for download links

    Note: User reported scraping issues - this is a basic implementation
    that may need refinement based on actual page structure.
    """
    print(f"  Fetching page: {url}")
    html = fetch_url(url).decode('utf-8')

    # Find ZIP file links
    zip_pattern = re.compile(r'href="([^"]+\.zip)"')
    zip_links = zip_pattern.findall(html)

    # Find FITS file links
    fits_pattern = re.compile(r'href="([^"]+\.(fits|fz))"')
    fits_links = fits_pattern.findall(html)

    # Find TIF preview
    tif_pattern = re.compile(r'href="([^"]+\.tif)"')
    tif_links = tif_pattern.findall(html)

    downloads = {
        'zips': [urljoin(url, link) for link in zip_links],
        'fits': [urljoin(url, link[0]) for link in fits_links],
        'tif': [urljoin(url, link) for link in tif_links]
    }

    return downloads


def download_dataset(dataset_id, url, output_dir):
    """Download a single dataset"""
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_id}")
    print(f"{'='*60}")

    dataset_dir = output_dir / 'NOIRLab' / 'examples' / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save link.txt
    link_file = dataset_dir / 'link.txt'
    link_file.write_text(url)

    try:
        # Scrape page for download links
        downloads = scrape_dataset_page(url)

        # Download preview TIF
        for tif_url in downloads['tif']:
            filename = Path(tif_url).name
            download_file(tif_url, dataset_dir / filename)

        # Download ZIP files
        for zip_url in downloads['zips']:
            filename = Path(zip_url).name
            zip_path = download_file(zip_url, dataset_dir / filename)

            # Extract to data/ subdirectory
            extract_zip(zip_path, dataset_dir / 'data')

        # Download individual FITS files (if not in ZIPs)
        if not downloads['zips'] and downloads['fits']:
            data_dir = dataset_dir / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)

            for fits_url in downloads['fits']:
                filename = Path(fits_url).name
                download_file(fits_url, data_dir / filename)

        print(f"\n✅ {dataset_id} complete!")

    except Exception as e:
        print(f"\n❌ Error downloading {dataset_id}: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Download NOIRLab FITS datasets')
    parser.add_argument('datasets', nargs='*', help='Dataset IDs to download (e.g., edu008)')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--small', action='store_true', help='Download only small datasets (<50MB)')
    parser.add_argument('--output', '-o', type=Path, default=Path('.'),
                       help='Output directory (default: current directory)')

    args = parser.parse_args()

    # Load dataset metadata
    script_dir = Path(__file__).parent
    metadata_file = script_dir / 'NOIRLab' / 'examples' / 'dataset_metadata.json'

    if not metadata_file.exists():
        print(f"Error: dataset_metadata.json not found at {metadata_file}")
        print("This file should be in examples/data/NOIRLab/examples/")
        return 1

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Determine which datasets to download
    if args.all:
        datasets = list(metadata.keys())
    elif args.small:
        datasets = SMALL_DATASETS
    elif args.datasets:
        datasets = args.datasets
    else:
        print("Error: Specify dataset IDs, --all, or --small")
        parser.print_help()
        return 1

    # Validate dataset IDs
    invalid = [d for d in datasets if d not in metadata]
    if invalid:
        print(f"Error: Unknown datasets: {', '.join(invalid)}")
        print(f"Available: {', '.join(sorted(metadata.keys()))}")
        return 1

    # Confirm large downloads
    if args.all or any(d in LARGE_DATASETS for d in datasets):
        print("\n⚠️  Warning: Some datasets are very large (>100MB, M106 is 3.1GB)")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

    # Download each dataset
    successful = []
    failed = []

    for dataset_id in datasets:
        url = metadata[dataset_id]['url']
        if download_dataset(dataset_id, url, args.output):
            successful.append(dataset_id)
        else:
            failed.append(dataset_id)

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")

    if failed:
        print(f"\nFailed datasets: {', '.join(failed)}")
        print("\nNote: Web scraping can be fragile. If downloads fail:")
        print("1. Check your internet connection")
        print("2. Visit the dataset URL in a browser to verify it's accessible")
        print("3. Download manually and extract to examples/data/NOIRLab/examples/")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
