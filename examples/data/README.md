# Example Data

This folder contains sample FITS files and images for demonstrations and tutorials.

**IMPORTANT:** All files in this directory are excluded from git via `.gitignore` to prevent bloating the repository. These files remain local to your machine.

## What's Here

### NOIRLab Educational Datasets
Curated astronomical images from NOIRLab's FITS Liberator collection:
- **18 datasets** (edu008 through edu027)
- **FITS files:** Raw narrowband and broadband observations (~10MB each)
- **Composites:** Pre-processed PSD files showing final results
- **Metadata:** Dataset descriptions and source links

These are excellent for:
- Testing the FITS processing pipeline
- Learning RGB mapping with narrowband data (H-alpha, OIII, SII)
- Comparing your results to professional composites
- Educational demonstrations

### Source Information
Originally downloaded from: https://noirlab.edu/public/products/education/
Organized in `.mission_control/NOIRLab` and copied here for convenience.

## Downloading Datasets

### Option 1: Use the Download Script
```bash
# Download a specific dataset
python examples/data/download_noirlab_datasets.py edu008

# Download multiple datasets
python examples/data/download_noirlab_datasets.py edu008 edu010 edu025

# Download all small datasets (<50MB)
python examples/data/download_noirlab_datasets.py --small

# Download everything (⚠️ ~15GB)
python examples/data/download_noirlab_datasets.py --all
```

### Option 2: Manual Download
Each dataset has its own page on NOIRLab. Visit the URLs below:

| ID | Object | URL |
|----|--------|-----|
| edu008 | Eagle Nebula (M16) | https://noirlab.edu/public/products/education/edu008/ |
| edu010 | Messier 17 | https://noirlab.edu/public/products/education/edu010/ |
| edu011 | Roberts 22 | https://noirlab.edu/public/products/education/edu011/ |
| edu012 | NGC 5307 | https://noirlab.edu/public/products/education/edu012/ |
| edu013 | NGC 6309 | https://noirlab.edu/public/products/education/edu013/ |
| edu014 | NGC 6881 | https://noirlab.edu/public/products/education/edu014/ |
| edu015 | Venus UV | https://noirlab.edu/public/products/education/edu015/ |
| edu016 | NGC 1068 | https://noirlab.edu/public/products/education/edu016/ |
| edu017 | N11B (LMC) | https://noirlab.edu/public/products/education/edu017/ |
| edu018 | NGC 6302 | https://noirlab.edu/public/products/education/edu018/ |
| edu019 | NGC 1569 | https://noirlab.edu/public/products/education/edu019/ |
| edu020 | Messier 12 | https://noirlab.edu/public/products/education/edu020/ |
| edu021 | NGC 6652 | https://noirlab.edu/public/products/education/edu021/ |
| edu022 | Messier 31 (Andromeda) | https://noirlab.edu/public/products/education/edu022/ |
| edu023 | Messier 35 | https://noirlab.edu/public/products/education/edu023/ |
| edu024 | Messier 42 (Orion) | https://noirlab.edu/public/products/education/edu024/ |
| edu025 | Antennae Galaxies | https://noirlab.edu/public/products/education/edu025/ |
| edu027 | Messier 106 | https://noirlab.edu/public/products/education/edu027/ |

**Recommended for Quick Testing:**
- edu025 (Antennae) - 4 small files, ~28MB total
- edu008 (Eagle Nebula) - Classic 3-band narrowband
- edu012 (NGC 5307) - Simple 2-band planetary nebula

## Usage Guidelines

### ✅ Safe to Add
- Reference FITS files for testing
- Example composites
- Documentation images
- Metadata/config files

### ⚠️ Already Gitignored
The following extensions are automatically excluded from git:
- `*.fits`, `*.fit`, `*.fts` - FITS astronomical data
- `*.png`, `*.jpg`, `*.tif` - Image outputs
- `*.zip`, `*.tar.gz` - Archives
- `*.psd` - Photoshop composites
- `*.hdf5`, `*.npy` - Scientific data formats

## Directory Structure

```
examples/data/
└── NOIRLab/
    ├── DATASETS.md                 # Overview of all datasets
    ├── dataset_metadata.json       # Structured metadata
    └── examples/
        ├── edu008/                 # Eagle Nebula (M16)
        ├── edu010/                 # M17 Star-forming Nebula
        ├── edu011/                 # Robert's Quartet
        └── ...                     # More datasets
```

Each dataset folder contains:
- `description.md` - Target info, filters used, processing notes
- `data/` - Extracted FITS files (raw observations)
- `*.tif` - Preview thumbnail
- `*.zip` - Original archives from NOIRLab
- `link.txt` - Source URL

## Network Path Alternative

For PanSTARRS and other large survey data, use the network mount instead:
```
\\127.0.0.1\Experiments\VisionProject\raw_fits\
```

Access via SCP:
```python
from astro_vision_composer.preprocessing import FITSLoader

loader = FITSLoader()
data = loader.load('//127.0.0.1/Experiments/VisionProject/raw_fits/object_name/file.fits')
```
