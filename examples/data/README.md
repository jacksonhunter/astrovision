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
