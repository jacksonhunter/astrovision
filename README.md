# Astro Vision Composer

**AI-Guided Astronomical Image Composition from FITS Data**

Create stunning, planetarium-style space images from astronomical FITS files using local vision AI to make aesthetic decisions about band mapping, color assignment, and image enhancement.

## Features

- **FITS Band Extraction**: Automatically extract and normalize multiple wavelength bands from FITS files
- **AI-Guided Color Mapping**: Use local vision models to intelligently assign bands to RGB/CMYK channels
- **Planetarium-Quality Enhancement**:
  - Adaptive contrast enhancement (CLAHE)
  - Detail enhancement via unsharp masking
  - Bright star highlighting
  - Color balancing
- **Flexible Composition Modes**: RGB or CMYK composite generation
- **Local Inference**: All AI processing runs locally (no API keys required)
- **Professional Astronomy Tools**: Built on Astropy, APLpy, and scikit-image

## Installation

### Prerequisites

- Python 3.11+
- 8GB+ RAM (16GB recommended for large FITS files)
- Optional: NVIDIA GPU with CUDA for faster AI inference

### Install Dependencies

```bash
# Clone or navigate to project
cd VisionProject

# Install the package and all dependencies
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### First-Time Model Download

The first run will download the Llama-3.2-11B Vision model (~11GB) to `~/experiments/Models/huggingface/`. This takes 10-30 minutes depending on connection speed. Subsequent runs use the cached model.

## Quick Start

### Basic Usage

Create an AI-guided RGB composite from a FITS file:

```bash
python scripts/create_astro_composite.py my_image.fits -o output.png
```

### Advanced Usage

```bash
# Create CMYK composite with AI guidance
python scripts/create_astro_composite.py image.fits -o output.png --mode cmyk

# Skip AI and use automatic band mapping (faster)
python scripts/create_astro_composite.py image.fits --no-ai -o output.png

# Manual band mapping
python scripts/create_astro_composite.py image.fits \
    --manual-mapping '{"red": "Band_2", "green": "Band_1", "blue": "Band_0"}' \
    -o output.png
```

## Library API

### Python Usage

```python
from astro_vision_composer import (
    FITSImageProcessor,
    CompositeImageGenerator,
    VisionGuidedCompositor
)

# Step 1: Load FITS and extract bands
with FITSImageProcessor("image.fits") as fits_proc:
    bands = fits_proc.extract_bands()
    metadata = fits_proc.get_band_info()

    # Normalize bands for visualization
    normalized = {
        name: fits_proc.normalize_band(name, method='zscale')
        for name in bands.keys()
    }

# Step 2: Get AI recommendations for band mapping
compositor = VisionGuidedCompositor()
mapping = compositor.recommend_band_mapping(
    normalized,
    metadata,
    mode='rgb'
)
# Returns: {'red': 'Band_2', 'green': 'Band_1', 'blue': 'Band_0'}

# Step 3: Create enhanced composite
generator = CompositeImageGenerator()

for name, data in normalized.items():
    generator.add_band(name, data)

composite = generator.create_rgb_composite(
    r_band=mapping['red'],
    g_band=mapping['green'],
    b_band=mapping['blue'],
    enhance_contrast=True,
    enhance_details=True,
    enhance_stars=True
)

# Step 4: Get optimization feedback
optimization = compositor.optimize_composite_parameters(composite)
print(f"Quality: {optimization['overall_quality']}")
print(f"Suggestions: {optimization['specific_recommendations']}")

# Step 5: Save result
generator.save_composite(composite, "output.png", quality=95)
```

## Architecture

### Core Components

#### 1. FITSImageProcessor
Handles FITS file loading and band extraction with astronomical data processing.

#### 2. CompositeImageGenerator
Creates enhanced composites with contrast, detail, and star improvements.

#### 3. VisionGuidedCompositor
Uses local AI to analyze bands and suggest optimal color mappings.

## Project Structure

```
VisionProject/
├── src/
│   └── astro_vision_composer/
│       ├── __init__.py
│       ├── fits_processor.py          # FITS loading & band extraction
│       ├── composite_generator.py     # Image composition & enhancement
│       └── vision_compositor.py       # AI-guided decisions
├── scripts/
│   ├── create_astro_composite.py      # Main workflow script
│   ├── local_vision_inference.py      # Vision model interface
│   └── theme_vision.py                # Color palette extraction (example)
├── requirements.txt                    # All dependencies
├── pyproject.toml                      # Package configuration
├── SETUP.md                            # Installation guide
└── README.md                           # This file
```

## See Also

- [SETUP.md](SETUP.md) - Detailed installation and configuration guide
- [scripts/create_astro_composite.py](scripts/create_astro_composite.py) - Full workflow example

## License

This project is for educational and research purposes.

## Credits

Built with:
- [Astropy](https://www.astropy.org/) - Astronomical data processing
- [Transformers](https://huggingface.co/transformers/) - Model inference
- [Unsloth](https://github.com/unslothai/unsloth) - Optimized inference library
- [unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit) - Vision model