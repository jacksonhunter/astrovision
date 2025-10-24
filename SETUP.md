# Local Vision Model Setup

This project uses the **Llama-3.2-11B-Vision-Instruct-bnb-4bit** model for local image analysis inference.

## Installation

### 1. Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Or using the project directly:
```bash
pip install -e .
```

### 2. Verify Model Cache Location

Models will be automatically downloaded to:
```
~/experiments/Models/huggingface/
```

The first run will download the model (~11GB for the 4-bit quantized version). Subsequent runs will use the cached model.

## Usage

### Basic Usage

Analyze an image:
```bash
python scripts/local_vision_inference.py path/to/image.jpg
```

### With Custom Prompt

Guide the analysis with a specific prompt:
```bash
python scripts/local_vision_inference.py path/to/image.jpg "Describe the colors and composition"
```

### Python API Usage

```python
from scripts.local_vision_inference import VisionModel

# Initialize model
model = VisionModel()

# Analyze an image
result = model.analyze_image("path/to/image.jpg")
print(result)

# With custom prompt
result = model.analyze_image(
    "path/to/image.jpg",
    prompt="Extract the main colors from this image",
    max_new_tokens=300
)
print(result)
```

## Model Details

- **Model**: `unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit`
- **Type**: Vision-Language Model (VLM)
- **Quantization**: 4-bit (unsloth-optimized bnb) for reduced memory usage
- **Size**: ~11GB download
- **Memory Requirements**: ~8-12GB RAM/VRAM recommended
- **Library**: Uses unsloth for optimized inference

## Hardware Requirements

### Minimum
- **RAM**: 8GB
- **Storage**: 15GB free space (for model + cache)
- **CPU**: Multi-core processor

### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA support)
- **RAM**: 16GB+
- **Storage**: SSD with 20GB+ free space

### GPU Acceleration

The script automatically uses GPU if available via `device_map="auto"`. To force CPU:
```python
model.pipe = pipeline("image-to-text", model=model_name, device_map="cpu")
```

## Troubleshooting

### Out of Memory Errors
- Close other applications
- Use CPU instead of GPU
- Consider using a smaller model

### Slow Download
- First download can take 10-30 minutes depending on connection
- Model is cached, subsequent runs are instant

### CUDA Errors
- Ensure PyTorch CUDA version matches your NVIDIA driver
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Windows-Specific Notes

### bitsandbytes on Windows
The `bitsandbytes` library may require additional setup on Windows. If you encounter issues:

1. Install the Windows-compatible version:
```bash
pip install bitsandbytes-windows
```

2. Or use the CPU-only version by removing `bitsandbytes` from requirements and using non-quantized models.

## Log Files

Inference logs are saved to:
- `vision_inference.log` - Detailed logging of all operations

## Next Steps

- Integrate with `theme_vision.py` for local color extraction
- Create batch processing scripts
- Add support for multiple vision models