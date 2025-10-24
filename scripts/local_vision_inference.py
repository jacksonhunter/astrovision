from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Optional

from PIL import Image
from transformers import pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set the cache directory for Hugging Face models
# This will download models to ~/experiments/Models/huggingface
MODELS_DIR = Path.home() / "experiments" / "Models" / "huggingface"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)

# Model configuration
DEFAULT_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit"

class VisionModel:
    """Wrapper for local vision model inference."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the vision model pipeline.

        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.pipe = None
        logger.info(f"Initializing model: {model_name}")
        logger.info(f"Models will be cached in: {MODELS_DIR}")

    def load_model(self):
        """Load the model pipeline. Downloads model if not cached."""
        if self.pipe is None:
            logger.info("Loading model pipeline...")
            try:
                # Configure for CPU offload
                from transformers import AutoModelForVision2Seq, AutoProcessor

                model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    llm_int8_enable_fp32_cpu_offload=True,
                    load_in_8bit=True  # Use 8-bit for better CPU compatibility
                )
                processor = AutoProcessor.from_pretrained(self.model_name)

                self.pipe = pipeline(
                    "image-to-text",
                    model=model,
                    tokenizer=processor,
                    device_map="auto"
                )
                logger.info("Model loaded successfully with CPU offload")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def analyze_image(
        self,
        image_path: str | Path,
        prompt: Optional[str] = None,
        max_new_tokens: int = 200
    ) -> str:
        """Analyze an image and generate description.

        Args:
            image_path: Path to the image file
            prompt: Optional text prompt to guide the analysis
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text description of the image
        """
        if self.pipe is None:
            self.load_model()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Analyzing image: {image_path}")

        try:
            # Load image
            image = Image.open(image_path)

            # Generate description
            if prompt:
                # Some models support text prompts in the input
                result = self.pipe(
                    image,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens
                )
            else:
                result = self.pipe(
                    image,
                    max_new_tokens=max_new_tokens
                )

            # Extract text from result
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
            else:
                text = str(result)

            logger.info(f"Generated text length: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

def main():
    """Example usage of the vision model."""
    if len(sys.argv) < 2:
        print("Usage: python local_vision_inference.py <image_path> [prompt]")
        print("\nExample:")
        print("  python local_vision_inference.py image.jpg")
        print("  python local_vision_inference.py image.jpg 'Describe the colors in this image'")
        return 1

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else None

    # Initialize model
    model = VisionModel()

    try:
        # Analyze image
        result = model.analyze_image(image_path, prompt=prompt)

        print("\n" + "="*80)
        print("ANALYSIS RESULT:")
        print("="*80)
        print(result)
        print("="*80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())