from __future__ import annotations
import base64  # for image encoding
import json
import logging
from pathlib import Path
import sys
from typing import List

from openai import OpenAI, APIConnectionError, APIStatusError

client = OpenAI()  # Reads OPENAI_API_KEY from environment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('theme_vision.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_image_colors(file_path: str) -> str:
    """Get color analysis from GPT-4 Vision."""
    logger.info(f"Processing image: {file_path}")
    
    prompt = """Analyze this image and create a representative pallet of the 8 most iconic colors of the image, along with two lighter hues of each and two darker hues of each, 24 colors total.  For each color I want:
    1. A descriptive name inspired by the images themes.
    2. The exact hex color value
    Format as JSON array of objects with 'name' and 'hex' properties.
    Example: [{"name": "Electric Blue", "hex": "#0066FF"}, ...]
    Include only the JSON, no other text."""
    
    # Log the request (without the base64 data to keep log readable)
    request_info = {
        "model": "gpt-4o-mini",
        "file_path": file_path,
        "mime_type": get_mime_type(file_path),
        "prompt": prompt
    }
    logger.info(f"API Request: {json.dumps(request_info, indent=2)}")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{get_mime_type(file_path)};base64,{encode_image(file_path)}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=16384
    )
    
    # Log the response
    response_content = response.choices[0].message.content
    response_info = {
        "file_path": file_path,
        "usage": {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        },
        "response_content": response_content
    }
    logger.info(f"API Response: {json.dumps(response_info, indent=2)}")
    
    return response_content

def get_mime_type(image_path: str) -> str:
    """Get MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg', 
        '.png': 'png',
        '.webp': 'webp',
        '.bmp': 'bmp'
    }
    return mime_map.get(ext, 'jpeg')

def encode_image(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def calculate_color_sort_value(hex_color: str) -> float:
    """Calculate red³ × blue² × green sorting value for a hex color."""
    # Remove # if present and convert to RGB
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return 0
    
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:6], 16)
        
        # Calculate red³ × blue² × green
        return (r ** 3) * (b ** 2) * g
    except ValueError:
        return 0

def process_folder(folder_path: str | Path, output_dir: str | Path) -> None:
    """Process all images in a folder and create palette JSON files."""
    folder_path = Path(folder_path)
    output_dir = Path(output_dir + "_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    # Store all palettes for the combined theme
    all_palettes = []
    
    # Process each image
    for img_path in folder_path.iterdir():
        if img_path.suffix.lower() in image_extensions:
            try:
                # Get color analysis from GPT-4 Vision
                json_str = analyze_image_colors(str(img_path))
                
                # Clean the response if it's wrapped in markdown code blocks
                if json_str.startswith("```json"):
                    json_str = json_str.strip("```json").strip("```").strip()
                elif json_str.startswith("```"):
                    json_str = json_str.strip("```").strip()
                
                palette = json.loads(json_str)
                
                # Save individual palette JSON
                palette_file = output_dir / f"{img_path.stem}_palette.json"
                with palette_file.open('w') as f:
                    json.dump(palette, f, indent=2)
                
                # Save individual palette markdown
                md_file = output_dir / f"{img_path.stem}_palette.md"
                # Calculate relative path from output dir to the image
                rel_path = Path("../theme_files") / img_path.name
                with md_file.open('w') as f:
                    f.write(f"# Color Palette: {img_path.name}\n\n")
                    f.write(f"![{img_path.name}]({rel_path})\n\n")
                    f.write("## Colors\n\n")
                    for i, color in enumerate(palette, 1):
                        f.write(f"{i}. **{color['name']}** - `{color['hex']}`\n")
                        f.write(f"   <div style=\"background-color: {color['hex']}; width: 50px; height: 20px; display: inline-block;\"></div>\n\n")
                
                all_palettes.extend(palette)
                print(f"Processed: {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
    
    # Create combined theme
    if all_palettes:
        # Sort all colors by red³ × blue² × green formula
        sorted_colors = sorted(
            all_palettes,
            key=lambda color: calculate_color_sort_value(color['hex']),
            reverse=True
        )
        
        # Save combined theme JSON
        with (output_dir / "combined_theme.json").open('w') as f:
            json.dump(sorted_colors, f, indent=2)
        
        # Save combined theme markdown
        with (output_dir / "combined_theme.md").open('w') as f:
            f.write("# Combined Theme Palette\n\n")
            f.write("All colors from processed images, sorted by red³ × blue² × green formula.\n\n")
            
            f.write("## All Colors\n\n")
            for i, color in enumerate(sorted_colors, 1):
                sort_value = calculate_color_sort_value(color['hex'])
                f.write(f"{i}. **{color['name']}** - `{color['hex']}` (sort value: {sort_value:,.0f})\n")
                f.write(f"   <div style=\"background-color: {color['hex']}; width: 50px; height: 20px; display: inline-block;\"></div>\n\n")

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python theme_vision.py <image-folder-path> [output-folder-path]")
        return 2

    input_folder = argv[1]
    output_folder = argv[2] if len(argv) > 2 else "palette_output"

    try:
        process_folder(input_folder, output_folder)
        print(f"\nPalettes saved to: {output_folder}")
        print("Check combined_theme.json for the overall theme.")
        return 0
    except APIStatusError as e:
        print(f"API error ({e.status_code}): {e.message}")
        return 1
    except APIConnectionError as e:
        print(f"Connection error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))