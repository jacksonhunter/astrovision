"""Create stunning astronomical composites from FITS files using AI guidance.

This script demonstrates the full workflow of the astro-vision-composer library:
1. Load FITS file and extract bands
2. Use vision AI to recommend optimal band-to-color mapping
3. Generate enhanced composite with detail, contrast, and star improvements
4. Iteratively optimize based on vision feedback
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astro_vision_composer import (
    FITSImageProcessor,
    CompositeImageGenerator,
    VisionGuidedCompositor
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('astro_composite.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main workflow for creating AI-guided astronomical composites."""
    parser = argparse.ArgumentParser(
        description="Create stunning astronomical composites from FITS files using AI guidance"
    )
    parser.add_argument(
        'fits_file',
        type=str,
        help='Path to FITS file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='composite_output.png',
        help='Output file path (default: composite_output.png)'
    )
    parser.add_argument(
        '--mode',
        choices=['rgb', 'cmyk'],
        default='rgb',
        help='Composite mode (default: rgb)'
    )
    parser.add_argument(
        '--manual-mapping',
        type=str,
        help='Manual band mapping as JSON, e.g. {"red": "Band_0", "green": "Band_1", "blue": "Band_2"}'
    )
    parser.add_argument(
        '--no-ai',
        action='store_true',
        help='Skip AI analysis (use automatic mapping)'
    )
    parser.add_argument(
        '--enhance-contrast',
        action='store_true',
        default=True,
        help='Apply contrast enhancement (default: True)'
    )
    parser.add_argument(
        '--enhance-details',
        action='store_true',
        default=True,
        help='Apply detail enhancement (default: True)'
    )
    parser.add_argument(
        '--enhance-stars',
        action='store_true',
        default=True,
        help='Apply star enhancement (default: True)'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("ASTRONOMICAL COMPOSITE GENERATOR")
    logger.info("="*80)

    # Step 1: Load FITS file and extract bands
    logger.info(f"\nStep 1: Loading FITS file: {args.fits_file}")

    try:
        with FITSImageProcessor(args.fits_file) as fits_proc:
            # Extract all bands
            bands = fits_proc.extract_bands()
            band_metadata = fits_proc.get_band_info()

            if len(bands) == 0:
                logger.error("No image bands found in FITS file")
                return 1

            logger.info(f"Extracted {len(bands)} bands: {list(bands.keys())}")

            # Normalize bands
            normalized_bands = {}
            for band_name in bands.keys():
                normalized = fits_proc.normalize_band(band_name, method='zscale')
                normalized_bands[band_name] = normalized

            # Step 2: Determine band mapping
            logger.info(f"\nStep 2: Determining band-to-color mapping")

            if args.manual_mapping:
                # Use manual mapping
                import json
                mapping = json.loads(args.manual_mapping)
                logger.info(f"Using manual mapping: {mapping}")

            elif args.no_ai:
                # Use simple automatic mapping
                band_names = list(normalized_bands.keys())
                if args.mode == 'rgb':
                    mapping = {
                        'red': band_names[0] if len(band_names) > 0 else None,
                        'green': band_names[1] if len(band_names) > 1 else band_names[0],
                        'blue': band_names[2] if len(band_names) > 2 else band_names[0]
                    }
                else:  # cmyk
                    mapping = {
                        'cyan': band_names[0] if len(band_names) > 0 else None,
                        'magenta': band_names[1] if len(band_names) > 1 else band_names[0],
                        'yellow': band_names[2] if len(band_names) > 2 else band_names[0],
                        'black': band_names[3] if len(band_names) > 3 else band_names[0]
                    }
                logger.info(f"Using automatic mapping: {mapping}")

            else:
                # Use AI-guided mapping
                logger.info("Using AI vision model to analyze bands...")
                compositor = VisionGuidedCompositor()
                mapping = compositor.recommend_band_mapping(
                    normalized_bands,
                    band_metadata,
                    mode=args.mode
                )
                logger.info(f"AI-recommended mapping: {mapping}")

            # Step 3: Create composite
            logger.info(f"\nStep 3: Creating {args.mode.upper()} composite")

            generator = CompositeImageGenerator()

            # Add all normalized bands to generator
            for band_name, band_data in normalized_bands.items():
                generator.add_band(band_name, band_data)

            # Create composite based on mode
            if args.mode == 'rgb':
                composite = generator.create_rgb_composite(
                    r_band=mapping['red'],
                    g_band=mapping['green'],
                    b_band=mapping['blue'],
                    enhance_contrast=args.enhance_contrast,
                    enhance_details=args.enhance_details,
                    enhance_stars=args.enhance_stars
                )
            else:  # cmyk
                composite = generator.create_cmyk_composite(
                    cyan_band=mapping['cyan'],
                    magenta_band=mapping['magenta'],
                    yellow_band=mapping['yellow'],
                    black_band=mapping['black'],
                    enhance_contrast=args.enhance_contrast,
                    enhance_details=args.enhance_details,
                    enhance_stars=args.enhance_stars
                )

            # Step 4: Optional AI optimization feedback
            if not args.no_ai:
                logger.info(f"\nStep 4: Getting AI optimization feedback")
                try:
                    compositor = VisionGuidedCompositor()
                    optimization = compositor.optimize_composite_parameters(composite)
                    logger.info(f"AI Quality Assessment: {optimization.get('overall_quality', 'unknown')}")
                    logger.info(f"Recommendations: {optimization.get('specific_recommendations', [])}")
                except Exception as e:
                    logger.warning(f"AI optimization feedback failed: {e}")

            # Step 5: Save composite
            logger.info(f"\nStep 5: Saving composite to: {args.output}")
            generator.save_composite(composite, args.output)

            logger.info("\n" + "="*80)
            logger.info("COMPOSITE GENERATION COMPLETE")
            logger.info("="*80)
            logger.info(f"Output saved to: {args.output}")

            return 0

    except Exception as e:
        logger.error(f"Error during composite generation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())