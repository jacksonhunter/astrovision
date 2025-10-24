"""Vision-guided composition using local AI model for aesthetic decisions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from scripts.local_vision_inference import VisionModel

logger = logging.getLogger(__name__)


class VisionGuidedCompositor:
    """Use vision AI to guide the creation of aesthetically optimal composites.

    This class leverages a local vision model to analyze preview composites
    and make recommendations for band assignments, color mappings, and
    processing parameters to create the most visually appealing result.
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the vision-guided compositor.

        Args:
            model_name: Optional specific model to use for vision inference
        """
        self.vision_model = VisionModel(model_name) if model_name else VisionModel()
        logger.info("Initialized VisionGuidedCompositor")

    def analyze_band_for_color(
        self,
        band_data: np.ndarray,
        band_metadata: dict,
        temp_preview_path: str | Path = "temp_preview.png"
    ) -> Dict[str, any]:
        """Analyze a single band to determine optimal color assignment.

        Args:
            band_data: Normalized band data (0-1 range)
            band_metadata: Metadata about the band (filter, wavelength, etc.)
            temp_preview_path: Temporary path for preview image

        Returns:
            Dictionary with color assignment recommendations
        """
        temp_preview_path = Path(temp_preview_path)

        # Save band as grayscale preview
        band_uint8 = (band_data * 255).astype(np.uint8)
        img = Image.fromarray(band_uint8, mode='L')
        img.save(temp_preview_path)

        # Create analysis prompt
        wavelength = band_metadata.get('wavelength', 'Unknown')
        filter_name = band_metadata.get('filter', 'Unknown')

        prompt = f"""Analyze this astronomical image band (Filter: {filter_name}, Wavelength: {wavelength}).

Consider:
1. What features are most prominent (nebulae, stars, dust, etc.)?
2. What color would best represent this data in a composite?
3. Should this be used for structure/detail (luminance) or color?

Respond in JSON format:
{{
    "prominent_features": ["feature1", "feature2"],
    "suggested_color": "red|green|blue|cyan|magenta|yellow|luminance",
    "reasoning": "brief explanation",
    "detail_level": "high|medium|low",
    "contrast_quality": "high|medium|low"
}}"""

        try:
            result_text = self.vision_model.analyze_image(
                temp_preview_path,
                prompt=prompt,
                max_new_tokens=300
            )

            # Parse JSON response
            result = self._parse_json_response(result_text)

            logger.info(f"Vision analysis for band: suggested_color={result.get('suggested_color')}")
            return result

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            # Return default recommendations
            return {
                "prominent_features": ["unknown"],
                "suggested_color": "luminance",
                "reasoning": "Analysis failed, using default",
                "detail_level": "medium",
                "contrast_quality": "medium"
            }
        finally:
            # Clean up temp file
            if temp_preview_path.exists():
                temp_preview_path.unlink()

    def recommend_band_mapping(
        self,
        bands: Dict[str, np.ndarray],
        band_metadata: Dict[str, dict],
        mode: str = 'rgb'
    ) -> Dict[str, str]:
        """Recommend optimal band-to-channel mapping using vision analysis.

        Args:
            bands: Dictionary of band name to data
            band_metadata: Metadata for each band
            mode: Composition mode ('rgb' or 'cmyk')

        Returns:
            Dictionary mapping channels to band names
            e.g., {'red': 'Band_1', 'green': 'Band_2', 'blue': 'Band_3'}
        """
        logger.info(f"Analyzing {len(bands)} bands for optimal {mode.upper()} mapping...")

        band_analyses = {}

        # Analyze each band
        for band_name, band_data in bands.items():
            metadata = band_metadata.get(band_name, {})
            analysis = self.analyze_band_for_color(band_data, metadata)
            band_analyses[band_name] = analysis

        # Determine optimal mapping based on analyses
        if mode == 'rgb':
            mapping = self._map_rgb_channels(band_analyses, bands)
        elif mode == 'cmyk':
            mapping = self._map_cmyk_channels(band_analyses, bands)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        logger.info(f"Recommended mapping: {mapping}")
        return mapping

    def _map_rgb_channels(
        self,
        analyses: Dict[str, dict],
        bands: Dict[str, np.ndarray]
    ) -> Dict[str, str]:
        """Map bands to RGB channels based on vision analysis.

        Args:
            analyses: Vision analysis results for each band
            bands: Band data

        Returns:
            RGB channel mapping
        """
        # Score each band for each channel
        red_scores = {}
        green_scores = {}
        blue_scores = {}

        for band_name, analysis in analyses.items():
            suggested = analysis.get('suggested_color', 'luminance')
            detail = analysis.get('detail_level', 'medium')
            contrast = analysis.get('contrast_quality', 'medium')

            # Score multipliers
            detail_score = {'high': 1.2, 'medium': 1.0, 'low': 0.8}[detail]
            contrast_score = {'high': 1.2, 'medium': 1.0, 'low': 0.8}[contrast]

            base_score = detail_score * contrast_score

            # Assign scores based on suggestion
            if suggested == 'red':
                red_scores[band_name] = base_score * 2.0
                green_scores[band_name] = base_score * 0.5
                blue_scores[band_name] = base_score * 0.5
            elif suggested == 'green':
                red_scores[band_name] = base_score * 0.5
                green_scores[band_name] = base_score * 2.0
                blue_scores[band_name] = base_score * 0.5
            elif suggested == 'blue':
                red_scores[band_name] = base_score * 0.5
                green_scores[band_name] = base_score * 0.5
                blue_scores[band_name] = base_score * 2.0
            elif suggested == 'cyan':
                red_scores[band_name] = base_score * 0.3
                green_scores[band_name] = base_score * 1.5
                blue_scores[band_name] = base_score * 1.5
            elif suggested == 'magenta':
                red_scores[band_name] = base_score * 1.5
                green_scores[band_name] = base_score * 0.3
                blue_scores[band_name] = base_score * 1.5
            elif suggested == 'yellow':
                red_scores[band_name] = base_score * 1.5
                green_scores[band_name] = base_score * 1.5
                blue_scores[band_name] = base_score * 0.3
            else:  # luminance or unknown
                red_scores[band_name] = base_score
                green_scores[band_name] = base_score
                blue_scores[band_name] = base_score

        # Select best bands for each channel
        mapping = {}
        used_bands = set()

        # Assign in order: red, green, blue
        for channel, scores in [('red', red_scores), ('green', green_scores), ('blue', blue_scores)]:
            # Find highest scoring unused band
            available = {k: v for k, v in scores.items() if k not in used_bands}
            if available:
                best_band = max(available, key=available.get)
                mapping[channel] = best_band
                used_bands.add(best_band)

        # Fill in any missing channels with remaining bands
        remaining = [b for b in bands.keys() if b not in used_bands]
        for channel in ['red', 'green', 'blue']:
            if channel not in mapping and remaining:
                mapping[channel] = remaining.pop(0)

        return mapping

    def _map_cmyk_channels(
        self,
        analyses: Dict[str, dict],
        bands: Dict[str, np.ndarray]
    ) -> Dict[str, str]:
        """Map bands to CMYK channels based on vision analysis.

        Args:
            analyses: Vision analysis results for each band
            bands: Band data

        Returns:
            CMYK channel mapping
        """
        # Score each band for CMYK channels
        cyan_scores = {}
        magenta_scores = {}
        yellow_scores = {}
        black_scores = {}

        for band_name, analysis in analyses.items():
            suggested = analysis.get('suggested_color', 'luminance')
            detail = analysis.get('detail_level', 'medium')

            detail_score = {'high': 1.5, 'medium': 1.0, 'low': 0.8}[detail]

            # CMYK assignment preferences
            if suggested == 'cyan' or suggested == 'blue':
                cyan_scores[band_name] = detail_score * 2.0
            elif suggested == 'magenta' or suggested == 'red':
                magenta_scores[band_name] = detail_score * 2.0
            elif suggested == 'yellow' or suggested == 'green':
                yellow_scores[band_name] = detail_score * 2.0
            elif suggested == 'luminance':
                black_scores[band_name] = detail_score * 2.0
            else:
                # Default scoring
                cyan_scores[band_name] = detail_score
                magenta_scores[band_name] = detail_score
                yellow_scores[band_name] = detail_score
                black_scores[band_name] = detail_score

        # Select best bands
        mapping = {}
        used_bands = set()

        for channel, scores in [('cyan', cyan_scores), ('magenta', magenta_scores),
                                ('yellow', yellow_scores), ('black', black_scores)]:
            available = {k: v for k, v in scores.items() if k not in used_bands}
            if available:
                best_band = max(available, key=available.get)
                mapping[channel] = best_band
                used_bands.add(best_band)

        # Fill missing channels
        remaining = [b for b in bands.keys() if b not in used_bands]
        for channel in ['cyan', 'magenta', 'yellow', 'black']:
            if channel not in mapping and remaining:
                mapping[channel] = remaining.pop(0)

        return mapping

    def optimize_composite_parameters(
        self,
        preview_rgb: np.ndarray,
        temp_preview_path: str | Path = "temp_composite_preview.png"
    ) -> Dict[str, any]:
        """Analyze a preview composite and suggest optimization parameters.

        Args:
            preview_rgb: Preview RGB composite (0-1 range)
            temp_preview_path: Temporary path for preview

        Returns:
            Dictionary of recommended processing parameters
        """
        temp_preview_path = Path(temp_preview_path)

        # Save preview
        preview_uint8 = (preview_rgb * 255).astype(np.uint8)
        img = Image.fromarray(preview_uint8, mode='RGB')
        img.save(temp_preview_path)

        prompt = """Analyze this astronomical composite image and suggest improvements.

Evaluate:
1. Overall contrast and brightness
2. Color balance and saturation
3. Detail visibility in nebulae and dust
4. Star prominence and definition
5. Any processing artifacts

Respond in JSON format:
{
    "contrast_adjustment": "increase|decrease|maintain",
    "brightness_adjustment": "increase|decrease|maintain",
    "color_saturation": "increase|decrease|maintain",
    "detail_enhancement": "more|less|maintain",
    "star_enhancement": "more|less|maintain",
    "overall_quality": "excellent|good|acceptable|needs_work",
    "specific_recommendations": ["recommendation1", "recommendation2"]
}"""

        try:
            result_text = self.vision_model.analyze_image(
                temp_preview_path,
                prompt=prompt,
                max_new_tokens=400
            )

            result = self._parse_json_response(result_text)
            logger.info(f"Composite optimization suggestions: quality={result.get('overall_quality')}")
            return result

        except Exception as e:
            logger.error(f"Optimization analysis failed: {e}")
            return {
                "contrast_adjustment": "maintain",
                "brightness_adjustment": "maintain",
                "color_saturation": "maintain",
                "detail_enhancement": "maintain",
                "star_enhancement": "maintain",
                "overall_quality": "unknown",
                "specific_recommendations": []
            }
        finally:
            if temp_preview_path.exists():
                temp_preview_path.unlink()

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from model response, handling markdown code blocks.

        Args:
            response_text: Raw response text from model

        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {}