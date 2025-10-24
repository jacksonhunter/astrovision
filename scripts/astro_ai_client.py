"""Client for remote AI with JSON structured responses.

Wraps the transformers_client pattern with JSON validation and retry logic.
"""

import json
import io
import base64
import re
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import requests
from PIL import Image


class AstroAIClient:
    """Client for astronomical image analysis with structured JSON responses."""

    def __init__(self, server_url: str = "http://localhost:5000", timeout: int = 600):
        """Initialize client.

        Args:
            server_url: URL of transformers server (via SSH tunnel)
            timeout: Request timeout in seconds (default 600 = 10 minutes for first model load)
        """
        self.server_url = server_url
        self.timeout = timeout

    def check_server(self) -> bool:
        """Check if server is accessible."""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _image_to_base64(self, image_array: np.ndarray) -> str:
        """Convert numpy array to base64 PNG."""
        # Ensure array is in the 0-255 uint8 range
        image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        # Create image and explicitly convert to RGB to handle both grayscale and color inputs
        img = Image.fromarray(image_uint8).convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def _analyze_with_prompt(
        self,
        image_array: np.ndarray,
        prompt: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Private helper to run analysis with a specific prompt and expect JSON.

        Args:
            image_array: Numpy array of the image to analyze.
            prompt: The full prompt to send to the model.
            max_tokens: The maximum number of tokens for the response.

        Returns:
            The structured dictionary from the 'analyze' method.
        """
        result = self.analyze(image_array, prompt, max_tokens=max_tokens, expect_json=True)
        return result

    def analyze(
        self,
        image_array: np.ndarray,
        prompt: str,
        max_tokens: int = 500,
        expect_json: bool = True
    ) -> Dict[str, Any]:
        """Analyze image with AI.

        Args:
            image_array: RGB image (H, W, 3) in 0-1 range
            prompt: Analysis prompt
            max_tokens: Max response tokens
            expect_json: If True, extract and parse JSON from response

        Returns:
            Dictionary with 'text' and optionally 'json' keys
        """
        image_b64 = self._image_to_base64(image_array)

        response = requests.post(
            f"{self.server_url}/generate",
            json={
                "image_base64": image_b64,
                "prompt": prompt,
                "max_tokens": max_tokens
            },
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )

        if response.status_code != 200:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            error_msg = error_data.get('error', response.text)
            raise Exception(f"Server error ({response.status_code}): {error_msg}")

        result = response.json()
        text = result.get("text", "")

        output = {"text": text, "raw": result}

        if expect_json:
            # Extract JSON from response (handles markdown code blocks)
            json_data = self._extract_json(text)
            if json_data:
                output["json"] = json_data
            else:
                output["json"] = None
                output["parse_error"] = "Could not extract valid JSON from response"

        return output

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract and parse JSON from text response.

        Handles:
        - Plain JSON
        - JSON in markdown code blocks (```json...```)
        - Finds the first valid JSON object or array in the text
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Code blocks (```json ... ``` or ``` ... ```)
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(code_block_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Use JSONDecoder to find first valid JSON object/array
        decoder = json.JSONDecoder()
        text = text.strip()
        for i in range(len(text)):
            if text[i] in '{[':
                try:
                    obj, _ = decoder.raw_decode(text, i)
                    return obj
                except json.JSONDecodeError:
                    continue

        return None

    def analyze_band(
        self,
        band_array: np.ndarray,
        band_name: str,
        wavelength: int
    ) -> Dict[str, Any]:
        """Analyze individual astronomical band.

        Returns structured JSON with:
        - quality: 1-10 rating
        - noise_level: low/medium/high
        - features: list of visible features
        - suitability_rgb: red/green/blue or none
        """
        # Create grayscale preview
        preview = np.dstack([band_array, band_array, band_array])

        prompt = f"""Analyze this {band_name} astronomical band ({wavelength}nm wavelength).

Respond ONLY with valid JSON in this exact format:
{{
  "quality": 7,
  "noise_level": "low",
  "dynamic_range": "good",
  "features_visible": ["stars", "nebulosity"],
  "brightness_distribution": "most pixels dark, bright stars",
  "best_rgb_channel": "red",
  "notes": "High quality data with good SNR"
}}

Do not include any text outside the JSON object."""

        return self._analyze_with_prompt(preview, prompt, max_tokens=300)

    def recommend_rgb_mapping(
        self,
        band_analyses: Dict[str, Dict],
        composite_preview: np.ndarray
    ) -> Dict[str, Any]:
        """Get AI recommendation for RGB channel mapping.

        Args:
            band_analyses: Dict of band analyses from analyze_band()
            composite_preview: Preview composite image

        Returns:
            Dict with:
            - red: band letter
            - green: band letter
            - blue: band letter
            - reasoning: explanation
        """
        # Format band info for prompt
        band_summary = "\n".join([
            f"{k}: quality={v.get('json', {}).get('quality', 'N/A')}, "
            f"best_channel={v.get('json', {}).get('best_rgb_channel', 'N/A')}"
            for k, v in band_analyses.items()
        ])

        prompt = f"""Given these astronomical band analyses:

{band_summary}

Recommend optimal RGB mapping following wavelength order (longer->red, shorter->blue).

Respond ONLY with valid JSON:
{{
  "red": "i",
  "green": "r",
  "blue": "g",
  "reasoning": "i-band (752nm) has longest wavelength for red, r-band (617nm) for green, g-band (481nm) shortest for blue"
}}

Do not include any text outside the JSON object."""

        return self._analyze_with_prompt(composite_preview, prompt, max_tokens=300)

    def assess_processing(
        self,
        composite: np.ndarray,
        processing_steps: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess quality of processing pipeline.

        Args:
            composite: Final processed composite
            processing_steps: Dict describing what was applied

        Returns:
            Dict with assessment of each step and recommendations
        """
        steps_text = "\n".join([f"- {k}: {v}" for k, v in processing_steps.items()])

        prompt = f"""Evaluate this processed astronomical composite.

Processing applied:
{steps_text}

Respond ONLY with valid JSON:
{{
  "zscale": {{"effective": true, "recommendation": "working well"}},
  "clahe": {{"effective": true, "suggested_clip_limit": 0.03, "notes": "good contrast"}},
  "unsharp_mask": {{"effective": true, "suggested_amount": 0.5}},
  "star_enhancement": {{"effective": true, "suggested_threshold": 99}},
  "luminance_masking": {{"effective": true, "suggested_percentile": 5}},
  "overall_quality": 8,
  "major_issues": [],
  "recommendations": ["Consider slight brightness boost"]
}}

Do not include any text outside the JSON object."""

        return self._analyze_with_prompt(composite, prompt, max_tokens=600)