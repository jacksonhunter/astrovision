"""Updated client using new server API with context_building strategy.

Uses few-shot learning pattern:
- Early messages: Explicit format instructions
- Later messages: Vague prompts (rely on learned pattern)
"""

import json
import base64
import requests
import numpy as np
from io import BytesIO
from typing import Dict, List, Any
from PIL import Image


class AstroAIClientV2:
    """Client for astronomical image analysis using context_building."""

    def __init__(self, server_url: str = "http://localhost:5000", timeout: int = 600):
        self.server_url = server_url
        self.timeout = timeout

    def check_server(self) -> bool:
        """Check if server is accessible."""
        try:
            response = requests.get(self.server_url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def _numpy_to_base64(self, array: np.ndarray) -> str:
        """Convert numpy array to base64 PNG."""
        img_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).convert('RGB')
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response text."""
        import re

        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass

        # Look for JSON in code blocks
        code_block = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except:
                pass

        # Look for JSON object
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass

        return None

    def analyze_bands(
        self,
        bands: Dict[str, np.ndarray],
        band_info: Dict[str, Dict]
    ) -> List[Dict]:
        """Analyze multiple astronomical bands using few-shot learning.

        Args:
            bands: Dict of band_name -> normalized image array
            band_info: Dict of band_name -> {"wavelength": int, "name": str}

        Returns:
            List of analysis results (one per band)
        """
        messages = []
        band_names = list(bands.keys())

        for idx, band_name in enumerate(band_names):
            band_array = bands[band_name]
            info = band_info[band_name]

            # Convert to grayscale preview
            preview = np.dstack([band_array, band_array, band_array])
            image_b64 = self._numpy_to_base64(preview)

            # First 2 bands: EXPLICIT format
            if idx < 2:
                text = (
                    f'Analyze this {info["name"]} astronomical band ({info["wavelength"]}nm wavelength). '
                    f'Respond in JSON format: {{'
                    f'"quality": <number 1-10>, '
                    f'"noise_level": "<low/medium/high>", '
                    f'"features_visible": [<list of features you see>], '
                    f'"brightness": "<dark/medium/bright>", '
                    f'"notes": "<brief assessment>"}}'
                )
            # Remaining bands: VAGUE (rely on examples)
            else:
                text = f'Analyze this {info["name"]} band ({info["wavelength"]}nm) in the same JSON format'

            messages.append({
                "image_base64": image_b64,
                "text": text
            })

        # Send all at once with context_building
        response = requests.post(
            f"{self.server_url}/generate",
            json={
                "messages": messages,
                "max_tokens": 200,
                "strategy": "context_building"
            },
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise Exception(f"Server error: {response.status_code} - {response.text}")

        result = response.json()

        # Parse results
        analyses = []
        for idx, (band_name, res) in enumerate(zip(band_names, result["results"])):
            answer = res['result'].split('assistant\n')[-1].strip()
            parsed = self._extract_json(answer)

            analyses.append({
                "band": band_name,
                "raw_text": answer,
                "json": parsed,
                "success": parsed is not None
            })

        return analyses

    def recommend_rgb_mapping(
        self,
        band_analyses: List[Dict],
        preview_composite: np.ndarray,
        available_bands: List[str]
    ) -> Dict:
        """Get RGB mapping recommendation using few-shot learning.

        Uses 2 explicit examples, then vague query.
        """
        # Create example composites (dummy for teaching format)
        dummy_composite = np.zeros((100, 100, 3))

        messages = [
            # Example 1: EXPLICIT format
            {
                "image_base64": self._numpy_to_base64(dummy_composite),
                "text": (
                    'For these bands: g(481nm), r(617nm), i(752nm) - recommend RGB mapping following wavelength order. '
                    'Respond in JSON: {"red": "<band>", "green": "<band>", "blue": "<band>", "reasoning": "<explanation>"}'
                )
            },
            # Example 2: REINFORCE
            {
                "image_base64": self._numpy_to_base64(dummy_composite),
                "text": (
                    'For these bands: r(617nm), i(752nm), z(866nm) - recommend RGB mapping following wavelength order. '
                    'Respond in JSON: {"red": "<band>", "green": "<band>", "blue": "<band>", "reasoning": "<explanation>"}'
                )
            },
            # Query: VAGUE
            {
                "image_base64": self._numpy_to_base64(preview_composite),
                "text": f'For these bands: {", ".join(available_bands)} - same RGB mapping format'
            }
        ]

        response = requests.post(
            f"{self.server_url}/generate",
            json={
                "messages": messages,
                "max_tokens": 200,
                "strategy": "context_building"
            },
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise Exception(f"Server error: {response.status_code}")

        result = response.json()

        # Get last result (the actual query)
        answer = result["results"][-1]['result'].split('assistant\n')[-1].strip()
        parsed = self._extract_json(answer)

        return {
            "raw_text": answer,
            "json": parsed,
            "success": parsed is not None
        }
