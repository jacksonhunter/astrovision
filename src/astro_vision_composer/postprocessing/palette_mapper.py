"""Advanced palette mapping for narrowband and false-color imaging.

This module provides sophisticated palette mapping capabilities for:
- Narrowband imaging (Ha/OIII/SII) with standard palettes (Hubble, HOO, etc.)
- Multi-band selection when >3 filters are available
- Custom false-color palette definitions
- Per-channel normalization strategies
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ColorChannel(Enum):
    """RGB color channel designation."""
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    CYAN = 'cyan'  # Synthesized from green + blue
    MAGENTA = 'magenta'  # Synthesized from red + blue
    YELLOW = 'yellow'  # Synthesized from red + green


@dataclass
class PaletteMapping:
    """Result of palette-based channel mapping.

    Attributes:
        red_band: Band name for red channel
        green_band: Band name for green channel
        blue_band: Band name for blue channel
        palette_name: Name of palette used
        is_false_color: Whether this is false-color (non-chromatic)
        synthetic_channels: Dict of synthetic color channels (e.g., cyan from OIII)
        notes: Human-readable description of the mapping
    """
    red_band: Optional[str]
    green_band: Optional[str]
    blue_band: Optional[str]
    palette_name: str
    is_false_color: bool = True
    synthetic_channels: Optional[Dict[str, str]] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Initialize synthetic channels dict if not provided."""
        if self.synthetic_channels is None:
            self.synthetic_channels = {}

    def __repr__(self):
        """Pretty representation."""
        channels = []
        if self.red_band:
            channels.append(f"R={self.red_band}")
        if self.green_band:
            channels.append(f"G={self.green_band}")
        if self.blue_band:
            channels.append(f"B={self.blue_band}")

        channel_str = ", ".join(channels)
        return f"PaletteMapping({self.palette_name}: {channel_str})"


class PaletteMapper:
    """Map bands to RGB channels using standard or custom palettes.

    This class implements standard narrowband imaging palettes and supports
    creating custom false-color mappings for scientific visualization.

    Standard Palettes:
        - Hubble (SHO): S-II→Red, H-α→Green, O-III→Blue
        - HOO: H-α→Red, O-III→Cyan (synthesized)
        - Natural narrowband: Wavelength-aware Ha/OIII/SII mix
        - Chromatic: Standard wavelength ordering

    Example:
        >>> mapper = PaletteMapper()
        >>> # Use Hubble palette for narrowband
        >>> mapping = mapper.map_narrowband(
        ...     bands={'ha': data_ha, 'oiii': data_oiii, 'sii': data_sii},
        ...     palette='hubble'
        ... )
        >>> print(mapping)  # PaletteMapping(hubble: R=sii, G=ha, B=oiii)
    """

    # Standard narrowband palettes
    NARROWBAND_PALETTES = {
        'hubble': {
            'description': 'Hubble palette (SHO): S-II→R, H-α→G, O-III→B',
            'mapping': {
                ColorChannel.RED: ['sii', 's2', 'sulfur', 'sul'],
                ColorChannel.GREEN: ['ha', 'halpha', 'h-alpha', 'hydrogen'],
                ColorChannel.BLUE: ['oiii', 'o3', 'oxygen']
            },
            'is_false_color': True,
            'notes': 'Classic Hubble narrowband palette emphasizing sulfur/hydrogen/oxygen'
        },
        'hoo': {
            'description': 'H-α→Red, O-III→Cyan (bicolor)',
            'mapping': {
                ColorChannel.RED: ['ha', 'halpha', 'h-alpha', 'hydrogen'],
                ColorChannel.CYAN: ['oiii', 'o3', 'oxygen']  # Synthesized G+B
            },
            'is_false_color': True,
            'notes': 'Bicolor palette popular for emission nebulae, O-III creates cyan'
        },
        'natural': {
            'description': 'Natural narrowband: H-α→R, O-III→Cyan, S-II→R (mixed)',
            'mapping': {
                ColorChannel.RED: ['ha', 'halpha', 'h-alpha', 'sii', 's2'],
                ColorChannel.GREEN: ['oiii', 'o3'],
                ColorChannel.BLUE: ['oiii', 'o3']
            },
            'is_false_color': False,
            'notes': 'More natural-looking narrowband with wavelength consideration'
        },
        'mapped': {
            'description': 'Wavelength-mapped: assigns based on actual wavelength',
            'mapping': None,  # Computed dynamically
            'is_false_color': False,
            'notes': 'Uses actual wavelengths to assign channels chromatically'
        }
    }

    # Standard filter wavelengths (nm) for common narrowband filters
    NARROWBAND_WAVELENGTHS = {
        # Hydrogen
        'ha': 656.3,
        'halpha': 656.3,
        'h-alpha': 656.3,
        'hydrogen': 656.3,

        # Oxygen
        'oiii': 500.7,
        'o3': 500.7,
        'oxygen': 500.7,

        # Sulfur
        'sii': 672.4,
        's2': 672.4,
        'sulfur': 672.4,
        'sul': 672.4,

        # Nitrogen
        'nii': 658.3,
        'n2': 658.3,
        'nitrogen': 658.3
    }

    def __init__(self):
        """Initialize the PaletteMapper."""
        pass

    def map_narrowband(
        self,
        bands: Dict[str, Any],
        palette: Union[str, Dict[str, str]] = 'hubble',
        wavelengths: Optional[Dict[str, float]] = None
    ) -> PaletteMapping:
        """Map narrowband data to RGB using a named or custom palette.

        Args:
            bands: Dict of band names to data arrays or just band names
            palette: Either:
                - Named palette string ('hubble', 'hoo', 'natural', 'mapped')
                - Custom dict mapping band names to color channels
                  e.g., {'ha': 'red', 'oiii': 'cyan', 'sii': 'blue'}
            wavelengths: Optional wavelength lookup for bands

        Returns:
            PaletteMapping with channel assignments

        Raises:
            ValueError: If palette is unknown or required bands are missing

        Example:
            >>> # Use Hubble palette
            >>> mapping = mapper.map_narrowband(
            ...     bands=['sii', 'ha', 'oiii'],
            ...     palette='hubble'
            ... )
            >>>
            >>> # Use custom palette
            >>> mapping = mapper.map_narrowband(
            ...     bands=['ha', 'oiii'],
            ...     palette={'ha': 'red', 'oiii': 'cyan'}
            ... )
        """
        # Extract band names if dict provided
        band_names = list(bands.keys()) if isinstance(bands, dict) else bands

        # Handle custom palette dict
        if isinstance(palette, dict):
            return self._apply_custom_palette(band_names, palette)

        # Handle named palettes
        if palette not in self.NARROWBAND_PALETTES:
            raise ValueError(
                f"Unknown palette '{palette}'. Available: {list(self.NARROWBAND_PALETTES.keys())}"
            )

        palette_def = self.NARROWBAND_PALETTES[palette]

        # Special case: wavelength-mapped palette
        if palette == 'mapped':
            return self._apply_wavelength_mapped(band_names, wavelengths)

        # Apply standard narrowband palette
        return self._apply_standard_palette(band_names, palette, palette_def)

    def _normalize_band_name(self, band: str) -> str:
        """Normalize band name for matching (lowercase, no special chars)."""
        return band.lower().replace('-', '').replace('_', '')

    def _find_band_for_channel(
        self,
        available_bands: List[str],
        channel_patterns: List[str]
    ) -> Optional[str]:
        """Find a band that matches any of the channel patterns."""
        normalized_available = {
            self._normalize_band_name(b): b for b in available_bands
        }

        for pattern in channel_patterns:
            normalized_pattern = self._normalize_band_name(pattern)
            if normalized_pattern in normalized_available:
                return normalized_available[normalized_pattern]

        return None

    def _apply_standard_palette(
        self,
        band_names: List[str],
        palette_name: str,
        palette_def: Dict
    ) -> PaletteMapping:
        """Apply a standard narrowband palette."""
        mapping_spec = palette_def['mapping']

        red_band = None
        green_band = None
        blue_band = None
        synthetic = {}

        for color_channel, patterns in mapping_spec.items():
            matched_band = self._find_band_for_channel(band_names, patterns)

            if color_channel == ColorChannel.RED:
                red_band = matched_band
            elif color_channel == ColorChannel.GREEN:
                green_band = matched_band
            elif color_channel == ColorChannel.BLUE:
                blue_band = matched_band
            elif color_channel == ColorChannel.CYAN:
                # Cyan is synthesized from same data in G and B
                if matched_band:
                    green_band = matched_band
                    blue_band = matched_band
                    synthetic['cyan'] = matched_band

        # Check if we have enough bands
        if not any([red_band, green_band, blue_band]):
            raise ValueError(
                f"No matching bands found for palette '{palette_name}'. "
                f"Available: {band_names}"
            )

        logger.info(
            f"Applied {palette_name} palette: R={red_band}, G={green_band}, B={blue_band}"
        )
        if synthetic:
            logger.info(f"  Synthetic channels: {synthetic}")

        return PaletteMapping(
            red_band=red_band,
            green_band=green_band,
            blue_band=blue_band,
            palette_name=palette_name,
            is_false_color=palette_def['is_false_color'],
            synthetic_channels=synthetic,
            notes=palette_def['notes']
        )

    def _apply_custom_palette(
        self,
        band_names: List[str],
        palette_dict: Dict[str, str]
    ) -> PaletteMapping:
        """Apply a custom user-defined palette."""
        red_band = None
        green_band = None
        blue_band = None
        synthetic = {}

        for band, color in palette_dict.items():
            if band not in band_names:
                logger.warning(f"Band '{band}' in palette not found in available bands")
                continue

            color_lower = color.lower()
            if color_lower == 'red':
                red_band = band
            elif color_lower == 'green':
                green_band = band
            elif color_lower == 'blue':
                blue_band = band
            elif color_lower == 'cyan':
                # Synthesize cyan from band
                green_band = band
                blue_band = band
                synthetic['cyan'] = band
            elif color_lower == 'magenta':
                red_band = band if not red_band else red_band
                blue_band = band
                synthetic['magenta'] = band
            elif color_lower == 'yellow':
                red_band = band
                green_band = band if not green_band else green_band
                synthetic['yellow'] = band

        logger.info(f"Applied custom palette: R={red_band}, G={green_band}, B={blue_band}")

        return PaletteMapping(
            red_band=red_band,
            green_band=green_band,
            blue_band=blue_band,
            palette_name='custom',
            is_false_color=True,
            synthetic_channels=synthetic,
            notes='User-defined custom palette'
        )

    def _apply_wavelength_mapped(
        self,
        band_names: List[str],
        wavelengths: Optional[Dict[str, float]]
    ) -> PaletteMapping:
        """Apply wavelength-based chromatic mapping."""
        # Merge provided wavelengths with defaults
        wl_lookup = dict(self.NARROWBAND_WAVELENGTHS)
        if wavelengths:
            wl_lookup.update(wavelengths)

        # Find wavelengths for available bands
        band_wavelengths = {}
        for band in band_names:
            normalized = self._normalize_band_name(band)
            if normalized in wl_lookup:
                band_wavelengths[band] = wl_lookup[normalized]
            elif band in wl_lookup:
                band_wavelengths[band] = wl_lookup[band]

        if len(band_wavelengths) < 3:
            raise ValueError(
                f"Need wavelengths for at least 3 bands. "
                f"Found {len(band_wavelengths)} with wavelengths from {len(band_names)} bands"
            )

        # Sort by wavelength (longest to shortest)
        sorted_bands = sorted(band_wavelengths.items(), key=lambda x: x[1], reverse=True)

        red_band = sorted_bands[0][0]
        green_band = sorted_bands[len(sorted_bands) // 2][0]
        blue_band = sorted_bands[-1][0]

        logger.info(
            f"Applied wavelength mapping: R={red_band}({sorted_bands[0][1]:.1f}nm), "
            f"G={green_band}({sorted_bands[len(sorted_bands)//2][1]:.1f}nm), "
            f"B={blue_band}({sorted_bands[-1][1]:.1f}nm)"
        )

        return PaletteMapping(
            red_band=red_band,
            green_band=green_band,
            blue_band=blue_band,
            palette_name='wavelength_mapped',
            is_false_color=False,
            notes='Chromatic ordering by wavelength'
        )

    def list_palettes(self) -> Dict[str, str]:
        """Get list of available palettes with descriptions.

        Returns:
            Dict mapping palette names to descriptions
        """
        return {
            name: info['description']
            for name, info in self.NARROWBAND_PALETTES.items()
        }
