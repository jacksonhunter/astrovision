"""Channel mapping for RGB composite generation.

This module provides the ChannelMapper class for assigning wavelength bands
to RGB color channels using chromatic ordering, narrowband palettes, and
advanced multi-band selection strategies.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import numpy as np

from .palette_mapper import PaletteMapper, PaletteMapping
from .band_selector import BandSelector, BandSelection, SelectionStrategy

logger = logging.getLogger(__name__)


@dataclass
class ChannelMapping:
    """Result of channel mapping.

    Attributes:
        red: Name/key of band assigned to red channel
        green: Name/key of band assigned to green channel
        blue: Name/key of band assigned to blue channel
        red_wavelength: Wavelength in nm for red channel
        green_wavelength: Wavelength in nm for green channel
        blue_wavelength: Wavelength in nm for blue channel
        chromatic_order: Whether mapping follows chromatic order (longest→red)
    """
    red: str
    green: str
    blue: str
    red_wavelength: Optional[float] = None
    green_wavelength: Optional[float] = None
    blue_wavelength: Optional[float] = None
    chromatic_order: bool = True

    def __repr__(self):
        """Pretty representation."""
        if self.red_wavelength:
            return (
                f"ChannelMapping(R={self.red}@{self.red_wavelength:.0f}nm, "
                f"G={self.green}@{self.green_wavelength:.0f}nm, "
                f"B={self.blue}@{self.blue_wavelength:.0f}nm)"
            )
        return f"ChannelMapping(R={self.red}, G={self.green}, B={self.blue})"


class ChannelMapper:
    """Map wavelength bands to RGB color channels with advanced capabilities.

    This class implements:
    - Chromatic ordering - assigning longer wavelengths to redder channels
    - Narrowband palette mapping (Hubble, HOO, etc.)
    - Multi-band selection from >3 filters
    - Custom false-color palette definitions

    Example:
        >>> mapper = ChannelMapper()
        >>>
        >>> # Auto-map PanSTARRS bands by wavelength
        >>> mapping = mapper.auto_map_by_wavelength({
        ...     'g': 481,  # nm
        ...     'r': 617,
        ...     'i': 752
        ... })
        >>> print(mapping)  # ChannelMapping(R=i@752nm, G=r@617nm, B=g@481nm)
        >>>
        >>> # Use Hubble palette for narrowband
        >>> mapping = mapper.map_narrowband(['ha', 'oiii', 'sii'], palette='hubble')
        >>>
        >>> # Select 3 from 10 bands using PCA
        >>> mapping = mapper.select_and_map(
        ...     bands=['f070w', 'f090w', ..., 'f480m'],
        ...     wavelengths={...},
        ...     data={...},
        ...     strategy='pca'
        ... )
    """

    def __init__(self):
        """Initialize the ChannelMapper with palette and selection support."""
        self.palette_mapper = PaletteMapper()
        self.band_selector = BandSelector()

    def auto_map_by_wavelength(
        self,
        bands: Union[Dict[str, float], List[str]],
        wavelength_lookup: Optional[Dict[str, float]] = None
    ) -> ChannelMapping:
        """Automatically map bands to RGB by wavelength (chromatic ordering).

        Assigns longest wavelength to red, shortest to blue, middle to green.

        Args:
            bands: Either:
                  - Dict mapping band names to wavelengths in nm
                  - List of band names (requires wavelength_lookup)
            wavelength_lookup: Optional wavelength database for band names

        Returns:
            ChannelMapping with assignments

        Raises:
            ValueError: If fewer than 3 bands or wavelengths cannot be determined

        Example:
            >>> # With wavelengths provided
            >>> mapping = mapper.auto_map_by_wavelength({
            ...     'F444W': 4421,
            ...     'F356W': 3568,
            ...     'F200W': 1989
            ... })
            >>> # R=F444W, G=F356W, B=F200W
            >>>
            >>> # With lookup table
            >>> mapping = mapper.auto_map_by_wavelength(
            ...     ['g', 'r', 'i'],
            ...     wavelength_lookup={'g': 481, 'r': 617, 'i': 752}
            ... )
        """
        # Convert list to dict using lookup
        if isinstance(bands, list):
            if wavelength_lookup is None:
                raise ValueError(
                    "wavelength_lookup required when bands is a list"
                )
            bands = {name: wavelength_lookup[name] for name in bands}

        if len(bands) < 3:
            raise ValueError(
                f"Need at least 3 bands for RGB mapping, got {len(bands)}"
            )

        # Sort bands by wavelength (longest to shortest)
        sorted_bands = sorted(bands.items(), key=lambda x: x[1], reverse=True)

        # Chromatic ordering: longest→red, middle→green, shortest→blue
        red_name, red_wl = sorted_bands[0]
        green_name, green_wl = sorted_bands[len(sorted_bands) // 2]
        blue_name, blue_wl = sorted_bands[-1]

        logger.info(
            f"Auto-mapped by wavelength: R={red_name}({red_wl:.0f}nm), "
            f"G={green_name}({green_wl:.0f}nm), B={blue_name}({blue_wl:.0f}nm)"
        )

        return ChannelMapping(
            red=red_name,
            green=green_name,
            blue=blue_name,
            red_wavelength=red_wl,
            green_wavelength=green_wl,
            blue_wavelength=blue_wl,
            chromatic_order=True
        )

    def manual_mapping(
        self,
        red: str,
        green: str,
        blue: str,
        wavelengths: Optional[Dict[str, float]] = None
    ) -> ChannelMapping:
        """Create a manual channel mapping.

        Args:
            red: Band name for red channel
            green: Band name for green channel
            blue: Band name for blue channel
            wavelengths: Optional wavelength lookup

        Returns:
            ChannelMapping with user-specified assignments

        Example:
            >>> # Artistic mapping (not chromatic order)
            >>> mapping = mapper.manual_mapping(
            ...     red='H-alpha',
            ...     green='OIII',
            ...     blue='SII'
            ... )
        """
        red_wl = wavelengths.get(red) if wavelengths else None
        green_wl = wavelengths.get(green) if wavelengths else None
        blue_wl = wavelengths.get(blue) if wavelengths else None

        # Check if chromatic order
        chromatic = True
        if all([red_wl, green_wl, blue_wl]):
            chromatic = red_wl > green_wl > blue_wl

        logger.info(f"Manual mapping: R={red}, G={green}, B={blue}")
        if not chromatic:
            logger.warning("Manual mapping does not follow chromatic order")

        return ChannelMapping(
            red=red,
            green=green,
            blue=blue,
            red_wavelength=red_wl,
            green_wavelength=green_wl,
            blue_wavelength=blue_wl,
            chromatic_order=chromatic
        )

    def validate_mapping(
        self,
        mapping: ChannelMapping,
        available_bands: List[str]
    ) -> Tuple[bool, List[str]]:
        """Validate that a mapping references existing bands.

        Args:
            mapping: ChannelMapping to validate
            available_bands: List of available band names

        Returns:
            Tuple of (is_valid, list_of_errors)

        Example:
            >>> mapping = ChannelMapping(red='i', green='r', blue='g')
            >>> valid, errors = mapper.validate_mapping(mapping, ['g', 'r', 'i'])
            >>> if not valid:
            ...     print(errors)
        """
        errors = []

        for channel, band in [('red', mapping.red),
                              ('green', mapping.green),
                              ('blue', mapping.blue)]:
            if band not in available_bands:
                errors.append(
                    f"{channel.capitalize()} channel '{band}' not in available bands: {available_bands}"
                )

        return (len(errors) == 0, errors)

    def suggest_mapping(
        self,
        available_bands: List[str],
        wavelengths: Dict[str, float]
    ) -> ChannelMapping:
        """Suggest a mapping given available bands and wavelengths.

        Convenience method that auto-maps whatever bands are available.

        Args:
            available_bands: List of band names
            wavelengths: Wavelength lookup table

        Returns:
            Suggested ChannelMapping

        Example:
            >>> bands = ['g', 'r', 'i', 'z']
            >>> wavelengths = {'g': 481, 'r': 617, 'i': 752, 'z': 866}
            >>> mapping = mapper.suggest_mapping(bands, wavelengths)
        """
        # Filter wavelengths to only available bands
        available_wavelengths = {
            band: wl for band, wl in wavelengths.items()
            if band in available_bands
        }

        if len(available_wavelengths) < 3:
            raise ValueError(
                f"Need at least 3 bands with known wavelengths, "
                f"got {len(available_wavelengths)}"
            )

        return self.auto_map_by_wavelength(available_wavelengths)

    def map_narrowband(
        self,
        bands: Union[List[str], Dict[str, Any]],
        palette: Union[str, Dict[str, str]] = 'hubble',
        wavelengths: Optional[Dict[str, float]] = None
    ) -> Union[ChannelMapping, PaletteMapping]:
        """Map narrowband data using a named or custom palette.

        Args:
            bands: Either list of band names or dict of band data
            palette: Palette name ('hubble', 'hoo', 'natural', 'mapped')
                    or custom dict mapping bands to colors
            wavelengths: Optional wavelength lookup

        Returns:
            ChannelMapping or PaletteMapping with assignments

        Example:
            >>> # Use Hubble palette (SHO: S-II→R, H-α→G, O-III→B)
            >>> mapping = mapper.map_narrowband(['sii', 'ha', 'oiii'], 'hubble')
            >>>
            >>> # Use HOO bicolor palette
            >>> mapping = mapper.map_narrowband(['ha', 'oiii'], 'hoo')
            >>>
            >>> # Custom palette
            >>> mapping = mapper.map_narrowband(
            ...     ['ha', 'oiii'],
            ...     palette={'ha': 'red', 'oiii': 'cyan'}
            ... )
        """
        palette_mapping = self.palette_mapper.map_narrowband(
            bands=bands,
            palette=palette,
            wavelengths=wavelengths
        )

        # Convert PaletteMapping to ChannelMapping for compatibility
        if isinstance(palette_mapping, PaletteMapping):
            logger.info(f"Narrowband palette applied: {palette_mapping}")
            return ChannelMapping(
                red=palette_mapping.red_band or '',
                green=palette_mapping.green_band or '',
                blue=palette_mapping.blue_band or '',
                chromatic_order=not palette_mapping.is_false_color
            )

        return palette_mapping

    def select_and_map(
        self,
        bands: List[str],
        wavelengths: Dict[str, float],
        strategy: Union[str, SelectionStrategy] = 'max_span',
        data: Optional[Dict[str, np.ndarray]] = None,
        science_goal: Optional[str] = None
    ) -> ChannelMapping:
        """Select optimal 3 bands from multiple available and map to RGB.

        This method is used when you have more than 3 bands available and need
        to intelligently choose which 3 to use for RGB composition.

        Args:
            bands: List of available band names (>3)
            wavelengths: Dict mapping band names to wavelengths (nm)
            strategy: Selection strategy ('max_span', 'pca', 'science')
            data: Optional image data dict (required for PCA)
            science_goal: Science goal name (required for 'science' strategy)

        Returns:
            ChannelMapping with selected and assigned bands

        Example:
            >>> # Select from 10 JWST filters using max wavelength span
            >>> mapping = mapper.select_and_map(
            ...     bands=['f070w', 'f090w', 'f115w', 'f150w', 'f200w',
            ...            'f277w', 'f356w', 'f410m', 'f444w', 'f480m'],
            ...     wavelengths={...},
            ...     strategy='max_span'
            ... )
            >>>
            >>> # Use PCA to find most informative combination
            >>> mapping = mapper.select_and_map(
            ...     bands=[...],
            ...     wavelengths={...},
            ...     data={'f070w': array(...), ...},
            ...     strategy='pca'
            ... )
        """
        if len(bands) < 3:
            raise ValueError(f"Need at least 3 bands, got {len(bands)}")

        if len(bands) == 3:
            # Only 3 bands, use chromatic ordering
            logger.info("Exactly 3 bands, using chromatic ordering")
            return self.auto_map_by_wavelength({b: wavelengths[b] for b in bands})

        # Select best 3 bands
        selection = self.band_selector.select_bands(
            bands=bands,
            wavelengths=wavelengths,
            strategy=strategy,
            data=data,
            science_goal=science_goal
        )

        logger.info(f"Selected bands using {selection.strategy.value}: {selection}")

        # Create mapping from selection
        selected_wavelengths = {
            selection.red_band: wavelengths[selection.red_band],
            selection.green_band: wavelengths[selection.green_band],
            selection.blue_band: wavelengths[selection.blue_band]
        }

        # Use manual mapping to preserve the selection's channel assignments
        return self.manual_mapping(
            red=selection.red_band,
            green=selection.green_band,
            blue=selection.blue_band,
            wavelengths=selected_wavelengths
        )

    def list_narrowband_palettes(self) -> Dict[str, str]:
        """List available narrowband palettes.

        Returns:
            Dict mapping palette names to descriptions
        """
        return self.palette_mapper.list_palettes()

    def list_science_goals(self) -> Dict[str, str]:
        """List available science-driven band selection presets.

        Returns:
            Dict mapping goal names to descriptions
        """
        return self.band_selector.list_science_goals()
