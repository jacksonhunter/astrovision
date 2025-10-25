"""Channel mapping for RGB composite generation.

This module provides the ChannelMapper class for assigning wavelength bands
to RGB color channels using chromatic ordering.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

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
    """Map wavelength bands to RGB color channels.

    This class implements chromatic ordering - assigning longer wavelengths
    to redder channels - which creates intuitive false-color representations
    of multi-wavelength astronomical data.

    Example:
        >>> mapper = ChannelMapper()
        >>> # Auto-map PanSTARRS bands by wavelength
        >>> mapping = mapper.auto_map_by_wavelength({
        ...     'g': 481,  # nm
        ...     'r': 617,
        ...     'i': 752
        ... })
        >>> print(mapping)  # ChannelMapping(R=i@752nm, G=r@617nm, B=g@481nm)
    """

    def __init__(self):
        """Initialize the ChannelMapper."""
        pass

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
