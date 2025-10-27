"""Multi-band selection strategies for RGB composition.

This module provides strategies for selecting the optimal 3 bands from a larger
set of available filters (e.g., selecting from 10 filters for RGB composite).

Strategies:
- Maximum wavelength span: Choose bands with maximum separation
- PCA-based: Choose bands capturing most variance
- Science-driven: Pre-defined combinations for specific science goals
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Band selection strategy."""
    MAX_SPAN = 'max_span'
    PCA = 'pca'
    SCIENCE = 'science'
    MANUAL = 'manual'


@dataclass
class BandSelection:
    """Result of band selection.

    Attributes:
        red_band: Selected band for red channel
        green_band: Selected band for green channel
        blue_band: Selected band for blue channel
        strategy: Strategy used for selection
        all_bands: All available bands
        metadata: Additional selection metadata
    """
    red_band: str
    green_band: str
    blue_band: str
    strategy: SelectionStrategy
    all_bands: List[str]
    metadata: Optional[Dict] = None

    def __repr__(self):
        """Pretty representation."""
        return (
            f"BandSelection({self.strategy.value}: "
            f"R={self.red_band}, G={self.green_band}, B={self.blue_band} "
            f"from {len(self.all_bands)} bands)"
        )


class BandSelector:
    """Select optimal 3 bands from multiple available filters.

    This class implements various strategies for choosing which 3 bands
    to use when more than 3 filters are available.

    Example:
        >>> selector = BandSelector()
        >>> # Select from 10 JWST filters
        >>> selection = selector.select_bands(
        ...     bands=['f070w', 'f090w', 'f115w', 'f150w', 'f200w',
        ...            'f277w', 'f356w', 'f410m', 'f444w', 'f480m'],
        ...     wavelengths={'f070w': 704, 'f090w': 901, ...},
        ...     strategy='max_span'
        ... )
        >>> print(selection)  # BandSelection(max_span: R=f480m, G=f200w, B=f070w)
    """

    # Science-driven pre-defined selections for common goals
    SCIENCE_PRESETS = {
        'star_formation': {
            'description': 'Emphasize young stars, ionized gas, dust',
            'priorities': ['uv', 'ha', 'ir'],
            'wavelength_ranges': [(100, 400), (650, 660), (3000, 10000)]
        },
        'stellar_population': {
            'description': 'Age and metallicity tracers',
            'priorities': ['u', 'g', 'i'],
            'wavelength_ranges': [(300, 400), (400, 550), (700, 900)]
        },
        'agn_activity': {
            'description': 'Multi-wavelength AGN emission',
            'priorities': ['xray', 'ir', 'radio'],
            'wavelength_ranges': [(0.1, 10), (3000, 10000), (1e6, 1e9)]
        },
        'dust_features': {
            'description': 'Dust emission and PAH features',
            'priorities': ['mir', 'fir', 'submm'],
            'wavelength_ranges': [(5000, 25000), (25000, 350000), (350000, 1000000)]
        }
    }

    def __init__(self):
        """Initialize the BandSelector."""
        pass

    def select_bands(
        self,
        bands: List[str],
        wavelengths: Dict[str, float],
        strategy: Union[str, SelectionStrategy] = SelectionStrategy.MAX_SPAN,
        data: Optional[Dict[str, np.ndarray]] = None,
        science_goal: Optional[str] = None
    ) -> BandSelection:
        """Select 3 bands from available filters.

        Args:
            bands: List of available band names
            wavelengths: Dict mapping band names to wavelengths (nm)
            strategy: Selection strategy to use
            data: Optional dict of actual image data (needed for PCA)
            science_goal: Science goal for SCIENCE strategy

        Returns:
            BandSelection with chosen bands

        Raises:
            ValueError: If fewer than 3 bands available or strategy invalid

        Example:
            >>> # Maximum wavelength span
            >>> selection = selector.select_bands(
            ...     bands=['g', 'r', 'i', 'z', 'y'],
            ...     wavelengths={'g': 481, 'r': 617, 'i': 752, 'z': 866, 'y': 962},
            ...     strategy='max_span'
            ... )
        """
        if len(bands) < 3:
            raise ValueError(f"Need at least 3 bands, got {len(bands)}")

        if len(bands) == 3:
            # Only 3 bands, no selection needed
            logger.info("Exactly 3 bands available, no selection needed")
            sorted_bands = sorted(
                [(b, wavelengths[b]) for b in bands],
                key=lambda x: x[1],
                reverse=True
            )
            return BandSelection(
                red_band=sorted_bands[0][0],
                green_band=sorted_bands[1][0],
                blue_band=sorted_bands[2][0],
                strategy=SelectionStrategy.MANUAL,
                all_bands=bands,
                metadata={'note': 'Only 3 bands available'}
            )

        # Convert string to enum if needed
        if isinstance(strategy, str):
            try:
                strategy = SelectionStrategy(strategy.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    f"Available: {[s.value for s in SelectionStrategy]}"
                )

        # Apply selected strategy
        if strategy == SelectionStrategy.MAX_SPAN:
            return self._select_max_span(bands, wavelengths)
        elif strategy == SelectionStrategy.PCA:
            if data is None:
                raise ValueError("PCA strategy requires data parameter")
            return self._select_pca(bands, wavelengths, data)
        elif strategy == SelectionStrategy.SCIENCE:
            if science_goal is None:
                raise ValueError("SCIENCE strategy requires science_goal parameter")
            return self._select_science_driven(bands, wavelengths, science_goal)
        else:
            raise ValueError(f"Strategy {strategy} not implemented")

    def _select_max_span(
        self,
        bands: List[str],
        wavelengths: Dict[str, float]
    ) -> BandSelection:
        """Select bands with maximum wavelength separation.

        This chooses the shortest, longest, and middle wavelength bands
        to maximize color separation in the final RGB image.
        """
        # Filter to bands with known wavelengths
        valid_bands = {b: wl for b, wl in wavelengths.items() if b in bands}

        if len(valid_bands) < 3:
            raise ValueError(
                f"Need wavelengths for at least 3 bands. "
                f"Got {len(valid_bands)} from {len(bands)} total"
            )

        # Sort by wavelength
        sorted_bands = sorted(valid_bands.items(), key=lambda x: x[1], reverse=True)

        # Select longest, middle, shortest
        red_band = sorted_bands[0][0]  # Longest wavelength
        blue_band = sorted_bands[-1][0]  # Shortest wavelength
        green_band = sorted_bands[len(sorted_bands) // 2][0]  # Middle

        span = sorted_bands[0][1] - sorted_bands[-1][1]

        logger.info(
            f"Max-span selection from {len(bands)} bands: "
            f"R={red_band}({sorted_bands[0][1]:.0f}nm), "
            f"G={green_band}({sorted_bands[len(sorted_bands)//2][1]:.0f}nm), "
            f"B={blue_band}({sorted_bands[-1][1]:.0f}nm) "
            f"(span={span:.0f}nm)"
        )

        return BandSelection(
            red_band=red_band,
            green_band=green_band,
            blue_band=blue_band,
            strategy=SelectionStrategy.MAX_SPAN,
            all_bands=bands,
            metadata={
                'wavelength_span_nm': span,
                'red_wavelength': sorted_bands[0][1],
                'green_wavelength': sorted_bands[len(sorted_bands)//2][1],
                'blue_wavelength': sorted_bands[-1][1]
            }
        )

    def _select_pca(
        self,
        bands: List[str],
        wavelengths: Dict[str, float],
        data: Dict[str, np.ndarray]
    ) -> BandSelection:
        """Select bands using Principal Component Analysis.

        This method finds the 3 bands that capture the most variance
        in the data, which often correspond to the most informative
        combination for visualization.
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "PCA selection requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )

        # Ensure all bands have data
        valid_bands = [b for b in bands if b in data]
        if len(valid_bands) < 3:
            raise ValueError(f"Need data for at least 3 bands, got {len(valid_bands)}")

        # Stack data into matrix (pixels x bands)
        # Flatten each image and stack as columns
        logger.info(f"Running PCA on {len(valid_bands)} bands...")

        data_matrix = []
        band_order = []
        for band in valid_bands:
            img = data[band]
            if img is not None:
                data_matrix.append(img.flatten())
                band_order.append(band)

        data_matrix = np.array(data_matrix).T  # Shape: (n_pixels, n_bands)

        # Remove NaNs
        valid_pixels = ~np.isnan(data_matrix).any(axis=1)
        data_matrix_clean = data_matrix[valid_pixels]

        # Run PCA to find 3 principal components
        pca = PCA(n_components=3)
        pca.fit(data_matrix_clean)

        # Find bands closest to principal components
        # by computing correlation between each band and each PC
        pca_transformed = pca.transform(data_matrix_clean)

        selected_bands = []
        for i in range(3):
            pc = pca_transformed[:, i]
            correlations = []

            for j, band in enumerate(band_order):
                band_data = data_matrix_clean[:, j]
                corr = np.corrcoef(pc, band_data)[0, 1]
                correlations.append((abs(corr), band))

            # Select band with highest correlation to this PC
            correlations.sort(reverse=True)
            best_band = correlations[0][1]

            # Avoid duplicates
            if best_band not in selected_bands:
                selected_bands.append(best_band)
            else:
                # Take next best
                for corr, band in correlations[1:]:
                    if band not in selected_bands:
                        selected_bands.append(band)
                        break

        # Assign to R, G, B based on wavelength (longest to shortest)
        selected_with_wl = [
            (b, wavelengths.get(b, 0)) for b in selected_bands[:3]
        ]
        selected_with_wl.sort(key=lambda x: x[1], reverse=True)

        red_band = selected_with_wl[0][0]
        green_band = selected_with_wl[1][0]
        blue_band = selected_with_wl[2][0]

        variance_explained = pca.explained_variance_ratio_[:3].sum()

        logger.info(
            f"PCA selection: R={red_band}, G={green_band}, B={blue_band} "
            f"(variance explained: {variance_explained:.1%})"
        )

        return BandSelection(
            red_band=red_band,
            green_band=green_band,
            blue_band=blue_band,
            strategy=SelectionStrategy.PCA,
            all_bands=bands,
            metadata={
                'variance_explained': variance_explained,
                'principal_components': 3,
                'n_pixels_used': data_matrix_clean.shape[0]
            }
        )

    def _select_science_driven(
        self,
        bands: List[str],
        wavelengths: Dict[str, float],
        science_goal: str
    ) -> BandSelection:
        """Select bands based on science goals.

        Uses predefined wavelength ranges or filter priorities
        to choose bands that best highlight specific astrophysical features.
        """
        if science_goal not in self.SCIENCE_PRESETS:
            raise ValueError(
                f"Unknown science goal '{science_goal}'. "
                f"Available: {list(self.SCIENCE_PRESETS.keys())}"
            )

        preset = self.SCIENCE_PRESETS[science_goal]
        logger.info(f"Applying science-driven selection: {preset['description']}")

        # Match available bands to target wavelength ranges
        target_ranges = preset['wavelength_ranges']

        selected = {}
        for i, (wl_min, wl_max) in enumerate(target_ranges):
            channel = ['red', 'green', 'blue'][i]

            # Find band closest to this range
            best_band = None
            best_score = float('inf')

            for band in bands:
                if band not in wavelengths:
                    continue

                wl = wavelengths[band]

                # Score: distance from range center
                range_center = (wl_min + wl_max) / 2
                distance = abs(wl - range_center)

                # Prefer bands within range
                if wl_min <= wl <= wl_max:
                    distance *= 0.1  # Strong preference for in-range

                if distance < best_score:
                    best_score = distance
                    best_band = band

            if best_band:
                selected[channel] = best_band

        if len(selected) < 3:
            raise ValueError(
                f"Could not find suitable bands for science goal '{science_goal}'. "
                f"Only matched {len(selected)}/3 channels"
            )

        logger.info(
            f"Science selection ({science_goal}): "
            f"R={selected['red']}, G={selected['green']}, B={selected['blue']}"
        )

        return BandSelection(
            red_band=selected['red'],
            green_band=selected['green'],
            blue_band=selected['blue'],
            strategy=SelectionStrategy.SCIENCE,
            all_bands=bands,
            metadata={
                'science_goal': science_goal,
                'description': preset['description'],
                'target_ranges': target_ranges
            }
        )

    def list_science_goals(self) -> Dict[str, str]:
        """Get list of available science-driven presets.

        Returns:
            Dict mapping goal names to descriptions
        """
        return {
            name: preset['description']
            for name, preset in self.SCIENCE_PRESETS.items()
        }
