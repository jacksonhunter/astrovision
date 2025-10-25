"""Processing history tracking for reproducibility.

This module provides the HistoryTracker class for recording all processing
steps applied to astronomical images, enabling reproducibility and documentation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStep:
    """Record of a single processing step.

    Attributes:
        timestamp: When the step was performed
        operation: Name of the operation (e.g., 'normalize', 'stretch', 'reproject')
        parameters: Dictionary of parameters used
        component: Which component performed the operation
        notes: Optional additional notes
    """
    timestamp: str
    operation: str
    parameters: Dict[str, Any]
    component: str
    notes: Optional[str] = None

    def to_string(self) -> str:
        """Convert to human-readable string."""
        params_str = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
        base = f"[{self.timestamp}] {self.component}.{self.operation}({params_str})"
        if self.notes:
            base += f" - {self.notes}"
        return base

    def to_fits_comment(self) -> str:
        """Convert to FITS-compatible HISTORY comment (max 80 chars)."""
        params_str = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
        comment = f"{self.component}.{self.operation}({params_str})"
        if len(comment) > 70:  # Leave room for "HISTORY " prefix
            comment = comment[:67] + "..."
        return comment


class HistoryTracker:
    """Track processing history for reproducibility.

    Records all operations applied to data, including normalization,
    stretching, reprojection, and composition. History can be exported
    as text, JSON, or FITS header comments.

    Example:
        >>> tracker = HistoryTracker()
        >>> tracker.record('normalize', {'method': 'zscale'}, 'Normalizer')
        >>> tracker.record('stretch', {'method': 'asinh', 'a': 0.1}, 'Stretcher')
        >>> print(tracker.to_text())
    """

    def __init__(self):
        """Initialize the HistoryTracker."""
        self.steps: List[ProcessingStep] = []

    def record(
        self,
        operation: str,
        parameters: Dict[str, Any],
        component: str,
        notes: Optional[str] = None
    ) -> None:
        """Record a processing step.

        Args:
            operation: Name of the operation
            parameters: Dictionary of parameters used
            component: Component that performed the operation
            notes: Optional additional notes

        Example:
            >>> tracker.record(
            ...     operation='reproject',
            ...     parameters={'method': 'interp', 'order': 1},
            ...     component='Reprojector',
            ...     notes='Aligned to i-band reference'
            ... )
        """
        step = ProcessingStep(
            timestamp=datetime.now().isoformat(timespec='seconds'),
            operation=operation,
            parameters=parameters.copy(),  # Copy to prevent mutation
            component=component,
            notes=notes
        )
        self.steps.append(step)

        logger.debug(f"Recorded: {step.to_string()}")

    def record_from_object(
        self,
        obj: Any,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a step, automatically extracting component name from object.

        Args:
            obj: Object that performed the operation
            operation: Name of the operation
            parameters: Optional parameters (extracted from object if not provided)

        Example:
            >>> normalizer = Normalizer()
            >>> # After normalization
            >>> tracker.record_from_object(
            ...     normalizer,
            ...     'normalize',
            ...     {'method': 'zscale'}
            ... )
        """
        component = obj.__class__.__name__
        params = parameters or {}
        self.record(operation, params, component)

    def get_history(self) -> List[ProcessingStep]:
        """Get list of all processing steps.

        Returns:
            List of ProcessingStep objects
        """
        return self.steps.copy()

    def to_text(self, include_timestamps: bool = True) -> str:
        """Export history as plain text.

        Args:
            include_timestamps: Include timestamps in output

        Returns:
            Multi-line string with processing history

        Example:
            >>> print(tracker.to_text())
            [2025-10-25T14:30:00] Normalizer.normalize(method=zscale)
            [2025-10-25T14:30:01] Stretcher.stretch(method=asinh, a=0.1)
            [2025-10-25T14:30:02] Compositor.create_lupton_rgb(stretch=0.5, Q=8)
        """
        if not self.steps:
            return "No processing history recorded"

        lines = []
        for step in self.steps:
            if include_timestamps:
                lines.append(step.to_string())
            else:
                params_str = ', '.join(f"{k}={v}" for k, v in step.parameters.items())
                line = f"{step.component}.{step.operation}({params_str})"
                if step.notes:
                    line += f" - {step.notes}"
                lines.append(line)

        return '\n'.join(lines)

    def to_fits_header(self) -> List[str]:
        """Export history as FITS HISTORY comments.

        Returns:
            List of strings suitable for FITS HISTORY cards

        Example:
            >>> from astropy.io import fits
            >>> header = fits.Header()
            >>> for comment in tracker.to_fits_header():
            ...     header['HISTORY'] = comment
        """
        if not self.steps:
            return []

        comments = []
        comments.append("Processing history:")
        for step in self.steps:
            comments.append(step.to_fits_comment())

        return comments

    def to_dict(self) -> Dict[str, Any]:
        """Export history as dictionary (JSON-serializable).

        Returns:
            Dictionary with processing history

        Example:
            >>> import json
            >>> history_dict = tracker.to_dict()
            >>> json.dump(history_dict, open('history.json', 'w'), indent=2)
        """
        return {
            'steps': [
                {
                    'timestamp': step.timestamp,
                    'operation': step.operation,
                    'parameters': step.parameters,
                    'component': step.component,
                    'notes': step.notes
                }
                for step in self.steps
            ],
            'total_steps': len(self.steps)
        }

    def clear(self) -> None:
        """Clear all recorded history.

        Example:
            >>> tracker.clear()
        """
        self.steps.clear()
        logger.debug("History tracker cleared")

    def merge(self, other: 'HistoryTracker') -> None:
        """Merge another tracker's history into this one.

        Args:
            other: Another HistoryTracker instance

        Example:
            >>> # Merge processing from different bands
            >>> main_tracker.merge(band_tracker)
        """
        self.steps.extend(other.steps)
        logger.debug(f"Merged {len(other.steps)} steps from another tracker")

    def __len__(self) -> int:
        """Return number of recorded steps."""
        return len(self.steps)

    def __repr__(self) -> str:
        """String representation."""
        return f"HistoryTracker({len(self.steps)} steps)"
