# ProcessingPipeline Design Document

## Purpose
Provide a high-level, user-friendly API for common astronomical image processing workflows, abstracting away the complexity of manually chaining multiple components.

---

## Design Philosophy

**Goals:**
1. **Ease of Use:** Single function call for common workflows
2. **Sensible Defaults:** Good parameters out-of-the-box
3. **Transparency:** Show what's happening under the hood
4. **Flexibility:** Allow customization when needed
5. **Reproducibility:** Auto-track processing history

**Non-Goals:**
- Replace low-level components (users can still use them directly)
- Support every possible workflow (focus on 80% use case)
- Be "magical" (prefer explicit over implicit)

---

## API Design

### High-Level Interface

```python
from astro_vision_composer import ProcessingPipeline

pipeline = ProcessingPipeline()

# Simple workflow: FITS files → RGB composite
rgb = pipeline.create_rgb_composite(
    files=['g_band.fits', 'r_band.fits', 'i_band.fits'],
    wavelengths={'g': 481, 'r': 617, 'i': 752},
    output='composite.png'
)

# Returns: RGB array + processing history + quality report
```

###  Workflow Presets

```python
# Preset 1: Quick Preview
pipeline.quick_preview(
    files=['g.fits', 'r.fits', 'i.fits'],
    output='preview.jpg',
    max_size=1024
)

# Preset 2: Publication Quality
pipeline.publication_quality(
    files=['g.fits', 'r.fits', 'i.fits'],
    output='publication.tiff',
    enhance=True,
    color_balance=True,
    bit_depth=16
)

# Preset 3: Single Band Processing
pipeline.process_single_band(
    file='raw.fits',
    calibration_frames={'bias': 'bias.fits', 'dark': 'dark.fits', 'flat': 'flat.fits'},
    output='processed.fits'
)
```

### Builder Pattern (Advanced Users)

```python
result = (ProcessingPipeline()
    .load_files(['g.fits', 'r.fits', 'i.fits'])
    .assess_quality()
    .normalize(method='zscale')
    .stretch(method='asinh', a=0.1)
    .enhance(unsharp_mask=True, sigma=2.0)
    .map_channels_by_wavelength({'g': 481, 'r': 617, 'i': 752})
    .create_composite(method='lupton', stretch=0.5, Q=8)
    .balance_colors(method='gray_world')
    .export('final.png', bit_depth=16)
    .get_result()
)
```

---

## Class Structure

```python
class ProcessingPipeline:
    """High-level pipeline for astronomical image processing."""

    def __init__(self,
                 mission: Optional[str] = None,
                 verbose: bool = True,
                 track_history: bool = True):
        """
        Args:
            mission: Mission hint for auto-detection
            verbose: Print progress messages
            track_history: Record all processing steps
        """
        self.mission = mission
        self.verbose = verbose
        self.track_history = track_history

        # Internal state
        self._loaded_data = {}
        self._processed_data = {}
        self._rgb = None
        self._history = HistoryTracker() if track_history else None
        self._validation = ValidationReport() if track_history else None

    # === Workflow Presets ===

    def create_rgb_composite(self, files, wavelengths, output, **kwargs):
        """One-call RGB composite generation."""
        pass

    def quick_preview(self, files, output, **kwargs):
        """Fast preview for inspection."""
        pass

    def publication_quality(self, files, output, **kwargs):
        """High-quality composite with all enhancements."""
        pass

    def process_single_band(self, file, calibration_frames, output, **kwargs):
        """Single-band reduction and enhancement."""
        pass

    # === Builder Pattern Methods ===

    def load_files(self, files):
        """Load FITS files."""
        # Uses FITSLoader
        return self  # For chaining

    def assess_quality(self):
        """Run quality assessment on all loaded data."""
        # Uses QualityAssessor
        return self

    def calibrate(self, bias=None, dark=None, flat=None):
        """Apply calibration corrections."""
        # Uses Calibrator
        return self

    def align(self, reference=None, method='interp'):
        """Align images to common WCS."""
        # Uses Reprojector
        return self

    def normalize(self, method='zscale', **kwargs):
        """Normalize data."""
        # Uses Normalizer
        return self

    def stretch(self, method='asinh', **kwargs):
        """Apply non-linear stretch."""
        # Uses Stretcher
        return self

    def enhance(self, **kwargs):
        """Apply enhancement techniques."""
        # Uses Enhancer
        return self

    def map_channels_by_wavelength(self, wavelengths):
        """Map bands to RGB channels."""
        # Uses ChannelMapper
        return self

    def create_composite(self, method='lupton', **kwargs):
        """Create RGB composite."""
        # Uses Compositor
        return self

    def balance_colors(self, method='gray_world', **kwargs):
        """Balance RGB colors."""
        # Uses ColorBalancer
        return self

    def export(self, filepath, **kwargs):
        """Export to file."""
        # Uses ImageExporter
        return self

    def get_result(self):
        """Get final result with metadata."""
        return {
            'data': self._rgb or self._processed_data,
            'history': self._history,
            'validation': self._validation,
            'metadata': self._collect_metadata()
        }

    # === Utility Methods ===

    def _collect_metadata(self):
        """Aggregate metadata from all loaded files."""
        pass

    def _log(self, message):
        """Log message if verbose."""
        if self.verbose:
            print(f"[Pipeline] {message}")
```

---

## Default Parameters

### Normalization Defaults:
- Method: `zscale`
- Why: Best general-purpose interval for astronomical data

### Stretching Defaults:
- Method: `asinh`
- Parameter `a`: `0.1`
- Why: Excellent for wide dynamic range

### Composite Defaults:
- Method: `lupton`
- Stretch: `0.5`
- Q: `8.0`
- Why: SDSS standard, works well for most data

### Enhancement Defaults:
- Unsharp mask sigma: `2.0`
- Unsharp mask strength: `1.5`
- Why: Moderate sharpening without artifacts

### Color Balance Defaults:
- Method: `gray_world`
- Why: Assumes average scene is neutral gray

---

## Error Handling

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class LoadError(PipelineError):
    """Failed to load FITS files."""
    pass

class ProcessingError(PipelineError):
    """Error during processing step."""
    pass

class ValidationError(PipelineError):
    """Data failed validation checks."""
    pass
```

**Strategy:**
- Validate inputs early
- Provide helpful error messages with suggestions
- Offer recovery options when possible
- Never silently fail

---

## Progress Reporting

```python
# Simple progress messages
[Pipeline] Loading 3 FITS files...
[Pipeline] Assessing quality...
[Pipeline] Normalizing with zscale...
[Pipeline] Stretching with asinh (a=0.1)...
[Pipeline] Creating Lupton RGB composite...
[Pipeline] Exporting to final.png (16-bit)...
[Pipeline] ✓ Complete! (12.3 seconds)

# With validation warnings
[Pipeline] ⚠ Warning: g-band has 2.3% saturated pixels
[Pipeline] ⚠ Warning: WCS rotation differs between bands (±3.2°)
```

---

## Usage Examples

### Example 1: Simplest Possible

```python
from astro_vision_composer import ProcessingPipeline

ProcessingPipeline().create_rgb_composite(
    files=['g.fits', 'r.fits', 'i.fits'],
    wavelengths={'g': 481, 'r': 617, 'i': 752},
    output='composite.png'
)
```

### Example 2: With Customization

```python
pipeline = ProcessingPipeline(verbose=True)

result = pipeline.create_rgb_composite(
    files=['g.fits', 'r.fits', 'i.fits'],
    wavelengths={'g': 481, 'r': 617, 'i': 752},
    normalize_method='percentile',
    normalize_params={'lower': 2, 'upper': 98},
    stretch_method='asinh',
    stretch_params={'a': 0.15},
    enhance=True,
    enhance_params={'unsharp_sigma': 2.0, 'unsharp_strength': 2.0},
    color_balance=True,
    output='enhanced_composite.png',
    bit_depth=16
)

# Access processing history
print(result['history'].to_text())

# Access validation report
print(result['validation'].summary())
```

### Example 3: Builder Pattern

```python
result = (ProcessingPipeline()
    .load_files(['F444W.fits', 'F356W.fits', 'F200W.fits'])
    .assess_quality()
    .normalize(method='zscale')
    .stretch(method='asinh', a=0.1)
    .enhance(unsharp_mask=True, sigma=2.0, strength=1.5)
    .map_channels_by_wavelength({'F200W': 1989, 'F356W': 3568, 'F444W': 4421})
    .create_composite(method='lupton', stretch=0.5, Q=8)
    .balance_colors(method='gray_world')
    .export('jwst_composite.png', bit_depth=16)
    .get_result()
)
```

### Example 4: Calibration Workflow

```python
pipeline = ProcessingPipeline()

processed = pipeline.process_single_band(
    file='raw_science.fits',
    calibration_frames={
        'bias': 'master_bias.fits',
        'dark': 'master_dark.fits',
        'flat': 'master_flat.fits'
    },
    normalize=True,
    stretch=True,
    output='calibrated.fits'
)
```

---

## Integration Points

### With Existing Components:

```python
# Pipeline internally uses:
from ..preprocessing import FITSLoader, Calibrator, QualityAssessor
from ..processing import Normalizer, Stretcher, Reprojector, Enhancer
from ..postprocessing import ChannelMapper, Compositor, ColorBalancer, ImageExporter, HistoryTracker
from .validation import ValidationReport
```

### With HistoryTracker:

```python
# Pipeline automatically records:
self._history.record('load_files', {'files': files}, 'ProcessingPipeline')
self._history.record('normalize', {'method': 'zscale'}, 'Normalizer')
# etc.
```

### With ValidationReport:

```python
# Pipeline automatically validates:
self._validation.add_quality_report('g_band', quality_report)
self._validation.check_wcs_alignment(wcs_comparison)
self._validation.check_saturation_levels(saturation_map)
```

---

## Testing Strategy

### Unit Tests:
- Test each builder method independently
- Test parameter validation
- Test error handling

### Integration Tests:
- Test complete workflows with synthetic data
- Test with real FITS files
- Test all presets

### Performance Tests:
- Benchmark on various image sizes
- Memory usage profiling
- Identify bottlenecks

---

## Future Enhancements

### Phase 5+:
1. **Batch Processing:** Process multiple targets in parallel
2. **Auto-Parameter Selection:** ML-based parameter optimization
3. **Resume Capability:** Save/load pipeline state
4. **GUI Integration:** Visual pipeline builder
5. **Web API:** REST API for pipeline execution
6. **Streaming:** Process very large images in chunks

---

## Implementation Priority

**Must Have (MVP):**
- `create_rgb_composite()` - Core use case
- Builder pattern methods - Flexibility
- Error handling - Robustness
- Progress reporting - User feedback

**Should Have:**
- `quick_preview()` - Common use case
- `publication_quality()` - Common use case
- Validation integration - Quality assurance

**Nice to Have:**
- `process_single_band()` - Specialized use case
- Parameter auto-tuning - Advanced feature
- Resume capability - Advanced feature

---

## Estimated Implementation

**Lines of Code:** ~500-600
**Time:** 1-2 days
**Dependencies:** None (uses existing components)
**Testing:** 2-3 days additional
