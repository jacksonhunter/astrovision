# ValidationReport Design Document

## Purpose
Aggregate quality assessments, warnings, and validation checks across the entire processing pipeline to provide comprehensive quality assurance.

---

## Design Philosophy

**Goals:**
1. **Comprehensive:** Check all aspects of data quality
2. **Actionable:** Provide specific recommendations
3. **Exportable:** Generate human and machine-readable reports
4. **Non-Blocking:** Warn but don't prevent processing
5. **Traceable:** Link issues to specific processing steps

**Use Cases:**
1. Pre-flight checks before processing
2. Post-processing quality verification
3. Batch processing quality reports
4. Publication documentation
5. Debugging processing issues

---

## API Design

### Basic Usage

```python
from astro_vision_composer.utilities import ValidationReport

# Create report
report = ValidationReport()

# Add quality assessments
report.add_quality_assessment('g_band', quality_report_g)
report.add_quality_assessment('r_band', quality_report_r)

# Add WCS checks
report.add_wcs_validation('g_band', wcs_info_g)
report.check_wcs_alignment(['g_band', 'r_band'])

# Add custom checks
report.add_check('saturation', 'High saturation in r-band', level='warning')

# Get summary
print(report.summary())

# Export
report.export_json('quality_report.json')
report.export_html('quality_report.html')
```

---

## Class Structure

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from datetime import datetime

CheckLevel = Literal['info', 'warning', 'error', 'critical']

@dataclass
class ValidationCheck:
    """Individual validation check result."""
    category: str  # 'quality', 'wcs', 'calibration', 'processing'
    name: str
    status: Literal['pass', 'fail', 'warning']
    level: CheckLevel
    message: str
    details: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: Optional[str] = None  # Which component raised this


class ValidationReport:
    """Aggregate validation and quality reports."""

    def __init__(self):
        self.checks: List[ValidationCheck] = []
        self.quality_assessments: Dict[str, 'QualityReport'] = {}
        self.wcs_validations: Dict[str, 'WCSInfo'] = {}
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()

    # === Adding Data ===

    def add_quality_assessment(self, band_name: str, quality_report):
        """Add QualityAssessor result for a band."""
        self.quality_assessments[band_name] = quality_report
        self._check_quality_thresholds(band_name, quality_report)

    def add_wcs_validation(self, band_name: str, wcs_info):
        """Add WCSHandler validation result."""
        self.wcs_validations[band_name] = wcs_info
        self._check_wcs_quality(band_name, wcs_info)

    def add_check(self, category: str, message: str,
                  level: CheckLevel = 'info',
                  status: str = 'pass',
                  **details):
        """Add custom validation check."""
        check = ValidationCheck(
            category=category,
            name=category,
            status=status,
            level=level,
            message=message,
            details=details
        )
        self.checks.append(check)

    def add_metadata(self, key: str, value: Any):
        """Add metadata about processing."""
        self.metadata[key] = value

    # === Automated Checks ===

    def check_wcs_alignment(self, band_names: List[str]):
        """Check if WCS are compatible across bands."""
        pass

    def check_saturation_levels(self, threshold: float = 0.05):
        """Check if any bands exceed saturation threshold."""
        pass

    def check_snr_levels(self, minimum_snr: float = 3.0):
        """Check if SNR is adequate."""
        pass

    def check_data_integrity(self):
        """Check for NaN, Inf, negative values."""
        pass

    # === Reporting ===

    def summary(self, verbose: bool = False) -> str:
        """Generate text summary."""
        pass

    def get_warnings(self) -> List[ValidationCheck]:
        """Get all warning-level checks."""
        return [c for c in self.checks if c.level == 'warning']

    def get_errors(self) -> List[ValidationCheck]:
        """Get all error-level checks."""
        return [c for c in self.checks if c.level in ['error', 'critical']]

    def has_errors(self) -> bool:
        """Check if any errors exist."""
        return len(self.get_errors()) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings exist."""
        return len(self.get_warnings()) > 0

    def is_valid(self) -> bool:
        """Check if data passes all critical checks."""
        return not any(c.level == 'critical' and c.status == 'fail'
                       for c in self.checks)

    # === Export ===

    def export_json(self, filepath: str):
        """Export as JSON."""
        pass

    def export_html(self, filepath: str):
        """Export as HTML report."""
        pass

    def export_text(self, filepath: str):
        """Export as plain text."""
        pass

    def to_dict(self) -> Dict:
        """Convert to dictionary (JSON-serializable)."""
        pass

    # === Internal Validation Logic ===

    def _check_quality_thresholds(self, band_name: str, quality_report):
        """Apply quality thresholds and add checks."""
        # Check saturation
        if quality_report.saturation_fraction > 0.05:
            self.add_check(
                'quality',
                f"{band_name}: High saturation ({quality_report.saturation_fraction*100:.1f}%)",
                level='warning',
                status='warning',
                band=band_name,
                saturation=quality_report.saturation_fraction
            )

        # Check SNR
        if quality_report.snr < 3.0:
            self.add_check(
                'quality',
                f"{band_name}: Low SNR ({quality_report.snr:.1f})",
                level='warning',
                status='warning',
                band=band_name,
                snr=quality_report.snr
            )

    def _check_wcs_quality(self, band_name: str, wcs_info):
        """Check WCS validation results."""
        if not wcs_info.is_valid:
            self.add_check(
                'wcs',
                f"{band_name}: Invalid WCS",
                level='error',
                status='fail',
                band=band_name,
                warnings=wcs_info.warnings
            )
```

---

## Validation Categories

### 1. Data Quality
**Checks:**
- SNR levels (minimum threshold)
- Saturation fraction (maximum threshold)
- Dynamic range
- Noise characteristics
- NaN/Inf presence
- Negative values (unexpected for counts)

**Thresholds:**
```python
QUALITY_THRESHOLDS = {
    'min_snr': 3.0,           # Minimum acceptable SNR
    'max_saturation': 0.05,   # 5% max saturated pixels
    'min_dynamic_range': 10,  # bits
    'max_nan_fraction': 0.01, # 1% max NaN pixels
}
```

### 2. WCS Quality
**Checks:**
- WCS validity (has celestial coordinates)
- Pixel scale reasonableness
- Projection type consistency across bands
- WCS alignment (rotation differences)
- Reference coordinate sanity

**Thresholds:**
```python
WCS_THRESHOLDS = {
    'max_scale_diff': 0.05,     # 5% max pixel scale difference
    'max_rotation_diff': 1.0,   # 1 degree max rotation difference
    'min_pixel_scale': 0.001,   # arcsec/pixel (very fine)
    'max_pixel_scale': 10.0,    # arcsec/pixel (very coarse)
}
```

### 3. Calibration Quality
**Checks:**
- Bias level after correction
- Dark current residuals
- Flat-field uniformity
- Background level consistency

### 4. Processing Quality
**Checks:**
- Normalization range
- Stretch parameter validity
- Color balance reasonableness
- Composite quality (color clipping, artifacts)

---

## Report Formats

### Text Summary

```
Validation Report
Generated: 2025-10-25 14:30:00
================================================================================

OVERALL STATUS: ✓ PASS (2 warnings)

Data Quality:
  ✓ g-band: SNR=15.3, Saturation=1.2%, Dynamic Range=14.2 bits
  ⚠ r-band: SNR=12.1, Saturation=6.3%, Dynamic Range=13.8 bits
    └─ WARNING: High saturation (6.3% > 5.0% threshold)
  ✓ i-band: SNR=18.7, Saturation=0.8%, Dynamic Range=14.5 bits

WCS Quality:
  ✓ All bands have valid WCS
  ✓ Pixel scale: 0.25 arcsec/pixel (consistent)
  ⚠ Rotation difference: 2.1° between g and r bands
    └─ WARNING: Exceeds 1.0° threshold
  ✓ Projection: TAN (consistent)

Processing:
  ✓ Normalization: zscale interval [125.3, 4821.7]
  ✓ Stretch: asinh (a=0.1)
  ✓ Composite: Lupton RGB (stretch=0.5, Q=8)

Recommendations:
  • Consider reprocessing r-band with different exposure
  • Check alignment - 2.1° rotation may affect registration
  • Overall quality: GOOD - suitable for publication
```

### JSON Export

```json
{
  "created_at": "2025-10-25T14:30:00",
  "overall_status": "pass_with_warnings",
  "quality_assessments": {
    "g_band": {
      "snr": 15.3,
      "saturation_fraction": 0.012,
      "dynamic_range": 14.2
    },
    "r_band": {...},
    "i_band": {...}
  },
  "checks": [
    {
      "category": "quality",
      "name": "saturation_check",
      "status": "warning",
      "level": "warning",
      "message": "r-band: High saturation (6.3%)",
      "details": {
        "band": "r_band",
        "saturation": 0.063,
        "threshold": 0.05
      },
      "timestamp": "2025-10-25T14:30:01"
    }
  ],
  "warnings_count": 2,
  "errors_count": 0,
  "recommendations": [...]
}
```

### HTML Export

Interactive HTML report with:
- Color-coded status indicators
- Expandable sections
- Embedded plots (if matplotlib available)
- Download as PDF option
- Filterable table of checks

---

## Integration with ProcessingPipeline

```python
class ProcessingPipeline:
    def __init__(self, track_validation=True):
        self._validation = ValidationReport() if track_validation else None

    def load_files(self, files):
        # After loading
        if self._validation:
            self._validation.add_metadata('files_loaded', len(files))
            # Check file integrity
            self._validation.check_data_integrity()
        return self

    def assess_quality(self):
        # After assessment
        for band, report in self._quality_reports.items():
            self._validation.add_quality_assessment(band, report)
        return self

    def align(self, ...):
        # After alignment
        self._validation.check_wcs_alignment(list(self._loaded_data.keys()))
        return self

    def get_result(self):
        result = {...}
        if self._validation:
            result['validation'] = self._validation
            if self._validation.has_errors():
                print("⚠ WARNING: Validation errors detected!")
                print(self._validation.summary())
        return result
```

---

## Usage Examples

### Example 1: Manual Validation

```python
from astro_vision_composer.preprocessing import FITSLoader, QualityAssessor
from astro_vision_composer.utilities import ValidationReport

# Load data
loader = FITSLoader()
g_data = loader.load('g.fits')
r_data = loader.load('r.fits')

# Assess quality
qa = QualityAssessor()
g_quality = qa.assess_quality(g_data.science)
r_quality = qa.assess_quality(r_data.science)

# Create validation report
report = ValidationReport()
report.add_quality_assessment('g', g_quality)
report.add_quality_assessment('r', r_quality)

# Check results
if report.has_warnings():
    print(report.summary())

# Export for records
report.export_json('quality_report.json')
report.export_html('quality_report.html')
```

### Example 2: With Pipeline

```python
result = ProcessingPipeline().create_rgb_composite(
    files=['g.fits', 'r.fits', 'i.fits'],
    wavelengths={'g': 481, 'r': 617, 'i': 752},
    output='composite.png'
)

# Automatic validation
validation = result['validation']
print(validation.summary())

if not validation.is_valid():
    print("ERROR: Critical validation failures!")
    for error in validation.get_errors():
        print(f"  - {error.message}")
```

### Example 3: Custom Thresholds

```python
report = ValidationReport()

# Add quality data
report.add_quality_assessment('hst_acs', quality_report)

# Custom checks
report.check_snr_levels(minimum_snr=5.0)  # Higher threshold for HST
report.check_saturation_levels(threshold=0.02)  # Stricter for space data

# Science-specific check
if some_condition:
    report.add_check(
        category='science',
        message='Target not detected in i-band',
        level='critical',
        status='fail'
    )
```

---

## Extensibility

### Custom Validators

```python
class CustomValidator:
    def validate(self, data, report: ValidationReport):
        # Custom logic
        if some_check_fails:
            report.add_check(
                'custom',
                'Custom check failed',
                level='warning'
            )

# Use in pipeline
pipeline.add_validator(CustomValidator())
```

### Validator Plugins

```python
# Register custom validators
ValidationReport.register_validator('my_check', my_validator_function)

# Use in reports
report.run_validator('my_check', data)
```

---

## Testing Strategy

### Unit Tests:
- Test each validation check independently
- Test threshold logic
- Test report generation
- Test export formats

### Integration Tests:
- Test with known good/bad data
- Test with edge cases (all NaN, all saturated, etc.)
- Test with real FITS files

### User Acceptance:
- Generate reports for sample datasets
- Verify readability and actionability
- Validate threshold recommendations

---

## Implementation Priority

**Must Have (MVP):**
- Quality assessment aggregation
- WCS validation
- Text summary output
- JSON export
- Basic threshold checks

**Should Have:**
- HTML export
- Custom checks API
- Integration with Pipeline
- Recommendations system

**Nice to Have:**
- PDF export
- Interactive visualization
- Custom validator plugins
- Email notifications

---

## Estimated Implementation

**Lines of Code:** ~400-500
**Time:** 2-3 days
**Dependencies:** None (HTML export may use jinja2)
**Testing:** 2 days additional
