# Mission-Specific FITS Data Acquisition Guide

**Version:** 2025-10-27
**For:** Astro Vision Composer Pipeline
**Audience:** Researchers and amateur astronomers acquiring FITS data for RGB processing

---

## Quick Reference Table

| Mission/Survey | Archive | Best Product | Access Method | Free? |
|----------------|---------|--------------|---------------|-------|
| **JWST** | MAST | `_cal.fits` | Portal/astroquery | YES |
| **HST** | MAST | `_flc.fits` | Portal/astroquery | YES |
| **Chandra** | CDA | `_evt2.fits` (binned) | ChaSeR/download script | YES |
| **Euclid** | ESA/IRSA | Processed stacks | ESASky/IRSA | YES (ERO only) |
| **PanSTARRS** | MAST | FITS cutouts | Cutout service/API | YES |
| **ZTF** | IRSA | Raw/processed FITS | Portal/ztfquery | YES |
| **Catalina** | PDS/AWS | FPACK compressed | AWS/PDS archive | YES |
| **NOIRLab** | Astro Data Lab | FITS cutouts | SIA service/portal | YES |

---

## Space Telescopes

### JWST (James Webb Space Telescope)

**Archive:** MAST (Mikulski Archive for Space Telescopes)
**URL:** https://mast.stsci.edu / https://archive.stsci.edu/jwst

#### Recommended Products for RGB

| Product | Suffix | Description | Use Case |
|---------|--------|-------------|----------|
| **Calibrated** | `_cal.fits` | Fully calibrated, includes **gwcs in ASDF** | **RECOMMENDED** - Best for Phase 3A/3B WCS features |
| **CR-flagged** | `_crf.fits` | Calibrated + cosmic ray flags | When CR rejection critical |
| **Resampled** | `_i2d.fits` | Drizzled to common grid (FITS WCS) | Multi-mission mosaics |

#### Access Methods

**Method 1: MAST Portal (Web Interface)**
```
1. Navigate to: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
2. Advanced Search → Mission: "JWST"
3. Enter target name or coordinates (RA/Dec)
4. Optional filters:
   - Instrument: NIRCam, MIRI, NIRSpec, FGS
   - Filters: F090W, F115W, F150W, F200W, F277W, F356W, F444W (NIRCam)
   - Exposure time: > 100s recommended
5. Select observations → Add to basket
6. Download basket → Choose product types:
   - Check "SCIENCE" for _cal.fits
   - Uncheck "PREVIEW" to save bandwidth
7. Download as .zip or direct links
```

**Method 2: Python/astroquery**
```python
from astroquery.mast import Observations
import astropy.units as u

# Search for JWST observations
results = Observations.query_criteria(
    obs_collection='JWST',
    target_name='SMACS 0723',  # or use coordinates
    instrument_name='NIRCAM/IMAGE',
    filters='F200W',
    calib_level=2  # Level 2 = calibrated (_cal.fits)
)

# Get product list
products = Observations.get_product_list(results)

# Filter for calibrated FITS files
cal_products = products[
    (products['productSubGroupDescription'] == 'CAL') &
    (products['dataproduct_type'] == 'image')
]

# Download
Observations.download_products(cal_products, download_dir='./jwst_data/')
```

**Method 3: Command-line (jwst_mast_query)**
```bash
# Install
pip install jwst-mast-query

# Download specific program/observation
jwst_download.py --propID 2736 --obsNum 1 --output ./data/

# Download by filter
jwst_download.py --target "NGC 3324" --filter F200W --output ./data/
```

#### File Naming Convention
```
jw[PROPID][OBS][VISIT]_[DETECTOR]_[FILTER]_[PRODUCT].fits

Example: jw02739001001_nrcb1_f200w_cal.fits
  - Program: 02739
  - Observation: 001
  - Visit: 001
  - Detector: NRCb1 (NIRCam module B, detector 1)
  - Filter: F200W (2.0 μm)
  - Product: cal (calibrated)
```

#### Recommended Filter Combinations

**NIRCam (Infrared RGB):**
- **Best:** F444W (4.4μm), F200W (2.0μm), F090W (0.9μm) - 4.9× wavelength ratio
- **Good:** F356W (3.6μm), F200W (2.0μm), F115W (1.15μm) - 3.1× ratio
- **Moderate:** F277W (2.8μm), F150W (1.5μm), F090W (0.9μm) - 3.1× ratio

**MIRI (Mid-IR RGB):**
- **Best:** F1800W (18μm), F1000W (10μm), F560W (5.6μm) - 3.2× ratio

#### Critical Notes
- **Requires stdatamodels:** Install with `pip install stdatamodels` to read gwcs from ASDF
- **File sizes:** `_cal.fits` are 50-200 MB each (full resolution)
- **Proprietary periods:** Check if data is public (12-month embargo for most programs)

---

### HST (Hubble Space Telescope)

**Archive:** MAST
**URL:** https://archive.stsci.edu/hst

#### Recommended Products for RGB

| Product | Suffix | Description | Use Case |
|---------|--------|-------------|----------|
| **CR-rejected** | `_flc.fits` | Calibrated + CR rejection (ACS/WFC3) | **RECOMMENDED** - Best quality |
| **Flat-fielded** | `_flt.fits` | Calibrated (older instruments) | When _flc unavailable |
| **Drizzled** | `_drz.fits` / `_drc.fits` | Multi-exposure mosaics | Pre-aligned mosaics |

#### Access Methods

**Method 1: MAST Portal**
```
1. Navigate to: https://mast.stsci.edu
2. Advanced Search → Mission: "HST"
3. Instrument: ACS/WFC, WFC3/UVIS, WFC3/IR, WFPC2
4. Target/Coordinates
5. Filters: F435W, F475W, F606W, F814W (optical common)
6. Add to basket → Download
```

**Method 2: Python/astroquery**
```python
from astroquery.mast import Observations

# Search HST ACS observations
results = Observations.query_criteria(
    obs_collection='HST',
    target_name='M51',
    instrument_name='ACS/WFC',
    filters=['F435W', 'F606W', 'F814W']
)

# Get calibrated products
products = Observations.get_product_list(results)
flc_products = products[products['productSubGroupDescription'] == 'FLC']

Observations.download_products(flc_products)
```

#### File Naming Convention
```
[dataset]_[instr]_[filter]_[product].fits

Example: hst_13003_01_acs_wfc_f606w_flc.fits
  - Program: 13003
  - Visit: 01
  - Instrument: ACS/WFC
  - Filter: F606W (V-band)
  - Product: flc (flat-fielded + CR-rejected)
```

#### Recommended Filter Combinations

**ACS/WFC (Optical RGB):**
- **Classic:** F814W (I-band), F606W (V-band), F435W (B-band) - 1.9× ratio
- **Extended red:** F850LP, F606W, F475W
- **Blue-shifted:** F775W, F555W, F435W

**WFC3/UVIS:**
- **UV-optical:** F438W, F555W, F814W
- **Narrowband:** F656N (Ha), F658N (Ha+[NII]), F502N ([OIII])

**WFC3/IR:**
- **Near-IR:** F160W (1.6μm), F125W (1.25μm), F105W (1.05μm)

#### Critical Notes
- **Distortion correction:** Use drizzlepac.stwcs for full distortion (Phase 3A feature)
- **CR rejection:** Prefer `_flc.fits` over `_flt.fits` when available
- **File sizes:** 20-100 MB per file

---

### Chandra (X-ray Observatory)

**Archive:** Chandra Data Archive (CDA)
**URL:** https://cxc.harvard.edu/cda/

#### Recommended Products for RGB

| Product | Suffix | Description | Use Case |
|---------|--------|-------------|----------|
| **Event list** | `_evt2.fits` | Photon table (TIME, X, Y, ENERGY) | **Raw data** - needs binning |
| **Binned image** | `_img.fits` | Events binned to 2D image | Ready for pipeline |

#### Access Methods

**Method 1: ChaSeR (Web Interface)**
```
1. Navigate to: https://cda.harvard.edu/chaser/
2. Enter target name or coordinates
3. Select observations
4. Check desired data products (evt2, img)
5. Submit → Receive email with download link
6. Download tarball from cdaftp.harvard.edu
```

**Method 2: download_chandra_obsid (Command-line)**
```bash
# Install CIAO (Chandra Interactive Analysis of Observations)
# https://cxc.cfa.harvard.edu/ciao/download/

# Download by ObsID
download_chandra_obsid 1843

# Extract event file
tar -xzf 1843.tar.gz
cd 1843/primary/
# evt2.fits is the calibrated event list
```

**Method 3: Python (via CDA API)**
```python
import urllib.request

obsid = 1843
url = f"https://cda.cfa.harvard.edu/csccli/retrieve?obsid={obsid}"
urllib.request.urlretrieve(url, f"obs_{obsid}.tar.gz")
```

#### Creating RGB from Event Lists

**IMPORTANT:** Our pipeline cannot process event lists directly. Must bin into energy bands first.

```bash
# Use CIAO dmcopy to bin events into energy bands
# Soft (0.5-1.5 keV) → Blue
dmcopy "evt2.fits[energy=500:1500][bin x=::1,y=::1]" soft_img.fits

# Medium (1.5-3.0 keV) → Green
dmcopy "evt2.fits[energy=1500:3000][bin x=::1,y=::1]" medium_img.fits

# Hard (3.0-7.0 keV) → Red
dmcopy "evt2.fits[energy=3000:7000][bin x=::1,y=::1]" hard_img.fits

# Now process with our pipeline
python
from astro_vision_composer import ProcessingPipeline
pipeline = ProcessingPipeline(mode='scientific')
rgb = pipeline.process_to_rgb(['hard_img.fits', 'medium_img.fits', 'soft_img.fits'])
```

#### Critical Notes
- **Event lists are tables, not images!** Must bin first
- **CIAO required** for event binning (https://cxc.cfa.harvard.edu/ciao/)
- **Time-dependent WCS:** Requires aspect solution file (`_asol1.fits`)
- **Phase 5 feature:** Full event list support planned but not yet implemented

---

### Euclid (Space Telescope)

**Archive:** ESA Archive + NASA IRSA
**URL:** https://www.cosmos.esa.int/web/euclid / https://irsa.ipac.caltech.edu/

#### Data Status (2025)

- **Early Release Observations (ERO):** Public since May 2024 (17 fields)
- **Quick Release 1 (Q1):** Public since March 2025 (53 sq. degrees)
- **Main survey data:** Proprietary (consortium access only until ~2027)

#### Access Methods

**Method 1: ESASky (Web Interface)**
```
1. Navigate to: https://sky.esa.int/esasky/
2. Search target or coordinates
3. Select "Euclid" in missions list
4. Click on observation → Download FITS
```

**Method 2: NASA IRSA (Alternative)**
```
1. Navigate to: https://irsa.ipac.caltech.edu/
2. Search for "Euclid"
3. ERO data available as processed image stacks
4. VIS band (optical) and NISP bands (near-IR)
```

#### Products Available

- **VIS band:** ~0.55-0.9 μm (optical)
- **NISP Y-band:** ~0.95-1.19 μm
- **NISP J-band:** ~1.17-1.47 μm
- **NISP H-band:** ~1.40-2.00 μm

#### Critical Notes
- **Limited public data:** Only ERO and Q1 currently available
- **Multi-detector mosaics:** Each field spans multiple CCDs (Phase 3 handles this)
- **High resolution:** 0.1 arcsec/pixel (VIS), 0.3 arcsec/pixel (NISP)

---

## Ground-Based Surveys

### PanSTARRS (Pan-STARRS1)

**Archive:** MAST PS1 Archive
**URL:** https://outerspace.stsci.edu/display/PANSTARRS

#### Access Methods

**Method 1: PS1 Image Cutout Service (Web)**
```
1. Navigate to: https://ps1images.stsci.edu/cgi-bin/ps1cutouts
2. Enter RA/Dec or target name
3. Select filters: g, r, i, z, y
4. Set cutout size (arcseconds or pixels)
5. Choose format: FITS
6. Download individual cutouts
```

**Method 2: Python API**
```python
import requests
from io import BytesIO
from astropy.io import fits

def get_ps1_cutout(ra, dec, size=240, filters="irg", format="fits"):
    """
    Get PS1 FITS cutout.

    Parameters
    ----------
    ra, dec : float
        Coordinates in degrees
    size : int
        Cutout size in pixels
    filters : str
        PS1 filters (g, r, i, z, y)
    """
    url = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
    params = {
        'ra': ra,
        'dec': dec,
        'size': size,
        'format': format,
        'filters': filters
    }

    response = requests.get(url, params=params)
    return fits.open(BytesIO(response.content))

# Example: M51
cutouts = get_ps1_cutout(ra=202.4695, dec=47.1951, filters="irg")
```

#### Recommended Filter Combinations

**Optical RGB:**
- **Best:** i (755nm), r (622nm), g (481nm) - 1.6× ratio
- **Extended red:** z (868nm), i (755nm), r (622nm)
- **Blue-shifted:** r, g, open (450-850nm)

#### Critical Notes
- **All-sky coverage:** Dec > -30° (Northern hemisphere)
- **Cutout limit:** 20 cutouts max via web interface (use API for bulk)
- **Resolution:** 0.25 arcsec/pixel
- **File format:** FITS cutouts are full resolution (no JPEG compression)

---

### ZTF (Zwicky Transient Facility)

**Archive:** IRSA (NASA/IPAC)
**URL:** https://irsa.ipac.caltech.edu/Missions/ztf.html

#### Data Releases

- **Latest:** DR23 (January 15, 2025)
- **Coverage:** Northern sky (Dec > -31°)
- **Cadence:** Entire sky every 2 days

#### Access Methods

**Method 1: IRSA Portal (Web)**
```
1. Navigate to: https://irsa.ipac.caltech.edu/applications/ztf/
2. Search by:
   - Coordinates (RA/Dec)
   - Object name
   - ObsID
3. Select products:
   - sciimg.fits (science image)
   - mskimg.fits (mask image)
   - scipsf.fits (PSF model)
4. Download individual files or batch
```

**Method 2: ztfquery (Python)**
```python
from ztfquery import query, marshal

# Initialize query
zquery = query.ZTFQuery()

# Search by coordinates
zquery.load_metadata(kind='sci', radec=[ra, dec], size=0.01)

# Download science images
zquery.download_data(
    nodl=False,
    show_progress=True,
    nprocess=4  # Parallel downloads
)

# Files saved to default location: $ZTFDATA
```

**Method 3: AWS Open Data**
```python
import boto3
from astropy.io import fits

# Access ZTF data on AWS
s3 = boto3.client('s3', region_name='us-west-2')
bucket = 'ztf-data-releases'

# Download specific image
s3.download_file(
    bucket,
    'dr23/field0123/ccd01/q1/ztf_20230415_123456_g_c01_o_q1_sciimg.fits',
    'local_image.fits'
)
```

#### Available Filters

- **g-band:** 400-552 nm (green)
- **r-band:** 562-695 nm (red)
- **i-band:** 695-820 nm (infrared)

**Recommended Combination:**
- i (757nm), r (641nm), g (464nm) - 1.6× ratio

#### Critical Notes
- **Raw CCD data:** Needs calibration (bias/dark/flat) - Use our Phase 2 CalibrationManager!
- **Large files:** 1-2 GB per field
- **ztfquery requires:** `pip install ztfquery`
- **AWS access:** Free but requires AWS credentials

---

### Catalina Sky Survey (CSS)

**Archive:** PDS Small Bodies Node + AWS Open Data
**URL:** https://sbn.psi.edu/pds/resource/css.html

#### Access Methods

**Method 1: PDS Archive**
```
1. Navigate to: https://sbn.psi.edu/pds/resource/css.html
2. Browse by date or object
3. Download FPACK compressed FITS (.fz extension)
4. Requires CFITSIO library to uncompress
```

**Method 2: AWS Open Data**
```python
import boto3
from astropy.io import fits

s3 = boto3.client('s3')
bucket = 'nasa-pds'

# List available CSS data
response = s3.list_objects_v2(
    Bucket=bucket,
    Prefix='gbo.ast.catalina.survey/'
)

# Download compressed FITS
s3.download_file(
    bucket,
    'gbo.ast.catalina.survey/data/G96_20220120_2B_N16089_01_0001.arch.fz',
    'local_file.fits.fz'
)

# Uncompress and read
# Option 1: Use fpack command-line tool
# Option 2: astropy.io.fits handles .fz automatically
data = fits.open('local_file.fits.fz')
```

**Method 3: Direct HTTP**
```bash
wget https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.catalina.survey/data/[path]
```

#### Data Format

- **File extension:** `.fits.fz` (FPACK tile-compressed)
- **Best files:** `*.arch.fz` (most complete metadata)
- **Uncompression:** Requires CFITSIO or astropy handles automatically

#### Critical Notes
- **3 million+ images** in archive
- **Multiple telescopes:** Different pixel scales (1.4-2.5 arcsec/pixel)
- **Raw data:** Needs calibration (Phase 2 CalibrationManager)
- **Delivered nightly** to PDS since Jan 2022

---

### NOIRLab Astro Data Lab

**Archive:** Astro Data Lab
**URL:** https://datalab.noirlab.edu

#### Surveys Available

- **Legacy Surveys (DECaLS, BASS, MzLS):** Optical imaging
- **unWISE:** Mid-infrared all-sky
- **NSC (NOIRLab Source Catalog):** Multi-epoch photometry
- **DES (Dark Energy Survey):** Deep imaging
- **DECaPS:** Plane of the Milky Way

#### Access Methods

**Method 1: Image Cutout Service (Web)**
```
1. Navigate to: https://datalab.noirlab.edu/sia.php
2. Select survey (e.g., "Legacy Surveys DR10")
3. Enter RA/Dec or target name
4. Set cutout size (arcmin)
5. Select bands: g, r, z (DECam)
6. Download up to 20 cutouts as .zip
```

**Method 2: Simple Image Access (SIA) API**
```python
from pyvo.dal import sia

# Connect to Data Lab SIA service
service = sia.SIAService(
    "https://datalab.noirlab.edu/sia/des_dr2"
)

# Query for images
results = service.search(
    pos=(ra, dec),  # degrees
    size=0.05,      # degrees radius
    format='image/fits'
)

# Download FITS cutouts
for result in results:
    result.cachedataset(filename=f"{result.title}.fits")
```

**Method 3: Bulk Downloads (for >20 images)**
```python
import requests

url = "https://datalab.noirlab.edu/sia/des_dr2"
params = {
    'POS': f'{ra},{dec}',
    'SIZE': 0.05,
    'FORMAT': 'image/fits',
    'BAND': 'g,r,i'
}

response = requests.get(url, params=params)
# Parse VOTable response for download URLs
```

#### Recommended Filter Combinations

**DECam (Legacy Surveys):**
- **Optical:** z (926nm), r (641nm), g (477nm) - 1.9× ratio
- **Extended:** z, r, g

**unWISE (Mid-IR):**
- **W1:** 3.4 μm
- **W2:** 4.6 μm
- (Need visible band from another survey for RGB)

#### Critical Notes
- **Cutout limit:** 20 images max via web portal (use SIA API for bulk)
- **Pre-calibrated:** All Data Lab images are already calibrated
- **High quality:** Deep stacked images with excellent SNR

---

## Practical Workflows by Mission

### JWST: Quick 3-Band Download

```bash
# 1. Search MAST for NIRCam observations
# Target: SMACS 0723 (First Deep Field)

# 2. Download via astroquery
python3 << EOF
from astroquery.mast import Observations

results = Observations.query_criteria(
    obs_collection='JWST',
    target_name='SMACS 0723',
    instrument_name='NIRCAM/IMAGE',
    filters=['F090W', 'F200W', 'F444W'],
    calib_level=2
)

products = Observations.get_product_list(results)
cal = products[products['productSubGroupDescription'] == 'CAL']
Observations.download_products(cal[:3])  # Download 3 filters
EOF

# 3. Process with our pipeline
python3 << EOF
from astro_vision_composer import ProcessingPipeline
pipeline = ProcessingPipeline(mode='scientific')
rgb = pipeline.process_to_rgb([
    'jw02736001001_nrcb1_f444w_cal.fits',
    'jw02736001001_nrcb1_f200w_cal.fits',
    'jw02736001001_nrcb1_f090w_cal.fits'
], output_path='smacs0723_jwst.png')
EOF
```

### HST: Classic Optical RGB

```python
from astroquery.mast import Observations
from astro_vision_composer import ProcessingPipeline

# 1. Download HST ACS data for M51
results = Observations.query_criteria(
    obs_collection='HST',
    target_name='M51',
    instrument_name='ACS/WFC',
    filters=['F435W', 'F606W', 'F814W']
)

products = Observations.get_product_list(results[:1])  # First observation
flc = products[products['productSubGroupDescription'] == 'FLC']
Observations.download_products(flc)

# 2. Process to RGB
pipeline = ProcessingPipeline(mode='sdss')
rgb = pipeline.process_to_rgb([
    'hst_13003_01_acs_wfc_f814w_flc.fits',  # Red
    'hst_13003_01_acs_wfc_f606w_flc.fits',  # Green
    'hst_13003_01_acs_wfc_f435w_flc.fits'   # Blue
])
```

### PanSTARRS: Quick Sky Survey Cutouts

```python
import requests
from astropy.io import fits
from io import BytesIO
from astro_vision_composer import ProcessingPipeline

# 1. Get cutouts for Andromeda (M31)
def get_ps1(ra, dec, filter, size=2400):
    url = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params={'ra':ra, 'dec':dec, 'size':size,
                                   'filters':filter, 'format':'fits'})
    return fits.open(BytesIO(r.content))[0].data

# M31 coordinates
ra, dec = 10.6847, 41.2687

# Download i, r, g bands
i_data = get_ps1(ra, dec, 'i')
r_data = get_ps1(ra, dec, 'r')
g_data = get_ps1(ra, dec, 'g')

# Save temporarily
fits.PrimaryHDU(i_data).writeto('m31_i.fits', overwrite=True)
fits.PrimaryHDU(r_data).writeto('m31_r.fits', overwrite=True)
fits.PrimaryHDU(g_data).writeto('m31_g.fits', overwrite=True)

# 2. Process to RGB
pipeline = ProcessingPipeline(mode='scientific')
rgb = pipeline.process_to_rgb(['m31_i.fits', 'm31_r.fits', 'm31_g.fits'])
```

### ZTF: Recent Transient Field

```python
from ztfquery import query
from astro_vision_composer import ProcessingPipeline

# 1. Download ZTF data
zquery = query.ZTFQuery()

# Search for recent observation of a field
zquery.load_metadata(
    kind='sci',
    field=579,  # Field ID
    ccdid=1,
    qid=1,
    filtercode='g'  # Then repeat for 'r', 'i'
)

# Download
zquery.download_data(show_progress=True)

# 2. Get file paths
files = zquery.get_local_data()

# 3. Calibrate with Phase 2 CalibrationManager
from astro_vision_composer.preprocessing import CalibrationManager

# Assuming you have bias/dark/flat in 'calibration/' directory
calib = CalibrationManager('calibration/')
calib.create_master_calibrations()

# Calibrate each band
from ccdproc import CCDData
for band in ['i', 'r', 'g']:
    raw = CCDData.read(f'ztf_{band}.fits', unit='adu')
    calibrated = calib.calibrate(raw, filter_name=band)
    calibrated.write(f'ztf_{band}_cal.fits')

# 4. Process to RGB
pipeline = ProcessingPipeline(mode='scientific')
rgb = pipeline.process_to_rgb([
    'ztf_i_cal.fits',
    'ztf_r_cal.fits',
    'ztf_g_cal.fits'
])
```

---

## Troubleshooting Common Issues

### Issue: "No data found for target"

**Cause:** Target name not recognized or no observations exist

**Solutions:**
1. Try coordinates instead of name (use SIMBAD to get RA/Dec)
2. Check mission coverage (JWST limited, HST extensive, surveys all-sky)
3. Verify target is in survey footprint (ZTF: Dec > -31°, PanSTARRS: Dec > -30°)

### Issue: "Downloaded file is not a FITS image"

**Cause:** Downloaded wrong product type (catalog, spectrum, event list)

**Solutions:**
1. Check file suffix matches expected product
2. For Chandra: Bin event lists to images first (see Chandra section)
3. Verify download completed (check file size > 0)

### Issue: "FITS file has no WCS"

**Cause:** Raw/uncalibrated data or corrupted download

**Solutions:**
1. Download calibrated products (_cal.fits, _flc.fits, not _raw.fits)
2. For ground-based raw data: Use Phase 2 CalibrationManager first
3. Check FITS header for WCS keywords (CTYPE1, CTYPE2, CRVAL1, CRVAL2)

### Issue: "Cutout service returns error 500"

**Cause:** Server overload or invalid parameters

**Solutions:**
1. Reduce cutout size (try < 2048 pixels)
2. Retry during off-peak hours
3. Use API instead of web interface for automation
4. Check coordinate format (decimal degrees, not HMS/DMS)

### Issue: "Cannot read JWST gwcs - no ASDF extension"

**Cause:** Missing stdatamodels package

**Solution:**
```bash
pip install stdatamodels
```

### Issue: "Compressed FITS (.fz) cannot be read"

**Cause:** Missing CFITSIO support

**Solutions:**
1. astropy.io.fits handles .fz automatically (update astropy)
2. Manually uncompress with fpack:
```bash
# Install fpack
# Ubuntu/Debian: apt-get install libcfitsio-bin
# macOS: brew install cfitsio

# Uncompress
funpack file.fits.fz
# Creates: file.fits
```

---

## Best Practices for Data Acquisition

### 1. **Always Download Calibrated Products**

| Mission | Calibrated Suffix | Uncalibrated (Avoid) |
|---------|-------------------|----------------------|
| JWST | `_cal.fits` | `_uncal.fits`, `_rate.fits` |
| HST | `_flc.fits`, `_flt.fits` | `_raw.fits` |
| Chandra | `_evt2.fits` (then bin) | N/A |
| Ground surveys | Pre-calibrated | N/A |

### 2. **Check File Completeness Before Processing**

```python
from astropy.io import fits

def validate_fits(filename):
    """Quick FITS validation."""
    with fits.open(filename) as hdul:
        # Check has data
        assert len(hdul) > 0, "No HDUs in FITS"

        # Check data not empty
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        assert data is not None, "No data array"
        assert data.size > 0, "Empty data array"

        # Check has WCS (for space telescopes)
        hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
        has_wcs = 'CTYPE1' in hdr or 'WCSAXES' in hdr

        print(f"[OK] {filename}")
        print(f"  Shape: {data.shape}")
        print(f"  WCS: {'Yes' if has_wcs else 'No'}")

        return True

# Validate before processing
validate_fits('jw02736_f200w_cal.fits')
```

### 3. **Organize Downloaded Data by Mission/Target**

```
data/
├── jwst/
│   ├── smacs0723/
│   │   ├── jw02736_f090w_cal.fits
│   │   ├── jw02736_f200w_cal.fits
│   │   └── jw02736_f444w_cal.fits
│   └── carina_nebula/
├── hst/
│   └── m51/
│       ├── f435w_flc.fits
│       ├── f606w_flc.fits
│       └── f814w_flc.fits
└── panstarrs/
    └── m31/
        ├── m31_g.fits
        ├── m31_r.fits
        └── m31_i.fits
```

### 4. **Document Data Provenance**

Create a `README.txt` or `metadata.json` for each target:

```json
{
  "target": "SMACS 0723",
  "mission": "JWST",
  "instrument": "NIRCam",
  "filters": ["F090W", "F200W", "F444W"],
  "observation_date": "2022-06-07",
  "program_id": "2736",
  "download_date": "2025-10-27",
  "archive": "MAST",
  "notes": "First Deep Field ERO"
}
```

### 5. **Use Caching to Avoid Re-downloads**

```python
from pathlib import Path
from astroquery.mast import Observations

def download_if_missing(target, filters, output_dir='./data/jwst/'):
    """Download JWST data only if not already cached."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    expected_files = [output_dir / f"{target}_{f}_cal.fits" for f in filters]
    if all(f.exists() for f in expected_files):
        print(f"[CACHE] Using cached files for {target}")
        return expected_files

    # Download
    print(f"[DOWNLOAD] Fetching {target} from MAST...")
    results = Observations.query_criteria(
        obs_collection='JWST',
        target_name=target,
        filters=filters,
        calib_level=2
    )

    products = Observations.get_product_list(results)
    cal = products[products['productSubGroupDescription'] == 'CAL']
    Observations.download_products(cal, download_dir=str(output_dir))

    return expected_files
```

---

## Additional Resources

### Official Documentation

- **MAST:** https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
- **JWST Archive:** https://archive.stsci.edu/jwst/
- **HST Archive:** https://archive.stsci.edu/hst/
- **Chandra Archive:** https://cxc.harvard.edu/cda/
- **IRSA (ZTF/Euclid):** https://irsa.ipac.caltech.edu/
- **PanSTARRS:** https://outerspace.stsci.edu/display/PANSTARRS/
- **NOIRLab Data Lab:** https://datalab.noirlab.edu/

### Python Packages

```bash
# Core tools
pip install astropy astroquery

# Mission-specific
pip install stdatamodels     # JWST gwcs
pip install drizzlepac       # HST distortion
pip install ztfquery          # ZTF data access

# Our pipeline
pip install astro-vision-composer
```

### Community Support

- **MAST Help Desk:** https://stsci.service-now.com/mast
- **Chandra Help:** https://cxc.harvard.edu/help/
- **IRSA Help:** https://irsa.ipac.caltech.edu/docs/help_desk.html
- **Our Pipeline Issues:** https://github.com/[your-repo]/issues

---

## Quick Start Checklist

Before starting your RGB project:

- [ ] Identify target coordinates (RA/Dec in decimal degrees)
- [ ] Choose appropriate mission/survey (based on wavelength, resolution, availability)
- [ ] Register for archive accounts if needed (MAST, IRSA - all free!)
- [ ] Download calibrated products (not raw/uncalibrated)
- [ ] Select 3 filters with good wavelength separation (2-5× ratio)
- [ ] Verify FITS files are complete and have WCS
- [ ] Check exposure times are adequate (>100s for space, >300s for ground)
- [ ] Organize data in clear directory structure
- [ ] Install required Python packages (stdatamodels, drizzlepac, etc.)
- [ ] Review our FITS selection guide in `examples/README.md`

**Ready to process?** See `examples/README.md` for pipeline workflows!

---

**Last Updated:** 2025-10-27
**Version:** 1.0
**Feedback:** Submit issues to project repository
