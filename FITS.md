From FITS to Finish: A Python-Centric Guide to Professional Astronomical Image Processing

Part I: Foundations of Astronomical Data Handling

The journey from raw observational data to a stunning astronomical image is a meticulous process of scientific interpretation and technical execution. At its heart lies the Flexible Image Transport System (FITS) file, the universal standard for astronomical data. This section establishes the foundational knowledge required to handle these files programmatically, introducing the structure of FITS itself and the powerful Python ecosystem designed to manipulate it. A firm grasp of these core concepts is essential before proceeding to the mission-specific workflows and advanced imaging techniques that follow.

1.  Anatomy of a FITS File: The Universal Astronomical Data Container

The Flexible Image Transport System (FITS) is far more than a simple image format like JPEG or PNG. It is a comprehensive data container designed specifically for the storage, transmission, and manipulation of scientific datasets.2 Originally developed in the late 1970s, its primary purpose is to store multidimensional data arrays (such as images, spectra, or data cubes) and two-dimensional tables in a self-describing, hardware-independent manner.4 This robust and extensible design has made it the default standard for data exchange and archiving throughout the astronomical community.1

Core Concepts

The fundamental building block of a FITS file is the Header Data Unit (HDU). A FITS file consists of one or more HDUs arranged sequentially. The first HDU is always the Primary HDU. Any subsequent HDUs are known as extensions.4 This multi-extension FITS (MEF) structure is particularly important for modern observatories, which often package related data—such as a science image, its corresponding error map, and a data quality mask—into a single, organized file.6 Each HDU is composed of two main parts 8: 1. ASCII Header: A human-readable text block that describes the data that follows. It contains metadata about the observation, the instrument, the data format, and the coordinate system. 2. Binary Data: The actual scientific data, stored in a binary format as an array or a table. This data block may be empty, particularly in the case of the Primary HDU in many modern data products, which often serves only as a container for global metadata.6 For historical reasons related to data storage on magnetic tapes, each HDU (both header and data sections) must be an exact multiple of 2880 bytes in length. Any unused space is padded with fill characters.4

The Header

The FITS header is a critical component, providing the context necessary to interpret the binary data. It consists of a series of 80-character ASCII records, often referred to as "cards".4 Each card typically follows the format KEYWORD = value / comment. ● Keywords: These are 8-character (or shorter) ASCII strings that identify the metadata field (e.g., NAXIS for the number of axes in the data array). ● Value: The value associated with the keyword. This can be a logical (T or F), an integer, a floating-point number, or a character string (enclosed in single quotes). ● Comment: An optional comment string that provides a human-readable description of the keyword. Several keywords are essential for defining the structure of the data within an HDU 4: ● SIMPLE: A logical value indicating whether the file conforms to the basic FITS standard. ● BITPIX: An integer specifying the number of bits per data pixel (e.g., 16 for 16-bit integers, -32 for 32-bit floating-point numbers). ● NAXIS: An integer defining the number of axes in the data array (e.g., 2 for a 2D image). ● NAXISn: A series of integers (where n is 1, 2, 3,...) specifying the length of each axis. ● EXTEND: A logical value in the Primary HDU indicating whether FITS extensions may be present. For HDUs that are extensions, the XTENSION keyword is required. It specifies the type of data contained within the extension, with common values being IMAGE for an image array, BINTABLE for a binary table, and TABLE for an ASCII table.1 The FITS standard is not static; it evolves to meet the needs of modern astronomy. For instance, recent versions have added support for 64-bit integers and variable-length arrays.2 Furthermore, missions like the James Webb Space Telescope (JWST) have adopted modern solutions for managing their extensive and complex metadata. JWST utilizes JavaScript Object Notation (JSON) files to define the collection of keywords and their properties, which are then used to populate the FITS headers during data processing.11 This hierarchical and human-readable format provides a flexible interface across the entire data reduction, calibration, and archival system. This adaptability demonstrates that while the core FITS structure is stable, the specific metadata conventions can be highly mission-specific. A robust processing workflow, therefore, must not rely on hardcoded assumptions about header content but should programmatically inspect the header to make informed decisions.

Practical Interaction with astropy.io.fits

The Python package Astropy provides the definitive, community-standard library for working with FITS files through its astropy.io.fits module. It abstracts away the low-level details of the FITS format, allowing users to interact with FITS files in an intuitive, "Pythonic" way.12 Opening and Inspecting Files The primary function for opening a FITS file is fits.open(). It is best practice to use this within a with statement, which ensures the file is automatically closed when the block is exited, preventing resource leaks.7 Once opened, the function returns an HDUList object, which behaves like a Python list of HDUs. The info() method of this object is the first and most crucial step in understanding a file's contents, providing a summary of all HDUs, their names, types, and data dimensions.7

Python

# Import the necessary module from Astropy

from astropy.io import fits

# Define the path to a FITS file

# (Replace with a real FITS file path)

fits_file_path = 'example_image.fits'

# Open the FITS file using a 'with' block for safety

try: with fits.open(fits_file_path) as hdul: \# The hdul object is an HDUList, a list-like collection of HDUs. \# Use the.info() method for a high-level summary of the file structure. print("FITS File Structure:") hdul.info() except FileNotFoundError: print(f"Error: The file '{fits_file_path}' was not found.") except OSError: print(f"Error: Could not read the file '{fits_file_path}'. It may be corrupted or not a FITS file.")

Accessing HDUs Individual HDUs within the HDUList can be accessed in two primary ways: by their zero-based index or by a combination of their name and version number. Astropy uses zero-based indexing, where hdul is the Primary HDU, hdul\[1\] is the first extension, and so on.7 Modern multi-extension FITS files, especially from missions like HST and JWST, assign names to their extensions using the EXTNAME keyword. This allows for more readable and robust access. For example, a science image might be in an extension named 'SCI'. If there are multiple extensions with the same name, they are distinguished by the EXTVER keyword.6 Astropy supports accessing these HDUs with a tuple: hdul or, more simply, hdul.7

Python

# (Continuing from the previous block, assuming 'hdul' is an open HDUList)

# Access the Primary HDU (index 0)

primary_hdu = hdul print(f"\nAccessed Primary HDU (index 0). HDU Type: {type(primary_hdu)}")

# Access the first extension HDU (index 1)

if len(hdul) \> 1: first_extension_hdu = hdul\[1\] print(f"Accessed First Extension HDU (index 1). HDU Type: {type(first_extension_hdu)}")

```         
# Access an extension by name (e.g., 'SCI' for a science image)
# This is the recommended method for clarity and robustness.
try:
    # For an imset with EXTVER=1
    science_hdu = hdul 
    print(f"Successfully accessed HDU by name 'SCI' and version 1.")
except KeyError:
    print("Could not find an HDU with EXTNAME='SCI' and EXTVER=1.")
```

Reading Headers and Data Each HDU object has two main attributes: .header and .data. The .header attribute returns a Header object, which behaves much like a Python dictionary. You can retrieve the value of a keyword by using its name as a key (e.g., header). For convenience, Astropy makes keyword access case-insensitive.7 You can also iterate through all keywords or print the entire header. The .data attribute contains the actual scientific data. For an image HDU, this attribute returns a NumPy ndarray object.12 This is the critical link to the rest of the scientific Python ecosystem; once the data is a NumPy array, it can be sliced, manipulated, and analyzed using the full power of libraries like NumPy, SciPy, and Matplotlib.

Python

# (Continuing from the previous block)

# Let's assume the science data is in the first extension

if len(hdul) \> 1: science_hdu = hdul\[1\] \# Or hdul if applicable

```         
# Access the header of the science HDU
header = science_hdu.header
print("\n--- Science HDU Header ---")
# Print the entire header (as it appears in the FITS file)
# print(repr(header))

# Access specific keywords like a dictionary
if 'NAXIS' in header:
    print(f"Number of axes (NAXIS): {header}")
if 'NAXIS1' in header:
    print(f"Axis 1 length (NAXIS1): {header}")
if 'NAXIS2' in header:
    print(f"Axis 2 length (NAXIS2): {header}")
if 'FILTER' in header:
    print(f"Filter used (FILTER): {header}")

# Access the data as a NumPy array
data = science_hdu.data

# Check if data exists and print its properties
if data is not None:
    print("\n--- Science Data Array ---")
    print(f"Data type: {data.dtype}")
    print(f"Data shape: {data.shape}")
    
    # Example: Print the value of a single pixel at (row=10, col=10)
    # Note: NumPy uses (row, column) indexing, which is (y, x)
    if data.ndim == 2 and data.shape > 10 and data.shape[1] > 10:
        pixel_value = data[10, 10]
        print(f"Value of pixel at (10, 10): {pixel_value}")
else:
    print("\nNo data associated with this HDU.")
```

Handling Tables If an HDU is a binary table (XTENSION='BINTABLE'), its .data attribute will return a FITS_rec object. While this is a NumPy structured array, it is often more convenient to work with it as an astropy.table.Table object. This provides a more powerful and user-friendly interface for sorting, filtering, and accessing columns by name.14

Python

from astropy.table import Table

# Example of reading a FITS file with a binary table extension

# (Replace with a FITS file containing a table, e.g., an event file)

table_fits_file = 'example_table.fits'

try: with fits.open(table_fits_file) as hdul: if len(hdul) \> 1 and hdul.\[1\]is_image is False: \# Access the table data table_data = hdul.\[1\]data

```         
        # Convert the FITS_rec object to an Astropy Table for easier use
        astropy_table = Table(table_data)
        
        print("\n--- FITS Table Data ---")
        # Print the table summary
        astropy_table.info()
        
        # Print the first 5 rows
        print("\nFirst 5 rows:")
        print(astropy_table[:5])
        
        # Access a column by name
        if 'RA' in astropy_table.colnames:
            ra_column = astropy_table
            print(f"\nFirst 5 values of the 'RA' column:\n{ra_column[:5]}")
```

except (FileNotFoundError, OSError, IndexError): print(f"\nCould not process table from '{table_fits_file}'.")

2.  The Python Astro-Imaging Ecosystem: A Curated Toolkit

Creating a professional astronomical image requires more than just reading a FITS file. It involves a workflow that spans data acquisition, calibration, alignment, scaling, and color composition. The Python ecosystem provides a suite of powerful, interoperable libraries designed to handle each stage of this process. These tools are built upon a common foundation provided by Astropy, ensuring that data flows seamlessly from one step to the next. Understanding the role of each library is key to constructing an efficient and robust imaging pipeline.

Core Libraries

● Astropy: This is the cornerstone of the Python astronomical software stack. It provides not only the astropy.io.fits module for file handling but also a rich set of sub-packages that are fundamental to data analysis.13 ○ astropy.table: Offers a powerful Table object for manipulating tabular data, essential for working with catalogs and FITS binary tables.14 ○ astropy.wcs: Provides tools for handling the World Coordinate System (WCS), which maps pixel coordinates in an image to celestial coordinates (e.g., Right Ascension and Declination). This is indispensable for image alignment. ○ astropy.units: Introduces a Quantity object that attaches physical units to numerical values, preventing common errors in calculations and ensuring dimensional consistency.17 ○ astropy.visualization: A critical module for image creation, offering a framework for image normalization (scaling) and non-linear stretching, as well as functions for creating publication-quality RGB composite images.18 ● Astroquery: This package serves as a standardized, programmatic interface to dozens of online astronomical archives and databases.22 Instead of manually searching web portals and downloading files, astroquery allows users to script these tasks. It provides mission-specific modules for accessing major archives like the Mikulski Archive for Space Telescopes (MAST), which hosts data for JWST, HST, GALEX, and Pan-STARRS, as well as archives for Euclid and Chandra.23 The results of queries are typically returned as astropy.table.Table objects, demonstrating the tight integration within the ecosystem. ● reproject: This package is dedicated to one crucial task: reprojecting astronomical images. Reprojection is the process of transforming an image from one WCS projection to another, effectively "warping" it to align with a reference image.26 This is a non-negotiable step when combining images taken with different filters, at different times, or with different instruments, as it ensures that a star at a given celestial coordinate appears at the same pixel location in all images.28 ● ccdproc: An Astropy-affiliated package designed for the fundamental processing of raw CCD (Charge-Coupled Device) images.30 It provides tools for essential calibration steps like overscan correction, bias subtraction, dark current removal, and flat-fielding.30 While many data products from major observatories are delivered pre-calibrated, ccdproc is invaluable for users working with raw data or needing to perform custom reductions. It introduces the CCDData object, which extends NumPy arrays with metadata, units, and uncertainty, propagating errors through the reduction process.16 ● NumPy & Matplotlib: These are the foundational libraries for all scientific computing and plotting in Python. NumPy provides the ndarray object, the high-performance array that holds all image data.12 Matplotlib is the primary library for creating static 2D plots and images, and it is the final destination for the RGB image arrays created by astropy.visualization.19 The power of this ecosystem lies not in any single library but in their designed interoperability. The system is built on a philosophy of shared, high-level data structures provided by Astropy. An astroquery search returns an astropy.table.Table. An image read by astropy.io.fits provides a NumPy array and a header that can be used to construct an astropy.wcs.WCS object. These components are then consumed directly by reproject. A ccdproc.CCDData object is built upon a NumPy array and an Astropy header. This layered architecture, where the output of one tool serves as the natural input for another, enables the creation of powerful, modular, and readable data processing pipelines. Mastering the core Astropy objects—Table, HDUList, WCS, and Quantity—is therefore the key to unlocking the full potential of this integrated toolkit. \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Part II: Data Acquisition and Preparation

Before any image can be created, the raw materials must be gathered and prepared. This section details the practical steps of programmatically acquiring data from major astronomical archives and understanding the specific file structures of different missions. It then addresses the critical process of image registration, ensuring that multiple datasets are spatially aligned and ready for combination. This preparatory phase is foundational; errors or misunderstandings at this stage will propagate through the entire workflow.

3.  Acquiring the Building Blocks: Programmatic Data Access

Manually downloading data from web archives is time-consuming, prone to error, and not reproducible. The astroquery package provides a robust, scriptable solution for accessing data from a wide range of astronomical archives. The API for each archive is tailored to its specific database structure and query logic, requiring a mission-aware approach to data acquisition.

The MAST Portal (astroquery.mast)

The Mikulski Archive for Space Telescopes (MAST) is the primary NASA archive for optical, ultraviolet, and near-infrared data, hosting observations from missions including JWST, HST, GALEX, and Pan-STARRS.23 The astroquery.mast module provides a comprehensive interface to this rich dataset. Authentication While much of the data in MAST is public, some datasets may be proprietary (e.g., recently acquired observations). Accessing these requires authentication. This is typically handled via an authentication token generated from the MAST portal, which can be used to log in to a session.23

Python

from astroquery.mast import Observations

# For proprietary data, you would first obtain a token from https://auth.mast.stsci.edu/

# and then log in. For public data, this step is not required.

# Example login (replace 'your_token_here' with an actual token):

# try:

# Observations.login('your_token_here')

# except Exception as e:

# print(f"MAST login failed: {e}")

Querying Observations The Observations class is the main tool for finding data. The query_object() method allows searching for all observations of a specific astronomical target, while query_criteria() enables more complex searches based on parameters like instrument, filter, exposure time, or proposal ID.23

Python

from astroquery.mast import Observations from astropy.time import Time

# Example 1: Query for all JWST NIRCam observations of the galaxy M101

target_name = "M101" print(f"Querying MAST for JWST/NIRCam observations of {target_name}...")

obs_table = Observations.query_object(target_name, radius="0.1 deg")

# Filter the results for specific criteria

jwst_nircam_obs = obs_table\[ (obs_table\['instrument_name'\] == 'NIRCAM/IMAGE') & (obs_table\['obs_collection'\] == 'JWST')\]

print(f"Found {len(jwst_nircam_obs)} JWST/NIRCam observations.") if len(jwst_nircam_obs) \> 0: print("Available filters:", set(jwst_nircam_obs\['filters'\]))

# Example 2: Query for HST WFC3 data using specific criteria

print("\nQuerying MAST for recent HST/WFC3 observations...")

criteria = { "instrument_name": "WFC3/UVIS", "filters": "F606W", "t_min": }

hst_obs_table = Observations.query_criteria(\*\*criteria) print(f"Found {len(hst_obs_table)} HST/WFC3 observations matching criteria.")

Downloading Products Once a table of desired observations is obtained, the next step is to download the associated data products. This is a two-step process: first, use get_product_list() to see all available files for a given observation, and then use download_products() to retrieve them. The download_products() function is powerful, allowing filtering by product type (e.g., productType="SCIENCE" to get only calibrated science files) and description.23

Python

from astroquery.mast import Observations

# Let's use the first JWST observation from the previous query

if 'jwst_nircam_obs' in locals() and len(jwst_nircam_obs) \> 0: obs_id_to_download = jwst_nircam_obs\['obsid'\] print(f"\nFetching product list for observation ID: {obs_id_to_download}")

```         
# Get the list of all data products for this observation
products = Observations.get_product_list(obs_id_to_download)

# Filter for calibrated science products (e.g., '_cal.fits' files)
science_products = Observations.filter_products(products,
                                                productSubGroupDescription="CAL",
                                                mrp_only=False) # mrp_only=False includes non-default products

print(f"Found {len(science_products)} calibrated science products.")

if len(science_products) > 0:
    # Download the first 3 science products for this example
    print("Downloading first 3 science products...")
    manifest = Observations.download_products(science_products[0:3],
                                              download_dir="mast_data")
    print("\nDownload manifest:")
    print(manifest)
```

The Euclid Science Archive (astroquery.esa.euclid)

The ESA Euclid Science Archive is structured as a large relational database, and the primary way to interact with it is through the Astronomical Data Query Language (ADQL), which is a variant of SQL.25 The astroquery.esa.euclid module provides the interface to this system. The archive is divided into several environments, with PDR (Public Data Release) being the default for public access.25 Connecting and Querying Queries are executed using the launch_job() (for synchronous, small queries) or launch_job_async() (for asynchronous, large queries) methods. The query itself is an ADQL string.

Python

from astroquery.esa.euclid import Euclid from astropy.coordinates import SkyCoord import astropy.units as u

# It's good practice to check the service status first

try: status = Euclid.get_status_messages() print("Euclid Archive Status:", status) except Exception as e: print(f"Could not retrieve Euclid status: {e}")

# Example: Perform a cone search for calibrated images (DpdMerBksMosaic)

# around a specific coordinate using an ADQL query.

coords = SkyCoord("150.116 deg", "2.201 deg", frame='icrs') radius_deg = 0.1

adql_query = f""" SELECT \* FROM catalogue.mer_mosaic_product WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {coords.ra.deg}, {coords.dec.deg}, {radius_deg})) AND instrument_name = 'NISP' """

print("\nLaunching ADQL query to Euclid archive...") try: job = Euclid.launch_job_async(adql_query) euclid_results = job.get_results() print(f"Query successful. Found {len(euclid_results)} results.") if len(euclid_results) \> 0: print(euclid_results\['file_name', 'filter_name', 'ra', 'dec'\]) except Exception as e: print(f"Euclid query failed: {e}")

Downloading Data Once a table of products is retrieved, specific files can be downloaded using methods like get_product().

Python

# (Continuing from the Euclid query block)

if 'euclid_results' in locals() and len(euclid_results) \> 0: file_to_download = euclid_results\['file_name'\] print(f"\nDownloading Euclid product: {file_to_download}") try: \# The get_product method downloads the file Euclid.get_product(file_to_download, filename=f"euclid_data/{file_to_download}") print("Download complete.") except Exception as e: print(f"Euclid download failed: {e}")

The Chandra Data Archive (pycrates)

The Chandra X-ray Center (CXC) provides its own software suite, CIAO (Chandra Interactive Analysis of Observations), for data analysis. The recommended Python interface for file I/O is the pycrates module, which is part of CIAO. This is because pycrates is designed to understand the specific Chandra Data Model, including virtual file syntax for filtering and binning, which astropy.io.fits does not recognize.40 While astroquery can search the Chandra archive, the official and most robust method for data manipulation is via CIAO tools. Chandra data is fundamentally different from optical/IR data; the primary science product is an "event list" (evt2.fits), which is a table where each row represents a single detected photon with its position, arrival time, and energy.41 Reading Event Files The pycrates.read_file() function is used to open a FITS file and load the most relevant HDU (typically the 'EVENTS' block) into a Crate object. From this object, columns can be extracted as CrateData objects, whose values are accessible as NumPy arrays.40

Python

# This code requires a working CIAO installation with pycrates.

try: import pycrates

```         
# Path to a Chandra level 2 event file
chandra_evt_file = 'acisf13770_repro_evt2.fits'

print(f"Reading Chandra event file: {chandra_evt_file}")

# read_file loads the 'EVENTS' block by default
cr = pycrates.read_file(chandra_evt_file)

print("Crate information:", cr)
print("Available columns:", cr.get_colnames())

# Extract columns needed for imaging: sky coordinates (x, y) and energy
x_col = cr.get_column('x')
y_col = cr.get_column('y')
energy_col = cr.get_column('energy')

# Access the data as NumPy arrays
x_vals = x_col.values
y_vals = y_col.values
energy_vals = energy_col.values

print(f"\nSuccessfully read {len(x_vals)} events.")
print(f"Energy range: {energy_vals.min():.2f} to {energy_vals.max():.2f} eV")
```

except ImportError: print("\nWarning: CIAO 'pycrates' module not found. Skipping Chandra data access example.") except Exception as e: print(f"\nAn error occurred while reading the Chandra file: {e}")

The distinct query methods for each archive highlight a fundamental principle of programmatic data access: the API is an abstraction of the archive's underlying architecture. MAST, serving many missions, provides an observation-centric query system. The Euclid archive, designed as a massive, unified database, is best queried with the SQL-like ADQL. Chandra's data products are event lists, and its tools are optimized for filtering and manipulating these lists. Therefore, an effective data acquisition script must be tailored to the logic of its target archive, asking "What observations exist?" for MAST, "Select from this table where..." for Euclid, and "Give me events that satisfy..." for Chandra. Table 1: Data Acquisition API Summary Archive Name Primary Python Module Key Query Function(s) Query Language/Paradigm Authentication Method MAST (JWST, HST, etc.) astroquery.mast Observations.query_object() Observations.query_criteria() Object- and parameter-based MAST API Token Euclid Science Archive astroquery.esa.euclid Euclid.launch_job_async() Euclid.cone_search() Astronomical Data Query Language (ADQL) ESA User Credentials Chandra Data Archive astroquery.cda (for search) pycrates (for I/O) Cda.query_object() pycrates.read_file() Object-based (search) Data Model I/O (local) N/A for public data

4.  Deconstructing the Data: A Guide to Mission-Specific FITS Formats

Once downloaded, the FITS files must be correctly interpreted. While all adhere to the same base standard, each mission and instrument has a unique convention for organizing data within the multi-extension FITS structure. Understanding these conventions is crucial for correctly extracting the science image, its associated uncertainties, and data quality information.

JWST & HST: The Multi-Extension FITS (MEF) Standard

Data from the Hubble Space Telescope (HST) and James Webb Space Telescope (JWST) are typically delivered as multi-extension FITS (MEF) files. These files group logically related data arrays into a single container. A set of these related extensions is called an "imset".6 The most common and fundamental imset for a calibrated image consists of three extensions 6: ● SCI (Science): A 2D or 3D array containing the calibrated science data. The pixel values are typically in physical units like electrons/second or surface brightness (e.g., MJy/sr). ● ERR (Error): An array of the same dimensions as SCI, providing an estimate of the uncertainty or standard deviation for each pixel. ● DQ (Data Quality): An integer array where each bit of the integer value at a given pixel location represents a specific flag (a "bitmask"). These flags indicate known issues with the pixel, such as it being a "hot" or "dead" pixel, saturated, or affected by a cosmic ray.43 These extensions are identified by the EXTNAME keyword in their respective headers. If an observation contains multiple exposures or integrations in a single file, they are distinguished by the EXTVER keyword (e.g., ('SCI', 1), ('SCI', 2)).

Python

from astropy.io import fits

# Example for a typical HST or JWST calibrated file (e.g., '\_flt.fits' or '\_cal.fits')

hst_jwst_file = 'jdl803dxq_flt.fits' \# Replace with an actual file path

try: with fits.open(hst_jwst_file) as hdul: hdul.info()

```         
    # Access the first science image, its error, and data quality arrays
    # The tuple ('EXTNAME', EXTVER) is used for access.
    sci_data = hdul.data
    err_data = hdul.data
    dq_data = hdul.data
    
    print(f"\nSuccessfully loaded data for imset 1:")
    print(f"  SCI array shape: {sci_data.shape}, dtype: {sci_data.dtype}")
    print(f"  ERR array shape: {err_data.shape}, dtype: {err_data.dtype}")
    print(f"  DQ array shape: {dq_data.shape}, dtype: {dq_data.dtype}")
```

except (FileNotFoundError, KeyError) as e: print(f"\nCould not process file '{hst_jwst_file}': {e}")

Euclid: Calibrated VIS and NISP Frames

Euclid data products also follow a MEF structure, though with slight variations between its two instruments, VIS (Visible instrument) and NISP (Near-Infrared Spectrometer and Photometer).44 ● NISP Calibrated Images (DpdNirCalibratedFrame): These files are highly structured. For each of the 16 NISP detectors, the MEF file contains three extensions: SCI (the calibrated science image), RMS (Root Mean Square, which serves as the error array), and DQ (Data Quality flags).44 ● VIS Calibrated Images (DpdVisCalibratedFrame): The documentation suggests that VIS data products may be delivered as separate files rather than a single MEF containing all components. A user might receive a ...-DET-...fits file for the science image, a ...-WGT-...fits file for the weight map (which is related to the error, often as $1/\sigma^2$), and potentially other files for background maps.44 This requires the user to associate these separate files based on their filenames and header information.

Chandra: From Event Lists to Images

As previously noted, the primary data product from the Chandra X-ray Observatory is not an image but an event list (evt2.fits). This file is a FITS binary table where each row corresponds to a detected photon.41 To create an image, these events must be binned into a 2D grid. This process fundamentally loses information (specifically, the precise time and energy of each individual photon) but is necessary for visualization and many types of spatial analysis.41 The process involves filtering the event list by energy to create specific bands and then using a 2D histogram to create the image array. The standard Chandra energy bands for creating three-color images are 47: ● Soft band: 0.5–1.2 keV ● Medium band: 1.2–2.0 keV ● Hard band: 2.0–7.0 keV The following code demonstrates this crucial transformation.

Python

import numpy as np from astropy.io import fits

# This example uses astropy.io.fits for simplicity, but pycrates is recommended for production work.

chandra_evt_file = 'acisf13770_repro_evt2.fits' \# Replace with actual file path

try: with fits.open(chandra_evt_file) as hdul: \# The event data is in the 'EVENTS' extension, which is a binary table. events = hdul.data

```         
    # Chandra energies are typically in eV, so we define our bands in eV.
    energy_bands = {
        'soft': (500, 1200),
        'medium': (1200, 2000),
        'hard': (2000, 7000)
    }
    
    # Get the image dimensions and binning from the header WCS keywords.
    # This assumes a simple tangent plane projection.
    header = hdul.header
    image_size_x = header * 2
    image_size_y = header * 2
    
    # Create a dictionary to hold the image arrays for each band
    chandra_images = {}
    
    for band, (emin, emax) in energy_bands.items():
        print(f"Binning events for '{band}' band ({emin}-{emax} eV)...")
        
        # 1. Filter events by energy
        energy_mask = (events['energy'] >= emin) & (events['energy'] < emax)
        filtered_events = events[energy_mask]
        
        # 2. Use numpy.histogram2d to create an image from x, y coordinates
        # The range and bins are determined from header info to match the native pixel grid.
        img, _, _ = np.histogram2d(
            x=filtered_events['x'],
            y=filtered_events['y'],
            bins=(int(image_size_x), int(image_size_y)),
            range=[[0.5, image_size_x + 0.5], [0.5, image_size_y + 0.5]]
        )
        
        # The histogram function returns a transposed array, so we correct it.
        chandra_images[band] = img.T
        
        print(f"  Created '{band}' image with shape {chandra_images[band].shape} and {len(filtered_events)} events.")
        
```

except (FileNotFoundError, KeyError) as e: print(f"\nCould not process Chandra file '{chandra_evt_file}': {e}")

GALEX & Pan-STARRS: Navigating Survey Products

Data from large sky surveys often have simpler FITS structures for individual images but rely on filename conventions and separate catalogs for context. ● GALEX: The Galaxy Evolution Explorer data products are typically single-extension FITS files where the image is in the Primary HDU.15 The key information is encoded in the filename. For example, ...-nd-int.fits indicates an intensity map (-int) from the NUV detector (-nd-).48 There are often corresponding -exp.fits (exposure map) and -skybg.fits (sky background) files. ● Pan-STARRS: The Pan-STARRS survey provides two main types of image products: "stack" images, which are deep co-adds of multiple exposures, and single-epoch "warp" images, which are individual exposures resampled onto a common sky grid.49 Both are typically delivered as FITS files with the science data in the Primary HDU. The filter used (g, r, i, z, y) is a critical piece of metadata found in the header. The extensive source catalogs derived from these images are usually provided separately in database format.49 Table 2: Mission FITS Structure Reference Mission/Instrument Product Type Primary HDU Content Key Extension Names (EXTNAME) Purpose of Extension JWST (e.g., NIRCam) Calibrated Image (\_cal.fits) Metadata only SCI, ERR, DQ, VAR_POISSON, VAR_RNOISE Science data, Uncertainty, Data Quality Flags, Variance components HST (e.g., ACS/WFC) Calibrated Image (\_flc.fits) Metadata only SCI, ERR, DQ Science data, Uncertainty, Data Quality Flags Euclid (NISP) Calibrated Frame Metadata only SCI, RMS, DQ (per detector) Science data, Root Mean Square (error), Data Quality Flags Chandra (ACIS/HRC) Level 2 Event File (\_evt2.fits) Metadata only EVENTS, GTI Photon event list, Good Time Intervals GALEX Intensity Map (-int.fits) Science Image N/A (single extension) NUV or FUV intensity map Pan-STARRS1 Stack/Warp Image Science Image N/A (typically single extension) Co-added or single-epoch image in a specific filter

5.  Achieving Spatial Coherence: Image Registration and Reprojection

Astronomical images taken through different filters, with different instruments, or at different times are almost never perfectly aligned on a pixel-by-pixel basis. They may have different pixel scales, orientations on the sky, and optical distortions. To combine these images into a single color composite, they must first be transformed onto a common pixel grid. This process is known as image registration or reprojection, and it is accomplished by leveraging the World Coordinate System (WCS) information embedded in the FITS headers of each image.27 The WCS acts as a bridge, providing the precise mathematical transformation between the internal pixel coordinates (x, y) of an image and the absolute celestial coordinates (e.g., Right Ascension, Declination) on the sky.

The reproject Library

The reproject Python package, an Astropy-affiliated package, is the standard tool for this task. It provides functions that can take a source image and its WCS and "warp" it to match the WCS and pixel grid of a target image.26 The library offers several algorithms for this process, with the most common being: ● reproject_interp: This function uses interpolation (e.g., bilinear, bicubic) to calculate the pixel values in the new grid. It is relatively fast and suitable for most visualization purposes.57 ● reproject_exact: This flux-conserving algorithm calculates the exact area of overlap between each pixel in the source and destination grids. It is significantly more computationally intensive but provides the most accurate results, making it ideal for scientific applications where photometry must be preserved.58

Practical Workflow for Image Alignment

The standard workflow for aligning a set of images for color compositing is as follows: 1. Select a Target Frame: Choose one of the images to serve as the reference. A common choice is the image taken with the longest wavelength filter, as it often has the lowest resolution due to diffraction, or simply the image with the desired final orientation and pixel scale. 2. Extract Target WCS and Shape: Load the target image's FITS file and extract its WCS object from the header using astropy.wcs.WCS. Also, note the shape of its data array, as this will define the dimensions of the output images. 3. Iterate and Reproject: Loop through the remaining source images. For each one, load its data and WCS. Call one of the reproject functions, providing the source data and WCS, the target WCS, and the desired output shape. The function will return a new NumPy array containing the reprojected image.

Python

import numpy as np from astropy.io import fits from astropy.wcs import WCS from reproject import reproject_interp

# Assume we have a list of FITS filenames for different filters of the same object

# These files must have valid WCS information in their headers.

# Example: \['m51_f435w.fits', 'm51_f555w.fits', 'm51_f814w.fits'\]

filter_files = { 'blue': 'path/to/blue_filter.fits', 'green': 'path/to/green_filter.fits', 'red': 'path/to/red_filter.fits' }

# 1. Select the 'red' filter image as the target reference frame.

target_filename = filter_files\['red'\]

try: with fits.open(target_filename) as hdul: \# 2. Extract the target WCS and shape from the primary science HDU target_wcs = WCS(hdul.header) target_shape = hdul.shape

```         
print(f"Target frame defined by '{target_filename}' with shape {target_shape}.")

aligned_images = {}
# The target image is already aligned to itself
aligned_images['red'] = fits.getdata(target_filename, ext=('SCI', 1))

# 3. Iterate through the other filters and reproject them
for color, filename in filter_files.items():
    if filename == target_filename:
        continue # Skip the target file itself
    
    print(f"Reprojecting '{filename}' to match the target frame...")
    
    with fits.open(filename) as hdul:
        source_hdu = hdul
        source_data = source_hdu.data
        source_wcs = WCS(source_hdu.header)
        
        # Perform the reprojection using interpolation
        # The output is a tuple: (reprojected_array, footprint_array)
        reprojected_data, footprint = reproject_interp(
            (source_data, source_wcs),
            target_wcs,
            shape_out=target_shape
        )
        
        aligned_images[color] = reprojected_data
        print(f"  Reprojection complete for '{color}' channel.")

print("\nAll images are now aligned to a common pixel grid.")
```

except Exception as e: print(f"\nAn error occurred during reprojection: {e}") print("Please ensure all FITS files exist and have valid WCS headers.")

It is important to recognize that reprojection is not a lossless operation. The process of interpolation inherently involves resampling the data, which can subtly smooth the image and introduce a small degree of error or uncertainty.59 For visualization, these effects are generally negligible. However, for high-precision scientific measurements like photometry, it is often preferable to perform the analysis on the original, un-reprojected images, using WCS transformations to correlate source positions between frames. The choice of reprojection algorithm represents a trade-off between computational speed and photometric accuracy, a consideration that should be guided by the ultimate goal of the analysis. \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Part III: The Art and Science of Color Compositing

With the data acquired, understood, and spatially aligned, the final stage is to transform these numerical arrays into a visually compelling and scientifically informative color image. This is not merely a cosmetic step but a crucial part of data visualization. It involves carefully scaling the vast dynamic range of astronomical data to fit the limited range of a computer display and then mapping different observational filters to color channels in a way that reveals the underlying physics of the object.

6.  Bringing Out the Faint Universe: Image Scaling and Stretching

A raw astronomical FITS image, when displayed directly, is often underwhelming. It typically appears as a mostly black frame with a few saturated white pixels corresponding to bright stars or the core of a galaxy. This is because the data has a very high dynamic range—the ratio of the brightest to the faintest detectable signal can be thousands or even millions to one. In contrast, a standard computer display can only show 256 levels of brightness per color channel (for an 8-bit display).60 To make the faint, scientifically interesting structures visible without completely saturating the bright regions, the data must be scaled and stretched. The astropy.visualization module provides a powerful and flexible framework for this task, conceptualizing it as a two-step process: defining an interval to map to the black-to-white range, and then applying a stretching function to the values within that interval.20

Step 1: Interval (Normalization)

The interval determines which data values will be mapped to black (the minimum) and white (the maximum). Any values below the minimum will also appear black, and any above the maximum will also appear white. Astropy provides several classes to automatically determine this interval: ● MinMaxInterval(): The simplest approach, mapping the absolute minimum value in the data to black and the absolute maximum to white. This is often not ideal, as a single hot pixel or cosmic ray can skew the entire scale. ● PercentileInterval(percentile): A much more robust method. For example, PercentileInterval(99.5) sets the 99.5th percentile of the data values as the white point. This effectively ignores the brightest 0.5% of pixels, preventing bright stars from dominating the color scale and allowing fainter details to become visible.20 ● ZScaleInterval(): Implements the zscale algorithm, a robust method used widely in astronomical software like DS9. It computes the interval based on the median and contrast of the image, providing a good starting point for a wide variety of images without requiring manual tweaking of percentiles.21

Step 2: Stretch (Transformation)

Once the interval is defined, the values within it are mapped to the display range of . A linear mapping is often insufficient for astronomical data. Non-linear stretching functions are used to compress the dynamic range, enhancing contrast in either the faint or bright parts of the image. Key BaseStretch classes include 21: ● LinearStretch(): A direct mapping. Data values halfway between the minimum and maximum are mapped to 50% gray. ● SqrtStretch(): A square root stretch, which enhances fainter features. ● LogStretch(): A logarithmic stretch, which strongly compresses the bright end of the data, revealing very faint details.62 ● AsinhStretch(): An inverse hyperbolic sine stretch. This function is particularly effective for astronomical images because it behaves linearly for faint signals (where noise often dominates) and logarithmically for bright signals. This unique property allows it to display faint nebulosity while preserving detail in bright stellar cores, making it a preferred choice for high-dynamic-range data.21

Practical Application with ImageNormalize

The ImageNormalize class combines an interval and a stretch object into a single normalization object that Matplotlib can use directly. This allows for a clean and modular approach to displaying single-band, scaled images.

Python

import matplotlib.pyplot as plt from astropy.io import fits from astropy.visualization import ( AsinhStretch, LogStretch, PercentileInterval, ImageNormalize )

# Load a single-band science image (e.g., one of the aligned images from before)

# aligned_images\['red'\] would be a NumPy array from the previous step.

# For a standalone example, we load a file:

try: image_data = fits.getdata('path/to/red_filter.fits', ext=('SCI', 1))

```         
# --- Create different normalization schemes ---

# 1. A simple log stretch with a 99th percentile cut
log_norm = ImageNormalize(image_data, 
                          interval=PercentileInterval(99.0), 
                          stretch=LogStretch(a=1000))

# 2. An asinh stretch using ZScale for interval determination
# This is often a very good starting point.
asinh_norm = ImageNormalize(image_data, 
                            interval=ZScaleInterval(), 
                            stretch=AsinhStretch(a=0.1))

# --- Display the images ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

im0 = axes.imshow(image_data, origin='lower', cmap='gray', norm=log_norm)
axes.set_title('Log Stretch with 99% Interval')
fig.colorbar(im0, ax=axes)

im1 = axes.[1]imshow(image_data, origin='lower', cmap='gray', norm=asinh_norm)
axes.[1]set_title('Asinh Stretch with ZScale Interval')
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
```

except (FileNotFoundError, KeyError) as e: print(f"Error loading image for visualization: {e}")

Table 3: Image Stretch Function Guide Stretch Name (astropy.visualization class) Mathematical Form Key Parameter(s) Use Case / Characteristics LinearStretch $y = x$ None Simple, direct mapping. Best for low dynamic range or linear data. LogStretch $y = \frac{\log(ax + 1)}{\log(a + 1)}$ a Strongly compresses bright values. Excellent for revealing very faint structures. SqrtStretch $y = \sqrt{x}$ None Mild non-linear stretch that boosts faint and mid-range values. PowerStretch $y = x^a$ a General power-law stretch. a \< 1 boosts faint values; a \> 1 boosts bright values. AsinhStretch $y = \frac{\text{asinh}(ax)}{\text{asinh}(a)}$ a Hybrid stretch: linear at low values, logarithmic at high values. Gold standard for high dynamic range.

7.  Synthesizing the View: Creating Multi-Band Composite Images

A color image is created by combining three single-band grayscale images, assigning each to one of the three primary color channels: Red, Green, and Blue (RGB). The choices made during this combination process determine both the aesthetic appeal and the scientific interpretability of the final image.

Principles of Astronomical Color

Since telescopes like JWST observe in infrared wavelengths invisible to the human eye, the resulting images are "false color" or, more accurately, "representative color".64 The goal is not to reproduce what a human would see, but to create a visually intuitive representation of the physical information contained in the data. The guiding principle for this is chromatic ordering: mapping the images from longest to shortest wavelength to the R, G, and B channels, respectively.66 For example, when combining images from red, green, and blue optical filters, the mapping is direct. When combining infrared images from JWST, an image from a 4.4 µm filter would be mapped to red, a 2.0 µm filter to green, and a 0.9 µm filter to blue. This convention ensures that intrinsically redder objects (which are brighter at longer wavelengths) appear red in the final image.

The Lupton Algorithm (make_lupton_rgb)

A simple combination of three independently scaled images often leads to a problem: very bright objects, like stars or galactic nuclei, will saturate all three channels, appearing as uniform white blobs. This "burnt-out" effect destroys all color information in the brightest parts of the image.63 The algorithm developed by Lupton et al. (2004) provides an elegant solution to this problem. It is implemented in astropy.visualization as the make_lupton_rgb function.18 Instead of stretching each R, G, and B channel independently, it first calculates a total intensity, $I$, by averaging the three channels. This intensity is then stretched using an asinh function. The final color of a pixel is determined by scaling its original R, G, and B values by this single, stretched intensity factor. Because the scaling is applied uniformly to all three channels based on the total brightness, the color ratios between the channels are preserved, even in highly saturated regions.67 A star that is intrinsically reddish will therefore remain reddish in its core instead of turning white. This makes make_lupton_rgb the preferred method for creating professional-quality, high-dynamic-range color composites. The key parameters for tuning a Lupton image are: ● stretch: A linear stretch factor that controls the overall brightness and contrast. Lower values increase contrast. ● Q: The asinh softening parameter, which controls the transition from linear to logarithmic scaling. Higher values make the stretch more linear. ● minimum: Sets the black point for the image.

Python

from astropy.visualization import make_lupton_rgb

# Assume 'aligned_images' is a dictionary from the reprojection step

# containing NumPy arrays for 'red', 'green', and 'blue' channels.

try: image_r = aligned_images\['red'\] image_g = aligned_images\['green'\] image_b = aligned_images\['blue'\]

```         
# Create an RGB image using the Lupton algorithm
# These parameters often require some experimentation to find the best values.
rgb_image = make_lupton_rgb(image_r, image_g, image_b,
                            stretch=0.5,
                            Q=8,
                            minimum=0.0) # Set background to black

# Display the final image
plt.figure(figsize=(10, 10))
plt.imshow(rgb_image, origin='lower')
plt.title("Lupton RGB Composite")
plt.axis('off')
plt.show()

# Save the image to a file
# The file type is determined from the extension.
# from matplotlib.image import imsave
# imsave('lupton_composite.png', rgb_image)
```

except NameError: print("Error: 'aligned_images' not defined. Please run reprojection step first.")

The Simpler Approach (make_rgb)

Astropy also provides a simpler function, make_rgb, which scales and stretches each of the three color channels independently before combining them.68 This is analogous to adjusting the "Levels" or "Curves" for each color channel separately in an image editing program. While this method gives the user more direct control over the color balance, it is susceptible to the color-desaturation issue in bright regions. It can be useful for artistic purposes or for specifically highlighting a feature that is bright in only one filter.19

Guided Techniques and Fine-Tuning

Creating a final, polished image is an iterative process that blends science with aesthetics. Beyond the basic combination, several guided techniques can be employed: ● Black Point Adjustment: Carefully setting the minimum value or using an Interval object (like ManualInterval or PercentileInterval) with make_lupton_rgb is crucial for ensuring the sky background is truly black and not a muddy gray. ● Channel Balancing: If one filter image is significantly fainter or noisier than the others, it can be pre-scaled by a constant factor before being passed to the compositing function to achieve a better color balance. ● Narrowband and Multi-Wavelength Compositing: The RGB paradigm is not limited to three broadband filters. A common technique is to combine broadband data with narrowband data that isolates specific emission lines. For example, an image of an HII region might map a narrowband H-alpha filter (which traces ionized hydrogen) to the red channel, and then use two broadband filters for the green and blue channels. This creates a visually striking image that also clearly delineates regions of active star formation.47 Similarly, data from entirely different wavelength regimes can be combined, such as mapping Chandra X-ray data to blue, HST optical data to green, and JWST infrared data to red, to visualize the interplay of different physical processes in a single object. \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Part IV: Practical Case Studies

This final part synthesizes the concepts and techniques from the preceding sections into complete, end-to-end workflows. Each case study addresses a distinct astronomical object and utilizes data from different observatories, providing practical, reusable code templates. These examples demonstrate the entire process from data acquisition to final color composite, reinforcing the decision-making and problem-solving steps involved in creating professional-quality astronomical images.

8.  Case Study 1: A Spiral Galaxy with HST

Objective: To create a classic, "natural-looking" color composite of a spiral galaxy using broadband optical data from the Hubble Space Telescope. This workflow represents a foundational use case for multi-extension FITS files, reprojection, and the Lupton algorithm. Target: Messier 51 (The Whirlpool Galaxy) Data Source: HST Advanced Camera for Surveys (ACS) Filters: F814W (I-band), F555W (V-band), F435W (B-band)

Workflow and Code

Step 1: Data Acquisition First, use astroquery.mast to find and download the relevant calibrated (\_flc.fits) files for M51. We will search for data from a specific proposal to ensure we get a consistent set of observations.

Python

import os from astroquery.mast import Observations

# Create a directory to store the data

data_dir = "m51_hst_data" os.makedirs(data_dir, exist_ok=True)

print("Querying MAST for HST/ACS data of M51...") obs_table = Observations.query_criteria( target_name="M51", obs_collection="HST", instrument_name="ACS/WFC", proposal_id="10452" \# A well-known HST survey of M51 )

# Filter for the specific filters we want

filters_to_download = filtered_obs = obs_table\[\[(f in filters_to_download) for f in obs_table\['filters'\]\]\]

print(f"Found {len(filtered_obs)} relevant observations.")

# Download the calibrated science products ('\_flc.fits')

if len(filtered_obs) \> 0: products = Observations.get_product_list(filtered_obs) science_products = Observations.filter_products(products, productSubGroupDescription="FLC")

```         
print(f"Downloading {len(science_products)} FLC files...")
manifest = Observations.download_products(science_products, download_dir=data_dir)
print("Download complete.")
print(manifest)
```

else: print("No matching observations found.")

Step 2: Data Inspection and File Identification After downloading, identify the specific FITS file for each filter. The filenames often contain the filter information.

Python

import glob

# Identify the downloaded files for each filter

try: file_paths = { "blue": glob.glob(os.path.join(data_dir, 'mastDownload/HST/*f435w*.fits')), "green": glob.glob(os.path.join(data_dir, 'mastDownload/HST/*f555w*.fits')), "red": glob.glob(os.path.join(data_dir, 'mastDownload/HST/*f814w*.fits')) } print("Identified FITS files for each filter:") for color, path in file_paths.items(): print(f" {color.capitalize()}: {os.path.basename(path)}") except IndexError: print("Error: Could not find downloaded FITS files for all required filters.") \# Exit or handle error appropriately

Step 3: Image Reprojection Align the F435W (blue) and F555W (green) images to the coordinate system of the F814W (red) image. The F814W image is chosen as the reference because it is the longest wavelength and may have a slightly different point spread function.

Python

from astropy.io import fits from astropy.wcs import WCS from reproject import reproject_interp

aligned_images = {}

try: \# Define the target frame using the red filter image target_filename = file_paths\['red'\] with fits.open(target_filename) as hdul: target_wcs = WCS(hdul.header) target_shape = hdul.shape

```         
aligned_images['red'] = fits.getdata(target_filename, ext=('SCI', 1))
print(f"Reference frame set from {os.path.basename(target_filename)}.")

# Reproject the other images
for color in ['green', 'blue']:
    filename = file_paths[color]
    print(f"Reprojecting {color} channel ({os.path.basename(filename)})...")
    with fits.open(filename) as hdul:
        source_data = hdul.data
        source_wcs = WCS(hdul.header)
    
    reprojected_data, _ = reproject_interp(
        (source_data, source_wcs),
        target_wcs,
        shape_out=target_shape
    )
    aligned_images[color] = reprojected_data

print("Reprojection complete.")
```

except (NameError, KeyError) as e: print(f"Reprojection failed. Ensure file paths are correct. Error: {e}")

Step 4: Color Compositing and Display Combine the three aligned, single-band images into a color composite using make_lupton_rgb. The parameters stretch and Q are chosen through experimentation to best reveal the galaxy's structure.

Python

import matplotlib.pyplot as plt from astropy.visualization import make_lupton_rgb import numpy as np

try: \# Retrieve the aligned NumPy arrays \# Chromatic ordering: Longest wavelength (F814W) to Red, shortest (F435W) to Blue image_r = aligned_images\['red'\] image_g = aligned_images\['green'\] image_b = aligned_images\['blue'\]

```         
# Handle potential NaN values from reprojection by setting them to 0
image_r = np.nan_to_num(image_r)
image_g = np.nan_to_num(image_g)
image_b = np.nan_to_num(image_b)

print("Creating Lupton RGB composite image...")
# These parameters are a good starting point for HST galaxy images.
# stretch: Controls contrast. Lower is higher contrast.
# Q: Controls the asinh softening. Higher is more linear.
# minimum: Sets the black level. Can be a single value or one per channel.
rgb_image = make_lupton_rgb(image_r, image_g, image_b,
                            stretch=0.2,
                            Q=10,
                            minimum=0.01) # A small minimum to set the background black

# Display the final image
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(rgb_image, origin='lower')
ax.set_title("M51 (Whirlpool Galaxy) - HST/ACS", fontsize=16)
ax.axis('off') # Hide axes for a cleaner look
plt.tight_layout()
plt.show()
```

except NameError: print("Error: Aligned images not available for compositing.")

9.  Case Study 2: A Supernova Remnant with Chandra & JWST

Objective: To create a multi-wavelength composite image combining X-ray data from Chandra and infrared data from JWST. This workflow highlights how to handle fundamentally different data types (event lists vs. images) and map them to color channels to reveal distinct physical processes: the hot, shocked gas seen in X-rays and the cooler dust and gas seen in the infrared. Target: Cassiopeia A (Cas A) Data Sources: Chandra ACIS, JWST MIRI Bands: Chandra Soft (0.5-1.2 keV), Medium (1.2-2.0 keV), Hard (2.0-7.0 keV); JWST F770W (7.7 µm)

Workflow and Code

Step 1: Data Acquisition Download a Chandra evt2.fits file for Cas A and a corresponding JWST MIRI \_i2d.fits (Stage 3 processed) file.

Python

# (Code for astroquery download would be here. For brevity, we assume files are local.)

chandra_file = 'chandra_cas_a_evt2.fits' \# Replace with actual file path jwst_file = 'jwst_miri_cas_a_i2d.fits' \# Replace with actual file path

Step 2: Chandra Image Creation Process the Chandra event list into three separate images corresponding to the soft, medium, and hard X-ray bands. This follows the procedure detailed in Section 4.

Python

import numpy as np from astropy.io import fits

chandra_images = {} try: with fits.open(chandra_file) as hdul: events = hdul.data header = hdul.header

```         
    # Define image grid from WCS keywords
    image_size = (header, header) # Assuming TDET axes
    
    energy_bands = {'soft': (500, 1200), 'medium': (1200, 2000), 'hard': (2000, 7000)}
    
    for band, (emin, emax) in energy_bands.items():
        energy_mask = (events['energy'] >= emin) & (events['energy'] < emax)
        filtered_events = events[energy_mask]
        
        # Use sky coordinates for binning
        img, _, _ = np.histogram2d(
            x=filtered_events['x'], y=filtered_events['y'],
            bins=(4096, 4096), # Example binning, adjust for desired resolution
            range=[[0.5, 4096.5], [0.5, 4096.5]]
        )
        chandra_images[band] = img.T
print("Chandra X-ray images created for Soft, Medium, and Hard bands.")
```

except (FileNotFoundError, KeyError) as e: print(f"Failed to process Chandra event file: {e}")

Step 3: Reprojection The JWST image will serve as the astrometric reference. Reproject the three Chandra images to align with the JWST WCS and pixel grid.

Python

from astropy.wcs import WCS from reproject import reproject_interp

aligned_multiwavelength = {} try: \# Load JWST data and define it as the target frame with fits.open(jwst_file) as hdul: jwst_hdu = hdul target_wcs = WCS(jwst_hdu.header) target_shape = jwst_hdu.shape aligned_multiwavelength\['infrared'\] = jwst_hdu.data

```         
print(f"JWST MIRI image loaded as reference frame.")

# Create a WCS for the Chandra images from the event file header
chandra_wcs = WCS(fits.getheader(chandra_file, ext='EVENTS'))

# Reproject each Chandra band
for band, data in chandra_images.items():
    print(f"Reprojecting Chandra '{band}' band...")
    reprojected_data, _ = reproject_interp(
        (data, chandra_wcs),
        target_wcs,
        shape_out=target_shape
    )
    aligned_multiwavelength[band] = reprojected_data

print("All images reprojected to JWST frame.")
```

except Exception as e: print(f"Reprojection failed: {e}")

Step 4: Color Compositing and Display This is a guided, creative step. A common and effective mapping for supernova remnants is to assign different physical components to different colors. Here, we will map the JWST infrared data (tracing dust) to Red, the hard X-rays (tracing the fastest, hottest material) to Blue, and a combination of soft/medium X-rays to Green.

Python

import matplotlib.pyplot as plt from astropy.visualization import make_lupton_rgb, AsinhStretch, ManualInterval import numpy as np

try: \# Assign channels based on physical properties \# Red: Dust and molecular gas (JWST) \# Green: Intermediate temperature gas (Chandra Soft+Medium) \# Blue: Hottest, highest-energy ejecta (Chandra Hard)

```         
image_r = aligned_multiwavelength['infrared']
image_g = aligned_multiwavelength['soft'] + aligned_multiwavelength['medium']
image_b = aligned_multiwavelength['hard']

# Normalize each channel individually before combining to balance their intensities
# This is a case where independent scaling is useful before the final composition.
# We use percentile intervals to handle the different dynamic ranges.
image_r = ManualInterval(vmin=np.percentile(image_r, 5), vmax=np.percentile(image_r, 99.8))(image_r)
image_g = ManualInterval(vmin=np.percentile(image_g, 10), vmax=np.percentile(image_g, 99.9))(image_g)
image_b = ManualInterval(vmin=np.percentile(image_b, 10), vmax=np.percentile(image_b, 99.9))(image_b)

# Use make_rgb with an AsinhStretch for a vibrant final image
# We use make_rgb here because we've already manually balanced the channels.
from astropy.visualization import make_rgb
rgb_image = make_rgb(image_r, image_g, image_b, stretch=AsinhStretch(a=0.05))

# Display the final image
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(rgb_image, origin='lower')
ax.set_title("Cassiopeia A: JWST (Red) & Chandra (Green/Blue)", fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.show()
```

except NameError: print("Error: Aligned images not available for compositing.")

10. Case Study 3: A Wide-Field Mosaic with Pan-STARRS

Objective: To demonstrate the processing of wide-field survey data. This workflow involves querying for multiple single-epoch "warp" images and combining them to create a color view of a large region of the sky, paying special attention to handling the sky background. Target: A region in the Draco constellation Data Source: Pan-STARRS1 Filters: g, r, i

Workflow and Code

Step 1: Data Acquisition Query the Pan-STARRS archive at MAST for warp images in the g, r, and i filters for a specific sky coordinate.

Python

from astroquery.mast import Observations from astropy.coordinates import SkyCoord import os

data_dir = "panstarrs_data" os.makedirs(data_dir, exist_ok=True)

print("Querying MAST for Pan-STARRS1 warp images...") coords = SkyCoord.from_name("NGC 5907") \# A galaxy in Draco

obs_table = Observations.query_criteria( obs_collection="PanSTARRS", object_name="NGC 5907", dataproduct_type="image", radius="0.2 deg" )

# Filter for warp images in the desired filters

filters_to_download = \['g', 'r', 'i'\] products = Observations.get_product_list(obs_table) filtered_products = Observations.filter_products(products, productSubGroupDescription=\['g.unconv.warp', 'r.unconv.warp', 'i.unconv.warp'\], mrp_only=False)

# Download one image for each filter for this example

download_list = for f in filters_to_download: \# Find the first product that matches the filter for row in filtered_products: if f'unconv.warp.{f}' in row\['description'\]: download_list.append(row) break

if download_list: print(f"Downloading {len(download_list)} Pan-STARRS warp files...") manifest = Observations.download_products(download_list, download_dir=data_dir) print("Download complete.") print(manifest) else: print("No matching warp images found.")

Step 2: Reprojection Even though warp images are projected onto a sky grid, different observations may have slightly different centers or orientations. Reprojecting to a common frame is still the most robust approach. We will use the i-band image as the reference.

Python

# (Code for identifying file paths and reprojection would be here,

# similar to the HST case study. For brevity, we assume this is done

# and the results are in a dictionary 'aligned_panstarrs'.)

Step 3: Color Compositing with Background Subtraction Wide-field images have a significant sky background level that must be handled correctly to produce a clean image. A simple but effective method is to estimate the background level (e.g., using the median of the image) and subtract it before compositing.

Python

import numpy as np import matplotlib.pyplot as plt from astropy.visualization import make_lupton_rgb from astropy.stats import sigma_clipped_stats

# Assume 'aligned_panstarrs' dictionary exists with 'g', 'r', 'i' images

try: \# Apply chromatic ordering: i -\> Red, r -\> Green, g -\> Blue image_i = aligned_panstarrs\['i'\] image_r = aligned_panstarrs\['r'\] image_g = aligned_panstarrs\['g'\]

```         
images = {'r': image_i, 'g': image_r, 'b': image_g}

# Background subtraction and scaling
final_images = {}
for channel, data in images.items():
    # Estimate sky background using sigma-clipped median
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    # Subtract the background
    subtracted_data = data - median
    final_images[channel] = subtracted_data

print("Background subtracted from all channels.")

# Create the Lupton RGB composite from the background-subtracted images
rgb_image = make_lupton_rgb(final_images['r'], final_images['g'], final_images['b'],
                            stretch=0.5,
                            Q=8,
                            minimum=0) # Black point is now 0 after subtraction

# Display the final image
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(rgb_image, origin='lower')
ax.set_title("Pan-STARRS1 Wide-Field Image", fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.show()
```

except NameError: print("Error: Aligned Pan-STARRS images not available.")

These case studies provide templates that can be adapted to a wide variety of astronomical targets and datasets. They illustrate the power and flexibility of the Python ecosystem, demonstrating a complete, reproducible, and scientifically grounded workflow for transforming raw FITS data into professional-quality color images. Table 4: Example Filter-to-Color Mapping Schemes Science Goal Wavelength Regime Filter → Red Filter → Green Filter → Blue Rationale "Natural" Optical Color Optical HST: F814W (I-band) HST: F555W (V-band) HST: F435W (B-band) Maps the standard astronomical broadband filters to their corresponding RGB channels, approximating how the human eye would perceive colors if it were sensitive enough. "Infrared Color" Near-Infrared JWST: F444W (4.4 µm) JWST: F200W (2.0 µm) JWST: F090W (0.9 µm) Applies chromatic ordering to infrared wavelengths. Redder colors trace cooler dust and gas, while bluer colors trace hotter stellar populations. X-ray Energy Bands X-ray Chandra: Hard (2-7 keV) Chandra: Medium (1.2-2 keV) Chandra: Soft (0.5-1.2 keV) Maps X-ray energy to color. Red indicates the highest-energy, most violent processes, while blue traces cooler, softer X-ray emission. Narrowband Emission Optical H-alpha (656 nm) OIII (501 nm) H-beta (486 nm) The "Hubble Palette" (often with SII mapped to Red). Assigns specific colors to emission from different elements, highlighting the physical structure and ionization state of nebulae.

Conclusions

This guide has provided a comprehensive, code-centric framework for creating professional-quality astronomical images from FITS files using the Python scientific ecosystem. The journey from raw data to final image is a multi-stage process that requires both technical precision and an understanding of the underlying scientific context. The foundation of this process is a thorough understanding of the FITS file format and the Astropy library, which provides the essential tools for data I/O, manipulation, and visualization. The Python astro-imaging ecosystem, including Astroquery, reproject, and ccdproc, forms an interoperable toolkit built upon this foundation, enabling powerful and reproducible workflows. A critical takeaway is that data acquisition and interpretation are mission-specific. The APIs for MAST, the Euclid Science Archive, and the Chandra Data Archive each reflect the unique structure and scientific goals of their respective missions. Similarly, the FITS file structures for JWST, HST, Euclid, Chandra, GALEX, and Pan-STARRS all have distinct conventions that must be handled programmatically. A particularly important distinction is the event-list nature of Chandra data, which requires an explicit binning step to create an image, a process fundamentally different from handling the pre-imaged data from other observatories. Image preparation, specifically reprojection, is a scientifically significant step that ensures spatial coherence between different datasets. The reproject library provides the necessary tools, but users must be cognizant that this process involves interpolation and can affect data integrity. Finally, the creation of a color composite is a blend of art and science. The high dynamic range of astronomical data necessitates the use of non-linear scaling and stretching functions, with the AsinhStretch being particularly effective. The Lupton algorithm, implemented in astropy.visualization.make_lupton_rgb, stands out as the superior method for creating broadband composites, as it is specifically designed to preserve color information in bright regions where simpler methods fail. The choice of filter-to-color mapping, guided by the principle of chromatic ordering, is what transforms multiple grayscale images into a scientifically meaningful color representation. By following the principles and code examples detailed in this guide—from programmatic data acquisition and mission-specific FITS parsing to robust image alignment and scientifically motivated color compositing—researchers and advanced amateurs can leverage the full power of the Python ecosystem to transform raw observational data into insightful and visually compelling astronomical images. Works cited 1. 2.2 FITS File Format - STScI, accessed October 23, 2025, https://www.stsci.edu/hst/wfpc2/Wfpc2_dhb/intro_ch23.html 2. Flexible Image Transport System (FITS), Version 3.0 - The Library of Congress, accessed October 23, 2025, https://www.loc.gov/preservation/digital/formats/fdd/fdd000317.shtml 3. Photo Album :: Open FITS & 3-Color Composite Images - Chandra X-ray Observatory, accessed October 23, 2025, https://chandra.si.edu/photo/openFITS/overview.html 4. Overview of the FITS Data Format - HEASARC, accessed October 23, 2025, https://heasarc.gsfc.nasa.gov/docs/heasarc/fits_overview.html 5. FITS Format, accessed October 23, 2025, https://www.astro.sunysb.edu/fwalter/AST443/fits.html 6. 3.2 FITS File Format - HST User Documentation - HDox - STScI, accessed October 23, 2025, https://hst-docs.stsci.edu/hstdhb/3-hst-file-formats/3-2-fits-file-format 7. FITS File Handling (astropy.io.fits), accessed October 23, 2025, https://docs.astropy.org/en/stable/io/fits/index.html 8. jwst-docs.stsci.edu, accessed October 23, 2025, https://jwst-docs.stsci.edu/accessing-jwst-data/jwst-science-data-overview#:\~:text=FITS%20format%20files%20consist%20of,may%20immediately%20follow%20the%20header. 9. hst-docs.stsci.edu, accessed October 23, 2025, https://hst-docs.stsci.edu/hstdhb/3-hst-file-formats/3-2-fits-file-format#:\~:text=A%20file%20in%20FITS%20format,may%20immediately%20follow%20the%20header. 10. Structure of FITS files, accessed October 23, 2025, https://www.eso.org/sci/software/esomidas/doc/user/18NOV/vola/node111.html 11. JWST FITS Keyword Dictionary - aspbooks.org, accessed October 23, 2025, https://www.aspbooks.org/publications/522/165.pdf 12. Handling FITS files — Astropy4MPIK 1.0 documentation - MPIK Astropy Workshop, accessed October 23, 2025, https://astropy4mpik.readthedocs.io/en/latest/fits.html 13. User Guide — Astropy v7.1.1, accessed October 23, 2025, https://docs.astropy.org/en/stable/index_user_docs.html 14. 4.4 Working with FITS Data in Python - HST User Documentation - HDox, accessed October 23, 2025, https://hst-docs.stsci.edu/hstdhb/4-hst-data-analysis/4-4-working-with-fits-data-in-python 15. Working with FITS Data – Programming for Astronomy and Astrophysics 2 - GitHub Pages, accessed October 23, 2025, https://philuttley.github.io/prog4aa_lesson2/10-fitsfiles/index.html 16. 1.8. Reading images — CCD Data Reduction Guide - Astropy, accessed October 23, 2025, https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-11-reading-images.html 17. FITS — Astropy v7.2.dev691+g2309a502d, accessed October 23, 2025, https://docs.astropy.org/en/latest/io/unified_table_fits.html 18. make_lupton_rgb — Astropy v7.1.1, accessed October 23, 2025, https://docs.astropy.org/en/stable/api/astropy.visualization.make_lupton_rgb.html 19. Creating color RGB images — Astropy v7.1.1, accessed October 23, 2025, https://docs.astropy.org/en/stable/visualization/rgb.html 20. Image stretching and normalization — Astropy v7.1.1, accessed October 23, 2025, https://docs.astropy.org/en/stable/visualization/normalization.html 21. Data Visualization (astropy.visualization) - Read the Docs, accessed October 23, 2025, https://astro-docs.readthedocs.io/en/latest/visualization/index.html 22. Astroquery — MAST Notebook Repository, accessed October 23, 2025, https://spacetelescope.github.io/mast_notebooks/notebooks/multi_mission/astroquery.html 23. MAST Queries (astroquery.mast) — astroquery v0.4.12.dev206, accessed October 23, 2025, https://astroquery.readthedocs.io/en/latest/mast/mast.html 24. MAST Queries — astroquery v0.4.11 - Read the Docs, accessed October 23, 2025, https://astroquery.readthedocs.io/en/stable/mast/mast_mastquery.html 25. ESA EUCLID Archive (astroquery.esa.euclid) — astroquery v0.4.12 ..., accessed October 23, 2025, https://astroquery.readthedocs.io/en/latest/esa/euclid/euclid.html 26. reproject 0.9 - Python Simple Repository Browser, accessed October 23, 2025, https://simple-repository.app.cern.ch/project/reproject/0.9 27. reproject - PyPI, accessed October 23, 2025, https://pypi.org/project/reproject/ 28. Cube Reprojection Tutorial - Learn Astropy, accessed October 23, 2025, https://learn.astropy.org/tutorials/SpectralCubeReprojectExample.html 29. Reproject.jl - JuliaAstro, accessed October 23, 2025, https://juliaastro.org/Reproject/stable/ 30. ccdproc — ccdproc v2.5.2.dev61+gbfc1e082b, accessed October 23, 2025, https://ccdproc.readthedocs.io/ 31. Ccdproc - Anaconda.org, accessed October 23, 2025, https://anaconda.org/astropy/ccdproc 32. astropy/ccdproc: Astropy affiliated package for reducing optical/IR CCD data - GitHub, accessed October 23, 2025, https://github.com/astropy/ccdproc 33. A User's Guide to CCD Reductions with IRAF - NOIRLab, accessed October 23, 2025, https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc0478.pdf 34. 5.1. Two science image calibration examples — CCD Data Reduction Guide - Astropy, accessed October 23, 2025, https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/06-03-science-images-calibration-examples.html 35. Handling overscan and bias for dark frames, accessed October 23, 2025, https://mwcraig.github.io/ccd-as-book/03-04-Handling-overscan-and-bias-for-dark-frames.html 36. Reduction toolbox — ccdproc v1.2.0 - Read the Docs, accessed October 23, 2025, https://ccdproc.readthedocs.io/en/v1.2.0/ccdproc/reduction_toolbox.html 37. 4.2. Calibrating the flats — CCD Data Reduction Guide - Astropy, accessed October 23, 2025, https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-03-Calibrating-the-flats.html 38. ASTR469: Project #2, Measuring X-ray Sources - Sarah Spolaor, accessed October 23, 2025, https://sarahspolaor.faculty.wvu.edu/files/d/c5375159-33b3-401a-9664-d21ce3302d9a/project_2.pdf 39. ObservationsClass — astroquery v0.4.12.dev206 - Read the Docs, accessed October 23, 2025, https://astroquery.readthedocs.io/en/latest/api/astroquery.mast.ObservationsClass.html 40. Reading and writing files in Python - Chandra X-ray Center, accessed October 23, 2025, https://cxc.cfa.harvard.edu/ciao/scripting/io.html 41. Introduction and Scripts Jonathan McDowell Chandra X-ray Center, SAO, accessed October 23, 2025, https://planet4589.org/talks/soft/2014/ws2.pdf 42. Science products — jwst 1.21.0.dev27+gbbadc65b6 documentation, accessed October 23, 2025, https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/science_products.html 43. JWST Science Data Overview, accessed October 23, 2025, https://jwst-docs.stsci.edu/accessing-jwst-data/jwst-science-data-overview 44. Euclid Archive at IRSA User Guide - NASA/IPAC Infrared Science ..., accessed October 23, 2025, https://irsa.ipac.caltech.edu/data/Euclid/docs/euclid_archive_at_irsa_user_guide.pdf 45. Euclid Help - NASA/IPAC Infrared Science Archive, accessed October 23, 2025, https://irsa.ipac.caltech.edu/onlinehelp/euclid/help.pdf?id=e3b78b3 46. Chandra Tutorial, accessed October 23, 2025, http://labx.iasfbo.inaf.it/2017/resources/chandra_tutorial2017.pdf 47. Ahelp: energy_hue_map - CIAO 4.17 - Chandra X-ray Center, accessed October 23, 2025, https://cxc.cfa.harvard.edu/ciao/ahelp/energy_hue_map.html 48. MAST.Galex.DDFAQ, accessed October 23, 2025, https://galex.stsci.edu/gr6/?page=ddfaq 49. \[1612.05243\] The Pan-STARRS1 Database and Data Products - ar5iv, accessed October 23, 2025, https://ar5iv.labs.arxiv.org/html/1612.05243 50. Pan-STARRS Pixel Processing: Detrending, Warping, Stacking \| Request PDF, accessed October 23, 2025, https://www.researchgate.net/publication/346561363_Pan-STARRS_Pixel_Processing_Detrending_Warping_Stacking 51. \[1612.05245\] Pan-STARRS Pixel Processing: Detrending, Warping, Stacking - arXiv, accessed October 23, 2025, https://arxiv.org/abs/1612.05245 52. The Pan-STARRS1 Data Archive \| STScI, accessed October 23, 2025, https://www.stsci.edu/contents/newsletters/2019-volume-36-issue-01/the-pan-starrs1-data-archive 53. MAST PanSTARRS - Mikulski Archive for Space Telescopes, accessed October 23, 2025, https://archive.stsci.edu/panstarrs/ 54. Pan-STARRS reference catalog in LSST format - DM Notifications, accessed October 23, 2025, https://community.lsst.org/t/pan-starrs-reference-catalog-in-lsst-format/1572 55. The Pan-STARRS1 Database and Data Products - mpe.mpg.de, accessed October 23, 2025, https://www.mpe.mpg.de/\~saglia/journals_pdf/flewelling2020.pdf 56. Reprojection — rasterio 1.5.0.dev documentation - Read the Docs, accessed October 23, 2025, https://rasterio.readthedocs.io/en/latest/topics/reproject.html 57. Aligning AIA and HMI Data with Reproject - SunPy, accessed October 23, 2025, https://docs.sunpy.org/en/stable/generated/gallery/map_transformations/reprojection_align_aia_hmi.html 58. reproject_exact — reproject v0.18.0 - Read the Docs, accessed October 23, 2025, https://reproject.readthedocs.io/en/stable/api/reproject.reproject_exact.html 59. Reproject Raster Data Python - Earth Data Science, accessed October 23, 2025, https://earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/ 60. How Are Webb's Full-Color Images Made? - NASA Science, accessed October 23, 2025, https://science.nasa.gov/mission/webb/science-overview/science-explainers/how-are-webbs-full-color-images-made/ 61. Source code for astropy.visualization.stretch, accessed October 23, 2025, https://docs.astropy.org/en/stable/\_modules/astropy/visualization/stretch.html 62. LogStretch — Astropy v7.1.1, accessed October 23, 2025, https://docs.astropy.org/en/stable/api/astropy.visualization.LogStretch.html 63. Preparing Red-Green-Blue (RGB) Images from CCD Data - Department of Astrophysical Sciences, accessed October 23, 2025, https://www.astro.princeton.edu/\~rhl/Papers/truecolor.pdf 64. Maybe asked already, but how are they coloring the JW pictures and how do they know what colors are accurate? : r/jameswebb - Reddit, accessed October 23, 2025, https://www.reddit.com/r/jameswebb/comments/w0kcpj/maybe_asked_already_but_how_are_they_coloring_the/ 65. What are the true colors of images from the James Webb Space Telescope? - Reddit, accessed October 23, 2025, https://www.reddit.com/r/space/comments/1bopcms/what_are_the_true_colors_of_images_from_the_james/ 66. Bringing JWST images to life \| Physics Today - AIP Publishing, accessed October 23, 2025, https://pubs.aip.org/physicstoday/online/42481/Bringing-JWST-images-to-life 67. Source code for astropy.visualization.lupton_rgb, accessed October 23, 2025, https://docs.astropy.org/en/stable/\_modules/astropy/visualization/lupton_rgb.html 68. make_rgb — Astropy v7.1.1, accessed October 23, 2025, https://docs.astropy.org/en/stable/api/astropy.visualization.make_rgb.html