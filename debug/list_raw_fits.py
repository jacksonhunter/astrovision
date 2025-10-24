"""List FITS files in the raw_fits directory."""

from pathlib import Path

raw_fits_dir = Path(r"C:\Users\jacks\experiments\PycharmProjects\Mission Control\examples\output\raw_fits")

print(f"FITS files in: {raw_fits_dir}\n")

if not raw_fits_dir.exists():
    print(f"ERROR: Directory not found!")
else:
    fits_files = sorted(raw_fits_dir.glob("*.fits"))

    if not fits_files:
        print("No FITS files found!")
    else:
        # Group by filter
        filters = {}
        for f in fits_files:
            # Extract filter from filename (e.g., .stk.i. or .stk.g.)
            parts = f.name.split('.')
            if 'stk' in parts:
                idx = parts.index('stk')
                if idx + 1 < len(parts):
                    filter_name = parts[idx + 1]
                    if filter_name not in filters:
                        filters[filter_name] = []
                    filters[filter_name].append(f.name)

        print(f"Found {len(fits_files)} FITS files\n")

        print("Grouped by filter:")
        for filter_name in sorted(filters.keys()):
            print(f"\n  {filter_name}-band ({len(filters[filter_name])} files):")
            for fname in filters[filter_name][:5]:  # Show first 5
                print(f"    {fname}")
            if len(filters[filter_name]) > 5:
                print(f"    ... and {len(filters[filter_name]) - 5} more")