import numpy as np
from netCDF4 import Dataset as ds
from pathlib import Path
import re
import xarray as xr

# Function to check if a year is a leap year
def leapyear(year):
    assert isinstance(year, int) and year > 0, "Year must be a positive integer"
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 1
    return 0

def extract_mod_sat_keys(taylor_dict):
    mod_key = next((k for k in taylor_dict if 'mod' in k.lower()), None)
    sat_key = next((k for k in taylor_dict if 'sat' in k.lower()), None)
    if mod_key is None or sat_key is None:
        raise ValueError("taylor_dict must contain keys with 'mod' and 'sat' data.")
    return mod_key, sat_key

def get_common_series_by_year(data_dict):
    """
    Extract and align model and satellite data by year from taylor_dict.

    Returns:
        List of tuples: (year, mod_values, sat_values)
    """
    common_series = []
    
    mod_key, sat_key = extract_mod_sat_keys(data_dict)

    for year in sorted(data_dict[mod_key].keys()):
        mod_series = data_dict[mod_key][year].dropna()
        sat_series = data_dict[sat_key][year].dropna()

        combined = mod_series.to_frame('mod').join(sat_series.to_frame('sat'), how='inner').dropna()
        if combined.empty:
            print(f"Warning: No overlapping data for year {year}. Skipping.")
            continue

        common_series.append((str(year), combined['mod'].values, combined['sat'].values))

    return common_series

def get_common_series_by_year_month(data_dict):
    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    result = []  # list of (year, month, mod_values, sat_values)

    years = sorted(data_dict[mod_key].keys())
    for year in years:
        for month_index in range(12):
            try:
                mod_vals = np.asarray(data_dict[mod_key][year][month_index])
                sat_vals = np.asarray(data_dict[sat_key][year][month_index])
            except (IndexError, KeyError):
                continue

            valid = ~np.isnan(mod_vals) & ~np.isnan(sat_vals)
            if not np.any(valid):
                continue

            result.append((year, month_index, mod_vals[valid], sat_vals[valid]))

    return result

def get_valid_mask(mod_vals, sat_vals):
    """Return boolean mask where both mod and sat values are not NaN."""
    return ~np.isnan(mod_vals) & ~np.isnan(sat_vals)

# Main function to calculate the true time series length in days
def true_time_series_length(nf, chlfstart, chlfend, DinY):
    # Input validations
    assert isinstance(nf, int) and nf > 0, "nf must be a positive integer"
    assert isinstance(chlfstart, list) and isinstance(chlfend, list), "chlfstart and chlfend must be lists"
    assert len(chlfstart) == nf and len(chlfend) == nf, "chlfstart and chlfend must have length equal to nf"
    assert all(isinstance(x, int) for x in chlfstart + chlfend), "chlfstart and chlfend must contain integers"
    assert all(end >= start for start, end in zip(chlfstart, chlfend)), "Each chlfend must be greater than or equal to corresponding chlfstart"
    assert isinstance(DinY, int) and DinY in [365], "DinY must be 365 (base number of days in a non-leap year)"

    Truedays = 0  # Initialize the total number of days
    fdays = [0] * nf  # List to store the number of days for each file
    nspan = [0] * nf  # List to store the span of years for each file

    for n in range(nf):
        # Define the time span (in years) for each file
        nspan[n] = chlfend[n] - chlfstart[n] + 1
        fdays[n] = 0

        # Define the "true" number of days in each file
        for y in range(chlfstart[n], chlfend[n] + 1):
            # If year "y" is a leap year, one day is added
            fdays[n] += DinY + leapyear(y)

        Truedays += fdays[n]

    return Truedays

# To read the mask
def mask_reader(BaseDIR):
    # Input validation
    assert isinstance(BaseDIR, (str, Path)), "BaseDIR must be a string or Path object"

    # -----MODEL LAND SEA MASK-----
    MASK = Path(BaseDIR, 'mesh_mask.nc')
    assert MASK.exists(), f"Mask file not found at {MASK}"

    # Open the NetCDF file
    with ds(MASK, 'r') as ncfile:
        # Check required variables exist
        required_vars = ['tmask', 'nav_lat', 'nav_lon']
        for var in required_vars:
            assert var in ncfile.variables, f"'{var}' not found in {MASK.name}"

        # Read the 3D mask and remove the degenerate dimension
        mask3d = ncfile.variables['tmask'][:]
        assert mask3d.ndim == 4, f"Expected 'tmask' to be 4D (time, depth, lat, lon), got shape {mask3d.shape}"

        mask3d = mask3d.squeeze()  # Remove degenerate dimensions
        assert mask3d.ndim == 3, f"'tmask' should be 3D after squeeze, got shape {mask3d.shape}"

        # -----ELIMINATE DEGENERATE DIMENSION-----
        Mmask = mask3d[0, :, :]  # Extract first layer
        assert Mmask.ndim == 2, f"Expected Mmask to be 2D, got shape {Mmask.shape}"

        # -----FIND LAND GRIDPOINTS INDEXES FROM MODEL MASK-----
        Mfsm = np.where(Mmask == 0)      # Land points (2D)
        Mfsm_3d = np.where(mask3d == 0)  # Land points (3D)

        # -----GET MODEL LAT & LON-----
        Mlat = ncfile.variables['nav_lat'][:]
        Mlon = ncfile.variables['nav_lon'][:]
        assert Mlat.shape == Mmask.shape, f"Mlat shape {Mlat.shape} does not match Mmask shape {Mmask.shape}"
        assert Mlon.shape == Mmask.shape, f"Mlon shape {Mlon.shape} does not match Mmask shape {Mmask.shape}"

    return Mmask, Mfsm, Mfsm_3d, Mlat, Mlon


def split_to_monthly(yearly_data):
    """Converts yearly data into monthly data."""
    monthly_data_dict = {}
    
    # Loop over each year
    for year, year_data in yearly_data.items():
        year_months = []
        
        # Loop over each month
        for month in range(1, 13):  # 1 to 12 for each month
            # Extract data for the current month
            month_data = year_data[year_data.index.month == month]
            year_months.append(month_data)
        
        # Store the monthly data for the current year
        monthly_data_dict[year] = year_months
    
    return monthly_data_dict

def split_to_yearly(series, unique_years):
    """Splits a pandas Series into a dictionary by year based on the datetime index."""
    yearly_data = {}
    for year in unique_years:
        # Extract data for the current year using the datetime index
        year_data = series[series.index.year == year]
        yearly_data[year] = year_data
    return yearly_data

def format_unit(unit):
    # First, handle chemical subscripts in numerator and denominator (e.g., O2 -> O_2)
    def add_subscripts(s):
        return re.sub(r'([A-Za-z]{1,2})(\d+)', r'\1_{\2}', s)

    # Then handle exponents (e.g., m3 -> m^3) in denominator
    def handle_exponents(s):
        return re.sub(r'([a-zA-Z])(\d+)', r'\1^{\2}', s)

    if '/' in unit:
        numerator, denominator = unit.split('/')
        numerator = add_subscripts(numerator.strip())
        denominator = handle_exponents(denominator.strip())
        return f'$\\frac{{{numerator}}}{{{denominator}}}$'
    else:
        unit = add_subscripts(unit)
        unit = handle_exponents(unit)
        return f'${unit}$'

def load_dataset(year, IDIR):
    file_path = Path(IDIR) / f"Msst_{year}.nc"
    if file_path.exists():
        print(f"Opening {file_path.name}...")
        return year, xr.open_dataset(file_path)
    else:
        print(f"Warning: {file_path.name} not found!")
        return year, None
    
def round_up_to_nearest(x, base=1.0):
        return base * np.ceil(x / base)