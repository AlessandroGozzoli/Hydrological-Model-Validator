import numpy as np
from netCDF4 import Dataset as ds
from pathlib import Path

from Costants import days_in_months_non_leap, days_in_months_leap, ysec

# Function to check if a year is a leap year
def leapyear(year):
    assert isinstance(year, int) and year > 0, "Year must be a positive integer"
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 1
    return 0


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

# Function to convert yearly data to monthly data
def convert_to_monthly_data(yearly_data):

    # Initialize an empty dictionary to hold the monthly data
    monthly_data_dict = {}

    # Loop over each year's data
    for i, year_data in enumerate(yearly_data):
        year = ysec[i]
        
        # Determine the number of days in the year
        if leapyear(year):
            days_in_months = days_in_months_leap
            expected_days = 366
        else:
            days_in_months = days_in_months_non_leap
            expected_days = 365
        
        # Pad shorter years with NaNs if necessary
        if len(year_data) < expected_days:
            year_data = np.pad(year_data, (0, expected_days - len(year_data)), constant_values=np.nan)
        
        # Initialize a list to store the months for this year
        year_months = []
        start_idx = 0  # Start at the beginning of the year
        
        # For each month, slice the data
        for month_days in days_in_months:
            end_idx = start_idx + month_days
            month_data = year_data[start_idx:end_idx]
            year_months.append(month_data)
            start_idx = end_idx  # Move to the next month
        
        # Save the monthly data for the current year in the dictionary
        monthly_data_dict[year] = year_months

    # Return the dictionary containing monthly data for each year
    return monthly_data_dict