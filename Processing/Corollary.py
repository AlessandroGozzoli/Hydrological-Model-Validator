import numpy as np
from netCDF4 import Dataset as ds
from pathlib import Path

# Function to check if a year is a leap year
def leapyear(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 1
    return 0

# Main function to calculate the true time series length in days
def true_time_series_length(nf, chlfstart, chlfend, DinY):
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

def mask_reader(BaseDIR):
    # -----MODEL LAND SEA MASK-----
    MASK = Path(BaseDIR, 'mesh_mask.nc')

    # Open the NetCDF file
    with ds(MASK, 'r') as ncfile:
        # Read the 3D mask and remove the degenerate dimension
        mask3d = ncfile.variables['tmask'][:].squeeze()
        
        # -----ELIMINATE DEGENERATE DIMENSION-----
        Mmask = mask3d[0, :, :]  # Extract first layer

        # -----FIND LAND GRIDPOINTS INDEXES FROM MODEL MASK-----
        Mfsm = np.where(Mmask == 0)  # Land points (2D)
        Mfsm_3d = np.where(mask3d == 0)  # Land points (3D)

        # -----GET MODEL LAT & LON-----
        Mlat = ncfile.variables['nav_lat'][:]
        Mlon = ncfile.variables['nav_lon'][:]

    return Mmask, Mfsm, Mfsm_3d, Mlat, Mlon