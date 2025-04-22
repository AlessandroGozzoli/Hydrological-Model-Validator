from netCDF4 import Dataset as ds
import numpy as np

def mask_reader():
    # -----MODEL LAND SEA MASK-----
    MASK = 'C:/Tesi Magistrale/Dati/mesh_mask.nc'

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