import numpy as np
from scipy.interpolate import RegularGridInterpolator
from Flood import flood
from Data_setupper import chldlev, Truedays, Schl_complete, Slat, Slon, satnan
from Mask_reader import mask_reader
from Model_data_reader import Mchl_complete

# Load mask data
Mmask, Mfsm, Mfsm_3d, Mlat, Mlon = mask_reader()

def interpolate_chl_data(Truedays, chldlev, Schl_complete, Slon, Slat, Mlon, Mlat, Mmask, Mchl_complete, Mfsm):
    M_TS_ave = np.full(Truedays, np.nan)
    S_TS_ave = np.full(Truedays, np.nan)
    
    if chldlev == "l4":
        print("Start interpolation loop")
    
        satmask = ~np.isnan(Schl_complete[0, :, :])
        Slat = Slat.T
    
        Schl_interp = np.full_like(Schl_complete, np.nan, dtype=float)
    
        for d in range(Truedays):
            noflood = Schl_complete[d, :, :]
            flooded = flood(noflood, 5)
        
            # Create the interpolator
            interpolator = RegularGridInterpolator((Slon[:, 0], Slat[0, :]), flooded, method='linear', bounds_error=False, fill_value=np.nan)
        
            # Define the model grid for interpolation
            Mlon_mesh, Mlat_mesh = np.meshgrid(Mlon, Mlat, indexing='ij')
            interp_points = np.column_stack([Mlat_mesh.ravel(), Mlon_mesh.ravel()])
        
            # Interpolate using the RegularGridInterpolator
            Stmp = interpolator(interp_points).reshape(Mlat_mesh.shape)
        
            # Apply mask
            Stmp[Mfsm] = np.nan
            Schl_interp[d, :, :] = Stmp
        
            print(f"Interpolating day {d+1}")
    
        print("Interpolation terminated")
        
        M_TS_ave = np.nanmean(Mchl_complete, axis=(1, 2))
        S_TS_ave = np.nanmean(Schl_interp, axis=(1, 2))
    
    elif chldlev == "l3":
        print("INTERPOLATING MODEL DATA ON SAT GRID")

        Slat = Slat.T
        MinMlat = np.min(Mlat)
        exSgrid = np.where(Slat <= MinMlat)

        Mchl_interp = np.full_like(Mchl_complete, np.nan, dtype=float)

        for d in range(Truedays):
            if d % 10 == 0:
                print(f"{d} {chldlev} MODEL SCHL DAILY FIELDS INTERPOLATED")

            # Generate a land-sea mask for satellite fields
            Stmp = Schl_complete[d, :, :].copy()
            Stmp[exSgrid] = np.nan
            outlierconc = 15
            outliers = np.where(Stmp >= outlierconc)
            Stmp[outliers] = np.nan
            Schl_complete[d, :, :] = Stmp

            # Apply satellite mask
            satmask = ~np.isnan(Stmp)
            nobs = np.nansum(satmask)
            if nobs <= 500:
                satmask[:, :] = 0
        
            Schl_complete[d, :, :][satmask == 0] = np.nan
            Mchl_complete[d, :, :][satnan == 0] = np.nan

            # Expand data over land
            noflood = Mchl_complete[d, :, :]
            flooded = flood(noflood, 5)

            # Create the interpolator
            interpolator = RegularGridInterpolator((Mlon[:, 0], Mlat[0, :]), flooded, method='linear', bounds_error=False, fill_value=np.nan)

            # Define the satellite grid for interpolation
            Slon_mesh, Slat_mesh = np.meshgrid(Slon, Slat, indexing='ij')
            interp_points = np.column_stack([Slat_mesh.ravel(), Slon_mesh.ravel()])

            # Interpolate using the RegularGridInterpolator
            Mtmp = interpolator(interp_points).reshape(Slat_mesh.shape)

            # Apply masks
            Mtmp[outliers] = np.nan
            Mtmp[satmask == 0] = np.nan
            Mtmp[np.where(Mtmp >= outlierconc)] = np.nan

            Mchl_interp[d, :, :] = Mtmp

        # Compute time-series averages
        M_TS_ave = np.nanmean(Mchl_interp, axis=(1, 2))
        S_TS_ave = np.nanmean(Schl_complete, axis=(1, 2))
    
    return M_TS_ave, S_TS_ave


# Call the interpolation function
interpolate_chl_data(Truedays, chldlev, Schl_complete, Slon, Slat, Mlon, Mlat, Mmask, Mchl_complete, Mfsm)