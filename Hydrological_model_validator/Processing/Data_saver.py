import scipy.io
import xarray as xr
import os
import numpy as np
import pathlib as Path
###############################################################################

###############################################################################
def save_satellite_data(output_path, Sat_lon, Sat_lat, SatData_complete):
    
    assert isinstance(output_path, (str, Path)), "output_path must be a string or Path object"
    assert os.path.isdir(output_path), f"'{output_path}' is not a valid directory"

    assert isinstance(Sat_lon, (np.ndarray, xr.DataArray)), "Slon must be a NumPy array or xarray DataArray"
    assert isinstance(Sat_lat, (np.ndarray, xr.DataArray)), "Slat must be a NumPy array or xarray DataArray"
    assert isinstance(SatData_complete, (np.ndarray, xr.DataArray)), "Schl_complete must be a NumPy array or xarray DataArray"

    assert Sat_lon.ndim == 2, f"Slon should be 2D, got shape {Sat_lon.shape}"
    assert Sat_lat.ndim == 2, f"Slat should be 2D, got shape {Sat_lat.shape}"
    assert SatData_complete.ndim == 3, f"Schl_complete should be 3D (time, lat, lon), got shape {SatData_complete.shape}"

    os.chdir(output_path)

    print("Saving the data in the folder...")
    
    data = {
        'Sat_lon': Sat_lon,
        'Sat_lat': Sat_lat,
        'SatData_complete': SatData_complete
    }

    # File format input
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-'*45)

    if choice == '1' or choice == '3':
        print("Saving data as a single .mat file...")
        scipy.io.savemat("SatData_clean.mat", data)
        print("Data saved as SatData_clean.mat")
        print("-"*45)

    elif choice == '2' or choice == '3':

        print("Saving the data as separate .nc files...")
        
        xr.DataArray(Sat_lon).to_netcdf("Sat_lon.nc")
        print("Sat_lon saved as Sat_lon.nc")

        xr.DataArray(Sat_lat).to_netcdf("Sat_lat.nc")
        print("Sat_lat saved as Sat_lat.nc")

        xr.DataArray(SatData_complete).to_netcdf("SatData_complete.nc")
        print("SatData_complete saved as SatData_complete.nc")
        print("-"*45)

    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*" * 45)
###############################################################################

###############################################################################    
def save_model_data(output_path, ModData_complete):
    
    assert isinstance(output_path, (str, Path)), "output_path must be a string or Path object"
    assert os.path.isdir(output_path), f"'{output_path}' is not a valid directory"

    assert isinstance(ModData_complete, (np.ndarray, xr.DataArray)), "Mchl_complete must be a NumPy array or xarray DataArray"
    assert ModData_complete.ndim == 3, f"Mchl_complete should be 3D (e.g., [time, lat, lon]), got shape {ModData_complete.shape}"

    os.chdir(output_path)
    
    print("\033[91m⚠️ Careful ⚠️\033[0m")
    print("\033[91m⚠️ These dataset do not have the satnan mask applied to them ⚠️\033[0m")
    print(" For further analysis it is suggested to pass these data")
    print(" though the Interpolator.m script provided alongise")
    print(" these Python scripts to ensure that shapes etc. match")
    print(" with the Satellite data, especially regarding the presence")
    print(" of the missing satellite values")
    print("\033[91m⚠️ This is necessary when using the level3 data ⚠️\033[0m")
    
    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()  # Remove extra spaces
    print('-'*45)

    if choice == "1" or choice == "3":
        print("Saving data as a .mat file...")
        # Saving as .mat file - wrap Mchl_complete in a dictionary
        scipy.io.savemat("ModData_complete.mat", {"ModData_complete": ModData_complete})
        print("Data saved as ModData_complete.mat")
        print("-"*45)

    elif choice == "2" or choice == "3":
        print("Saving ModData_complete as a .nc file...")
        ModData_complete_xr = xr.DataArray(ModData_complete)
        ModData_complete_xr.to_netcdf("ModData_complete.nc")
        print("ModData_complete saved as ModData_complete.nc")
        print("-"*45)
    
    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*"*45)
###############################################################################

###############################################################################    
def save_to_netcdf(data_dict, output_path):
    """
    Save a dictionary of arrays or DataArrays to a NetCDF file.

    Parameters
    ----------
    data_dict : dict
        Dictionary where keys are variable names and values are numpy arrays or xarray.DataArrays.
    output_path : str
        Full path to save the NetCDF file.

    Returns
    -------
    None
    """
    os.chdir(output_path)
    
    for var_name, data in data_dict.items():
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data, name=var_name)
        ds = xr.Dataset({var_name: data})
        
        filepath = os.path.join(output_path, f"{var_name}.nc")
        ds.to_netcdf(filepath)