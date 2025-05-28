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

    if choice == '2' or choice == '3':

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

    if choice == "2" or choice == "3":
        print("Saving ModData_complete as a .nc file...")
        ModData_complete_xr = xr.DataArray(ModData_complete)
        ModData_complete_xr.to_netcdf("ModData_complete.nc")
        print("ModData_complete saved as ModData_complete.nc")
        print("-"*45)
    
    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*"*45)

def save_model_SST_data(output_path, Msst_complete):
    
    assert isinstance(output_path, (str, Path)), "output_path must be a string or Path object"
    assert os.path.isdir(output_path), f"'{output_path}' is not a valid directory"

    assert isinstance(Msst_complete, dict), "Msst_complete must be a dictionary"
    assert all(isinstance(k, (str, int)) for k in Msst_complete.keys()), "All keys in Msst_complete should be strings or integers (years)"
    assert all(isinstance(v, np.ndarray) for v in Msst_complete.values()), "All values in Msst_complete should be NumPy arrays"
    assert all(v.ndim == 3 for v in Msst_complete.values()), "Each SST array must be 3D (e.g., time x lat x lon)"

    os.chdir(output_path)
    
    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-'*45)

    # Prepare .mat data
    mat_data = {}
    for year, array in Msst_complete.items():
        mat_data[str(year)] = array  # Store each array under its corresponding year

    if choice == "1" or choice == "3":
        print("Saving data as a .mat file...")
        try:
            scipy.io.savemat("Msst_complete.mat", mat_data)
            print("Data saved as Msst_complete.mat")
        except Exception as e:
            print(f"Error saving MAT file: {e}")
        print("-"*45)

    if choice == "2" or choice == "3":
        print("Saving each year separately as .nc files...")
        for year, array in Msst_complete.items():
            try:
                ds = xr.Dataset({str(year): (["time", "lat", "lon"], array)})  # Assuming array has shape (time, lat, lon)
                filename = f"Msst_{year}.nc"
                ds.to_netcdf(filename)
                print(f"Saved {filename}")
            except Exception as e:
                print(f"Error saving {year} NetCDF file: {e}")
        print("-"*45)
    
    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*"*45)
###############################################################################

###############################################################################     
def save_SST_Bavg(output_path, BASSTmod, BASSTsat):
    
    os.chdir(output_path)
    
    assert isinstance(output_path, (str, Path)), "output_path must be a string or Path object"
    assert os.path.isdir(output_path), f"{output_path} is not a valid directory"

    assert isinstance(BASSTmod, (list, np.ndarray)), "BASSTmod must be a list or NumPy array"
    assert isinstance(BASSTsat, (list, np.ndarray)), "BASSTsat must be a list or NumPy array"
    assert len(BASSTmod) == len(BASSTsat), "BASSTmod and BASSTsat must have the same length"

    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-'*45)

    # Prepare .mat data
    mat_data = {
        "BASSTmod": BASSTmod,
        "BASSTsat": BASSTsat
    }

    if choice == "1" or choice == "3":
        print("Saving data as a .mat file...")
        try:
            scipy.io.savemat("BASST_data.mat", mat_data)
            print("Data saved as BASST_data.mat")
        except Exception as e:
            print(f"Error saving MAT file: {e}")
        print("-"*45)

    if choice == "2" or choice == "3":
        print("Saving each dataset separately as .nc files...")
        
        # Save BASSTmod as NetCDF
        try:
            ds_mod = xr.Dataset({"BASSTmod": ("time", BASSTmod)})
            filename_mod = "BASSTmod.nc"
            ds_mod.to_netcdf(filename_mod)
            print(f"Saved {filename_mod}")
        except Exception as e:
            print(f"Error saving BASSTmod NetCDF file: {e}")
        
        # Save BASSTsat as NetCDF
        try:
            ds_sat = xr.Dataset({"BASSTsat": ("time", BASSTsat)})
            filename_sat = "BASSTsat.nc"
            ds_sat.to_netcdf(filename_sat)
            print(f"Saved {filename_sat}")
        except Exception as e:
            print(f"Error saving BASSTsat NetCDF file: {e}")
        
        print("-"*45)

    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*"*45)    