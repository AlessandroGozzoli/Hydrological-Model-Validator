import scipy.io
import xarray as xr
import os
import numpy as np
import pathlib as Path

def save_satellite_CHL_data(output_path, Slon, Slat, Schl_complete):
    
    assert isinstance(output_path, (str, Path)), "output_path must be a string or Path object"
    assert os.path.isdir(output_path), f"'{output_path}' is not a valid directory"

    assert isinstance(Slon, (np.ndarray, xr.DataArray)), "Slon must be a NumPy array or xarray DataArray"
    assert isinstance(Slat, (np.ndarray, xr.DataArray)), "Slat must be a NumPy array or xarray DataArray"
    assert isinstance(Schl_complete, (np.ndarray, xr.DataArray)), "Schl_complete must be a NumPy array or xarray DataArray"

    assert Slon.ndim == 2, f"Slon should be 2D, got shape {Slon.shape}"
    assert Slat.ndim == 2, f"Slat should be 2D, got shape {Slat.shape}"
    assert Schl_complete.ndim == 3, f"Schl_complete should be 3D (time, lat, lon), got shape {Schl_complete.shape}"

    os.chdir(output_path)

    print("Saving the data in the folder...")
    
    data = {
        'Slon': Slon,
        'Slat': Slat,
        'Schl_complete': Schl_complete
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
        scipy.io.savemat("chl_clean.mat", data)
        print("Data saved as chl_clean.mat")
        print("-"*45)

    if choice == '2' or choice == '3':

        print("Saving the data as separate .nc files...")
        
        xr.DataArray(Slon).to_netcdf("Slon.nc")
        print("Slon saved as Slon.nc")

        xr.DataArray(Slat).to_netcdf("Slat.nc")
        print("Slat saved as Slat.nc")

        xr.DataArray(Schl_complete).to_netcdf("Schl_complete.nc")
        print("Schl_complete saved as Schl_complete.nc")
        print("-"*45)

    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*" * 45)

def save_satellite_SST_data(output_path, Sat_sst):
    
    assert isinstance(output_path, (str, Path)), "output_path must be a string or Path object"
    assert os.path.isdir(output_path), f"'{output_path}' is not a valid directory"

    assert isinstance(Sat_sst, (np.ndarray, xr.DataArray)), "Sat_sst must be a NumPy array or xarray DataArray"
    assert Sat_sst.ndim in [2, 3], f"Sat_sst should be 2D or 3D (e.g., [time, lat, lon]), got shape {Sat_sst.shape}"

    # Data to save
    data = {
        'Sat_sst': Sat_sst
    }
    
    os.chdir(output_path)

    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()  # Remove extra spaces
    print('-'*45)

    # Save based on user choice
    if choice == '1' or choice == '3':
        print("Saving the SST data as a .mat file...")
        # Saving as .mat file
        scipy.io.savemat("Sat_sst.mat", data)
        print("Data saved as Sat_sst.mat")
        print("-"*45)

    if choice == '2' or choice == '3':
        print("Saving the SST data as .nc file...")
        # Saving as .nc file
        Sat_sst_xr = xr.DataArray(Sat_sst)
        Sat_sst_xr.to_netcdf("Sat_sst.nc")
        print("Sat_sst saved as Sat_sst.nc")
        print("-"*45)

    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*"*45)
    
def save_model_CHL_data(output_path, Mchl_complete):
    
    assert isinstance(output_path, (str, Path)), "output_path must be a string or Path object"
    assert os.path.isdir(output_path), f"'{output_path}' is not a valid directory"

    assert isinstance(Mchl_complete, (np.ndarray, xr.DataArray)), "Mchl_complete must be a NumPy array or xarray DataArray"
    assert Mchl_complete.ndim == 3, f"Mchl_complete should be 3D (e.g., [time, lat, lon]), got shape {Mchl_complete.shape}"

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
        scipy.io.savemat("Mchl_complete.mat", {"Mchl_complete": Mchl_complete})
        print("Data saved as Mchl_complete.mat")
        print("-"*45)

    if choice == "2" or choice == "3":
        print("Saving Mchl_complete as a .nc file...")
        Mchl_complete_xr = xr.DataArray(Mchl_complete)
        Mchl_complete_xr.to_netcdf("Mchl_complete.nc")
        print("Mchl_complete saved as Mchl_complete.nc")
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