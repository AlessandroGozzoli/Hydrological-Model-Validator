import scipy.io
import xarray as xr
from typing import Dict, Union
import numpy as np
from pathlib import Path
###############################################################################

###############################################################################
def save_satellite_data(output_path, Sat_lon, Sat_lat, SatData_complete):
    """
    Save satellite longitude, latitude, and data arrays to disk in chosen formats.

    Parameters
    ----------
    output_path : str or Path
        Directory path where files will be saved.
    Sat_lon : np.ndarray or xr.DataArray
        2D array of satellite longitudes.
    Sat_lat : np.ndarray or xr.DataArray
        2D array of satellite latitudes.
    SatData_complete : np.ndarray or xr.DataArray
        3D array of satellite data (time, lat, lon).

    Raises
    ------
    TypeError
        If inputs are not of the expected types.
    ValueError
        If shapes are not as expected or directory path is invalid.
    """
    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a string or Path object")
    output_path = Path(output_path)
    if not output_path.is_dir():
        raise ValueError(f"'{output_path}' is not a valid directory")

    if not isinstance(Sat_lon, (np.ndarray, xr.DataArray)):
        raise TypeError("Sat_lon must be a NumPy array or xarray DataArray")
    if not isinstance(Sat_lat, (np.ndarray, xr.DataArray)):
        raise TypeError("Sat_lat must be a NumPy array or xarray DataArray")
    if not isinstance(SatData_complete, (np.ndarray, xr.DataArray)):
        raise TypeError("SatData_complete must be a NumPy array or xarray DataArray")

    if Sat_lon.ndim != 2:
        raise ValueError(f"Sat_lon should be 2D, got shape {Sat_lon.shape}")
    if Sat_lat.ndim != 2:
        raise ValueError(f"Sat_lat should be 2D, got shape {Sat_lat.shape}")
    if SatData_complete.ndim != 3:
        raise ValueError(f"SatData_complete should be 3D (time, lat, lon), got shape {SatData_complete.shape}")

    print("Saving the data in the folder...")

    data = {
        'Sat_lon': Sat_lon,
        'Sat_lat': Sat_lat,
        'SatData_complete': SatData_complete
    }

    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-' * 45)

    if choice == '1' or choice == '3':
        print("Saving data as a single .mat file...")
        scipy.io.savemat(str(output_path / "SatData_clean.mat"), data)
        print("Data saved as SatData_clean.mat")
        print("-" * 45)

    if choice == '2' or choice == '3':
        print("Saving the data as separate .nc files...")

        xr.DataArray(Sat_lon).to_netcdf(str(output_path / "Sat_lon.nc"))
        print("Sat_lon saved as Sat_lon.nc")

        xr.DataArray(Sat_lat).to_netcdf(str(output_path / "Sat_lat.nc"))
        print("Sat_lat saved as Sat_lat.nc")

        xr.DataArray(SatData_complete).to_netcdf(str(output_path / "SatData_complete.nc"))
        print("SatData_complete saved as SatData_complete.nc")
        print("-" * 45)

    if choice not in {'1', '2', '3'}:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*" * 45)
###############################################################################

###############################################################################    
def save_model_data(output_path, ModData_complete):
    """
    Save model data array to disk in chosen formats.

    Parameters
    ----------
    output_path : str or Path
        Directory path where files will be saved.
    ModData_complete : np.ndarray or xr.DataArray
        3D array of model data (e.g., [time, lat, lon]).

    Raises
    ------
    TypeError
        If inputs are not of the expected types.
    ValueError
        If shapes are not as expected or directory path is invalid.
    """
    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a string or Path object")
    output_path = Path(output_path)
    if not output_path.is_dir():
        raise ValueError(f"'{output_path}' is not a valid directory")

    if not isinstance(ModData_complete, (np.ndarray, xr.DataArray)):
        raise TypeError("ModData_complete must be a NumPy array or xarray DataArray")

    if ModData_complete.ndim != 3:
        raise ValueError(f"ModData_complete should be 3D (e.g., [time, lat, lon]), got shape {ModData_complete.shape}")

    print("\033[91m⚠️ Careful ⚠️\033[0m")
    print("\033[91m⚠️ These dataset do not have the satnan mask applied to them ⚠️\033[0m")
    print(" For further analysis it is suggested to pass these data")
    print(" through the Interpolator.m script provided alongside")
    print(" these Python scripts to ensure that shapes etc. match")
    print(" with the Satellite data, especially regarding the presence")
    print(" of the missing satellite values")
    print("\033[91m⚠️ This is necessary when using the level3 data ⚠️\033[0m")

    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-' * 45)

    if choice == "1" or choice == "3":
        print("Saving data as a .mat file...")
        scipy.io.savemat(str(output_path / "ModData_complete.mat"), {"ModData_complete": ModData_complete})
        print("Data saved as ModData_complete.mat")
        print("-" * 45)

    if choice == "2" or choice == "3":
        print("Saving ModData_complete as a .nc file...")
        xr.DataArray(ModData_complete).to_netcdf(str(output_path / "ModData_complete.nc"))
        print("ModData_complete saved as ModData_complete.nc")
        print("-" * 45)

    if choice not in {"1", "2", "3"}:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*" * 45)
###############################################################################

###############################################################################    
def save_to_netcdf(data_dict: Dict[str, Union[np.ndarray, xr.DataArray]], output_path: Union[str, Path]) -> None:
    """
    Save each variable from a dictionary of arrays or DataArrays as separate NetCDF files.

    Parameters
    ----------
    data_dict : dict
        Dictionary where keys are variable names and values are numpy arrays or xarray.DataArrays.
    output_path : str or Path
        Directory path where NetCDF files will be saved.

    Raises
    ------
    ValueError
        If output_path is not a valid directory.
    """
    output_path = Path(output_path)
    if not output_path.is_dir():
        raise ValueError(f"'{output_path}' is not a valid directory")

    for var_name, data in data_dict.items():
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data, name=var_name)

        ds = xr.Dataset({var_name: data})
        filepath = output_path / f"{var_name}.nc"
        ds.to_netcdf(filepath)