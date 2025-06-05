import scipy.io
import xarray as xr
from typing import Dict, Union
import numpy as np
from pathlib import Path
import pandas as pd
import json
###############################################################################

###############################################################################
def save_satellite_data(output_path, Sat_lon, Sat_lat, SatData_complete):
    """
    Save satellite longitude, latitude, and data arrays to disk in user-selected formats.

    This function saves the provided satellite coordinate and data arrays to the specified directory,
    allowing the user to choose between MATLAB .mat format, NetCDF .nc format, .json format or all of the above.
    Longitude and latitude arrays must be 2D; the satellite data array must be 3D (time, lat, lon).

    Parameters
    ----------
    output_path : str or pathlib.Path
        Directory path where the output files will be saved.
    Sat_lon : np.ndarray or xarray.DataArray
        2D array representing satellite longitudes.
    Sat_lat : np.ndarray or xarray.DataArray
        2D array representing satellite latitudes.
    SatData_complete : np.ndarray or xarray.DataArray
        3D array of satellite data with dimensions (time, latitude, longitude).

    Raises
    ------
    TypeError
        If any input is not of the expected type.
    ValueError
        If input array dimensions are incorrect or if output_path is not a valid directory.

    Notes
    -----
    The user is prompted to select the desired file format(s) interactively.

    Example
    -------
    >>> save_satellite_data('./output', Sat_lon, Sat_lat, SatData_complete)
    Choose a file format to save the data:
    1. MAT-File (.mat)
    2. NetCDF (.nc)
    3. Both MAT and NetCDF
    4. JSON (.json)
    5. All of the above
    Enter the number corresponding to your choice: 3
    Saving data as a single .mat file...
    Data saved as SatData_clean.mat
    Saving the data as separate .nc files...
    Sat_lon saved as Sat_lon.nc
    Sat_lat saved as Sat_lat.nc
    SatData_complete saved as SatData_complete.nc
    ✅ The requested data has been saved!
    """
    # Validate output path
    if not isinstance(output_path, (str, Path)):
        raise TypeError("❌ output_path must be a string or Path object ❌")
    output_path = Path(output_path)
    if not output_path.is_dir():
        raise ValueError(f"❌ '{output_path}' is not a valid directory ❌")

    # Validate data types
    if not isinstance(Sat_lon, (np.ndarray, xr.DataArray)):
        raise TypeError("❌ Sat_lon must be a NumPy array or xarray DataArray ❌")
    if not isinstance(Sat_lat, (np.ndarray, xr.DataArray)):
        raise TypeError("❌ Sat_lat must be a NumPy array or xarray DataArray ❌")
    if not isinstance(SatData_complete, (np.ndarray, xr.DataArray)):
        raise TypeError("❌ SatData_complete must be a NumPy array or xarray DataArray ❌")

    # Validate dimensions
    if Sat_lon.ndim != 2:
        raise ValueError(f"❌ Sat_lon should be 2D, got shape {Sat_lon.shape} ❌")
    if Sat_lat.ndim != 2:
        raise ValueError(f"❌ Sat_lat should be 2D, got shape {Sat_lat.shape} ❌")
    if SatData_complete.ndim != 3:
        raise ValueError(f"❌ SatData_complete should be 3D (time, lat, lon), got shape {SatData_complete.shape} ❌")

    # Convert xarray to numpy if needed
    if isinstance(Sat_lon, xr.DataArray):
        Sat_lon = Sat_lon.values
    if isinstance(Sat_lat, xr.DataArray):
        Sat_lat = Sat_lat.values
    if isinstance(SatData_complete, xr.DataArray):
        SatData_complete = SatData_complete.values

    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    print("4. JSON (.json)")
    print("5. All formats")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-' * 45)

    # Save as .mat
    if choice in {'1', '3', '5'}:
        print("Saving data as a single .mat file...")
        scipy.io.savemat(str(output_path / "SatData_clean.mat"), {
            'Sat_lon': Sat_lon,
            'Sat_lat': Sat_lat,
            'SatData_complete': SatData_complete
        })
        print("Data saved as SatData_clean.mat")
        print("-" * 45)

    # Save as .nc
    if choice in {'2', '3', '5'}:
        print("Saving the data as separate .nc files...")
        xr.DataArray(Sat_lon).to_netcdf(str(output_path / "Sat_lon.nc"))
        print("Sat_lon saved as Sat_lon.nc")
        xr.DataArray(Sat_lat).to_netcdf(str(output_path / "Sat_lat.nc"))
        print("Sat_lat saved as Sat_lat.nc")
        xr.DataArray(SatData_complete).to_netcdf(str(output_path / "SatData_complete.nc"))
        print("SatData_complete saved as SatData_complete.nc")
        print("-" * 45)

    # Save as .json
    if choice in {'4', '5'}:
        print("Saving data as a JSON file (flattened)...")
        data_json = {
            'Sat_lon': Sat_lon.tolist(),
            'Sat_lat': Sat_lat.tolist(),
            'SatData_complete': SatData_complete.tolist()
        }
        json_path = output_path / "SatData_clean.json"
        with open(json_path, 'w') as f:
            json.dump(data_json, f)
        print("Data saved as SatData_clean.json")
        print("-" * 45)

    # Invalid choice
    if choice not in {'1', '2', '3', '4', '5'}:
        print("❌ Invalid choice. Please run the script again and select a valid option.")
        return

    # Done
    print("\033[92m✅ The requested data has been saved!\033[0m")
    print("*" * 45)
###############################################################################

###############################################################################    
def save_model_data(output_path, ModData_complete):
    """
    Save model data array to disk in user-selected formats.

    This function saves the provided 3D model data array to the specified directory.
    The user is prompted to choose between MATLAB .mat format, NetCDF .nc format, or both.
    A warning is displayed regarding the absence of satellite mask application.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Directory path where the output files will be saved.
    ModData_complete : np.ndarray or xarray.DataArray
        3D array of model data (e.g., dimensions [time, lat, lon]).

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If ModData_complete does not have 3 dimensions or if output_path is invalid.

    Notes
    -----
    It is recommended to apply the satellite mask via the Interpolator.m script
    before further analysis to ensure alignment with satellite data, especially
    when using Level 3 data.
    """
    # Verify output_path is a string or Path, then convert to Path object for consistency
    if not isinstance(output_path, (str, Path)):
        raise TypeError("❌ output_path must be a string or Path object ❌")
    output_path = Path(output_path)
    
    # Ensure output_path exists and is a directory to avoid write errors
    if not output_path.is_dir():
        raise ValueError(f"❌ '{output_path}' is not a valid directory ❌")

    # Confirm ModData_complete is either a NumPy array or an xarray DataArray for compatibility
    if not isinstance(ModData_complete, (np.ndarray, xr.DataArray)):
        raise TypeError("❌ ModData_complete must be a NumPy array or xarray DataArray ❌")

    # Check that ModData_complete is 3D to fit the expected (time, lat, lon) format
    if ModData_complete.ndim != 3:
        raise ValueError(f"❌ ModData_complete should be 3D (e.g., [time, lat, lon]), got shape {ModData_complete.shape} ❌")

    # Display important warning that the satellite mask (satnan) has not been applied,
    # which could affect analysis due to missing satellite data masking
    print("\033[91m⚠️ Careful ⚠️\033[0m")
    print("\033[91m⚠️ These dataset do not have the satnan mask applied to them ⚠️\033[0m")
    print(" For further analysis it is suggested to pass these data")
    print(" through the Interpolator.m script provided alongside")
    print(" these Python scripts to ensure that shapes etc. match")
    print(" with the Satellite data, especially regarding the presence")
    print(" of the missing satellite values")
    print("\033[91m⚠️ This is necessary when using the level3 data ⚠️\033[0m")

    # Prompt user to choose preferred output format(s)
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-' * 45)

    # Save as .mat file if chosen or if both formats chosen
    if choice == "1" or choice == "3":
        print("Saving data as a .mat file...")
        # Save the 3D array under the variable name "ModData_complete"
        scipy.io.savemat(str(output_path / "ModData_complete.mat"), {"ModData_complete": ModData_complete})
        print("Data saved as ModData_complete.mat")
        print("-" * 45)

    # Save as NetCDF if chosen or if both formats chosen
    if choice == "2" or choice == "3":
        print("Saving ModData_complete as a .nc file...")
        # Convert to xarray DataArray for easy NetCDF saving, preserving structure
        xr.DataArray(ModData_complete).to_netcdf(str(output_path / "ModData_complete.nc"))
        print("ModData_complete saved as ModData_complete.nc")
        print("-" * 45)

    # If user enters an invalid option, notify and advise to rerun
    if choice not in {"1", "2", "3"}:
        print("❌ Invalid choice. Please run the script again and select a valid option. ❌")

    # Confirm completion of saving with colored output for visibility
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
    TypeError
        If data items are not NumPy arrays or xarray DataArrays.

    Example
    -------
    >>> import numpy as np
    >>> import xarray as xr
    >>> data_dict = {
    ...     'temperature': np.random.rand(10, 5, 5),
    ...     'precipitation': xr.DataArray(np.random.rand(10, 5, 5))
    ... }
    >>> save_to_netcdf(data_dict, "./output_data")
    """
    # Convert output_path to Path object to standardize path operations
    output_path = Path(output_path)
    # Validate the output_path exists and is a directory to avoid file saving errors
    if not output_path.is_dir():
        raise ValueError(f"❌ '{output_path}' is not a valid directory ❌")

    # Iterate over each key-value pair in the data dictionary
    for var_name, data in data_dict.items():
        # If the data is a numpy ndarray, convert it to an xarray DataArray for easier NetCDF saving
        if isinstance(data, np.ndarray):
            data = xr.DataArray(data, name=var_name)
        # If already an xarray DataArray but lacks a name, assign the dict key as its name
        elif isinstance(data, xr.DataArray):
            if data.name is None:
                data.name = var_name
        else:
            # Raise an error if the data type is unsupported for saving
            raise TypeError(f"❌ Data for variable '{var_name}' must be a NumPy array or xarray DataArray ❌")

        # Wrap the DataArray inside a Dataset so we can save it to NetCDF format
        ds = xr.Dataset({data.name: data})
        # Define the full output path filename for the NetCDF file
        filepath = output_path / f"{var_name}.nc"
        # Save the Dataset to a NetCDF file at the specified path
        ds.to_netcdf(filepath)
###############################################################################

###############################################################################         
def convert_to_serializable(obj):
    """
    Recursively convert an object to a form compatible with JSON serialization.

    Parameters
    ----------
    obj : any
        The object to convert.

    Returns
    -------
    obj_serializable : JSON-compatible representation

    Notes
    -----
    Converts NumPy arrays, pandas DataFrames/Series, and xarray DataArrays/Datasets to JSON-friendly types.
    Fallbacks to string representation for unsupported objects.
    """

    # Directly return JSON-native types
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Recursively handle iterable types
    elif isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(i) for i in obj]

    # Recursively handle dictionaries
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}

    # Convert NumPy arrays to nested lists
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # Convert pandas DataFrame to list of row dictionaries
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    # Convert pandas Series to dictionary
    elif isinstance(obj, pd.Series):
        return obj.to_dict()

    # xarray DataArray: separate out dims, coords, and values
    elif isinstance(obj, xr.DataArray):
        return {
            "dims": obj.dims,
            "coords": {k: v.values.tolist() for k, v in obj.coords.items()},
            "data": obj.values.tolist()
        }

    # xarray Dataset: convert to dict
    elif isinstance(obj, xr.Dataset):
        return obj.to_dict(data=True)

    # Try using .to_dict() if available (e.g., dataclass)
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()

    # Fallback: return string representation
    return str(obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict(orient="records") if isinstance(obj, pd.DataFrame) else obj.to_dict()
    elif isinstance(obj, xr.DataArray):
        return {
            "dims": obj.dims,
            "coords": {k: v.values.tolist() for k, v in obj.coords.items()},
            "data": obj.values.tolist()
        }
    elif isinstance(obj, xr.Dataset):
        return obj.to_dict(data=True)
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    else:
        return str(obj)
###############################################################################

############################################################################### 
def save_variable_to_json(variable, output_path):
    """
    Save any Python variable to a JSON file in a serializable format.

    Supports basic Python types, NumPy arrays, pandas DataFrames/Series,
    xarray DataArrays/Datasets, dictionaries, and nested combinations.

    Parameters
    ----------
    variable : any
        The Python object or data structure to save (e.g., dict, array, DataFrame, DataArray).
    output_path : str or Path
        File path (must end in .json) where the data will be saved.

    Raises
    ------
    ValueError
        If the output_path does not end with '.json'.
    TypeError
        If the object cannot be serialized to JSON and no fallback is possible.

    Example
    -------
    >>> import numpy as np
    >>> save_variable_to_json(np.array([[1, 2], [3, 4]]), "array_data.json")
    """
    output_path = Path(output_path)

    # Ensure output is a JSON file
    if output_path.suffix.lower() != '.json':
        raise ValueError("❌ Output file must have a .json extension ❌")

    # Convert the variable to a JSON-compatible object
    serializable_obj = convert_to_serializable(variable)

    # Write to file using built-in JSON module
    with open(output_path, 'w') as f:
        json.dump(serializable_obj, f, indent=2)

    print(f"\033[92m✅ Variable saved to {output_path}\033[0m")