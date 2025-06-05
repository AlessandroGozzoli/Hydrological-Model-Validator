import scipy.io
import xarray as xr
from typing import Dict, Union
import numpy as np
from pathlib import Path
###############################################################################

###############################################################################
def save_satellite_data(output_path, Sat_lon, Sat_lat, SatData_complete):
    """
    Save satellite longitude, latitude, and data arrays to disk in user-selected formats.

    This function saves the provided satellite coordinate and data arrays to the specified directory,
    allowing the user to choose between MATLAB .mat format, NetCDF .nc format, or both.
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
    Enter the number corresponding to your choice: 3
    Saving data as a single .mat file...
    Data saved as SatData_clean.mat
    Saving the data as separate .nc files...
    Sat_lon saved as Sat_lon.nc
    Sat_lat saved as Sat_lat.nc
    SatData_complete saved as SatData_complete.nc
    ✅ The requested data has been saved!
    """
    # Check if output_path is a string or Path; convert to Path object for consistent path handling
    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a string or Path object")
    output_path = Path(output_path)
    
    # Verify output_path exists and is a directory to prevent writing errors
    if not output_path.is_dir():
        raise ValueError(f"'{output_path}' is not a valid directory")

    # Validate input types for Sat_lon, Sat_lat, and SatData_complete to ensure compatibility
    if not isinstance(Sat_lon, (np.ndarray, xr.DataArray)):
        raise TypeError("Sat_lon must be a NumPy array or xarray DataArray")
    if not isinstance(Sat_lat, (np.ndarray, xr.DataArray)):
        raise TypeError("Sat_lat must be a NumPy array or xarray DataArray")
    if not isinstance(SatData_complete, (np.ndarray, xr.DataArray)):
        raise TypeError("SatData_complete must be a NumPy array or xarray DataArray")

    # Check that longitude and latitude arrays are 2D as expected for spatial grid coordinates
    if Sat_lon.ndim != 2:
        raise ValueError(f"Sat_lon should be 2D, got shape {Sat_lon.shape}")
    if Sat_lat.ndim != 2:
        raise ValueError(f"Sat_lat should be 2D, got shape {Sat_lat.shape}")
    
    # Satellite data should be 3D: time, latitude, and longitude dimensions
    if SatData_complete.ndim != 3:
        raise ValueError(f"SatData_complete should be 3D (time, lat, lon), got shape {SatData_complete.shape}")

    print("Saving the data in the folder...")

    # Prepare dictionary for MATLAB .mat saving; keys are variable names
    data = {
        'Sat_lon': Sat_lon,
        'Sat_lat': Sat_lat,
        'SatData_complete': SatData_complete
    }

    # Prompt user to choose output format(s), giving flexibility in saving
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-' * 45)

    # If user selects MAT or both, save all variables together in one .mat file
    if choice == '1' or choice == '3':
        print("Saving data as a single .mat file...")
        # scipy.io.savemat expects numpy arrays or dict of arrays; xarray objects can be passed if compatible
        scipy.io.savemat(str(output_path / "SatData_clean.mat"), data)
        print("Data saved as SatData_clean.mat")
        print("-" * 45)

    # If user selects NetCDF or both, save each variable separately as .nc files using xarray.DataArray
    if choice == '2' or choice == '3':
        print("Saving the data as separate .nc files...")

        # Convert Sat_lon to xarray DataArray and save as NetCDF, preserving metadata if present
        xr.DataArray(Sat_lon).to_netcdf(str(output_path / "Sat_lon.nc"))
        print("Sat_lon saved as Sat_lon.nc")

        # Similarly save Sat_lat coordinates
        xr.DataArray(Sat_lat).to_netcdf(str(output_path / "Sat_lat.nc"))
        print("Sat_lat saved as Sat_lat.nc")

        # Save the 3D satellite data variable in NetCDF format
        xr.DataArray(SatData_complete).to_netcdf(str(output_path / "SatData_complete.nc"))
        print("SatData_complete saved as SatData_complete.nc")
        print("-" * 45)

    # Warn user if input choice was invalid; no files saved in this case
    if choice not in {'1', '2', '3'}:
        print("Invalid choice. Please run the script again and select a valid option.")

    # Confirm success to user with colored output for clarity
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
        raise TypeError("output_path must be a string or Path object")
    output_path = Path(output_path)
    
    # Ensure output_path exists and is a directory to avoid write errors
    if not output_path.is_dir():
        raise ValueError(f"'{output_path}' is not a valid directory")

    # Confirm ModData_complete is either a NumPy array or an xarray DataArray for compatibility
    if not isinstance(ModData_complete, (np.ndarray, xr.DataArray)):
        raise TypeError("ModData_complete must be a NumPy array or xarray DataArray")

    # Check that ModData_complete is 3D to fit the expected (time, lat, lon) format
    if ModData_complete.ndim != 3:
        raise ValueError(f"ModData_complete should be 3D (e.g., [time, lat, lon]), got shape {ModData_complete.shape}")

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
        print("Invalid choice. Please run the script again and select a valid option.")

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
        raise ValueError(f"'{output_path}' is not a valid directory")

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
            raise TypeError(f"Data for variable '{var_name}' must be a NumPy array or xarray DataArray")

        # Wrap the DataArray inside a Dataset so we can save it to NetCDF format
        ds = xr.Dataset({data.name: data})
        # Define the full output path filename for the NetCDF file
        filepath = output_path / f"{var_name}.nc"
        # Save the Dataset to a NetCDF file at the specified path
        ds.to_netcdf(filepath)
