###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Standard library imports
import io
import os
import shutil
import gzip
from pathlib import Path
from typing import Tuple, Union, Optional, List, Any

# Third-party libraries
import numpy as np
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset

# Logging and tracing
import logging
from eliot import start_action, log_message

# Module utilities
from .time_utils import Timer

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def mask_reader(BaseDIR: Union[str, Path]) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Load the land-sea mask and associated latitude/longitude fields from a NEMO 'mesh_mask.nc' file.

    This function extracts:
    - A 2D land-sea surface mask (0 = land, 1 = ocean),
    - Indices of land grid points in 2D and 3D masks,
    - Latitude and longitude arrays on the model grid.

    Parameters
    ----------
    BaseDIR : str or Path
        Path to the directory containing the 'mesh_mask.nc' NetCDF file.

    Returns
    -------
    tuple
        A tuple containing:
        - Mmask (np.ndarray): 2D surface land-sea mask array with shape (y, x).
        - Mfsm (tuple of np.ndarray): Tuple of indices (y, x) where surface mask equals 0 (land).
        - Mfsm_3d (tuple of np.ndarray): Tuple of indices (z, y, x) where full 3D mask equals 0 (land).
        - Mlat (np.ndarray): Latitude array with same shape as Mmask.
        - Mlon (np.ndarray): Longitude array with same shape as Mmask.
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(BaseDIR, (str, Path)):
        raise TypeError("❌ BaseDIR must be a string or Path object ❌")

    with Timer("mask_reader function"):
        with start_action(action_type="mask_reader function", basedir=str(BaseDIR)):

            # ===== GETTING THE MASK =====
            BaseDIR = Path(BaseDIR)

            # --- Handle both folder or file input
            if BaseDIR.is_dir():
                mask_file = BaseDIR / 'mesh_mask.nc'
            else:
                mask_file = BaseDIR

            if not mask_file.exists():
                raise FileNotFoundError(f"❌ Mask file not found at {mask_file} ❌")

            log_message("Opening mask file", path=str(mask_file))
            logging.info(f"Opening NetCDF file: {mask_file}")

            with nc.Dataset(mask_file, 'r') as ds:
                # Check required variables
                required_vars = ['tmask', 'nav_lat', 'nav_lon']
                for var in required_vars:
                    if var not in ds.variables:
                        raise KeyError(f"❌ '{var}' not found in {mask_file.name} ❌")

                log_message("All required variables found", variables=required_vars)
                logging.info(f"Required variables present: {required_vars}")

                mask3d = ds.variables['tmask'][:]

                if mask3d.ndim == 4:
                    mask3d = mask3d[0, :, :, :]
                elif mask3d.ndim != 3:
                    raise ValueError(f"❌ Expected 'tmask' to be 3D or 4D, got shape {mask3d.shape} ❌")

                Mmask = mask3d[0, :, :]

                if Mmask.ndim != 2:
                    raise ValueError(f"❌ Expected 2D surface mask (y, x), got shape {Mmask.shape} ❌")

                Mfsm = np.where(Mmask == 0)
                Mfsm_3d = np.where(mask3d == 0)

                Mlat = ds.variables['nav_lat'][:]
                Mlon = ds.variables['nav_lon'][:]

                if Mlat.shape != Mmask.shape:
                    raise ValueError(f"❌ Shape mismatch: Mlat {Mlat.shape} vs Mmask {Mmask.shape} ❌")
                if Mlon.shape != Mmask.shape:
                    raise ValueError(f"❌ Shape mismatch: Mlon {Mlon.shape} vs Mmask {Mmask.shape} ❌")

                log_message("Mask and coordinates loaded",
                            surface_mask_shape=Mmask.shape,
                            land_points_2d=len(Mfsm[0]),
                            land_points_3d=len(Mfsm_3d[0]))
                logging.info(f"Loaded mask and coordinates: Mmask shape={Mmask.shape}, "
                             f"2D land points={len(Mfsm[0])}, 3D land points={len(Mfsm_3d[0])}")

    return Mmask, Mfsm, Mfsm_3d, Mlat, Mlon

###############################################################################

###############################################################################

def load_dataset(
    year: Union[int, str], 
    IDIR: Union[str, Path]
) -> Tuple[Union[int, str], Optional[xr.Dataset]]:
    """
    Load a NetCDF dataset file for a specified year from a given directory.

    Parameters
    ----------
    year : int or str
        Year identifier used to construct the filename 'Msst_{year}.nc'.
    IDIR : str or Path
        Path to the directory containing the dataset files.

    Returns
    -------
    tuple
        A tuple containing:
        - year (int or str): The input year identifier.
        - ds (xarray.Dataset or None): The loaded dataset if the file exists and opens successfully,
          otherwise None.

    Raises
    ------
    ValueError
        If the input directory does not exist or is not a directory.

    Notes
    -----
    - Prints messages indicating progress or warnings.
    - Catches exceptions during dataset loading and returns None if loading fails.

    Example
    -------
    >>> year, dataset = load_dataset(2020, "/data/sea_surface_temp")
    >>> if dataset is not None:
    ...     print(dataset)
    ... else:
    ...     print("Dataset not found or failed to load.")
    """
    # ===== INPUT VAIDATION =====
    if not isinstance(year, (int, str)):
        raise TypeError("❌ year must be an int or str ❌")
    if not isinstance(IDIR, (str, Path)):
        raise TypeError("❌ IDIR must be a string or Path object ❌")

    IDIR = Path(IDIR)
    # Ensure the input directory exists and is valid
    if not (IDIR.exists() and IDIR.is_dir()):
        raise ValueError(f"❌ Input directory {IDIR} does not exist or is not a directory ❌")

    # Build full file path based on the year
    file_path = IDIR / f"Msst_{year}.nc"

    # ===== OPEN THE FILE =====
    with Timer("load_dataset function"):
        with start_action(action_type="load_dataset function", year=year, path=str(file_path)):
            if file_path.exists():
                print(f"Opening {file_path.name}...")
                log_message("Dataset file found", filename=file_path.name)
                logging.info(f"Attempting to open file: {file_path}")

                try:
                    ds = xr.open_dataset(file_path)
                    log_message("Dataset loaded successfully", dimensions=dict(ds.dims))
                    logging.info(f"Dataset loaded successfully: {file_path.name}")
                    return year, ds
                except Exception as e:
                    print(f"❌ Error opening {file_path.name}: {e} ❌")
                    log_message("Failed to load dataset", error=str(e))
                    logging.error(f"Failed to open dataset {file_path.name}: {e}")
                    return year, None
            else:
                print(f"❌ Warning: {file_path.name} not found! ❌")
                log_message("Dataset file not found", filename=file_path.name)
                logging.warning(f"Dataset file not found: {file_path}")
                return year, None
            
###############################################################################

###############################################################################
def unzip_gz_to_file(file_gz: Path, target_file: Path) -> None:
    
    """
    Decompress a .gz compressed file to a specified target file.

    Parameters
    ----------
    file_gz : Path
        Path to the input .gz compressed file.
    target_file : Path
        Path to the output decompressed file.

    Raises
    ------
    FileNotFoundError
        If the input .gz file does not exist.

    Notes
    -----
    - Creates the parent directory of the target file if it does not exist.
    - Overwrites the target file if it already exists.

    Example
    -------
    >>> from pathlib import Path
    >>> unzip_gz_to_file(Path("data/archive.nc.gz"), Path("data/archive.nc"))
    """
    # ===== VALIDATION INPUT =====
    if not isinstance(file_gz, Path):
        raise TypeError("❌ file_gz must be a Path object ❌")
    if not isinstance(target_file, Path):
        raise TypeError("❌ target_file must be a Path object ❌")

    if not file_gz.exists():
        # Check input file presence before decompressing
        raise FileNotFoundError(f"❌ File not found: {file_gz} ❌")

    with Timer("unzip_gz_to_file function"):
        with start_action(action_type="unzip_gz_to_file function", input=str(file_gz), output=str(target_file)):
            log_message("Starting decompression", input_file=str(file_gz), output_file=str(target_file))
            logging.info(f"Decompressing {file_gz} to {target_file}")

            # Make sure target directory exists, create if missing
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # ===== UNZIP =====
            # Open compressed file and copy decompressed content to target file
            with gzip.open(file_gz, 'rb') as f_in, open(target_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            log_message("Decompression completed", output_file=str(target_file))
            logging.info(f"Decompression completed: {target_file}")
            
###############################################################################

###############################################################################

def read_nc_variable_from_unzipped_file(
    file_nc: Path,
    variable_key: str
) -> np.ndarray:
    """
    Read a variable array from a NetCDF (.nc) file.

    Parameters
    ----------
    file_nc : Path
        Path to the NetCDF (.nc) file.
    variable_key : str
        Name of the variable to extract from the file.

    Returns
    -------
    np.ndarray
        The data array corresponding to the specified variable.

    Raises
    ------
    FileNotFoundError
        If the NetCDF file does not exist at the given path.
    KeyError
        If the specified variable_key is not found in the NetCDF file.

    Example
    -------
    >>> from pathlib import Path
    >>> data = read_nc_variable_from_unzipped_file(Path("data/sample.nc"), "temperature")
    >>> print(data.shape)
    (50, 100)
    """
    # ===== INPUT VALDATION =====
    if not isinstance(file_nc, Path):
        raise TypeError("❌ file_nc must be a Path object ❌")
    if not isinstance(variable_key, str):
        raise TypeError("❌ variable_key must be a string ❌")

    # Ensure the NetCDF file exists before attempting to read
    if not file_nc.exists():
        raise FileNotFoundError(f"❌ NetCDF file not found: {file_nc} ❌")

    with Timer("read_nc_variable_from_unzipped_file function"):
        with start_action(action_type="read_nc_variable_from_unzipped_file function", file=str(file_nc), variable=variable_key):
            log_message("Opening NetCDF file", filename=str(file_nc))
            logging.info(f"Opening NetCDF file: {file_nc}")

            with Dataset(file_nc, 'r') as nc:
                if variable_key not in nc.variables:
                    log_message("Variable not found", variable=variable_key)
                    logging.error(f"Variable '{variable_key}' not found in {file_nc}")
                    raise KeyError(f"❌ Variable '{variable_key}' not found in {file_nc} ❌")

                data = nc.variables[variable_key][:]
                log_message("Variable data read", variable=variable_key, shape=data.shape)
                logging.info(f"Read variable '{variable_key}' with shape {data.shape} from {file_nc}")

            return data
        
###############################################################################

###############################################################################

def read_nc_variable_from_gz_in_memory(
    file_gz: Path,
    variable_key: str
) -> np.ndarray:
    """
    Read a variable directly from a gzipped NetCDF (.nc.gz) file in memory using xarray.

    This function decompresses the gzipped file in memory without writing to disk,
    then opens the NetCDF dataset and extracts the requested variable.

    Parameters
    ----------
    file_gz : Path
        Path to the gzipped NetCDF (.nc.gz) file.
    variable_key : str
        Name of the variable to extract from the dataset.

    Returns
    -------
    np.ndarray
        Numpy array containing the data of the requested variable.

    Raises
    ------
    FileNotFoundError
        If the gzipped file does not exist.
    KeyError
        If the specified variable_key is not found in the dataset.

    Example
    -------
    >>> from pathlib import Path
    >>> data = read_nc_variable_from_gz_in_memory(Path("data/sample.nc.gz"), "temperature")
    >>> print(data.shape)
    (50, 100)
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(file_gz, Path):
        raise TypeError("❌ file_gz must be a Path object ❌")
    if not isinstance(variable_key, str):
        raise TypeError("❌ variable_key must be a string ❌")

    # Verify the gzipped file exists before proceeding
    if not file_gz.exists():
        raise FileNotFoundError(f"❌ File not found: {file_gz} ❌")

    with Timer("read_nc_variable_from_gz_in_memory function"):
        with start_action(action_type="read_nc_variable_from_gz_in_memory function", file=str(file_gz), variable=variable_key):
            log_message("Opening gzipped NetCDF file in memory", filename=str(file_gz))
            logging.info(f"Opening gzipped NetCDF file in memory: {file_gz}")

            # Read and decompress the gzipped file fully into memory as bytes
            with gzip.open(file_gz, 'rb') as f_in:
                decompressed_bytes = f_in.read()
                log_message("Decompressed gzipped file", bytes_length=len(decompressed_bytes))
                logging.info(f"Decompressed {len(decompressed_bytes)} bytes from {file_gz}")

            # Open the decompressed bytes as an in-memory NetCDF dataset using xarray
            with xr.open_dataset(io.BytesIO(decompressed_bytes)) as ds:
                if variable_key not in ds.variables:
                    log_message("Variable not found", variable=variable_key)
                    logging.error(f"Variable '{variable_key}' not found in {file_gz}")
                    raise KeyError(f"❌ Variable '{variable_key}' not found in {file_gz} ❌")

                data = ds[variable_key].values
                log_message("Variable data read", variable=variable_key, shape=data.shape)
                logging.info(f"Read variable '{variable_key}' with shape {data.shape} from gzipped file {file_gz}")

            return data
        
###############################################################################

###############################################################################

def call_interpolator(
    varname: str,
    data_level: int,
    input_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    mask_file: Union[str, os.PathLike]
) -> None:
    """
    Call the MATLAB interpolation function `Interpolator_v2` via the MATLAB Engine API for Python.

    This function starts a MATLAB session, adds the directory containing the MATLAB
    function to the MATLAB path, and calls `Interpolator_v2` with the provided arguments.

    Parameters
    ----------
    varname : str
        Variable name to interpolate (e.g., 'temperature').
    data_level : int
        Data level identifier (such as depth or layer index).
    input_dir : str or os.PathLike
        Path to the directory containing input data files.
    output_dir : str or os.PathLike
        Path to the directory where output files will be saved.
    mask_file : str or os.PathLike
        Path to the mask file required by the interpolation function.

    Raises
    ------
    RuntimeError
        If the MATLAB engine fails to start or if the interpolation function raises an error.

    Example
    -------
    >>> call_interpolator(
            varname='sst',
            data_level=0,
            input_dir='/data/input',
            output_dir='/data/output',
            mask_file='/data/mask/mesh_mask.nc'
        )
    """
    # ===== ENGINE IMPORT ====
    import matlab.engine
    
    # ===== INPUT VALIDATION =====
    if not isinstance(varname, str):
        raise TypeError("❌ varname must be a string ❌")
    if not isinstance(data_level, str):
        raise TypeError("❌ data_level must be a string ❌")
    if not isinstance(input_dir, (str, os.PathLike)):
        raise TypeError("❌ input_dir must be a string or os.PathLike object ❌")
    if not isinstance(output_dir, (str, os.PathLike)):
        raise TypeError("❌ output_dir must be a string or os.PathLike object ❌")
    if not isinstance(mask_file, (str, os.PathLike)):
        raise TypeError("❌ mask_file must be a string or os.PathLike object ❌")

    # Determine the directory containing this script to locate the MATLAB .m files
    pkg_root = os.path.dirname(os.path.abspath(__file__))
    matlab_func_folder = pkg_root  # Assuming MATLAB functions are here

    print("Adding MATLAB folder to path...")

    # ===== BEGINNING MATLAB SESSION =====
    try:
        # Start the MATLAB engine session
        eng = matlab.engine.start_matlab()
    except Exception as e:
        # Raise error if MATLAB cannot start
        raise RuntimeError(f"❌ Failed to start MATLAB engine: {e} ❌")

    try:
        # Add the MATLAB function folder to the MATLAB search path
        eng.addpath(matlab_func_folder, nargout=0)

        print("Beginning the interpolation...")

        # Call the MATLAB interpolation function with provided arguments
        eng.Interpolator_v2(varname, data_level, input_dir, output_dir, mask_file, nargout=0)
    except Exception as e:
        # Raise error if the MATLAB function call fails
        raise RuntimeError(f"❌ MATLAB interpolation failed: {e} ❌")
    finally:
        # Ensure MATLAB engine quits to free resources
        eng.quit()
        
###############################################################################

###############################################################################

def find_file_with_keywords(
    files: List[Any],
    keywords: List[str],
    description: str
) -> Any:
    """
    Search for a file among a list whose name contains any of the specified keywords.

    This function filters a list of file-like objects to find those whose filenames include
    any of the provided keywords (case-insensitive). If exactly one match is found, it is returned.
    If multiple matches are found, the user is prompted to select the exact filename.
    If no matches are found, the user is prompted to input the filename manually.

    Parameters
    ----------
    files : list of file-like objects
        List of objects with a `.name` attribute representing the filename.
    keywords : list of str
        List of keywords to search for in filenames (case-insensitive).
    description : str
        Description of the file purpose, used in prompts to the user.

    Returns
    -------
    file-like object
        The selected file object matching the keywords or user input.

    Raises
    ------
    FileNotFoundError
        Raised if the user-input filename does not exist among the candidate files or in the folder.

    Example
    -------
    >>> files = [Path("data_obs.nc"), Path("data_sim.nc"), Path("readme.txt")]
    >>> keywords = ["obs", "observed"]
    >>> selected_file = find_file_with_keywords(files, keywords, "observed data")
    >>> print(selected_file.name)
    data_obs.nc
    """
    # Filter files whose names contain any keyword (case-insensitive)
    matches = [f for f in files if any(k in f.name.lower() for k in keywords)]

    # If exactly one match, return it directly
    if len(matches) == 1:
        return matches[0]

    # If multiple matches, ask user to specify the exact filename
    elif len(matches) > 1:
        print(f"Multiple candidate files found for {description}: {[f.name for f in matches]}")
        chosen = input(f"Please enter the exact filename to use for {description}: ").strip()
        
        # Check if user's choice is among the matches
        chosen_path = [f for f in matches if f.name == chosen]
        if chosen_path:
            return chosen_path[0]
            # Note: unreachable code after return - print is redundant here
        else:
            # Raise error if chosen file is not found among candidates
            raise FileNotFoundError(f"File named '{chosen}' not found among candidates.")

    # If no matches found, ask user to input filename directly from all files
    else:
        chosen = input(f"No files matching keywords for {description} found.\nPlease enter the filename for {description}: ").strip()
        chosen_path = [f for f in files if f.name == chosen]
        if chosen_path:
            return chosen_path[0]
            # Note: unreachable code after return - print is redundant here
        else:
            # Raise error if chosen file is not found in the entire folder
            raise FileNotFoundError(f"File named '{chosen}' not found in the folder.")

###############################################################################

###############################################################################

def select_3d_variable(ds: xr.Dataset, label: str) -> xr.DataArray:
    """
    Select a 3D variable from an xarray Dataset, prompting the user if multiple candidates exist.

    This function searches for variables within the given Dataset that have exactly
    three dimensions. If none are found, it raises an error. If multiple 3D variables
    are found, it prompts the user to select one by displaying the variable names and shapes.
    If only one 3D variable exists, it is selected automatically.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the variables to search.
    label : str
        A descriptive label for the Dataset, used in prompts and error messages.

    Returns
    -------
    xr.DataArray
        The selected 3D variable as an xarray DataArray.

    Raises
    ------
    ValueError
        If no 3D variables are found in the Dataset.

    Example
    -------
    >>> var = select_3d_variable(my_dataset, "Observed data")
    ⚠️ Multiple 3D variables found in Observed data: ['temp', 'salinity']
    1: temp (shape: (time, depth, lat, lon))
    2: salinity (shape: (time, depth, lat, lon))
    Select variable number: 1
    """

    # Filter dataset variables to those that have exactly 3 dimensions
    candidate_vars = [v for v in ds.data_vars if ds[v].ndim == 3]

    # Raise error if no 3D variables found
    if not candidate_vars:
        raise ValueError(f"❌ No 3D variables found in {label}.")

    # If multiple 3D variables found, prompt user to select one
    elif len(candidate_vars) > 1:
        print(f"⚠️ Multiple 3D variables found in {label}: {candidate_vars}")

        # Display each candidate variable with its index and shape for user reference
        for i, v in enumerate(candidate_vars):
            print(f"{i+1}: {v} (shape: {ds[v].shape})")

        # Loop until user inputs a valid selection index
        while True:
            try:
                idx = int(input("Select variable number: ")) - 1
                if 0 <= idx < len(candidate_vars):
                    var = candidate_vars[idx]
                    break
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")

    # If exactly one candidate, select it automatically
    else:
        var = candidate_vars[0]

    # Return the selected DataArray
    return ds[var]
