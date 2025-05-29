import numpy as np
from typing import Tuple, Union, Optional
import netCDF4 as nc
from pathlib import Path
import xarray as xr
import shutil
import gzip
from netCDF4 import Dataset
import io
import os
import matlab.engine

###############################################################################
def mask_reader(BaseDIR: Union[str, Path]) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Reads the model mask and latitude/longitude data from a NetCDF file named 'mesh_mask.nc' in the specified directory.

    Parameters
    ----------
    BaseDIR : str or Path
        Base directory path containing the 'mesh_mask.nc' file.

    Returns
    -------
    Tuple containing:
        - Mmask (np.ndarray): 2D model mask array (shape: [y, x]).
        - Mfsm (Tuple[np.ndarray, ...]): Indices of land grid points in 2D mask.
        - Mfsm_3d (Tuple[np.ndarray, ...]): Indices of land grid points in 3D mask.
        - Mlat (np.ndarray): Latitude array matching mask shape.
        - Mlon (np.ndarray): Longitude array matching mask shape.

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist.
    KeyError
        If required variables are missing in the NetCDF file.
    ValueError
        If variable dimensions or shapes do not match expected values.
    TypeError
        If BaseDIR is not a string or Path.
    """
    if not isinstance(BaseDIR, (str, Path)):
        raise TypeError("BaseDIR must be a string or Path object")

    BaseDIR = Path(BaseDIR)
    mask_file = BaseDIR / 'mesh_mask.nc'
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found at {mask_file}")

    with nc.Dataset(mask_file, 'r') as ds:
        required_vars = ['tmask', 'nav_lat', 'nav_lon']
        for var in required_vars:
            if var not in ds.variables:
                raise KeyError(f"'{var}' not found in {mask_file.name}")

        mask3d = ds.variables['tmask'][:]
        if mask3d.ndim != 4:
            raise ValueError(f"Expected 'tmask' to be 4D, got shape {mask3d.shape}")

        # Extract first layer explicitly (e.g. first vertical or time slice)
        mask3d = mask3d[0, :, :]
        if mask3d.ndim != 3:
            raise ValueError(f"'tmask' slice should be 3D, got shape {mask3d.shape}")

        Mmask = mask3d[0, :, :]
        if Mmask.ndim != 2:
            raise ValueError(f"Expected 2D Mmask, got shape {Mmask.shape}")

        Mfsm = np.where(Mmask == 0)
        Mfsm_3d = np.where(mask3d == 0)

        Mlat = ds.variables['nav_lat'][:]
        Mlon = ds.variables['nav_lon'][:]
        if Mlat.shape != Mmask.shape:
            raise ValueError(f"Shape mismatch: Mlat {Mlat.shape} vs Mmask {Mmask.shape}")
        if Mlon.shape != Mmask.shape:
            raise ValueError(f"Shape mismatch: Mlon {Mlon.shape} vs Mmask {Mmask.shape}")

    return Mmask, Mfsm, Mfsm_3d, Mlat, Mlon
###############################################################################

###############################################################################
def load_dataset(
    year: Union[int, str], 
    IDIR: Union[str, Path]
) -> Tuple[Union[int, str], Optional[xr.Dataset]]:
    """
    Load the NetCDF dataset for a given year from the specified directory.

    Parameters
    ----------
    year : int or str
        Year identifier for the dataset file.
    IDIR : str or Path
        Input directory containing the dataset files.

    Returns
    -------
    Tuple[year, Optional[xr.Dataset]]
        The year and the loaded xarray Dataset if found, otherwise None.

    Raises
    ------
    ValueError
        If the input directory does not exist or is not a directory.
    """
    IDIR = Path(IDIR)
    if not (IDIR.exists() and IDIR.is_dir()):
        raise ValueError(f"Input directory {IDIR} does not exist or is not a directory")

    file_path = IDIR / f"Msst_{year}.nc"

    if file_path.exists():
        print(f"Opening {file_path.name}...")
        try:
            ds = xr.open_dataset(file_path)
            return year, ds
        except Exception as e:
            print(f"Error opening {file_path.name}: {e}")
            return year, None
    else:
        print(f"Warning: {file_path.name} not found!")
        return year, None
###############################################################################

###############################################################################
def unzip_gz_to_file(file_gz: Path, target_file: Path) -> None:
    """
    Unzip a .gz compressed file to a target file.

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
    """
    if not file_gz.exists():
        raise FileNotFoundError(f"File not found: {file_gz}")

    # Ensure the target directory exists
    target_file.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(file_gz, 'rb') as f_in, open(target_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
###############################################################################

###############################################################################
def read_nc_variable_from_unzipped_file(
    file_nc: Path,
    variable_key: str
) -> np.ndarray:
    """
    Read a variable array from a NetCDF file.

    Parameters
    ----------
    file_nc : Path
        Path to the NetCDF (.nc) file.
    variable_key : str
        Name of the variable to read from the file.

    Returns
    -------
    np.ndarray
        The variable data read from the NetCDF file.

    Raises
    ------
    FileNotFoundError
        If the NetCDF file does not exist.
    KeyError
        If the variable_key does not exist in the NetCDF file.
    """
    if not file_nc.exists():
        raise FileNotFoundError(f"NetCDF file not found: {file_nc}")

    with Dataset(file_nc, 'r') as nc:
        if variable_key not in nc.variables:
            raise KeyError(f"Variable '{variable_key}' not found in {file_nc}")
        data = nc.variables[variable_key][:]
    return data
###############################################################################

###############################################################################
def read_nc_variable_from_gz_in_memory(
    file_gz: Path,
    variable_key: str
) -> np.ndarray:
    """
    Read a variable directly from a gzipped NetCDF file in memory using xarray.

    Parameters
    ----------
    file_gz : Path
        Path to the gzipped NetCDF (.nc.gz) file.
    variable_key : str
        Name of the variable to extract from the dataset.

    Returns
    -------
    np.ndarray
        Numpy array of the variable data.

    Raises
    ------
    FileNotFoundError
        If the gzipped file does not exist.
    KeyError
        If the specified variable_key is not found in the dataset.
    """
    if not file_gz.exists():
        raise FileNotFoundError(f"File not found: {file_gz}")

    with gzip.open(file_gz, 'rb') as f_in:
        decompressed_bytes = f_in.read()

    with xr.open_dataset(io.BytesIO(decompressed_bytes)) as ds:
        if variable_key not in ds.variables:
            raise KeyError(f"Variable '{variable_key}' not found in {file_gz}")
        data = ds[variable_key].values

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
    Call the MATLAB interpolation function `Interpolator_v2` via matlab.engine.

    Parameters
    ----------
    varname : str
        Variable name to interpolate.
    data_level : int
        Data level identifier (e.g., depth or layer index).
    input_dir : str or PathLike
        Directory path containing input data files.
    output_dir : str or PathLike
        Directory path where output files will be saved.
    mask_file : str or PathLike
        Path to the mask file required for interpolation.

    Raises
    ------
    RuntimeError
        If MATLAB engine fails to start or the interpolation function errors.
    """
    pkg_root = os.path.dirname(os.path.abspath(__file__))
    matlab_func_folder = pkg_root  # Assuming .m files are in the package root

    print("Adding MATLAB folder to path...")

    try:
        eng = matlab.engine.start_matlab()
    except Exception as e:
        raise RuntimeError(f"Failed to start MATLAB engine: {e}")

    try:
        # Add the folder where the .m files reside
        eng.addpath(matlab_func_folder, nargout=0)

        print("Beginning the interpolation...")

        # Call the MATLAB function
        eng.Interpolator_v2(varname, data_level, input_dir, output_dir, mask_file, nargout=0)
    except Exception as e:
        raise RuntimeError(f"MATLAB interpolation failed: {e}")
    finally:
        eng.quit()
###############################################################################
