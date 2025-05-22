import numpy as np
from typing import Tuple, Union, Optional
import netCDF4 as nc
from pathlib import Path
import xarray as xr

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
    AssertionError
        If the file does not exist, or required variables are missing, or dimensions do not match expected shapes.
    """
    # Validate input type
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

        # Remove singleton dimensions
        mask3d = np.squeeze(mask3d)
        if mask3d.ndim != 3:
            raise ValueError(f"'tmask' should be 3D after squeeze, got shape {mask3d.shape}")

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
        ds = xr.open_dataset(file_path)
        return year, ds
    else:
        print(f"Warning: {file_path.name} not found!")
        return year, None

###############################################################################
