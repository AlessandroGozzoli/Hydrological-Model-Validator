import numpy as np
from typing import Tuple, Union, Optional
from netCDF4 import Dataset as ds
from pathlib import Path
import xarray as xr

###############################################################################
def mask_reader(
    BaseDIR: Union[str, Path]
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Reads the model mask and lat/lon data from a NetCDF file.

    Args:
        BaseDIR (str or Path): Base directory containing the 'mesh_mask.nc' file.

    Returns:
        Tuple containing:
            - 2D model mask (np.ndarray),
            - 2D land grid indexes (tuple of arrays),
            - 3D land grid indexes (tuple of arrays),
            - Latitude array (np.ndarray),
            - Longitude array (np.ndarray).
    """
    assert isinstance(BaseDIR, (str, Path)), "BaseDIR must be a string or Path object"
    MASK = Path(BaseDIR) / 'mesh_mask.nc'
    assert MASK.exists(), f"Mask file not found at {MASK}"

    with ds(MASK, 'r') as ncfile:
        for var in ['tmask', 'nav_lat', 'nav_lon']:
            assert var in ncfile.variables, f"'{var}' not found in {MASK.name}"

        mask3d = ncfile.variables['tmask'][:]
        assert mask3d.ndim == 4, f"Expected 'tmask' to be 4D, got {mask3d.shape}"

        mask3d = mask3d.squeeze()
        assert mask3d.ndim == 3, f"'tmask' should be 3D after squeeze, got {mask3d.shape}"

        Mmask = mask3d[0, :, :]
        assert Mmask.ndim == 2, f"Expected Mmask to be 2D, got {Mmask.shape}"

        Mfsm = np.where(Mmask == 0)
        Mfsm_3d = np.where(mask3d == 0)

        Mlat = ncfile.variables['nav_lat'][:]
        Mlon = ncfile.variables['nav_lon'][:]
        assert Mlat.shape == Mmask.shape, "Mlat shape does not match Mmask shape"
        assert Mlon.shape == Mmask.shape, "Mlon shape does not match Mmask shape"

    return Mmask, Mfsm, Mfsm_3d, Mlat, Mlon
###############################################################################

###############################################################################
def load_dataset(year: Union[int, str], IDIR: Union[str, Path]) -> Tuple[Union[int, str], Optional[xr.Dataset]]:
    """
    Load the NetCDF dataset for a given year from the specified directory.

    Args:
        year (int or str): Year identifier for the dataset file.
        IDIR (str or Path): Input directory containing the dataset files.

    Returns:
        Tuple[year, xr.Dataset or None]: The year and the loaded dataset or None if file not found.
    """
    file_path = Path(IDIR) / f"Msst_{year}.nc"
    assert file_path.parent.exists(), f"Input directory {file_path.parent} does not exist"

    if file_path.exists():
        print(f"Opening {file_path.name}...")
        ds = xr.open_dataset(file_path)
        return year, ds
    else:
        print(f"Warning: {file_path.name} not found!")
        return year, None
###############################################################################
