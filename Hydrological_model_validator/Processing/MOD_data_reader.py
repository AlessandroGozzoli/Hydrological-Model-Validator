import numpy as np
import os
import gzip
import shutil
from netCDF4 import Dataset
from pathlib import Path
from typing import Union, Tuple

from .time_utils import leapyear
from .utils import infer_years_from_path

###############################################################################
def read_model_data(
    Dmod: Union[str, Path],
    Mfsm: Tuple[np.ndarray, np.ndarray],
    variable_name: str
) -> np.ndarray:
    """
    Load and mask model data (chlorophyll or SST) from yearly NetCDF files.

    Parameters
    ----------
    Dmod : Union[str, Path]
        Directory path containing yearly output folders.
    Mfsm : Tuple[np.ndarray, np.ndarray]
        Tuple of 2D numpy arrays representing indices to mask (row, col).
    variable_name : str
        'chl' for chlorophyll or 'sst' for sea surface temperature.

    Returns
    -------
    np.ndarray
        Concatenated model data array with masked values set to NaN.

    Raises
    ------
    TypeError, FileNotFoundError, ValueError, KeyError
        On invalid inputs or missing data files.
    """
    Dmod = Path(Dmod)  # Ensure Path object

    # Validate Mfsm
    if (
        not isinstance(Mfsm, tuple)
        or len(Mfsm) != 2
        or not all(isinstance(arr, np.ndarray) for arr in Mfsm)
    ):
        raise TypeError("Mfsm must be a tuple of two numpy arrays for indexing")

    # Determine variable key in NetCDF files
    if variable_name == 'chl':
        key = 'Chlasat_od'
    elif variable_name == 'sst':
        key = 'sst'
    else:
        raise ValueError("variable_name must be either 'chl' or 'sst'.")

    print("Scanning directory to determine available years...")

    Ybeg, Yend, ysec = infer_years_from_path(Dmod, target_type="folder", pattern=r'output(\d{4})')

    print(f"Found years from {Ybeg} to {Yend}: {ysec}")
    print("-" * 45)

    ModData_complete_list = []

    for y in ysec:
        YDIR = f"output{y}"
        folder_path = Dmod / YDIR

        amileap = 365 + leapyear(y)
        if amileap not in (365, 366):
            raise ValueError(f"Leap year calculation failed for year {y}")

        # Find candidate files based on variable
        if variable_name == 'chl':
            candidates = list(folder_path.glob("*_Chl.nc")) + list(folder_path.glob("*_Chl.nc.gz"))
        else:  # sst
            candidates = list(folder_path.glob("*_1d_*_grid_T.nc")) + list(folder_path.glob("*_1d_*_grid_T.nc.gz"))

        if not candidates:
            raise FileNotFoundError(f"No matching file for {variable_name.upper()} in {folder_path}")

        # Prefer uncompressed file if present
        candidate = next((f for f in candidates if f.suffix != '.gz'), candidates[0])

        ModData_path = candidate.with_suffix('') if candidate.suffix == '.gz' else candidate
        ModData_pathgz = candidate if candidate.suffix == '.gz' else candidate.with_suffix('.gz')

        print(f"Obtaining the {variable_name.upper()} data for the year {y}...")

        decompressed = False
        if not ModData_path.exists() and ModData_pathgz.exists():
            with gzip.open(ModData_pathgz, 'rb') as f_in, open(ModData_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            decompressed = True

        with Dataset(ModData_path, 'r') as nc_file:
            if key not in nc_file.variables:
                raise KeyError(f"{key} variable not found in {ModData_path}")
            ModData_orig = nc_file.variables[key][:]
            if ModData_orig.shape[0] != amileap:
                raise ValueError(
                    f"Unexpected number of days in {variable_name.upper()} data for year {y}, "
                    f"expected {amileap} got {ModData_orig.shape[0]}"
                )

        if decompressed:
            os.remove(ModData_path)
        else:
            print("No decompression needed, skipping re-zipping.")

        print(f"\033[92m✅ The model {variable_name.upper()} data for the year {y} has been retrieved!\033[0m")
        print('-' * 45)

        # Mask specified indices with NaN for all days
        ModData_orig[:, Mfsm[0], Mfsm[1]] = np.nan

        ModData_complete_list.append(ModData_orig[:amileap])

    ModData_complete = np.concatenate(ModData_complete_list, axis=0)

    print("\033[92m✅ Model data fully loaded across all years!\033[0m")
    print('*' * 45)

    return ModData_complete
###############################################################################
