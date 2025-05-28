import numpy as np
import os
import gzip
import shutil
from netCDF4 import Dataset
from pathlib import Path

from .time_utils import leapyear
from .utils import infer_years_from_path

###############################################################################
def read_model_data(Dmod, Mfsm, variable_name):
    """
    Reads and processes model CHL or SST data from NetCDF files over multiple years.

    Parameters
    ----------
    Dmod : str or Path
        Base directory containing yearly output subfolders.
    Ybeg : int
        Starting year for data reading (inferred again inside).
    Mfsm : tuple of np.ndarray
        Masking indices (e.g., from np.where) to apply NaNs to model data.
    variable_name : str
        Either 'chl' for chlorophyll or 'sst' for sea surface temperature.

    Returns
    -------
    np.ndarray
        Concatenated array of model data across years with masking applied.

    Raises
    ------
    AssertionError
        If inputs are not of expected types or data files are missing/inconsistent.
    """
    assert isinstance(Dmod, (str, Path)), "Dmod must be a string or Path object"
    assert (
        isinstance(Mfsm, tuple)
        and len(Mfsm) == 2
        and all(isinstance(i, np.ndarray) for i in Mfsm)
    ), "Mfsm must be a tuple of two numpy arrays for indexing"

    if variable_name == 'chl':
        key = 'Chlasat_od'
    elif variable_name == 'sst':
        key = 'sst'
    else:
        raise ValueError("variable_name must be either 'chl' or 'sst'.")

    print("Scanning directory to determine available years...")
    Ybeg, Yend, ysec = infer_years_from_path(Dmod, target_type="folder", pattern=r'output\s*(\d{4})')
    print(f"Found years from {Ybeg} to {Yend}: {ysec}")
    print("-" * 45)

    ymod = Ybeg - 1
    ModData_complete_list = []

    for y in ysec:
        ymod += 1
        ymod_str = str(ymod)
        YDIR = "output" + ymod_str

        # Leap year support
        amileap = 365 + leapyear(ymod)
        assert amileap in [365, 366], f"Leap year calculation failed for year {ymod}"

        # Construct file paths
        # Generalize file search
        folder_path = Path(Dmod, YDIR)
        file_suffix = '_Chl.nc' if variable_name == 'chl' else '_grid_T.nc'

        # Look for .nc and .nc.gz variants
        candidates = list(folder_path.glob(f"*{file_suffix}")) + list(folder_path.glob(f"*{file_suffix}.gz"))
        if not candidates:
            raise FileNotFoundError(f"No matching file for {variable_name.upper()} in {folder_path}")
        ModData_path = candidates[0].with_suffix('') if candidates[0].suffix == '.gz' else candidates[0]
        ModData_pathgz = candidates[0].with_suffix('.nc.gz') if candidates[0].suffix == '' else candidates[0]

        print(f"Obtaining the {variable_name.upper()} data for the year {ymod_str}...")

        # Extract gzipped file if needed
        if not os.path.exists(ModData_path):
            if os.path.exists(ModData_pathgz):
                with gzip.open(ModData_pathgz, 'rb') as f_in, open(ModData_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # Read and validate NetCDF file
        with Dataset(ModData_path, 'r') as nc_file:
            assert key in nc_file.variables, f"{key} variable not found in {ModData_path}"
            ModData_orig = nc_file.variables[key][:]
            assert ModData_orig.shape[0] == amileap, f"Unexpected number of days in {variable_name.upper()} data for year {ymod}"

        # Re-zip file if uncompressed copy was created
        if os.path.exists(ModData_pathgz):
            print("Zipped file already existing")
            os.remove(ModData_path)
        else:
            print("Zipping...")
            with open(ModData_path, 'rb') as f_in, gzip.open(ModData_pathgz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"\033[92m✅ The model {variable_name.upper()} data for the year {ymod_str} has been retrieved!\033[0m")
        print('-' * 45)

        # Mask and accumulate
        if y == Ybeg:
            ModData_row, ModData_col = ModData_orig.shape[1], ModData_orig.shape[2]

        tempo = np.full((ModData_row, ModData_col), np.nan)
        for t in range(amileap):
            tempo[:, :] = ModData_orig[t, :, :]
            tempo[Mfsm] = np.nan
            ModData_orig[t, :, :] = tempo[:, :]

        ModData_complete_list.append(ModData_orig[:amileap, :, :])

    # Concatenate all yearly data into a full dataset
    ModData_complete = np.concatenate(ModData_complete_list, axis=0)

    print("\033[92m✅ Model data fully loaded across all years!\033[0m")
    print('*' * 45)

    return ModData_complete
###############################################################################
