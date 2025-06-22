###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Standard library imports
import os
import gzip
import shutil
from pathlib import Path
from typing import Union, Tuple

# Third-party libraries
import numpy as np
from netCDF4 import Dataset

# Logging and tracing
import logging
from eliot import start_action, log_message

# Module utilities
from Hydrological_model_validator.Processing.time_utils import Timer, leapyear
from Hydrological_model_validator.Processing.utils import infer_years_from_path

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
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

    # ===== INPUT VALIDATION BLOCK =====
    # Ensure that inputs are of expected types and point to valid locations

    # Validate Dmod
    if not isinstance(Dmod, (str, Path)):
        raise TypeError("❌ Dmod must be a string or a Path object. ❌")
    Dmod = Path(Dmod)
    if not Dmod.exists():
        raise FileNotFoundError(f"❌ The specified path '{Dmod}' does not exist. ❌")
    if not Dmod.is_dir():
        raise NotADirectoryError(f"❌ The specified path '{Dmod}' is not a directory. ❌")

    # Validate Mfsm
    if (
        not isinstance(Mfsm, tuple)
        or len(Mfsm) != 2
        or not all(isinstance(arr, np.ndarray) for arr in Mfsm)
    ):
        raise TypeError("❌ Mfsm must be a tuple of two numpy arrays for indexing. ❌")
    if Mfsm[0].shape != Mfsm[1].shape:
        raise ValueError("❌ The two arrays in Mfsm must have the same shape. ❌")

    # Validate variable_name
    if variable_name == 'chl':
        key = 'Chlasat_od'
    elif variable_name == 'sst':
        key = 'sst'
    else:
        raise ValueError("❌ variable_name must be either 'chl' or 'sst'. ❌")

    with Timer("read_model_data function"):
        with start_action(
            action_type="read_model_data function",
            directory=str(Dmod),
            variable=variable_name
        ):
            # ===== YEAR RANGE INFERENCE BLOCK =====
            # Determine years from folder structure based on naming pattern

            print("Scanning directory to determine available years...")
            Ybeg, Yend, ysec = infer_years_from_path(Dmod, target_type="folder", pattern=r'output(\d{4})')
            print(f"Found years from {Ybeg} to {Yend}: {ysec}")
            print("-" * 45)
            log_message("Year range inferred", start_year=Ybeg, end_year=Yend, years_found=len(ysec))
            logging.info(f"Year range inferred from directory: {Ybeg}-{Yend}")

            ModData_complete_list = []

            # ===== YEARLY FILE HANDLING BLOCK =====
            # Loop over each year to extract, validate, and collect model data

            for y in ysec:
                YDIR = f"output{y}"
                folder_path = Dmod / YDIR

                # Determine number of days in the year (consider leap years)
                amileap = 365 + leapyear(y)
                if amileap not in (365, 366):
                    raise ValueError(f"❌ Leap year calculation failed for year {y} ❌")

                # Find candidate files based on variable
                if variable_name == 'chl':
                    candidates = list(folder_path.glob("*_Chl.nc")) + list(folder_path.glob("*_Chl.nc.gz"))
                else:
                    candidates = list(folder_path.glob("*_1d_*_grid_T.nc")) + list(folder_path.glob("*_1d_*_grid_T.nc.gz"))

                if not candidates:
                    raise FileNotFoundError(f"❌ No matching file for {variable_name.upper()} in {folder_path} ❌")

                # Prefer uncompressed file if available
                candidate = next((f for f in candidates if f.suffix != '.gz'), candidates[0])

                ModData_path = candidate.with_suffix('') if candidate.suffix == '.gz' else candidate
                ModData_pathgz = candidate if candidate.suffix == '.gz' else candidate.with_suffix('.gz')

                print(f"Obtaining the {variable_name.upper()} data for the year {y}...")
                logging.info(f"Reading {variable_name.upper()} data for year {y}")
                log_message("Reading yearly data", year=y, variable=variable_name)

                # ===== DECOMPRESSION BLOCK =====
                # Decompress gz file only if the uncompressed version is missing

                decompressed = False
                if not ModData_path.exists() and ModData_pathgz.exists():
                    with gzip.open(ModData_pathgz, 'rb') as f_in, open(ModData_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    decompressed = True
                    logging.info(f"Decompressed file for year {y}")

                # ===== DATA EXTRACTION BLOCK =====
                # Open NetCDF, extract data by key, validate shape

                with Dataset(ModData_path, 'r') as nc_file:
                    if key not in nc_file.variables:
                        raise KeyError(f"{key} variable not found in {ModData_path}")
                    ModData_orig = nc_file.variables[key][:]
                    if ModData_orig.shape[0] != amileap:
                        raise ValueError(
                            f"❌ Unexpected number of days in {variable_name.upper()} data for year {y}, ❌"
                            f"❌ expected {amileap} got {ModData_orig.shape[0]} ❌"
                        )

                # Clean up decompressed file after use
                if decompressed:
                    os.remove(ModData_path)
                else:
                    print("No decompression needed, skipping re-zipping.")
                    logging.info("No decompression needed")

                print(f"\033[92m✅ The model {variable_name.upper()} data for the year {y} has been retrieved!\033[0m")
                print('-' * 45)
                log_message("Yearly data retrieved", year=y, variable=variable_name)

                # ===== MASKING BLOCK =====
                # Apply NaN to specified mask indices across all days

                ModData_orig[:, Mfsm[0], Mfsm[1]] = np.nan

                # Store valid year data for final concatenation
                ModData_complete_list.append(ModData_orig[:amileap])

            # ===== FINAL CONCATENATION BLOCK =====
            # Combine data across all years into a single array

            ModData_complete = np.concatenate(ModData_complete_list, axis=0)

            print("\033[92m✅ Model data fully loaded across all years!\033[0m")
            print('*' * 45)
            logging.info("✅ Model data fully loaded across all years!")
            log_message("Model data loaded", total_years=len(ysec), total_days=ModData_complete.shape[0])

            return ModData_complete
###############################################################################
