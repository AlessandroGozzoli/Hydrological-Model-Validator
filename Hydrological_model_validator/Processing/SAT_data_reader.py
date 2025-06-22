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
from netCDF4 import Dataset as ds
import numpy as np

# Logging and tracing
import logging
from eliot import start_action, log_message

# Module utilities
from Hydrological_model_validator.Processing.time_utils import Timer
from Hydrological_model_validator.Processing.utils import find_key_variable

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def sat_data_loader(
    data_level: str,
    D_sat: Union[str, Path],
    varname: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load satellite data (chlorophyll or SST) from yearly NetCDF files.

    Parameters
    ----------
    data_level: str
        Value representing the type of data to be handled; must be 'l3' or 'l4'.
    D_sat : Union[str, Path]
        Directory path containing summary output folders.
    varname : str
        'chl' for chlorophyll or 'sst' for sea surface temperature.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays of time, data, longitude, and latitude.

    Raises
    ------
    TypeError, FileNotFoundError, ValueError, KeyError, RuntimeError
        On invalid inputs or missing data files.
    """
    # ===== INPUT VALIDATION BLOCK =====
    if not isinstance(D_sat, (str, Path)):
        raise TypeError("❌ D_sat must be a string or a Path object. ❌")
    D_sat = Path(D_sat)
    if not D_sat.exists():
        raise FileNotFoundError(f"❌ The specified path '{D_sat}' does not exist. ❌")
    if not D_sat.is_dir():
        raise NotADirectoryError(f"❌ The specified path '{D_sat}' is not a directory. ❌")

    if not isinstance(data_level, str):
        raise TypeError("❌ data_level must be a string. ❌")
    data_level = data_level.lower()
    if data_level not in ['l3', 'l4']:
        raise ValueError("❌ Invalid data level — must be 'l3' or 'l4' ❌")

    if not isinstance(varname, str):
        raise TypeError("❌ varname must be a string. ❌")
    varname_lower = varname.lower()
    if varname_lower not in ['chl', 'sst']:
        raise ValueError("❌ varname must be either 'chl' or 'sst' ❌")

    with Timer("sat_data_loader function"):
        with start_action(
            action_type="sat_data_loader function",
            data_level=data_level,
            variable=varname_lower,
            directory=str(D_sat)
        ):
            logging.info(f"Starting satellite data loading for level '{data_level}' and variable '{varname_lower}'.")
            log_message("Satellite data loading started", data_level=data_level, variable=varname_lower)

            # ===== FILE DISCOVERY BLOCK =====
            all_files = sorted(D_sat.glob('*.gz'))
            data_files = [f for f in all_files if data_level in f.name]

            # If no .gz files found, try .nc files instead
            if not data_files:
                all_nc_files = sorted(D_sat.glob('*.nc'))
                data_files = [f for f in all_nc_files if data_level in f.name]

            # If still no files, raise error
            if not data_files:
                raise FileNotFoundError(
                    f"❌ No .gz or .nc data files found in '{D_sat}' for data level '{data_level}'. ❌"
                    )

            print(f"Reading satellite data for level '{data_level}'...")
            print(f"\033[91m⚠️ Found {len(data_files)} data files ⚠️\033[0m")

            # ===== INITIALIZATION BLOCK =====
            lon = None
            lat = None
            T_orig = []
            data_orig_list = []
            total_time_count = 0

            # ===== FILE PROCESSING LOOP =====
            for n, gz_file in enumerate(data_files, start=1):
                nc_file = gz_file.with_suffix('')  # remove .gz extension

                if not nc_file.exists():
                    with gzip.open(gz_file, 'rb') as f_in, open(nc_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(gz_file)  # remove original compressed file after decompression
                    logging.info(f"Decompressed file: {gz_file.name}")

                with ds(nc_file, 'r') as nc:
                    # ===== LONGITUDE & LATITUDE EXTRACTION BLOCK =====
                    if lon is None or lat is None:
                        lon_var = find_key_variable(nc.variables, ['lon', 'longitude'])
                        lat_var = find_key_variable(nc.variables, ['lat', 'latitude'])

                        if lon_var is None or lat_var is None:
                            raise KeyError("❌ Longitude or latitude variable not found in NetCDF file. ❌")

                        lon_1d = nc.variables[lon_var][:]
                        lat_1d = nc.variables[lat_var][:]

                        if lon_1d.ndim != 1 or lat_1d.ndim != 1:
                            raise ValueError("❌ Longitude and latitude variables must be 1D arrays. ❌")

                        lon = np.tile(lon_1d, (len(lat_1d), 1))
                        lat = np.tile(lat_1d, (len(lon_1d), 1))

                    # ===== TIME EXTRACTION BLOCK =====
                    time_arr = nc.variables.get('time')
                    if time_arr is None:
                        raise KeyError("❌ 'time' variable not found in NetCDF file. ❌")
                    time_arr = time_arr[:]

                    # ===== DATA VARIABLE NAME RESOLUTION BLOCK =====
                    vars_lower = {k.lower(): k for k in nc.variables.keys()}

                    if varname_lower in vars_lower:
                        real_varname = vars_lower[varname_lower]
                    elif varname_lower == 'sst':
                        alt_varname = 'adjusted_sea_surface_temperature'
                        if alt_varname in vars_lower:
                            real_varname = vars_lower[alt_varname]
                        else:
                            raise KeyError(
                                f"❌ Variable '{varname_lower}' or '{alt_varname}' not found in file ❌"
                            )
                    else:
                        raise KeyError(f"❌ Variable '{varname_lower}' not found in file ❌")

                    # ===== DATA EXTRACTION & CLEANING BLOCK =====
                    data_arr = nc.variables[real_varname][:]
                    data_arr = np.array(data_arr, dtype=float)
                    data_arr[data_arr == -999] = np.nan

                    if data_arr.ndim != 3:
                        raise ValueError("❌ Expected 3D data (time, lat, lon) ❌")

                    SZTtmp = time_arr.shape[0]
                    total_time_count += SZTtmp
                    print(f"File {n}: {SZTtmp} time points, cumulative: {total_time_count}")

                    T_orig.extend(time_arr)
                    data_orig_list.append(data_arr)

            # ===== FINAL VALIDATION BLOCK =====
            if lon is None or lat is None:
                raise RuntimeError("❌ Longitude or latitude not found in any file ❌")

            T_orig = np.array(T_orig)
            data_orig = np.concatenate(data_orig_list, axis=0)

            print("*" * 45)
            print("Attempting to merge datasets...")

            if T_orig.shape[0] != total_time_count:
                raise ValueError(
                    f"❌ Merge failed: expected {total_time_count} time points, got {T_orig.shape[0]} ❌"
                )
            print("\033[92m✅ The data merging has been successful!\033[0m")
            print("*" * 45)

            # ===== UNIT CONVERSION BLOCK =====
            if varname_lower in ['sst', 'adjusted_sea_surface_temperature']:
                print("Converting the SST data from Kelvin into Celsius...")
                data_orig -= 273.15
                print("\033[92m✅ SST successfully converted to Celsius!\033[0m")

            logging.info(f"Satellite data loading completed successfully: {total_time_count} time points")
            log_message("Satellite data loading completed", total_time_count=total_time_count)

            return T_orig, data_orig, lon, lat
###############################################################################
