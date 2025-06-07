import os
import gzip
from netCDF4 import Dataset as ds
import numpy as np
from typing import Union, Tuple
from pathlib import Path
import shutil

from .utils import find_key_variable

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
    # Validate directory path for satellite data and basic parameter checks

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

    # ===== FILE DISCOVERY BLOCK =====
    # Find and filter compressed files matching the data_level

    all_files = sorted(D_sat.glob('*.gz'))
    data_files = [f for f in all_files if data_level in f.name]

    if not data_files:
        raise FileNotFoundError(f"❌ No .gz data files found in '{D_sat}' for data level '{data_level}'. ❌")

    print(f"Reading satellite data for level '{data_level}'...")
    print(f"\033[91m⚠️ Found {len(data_files)} data files ⚠️\033[0m")

    # ===== INITIALIZATION BLOCK =====
    # Initialize variables for storing lon, lat, time, and data

    lon = None
    lat = None
    T_orig = []
    data_orig_list = []
    total_time_count = 0

    # ===== FILE PROCESSING LOOP =====
    # Loop through each file: decompress if needed, extract variables, and accumulate data

    for n, gz_file in enumerate(data_files, start=1):
        nc_file = gz_file.with_suffix('')  # remove .gz extension

        # Decompress only if uncompressed file does not exist
        if not nc_file.exists():
            with gzip.open(gz_file, 'rb') as f_in, open(nc_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_file)  # remove original compressed file after decompression

        with ds(nc_file, 'r') as nc:
            # ===== LONGITUDE & LATITUDE EXTRACTION BLOCK =====
            # Extract lon/lat once, ensure 1D arrays and tile to 2D grid shape

            if lon is None or lat is None:
                lon_var = find_key_variable(nc.variables, ['lon', 'longitude'])
                lat_var = find_key_variable(nc.variables, ['lat', 'latitude'])

                if lon_var is None or lat_var is None:
                    raise KeyError("❌ Longitude or latitude variable not found in NetCDF file. ❌")

                lon_1d = nc.variables[lon_var][:]
                lat_1d = nc.variables[lat_var][:]

                if lon_1d.ndim != 1 or lat_1d.ndim != 1:
                    raise ValueError("❌ Longitude and latitude variables must be 1D arrays. ❌")

                # Create 2D grid by tiling 1D arrays to match expected shape (lat x lon)
                lon = np.tile(lon_1d, (len(lat_1d), 1))
                lat = np.tile(lat_1d, (len(lon_1d), 1))

            # ===== TIME EXTRACTION BLOCK =====
            # Extract time variable for the current file

            time_arr = nc.variables.get('time')
            if time_arr is None:
                raise KeyError("❌ 'time' variable not found in NetCDF file. ❌")
            time_arr = time_arr[:]

            # ===== DATA VARIABLE NAME RESOLUTION BLOCK =====
            # Resolve real variable name ignoring case; handle SST special cases

            vars_lower = {k.lower(): k for k in nc.variables.keys()}

            if varname_lower in vars_lower:
                real_varname = vars_lower[varname_lower]
            elif varname_lower == 'sst':
                alt_varname = 'adjusted_sea_surface_temperature'
                if alt_varname in vars_lower:
                    real_varname = vars_lower[alt_varname]
                else:
                    raise KeyError(
                        f"❌ Variable '{varname}' or '{alt_varname}' not found in file ❌"
                    )
            else:
                raise KeyError(
                    f"❌ Variable '{varname}' not found in file ❌"
                )

            # ===== DATA EXTRACTION & CLEANING BLOCK =====
            # Extract data array, convert to float, mask invalid fill values (-999) as NaN

            data_arr = nc.variables[real_varname][:]
            data_arr = np.array(data_arr, dtype=float)  # force float type for NaN assignment
            data_arr[data_arr == -999] = np.nan

            # Verify data shape matches expected 3D (time, lat, lon)
            if data_arr.ndim != 3:
                raise ValueError("❌ Expected 3D data (time, lat, lon) ❌")

            # Update counters and accumulate data/time
            SZTtmp = time_arr.shape[0]
            total_time_count += SZTtmp
            print(f"File {n}: {SZTtmp} time points, cumulative: {total_time_count}")

            T_orig.extend(time_arr)
            data_orig_list.append(data_arr)

    # ===== FINAL VALIDATION BLOCK =====
    # Ensure longitude and latitude were found and accumulated data/time sizes match

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
    # Convert SST data from Kelvin to Celsius, if applicable

    if varname_lower in ['sst', 'adjusted_sea_surface_temperature']:
        print("Converting the SST data from Kelvin into Celsius...")
        data_orig -= 273.15
        print("\033[92m✅ SST successfully converted to Celsius!\033[0m")

    return T_orig, data_orig, lon, lat
###############################################################################
