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
    D_sat = Path(D_sat)

    if data_level not in ['l3', 'l4']:
        raise ValueError("\033[91m❌ Invalid data level — must be 'l3' or 'l4'\033[0m")

    all_files = sorted(D_sat.glob('*.gz'))
    data_files = [f for f in all_files if data_level in f.name]

    print(f"Reading satellite data for level '{data_level}'...")
    print(f"\033[91m⚠️ Found {len(data_files)} data files ⚠️\033[0m")

    lon = None
    lat = None
    T_orig = []
    data_orig_list = []
    total_time_count = 0

    for n, gz_file in enumerate(data_files, start=1):
        nc_file = gz_file.with_suffix('')  # remove .gz extension

        if not nc_file.exists():
            with gzip.open(gz_file, 'rb') as f_in, open(nc_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_file)

        with ds(nc_file, 'r') as nc:
            if lon is None or lat is None:
                lon_var = find_key_variable(nc.variables, ['lon', 'longitude'])
                lat_var = find_key_variable(nc.variables, ['lat', 'latitude'])

                lon_1d = nc.variables[lon_var][:]
                lat_1d = nc.variables[lat_var][:]

                lon = np.tile(lon_1d, (len(lat_1d), 1))
                lat = np.tile(lat_1d, (len(lon_1d), 1))

            time_arr = nc.variables['time'][:]

            vars_lower = {k.lower(): k for k in nc.variables.keys()}
            varname_lower = varname.lower()

            if varname_lower in vars_lower:
                real_varname = vars_lower[varname_lower]
            elif varname_lower == 'sst':
                alt_varname = 'adjusted_sea_surface_temperature'
                if alt_varname in vars_lower:
                    real_varname = vars_lower[alt_varname]
                else:
                    raise KeyError(
                        f"\033[91m❌ Variable '{varname}' or '{alt_varname}' not found in file\033[0m"
                    )
            else:
                raise KeyError(
                    f"\033[91m❌ Variable '{varname}' not found in file\033[0m"
                )

            data_arr = nc.variables[real_varname][:]
            data_arr = np.array(data_arr, dtype=float)  # force float type
            data_arr[data_arr == -999] = np.nan

            if data_arr.ndim != 3:
                raise ValueError("\033[91m❌ Expected 3D data (time, lat, lon)\033[0m")

            SZTtmp = time_arr.shape[0]
            total_time_count += SZTtmp
            print(f"File {n}: {SZTtmp} time points, cumulative: {total_time_count}")

            T_orig.extend(time_arr)
            data_orig_list.append(data_arr)

    if lon is None or lat is None:
        raise RuntimeError("\033[91m❌ Longitude or latitude not found in any file\033[0m")

    T_orig = np.array(T_orig)
    data_orig = np.concatenate(data_orig_list, axis=0)

    print("*" * 45)
    print("Attempting to merge datasets...")

    if T_orig.shape[0] != total_time_count:
        raise ValueError(
            f"\033[91m❌ Merge failed: expected {total_time_count} time points, got {T_orig.shape[0]}\033[0m"
        )
    print("\033[92m✅ The data merging has been successful!\033[0m")
    print("*" * 45)

    if varname_lower in ['sst', 'adjusted_sea_surface_temperature']:
        print("Converting the SST data from Kelvin into Celsius...")
        data_orig -= 273.15
        print("\033[92m✅ SST successfully converted to Celsius!\033[0m")

    return T_orig, data_orig, lon, lat
###############################################################################
