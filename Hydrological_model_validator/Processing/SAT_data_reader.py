import os
import gzip
from netCDF4 import Dataset as ds
import numpy as np
import glob

###############################################################################
def sat_data_loader(data_level, D_sat, varname):

    assert isinstance(data_level, str) and data_level in ['l3', 'l4'], \
        "\033[91m❌ Invalid data level — must be 'l3' or 'l4'\033[0m"

    all_files = sorted(glob.glob(os.path.join(D_sat, '*.gz')))
    data_files = [f for f in all_files if data_level in os.path.basename(f)]

    print(f"Reading satellite data for level '{data_level}'...")
    print(f"\033[91m⚠️ Found {len(data_files)} data files ⚠️\033[0m")

    lon = None
    lat = None
    filepaths = []
    T_orig = []
    data_orig = []

    LE = 0
    total_time_count = 0

    for n, gz_file in enumerate(data_files):
        nc_file = gz_file[:-3]  # remove .gz extension

        # Uncompress if needed
        if not os.path.exists(nc_file):
            with gzip.open(gz_file, 'rb') as f_in, open(nc_file, 'wb') as f_out:
                f_out.write(f_in.read())
            os.remove(gz_file)

        filepaths.append(nc_file)

        with ds(nc_file, 'r') as nc:
            # Read and tile lon/lat only once (first file)
            if lon is None or lat is None:
                if lon is None or lat is None:
                    lon_keys = ['lon', 'longitude']
                    lat_keys = ['lat', 'latitude']

                    # Find lon
                    for lk in lon_keys:
                        if lk in nc.variables:
                            lon_1d = nc.variables[lk][:]
                            break
                        else:
                            raise KeyError("\033[91m❌ No longitude variable ('lon' or 'longitude') found\033[0m")

                    # Find lat
                    for lk in lat_keys:
                        if lk in nc.variables:
                            lat_1d = nc.variables[lk][:]
                            break
                        else:
                            raise KeyError("\033[91m❌ No latitude variable ('lat' or 'latitude') found\033[0m")

                    # Tile 1D arrays to 2D
                    lon = np.tile(lon_1d, (len(lat_1d), 1))
                    lat = np.tile(lat_1d, (len(lon_1d), 1))

            # Read time variable
            time_arr = nc.variables['time'][:]

            # Case-insensitive variable name lookup
            varname_lower = varname.lower()
            vars_lower = {k.lower(): k for k in nc.variables.keys()}

            if varname_lower in vars_lower:
                real_varname = vars_lower[varname_lower]
            elif varname == 'sst':
                # Further check, sometimes CMEMS uses other labels
                varname = 'adjusted_sea_surface_temperature'
                varname_lower = varname.lower()
                real_varname = vars_lower[varname_lower]
            else:
                raise KeyError(f"\033[91m❌ Variable '{varname}' not found in file\033[0m")

            data_arr = nc.variables[real_varname][:]
            data_arr = np.where(data_arr == -999, np.nan, data_arr)

            assert data_arr.ndim == 3, "\033[91m❌ Expected 3D data (time, lat, lon)\033[0m"

            SZTtmp = time_arr.shape[0]
            Ysz, Xsz = data_arr.shape[1], data_arr.shape[2]

            total_time_count += SZTtmp
            print(f"File {n + 1}: {SZTtmp} time points, cumulative: {total_time_count}")

            LS = LE
            LE += SZTtmp

            T_orig.extend(time_arr[:SZTtmp])

            if len(data_orig) < LE:
                data_orig.extend([np.full((Ysz, Xsz), np.nan)] * (LE - len(data_orig)))

            data_orig[LS:LE] = data_arr[:SZTtmp, :, :]

    if lon is None or lat is None:
        raise RuntimeError("\033[91m❌ Longitude or latitude not found in any file\033[0m")

    T_orig = np.array(T_orig)
    data_orig = np.array(data_orig)

    print("*" * 45)
    print("Attempting to merge datasets...")

    assert T_orig.shape[0] == total_time_count, (
        f"\033[91m❌ Merge failed: expected {total_time_count} time points, got {T_orig.shape[0]}\033[0m"
    )
    print("\033[92m✅ The data merging has been successful!\033[0m")
    print("*" * 45)
    
    data_orig[data_orig == -999] = np.nan
    
    if varname == 'sst' or varname == 'adjusted_sea_surface_temperature':
        # Satellite data by CMEMS is usually in kelvin, they need to be converted
        print("Converting the SST data from Kelvin into Celsius...")
        data_orig = data_orig - 273.15
        print("\033[92m✅ SST successfully converted to Celsius!\033[0m")

    return T_orig, data_orig, lon, lat
###############################################################################
