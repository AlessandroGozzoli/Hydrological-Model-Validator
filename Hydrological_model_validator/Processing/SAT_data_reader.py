import os
import gzip
from netCDF4 import Dataset as ds
import numpy as np
import xarray as xr

def sat_chldata(total_SZTtmp, LE, LS, nf, ffrag1, chlfstart, chlfend, chldlev, ffrag2, DCHL_sat, Schl2r):
    assert nf > 0, "\033[91m❌ 'nf' must be greater than 0 — no dataset instances provided\033[0m"
    assert chldlev in ['l3', 'l4'], "\033[91m❌ Invalid data level — must be 'l3' or 'l4'\033[0m"

    Slon = None
    Slat = None
    Schlpath = []       # To store file paths
    T_orig = []         # To store time data
    Schl_orig = []      # To store chlorophyll data
    
    LE=0
    LS=0
    total_SZTtmp=0

    print("Reading the satellite chlorophyll data...")
    print(f"\033[91m⚠️ The dataset is divided into {nf} instances ⚠️\033[0m")

    for n in range(nf):
        fSchl = f"{ffrag1}{chlfstart[n]}-{chlfend[n]}-{chldlev}-{ffrag2}.nc"
        fSchlgz = f"{fSchl}.gz"
        Schlpath_n = os.path.join(DCHL_sat, fSchl)

        # Uncompress if .nc is missing but .gz is present
        if not os.path.exists(Schlpath_n):
            zf = os.path.join(DCHL_sat, fSchlgz)
            assert os.path.exists(zf), f"\033[91m❌ Neither {Schlpath_n} nor {zf} found\033[0m"
            with gzip.open(zf, 'rb') as f_in:
                with open(Schlpath_n, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(zf)

        Schlpath.append(Schlpath_n)

        # Read lon/lat (only once)
        if n == 0:
            with ds(Schlpath_n, 'r') as ncfile:
                assert 'lon' in ncfile.variables, "\033[91m❌ 'lon' variable not found\033[0m"
                assert 'lat' in ncfile.variables, "\033[91m❌ 'lat' variable not found\033[0m"
                xc = ncfile.variables['lon'][:]
                yc = ncfile.variables['lat'][:]
                Slon = np.tile(xc, (len(yc), 1))
                Slat = np.tile(yc, (len(xc), 1))

        with ds(Schlpath_n, 'r') as nc_file:
            assert Schl2r in nc_file.variables, f"\033[91m❌ '{Schl2r}' variable not found in file\033[0m"
            Ttmp = nc_file.variables['time'][:]
            Dtmp = nc_file.variables[Schl2r][:]
            Dtmp[Dtmp == -999] = np.nan

            assert Dtmp.ndim == 3, "\033[91m❌ Expected 3D chlorophyll data (time, lat, lon)\033[0m"
            SZTtmp = Ttmp.shape[0]
            Ysz, Xsz = Dtmp.shape[1], Dtmp.shape[2]

            total_SZTtmp += SZTtmp
            print(f"Instance {n+1}: {SZTtmp} days, Cumulative: {total_SZTtmp}")

            LS = LE
            LE += SZTtmp

            T_orig.extend(Ttmp[:SZTtmp])

            if len(Schl_orig) < LE:
                Schl_orig.extend([np.full((Ysz, Xsz), np.nan)] * (LE - len(Schl_orig)))

            Schl_orig[LS:LE] = Dtmp[:SZTtmp, :, :]

    # Convert to numpy arrays
    T_orig = np.array(T_orig)
    Schl_orig = np.array(Schl_orig)

    SZT_orig = T_orig.shape[0]
    print("*" * 45)
    print("Attempting to merge datasets...")

    # Assert successful merge
    assert SZT_orig == total_SZTtmp, (
        f"\033[91m❌ Merge failed: expected {total_SZTtmp} days, got {SZT_orig}\033[0m"
    )
    print("\033[92m✅ The data merging has been successful!\033[0m")
    print("*" * 45)

    Schl_orig[Schl_orig == -999] = np.nan  # Just in case

    return T_orig, Schl_orig, Slon, Slat, Schlpath

def read_sst_satellite_data(DSST_sat, Truedays):
    """
    Reads SST satellite data from a NetCDF file, extracts the required time period,
    and converts temperatures from Kelvin to Celsius.
    
    Parameters:
    DSST_sat (str): Path to the directory containing SST data.
    Truedays (int): Expected number of days in the dataset.
    
    Returns:
    Ssst_orig (numpy.ndarray): SST data in Celsius for the selected time period.
    """
    file_name = "ADR_Tsat_masked-all.nc"
    file_path = os.path.join(DSST_sat, file_name)
    file_gz_path = file_path + ".gz"
    
    # Uncompress if necessary
    if not os.path.exists(file_path) and os.path.exists(file_gz_path):
        os.system(f"gunzip {file_gz_path}")
    
    # Check file existence
    assert os.path.exists(file_path), f"\033[91m❌ File not found: {file_path}\033[0m"

    # Open NetCDF file
    ds = xr.open_dataset(file_path)
    
    # Read SST data and time
    Ssst_all = ds["analysed_sst"].values
    tsst = ds["time"].values  # Already in datetime64 format

    # Make sure SST and time arrays match in the first dimension
    assert Ssst_all.shape[0] == len(tsst), "\033[91m❌ Mismatch in SST and time dimensions.\033[0m"

    # Convert reference timestamps to datetime64
    start_date = np.datetime64("2000-01-01T00:00:00")
    end_date = np.datetime64("2009-12-31T00:00:00")  # Adjust as needed
    
    # Make sure start and end dates exist in time series
    assert start_date in tsst, f"\033[91m❌ Start date {start_date} not found in time vector.\033[0m"
    assert end_date in tsst, f"\033[91m❌ End date {end_date} not found in time vector.\033[0m"
    
    # Slice the appropriate section
    Ip = np.where(tsst == start_date)[0][0]
    Ep = np.where(tsst == end_date)[0][0]
    Tdayssst = (Ep - Ip) + 1

    # Assert on time length match
    assert Tdayssst == Truedays, (
        f"\033[91m❌ Mismatch in expected days.\n\033[0m"
        f"Expected: {Truedays}, Got: {Tdayssst}"
    )
    print("\033[92m✅ The selected timeframe matches the CHL!\033[0m")
    
    # Extract and convert data
    Ssst_K = Ssst_all[Ip:Ep + 1, :, :]
    print("\033[92m✅ Satellite SST data obtained!\033[0m")
    print("-" * 45)

    print("Converting the SST data from Kelvin into Celsius...")
    K2C = -273.15
    Ssst_orig = Ssst_K + K2C
    print("\033[92m✅ SST successfully converted to Celsius!\033[0m")
    print("-" * 45)

    return Ssst_orig