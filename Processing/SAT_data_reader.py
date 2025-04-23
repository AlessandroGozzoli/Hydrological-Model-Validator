import os
import gzip
from netCDF4 import Dataset as ds
import numpy as np

# Retrieve some costants to be used as counters
from Costants import (
                      LS,
                      LE,
                      total_SZTtmp
                      )

def sat_chldata(total_SZTtmp, LE, LS, nf, ffrag1, chlfstart, chlfend, chldlev, ffrag2, DCHL_sat, Schl2r):
    assert nf > 0, "❌ 'nf' must be greater than 0 — no dataset instances provided"
    assert chldlev in ['l3', 'l4'], "❌ Invalid data level — must be 'l3' or 'l4'"

    Slon = None
    Slat = None
    Schlpath = []       # To store file paths
    T_orig = []         # To store time data
    Schl_orig = []      # To store chlorophyll data

    print("Reading the satellite chlorophyll data...")
    print(f"\033[91m⚠️ The dataset is divided into {nf} instances ⚠️\033[0m")

    for n in range(nf):
        fSchl = f"{ffrag1}{chlfstart[n]}-{chlfend[n]}-{chldlev}-{ffrag2}.nc"
        fSchlgz = f"{fSchl}.gz"
        Schlpath_n = os.path.join(DCHL_sat, fSchl)

        # Uncompress if .nc is missing but .gz is present
        if not os.path.exists(Schlpath_n):
            zf = os.path.join(DCHL_sat, fSchlgz)
            assert os.path.exists(zf), f"❌ Neither {Schlpath_n} nor {zf} found"
            with gzip.open(zf, 'rb') as f_in:
                with open(Schlpath_n, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(zf)

        Schlpath.append(Schlpath_n)

        # Read lon/lat (only once)
        if n == 0:
            with ds(Schlpath_n, 'r') as ncfile:
                assert 'lon' in ncfile.variables, "❌ 'lon' variable not found"
                assert 'lat' in ncfile.variables, "❌ 'lat' variable not found"
                xc = ncfile.variables['lon'][:]
                yc = ncfile.variables['lat'][:]
                Slon = np.tile(xc, (len(yc), 1))
                Slat = np.tile(yc, (len(xc), 1))

        with ds(Schlpath_n, 'r') as nc_file:
            assert Schl2r in nc_file.variables, f"❌ '{Schl2r}' variable not found in file"
            Ttmp = nc_file.variables['time'][:]
            Dtmp = nc_file.variables[Schl2r][:]
            Dtmp[Dtmp == -999] = np.nan

            assert Dtmp.ndim == 3, "❌ Expected 3D chlorophyll data (time, lat, lon)"
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
        f"❌ Merge failed: expected {total_SZTtmp} days, got {SZT_orig}"
    )
    print("\033[92m✅ The data merging has been successful!\033[0m")
    print("*" * 45)

    Schl_orig[Schl_orig == -999] = np.nan  # Just in case

    return T_orig, Schl_orig, Slon, Slat, Schlpath