import os
import gzip
import shutil
import xarray as xr
import numpy as np

from pathlib import Path

def read_model_sst(Tspan, ysec, Dmod, Sat_sst, Mfsm):
    """
    Reads and processes SST model data for multiple years, applying masking 
    consistent with satellite SST data and handling missing values.
    
    Parameters:
    - Tspan: number of years to process
    - ysec: list or array of years (e.g., [1998, 1999, ..., 2020])
    - DSST_mod: base path to model SST files
    - Ssst_orig: full satellite SST data, shape (Truedays, lat, lon)
    - Mfsm: indices (boolean array or mask) where values should be NaN’d
    
    Returns:
    - BASSTmod: list of daily mean SST values from the model
    - BASSTsat: list of corresponding satellite SST values
    """
    DafterD = 0
    BASSTmod = []
    BASSTsat = []

    for y in range(Tspan):
        Ynow = str(ysec[y])
        print(f"Processing year {Ynow}")
        
        YDIR = 'output' + str(Ynow)
        DSST_mod = Path(Dmod, YDIR)

        # File path construction
        fname = f"{DSST_mod}/ADR{Ynow}new_g100_1d_{Ynow}0101_{Ynow}1231_grid_T.nc"
        fname_gz = fname + ".gz"

        # Unzip if not already unzipped
        if not os.path.exists(fname) and os.path.exists(fname_gz):
            print(f"Unzipping {fname_gz}")
            with gzip.open(fname_gz, 'rb') as f_in:
                with open(fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # Load SST data
        ds = xr.open_dataset(fname)
        Msst_orig = ds['sst'].values  # Shape: (days, lat, lon)
        ydays = Msst_orig.shape[0]

        for d in range(ydays):
            DafterD += 1
            Msst = Msst_orig[d, :, :]
            Ssst = Sat_sst[DafterD - 1, :, :]

            # Find NaNs in satellite SST
            Ssstfsm = np.isnan(Ssst)

            # Apply NaNs based on Mfsm and satellite NaNs
            Msst[Mfsm] = np.nan
            Msst[Ssstfsm] = np.nan
            Ssst[Mfsm] = np.nan

            # Store daily mean values
            BASSTmod.append(np.nanmean(Msst))
            BASSTsat.append(np.nanmean(Ssst))

    return BASSTmod, BASSTsat