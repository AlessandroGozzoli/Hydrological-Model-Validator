import numpy as np
import os
import gzip
import shutil
from netCDF4 import Dataset
from Leap_year import leapyear
from pathlib import Path
import xarray as xr

def read_model_chl_data(Dmod, Ybeg, Tspan, Truedays, DinY, Mfsm):
    """Reads and processes model CHL data from netCDF files."""
    
    assert isinstance(Dmod, (str, Path)), "Dmod must be a string or Path object"
    assert isinstance(Ybeg, int), "Ybeg must be an integer"
    assert isinstance(Tspan, int) and Tspan > 0, "Tspan must be a positive integer"
    assert isinstance(Truedays, int) and Truedays > 0, "Truedays must be a positive integer"
    assert isinstance(DinY, int) and DinY in [365, 366], "DinY must be 365 or 366"
    assert (
        isinstance(Mfsm, tuple)
        and len(Mfsm) == 2
        and all(isinstance(i, np.ndarray) for i in Mfsm)
        ), "Mfsm must be a tuple of two numpy arrays for indexing"

    fMchl = 'Chlasat_od'
    ib = 0
    ie = 0
    ymod = Ybeg - 1
    Mchl_complete = []  # Initialize dynamically
    
    for y in range(Tspan):
        ymod += 1
        ymod_str = str(ymod)
        YDIR = "output" + ymod_str
        
        ib = ie
        amileap = DinY + leapyear(ymod)
        assert amileap in [365, 366], f"Leap year calculation failed for year {ymod}"
        ie = ib + amileap
        
        Mchlpath = Path(Dmod, YDIR) / f"ADR{ymod_str}new15bb_Chlsat.nc"
        Mchlpathgz = Path(Dmod, YDIR) / f"ADR{ymod_str}new15bb_Chlsat.nc.gz"
        
        print(f"Obtaining the CHL data for the year {ymod_str}...")
        
        if not os.path.exists(Mchlpath):
            if os.path.exists(Mchlpathgz):
                with gzip.open(Mchlpathgz, 'rb') as f_in, open(Mchlpath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        with Dataset(Mchlpath, 'r') as nc_file:
            assert fMchl in nc_file.variables, f"{fMchl} variable not found in {Mchlpath}"
            Mchl_orig = nc_file.variables[fMchl][:]
            assert Mchl_orig.shape[0] == amileap, f"Unexpected number of days in CHL data for year {ymod}"
        
        if os.path.exists(Mchlpathgz):
            print("Zipped file already existing")
            os.remove(Mchlpath)
        else:
            print("Zipping...")
            with open(Mchlpath, 'rb') as f_in, gzip.open(Mchlpathgz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"\033[92m✅ The model CHL data for the year {ymod_str} has been retrieved!\033[0m")
        print('-'*45)
        
        if y == 0:
            Mchlrow, Mchlcol = Mchl_orig.shape[1], Mchl_orig.shape[2]
            Mchl_complete = np.zeros((Truedays, Mchlrow, Mchlcol))
        
        tempo = np.full((Mchlrow, Mchlcol), np.nan)
        for t in range(amileap):
            tempo[:, :] = Mchl_orig[t, :, :]
            tempo[Mfsm] = np.nan
            Mchl_orig[t, :, :] = tempo[:, :]
        
        Mchl_complete[ib:ie, :, :] = Mchl_orig[:amileap, :, :]

    assert ie == Truedays, f"Total days mismatch: expected {Truedays}, got {ie}"

    print("\033[92m✅ Model CHL data fully loaded!\033[0m")
    print('*'*45)

    return Mchl_complete

def read_model_sst(Dmod, ysec, Mfsm):
    """
    Reads model SST data across multiple years and returns structured SST data.
    
    Parameters:
        DSST_mod (str): Path to the model SST data directory.
        ysec (list): List of years to iterate over.
    
    Returns:
        dict: Dictionary containing SST data organized by year.
    """
    
    # Input validations
    assert isinstance(Dmod, (str, Path)), "Dmod must be a string or Path object"
    assert isinstance(ysec, (list, tuple)) and all(isinstance(y, int) for y in ysec), "ysec must be a list or tuple of integers"
    assert (
        isinstance(Mfsm, tuple)
        and len(Mfsm) == 2
        and all(isinstance(i, np.ndarray) for i in Mfsm)
    ), "Mfsm must be a tuple of two numpy arrays for indexing"
    
    sst_data = {}  # Dictionary to store SST data by year
    
    for y in ysec:
        Ynow = str(y)
        print(f"Processing year {Ynow}...")
        current_year = str('output' + Ynow)
        
        DSST_mod = os.path.join(Dmod, current_year)
        
        # Construct the file path
        Msstpath = os.path.join(DSST_mod, 
                                f"ADR{Ynow}new_g100_1d_{Ynow}0101_{Ynow}1231_grid_T.nc")
        
        # Generate zipped file path
        Msstpathgz = f"{Msstpath}.gz"
        
        # Unzip if necessary
        if not os.path.exists(Msstpath) and os.path.exists(Msstpathgz):
            with gzip.open(Msstpathgz, 'rb') as f_in:
                with open(Msstpath, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        # Read SST data
        if os.path.exists(Msstpath):
            ds = xr.open_dataset(Msstpath)
            assert 'sst' in ds.variables, f"'sst' variable not found in file {Msstpath}"

            Msst = ds['sst'].values  # Extract SST variable
            assert Msst.ndim == 3, f"Expected 3D SST data, got shape {Msst.shape}"

            for t in range(Msst.shape[0]):
                # Apply mask to latitude/longitude using Mfsm
                Msst[t, Mfsm[0], Mfsm[1]] = np.nan  # Index mask

            sst_data[Ynow] = Msst
            print(f"\033[92m✅ The model SST data for the year {Ynow} has been retrieved!\033[0m")
            print('-'*45)
        else:
            print(f"\033[91m⚠️ Warning: SST file for {Ynow} not found.\033[0m")
            print('-'*45)

    return sst_data