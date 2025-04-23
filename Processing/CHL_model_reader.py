import numpy as np
import os
import gzip
import shutil
from netCDF4 import Dataset
from Leap_year import leapyear
from pathlib import Path

def read_model_chl_data(Dmod, Ybeg, Tspan, Truedays, DinY, Mfsm):
    """Reads and processes model CHL data from netCDF files."""
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
        ie = ib + amileap
        
        Mchlpath = Path(Dmod + YDIR) / f"ADR{ymod_str}new15bb_Chlsat.nc"
        Mchlpathgz = Path(Dmod + YDIR) / f"ADR{ymod_str}new15bb_Chlsat.nc.gz"
        
        print(f"Obtaining the CHL data for the year {ymod_str}...")
        
        if not os.path.exists(Mchlpath):
            if os.path.exists(Mchlpathgz):
                with gzip.open(Mchlpathgz, 'rb') as f_in, open(Mchlpath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        with Dataset(Mchlpath, 'r') as nc_file:
            Mchl_orig = nc_file.variables[fMchl][:]
        
        if os.path.exists(Mchlpathgz):
            print("Zipped file already existing")
            os.remove(Mchlpath)
        else:
            print("Zipping...")
            with open(Mchlpath, 'rb') as f_in, gzip.open(Mchlpathgz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Model CHL data for the year {ymod_str} obtained!")
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
    
    if ie != Truedays:
        print("MODEL CHL DATA INCORRECTLY LOADED")
        print('*'*45)
        return None
    else:
        print("MODEL CHL DATA CORRECTLY LOADED")
        print('*'*45)

    return Mchl_complete, Mchl_orig