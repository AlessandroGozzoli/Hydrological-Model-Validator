import os
import gzip
import xarray as xr
import numpy as np

def read_model_sst_yearly(Dmod, ysec, Mfsm):
    """
    Reads model SST data across multiple years and returns structured SST data.
    
    Parameters:
        DSST_mod (str): Path to the model SST data directory.
        ysec (list): List of years to iterate over.
    
    Returns:
        dict: Dictionary containing SST data organized by year.
    """
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
            Msst = ds['sst'].values  # Extract SST variable
            
            for t in range(Msst.shape[0]):
                # Apply mask to latitude/longitude using Mfsm (similar to the CHL case)
                Msst[t, Mfsm[0], Mfsm[1]] = np.nan  # Assuming Mfsm is a tuple with lat/lon indices
            
            sst_data[Ynow] = Msst
            print(f"Got the model SST data for the year {Ynow}!")
            print('-'*45)
        else:
            print(f"Warning: SST file for {Ynow} not found.")
            print('-'*45)
    
    return sst_data