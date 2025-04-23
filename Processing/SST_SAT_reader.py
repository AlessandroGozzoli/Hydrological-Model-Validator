import os
import numpy as np
import xarray as xr

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
    
    # Open NetCDF file
    ds = xr.open_dataset(file_path)
    
    # Read SST data and time
    Ssst_all = ds["analysed_sst"].values
    tsst = ds["time"].values  # Already in datetime64 format
    
    # Convert reference timestamps to datetime64
    start_date = np.datetime64("2000-01-01T00:00:00")
    end_date = np.datetime64("2009-12-31T00:00:00")  # Adjusted based on description
    
    # Select time period, the full dataframe goes up to 2018
    # it needs to be sliced down to 31/12/2009
    Ip = np.where(tsst == start_date)[0][0]
    Ep = np.where(tsst == end_date)[0][0]
    Tdayssst = (Ep - Ip) + 1
    
    # Further check
    if Tdayssst != Truedays:
        print("Problem in the timeframe selected!")
        print(f"It should be: {Truedays}")
        print(f"Instead, it is: {Tdayssst}")
        dday = Tdayssst - Truedays
        flag = "in excess" if dday > 0 else "missing"
        print(f"There are {dday} days {flag}")
    else:
        print("The selected timeframe matches the CHL!")
    
    # Extract the required SST period
    Ssst_K = Ssst_all[Ip:Ep+1, :, :]
    print("Satellite SST data obtained!")
    print("-"*45)
    
    print("Attempting to convert the SST data from Kelvin into Celsius...")
    # Convert Kelvin to Celsius
    K2C = -273.15
    Ssst_orig = Ssst_K + K2C
    print("The satellite SST has been successfully converted into Celsius!")
    print("-"*45)
    
    return Ssst_orig