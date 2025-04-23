import os
import gzip
from netCDF4 import Dataset as ds
import numpy as np

from Costants import (
                      LS,
                      LE,
                      total_SZTtmp
                      )

def sat_chldata(total_SZTtmp, LE, LS, nf, ffrag1, chlfstart, chlfend, chldlev, ffrag2, DCHL_sat, Schl2r):
    Slon = None
    Slat = None
    Schlpath = []  # Initialize the list to store paths
    T_orig = []  # List to store time data
    Schl_orig = []  # List to store chlorophyll data
    
    print("Reading the satellite chlorofille data...")
    print(f"The dataset is divided into {nf} instances...")

    for n in range(nf):
        # Compose file name
        fSchl = f"{ffrag1}{chlfstart[n]}-{chlfend[n]}-{chldlev}-{ffrag2}.nc"
        
        # Zipped file
        fSchlgz = f"{fSchl}.gz"
        
        # Path to data
        Schlpath_n = os.path.join(DCHL_sat, fSchl)
        
        # Uncompress (if necessary)
        if not os.path.exists(Schlpath_n):
            zf = os.path.join(DCHL_sat, fSchlgz)
            with gzip.open(zf, 'rb') as f_in:
                with open(Schlpath_n, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(zf)
        
        # Append path to Schlpath
        Schlpath.append(Schlpath_n)
        
        # Extract CHL satellite data (lat & lon) (executed only once)
        if n == 0:
            with ds(Schlpath_n, 'r') as ncfile:
                # Read lon and lat
                xc = ncfile.variables['lon'][:]
                yc = ncfile.variables['lat'][:]
                
                # Convert to 2D arrays
                Slon = np.tile(xc, (len(yc), 1))  # Each row is the same longitude values
                Slat = np.tile(yc, (len(xc), 1))  # Each column is the same latitude values
        
        # Read time and chl data for the current file
        with ds(Schlpath_n, 'r') as nc_file:
            # Get time and chl data
            Ttmp = nc_file.variables['time'][:]
            Dtmp = nc_file.variables[Schl2r][:]
            Dtmp[Dtmp == -999] = np.nan
            
            SZTtmp = Ttmp.shape[0]  # Get the size of SZTtmp for this iteration
            total_SZTtmp += SZTtmp  # Add to cumulative total

            print(f"Number of days in instance {n+1}: {SZTtmp}, Cumulative = {total_SZTtmp}")

            # Get dimensions
            SZTtmp = Ttmp.shape[0]
            Ysz = Dtmp.shape[1]
            Xsz = Dtmp.shape[2]

            # Update LS and LE indices
            LS = LE
            LE = LE + SZTtmp

            # Transfer time and chl data into a single array
            T_orig.extend(Ttmp[:SZTtmp])
            
            # Extend Schl_orig only if needed
            if len(Schl_orig) < LE:
                Schl_orig.extend([np.full((Ysz, Xsz), np.nan)] * (LE - len(Schl_orig)))

            # Assign the chlorophyll data in the correct range
            Schl_orig[LS:LE] = Dtmp[:SZTtmp, :, :]

    # Convert lists to numpy arrays for easier manipulation
    T_orig = np.array(T_orig)
    Schl_orig = np.array(Schl_orig)
    
    # -----TOTAL DAYS IN THE MERGED FILE-----
    SZT_orig = T_orig.shape[0]  # Get the number of time steps
    
    print('*'*45)

    # -----CHECK THE MERGING-----
    if SZT_orig == total_SZTtmp:
        string = "Data merging is OK"
        print(string)
    else:
        string = "something wrong in merging data....."
        print(string)
    
    print('*'*45)
    
    Schl_orig[Schl_orig == -999]=np.nan

    return T_orig, Schl_orig, Slon, Slat, Schlpath