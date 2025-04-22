### Libraries ###
import os
import scipy.io
import xarray as xr
import pandas as pd

# Importing the functions from Leap_year.py
from Leap_year import true_time_series_length

# Importing the functions form CHL_array.py
from CHL_array import chldata_into_array

from SST_SAT_reader import read_sst_satellite_data

#Import the functions to check for missing data
from Missing_data import (
                          check_missing_days,
                          find_missing_observations,
                          eliminate_empty_fields
                          )

# Importing the values from Constants.py
from Costants import (
                      DinY,
                      SinD,
                      nf,
                      chlfstart,
                      chlfend,
                      startingyear,
                      startingmonth,
                      startingday,
                      LE,
                      LS,
                      total_SZTtmp
                      )

### Setting up the working directory ###
# Change working directory
print("Accessing the satellite directory data for cleanup...")
WDIR = "C:/Tesi Magistrale/"
os.chdir(WDIR)  # Set the working directory
print('*'*45)

# -----DATA BASE DIRECTORY-----
BDIR = os.path.join(WDIR, "Dati/")

# -----DIRECTORIES FOR SATELLITE DATA-----
DSAT = os.path.join(BDIR, "SATELLITE/")
# SST
DSST_sat = os.path.join(DSAT, "SST/")
# SCHL
DCHL_sat = os.path.join(DSAT, "SCHL/")

# -----SATELLITE DATA LEVEL (NEEDED FOR CHLOROPHYLL DATA)-----
# N.B: chldlev='l3' data level 3
#      chldlev='l4' data level 4
chldlev = "l4"

# -----THESE ARE NEEDED TO COMPOSE THE SCHL FILE NAMES-----
ffrag1 = "NADR-CHL-SAT-CMEMS-"
ffrag2 = "daily"

# -----THIS IS THE VARIABLE NAME TO BE READ----
Schl2r = 'CHL'

# Calculate the true number of days
Truedays = true_time_series_length(nf, chlfstart, chlfend, DinY)

# Assert to check if the calculated number of days is 3652
assert Truedays == 3653, f"Error: Expected 3653 days, but got {Truedays}"

print("The number of true days in the dataset is:", Truedays)
print('*'*45)

# Call the function to read the data, merge it and check for SLON, SLAT shapes
T_orig, Schl_orig, Slon, Slat, Schlpath = chldata_into_array(
    total_SZTtmp, LE, LS, nf, ffrag1, chlfstart, chlfend, chldlev, ffrag2, DCHL_sat, Schl2r
)

# Run the missing days checker
Ttrue, Schl_complete=check_missing_days(T_orig, Schl_orig, Truedays, SinD, startingyear, startingmonth, startingday)

# Run the missing observation checker
satnan = find_missing_observations(Schl_complete, Truedays)

Schl_complete = eliminate_empty_fields(Schl_complete, Truedays)

again = input("Do you want to save the Satellite CHL data in case of interpolation? (yes/no): ").strip().lower()
print('-'*45)
if again in ["yes", "y"]:

    # Moving into the folder to save the data
    os.chdir(BDIR)

    #Moving into the OUTPUT FOLDER
    ODIR = os.path.join(BDIR, "OUTPUT/")
    os.chdir(ODIR)

    #Saving into the appropriate folder
    SODIR = os.path.join(ODIR, "SATELLITE/")
    os.chdir(SODIR)  # Set the working directory
    print(f"Saving the data in the folder {SODIR}")

    # Data to save
    data = {
        'Truedays': Truedays,
        'Slon': Slon,
        'Slat': Slat,
        'Schl_complete': Schl_complete,
        'satnan': satnan
        }

    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()  # Remove extra spaces
    print('-'*45)

    # Save based on user choice
    if choice == '1' or choice == '3':
        print("Saving data as a single .mat file...")
        # Saving as .mat file
        scipy.io.savemat("chl_clean.mat", data)
        print("Data saved as chl_clean.mat")
        print("-"*45)

    if choice == '2' or choice == '3':
        print("Saving the datasets as .nc files...")
        print("!!! satnan cannot be saved as a .nc file !!!")
        print("Saving satnan as a .csv file...")
        # Save `satnan` as a CSV file (count in the first column and indices in the second column)
        df = pd.DataFrame({'Count': [satnan[0]], 'Indices': [satnan[1]]})
        df.to_csv("satnan.csv", index=False)  # Save it as a CSV
        print("satnan saved as satnan.csv")
        print("-"*45)

        # Save other variables as .nc files
        print("Saving Slon as a .nc file...")
        Slon_xr = xr.DataArray(Slon)
        Slon_xr.to_netcdf("Slon.nc")
        print("Slon saved as Slon.nc")
        print("-"*45)
        
        print("Saving Slat as a .nc file...")
        Slat_xr = xr.DataArray(Slat)
        Slat_xr.to_netcdf("Slat.nc")
        print("Slat saved as Slat.nc")
        print("-"*45)
        
        print("Saving Schl_complete as a .nc file...")
        Schl_complete_xr = xr.DataArray(Schl_complete)
        Schl_complete_xr.to_netcdf("Schl_complete.nc")
        print("Schl_complete saved as Schl_complete.nc")
        print("-"*45)
        
    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("The requested clean data has been saved!")
    print("*"*45)

else:
    
    print("You chose not to save the data")
    print('*'*45)

print("Starting to read the satellite SST data...")
Sat_sst = read_sst_satellite_data(DSST_sat, Truedays)
print("Satellite SST retrieval completed!")
print("*"*45)

# Ask the user if they want to save the Satellite SST data
again = input("Do you want to save the Satellite SST data? (yes/no): ").strip().lower()
print('-'*45)

if again in ["yes", "y"]:
    # Moving into the folder to save the data
    os.chdir(BDIR)

    # Moving into the OUTPUT FOLDER
    ODIR = os.path.join(BDIR, "OUTPUT/")
    os.chdir(ODIR)

    # Saving into the SATELLITE_SST folder
    SODIR = os.path.join(ODIR, "SATELLITE/")
    os.chdir(SODIR)  # Set the working directory
    print(f"Saving the data in the folder {SODIR}")

    # Data to save
    data = {
        'Sat_sst': Sat_sst
    }

    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()  # Remove extra spaces
    print('-'*45)

    # Save based on user choice
    if choice == '1' or choice == '3':
        print("Saving the SST data as a .mat file...")
        # Saving as .mat file
        scipy.io.savemat("Sat_sst.mat", data)
        print("Data saved as Sat_sst.mat")
        print("-"*45)

    if choice == '2' or choice == '3':
        print("Saving the SST data as .nc file...")
        # Saving as .nc file
        Sat_sst_xr = xr.DataArray(Sat_sst)
        Sat_sst_xr.to_netcdf("Sat_sst.nc")
        print("Sat_sst saved as Sat_sst.nc")
        print("-"*45)

    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("The requested SST data has been saved!")
    print("*"*45)

else:
    print("You chose not to save the SST data")
    print('*'*45)

print("Resetting the working directory...")
WDIR = "C:/Tesi Magistrale/Codici/Python"
os.chdir(WDIR)  # Set the working directory