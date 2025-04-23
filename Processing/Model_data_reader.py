import os
import scipy.io
import xarray as xr

from Mask_reader import mask_reader

from CHL_model_reader import read_model_chl_data

from Model_SST_reader import read_model_sst_yearly

from SST_Reader_test import read_model_sst

from Costants import (
                      Ybeg,
                      ysec,
                      Tspan,
                      DinY,
                      nf,
                      chlfstart,
                      chlfend
                      )

from Leap_year import true_time_series_length

from Data_setupper import Sat_sst

### Setting up the working directory ###
# Change working directory
print("Accessing the directory data...")
WDIR = "C:/Tesi Magistrale/"
os.chdir(WDIR)  # Set the working directory

# -----DATA BASE DIRECTORY-----
BDIR = os.path.join(WDIR, "Dati/")

# -----DIRECTORIES FOR MODEL DATA----
Dmod = os.path.join(BDIR, "MODEL/")

# Calculate the true number of days
Truedays = true_time_series_length(nf, chlfstart, chlfend, DinY)

# Call the function and extract values
Mmask, Mfsm, Mfsm_3d, Mlat, Mlon = mask_reader()
print("Mask succesfully imported!")

print('*'*45)
print("Reading the CHL data...")
print('-'*45)
Mchl_complete, Mchl_orig = read_model_chl_data(Dmod, Ybeg, Tspan, Truedays, DinY, Mfsm)

again = input("Do you want to save the MODEL CHL data? (yes/no): ").strip().lower()
if again in ["yes", "y"]:
    
    print("!!! Careful !!!")
    print("!!!These dataset do not have the satnan mask applied to them!!!")
    print("For further analysis it is suggested to pass these data")
    print("though the Interpolator.m script provided alongise")
    print("these Python scripts to ensure that shapes etc. match")
    print("with the Satellite data, especially regarding the presence")
    print("of the missing satellite values")
    print("!!! This is necessary when using the level3 data !!!")
    
    #Moving to save the data in OUTPUT folder
    #Moving into the OUTPUT FOLDER
    ODIR = os.path.join(BDIR, "OUTPUT/")
    os.chdir(ODIR)

    #Saving into the appropriate folder
    MODIR = os.path.join(ODIR, "MODEL/")
    os.chdir(MODIR)  # Set the working directory
    print(f"Saving the data in the folder {MODIR}")

    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()  # Remove extra spaces
    print('-'*45)

    if choice == "1" or choice == "3":
        print("Saving data as a .mat file...")
        # Saving as .mat file - wrap Mchl_complete in a dictionary
        scipy.io.savemat("Mchl_complete.mat", {"Mchl_complete": Mchl_complete})
        print("Data saved as Mchl_complete.mat")
        print("-"*45)

    if choice == "2" or choice == "3":
        print("Saving Mchl_complete as a .nc file...")
        Mchl_complete_xr = xr.DataArray(Mchl_complete)
        Mchl_complete_xr.to_netcdf("Mchl_complete.nc")
        print("Mchl_complete saved as Mchl_complete.nc")
        print("-"*45)
    
    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("The requested data has been saved!")
    print("*"*45)
    
else:
    
    print("You chose to not save the model CHL dataset")
    print('*'*45)
    
print("Starting to read the MODEL SST data...")
Msst_complete = read_model_sst_yearly(Dmod, ysec, Mfsm)
print("The full Model SST data has been retrieved!")
print('*'*45)

import os
import scipy.io
import xarray as xr

again = input("Do you want to save the MODEL SST data? (yes/no): ").strip().lower()
if again in ["yes", "y"]:
    
    # Moving to save the data in OUTPUT folder
    ODIR = os.path.join(BDIR, "OUTPUT/")
    MODIR = os.path.join(ODIR, "MODEL/")
    
    # Ensure directories exist before changing into them
    os.makedirs(MODIR, exist_ok=True)
    os.chdir(MODIR)
    print(f"Saving the data in the folder {MODIR}")

    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-'*45)

    # Prepare .mat data
    mat_data = {}
    for year, array in Msst_complete.items():
        mat_data[str(year)] = array  # Store each array under its corresponding year

    if choice == "1" or choice == "3":
        print("Saving data as a .mat file...")
        try:
            scipy.io.savemat("Msst_complete.mat", mat_data)
            print("Data saved as Msst_complete.mat")
        except Exception as e:
            print(f"Error saving MAT file: {e}")
        print("-"*45)

    if choice == "2" or choice == "3":
        print("Saving each year separately as .nc files...")
        for year, array in Msst_complete.items():
            try:
                ds = xr.Dataset({str(year): (["time", "lat", "lon"], array)})  # Assuming array has shape (time, lat, lon)
                filename = f"Msst_{year}.nc"
                ds.to_netcdf(filename)
                print(f"Saved {filename}")
            except Exception as e:
                print(f"Error saving {year} NetCDF file: {e}")
        print("-"*45)
    
    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("The requested data has been saved!")
    print("*"*45)

else:
    
    print("You chose not to save the model SST dataset")
    print('*'*45)
    
print("The model data has been read")
print("*"*45)

print("Computing the daily mean...")
BASSTmod, BASSTsat = read_model_sst(Tspan, ysec, Dmod, Sat_sst, Mfsm)
print("Basin Average Daily Mean Timeseries computed!")

# Ask user if they want to save the model SST data
again = input("Do you want to save the basin average SST data? (yes/no): ").strip().lower()

if again in ["yes", "y"]:
    
    # Moving to save the data in OUTPUT folder
    ODIR = os.path.join(BDIR, "OUTPUT/")  # Adjust BDIR to where your base directory is
    BADIR = os.path.join(ODIR, "BASIN_AVERAGE/")
    
    # Ensure directories exist before changing into them
    os.makedirs(BADIR, exist_ok=True)
    os.chdir(BADIR)
    print(f"Saving the data in the folder {BADIR}")

    # Ask the user for the preferred format
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-'*45)

    # Prepare .mat data
    mat_data = {
        "BASSTmod": BASSTmod,
        "BASSTsat": BASSTsat
    }

    if choice == "1" or choice == "3":
        print("Saving data as a .mat file...")
        try:
            scipy.io.savemat("BASST_data.mat", mat_data)
            print("Data saved as BASST_data.mat")
        except Exception as e:
            print(f"Error saving MAT file: {e}")
        print("-"*45)

    if choice == "2" or choice == "3":
        print("Saving each dataset separately as .nc files...")
        
        # Save BASSTmod as NetCDF
        try:
            ds_mod = xr.Dataset({"BASSTmod": ("time", BASSTmod)})
            filename_mod = "BASSTmod.nc"
            ds_mod.to_netcdf(filename_mod)
            print(f"Saved {filename_mod}")
        except Exception as e:
            print(f"Error saving BASSTmod NetCDF file: {e}")
        
        # Save BASSTsat as NetCDF
        try:
            ds_sat = xr.Dataset({"BASSTsat": ("time", BASSTsat)})
            filename_sat = "BASSTsat.nc"
            ds_sat.to_netcdf(filename_sat)
            print(f"Saved {filename_sat}")
        except Exception as e:
            print(f"Error saving BASSTsat NetCDF file: {e}")
        
        print("-"*45)

    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("The requested data has been saved!")
    print("*"*45)

else:
    
    print("You chose not to save the model SST dataset")
    print('*'*45)

print("The model data has been read")
print("You can proceed with the analysis")

print("Resetting the working directory...")
WDIR = "C:/Tesi Magistrale/Codici/Python"
os.chdir(WDIR)  # Set the working directory