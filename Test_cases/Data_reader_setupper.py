###############################################################################
##                     Author: Gozzoli Alessandro                            ##
##              email: alessandro.gozzoli4@studio.unibo.it                   ##
##                        UniBO id: 0001126381                               ##
###############################################################################

# Ignoring a depracation warning to ensure a better console run
import warnings
from cryptography.utils import CryptographyDeprecationWarning

# Ignore specific deprecation warning from cryptography (used by paramiko)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

###############################################################################
###############################################################################
##    This code retrieves and sets up the Satellite and Model Sea Surface    ##
##   Temperature and Chlorofille data to pass them through an interpolator.  ##
## Everything is done to prepare the data to be analyzed and to validate and ##
##  evaluate the performance of the Bio-Physical model to indentify possible ##
##                          areas of improvements.                           ##
###############################################################################
###############################################################################

print("### Welcome to the Data Setupping script ###")
print('*'*45)

###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Library used to set up the working directory and paths
import os
from pathlib import Path
import sys

# To create necessary folders
from datetime import datetime

###############################################################################
##                                                                           ##
##                             PATH BUILDERS                                 ##
##                                                                           ##
###############################################################################

print("Setting up the necessary directories...")

# Set up the working directory for an easier time when building paths
print("The scripts are currently located in the following path:")
WDIR = os.getcwd()
print(WDIR)
print('-'*45)

# ----- BUILDING THE DATA FOLDER PATH -----
print("Locating the datasets...")
# Defining the base data directory
BaseDIR = Path(WDIR, "Data/")
sys.path.append(str(BaseDIR))  # Add the folder to the system path
print(f"The datasets are located in the folder {BaseDIR} !")

# ----- DIRECTORIES FOR SATELLITE DATA -----
DSAT = Path(BaseDIR, "SATELLITE/")
sys.path.append(str(DSAT))  # Add the folder to the system path
print(f"Satellite data located at {DSAT} !")
# SST
DSST_sat = Path(DSAT, "SST/")
sys.path.append(str(DSST_sat))  # Add the folder to the system path
print(f"Satellite SST data located at {DSST_sat}")
# SCHL
DCHL_sat = Path(DSAT, "SCHL/")
sys.path.append(str(DCHL_sat))  # Add the folder to the system path
print(f"Satellite CHL data locate at {DCHL_sat}")

# ----- DIRECTORIES FOR MODEL DATA
Dmod = Path(BaseDIR, "MODEL")
print(f"Model data lcated at {Dmod}")

print("\033[91m⚠️ MAKE SURE TO UPDATE THE PATHS ACCORDINGLY ⚠️\033[0m")
print('*'*45)

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################

print("Retrieving the time utility necessary functions...")
from Hydrological_model_validator.Processing.time_utils import true_time_series_length
print("\033[92m✅ Time utility functions retrieved!\033[0m")

print("Retrieving the File I/O necessary functions...")
from Hydrological_model_validator.Processing.file_io import mask_reader
print("\033[92m✅ File I/O functions retrieved!\033[0m")

print("Retrieving the necessary functions to read the satellite and model data...")
from Hydrological_model_validator.Processing.SAT_data_reader import sat_chldata, read_sst_satellite_data

# Reads the model datasets
from Hydrological_model_validator.Processing.MOD_data_reader import read_model_chl_data, read_model_sst, Bavg_sst
print("\033[92m✅ Functions to read the model and satellite data retrieved!\033[0m")

print("Retrieving the functions necessary to identify missing data...")
from Hydrological_model_validator.Processing.Missing_data import (
                          check_missing_days,
                          find_missing_observations,
                          eliminate_empty_fields
                          )
print("\033[92m✅ Data functions for the identification of missing data retrieved!\033[0m")

print("Retrieving the data saving functions...")
from Hydrological_model_validator.Processing.Data_saver import save_satellite_CHL_data, save_satellite_SST_data, save_model_CHL_data, save_model_SST_data, save_SST_Bavg
print("\033[92m✅ Data saving functions retrieved!\033[0m")

print("Retrieving some costants...")
print("These will be deprecated in a future update!")
from Hydrological_model_validator.Processing.Costants import (
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
                      total_SZTtmp,
                      Ybeg,
                      Tspan,
                      ysec
                      )
print("\033[92m✅ Costants retrieved!\033[0m")

###############################################################################
##                                                                           ##
##                           PROCESSING THE DATA                             ##
##                                                                           ##
###############################################################################

###############################################################################
##                                                                           ##
##                         SATELLITE - CHLOROPHYLLE                          ##
##                                                                           ##
###############################################################################

print("Beginning the data cleanup...")

print("Which level of satellite data would you like to handle?")
print("\033[91mCareful, the datasets might be to big to be handled by the code\033[0m")
print("\033[91mHence why it is suggested to make a run of the script for each data level\033[0m")
# ASK THE USER FOR THE LEVEL OF CHL DATA TO BE READ
# The code cannot run both toghether due to excessive RAM requirements (test case is 22+2Gb)
# Data divided in level 3 or 4 data
while True:
    chldlev = input("Enter CHL satellite data level (l3 or l4): ").lower()
    if chldlev in ['l3', 'l4']:
        break
    print("\033[91m⚠️ Invalid input. Please enter either 'l3' or 'l4' ⚠️\033[0m")

print(f"\033[92m✅ Satellite CHL data level set to: {chldlev}\033[0m")
print('-'*45)

# Definition of some variables needed to compose the file output names
ffrag1 = "NADR-CHL-SAT-CMEMS-"
ffrag2 = "daily"

Schl2r = 'CHL'

# COMPUTE THE NUMBER OF DAYS IN THE DATASET
Truedays = true_time_series_length(nf, chlfstart, chlfend, DinY)

# Assert to check if the calculated number of days is 3652
assert Truedays == 3653, f"\033[91m⚠️ Error: Expected 3653 days, but got {Truedays}⚠️\033[0m"

print("The number of days in the dataset accounting for leap years is:", Truedays)
print('-'*45)

# READ THE CHL SATELLITE DATA
# Call the function to read the data, merge it and check for SLON, SLAT shapes
T_orig, Schl_orig, Slon, Slat, Schlpath = sat_chldata(
    total_SZTtmp, LE, LS, nf, ffrag1, chlfstart, chlfend, chldlev, ffrag2, DCHL_sat, Schl2r
)

# Run the missing days checker
Ttrue, Schl_complete=check_missing_days(T_orig, Schl_orig, Truedays, SinD, startingyear, startingmonth, startingday)

# Run the missing observation checker
satnan = find_missing_observations(Schl_complete, Truedays)

# Run the empty field checker
Schl_complete = eliminate_empty_fields(Schl_complete, Truedays)

save = input("Do you want to save the Satellite CHL data? (yes/no): ").strip().lower()
print('-' * 45)

if save in ["yes", "y"]:
    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    output_path = os.path.join(BaseDIR, "OUTPUT", "SATELLITE", chldlev, timestamp)
    os.makedirs(output_path, exist_ok=True)

    print(f"Saving files in folder: {output_path}")
    print('-' * 45)

    # Call the save function and pass the new path
    save_satellite_CHL_data(output_path, Slon, Slat, Schl_complete)

else:
    print("You chose not to save the data")
    print('*' * 45)

###############################################################################
##                                                                           ##
##                    SATELLITE - SEA SURFACE TEMPERATURE                    ##
##                                                                           ##
###############################################################################

print("Starting to read the satellite SST data...")
Sat_sst = read_sst_satellite_data(DSST_sat, Truedays)
print("Satellite SST retrieval completed!")
print("*"*45)

save = input("Do you want to save the Satellite SST data? (yes/no): ").strip().lower()
print('-' * 45)

if save in ["yes", "y"]:
    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    output_path = os.path.join(BaseDIR, "OUTPUT", "SATELLITE", "SST", timestamp)
    os.makedirs(output_path, exist_ok=True)

    print(f"Saving files in folder: {output_path}")
    print('-' * 45)

    # Call the save function and pass the new path
    save_satellite_SST_data(output_path, Sat_sst)

else:
    print("You chose not to save the data")
    print('*' * 45)
    
print("Satellite data completed!")
print("*"*45)

###############################################################################
##                                                                           ##
##                           MODEL - CHLOROPHYLLE                            ##
##                                                                           ##
###############################################################################

print("Starting to work on the model data...")

print("\033[91m⚠️ The model data needs to be masked ⚠️\033[0m")
print("\033[91m⚠️ Please make sure that the data is masked or provide the mask yourself ⚠️\033[0m")

# Ask user if the model data is already masked
masking = input("Is the model data provided already masked? (yes/no): ").strip().lower()

if masking in ["yes", "y"]:
    print("\033[92m✅ Model data is already masked. Proceeding...\033[0m")
    print("-" * 45)

elif masking in ["no", "n"]:
    # Ask if the raw mask file is already in the folder
    while True:
        raw_mask = input("Is the raw mask file already provided in the data folder? (yes/no): ").strip().lower()

        if raw_mask in ["yes", "y"]:
            print("Retrieving land mask data...")
            Mmask, Mfsm, Mfsm_3d, Mlat, Mlon = mask_reader(BaseDIR)
            print("\033[92m✅ Mask successfully imported!\033[0m")
            print("-" * 45)
            break

        elif raw_mask in ["no", "n"]:
            print("Please add the raw mask file to the data folder.")
            
            # Ask for confirmation once the file has been added
            while True:
                confirm = input("Have you added the raw mask file to the folder? (yes/no): ").strip().lower()
                
                if confirm in ["yes", "y"]:
                    print("Retrieving land mask data...")
                    Mmask, Mfsm, Mfsm_3d, Mlat, Mlon = mask_reader(BaseDIR)
                    print("\033[92m✅ Mask successfully imported!\033[0m")
                    print("-" * 45)
                    break  # Break inner loop

                elif confirm in ["no", "n"]:
                    print("Waiting for you to add the raw mask file...")
                else:
                    print("Please answer with 'yes' or 'no'.")

            break  # Break outer loop once mask is loaded

        else:
            print("Please answer with 'yes' or 'no'.")

else:
    print("Invalid input. Please answer with 'yes' or 'no'.")

# READING THE DATA
print('*'*45)
print("Reading the CHL data...")
Mchl_complete = read_model_chl_data(Dmod, Ybeg, Tspan, Truedays, DinY, Mfsm)

save = input("Do you want to save the Model CHL data? (yes/no): ").strip().lower()
print('-' * 45)

if save in ["yes", "y"]:
    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    output_path = os.path.join(BaseDIR, "OUTPUT", "MODEL", "CHL", timestamp)
    os.makedirs(output_path, exist_ok=True)

    print(f"Saving files in folder: {output_path}")
    print('-' * 45)

    # Call the save function and pass the new path
    save_model_CHL_data(output_path, Mchl_complete)

else:
    print("You chose not to save the data")
    print('*' * 45)

###############################################################################
##                                                                           ##
##                      MODEL - SEA SURFACE TEMPERATURE                      ##
##                                                                           ##
###############################################################################

print("Starting to read the MODEL SST data...")
Msst_complete = read_model_sst(Dmod, ysec, Mfsm)
print("\033[92m✅ The full Model SST data has been retrieved!\033[0m")
print('*'*45)

save = input("Do you want to save the Model SST data? (yes/no): ").strip().lower()
print('-' * 45)

if save in ["yes", "y"]:
    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    output_path = os.path.join(BaseDIR, "OUTPUT", "MODEL", "SST", timestamp)
    os.makedirs(output_path, exist_ok=True)

    print(f"Saving files in folder: {output_path}")
    print('-' * 45)

    # Call the save function and pass the new path
    save_model_SST_data(output_path, Msst_complete)

else:
    print("You chose not to save the data")
    print('*' * 45)
    
###############################################################################
##                                                                           ##
##                              BASIN AVERAGES                               ##
##                                                                           ##
###############################################################################

Bavg = input("Do you wish to compute the Basin Averages? (yes/no): ").strip().lower()
print('-'*45)

if Bavg in ["yes", "y"]:
    # WARNING ABOUT THE CHL DATA
    print("\033[91m⚠️ WARNING ABOUT THE CHLOROPHYLLE DATA ⚠️")
    print("\033[91m Due to the high concentration of Nan fields and other ")
    print(" missing data in the CHL fields (especially the satellite ones) ")
    print(" the basin avegares computations can be done only after the ")
    print(" run of the interpolator.m script to ensure that the missing ")
    print(" fields are appropriatelly reconstructed. \033[0m")
    
    # Computing the SST Basin Averages
    print("Creating the Basin Average timeseries of model and satellite SST data...")
    BASSTmod, BASSTsat = Bavg_sst(Tspan, ysec, Dmod, Sat_sst, Mfsm)
    print("\033[92m✅ Basin Average Daily Mean Timeseries computed! \033[0m")
    
    save = input("Do you want to save the basin average SST data? (yes/no): ").strip().lower()

    if save in ["yes", "y"]:
        
        # Create a timestamped folder for this run
        timestamp = datetime.now().strftime("run_%Y-%m-%d")
        output_path = os.path.join(BaseDIR, "OUTPUT", "BASIN_AVERAGES", "SST", timestamp)
        os.makedirs(output_path, exist_ok=True)

        print(f"Saving files in folder: {output_path}")
        print('-' * 45)

        # Call the save function and pass the new path
        save_SST_Bavg(output_path, BASSTmod, BASSTsat)
        
    print("Please proceed with either the interpolation of the missing satellite data")
    print("or with the analysis and validation of the datasets")
    print('*'*45)
    
else:
    print("You chose not to compute the Basin Averages data")
    print("Please proceed with either the interpolation of the missing satellite data")
    print("or with the analysis and validation of the datasets")
    print('*'*45)