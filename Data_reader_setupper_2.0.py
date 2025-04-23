###############################################################################
##                     Author: Gozzoli Alessandro                            ##
##              email: alessandro.gozzoli4@studio.unibo.it                   ##
##                        UniBO id: 0001126381                               ##
###############################################################################

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

# Building the processing function directory path
print("Retrieving the processing functions' folder...")
ProcessingDIR = Path(WDIR, "Processing/")
sys.path.append(str(ProcessingDIR))  # Add the folder to the system path
print(f"The processing functions are located in the folder {ProcessingDIR} !")
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
print("\033[91m⚠️ MAKE SURE TO UPDATE THE PATHS ACCORDINGLY ⚠️\033[0m")
print('*'*45)

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################

# Knowing the size of the datasets counts the number of days in the timeseries
# by taking into accounts the leap years
from Leap_year import true_time_series_length

# Reads the satellite chlorophylle dataset
from SAT_data_reader import sat_chldata

# Reads the satellite sea surface temperature dataset
from SST_SAT_reader import read_sst_satellite_data

# Series of functions to check for the missing data
from Missing_data import (
                          check_missing_days,
                          find_missing_observations,
                          eliminate_empty_fields
                          )

from Data_saver import save_satellite_data

# A series of user define costants used for multiple computations
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

###############################################################################
##                                                                           ##
##                           PROCESSING THE DATA                             ##
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

again = input("Do you want to save the Satellite CHL data in case of interpolation? (yes/no): ").strip().lower()
print('-' * 45)

if again in ["yes", "y"]:
    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    output_path = os.path.join(BaseDIR, "OUTPUT", "SATELLITE", chldlev, timestamp)
    os.makedirs(output_path, exist_ok=True)

    print(f"📁 Saving files in folder: {output_path}")
    print('-' * 45)

    # Call the save function and pass the new path
    save_satellite_data(output_path, Truedays, Slon, Slat, Schl_complete)

else:
    print("You chose not to save the data")
    print('*' * 45)