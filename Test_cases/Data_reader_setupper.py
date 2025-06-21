###############################################################################
##                     Author: Gozzoli Alessandro                            ##
##              email: alessandro.gozzoli4@studio.unibo.it                   ##
##                        UniBO id: 0001126381                               ##
###############################################################################

def main():

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

    title = "WELCOME TO THE DATA SETUPPING TEST CASE SCRIPT"
    border = "#" * 60
    print(border)
    print(title.center(60))
    print(border)


    print("""\nThis is a test case which illustrated how to best process the
data before analysing it using the other test cases.

This code can handle both Chlorophylle and Sea Surface Temperature datsets
and both Level 3s and level 4 data though to avoid excessive memory usage
only one dataset and one data level can be handled by each run of the
script.

This script is also combined with an interpolator made using a Matlab script.
To fully run the test matlab needs to be installed, please refer to the
installation guide.""")

    input("Please press any key to confirm and move on: \n")

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

    print(border)
    print("\nSetting up the necessary directories...")

    # Set up the working directory for an easier time when building paths
    print("The scripts are currently located in the following path:")
    WDIR = os.path.dirname(os.path.abspath(__file__))
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

    # ----- DIRECTORIES FOR MODEL DATA
    Dmod = Path(BaseDIR, "MODEL")
    print(f"Model data located at {Dmod}")
    
    print("All of the paths to access the test data have been built!\n")
    print(border)

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################

    print("\nLoading the necessary modules provided by the package...")

    print("Retrieving the File I/O necessary functions...")
    from Hydrological_model_validator.Processing.file_io import (mask_reader,
                                                             call_interpolator)
    print("\033[92m✅ File I/O functions retrieved!\033[0m")

    print("Retrieving the necessary functions to read the satellite and model data...")
    from Hydrological_model_validator.Processing.SAT_data_reader import sat_data_loader

    from Hydrological_model_validator.Processing.MOD_data_reader import read_model_data
    print("\033[92m✅ Functions to read the model and satellite data retrieved!\033[0m")

    print("Retrieving the functions necessary to identify missing data...")
    from Hydrological_model_validator.Processing.Missing_data import (
                          check_missing_days,
                          find_missing_observations,
                          eliminate_empty_fields
                          )
    print("\033[92m✅ Data functions for the identification of missing data retrieved!\033[0m")

    print("Retrieving the functions necessary to compute statistics...")
    from Hydrological_model_validator.Processing.stats_math_utils import compute_coverage_stats
    print("\033[92m✅ Functions for the computations of statistics retrieved!\033[0m")

    print("Retrieving the data saving functions...")
    from Hydrological_model_validator.Processing.Data_saver import (save_satellite_data,
                                                                save_model_data,
                                                                save_to_netcdf)
    print("\033[92m✅ Data saving functions retrieved!\033[0m")
    
    print("All of the modules have been retrieved!\n")
    print(border)
    
###############################################################################
##                                                                           ##
##                           PROCESSING THE DATA                             ##
##                                                                           ##
###############################################################################

    print("""\nAs already said in the initial explanation to avoid excessive
memory usage only one type of dataset and only one dqta level\n""")

    while True:
        varname = input("What kind of dataset would you like to read and process? (CHL or SST): ").lower()
        if varname in ['chl', 'sst']:
            break
        print("\033[91m⚠️ Invalid input. Please enter either 'CHL' or 'SST' ⚠️\033[0m")

    print(f"\033[92m✅ We are going to work on the {varname.upper()} dataset\033[0m")
    print('-' * 45)

    print("The possible satellite dataset levels are:")
    print(" - L3 (Super-collated): A single gridded product without interpolation.")
    print(" - L4 : A gap-free dataset already interpolated.")

    while True:
        data_level = input("Enter the satellite data level (L3 or L4): ").lower()
        if data_level in ['l3', 'l4']:
            break
        print("\033[91m⚠️ Invalid input. Please enter either 'L3' or 'L4' ⚠️\033[0m")

    print(f"\033[92m✅ Satellite data level set to: {data_level.upper()}\033[0m")
    print('-' * 45)

    print("""\033[91m⚠️ For some of the computations a mask is required! ⚠️
Such mask is provided alongside the other test case data ⚠️
⚠️ Please make sure that the data is masked (not the case for the test case datasets)
or provide the mask yourself ⚠️\033[0m""")

    # Ask user if the model data is already masked
    masking = input("Is the data provided already masked? (yes/no): ").strip().lower()

    if masking in ["yes", "y"]:
        print("\033[92m✅ Data is already masked. Proceeding...\033[0m")
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
        
    print("\n" + border)

###############################################################################
##                                                                           ##
##                                SATELLITE                                  ##
##                                                                           ##
###############################################################################

    print("""\nOnce the preliminary work has been completed the actual processing
can begin, this include the reading of the datasets (usually saved as .nc, they
can be read in both zipped or unzipped form) and the identification of missing
days in the timeseries, missing data or empty fields\n""")

    if varname == 'sst':
        Data_sat = Path(DSAT, "SST/")
        sys.path.append(str(Data_sat))  # Add the folder to the system path
        print(f"Satellite SST data located at {Data_sat}")
    elif varname == 'chl':
        Data_sat = Path(DSAT, "SCHL/")
        sys.path.append(str(Data_sat))  # Add the folder to the system path
        print(f"Satellite CHL data locate at {Data_sat}")

    print("Extracting the satellite data...")
    # Call the function to read the data, merge it and check for SLON, SLAT shapes
    T_orig, SatData_orig, Sat_lon, Sat_lat = sat_data_loader(data_level, Data_sat, varname)
    print("Data extracted!")

    print("Start looking for missing days...")
    # Run the functions to check for missing days/values/fields
    Ttrue, SatData_complete=check_missing_days(T_orig, SatData_orig)
    print("Missing days indentified and filled!")

    print("Start looking for missing data...")
    # Run the missing observation checker
    satnan = find_missing_observations(SatData_complete)

    # Run the empty field checker
    SatData_complete = eliminate_empty_fields(SatData_complete)
    print("Missing satellite data found and filled!!")

    print("Computing the cloud coverage...")
    data_available, cloud_cover = compute_coverage_stats(SatData_complete, Mmask)
    print("Cloud coverage computed!")

    save = input("Do you want to save the Satellite data? (yes/no): ").strip().lower()
    print('-' * 45)

    if save in ["yes", "y"]:
        # Create a timestamped folder for this run
        timestamp = datetime.now().strftime("run_%Y-%m-%d")
        output_path = os.path.join(BaseDIR, "OUTPUT", "CLEANING", varname, timestamp)
        print(f"The data is being saved in the {output_path} folder for easier access for the interpolator")
        os.makedirs(output_path, exist_ok=True)

        print(f"Saving files in folder: {output_path}")
        print('-' * 45)

        # Call the save function and pass the new path
        save_satellite_data(output_path, Sat_lon, Sat_lat, SatData_complete)
    
        print("The cloud coverage data is being saved in the folder user by the data analysis scripts")
        output_path_to_analysis = os.path.join(BaseDIR, "PROCESSING_INPUT")
        os.makedirs(output_path_to_analysis, exist_ok=True)
        print(f"The folder is named {output_path_to_analysis}")
    
        print("Saving the available data/cloud coverage data...")
    
        data_to_save = {
            f'data_available_{varname}': data_available,
            f'cloud_cover_{varname}': cloud_cover,
            }
    
        save_to_netcdf(data_to_save, output_path_to_analysis)
    
        print("Cloud coverage/available data has been saved!")

    else:
        print("You chose not to save the data")
        print('*' * 45)
        
        print("Please do not move the newly saved files!\n")
        
    print(border)
    
###############################################################################
##                                                                           ##
##                                  MODEL                                    ##
##                                                                           ##
###############################################################################

    print("""\nNow we can move on to the model data. The model data is guaranteed
to be complete so we can avoid checking for missing values.\n""")

    print("Starting to work on the model data...")

    # READING THE DATA
    print('*'*45)
    print("Reading the data...")
    ModData_complete = read_model_data(Dmod, Mfsm, varname)

    save = input("Do you want to save the Model CHL data? (yes/no): ").strip().lower()
    print('-' * 45)

    if save in ["yes", "y"]:
        # Create a timestamped folder for this run
        timestamp = datetime.now().strftime("run_%Y-%m-%d")
        output_path = os.path.join(BaseDIR, "OUTPUT", "CLEANING", varname, timestamp)
        os.makedirs(output_path, exist_ok=True)

        print(f"Saving files in folder: {output_path}")
        print('-' * 45)

        # Call the save function and pass the new path
        save_model_data(output_path, ModData_complete)

    else:
        print("You chose not to save the data")
        print('*' * 45)

    print("""\nAll of the newly saved data can be used to run the interpolator
Matlab script.

To guarantee the correct run of the interpolator please do not move these files!\n""")

    print(border)

###############################################################################
##                                                                           ##
##                              INTERPOLATOR                                 ##
##                                                                           ##
###############################################################################

    interpolate = input("\nDo you wish to call the interpolator to process the data? (yes/no): ").strip().lower()
    print('-' * 45)

    if interpolate in ["yes", "y"]:

        print("The data will be directly saved in the folder used by the analysis scripts...")
        output_path_to_analysis = os.path.join(BaseDIR, "PROCESSING_INPUT")
        os.makedirs(output_path, exist_ok=True)
        print(f"The folder is named {output_path_to_analysis}")

        # Reuse the output path of the savng functions
        input_path = output_path

        Mask_path=os.path.join(BaseDIR)

        print("Calling the MatLab interpolator function...")
        call_interpolator(varname, data_level, input_dir=input_path, output_dir=output_path_to_analysis, mask_file=Mask_path)

        print("The data has been succesfully interpolated and the Basin Averages have been computed!")
        print("Please retrieve the new datasets from the folder!")
    
    else:
        print("You chose not to interpolate the data")
   
    print("You may proceed with the analysis scripts!")
    print(border)
    
if __name__ == "__main__":
    main()