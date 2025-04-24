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
##    This code retrieves the data from the setupper (SST specifically)      ##
## analyzes it and tests it for efficiency. The analysis is done by creating ##
##    multiple dictionaries and then plotting timeseries and scatterplots.   ##
##       The efficiency is tested by using taylor diagrams, target plots     ##
##        and multiple coefficents and metrics, which are also plotted.      ##
###############################################################################
###############################################################################


###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Libraries for paths
import os
from pathlib import Path
import sys

# Utility libraries
import numpy as np
import pandas as pd
import xarray as xr
import calendar
from datetime import datetime

###############################################################################
##                                                                           ##
##                                MODULES                                    ##
##                                                                           ##
###############################################################################

print("Loading the necessary modules...")
WDIR = os.getcwd()
os.chdir(WDIR)  # Set the working directory

print("Loading the Pre-Processing modules and constants...")
ProcessingDIR = Path(WDIR, "Processing/")
sys.path.append(str(ProcessingDIR))  # Add the folder to the system path

from Costants import ysec
from Corollary import convert_to_monthly_data

print("\033[92m✅ Pre-processing modules have been loaded!\033[0m")
print("-"*45)

print("Loading the plotting modules...")
PlottingDIR = Path(WDIR, "Plotting")
sys.path.append(str(PlottingDIR))  # Add the folder to the system path

from Plots import plot_daily_means, plot_metric, scatter_plot
from Taylor_diagrams import (
                             comprehensive_taylor_diagram,
                             monthly_taylor_diagram
                             )
from Target_plots import (
                          comprehensive_target_diagram,
                          target_diagram_by_month
                          )

taylor_options = str(Path(PlottingDIR, 'taylor_option_config.csv'))
taylor_options_monthly = str(Path(PlottingDIR, 'taylor_option_config - monthly.csv'))

print("\033[92m✅ The plotting modules have been loaded!\033[0m")
print('-'*45)

print("Loading the validation modules...")
from Efficiency_metrics import (
                                r_squared,
                                weighted_r_squared,
                                nse,
                                index_of_agreement,
                                ln_nse,
                                nse_j,
                                index_of_agreement_j,
                                relative_nse,
                                relative_index_of_agreement,
                                monthly_r_squared,
                                monthly_weighted_r_squared,
                                monthly_nse,
                                monthly_index_of_agreement,
                                monthly_ln_nse,
                                monthly_nse_j,
                                monthly_index_of_agreement_j,
                                monthly_relative_nse,
                                monthly_relative_index_of_agreement
                                )
print("\033[92m✅ The validation modules have been loaded!\033[0m")
print('*'*45)

###############################################################################
##                                                                           ##
##                             DATA LOADING                                  ##
##                                                                           ##
###############################################################################

# ----- SETTING UP THE WORKING DIRECTOTY -----
print("Resetting the working directory...")
WDIR = os.getcwd()
os.chdir(WDIR)  # Set the working directory
print('*'*45)

# ----- BASE DATA DIRECTORY -----
BDIR = Path(WDIR, "Data")

# ----- INPUT DATA DIRECTORY -----
IDIR = Path(BDIR, "PROCESSING_INPUT/")
print("Loading the input data...")
print(f"\033[91m⚠️ The input data needs to be located in the {IDIR} folder ⚠️\033[0m")
print("\033[91m⚠️ Make sure that it contains all of the necessary datasets ⚠️\033[0m")
print("-"*45)

print("The folder contains the following datasets")
# List the contents of the folder
contents = os.listdir(IDIR)
# Print the contents
print(contents)
print("*"*45)

###############################################################################
##                                                                           ##
##                              DICTIONARIES                                 ##
##                                                                           ##
###############################################################################

print("Setting up the SST dictionary...")
print('-'*45)

# Starting with the model SST

# Dictionary to store datasets
Msst_data = {}

print("Getting the yearly model SST datasets...")
# Loop through the years and load the datasets
for year in ysec:
    file_name = f"Msst_{year}.nc"  # Create the file name
    file_path = os.path.join(IDIR, file_name)  # Full path
    
    if os.path.exists(file_path):  # Check if the file exists
        print(f"Opening {file_name}...")
        Msst_data[year] = xr.open_dataset(file_path)  # Store dataset in dictionary
    else:
        print(f"Warning: {file_name} not found!")

print("\033[92m✅ Model SST obtained!\033[0m")

# Moving onto the satellite SST data

print("Getting the satellite SST data...")
SAT_SST = xr.open_dataset(Path(IDIR, 'Sat_sst.nc'))
print("\033[92m✅ Satellite SST obtained!\033[0m")

SST = {
       'MODEL' : Msst_data,
       'SATELLITE' : SAT_SST
       }

print("-"*45)
print("\033[92m✅ SST dictionary created!\033[0m")
print("*"*45)

# ----- IMPORTING BASIN AVERAGES -----

print("Importing the Basin Average SST timeseries")
BASSTmod = xr.open_dataset(Path(IDIR, 'BASSTmod.nc'))
BASSTsat = xr.open_dataset(Path(IDIR, 'BASSTsat.nc'))
BASSTmod = BASSTmod['BASSTmod'].values
BASSTsat = BASSTsat['BASSTsat'].values
print("\033[92m✅ Basin Average Timseries obtained!\033[0m")

print("Adding them to a dictionary...")
BASST = {
        'BAmod' : BASSTmod,
        'BAsat' : BASSTsat
        }
print("\033[92m✅ Basin Average dictionary created!\033[0m")
print('*'*45)

print("Splitting the data to better hanlde it...")

# Generate datetime index
dates = pd.date_range(start='2000-01-01', end='2009-12-31', freq='D')
years = np.array([d.year for d in dates])
unique_years = np.unique(years)

# Split data into a list of arrays, one per year
mod_split = [BASSTmod[years == y] for y in unique_years]
sat_split = [BASSTsat[years == y] for y in unique_years]

# Find max days in any year (i.e., 366)
max_days = max(len(year_data) for year_data in mod_split)

# Pad shorter years with NaNs to make uniform arrays
BASSTmod_yearly = np.array([np.pad(year_data, (0, max_days - len(year_data)), constant_values=np.nan)
                              for year_data in mod_split])
BASSTsat_yearly = np.array([np.pad(year_data, (0, max_days - len(year_data)), constant_values=np.nan)
                              for year_data in sat_split])

BASST_yearly = {
                'BAmod_year' : BASSTmod_yearly,
                'BAsat_year' : BASSTsat_yearly
                }

print("\033[92m✅ Yearly datasets computed and added to the dictionary!\033[0m")

# Conversion to a dictionary divided in months
BASSTmod_monthly_dict = convert_to_monthly_data(BASSTmod_yearly)
BASSTsat_monthly_dict = convert_to_monthly_data(BASSTsat_yearly)

BASSTmonthly = {
                'BASSTmod_monthly' : BASSTmod_monthly_dict,
                'BASSTsat_monthly' : BASSTsat_monthly_dict
                }

print("\033[92m✅ Monthly datasets computed and added to a dictionary! \033[0m")
print("\033[92m✅ All of the dictionaries have been created!\033[0m")

print("\033[91m⚠️ HEADS UP ⚠️")
print("To ensure that the code runs smoothly the plots will be")
print("displayed only for 3 seconds. This time can be changed")
print("in the script. After the plot's window closes it will be")
print("saved in the appropriate folder for further analysis.\033[0m")
confirm = input("Please press any key to confirm and move on: ")

###############################################################################
##                                                                           ##
##                            TAYLOR DIAGRAMS                                ##
##                                                                           ##
###############################################################################

std_ref = np.std(BASSTsat)

# Plotting the Taylor Diagram
print("Plotting the Taylor diagrams...")
print("-"*45)

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "TAYLOR", "SST", timestamp)
os.makedirs(output_path, exist_ok=True)

print("Plotting the SST Taylor diagram for yearly data...")
comprehensive_taylor_diagram(BASST_yearly, taylor_options, std_ref, output_path)
print("\033[92m✅ Yearly data Taylor diagram has been plotted!\033[0m")
print("-"*45)

print("Plotting the monthly data diagrams...")
for i in range(12):
    monthly_taylor_diagram(BASSTmonthly, i, taylor_options_monthly, output_path)
print("\033[92m✅ All of the Taylor diagrams have been plotted!\033[0m")
print("*"*45)

###############################################################################
##                                                                           ##
##                              TARGET PLOTS                                 ##
##                                                                           ##
###############################################################################

# Making the target plots
print("Plotting the Target plots...")
print("-"*45)

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "TARGET", "SST", timestamp)
os.makedirs(output_path, exist_ok=True)

print("Plotting the Target plot for the yearly data...")
comprehensive_target_diagram(BASST_yearly, output_path)
print("\033[92m✅ Yearly data Target plot has been plotted!\033[0m")
print("-"*45)

print("Plotting the monthly data plots...")
for i in range(12):
    target_diagram_by_month(BASSTmonthly, i, output_path)
print("\033[92m✅ All of the Target plots has been plotted!\033[0m")
print("*"*45)

###############################################################################
##                                                                           ##
##                              OTHER PLOTS                                  ##
##                                                                           ##
###############################################################################

print("Plotting remaining plots...")

# Time series and scatter plots
BIAS_Bavg = BASSTsat - BASSTmod

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "OTHER", "SST", timestamp)
os.makedirs(output_path, exist_ok=True)

print("Plotting the time-series...")
plot_daily_means(output_path, BASST, 'SST', BIAS_Bavg, BA=True)
print("\033[92m✅ Time-series plotted succesfully!\033[0m")

print("Plotting the scatter plot...")
scatter_plot(output_path, BASST, 'SST', BA=False)
print("\033[92m✅ Scatter plot plotted succesfully!\033[0m")
print("*"*45)

###############################################################################
##                                                                           ##
##                          EFFICIENCY METRCIS                               ##
##                                                                           ##
###############################################################################

print("Computing the Efficiency Metrics...")
print("-"*45)

# ----- Coefficient of determination -----
print("\033[93mComputing Coefficient of Determination (r²)...\033[0m")

r2_value = r_squared(BASST['BAsat'], BASST['BAmod'])
print(f"r² (Coefficient of Determination) = {r2_value:.4f}")

r2_monthly = monthly_r_squared(BASSTmonthly)

for i, val in enumerate(r2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: r² = {val:.4f}")

print("-" * 45)

# ----- Weighted Coeddificient of Determination -----
print("\033[93mComputing Weighted Coefficient of Determination (wr²)...\033[0m")

wr2_value = weighted_r_squared(BASST['BAsat'], BASST['BAmod'])
print(f"wr² (Weighted Coefficient of Determination) = {wr2_value:.4f}")

wr2_monthly = monthly_weighted_r_squared(BASSTmonthly)

for i, val in enumerate(wr2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: wr² = {val:.4f}")

print("-" * 45)

# ----- Nash-Sutcliffe -----
print("\033[93mComputing Nash–Sutcliffe Efficiency (NSE)...\033[0m")

nse_value = nse(BASST['BAsat'], BASST['BAmod'])
print(f"NSE (Nash–Sutcliffe Efficiency) = {nse_value:.4f}")

nse_monthly = monthly_nse(BASSTmonthly)

for i, val in enumerate(nse_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: NSE = {val:.4f}")

print("-" * 45)

# ----- Index of Agreement -----
print("\033[93mComputing Index of Agreement (d)...\033[0m")

d_value = index_of_agreement(BASST['BAsat'], BASST['BAmod'])
print(f"Index of Agreement (d) = {d_value:.4f}")

d_monthly = monthly_index_of_agreement(BASSTmonthly)

for i, val in enumerate(d_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: Index of Agreement (d) = {val:.4f}")

print("-" * 45)

# ----- Logarithmic Nash–Sutcliffe Efficiency (ln NSE) -----
print("\033[93mComputing NSE with Logarithmic Values (ln NSE)...\033[0m")

# Remove NaNs and ensure values are positive before computing
mask = ~np.isnan(BASST['BAsat']) & ~np.isnan(BASST['BAmod']) & \
       (BASST['BAsat'] > 0) & (BASST['BAmod'] > 0)

ln_nse_value = ln_nse(BASST['BAsat'][mask], BASST['BAmod'][mask])
print(f"ln NSE = {ln_nse_value:.4f}")

ln_nse_monthly = monthly_ln_nse(BASSTmonthly)

for i, val in enumerate(ln_nse_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: ln NSE = {val:.4f}")

print("-" * 45)

# ----- Modified Nash–Sutcliffe Efficiency (E₁, j=1) -----
print("\033[93mComputing Modified NSE (E₁, j=1)...\033[0m")

e1_value = nse_j(BASST['BAsat'], BASST['BAmod'], j=1)
print(f"Modified NSE (E₁, j=1) = {e1_value:.4f}")

e1_monthly = monthly_nse_j(BASSTmonthly, j=1)

for i, val in enumerate(e1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified NSE (E₁) = {val:.4f}")

print("-" * 45)

# ----- Modified Index of Agreement (d₁, j=1) -----
print("\033[93mComputing Modified Index of Agreement (d₁, j=1)...\033[0m")

d1_value = index_of_agreement_j(BASST['BAsat'], BASST['BAmod'], j=1)
print(f"Modified Index of Agreement (d₁, j=1) = {d1_value:.4f}")

d1_monthly = monthly_index_of_agreement_j(BASSTmonthly, j=1)

for i, val in enumerate(d1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified Index of Agreement (d₁) = {val:.4f}")

print("-" * 45)

# ----- Relative Nash–Sutcliffe Efficiency (E_rel) -----
print("\033[93mComputing Relative NSE (E_rel)...\033[0m")

mask = ~np.isnan(BASST['BAsat']) & ~np.isnan(BASST['BAmod']) & (BASST['BAsat'] != 0)
e_rel_value = relative_nse(BASST['BAsat'][mask], BASST['BAmod'][mask])
print(f"Relative NSE (E_rel) = {e_rel_value:.4f}")

e_rel_monthly = monthly_relative_nse(BASSTmonthly)

for i, val in enumerate(e_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative NSE = {val:.4f}")

print("-" * 45)

# ----- Relative Index of Agreement (d_rel) -----
print("\033[93mComputing Relative Index of Agreement (d_rel)...\033[0m")

mask = ~np.isnan(BASST['BAsat']) & ~np.isnan(BASST['BAmod']) & (BASST['BAsat'] != 0)
d_rel_value = relative_index_of_agreement(BASST['BAsat'][mask], BASST['BAmod'][mask])
print(f"Relative Index of Agreement (d_rel) = {d_rel_value:.4f}")

d_rel_monthly = monthly_relative_index_of_agreement(BASSTmonthly)

for i, val in enumerate(d_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative Index of Agreement = {val:.4f}")

print("-" * 45)

print("\033[92m✅ All of the metrcis have been computed!\033[0m")
print('*'*45)

###############################################################################
##                                                                           ##
##                           EFFICIENCY PLOTS                                ##
##                                                                           ##
###############################################################################

print("Plotting the efficiency metrcis results...")

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "SST", timestamp)
os.makedirs(output_path, exist_ok=True)

# ----- Plot Coefficient of Determination (r²) ----- 
plot_metric('Coefficient of Determination (r²)', r2_value, r2_monthly, 'r² Value', output_path)
print("\033[92m✅ Coefficient of determination plotted!\033[0m")

# ----- Plot Weighted Coefficient of Determination (wr²) ----- 
plot_metric('Weighted Coefficient of Determination (wr²)', wr2_value, wr2_monthly, 'wr² Value', output_path)
print("\033[92m✅ Weighted coefficient of determination plotted!\033[0m")

# ----- Plot Nash-Sutcliffe Efficiency (NSE) ----- 
plot_metric('Nash-Sutcliffe Efficiency', nse_value, nse_monthly, 'NSE Value', output_path)
print("\033[92m✅ Nash-Sutcliffe index plotted!")

# ----- Plot Index of Agreement (d) ----- 
plot_metric('Index of Agreement (d)', d_value, d_monthly, 'Index of Agreement (d)', output_path)
print("\033[92m✅ Index of Aggreement plotted!\033[0m")

# ----- Plot Logarithmic Nash–Sutcliffe Efficiency (ln NSE) ----- 
plot_metric('Nash-Sutcliffe Efficiency (Logarithmic)', ln_nse_value, ln_nse_monthly, 'ln NSE Value', output_path)
print("\033[92m✅ Logarithmic Nash-Sutcliffe index plotted!\033[0m")

# ----- Plot Modified Nash–Sutcliffe Efficiency (E₁, j=1) ----- 
plot_metric('Modified NSE (E₁, j=1)', e1_value, e1_monthly, 'E₁ Value', output_path)
print("\033[92m✅ Modified Nash-Sutcliffe index plitted!\033[0m")

# ----- Plot Modified Index of Agreement (d₁, j=1) ----- 
plot_metric('Modified Index of Agreement (d₁, j=1)', d1_value, d1_monthly, 'd₁ Value', output_path)
print("\033[92m✅ Modified index of aggreement plotted!\033[0m")

# ----- Plot Relative Nash–Sutcliffe Efficiency (E_rel) ----- 
plot_metric('Relative NSE ($E_{rel}$)', e_rel_value, e_rel_monthly, 'E_rel Value', output_path)
print("\033[92m✅ Relative Nash-Sutcliffe index plotted!\033[0m")

# ----- Plot Relative Index of Agreement (d_rel) ----- 
plot_metric('Relative Index of Agreement ($d_{rel}$)', d_rel_value, d_rel_monthly, 'd_rel Value', output_path)
print("\033[92m✅ Relative index of aggreement plotted!")

print("\033[92m✅ The efficiency metrcis plots have been succesfully created!\033[0m")
print("*"*45)