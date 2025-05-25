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

# Utility libraries
import numpy as np
import pandas as pd
import xarray as xr
import calendar
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import re

###############################################################################
##                                                                           ##
##                                MODULES                                    ##
##                                                                           ##
###############################################################################

print("Loading the necessary modules...")

print("Loading the Pre-Processing modules and constants...")
from Hydrological_model_validator.Processing.time_utils import split_to_monthly, split_to_yearly
print("\033[92m✅ Pre-processing modules have been loaded!\033[0m")
print("-"*45)

print("Loading the file I/O modules and constants...")
from Hydrological_model_validator.Processing.file_io import load_dataset
print("\033[92m✅ File I/O modules have been loaded!\033[0m")
print("-"*45)

print("Loading the utility functions...")
from Hydrological_model_validator.Processing.utils import infer_years_from_path
print("\033[92m✅ File I/O modules have been loaded!\033[0m")
print("-"*45)

print("Loading the plotting modules...")
from Hydrological_model_validator.Plotting.Plots import (timeseries,
                                                           scatter_plot,
                                                           seasonal_scatter_plot,
                                                           whiskerbox,
                                                           violinplot,
                                                           efficiency_plot)
from Hydrological_model_validator.Plotting.Taylor_diagrams import (comprehensive_taylor_diagram,
                                                                   monthly_taylor_diagram)
from Hydrological_model_validator.Plotting.Target_plots import (comprehensive_target_diagram,
                                                                target_diagram_by_month)
print("\033[92m✅ The plotting modules have been loaded!\033[0m")
print('-'*45)

print("Loading the validation modules...")
from Hydrological_model_validator.Processing.Efficiency_metrics import (r_squared,
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
                                                                        monthly_relative_index_of_agreement)
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

# ----- INFER YEARS FROM FILE NAMES -----
print("Scanning directory to determine available years...")
Ybeg, Yend, ysec = infer_years_from_path(IDIR, target_type="file", pattern=r'_(\d{4})\.nc$')

print("Setting up the SST dictionary...")
print('-' * 45)

print(f"Years detected: {ysec}")
print("Getting the yearly model SST datasets...")

with ThreadPoolExecutor() as executor:
    results = list(executor.map(load_dataset, ysec, repeat(IDIR)))

Msst_data = {year: ds for year, ds in results if ds is not None}

# ----- IMPORTING BASIN AVERAGES -----

print("Importing the Basin Average SST timeseries...")
idir_path = Path(IDIR)
BASSTmod = xr.open_dataset(idir_path / 'BASSTmod.nc')['BASSTmod'].values
BASSTsat = xr.open_dataset(idir_path / 'BASSTsat.nc')['BASSTsat'].values
print("\033[92m✅ Basin Average Timeseries obtained!\033[0m")

# Generate datetime index (Daily from 2000 to 2009)
dates = pd.date_range(start='2000-01-01', end='2009-12-31', freq='D')

# ----- CREATING DATETIME-INDEXED SERIES -----

BASSTmod_series = pd.Series(BASSTmod, index=dates)
BASSTsat_series = pd.Series(BASSTsat, index=dates)

# ----- BUILDING DICTIONARY WITH DATETIME INDEX -----

print("Adding them to a dictionary...")
BASST = {
    'BASSTsat': BASSTsat_series,
    'BASSTmod': BASSTmod_series
}
print("\033[92m✅ Basin Average dictionary created!\033[0m")
print('*' * 45)

# ----- SPLITTING INTO YEARLY DATASETS -----

print("Splitting the data to better handle it...")

print("Creating the yearly datasets...")
BASST_yearly = {}

for key in BASST:
    BASST_yearly[key] = split_to_yearly(BASST[key], ysec)
    
print("Yearly dataset created!")

# ----- SPLITTING INTO MONTHLY DATASETS -----

print("Creating the yearly datasets...")
BASST_monthly = {}

for key in BASST:
    # Split the yearly data into months
    BASST_monthly[key] = split_to_monthly(BASST_yearly[key])

print("Monthly dataset created!")

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
##                              OTHER PLOTS                                  ##
##                                                                           ##
###############################################################################

print("Plotting remaining plots...")

# ----- CREATE THE FOLDER TO SAVE THE PLOTS -----

timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "OTHER", "SST", timestamp)
os.makedirs(output_path, exist_ok=True)

# ----- TIMESERIES PLOTS -----

print("Computing the BIAS...")
BIAS_Bavg = BASSTmod - BASSTsat
BIAS = pd.Series(BIAS_Bavg, index=dates)
print("\033[92m✅ BIAS computed! \033[0m")
print("-"*45)

print("Plotting the time-series...")
# Defaylt options
timeseries(BASST, BIAS, variable_name='SST', BA=True, output_path=output_path)

print("\033[92m✅ Time-series plotted succesfully!\033[0m")

# ----- SCATTERPLOTS -----

print("Plotting the scatter plot...")
# Defaylt options
scatter_plot(BASST, variable_name='SST', BA=False, output_path=output_path)
print("\033[92m✅ Scatter plot plotted succesfully!\033[0m")

print("Plotting the seasonal data as scatterplots...")
# Defaylt options
seasonal_scatter_plot(BASST, variable_name='SST', BA=False, output_path=output_path)
print("\033[92m✅ Seasonal scatterplots plotted succesfully!\033[0m")

# ----- WHISKERBOX PLOTS -----

print("Plotting the whisker-box plots...")
# Defaylt options
whiskerbox(BASST_monthly, variable_name='SST', output_path=output_path)
print("\033[92m✅ Whisker-box plotted succesfully!\033[0m")

# ----- VIOLIN PLOTS -----

print("Plotting the violinplots...")
# Defaylt options
violinplot(BASST_monthly, variable_name='SST', output_path=output_path)
print("\033[92m✅ Violinplots plotted succesfully!\033[0m")

###############################################################################
##                                                                           ##
##                            TAYLOR DIAGRAMS                                ##
##                                                                           ##
###############################################################################

# Plotting the Taylor Diagram
print("Plotting the Taylor diagrams...")
print("-"*45)

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "TAYLOR", "SST", timestamp)
os.makedirs(output_path, exist_ok=True)

print("Plotting the SST Taylor diagram for yearly data...")
comprehensive_taylor_diagram(BASST_yearly, output_path=output_path, variable_name='SST')
print("\033[92m✅ Yearly data Taylor diagram has been plotted!\033[0m")
print("-"*45)

print("Plotting the monthly data diagrams...")
monthly_taylor_diagram(BASST_monthly, output_path=output_path, variable_name='SST')
print("\033[92m✅ Monthly Taylor diagrams have been plotted!\033[0m")
print("-"*45)

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
comprehensive_target_diagram(BASST_yearly, output_path=output_path, variable_name='SST')
print("\033[92m✅ Yearly data Target plot has been plotted!\033[0m")
print("-"*45)

print("Plotting the monthly data plots...")
target_diagram_by_month(BASST_monthly, output_path=output_path, variable_name='SST')
print("\033[92m✅ All of the Target plots has been plotted!\033[0m")
print("*"*45)

###############################################################################
##                                                                           ##
##                          EFFICIENCY METRCIS                               ##
##                                                                           ##
###############################################################################

print("Computing the Efficiency Metrics...")
print("-" * 45)

# Initialize the DataFrame
months = list(calendar.month_name)[1:]  # ['January', ..., 'December']
metrics = ['r²', 'wr²', 'NSE', 'd', 'ln NSE', 'E_1', 'd_1', 'E_{rel}', 'd_{rel}']
efficiency_df = pd.DataFrame(index=metrics, columns=['Total'] + months)

# List of metric functions and display names
metric_functions = [
    ('r²', r_squared, monthly_r_squared),
    ('wr²', weighted_r_squared, monthly_weighted_r_squared),
    ('NSE', nse, monthly_nse),
    ('d', index_of_agreement, monthly_index_of_agreement),
    ('ln NSE', ln_nse, monthly_ln_nse),
    ('E_1', lambda x, y: nse_j(x, y, j=1), lambda m: monthly_nse_j(m, j=1)),
    ('d_1', lambda x, y: index_of_agreement_j(x, y, j=1), lambda m: monthly_index_of_agreement_j(m, j=1)),
    ('E_{rel}', relative_nse, monthly_relative_nse),
    ('d_{rel}', relative_index_of_agreement, monthly_relative_index_of_agreement),
]

# Handle log-transformed and relative metrics (which need filtering)
for name, func, monthly_func in metric_functions:
    print(f"\033[93mComputing {name}...\033[0m")

    if name in ['ln NSE', 'E_rel', 'd_rel']:
        mask = ~np.isnan(BASST['BASSTsat']) & ~np.isnan(BASST['BASSTmod'])

        if name == 'ln NSE':
            mask &= (BASST['BASSTsat'] > 0) & (BASST['BASSTmod'] > 0)
        if name in ['E_rel', 'd_rel']:
            mask &= BASST['BASSTsat'] != 0

        x = BASST['BASSTsat'][mask]
        y = BASST['BASSTmod'][mask]
    else:
        x = BASST['BASSTsat']
        y = BASST['BASSTmod']

    total_val = func(x, y)
    monthly_vals = monthly_func(BASST_monthly)

    # Store in DataFrame
    efficiency_df.loc[name, 'Total'] = total_val
    efficiency_df.loc[name, months] = monthly_vals

    # Print values
    print(f"{name} (Total) = {total_val:.4f}")
    for month, val in zip(months, monthly_vals):
        print(f"{month}: {name} = {val:.4f}")

    print("-" * 45)

print("\033[92m✅ All of the metrics have been computed!\033[0m")
print("*" * 45)

###############################################################################
##                                                                           ##
##                           EFFICIENCY PLOTS                                ##
##                                                                           ##
###############################################################################

print("Plotting the efficiency metrics results...")

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "SST", timestamp)
os.makedirs(output_path, exist_ok=True)

# Mapping of display names for plots
plot_titles = {
    'r²': 'Coefficient of Determination (r²)',
    'wr²': 'Weighted Coefficient of Determination (wr²)',
    'NSE': 'Nash-Sutcliffe Efficiency',
    'd': 'Index of Agreement (d)',
    'ln NSE': 'Nash-Sutcliffe Efficiency (Logarithmic)',
    'E_1': 'Modified NSE ($E_1$, j=1)',
    'd_1': 'Modified Index of Agreement ($d_1$, j=1)',
    'E_{rel}': r'Relative NSE ($E_{rel}$)',
    'd_{rel}': r'Relative Index of Agreement ($d_{rel}$)',
}

# Plotting all metrics in a loop
for metric_key, title in plot_titles.items():
    total_value = efficiency_df.loc[metric_key, 'Total']
    monthly_values = efficiency_df.loc[metric_key, efficiency_df.columns[1:]].values.astype(float)

    efficiency_plot(total_value, monthly_values, 
                    title=f'{title}', 
                    y_label=f'{metric_key}', 
                    output_path=output_path)
    
    # Remove any parentheses and their contents for the print message
    clean_title = re.sub(r'\s*\([^)]*\)', '', title)
    print(f"\033[92m✅ {clean_title} plotted!\033[0m")

print("\033[92m✅ All efficiency metric plots have been successfully created!\033[0m")
print("*" * 45)