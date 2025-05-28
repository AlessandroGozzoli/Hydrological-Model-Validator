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
##    This code retrieves the data from the setupper (CHL specifically)      ##
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
from Hydrological_model_validator.Processing.file_io import mask_reader
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
BaseDIR = Path(WDIR, "Data")

# ----- INPUT DATA DIRECTORY -----
IDIR = Path(BaseDIR, "PROCESSING_INPUT/")
print("Loading the input data...")
print(f"\033[91m⚠️ The input data needs to be located in the {IDIR} folder ⚠️\033[0m")
print("\033[91m⚠️ Make sure that it contains all of the necessary datasets ⚠️\033[0m")
print("\033[91m⚠️ Make sure that it contains the mask ⚠️\033[0m")
print("-"*45)

print("The folder contains the following datasets")
# List the contents of the folder
contents = os.listdir(IDIR)
# Print the contents
print(contents)
print("*"*45)

print("Retrieving the mask...")
# Call the function and extract values
Mmask, Mfsm, Mfsm_3d, Mlat, Mlon = mask_reader(BaseDIR)
print("\033[92m✅ Mask succesfully imported! \033[0m")
print('*'*45)

print("Importing the datasets...")
Mchll3 = xr.open_dataset(Path(IDIR, 'ModData_chl_interp_l3.nc'))
Schll3 = xr.open_dataset(Path(IDIR, 'SatData_chl_interp_l3.nc'))
print("\033[92m✅ Datasets imported! \033[0m")
print("*"*45)

print("\033[91m⚠️ HEADS UP ⚠️")
print("To ensure that the code runs smoothly the plots will be")
print("displayed only for 3 seconds. This time can be changed")
print("in the script. After the plot's window closes it will be")
print("saved in the appropriate folder for further analysis.\033[0m")
confirm = input("Please press any key to confirm and move on: ")

###############################################################################
##                                                                           ##
##                                 LEVEL 3                                   ##
##                                                                           ##
###############################################################################

###############################################################################
##                                                                           ##
##                              DICTIONARIES                                 ##
##                                                                           ##
###############################################################################

print("Setting up the level 3 datasets for the analysis...")
print('-' * 45)

# ----- IMPORTING LEVEL 3 DATASETS -----

print("Importing Level 3 model and satellite datasets...")
idir_path = Path(IDIR)
# Get the datasets, need to be changed if the L4 data is used
BACHLmod = xr.open_dataset(idir_path / 'BA_chl_mod_L3.nc')['BAmod_L3'].values
BACHLsat = xr.open_dataset(idir_path / 'BA_chl_sat_l3.nc')['BAsat_L3'].values
print("\033[92m✅ Level 3 datasets loaded!\033[0m")

# ----- CREATING DATETIME-INDEXED SERIES -----

print("Generating datetime-indexed series...")
# Define the dates range, needs to be user defined
dates = pd.date_range(start='2000-01-01', end='2009-12-31', freq='D')
BACHLmod_series = pd.Series(BACHLmod, index=dates)
BACHLsat_series = pd.Series(BACHLsat, index=dates)

# ----- BUILDING DICTIONARY -----

print("Adding them to a dictionary...")
BACHL = {
    'BACHLmod': BACHLmod_series,
    'BACHLsat': BACHLsat_series
}
print("\033[92m✅ 3D dictionary created!\033[0m")
print("-" * 45)

# ----- SPLITTING INTO YEARLY DATASETS -----

print("Splitting the data to better handle it...")

print("Creating the yearly datasets...")
BACHL_yearly = {}

# Define the number of years in which to split the dataset
Ybeg, Yend = dates[0].year, dates[-1].year
ysec = list(range(Ybeg, Yend + 1))

for key in BACHL:
    BACHL_yearly[key] = split_to_yearly(BACHL[key], ysec)

print("\033[92m✅ Yearly dictionary created!\033[0m")

# ----- SPLITTING INTO MONTHLY DATASETS -----

print("Creating the monthly datasets...")
BACHLmonthly = {}

for key in BACHL:
    BACHLmonthly[key] = split_to_monthly(BACHL_yearly[key])

print("\033[92m✅ Monthly dictionary created!\033[0m")
print("*" * 45)

###############################################################################
##                                                                           ##
##                                  PLOTS                                    ##
##                                                                           ##
###############################################################################

print("Beginning to plot...")
print("-"*45)

# ----- CREATE THE FOLDER TO SAVE THE PLOTS -----

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "OTHER", "CHL", "l3", timestamp)
os.makedirs(output_path, exist_ok=True)

# ----- TIMESERIES PLOTS -----

print("Computing the BIAS...")
BIAS_Bavg = BACHLsat - BACHLmod
print("\033[92m✅ BIAS computed! \033[0m")
print("-"*45)

print("Plotting the timeseries...")
timeseries(BACHL, BIAS_Bavg, output_path=output_path, variable_name='CHL_L3', BA=True)
print("\033[92m✅ Time series plotted! \033[0m")

# ----- SCATTERPLOTS -----

print("Plotting the scatter plot...")
scatter_plot(BACHL, output_path=output_path, variable_name='CHL_L3', BA=False)
print("\033[92m✅ Scatter plot succesfully plotted! \033[0m")

print("Plotting the seasonal data as scatterplots...")
seasonal_scatter_plot(BACHL, output_path=output_path, variable_name='CHL_L3', BA=False)
print("\033[92m✅ Seasonal scatterplots plotted succesfully!\033[0m")
print('*'*45)

# ----- WHISKERBOX PLOTS -----

print("Plotting the whisker-box plots...")
whiskerbox(BACHLmonthly, output_path=output_path, variable_name='CHL_L3')
print("\033[92m✅ Whisker-box plotted succesfully!\033[0m")

# ----- VIOLIN PLOTS -----

print("Plotting the violinplots...")
violinplot(BACHLmonthly, output_path=output_path, variable_name='CHL_L3')
print("\033[92m✅ Violinplots plotted succesfully!\033[0m")

###############################################################################
##                                                                           ##
##                             TAYLOR DIAGRAMS                               ##
##                                                                           ##
###############################################################################

print("Plotting the Taylor diagrams...")

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "TAYLOR", "CHL", "l3", timestamp)
os.makedirs(output_path, exist_ok=True)

# Plotting the Taylor Diagram
comprehensive_taylor_diagram(BACHL_yearly, output_path=output_path, variable_name='CHL_L3')
print("\033[92m✅ Yearly Taylor diagram plotted! \033[0m")

monthly_taylor_diagram(BACHLmonthly, output_path=output_path, variable_name='CHL_L3')
print("\033[92m✅ All of the monthly Taylor diagrams have been plotted! \033[0m")
print("-"*45)

###############################################################################
##                                                                           ##
##                               TARGET PLOTS                                ##
##                                                                           ##
###############################################################################

print("Beginning to create the Target plots...")

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "TARGET", "CHL", "l3", timestamp)
os.makedirs(output_path, exist_ok=True)

comprehensive_target_diagram(BACHL_yearly, output_path=output_path, variable_name='CHL_L3')
print("\033[92m✅ Yearly target plot has been made! \033[0m")

target_diagram_by_month(BACHLmonthly, output_path=output_path, variable_name='CHL_L3')
print("\033[92m✅ All of the monthly target plots have been made! \033[0m")
print("*"*45)

###############################################################################
##                                                                           ##
##                          EFFICIENCY METRCIS                               ##
##                                                                           ##
###############################################################################

print("Computing the Efficiency Metrics...")
print("-" * 45)

# Initialize
months = list(calendar.month_name)[1:]  # ['January', ..., 'December']
metrics = ['r²', 'wr²', 'NSE', 'd', 'ln NSE', 'E_1', 'd_1', 'E_{rel}', 'd_{rel}']
efficiency_df = pd.DataFrame(index=metrics, columns=['Total'] + months)

# List of metric functions and names
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

# Metric computation loop
for name, func, monthly_func in metric_functions:
    print(f"\033[93mComputing {name}...\033[0m")

    if name in ['ln NSE', 'E_rel', 'd_rel']:
        mask = ~np.isnan(BACHL['BACHLsat']) & ~np.isnan(BACHL['BACHLmod'])

        if name == 'ln NSE':
            mask &= (BACHL['BACHLsat'] > 0) & (BACHL['BACHLmod'] > 0)
        if name in ['E_rel', 'd_rel']:
            mask &= BACHL['BACHLsat'] != 0

        x = BACHL['BACHLsat'][mask]
        y = BACHL['BACHLmod'][mask]
    else:
        x = BACHL['BACHLsat']
        y = BACHL['BACHLmod']

    total_val = func(x, y)
    monthly_vals = monthly_func(BACHLmonthly)

    efficiency_df.loc[name, 'Total'] = total_val
    efficiency_df.loc[name, months] = monthly_vals

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
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "CHL", "l3", timestamp)
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

# Plotting all metrics in a loop using efficiency_df
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

print("\033[92m✅ All efficiency metrics plots have been successfully created!\033[0m")
print("*" * 45)
