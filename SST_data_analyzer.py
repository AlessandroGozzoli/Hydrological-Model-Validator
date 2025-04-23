import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import calendar

############### MODULES LOADING ###############

import sys

print("Loading the necessary modules...")
CodingDIR = "C:/Tesi Magistrale/Codici/Python/"
os.chdir(CodingDIR)  # Set the working directory

print("Loading the Pre-Processing modules and constants...")
ProcessingDIR = Path(CodingDIR, "Processing/")
sys.path.append(str(ProcessingDIR))  # Add the folder to the system path

from Costants import ysec
from Leap_year import convert_to_monthly_data

print("Pre-processing modules have been loaded!")
print("-"*45)

print("Loading the plotting modules...")
PlottingDIR = Path(CodingDIR, "Plotting")
sys.path.append(str(PlottingDIR))  # Add the folder to the system path

from Daily_means_timeseries import plot_daily_means, plot_metric
from Scatter_plot import scatter_plot_BASST
from Taylor_diagrams import (
                             comprehensive_taylor_diagram,
                             monthly_taylor_diagram
                             )
from Target_plots import (
                          comprehensive_target_diagram,
                          target_diagram_by_month
                          )

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

taylor_options = str(Path(PlottingDIR, 'taylor_option_config.csv'))
taylor_options_monthly = str(Path(PlottingDIR, 'taylor_option_config - monthly.csv'))

print("The plotting modules have been loaded!")
print('-'*45)

print("Loading the validation modules...")
ValidationDIR = Path(CodingDIR, "Validation")
print("The validation modules have been loaded!")
print('*'*45)

# ----- SETTING UP THE WORKING DIRECTOTY -----
print("Resetting the working directory...")
WDIR = "C:/Tesi Magistrale/"
os.chdir(WDIR)  # Set the working directory
print('*'*45)

# ----- BASE DATA DIRECTORY -----
BDIR = Path(WDIR, "Dati")

# ----- INPUT DATA DIRECTORY -----
IDIR = Path(BDIR, "PROCESSING_INPUT/")
print("Loading the input data...")
print(f"!!! The input data needs to be located in the {IDIR} folder !!!")
print("!!! Make sure that it contains all of the necessary datasets !!!")
print("-"*45)

print("The folder contains the following datasets")
# List the contents of the folder
contents = os.listdir(IDIR)
# Print the contents
print(contents)
print("*"*45)

# ----- SETTING UP THE DICTIONARIES TO KEEP THE DATA -----

# ----- SETTING UP THE SST DICTIONARY

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

print("Model SST obtained!")

# Moving onto the satellite SST data

print("Getting the satellite SST data...")
SAT_SST = xr.open_dataset(Path(IDIR, 'Sat_sst.nc'))
print("Satellite SST obtained!")

SST = {
       'MODEL' : Msst_data,
       'SATELLITE' : SAT_SST
       }

print("-"*45)
print("SST dictionary created!")
print("All of the dictionaries have been created!")
print("*"*45)

# ----- IMPORTING BASIN AVERAGES -----

print("Importing the Basin Average SST timeseries")
BASSTmod = xr.open_dataset(Path(IDIR, 'BASSTmod.nc'))
BASSTsat = xr.open_dataset(Path(IDIR, 'BASSTsat.nc'))
BASSTmod = BASSTmod['BASSTmod'].values
BASSTsat = BASSTsat['BASSTsat'].values
print("Basin Average Timseries obtained!")

print("Adding them to a dictionary...")
BASST = {
        'BAmod' : BASSTmod,
        'BAsat' : BASSTsat
        }
print("Basin Average dictionary created!")
print('*'*45)

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

# Conversion to a dictionary divided in months
BASSTmod_monthly_dict = convert_to_monthly_data(BASSTmod_yearly)
BASSTsat_monthly_dict = convert_to_monthly_data(BASSTsat_yearly)

BASSTmonthly = {
                'BASSTmod_monthly' : BASSTmod_monthly_dict,
                'BASSTsat_monthly' : BASSTsat_monthly_dict
                }

std_ref = np.std(BASSTsat)

# Plotting the Taylor Diagram
print("Plotting the Taylor diagrams...")
print("-"*45)

print("Plotting the SST Taylor diagram for yearly data...")
comprehensive_taylor_diagram(BASST_yearly, taylor_options, std_ref)
print("Yearly data Taylor diagram has been plotted!")
print("-"*45)

print("Plotting the monthly data diagrams...")
for i in range(12):
    monthly_taylor_diagram(BASSTmonthly, i, taylor_options_monthly)
print("All of the Taylor diagrams have been plotted!")
print("*"*45)

# Making the target plots
print("Plotting the Target plots...")
print("-"*45)

print("Plotting the Target plot for the yearly data...")
comprehensive_target_diagram(BASST_yearly)
print("Yearly data Target plot has been plotted!")
print("-"*45)

print("Plotting the monthly data plots...")
for i in range(12):
    target_diagram_by_month(BASSTmonthly, i)
print("All of the Target plots has been plotted!")
print("*"*45)

# Time series and scatter plots
BIAS_Bavg = BASSTsat - BASSTmod

print("Plotting the time-series...")
plot_daily_means(BASST, 'SST', BIAS_Bavg, BA=True)
print("Time-series plotted succesfully!")

print("Plotting the scatter plot...")
scatter_plot_BASST(BASST, 'SST', BA=False)
print("Scatter plot plotted succesfully!")
print("*"*45)

################################################
# ----- COMPUTING THE EFFICIENCY METRCIS ----- #
################################################
print("Computing the Efficiency Metrics...")
print("They are based onto Krause et al. 2005")
print("-"*45)

# ----- Coefficient of determination -----
print("Computing Coefficient of Determination (r²)...")

r2_value = r_squared(BASST['BAsat'], BASST['BAmod'])
print(f"r² (Coefficient of Determination) = {r2_value:.4f}")

r2_monthly = monthly_r_squared(BASSTmonthly)

for i, val in enumerate(r2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: r² = {val:.4f}")

print("-" * 45)

# ----- Weighted Coeddificient of Determination -----
print("Computing Weighted Coefficient of Determination (wr²)...")

wr2_value = weighted_r_squared(BASST['BAsat'], BASST['BAmod'])
print(f"wr² (Weighted Coefficient of Determination) = {wr2_value:.4f}")

wr2_monthly = monthly_weighted_r_squared(BASSTmonthly)

for i, val in enumerate(wr2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: wr² = {val:.4f}")

print("-" * 45)

# ----- Nash-Sutcliffe -----
print("Computing Nash–Sutcliffe Efficiency (NSE)...")

nse_value = nse(BASST['BAsat'], BASST['BAmod'])
print(f"NSE (Nash–Sutcliffe Efficiency) = {nse_value:.4f}")

nse_monthly = monthly_nse(BASSTmonthly)

for i, val in enumerate(nse_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: NSE = {val:.4f}")

print("-" * 45)

# ----- Index of Agreement -----
print("Computing Index of Agreement (d)...")

d_value = index_of_agreement(BASST['BAsat'], BASST['BAmod'])
print(f"Index of Agreement (d) = {d_value:.4f}")

d_monthly = monthly_index_of_agreement(BASSTmonthly)

for i, val in enumerate(d_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: Index of Agreement (d) = {val:.4f}")

print("-" * 45)

# ----- Logarithmic Nash–Sutcliffe Efficiency (ln NSE) -----
print("Computing NSE with Logarithmic Values (ln NSE)...")

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
print("Computing Modified NSE (E₁, j=1)...")

e1_value = nse_j(BASST['BAsat'], BASST['BAmod'], j=1)
print(f"Modified NSE (E₁, j=1) = {e1_value:.4f}")

e1_monthly = monthly_nse_j(BASSTmonthly, j=1)

for i, val in enumerate(e1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified NSE (E₁) = {val:.4f}")

print("-" * 45)

# ----- Modified Index of Agreement (d₁, j=1) -----
print("Computing Modified Index of Agreement (d₁, j=1)...")

d1_value = index_of_agreement_j(BASST['BAsat'], BASST['BAmod'], j=1)
print(f"Modified Index of Agreement (d₁, j=1) = {d1_value:.4f}")

d1_monthly = monthly_index_of_agreement_j(BASSTmonthly, j=1)

for i, val in enumerate(d1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified Index of Agreement (d₁) = {val:.4f}")

print("-" * 45)

# ----- Relative Nash–Sutcliffe Efficiency (E_rel) -----
print("Computing Relative NSE (E_rel)...")

mask = ~np.isnan(BASST['BAsat']) & ~np.isnan(BASST['BAmod']) & (BASST['BAsat'] != 0)
e_rel_value = relative_nse(BASST['BAsat'][mask], BASST['BAmod'][mask])
print(f"Relative NSE (E_rel) = {e_rel_value:.4f}")

e_rel_monthly = monthly_relative_nse(BASSTmonthly)

for i, val in enumerate(e_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative NSE = {val:.4f}")

print("-" * 45)

# ----- Relative Index of Agreement (d_rel) -----
print("Computing Relative Index of Agreement (d_rel)...")

mask = ~np.isnan(BASST['BAsat']) & ~np.isnan(BASST['BAmod']) & (BASST['BAsat'] != 0)
d_rel_value = relative_index_of_agreement(BASST['BAsat'][mask], BASST['BAmod'][mask])
print(f"Relative Index of Agreement (d_rel) = {d_rel_value:.4f}")

d_rel_monthly = monthly_relative_index_of_agreement(BASSTmonthly)

for i, val in enumerate(d_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative Index of Agreement = {val:.4f}")

print("-" * 45)

print("All of the metrcis have been computed!")
print('*'*45)

####################################
# ----- PLOTTING THE RESULTS ----- #
####################################
print("Plotting the efficiency metrcis results...")

# ----- Plot Coefficient of Determination (r²) ----- 
plot_metric('Coefficient of Determination (r²)', r2_value, r2_monthly, 'r² Value')
print("Coefficient of determination plotted!")

# ----- Plot Weighted Coefficient of Determination (wr²) ----- 
plot_metric('Weighted Coefficient of Determination (wr²)', wr2_value, wr2_monthly, 'wr² Value')
print("Weighted coefficient of determination plotted!")

# ----- Plot Nash-Sutcliffe Efficiency (NSE) ----- 
plot_metric('Nash-Sutcliffe Efficiency', nse_value, nse_monthly, 'NSE Value')
print("Nash-Sutcliffe index plotted!")

# ----- Plot Index of Agreement (d) ----- 
plot_metric('Index of Agreement (d)', d_value, d_monthly, 'Index of Agreement (d)')
print("Index of Aggreement plotted!")

# ----- Plot Logarithmic Nash–Sutcliffe Efficiency (ln NSE) ----- 
plot_metric('Nash-Sutcliffe Efficiency (Logarithmic)', ln_nse_value, ln_nse_monthly, 'ln NSE Value')
print("Logarithmic Nash-Sutcliffe index plotted!")

# ----- Plot Modified Nash–Sutcliffe Efficiency (E₁, j=1) ----- 
plot_metric('Modified NSE (E₁, j=1)', e1_value, e1_monthly, 'E₁ Value')
print("Modified Nash-Sutcliffe index plitted!")

# ----- Plot Modified Index of Agreement (d₁, j=1) ----- 
plot_metric('Modified Index of Agreement (d₁, j=1)', d1_value, d1_monthly, 'd₁ Value')
print("Modified index of aggreement plotted!")

# ----- Plot Relative Nash–Sutcliffe Efficiency (E_rel) ----- 
plot_metric('Relative NSE ($E_{rel}$)', e_rel_value, e_rel_monthly, 'E_rel Value')
print("Relative Nash-Sutcliffe index plotted!")

# ----- Plot Relative Index of Agreement (d_rel) ----- 
plot_metric('Relative Index of Agreement ($d_{rel}$)', d_rel_value, d_rel_monthly, 'd_rel Value')
print("Relative index of aggreement plotted!")

print("The efficiency metrcis plots have been succesfully created!")
print("*"*45)