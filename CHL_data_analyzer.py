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

from Corollary import convert_to_monthly_data, mask_reader

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
Mchll3 = xr.open_dataset(Path(IDIR, 'Mchl_interp_l3.nc'))
Mchll4 = xr.open_dataset(Path(IDIR, 'Mchl_interp_l4.nc'))
Schll3 = xr.open_dataset(Path(IDIR, 'Schl_interp_l3.nc'))
Schll4 = xr.open_dataset(Path(IDIR, 'Schl_interp_l4.nc'))
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

BACHLmod_L3 = xr.open_dataset(Path(IDIR, 'BACHLmod_L3.nc'))
BACHLsat_L3 = xr.open_dataset(Path(IDIR, 'BACHLsat_L3.nc'))
BACHLmod_L3 = BACHLmod_L3['BACHLmod_L3'].values
BACHLsat_L3 = BACHLsat_L3['BACHLsat_L3'].values

BACHL_L3 = {
            'BACHLmod_L3' : BACHLmod_L3,
            'BACHLsat_L3' : BACHLsat_L3
            }
print("\033[92m✅ 3D dictionary created! \033[0m")
print("-"*45)

print("Generating the yearly and monthly datasets...")
# Generate datetime index
dates = pd.date_range(start='2000-01-01', end='2009-12-31', freq='D')
years = np.array([d.year for d in dates])
unique_years = np.unique(years)

# Split data into a list of arrays, one per year
mod_split = [BACHLmod_L3[years == y] for y in unique_years]
sat_split = [BACHLsat_L3[years == y] for y in unique_years]

# Find max days in any year (i.e., 366)
max_days = max(len(year_data) for year_data in mod_split)

# Pad shorter years with NaNs to make uniform arrays
BACHLmod_L3_yearly = np.array([np.pad(year_data, (0, max_days - len(year_data)), constant_values=np.nan)
                              for year_data in mod_split])
BACHLsat_L3_yearly = np.array([np.pad(year_data, (0, max_days - len(year_data)), constant_values=np.nan)
                              for year_data in sat_split])

BACHL_yearly_L3 = {
                'BAmod_year' : BACHLmod_L3_yearly,
                'BAsat_year' : BACHLsat_L3_yearly
                }
print("\033[92m✅ Yearly dictionary created! \033[0m")

# Conversion to a dictionary divided in months
BACHLmod_L3_monthly_dict = convert_to_monthly_data(BACHLmod_L3_yearly)
BACHLsat_L3_monthly_dict = convert_to_monthly_data(BACHLsat_L3_yearly)

BACHLmonthly_L3 = {
                'BACHLmod_L3_monthly' : BACHLmod_L3_monthly_dict,
                'BACHLsat_L3_monthly' : BACHLsat_L3_monthly_dict
                }
print("\033[92m✅ Monthly dictionary created! \033[0m")
print("*"*45)

###############################################################################
##                                                                           ##
##                                  PLOTS                                    ##
##                                                                           ##
###############################################################################

print("Beginning to plot...")
print("-"*45)

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "OTHER", "CHL", "l3", timestamp)
os.makedirs(output_path, exist_ok=True)

# 1. BIAS, TIMESERIES AND SCATTERPLOTS
print("Computing the BIAS...")
BIAS_Bavg = BACHLsat_L3 - BACHLmod_L3
print("\033[92m✅ BIAS computed! \033[0m")
print("-"*45)

print("Plotting the timeseries...")
plot_daily_means(output_path, BACHL_L3, 'CHL', BIAS_Bavg, BA=True)
print("\033[92m✅ Time series plotted! \033[0m")

print("Plotting the scatter plot...")
scatter_plot(output_path, BACHL_L3, 'CHL', BA=False)
print("\033[92m✅ Scatter plot succesfully plotted! \033[0m")
print("-"*45)

###############################################################################
##                                                                           ##
##                             TAYLOR DIAGRAMS                               ##
##                                                                           ##
###############################################################################

print("Plotting the Taylor diagrams...")
std_ref = np.nanstd(BACHLsat_L3)

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "TAYLOR", "CHL", "l3", timestamp)
os.makedirs(output_path, exist_ok=True)

# Plotting the Taylor Diagram
comprehensive_taylor_diagram(BACHL_yearly_L3, taylor_options, std_ref, output_path)
print("\033[92m✅ Yearly Taylor diagram plotted! \033[0m")

for i in range(12):
    monthly_taylor_diagram(BACHLmonthly_L3, i, taylor_options_monthly, output_path)
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

comprehensive_target_diagram(BACHL_yearly_L3, output_path)
print("\033[92m✅ Yearly target plot has been made! \033[0m")

for i in range(12):
    target_diagram_by_month(BACHLmonthly_L3, i, output_path)
print("\033[92m✅ All of the monthly target plots have been made! \033[0m")
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

r2_value = r_squared(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'])
print(f"r² (Coefficient of Determination) = {r2_value:.4f}")

r2_monthly = monthly_r_squared(BACHLmonthly_L3)

for i, val in enumerate(r2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: r² = {val:.4f}")

print("-" * 45)

# ----- Weighted Coeddificient of Determination -----
print("\033[93mComputing Weighted Coefficient of Determination (wr²)...\033[0m")

wr2_value = weighted_r_squared(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'])
print(f"wr² (Weighted Coefficient of Determination) = {wr2_value:.4f}")

wr2_monthly = monthly_weighted_r_squared(BACHLmonthly_L3)

for i, val in enumerate(wr2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: wr² = {val:.4f}")

print("-" * 45)

# ----- Nash-Sutcliffe -----
print("\033[93mComputing Nash–Sutcliffe Efficiency (NSE)...\033[0m")

nse_value = nse(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'])
print(f"NSE (Nash–Sutcliffe Efficiency) = {nse_value:.4f}")

nse_monthly = monthly_nse(BACHLmonthly_L3)

for i, val in enumerate(nse_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: NSE = {val:.4f}")

print("-" * 45)

# ----- Index of Agreement -----
print("\033[93mComputing Index of Agreement (d)...\033[0m")

d_value = index_of_agreement(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'])
print(f"Index of Agreement (d) = {d_value:.4f}")

d_monthly = monthly_index_of_agreement(BACHLmonthly_L3)

for i, val in enumerate(d_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: Index of Agreement (d) = {val:.4f}")

print("-" * 45)

# ----- Logarithmic Nash–Sutcliffe Efficiency (ln NSE) -----
print("\033[93mComputing NSE with Logarithmic Values (ln NSE)...\033[0m")

# Remove NaNs and ensure values are positive before computing
mask = ~np.isnan(BACHL_L3['BACHLsat_L3']) & ~np.isnan(BACHL_L3['BACHLmod_L3']) & \
       (BACHL_L3['BACHLsat_L3'] > 0) & (BACHL_L3['BACHLmod_L3'] > 0)

ln_nse_value = ln_nse(BACHL_L3['BACHLsat_L3'][mask], BACHL_L3['BACHLmod_L3'][mask])
print(f"ln NSE = {ln_nse_value:.4f}")

ln_nse_monthly = monthly_ln_nse(BACHLmonthly_L3)

for i, val in enumerate(ln_nse_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: ln NSE = {val:.4f}")

print("-" * 45)

# ----- Modified Nash–Sutcliffe Efficiency (E₁, j=1) -----
print("\033[93mComputing Modified NSE (E₁, j=1)...\033[0m")

e1_value = nse_j(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'], j=1)
print(f"Modified NSE (E₁, j=1) = {e1_value:.4f}")

e1_monthly = monthly_nse_j(BACHLmonthly_L3, j=1)

for i, val in enumerate(e1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified NSE (E₁) = {val:.4f}")

print("-" * 45)

# ----- Modified Index of Agreement (d₁, j=1) -----
print("\033[93mComputing Modified Index of Agreement (d₁, j=1)...\033[0m")

d1_value = index_of_agreement_j(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'], j=1)
print(f"Modified Index of Agreement (d₁, j=1) = {d1_value:.4f}")

d1_monthly = monthly_index_of_agreement_j(BACHLmonthly_L3, j=1)

for i, val in enumerate(d1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified Index of Agreement (d₁) = {val:.4f}")

print("-" * 45)

# ----- Relative Nash–Sutcliffe Efficiency (E_rel) -----
print("\033[93mComputing Relative NSE (E_rel)...\033[0m")

mask = ~np.isnan(BACHL_L3['BACHLsat_L3']) & ~np.isnan(BACHL_L3['BACHLmod_L3']) & (BACHL_L3['BACHLsat_L3'] != 0)
e_rel_value = relative_nse(BACHL_L3['BACHLsat_L3'][mask], BACHL_L3['BACHLmod_L3'][mask])
print(f"Relative NSE (E_rel) = {e_rel_value:.4f}")

e_rel_monthly = monthly_relative_nse(BACHLmonthly_L3)

for i, val in enumerate(e_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative NSE = {val:.4f}")

print("-" * 45)

# ----- Relative Index of Agreement (d_rel) -----
print("\033[93mComputing Relative Index of Agreement (d_rel)...\033[0m")

mask = ~np.isnan(BACHL_L3['BACHLsat_L3']) & ~np.isnan(BACHL_L3['BACHLmod_L3']) & (BACHL_L3['BACHLsat_L3'] != 0)
d_rel_value = relative_index_of_agreement(BACHL_L3['BACHLsat_L3'][mask], BACHL_L3['BACHLmod_L3'][mask])
print(f"Relative Index of Agreement (d_rel) = {d_rel_value:.4f}")

d_rel_monthly = monthly_relative_index_of_agreement(BACHLmonthly_L3)

for i, val in enumerate(d_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative Index of Agreement = {val:.4f}")

print("-" * 45)

print("\033[92m✅ All of the metrcis have been computed! \033[0m")
print('*'*45)

###############################################################################
##                                                                           ##
##                           EFFICIENCY PLOTS                                ##
##                                                                           ##
###############################################################################

print("Plotting the efficiency metrcis results...")

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "CHL", "l3", timestamp)
os.makedirs(output_path, exist_ok=True)

# ----- Plot Coefficient of Determination (r²) ----- 
plot_metric('Coefficient of Determination (r²)', r2_value, r2_monthly, 'r² Value', output_path)
print("\033[92m✅ Coefficient of determination plotted! \033[0m")

# ----- Plot Weighted Coefficient of Determination (wr²) ----- 
plot_metric('Weighted Coefficient of Determination (wr²)', wr2_value, wr2_monthly, 'wr² Value', output_path)
print("\033[92m✅ Weighted coefficient of determination plotted! \033[0m")

# ----- Plot Nash-Sutcliffe Efficiency (NSE) ----- 
plot_metric('Nash-Sutcliffe Efficiency', nse_value, nse_monthly, 'NSE Value', output_path)
print("\033[92m✅ Nash-Sutcliffe index plotted! \033[0m")

# ----- Plot Index of Agreement (d) ----- 
plot_metric('Index of Agreement (d)', d_value, d_monthly, 'Index of Agreement (d)', output_path)
print("\033[92m✅ Index of Aggreement plotted! \033[0m")

# ----- Plot Logarithmic Nash–Sutcliffe Efficiency (ln NSE) ----- 
plot_metric('Nash-Sutcliffe Efficiency (Logarithmic)', ln_nse_value, ln_nse_monthly, 'ln NSE Value', output_path)
print("\033[92m✅ Logarithmic Nash-Sutcliffe index plotted! \033[0m")

# ----- Plot Modified Nash–Sutcliffe Efficiency (E₁, j=1) ----- 
plot_metric('Modified NSE (E₁, j=1)', e1_value, e1_monthly, 'E₁ Value', output_path)
print("\033[92m✅ Modified Nash-Sutcliffe index plitted! \033[0m")

# ----- Plot Modified Index of Agreement (d₁, j=1) ----- 
plot_metric('Modified Index of Agreement (d₁, j=1)', d1_value, d1_monthly, 'd₁ Value', output_path)
print("\033[92m✅ Modified index of aggreement plotted! \033[0m")

# ----- Plot Relative Nash–Sutcliffe Efficiency (E_rel) ----- 
plot_metric('Relative NSE ($E_{rel}$)', e_rel_value, e_rel_monthly, 'E_rel Value', output_path)
print("\033[92m✅ Relative Nash-Sutcliffe index plotted! \033[0m")

# ----- Plot Relative Index of Agreement (d_rel) ----- 
plot_metric('Relative Index of Agreement ($d_{rel}$)', d_rel_value, d_rel_monthly, 'd_rel Value', output_path)
print("\033[92m✅ Relative index of aggreement plotted! \033[0m")

print("\033[92m✅ The efficiency metrcis plots have been succesfully created! \033[0m", output_path)
print("*"*45)

###############################################################################
##                                                                           ##
##                                 LEVEL 4                                   ##
##                                                                           ##
###############################################################################

###############################################################################
##                                                                           ##
##                              DICTIONARIES                                 ##
##                                                                           ##
###############################################################################

print("Setting up the level 4 datasets for the analysis...")

BACHLmod_L4 = xr.open_dataset(Path(IDIR, 'BACHLmod_L4.nc'))
BACHLsat_L4 = xr.open_dataset(Path(IDIR, 'BACHLsat_L4.nc'))
BACHLmod_L4 = BACHLmod_L4['BACHLmod_L4'].values
BACHLsat_L4 = BACHLsat_L4['BACHLsat_L4'].values

BACHL_L4 = {
            'BACHLmod_L4' : BACHLmod_L4,
            'BACHLsat_L4' : BACHLsat_L4
            }
print("\033[92m✅ 3D dictionary created! \033[0m")
print("-"*45)

print("Generating the yearly and monthly datasets...")
# Generate datetime index
dates = pd.date_range(start='2000-01-01', end='2009-12-31', freq='D')
years = np.array([d.year for d in dates])
unique_years = np.unique(years)

# Split data into a list of arrays, one per year
mod_split = [BACHLmod_L4[years == y] for y in unique_years]
sat_split = [BACHLsat_L4[years == y] for y in unique_years]

# Find max days in any year (i.e., 366)
max_days = max(len(year_data) for year_data in mod_split)

# Pad shorter years with NaNs to make uniform arrays
BACHLmod_L4_yearly = np.array([np.pad(year_data, (0, max_days - len(year_data)), constant_values=np.nan)
                              for year_data in mod_split])
BACHLsat_L4_yearly = np.array([np.pad(year_data, (0, max_days - len(year_data)), constant_values=np.nan)
                              for year_data in sat_split])

BACHL_yearly_L4 = {
                'BAmod_year' : BACHLmod_L4_yearly,
                'BAsat_year' : BACHLsat_L4_yearly
                }
print("\033[92m✅ Yearly dictionary created! \033[0m")

# Conversion to a dictionary divided in months
BACHLmod_L4_monthly_dict = convert_to_monthly_data(BACHLmod_L4_yearly)
BACHLsat_L4_monthly_dict = convert_to_monthly_data(BACHLsat_L4_yearly)

BACHLmonthly_L4 = {
                'BACHLmod_L4_monthly' : BACHLmod_L4_monthly_dict,
                'BACHLsat_L4_monthly' : BACHLsat_L4_monthly_dict
                }
print("\033[92m✅ Monthly dictionary created! \033[0m")
print("*"*45)

###############################################################################
##                                                                           ##
##                                  PLOTS                                    ##
##                                                                           ##
###############################################################################

print("Beginning to plot...")
print("-"*45)

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "OTHER", "CHL", "l4", timestamp)
os.makedirs(output_path, exist_ok=True)

# 1. BIAS, TIMESERIES AND SCATTERPLOTS
print("Computing the BIAS...")
BIAS_Bavg = BACHLsat_L4 - BACHLmod_L4
print("\033[92m✅ BIAS computed! \033[0m")
print("-"*45)

print("Plotting the timeseries...")
plot_daily_means(output_path, BACHL_L4, 'CHL', BIAS_Bavg, BA=True)
print("\033[92m✅ Time series plotted! \033[0m")

print("Plotting the scatter plot...")
scatter_plot(output_path, BACHL_L4, 'CHL', BA=False)
print("\033[92m✅ Scatter plot succesfully plotted! \033[0m")
print("-"*45)

###############################################################################
##                                                                           ##
##                             TAYLOR DIAGRAMS                               ##
##                                                                           ##
###############################################################################

print("Plotting the Taylor diagrams...")
std_ref = np.nanstd(BACHLsat_L4)

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "TAYLOR", "CHL", "l4", timestamp)
os.makedirs(output_path, exist_ok=True)

# Plotting the Taylor Diagram
comprehensive_taylor_diagram(BACHL_yearly_L4, taylor_options, std_ref, output_path)
print("\033[92m✅ Yearly Taylor diagram plotted! \033[0m")

for i in range(12):
    monthly_taylor_diagram(BACHLmonthly_L4, i, taylor_options_monthly, output_path)
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
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "TARGET", "CHL", "l4", timestamp)
os.makedirs(output_path, exist_ok=True)

comprehensive_target_diagram(BACHL_yearly_L4, output_path)
print("\033[92m✅ Yearly target plot has been made! \033[0m")

for i in range(12):
    target_diagram_by_month(BACHLmonthly_L4, i, output_path)
print("\033[92m✅ All of the monthly target plots have been made! \033[0m")
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

r2_value = r_squared(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'])
print(f"r² (Coefficient of Determination) = {r2_value:.4f}")

r2_monthly = monthly_r_squared(BACHLmonthly_L4)

for i, val in enumerate(r2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: r² = {val:.4f}")

print("-" * 45)

# ----- Weighted Coeddificient of Determination -----
print("\033[93mComputing Weighted Coefficient of Determination (wr²)...\033[0m")

wr2_value = weighted_r_squared(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'])
print(f"wr² (Weighted Coefficient of Determination) = {wr2_value:.4f}")

wr2_monthly = monthly_weighted_r_squared(BACHLmonthly_L4)

for i, val in enumerate(wr2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: wr² = {val:.4f}")

print("-" * 45)

# ----- Nash-Sutcliffe -----
print("\033[93mComputing Nash–Sutcliffe Efficiency (NSE)...\033[0m")

nse_value = nse(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'])
print(f"NSE (Nash–Sutcliffe Efficiency) = {nse_value:.4f}")

nse_monthly = monthly_nse(BACHLmonthly_L4)

for i, val in enumerate(nse_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: NSE = {val:.4f}")

print("-" * 45)

# ----- Index of Agreement -----
print("\033[93mComputing Index of Agreement (d)...\033[0m")

d_value = index_of_agreement(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'])
print(f"Index of Agreement (d) = {d_value:.4f}")

d_monthly = monthly_index_of_agreement(BACHLmonthly_L4)

for i, val in enumerate(d_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: Index of Agreement (d) = {val:.4f}")

print("-" * 45)

# ----- Logarithmic Nash–Sutcliffe Efficiency (ln NSE) -----
print("\033[93mComputing NSE with Logarithmic Values (ln NSE)...\033[0m")

# Remove NaNs and ensure values are positive before computing
mask = ~np.isnan(BACHL_L4['BACHLsat_L4']) & ~np.isnan(BACHL_L4['BACHLmod_L4']) & \
       (BACHL_L4['BACHLsat_L4'] > 0) & (BACHL_L4['BACHLmod_L4'] > 0)

ln_nse_value = ln_nse(BACHL_L4['BACHLsat_L4'][mask], BACHL_L4['BACHLmod_L4'][mask])
print(f"ln NSE = {ln_nse_value:.4f}")

ln_nse_monthly = monthly_ln_nse(BACHLmonthly_L4)

for i, val in enumerate(ln_nse_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: ln NSE = {val:.4f}")

print("-" * 45)

# ----- Modified Nash–Sutcliffe Efficiency (E₁, j=1) -----
print("\033[93mComputing Modified NSE (E₁, j=1)...\033[0m")

e1_value = nse_j(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'], j=1)
print(f"Modified NSE (E₁, j=1) = {e1_value:.4f}")

e1_monthly = monthly_nse_j(BACHLmonthly_L4, j=1)

for i, val in enumerate(e1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified NSE (E₁) = {val:.4f}")

print("-" * 45)

# ----- Modified Index of Agreement (d₁, j=1) -----
print("\033[93mComputing Modified Index of Agreement (d₁, j=1)...\033[0m")

d1_value = index_of_agreement_j(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'], j=1)
print(f"Modified Index of Agreement (d₁, j=1) = {d1_value:.4f}")

d1_monthly = monthly_index_of_agreement_j(BACHLmonthly_L4, j=1)

for i, val in enumerate(d1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified Index of Agreement (d₁) = {val:.4f}")

print("-" * 45)

# ----- Relative Nash–Sutcliffe Efficiency (E_rel) -----
print("\033[93mComputing Relative NSE (E_rel)...\033[0m")

mask = ~np.isnan(BACHL_L4['BACHLsat_L4']) & ~np.isnan(BACHL_L4['BACHLmod_L4']) & (BACHL_L4['BACHLsat_L4'] != 0)
e_rel_value = relative_nse(BACHL_L4['BACHLsat_L4'][mask], BACHL_L4['BACHLmod_L4'][mask])
print(f"Relative NSE (E_rel) = {e_rel_value:.4f}")

e_rel_monthly = monthly_relative_nse(BACHLmonthly_L4)

for i, val in enumerate(e_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative NSE = {val:.4f}")

print("-" * 45)

# ----- Relative Index of Agreement (d_rel) -----
print("\033[93mComputing Relative Index of Agreement (d_rel)...\033[0m")

mask = ~np.isnan(BACHL_L4['BACHLsat_L4']) & ~np.isnan(BACHL_L4['BACHLmod_L4']) & (BACHL_L4['BACHLsat_L4'] != 0)
d_rel_value = relative_index_of_agreement(BACHL_L4['BACHLsat_L4'][mask], BACHL_L4['BACHLmod_L4'][mask])
print(f"Relative Index of Agreement (d_rel) = {d_rel_value:.4f}")

d_rel_monthly = monthly_relative_index_of_agreement(BACHLmonthly_L4)

for i, val in enumerate(d_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative Index of Agreement = {val:.4f}")

print("-" * 45)

print("\033[92m✅ All of the metrcis have been computed! \033[0m")
print('*'*45)

###############################################################################
##                                                                           ##
##                           EFFICIENCY PLOTS                                ##
##                                                                           ##
###############################################################################

print("Plotting the efficiency metrcis results...")

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "CHL", "l4", timestamp)
os.makedirs(output_path, exist_ok=True)

# ----- Plot Coefficient of Determination (r²) ----- 
plot_metric('Coefficient of Determination (r²)', r2_value, r2_monthly, 'r² Value', output_path)
print("\033[92m✅ Coefficient of determination plotted! \033[0m")

# ----- Plot Weighted Coefficient of Determination (wr²) ----- 
plot_metric('Weighted Coefficient of Determination (wr²)', wr2_value, wr2_monthly, 'wr² Value', output_path)
print("\033[92m✅ Weighted coefficient of determination plotted! \033[0m")

# ----- Plot Nash-Sutcliffe Efficiency (NSE) ----- 
plot_metric('Nash-Sutcliffe Efficiency', nse_value, nse_monthly, 'NSE Value', output_path)
print("\033[92m✅ Nash-Sutcliffe index plotted! \033[0m")

# ----- Plot Index of Agreement (d) ----- 
plot_metric('Index of Agreement (d)', d_value, d_monthly, 'Index of Agreement (d)', output_path)
print("\033[92m✅ Index of Aggreement plotted! \033[0m")

# ----- Plot Logarithmic Nash–Sutcliffe Efficiency (ln NSE) ----- 
plot_metric('Nash-Sutcliffe Efficiency (Logarithmic)', ln_nse_value, ln_nse_monthly, 'ln NSE Value', output_path)
print("\033[92m✅ Logarithmic Nash-Sutcliffe index plotted! \033[0m")

# ----- Plot Modified Nash–Sutcliffe Efficiency (E₁, j=1) ----- 
plot_metric('Modified NSE (E₁, j=1)', e1_value, e1_monthly, 'E₁ Value', output_path)
print("\033[92m✅ Modified Nash-Sutcliffe index plitted! \033[0m")

# ----- Plot Modified Index of Agreement (d₁, j=1) ----- 
plot_metric('Modified Index of Agreement (d₁, j=1)', d1_value, d1_monthly, 'd₁ Value', output_path)
print("\033[92m✅ Modified index of aggreement plotted! \033[0m")

# ----- Plot Relative Nash–Sutcliffe Efficiency (E_rel) ----- 
plot_metric('Relative NSE ($E_{rel}$)', e_rel_value, e_rel_monthly, 'E_rel Value', output_path)
print("\033[92m✅ Relative Nash-Sutcliffe index plotted! \033[0m")

# ----- Plot Relative Index of Agreement (d_rel) ----- 
plot_metric('Relative Index of Agreement ($d_{rel}$)', d_rel_value, d_rel_monthly, 'd_rel Value', output_path)
print("\033[92m✅ Relative index of aggreement plotted! \033[0m")

print("\033[92m✅ The efficiency metrcis plots have been succesfully created! \033[0m")
print("*"*45)