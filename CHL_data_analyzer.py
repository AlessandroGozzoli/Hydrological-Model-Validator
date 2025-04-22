import os
from pathlib import Path
import numpy as np
import calendar
import pandas as pd
import xarray as xr

############### MODULES LOADING ###############

import sys

print("Loading the necessary modules...")
CodingDIR = "C:/Tesi Magistrale/Codici/Python/"
os.chdir(CodingDIR)  # Set the working directory

print("Loading the Pre-Processing modules and constants...")
ProcessingDIR = Path(CodingDIR, "Processing/")
sys.path.append(str(ProcessingDIR))  # Add the folder to the system path

from Leap_year import convert_to_monthly_data
from Mask_reader import mask_reader

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

# Call the function and extract values
Mmask, Mfsm, Mfsm_3d, Mlat, Mlon = mask_reader()
print("Mask succesfully imported!")
print('*'*45)

print("Importing the datasets...")
Mchll3 = xr.open_dataset(Path(IDIR, 'Mchl_interp_l3.nc'))
Mchll4 = xr.open_dataset(Path(IDIR, 'Mchl_interp_l4.nc'))
Schll3 = xr.open_dataset(Path(IDIR, 'Schl_interp_l3.nc'))
Schll4 = xr.open_dataset(Path(IDIR, 'Schl_interp_l4.nc'))
print("Datasets imported!")
print("*"*45)

################################
# ----- LEVEL 3 ANALYSIS ----- #
################################

print("Setting up the level 3 datasets for the analysis...")

BACHLmod_L3 = xr.open_dataset(Path(IDIR, 'BACHLmod_L3.nc'))
BACHLsat_L3 = xr.open_dataset(Path(IDIR, 'BACHLsat_L3.nc'))
BACHLmod_L3 = BACHLmod_L3['BACHLmod_L3'].values
BACHLsat_L3 = BACHLsat_L3['BACHLsat_L3'].values

BACHL_L3 = {
            'BACHLmod_L3' : BACHLmod_L3,
            'BACHLsat_L3' : BACHLsat_L3
            }
print("3D dictionary created!")
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
print("Yearly dictionary created!")

# Conversion to a dictionary divided in months
BACHLmod_L3_monthly_dict = convert_to_monthly_data(BACHLmod_L3_yearly)
BACHLsat_L3_monthly_dict = convert_to_monthly_data(BACHLsat_L3_yearly)

BACHLmonthly_L3 = {
                'BACHLmod_L3_monthly' : BACHLmod_L3_monthly_dict,
                'BACHLsat_L3_monthly' : BACHLsat_L3_monthly_dict
                }
print("Monthly dictionary created!")
print("*"*45)

#############################
# ----- LEVEL 3 PLOTS ----- #
#############################

print("Beginning to plot...")
print("-"*45)

# 1. BIAS, TIMESERIES AND SCATTERPLOTS
print("Computing the BIAS...")
BIAS_Bavg = BACHLsat_L3 - BACHLmod_L3
print("BIAS computed!")
print("-"*45)

print("Plotting the timeseries...")
plot_daily_means(BACHL_L3, 'CHL', BIAS_Bavg, BA=True)
print("Time series plotted!")
print("Plotting the scatter plot...")
scatter_plot_BASST(BACHL_L3, 'CHL', BA=False)
print("Scatter plot succesfully plotted!")
print("-"*45)

# 2. TAYLOR DIAGRAMS
print("Plotting the Taylor diagrams...")
std_ref = np.nanstd(BACHLsat_L3)

# Plotting the Taylor Diagram
comprehensive_taylor_diagram(BACHL_yearly_L3, taylor_options, std_ref)
print("Yearly Taylor diagram plotted!")

for i in range(12):
    monthly_taylor_diagram(BACHLmonthly_L3, i, taylor_options_monthly)
print("All of the monthly Taylor diagrams have been plotted!")
print("-"*45)

# 3. TARGET PLOTS
print("Beginning to create the Target plots...")

comprehensive_target_diagram(BACHL_yearly_L3)
print("Yearly target plot has been made!")

for i in range(12):
    target_diagram_by_month(BACHLmonthly_L3, i)
print("All of the monthly target plots have been made!")
print("*"*45)

########################################################
# ----- COMPUTING THE LEVEL 3 EFFICIENCY METRCIS ----- #
########################################################
print("Computing the Efficiency Metrics...")
print("They are based onto Krause et al. 2005")
print("-"*45)

# ----- Coefficient of determination -----
print("Computing Coefficient of Determination (r²)...")

r2_value = r_squared(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'])
print(f"r² (Coefficient of Determination) = {r2_value:.4f}")

r2_monthly = monthly_r_squared(BACHLmonthly_L3)

for i, val in enumerate(r2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: r² = {val:.4f}")

print("-" * 45)

# ----- Weighted Coeddificient of Determination -----
print("Computing Weighted Coefficient of Determination (wr²)...")

wr2_value = weighted_r_squared(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'])
print(f"wr² (Weighted Coefficient of Determination) = {wr2_value:.4f}")

wr2_monthly = monthly_weighted_r_squared(BACHLmonthly_L3)

for i, val in enumerate(wr2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: wr² = {val:.4f}")

print("-" * 45)

# ----- Nash-Sutcliffe -----
print("Computing Nash–Sutcliffe Efficiency (NSE)...")

nse_value = nse(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'])
print(f"NSE (Nash–Sutcliffe Efficiency) = {nse_value:.4f}")

nse_monthly = monthly_nse(BACHLmonthly_L3)

for i, val in enumerate(nse_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: NSE = {val:.4f}")

print("-" * 45)

# ----- Index of Agreement -----
print("Computing Index of Agreement (d)...")

d_value = index_of_agreement(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'])
print(f"Index of Agreement (d) = {d_value:.4f}")

d_monthly = monthly_index_of_agreement(BACHLmonthly_L3)

for i, val in enumerate(d_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: Index of Agreement (d) = {val:.4f}")

print("-" * 45)

# ----- Logarithmic Nash–Sutcliffe Efficiency (ln NSE) -----
print("Computing NSE with Logarithmic Values (ln NSE)...")

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
print("Computing Modified NSE (E₁, j=1)...")

e1_value = nse_j(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'], j=1)
print(f"Modified NSE (E₁, j=1) = {e1_value:.4f}")

e1_monthly = monthly_nse_j(BACHLmonthly_L3, j=1)

for i, val in enumerate(e1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified NSE (E₁) = {val:.4f}")

print("-" * 45)

# ----- Modified Index of Agreement (d₁, j=1) -----
print("Computing Modified Index of Agreement (d₁, j=1)...")

d1_value = index_of_agreement_j(BACHL_L3['BACHLsat_L3'], BACHL_L3['BACHLmod_L3'], j=1)
print(f"Modified Index of Agreement (d₁, j=1) = {d1_value:.4f}")

d1_monthly = monthly_index_of_agreement_j(BACHLmonthly_L3, j=1)

for i, val in enumerate(d1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified Index of Agreement (d₁) = {val:.4f}")

print("-" * 45)

# ----- Relative Nash–Sutcliffe Efficiency (E_rel) -----
print("Computing Relative NSE (E_rel)...")

mask = ~np.isnan(BACHL_L3['BACHLsat_L3']) & ~np.isnan(BACHL_L3['BACHLmod_L3']) & (BACHL_L3['BACHLsat_L3'] != 0)
e_rel_value = relative_nse(BACHL_L3['BACHLsat_L3'][mask], BACHL_L3['BACHLmod_L3'][mask])
print(f"Relative NSE (E_rel) = {e_rel_value:.4f}")

e_rel_monthly = monthly_relative_nse(BACHLmonthly_L3)

for i, val in enumerate(e_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative NSE = {val:.4f}")

print("-" * 45)

# ----- Relative Index of Agreement (d_rel) -----
print("Computing Relative Index of Agreement (d_rel)...")

mask = ~np.isnan(BACHL_L3['BACHLsat_L3']) & ~np.isnan(BACHL_L3['BACHLmod_L3']) & (BACHL_L3['BACHLsat_L3'] != 0)
d_rel_value = relative_index_of_agreement(BACHL_L3['BACHLsat_L3'][mask], BACHL_L3['BACHLmod_L3'][mask])
print(f"Relative Index of Agreement (d_rel) = {d_rel_value:.4f}")

d_rel_monthly = monthly_relative_index_of_agreement(BACHLmonthly_L3)

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

################################
# ----- LEVEL 4 ANALYSIS ----- #
################################

print("Setting up the level 4 datasets for the analysis...")

BACHLmod_L4 = xr.open_dataset(Path(IDIR, 'BACHLmod_L4.nc'))
BACHLsat_L4 = xr.open_dataset(Path(IDIR, 'BACHLsat_L4.nc'))
BACHLmod_L4 = BACHLmod_L4['BACHLmod_L4'].values
BACHLsat_L4 = BACHLsat_L4['BACHLsat_L4'].values

BACHL_L4 = {
            'BACHLmod_L4' : BACHLmod_L4,
            'BACHLsat_L4' : BACHLsat_L4
            }
print("3D dictionary created!")
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
print("Yearly dictionary created!")

# Conversion to a dictionary divided in months
BACHLmod_L4_monthly_dict = convert_to_monthly_data(BACHLmod_L4_yearly)
BACHLsat_L4_monthly_dict = convert_to_monthly_data(BACHLsat_L4_yearly)

BACHLmonthly_L4 = {
                'BACHLmod_L4_monthly' : BACHLmod_L4_monthly_dict,
                'BACHLsat_L4_monthly' : BACHLsat_L4_monthly_dict
                }
print("Monthly dictionary created!")
print("*"*45)

#############################
# ----- LEVEL 4 PLOTS ----- #
#############################

print("Beginning to plot...")
print("-"*45)

# 1. BIAS, TIMESERIES AND SCATTERPLOTS
print("Computing the BIAS...")
BIAS_Bavg = BACHLsat_L4 - BACHLmod_L4
print("BIAS computed!")
print("-"*45)

print("Plotting the timeseries...")
plot_daily_means(BACHL_L4, 'CHL', BIAS_Bavg, BA=True)
print("Time series plotted!")
print("Plotting the scatter plot...")
scatter_plot_BASST(BACHL_L4, 'CHL', BA=False)
print("Scatter plot succesfully plotted!")
print("-"*45)

# 2. TAYLOR DIAGRAMS
print("Plotting the Taylor diagrams...")
std_ref = np.nanstd(BACHLsat_L4)

# Plotting the Taylor Diagram
comprehensive_taylor_diagram(BACHL_yearly_L4, taylor_options, std_ref)
print("Yearly Taylor diagram plotted!")

for i in range(12):
    monthly_taylor_diagram(BACHLmonthly_L4, i, taylor_options_monthly)
print("All of the monthly Taylor diagrams have been plotted!")
print("-"*45)

# 3. TARGET PLOTS
print("Beginning to create the Target plots...")

comprehensive_target_diagram(BACHL_yearly_L4)
print("Yearly target plot has been made!")

for i in range(12):
    target_diagram_by_month(BACHLmonthly_L4, i)
print("All of the monthly target plots have been made!")
print("*"*45)

########################################################
# ----- COMPUTING THE LEVEL 4 EFFICIENCY METRCIS ----- #
########################################################
print("Computing the Efficiency Metrics...")
print("They are based onto Krause et al. 2005")
print("-"*45)

# ----- Coefficient of determination -----
print("Computing Coefficient of Determination (r²)...")

r2_value = r_squared(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'])
print(f"r² (Coefficient of Determination) = {r2_value:.4f}")

r2_monthly = monthly_r_squared(BACHLmonthly_L4)

for i, val in enumerate(r2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: r² = {val:.4f}")

print("-" * 45)

# ----- Weighted Coeddificient of Determination -----
print("Computing Weighted Coefficient of Determination (wr²)...")

wr2_value = weighted_r_squared(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'])
print(f"wr² (Weighted Coefficient of Determination) = {wr2_value:.4f}")

wr2_monthly = monthly_weighted_r_squared(BACHLmonthly_L4)

for i, val in enumerate(wr2_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: wr² = {val:.4f}")

print("-" * 45)

# ----- Nash-Sutcliffe -----
print("Computing Nash–Sutcliffe Efficiency (NSE)...")

nse_value = nse(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'])
print(f"NSE (Nash–Sutcliffe Efficiency) = {nse_value:.4f}")

nse_monthly = monthly_nse(BACHLmonthly_L4)

for i, val in enumerate(nse_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: NSE = {val:.4f}")

print("-" * 45)

# ----- Index of Agreement -----
print("Computing Index of Agreement (d)...")

d_value = index_of_agreement(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'])
print(f"Index of Agreement (d) = {d_value:.4f}")

d_monthly = monthly_index_of_agreement(BACHLmonthly_L4)

for i, val in enumerate(d_monthly):
    month_name = calendar.month_name[i + 1]  # month_name[1] = 'January'
    print(f"{month_name}: Index of Agreement (d) = {val:.4f}")

print("-" * 45)

# ----- Logarithmic Nash–Sutcliffe Efficiency (ln NSE) -----
print("Computing NSE with Logarithmic Values (ln NSE)...")

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
print("Computing Modified NSE (E₁, j=1)...")

e1_value = nse_j(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'], j=1)
print(f"Modified NSE (E₁, j=1) = {e1_value:.4f}")

e1_monthly = monthly_nse_j(BACHLmonthly_L4, j=1)

for i, val in enumerate(e1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified NSE (E₁) = {val:.4f}")

print("-" * 45)

# ----- Modified Index of Agreement (d₁, j=1) -----
print("Computing Modified Index of Agreement (d₁, j=1)...")

d1_value = index_of_agreement_j(BACHL_L4['BACHLsat_L4'], BACHL_L4['BACHLmod_L4'], j=1)
print(f"Modified Index of Agreement (d₁, j=1) = {d1_value:.4f}")

d1_monthly = monthly_index_of_agreement_j(BACHLmonthly_L4, j=1)

for i, val in enumerate(d1_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Modified Index of Agreement (d₁) = {val:.4f}")

print("-" * 45)

# ----- Relative Nash–Sutcliffe Efficiency (E_rel) -----
print("Computing Relative NSE (E_rel)...")

mask = ~np.isnan(BACHL_L4['BACHLsat_L4']) & ~np.isnan(BACHL_L4['BACHLmod_L4']) & (BACHL_L4['BACHLsat_L4'] != 0)
e_rel_value = relative_nse(BACHL_L4['BACHLsat_L4'][mask], BACHL_L4['BACHLmod_L4'][mask])
print(f"Relative NSE (E_rel) = {e_rel_value:.4f}")

e_rel_monthly = monthly_relative_nse(BACHLmonthly_L4)

for i, val in enumerate(e_rel_monthly):
    month_name = calendar.month_name[i + 1]
    print(f"{month_name}: Relative NSE = {val:.4f}")

print("-" * 45)

# ----- Relative Index of Agreement (d_rel) -----
print("Computing Relative Index of Agreement (d_rel)...")

mask = ~np.isnan(BACHL_L4['BACHLsat_L4']) & ~np.isnan(BACHL_L4['BACHLmod_L4']) & (BACHL_L4['BACHLsat_L4'] != 0)
d_rel_value = relative_index_of_agreement(BACHL_L4['BACHLsat_L4'][mask], BACHL_L4['BACHLmod_L4'][mask])
print(f"Relative Index of Agreement (d_rel) = {d_rel_value:.4f}")

d_rel_monthly = monthly_relative_index_of_agreement(BACHLmonthly_L4)

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