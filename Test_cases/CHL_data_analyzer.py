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
import subprocess

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
from Hydrological_model_validator.Processing.time_utils import (split_to_monthly, 
                                                                split_to_yearly,
                                                                resample_and_compute)
from Hydrological_model_validator.Plotting.formatting import compute_geolocalized_coords
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
                                                           efficiency_plot,
                                                           plot_spatial_efficiency)
from Hydrological_model_validator.Plotting.Taylor_diagrams import (comprehensive_taylor_diagram,
                                                                   monthly_taylor_diagram)
from Hydrological_model_validator.Plotting.Target_plots import (comprehensive_target_diagram,
                                                                target_diagram_by_month)
from Hydrological_model_validator.Processing.stats_math_utils import (detrend_dim)
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
                                                                        monthly_relative_index_of_agreement,
                                                                        compute_spatial_efficiency)
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
Mfsm = mask_reader(BaseDIR)
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
cloud_cover = xr.open_dataset(idir_path / "cloud_cover_chl.nc")['cloud_cover_chl']
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
    'BACHLsat': BACHLsat_series,
    'BACHLmod': BACHLmod_series
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
BIAS_Bavg = BACHLmod - BACHLsat
BIAS = pd.Series(BIAS_Bavg, index=dates)
# Also adding the dates to the cloud cover %
cloud_coverage = pd.Series(cloud_cover, index=dates)
print("\033[92m✅ BIAS computed! \033[0m")
print("-"*45)

print("Plotting the timeseries...")
timeseries(BACHL,
           BIAS, 
           cloud_coverage=cloud_coverage,
           output_path=output_path,
           variable_name='CHL_L3',
           BA=True,
           smooth7=True,   
           smooth30=True  
           )
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

###############################################################################
##                                                                           ##
##                           SPATIAL PERFORMANCE                             ##
##                                                                           ##
###############################################################################

# ----- RETRIEVING THE MASK -----
print("Retrieving the mask...")
Mfsm = mask_reader(BaseDIR)
ocean_mask = Mfsm  # This returns a NumPy array
print("\033[92m✅ Mask succesfully imported! \033[0m")
print('*'*45)

# ----- GEOLOCALIZE THE DATASET -----
print("Computing the geolocalization...")
# Known values from the dataset, need to be changed if the area of analysis is changed
grid_shape = (278, 315)
epsilon = 0.06  # Correction factor linked to the resolution of the dataset
x_start = 12.1986847
x_step = 0.00948
y_start = 43.5037956
y_step = 0.00897

geo_coords = compute_geolocalized_coords(grid_shape, epsilon, x_start, x_step, y_start, y_step)
print("\033[92m✅ Geolocalization complete! \033[0m")
print('*'*45)

# ----- LOADING THE DAILY DATASETS, OUTPUT OF THE INTERPOLATOR.M -----
print("Loading the datasets...")
ds_model = xr.open_dataset(IDIR / "ModData_chl_interp_l3.nc")
ds_sat = xr.open_dataset(IDIR / "SatData_chl_interp_l3.nc")
print("The datasets have been loaded!")
print('-'*45)

# ----- TRANSPOSING -----
print("Due to the necessity to resample the data the datasets need to be")
print("Transposed so that the 1st dimension is the time")
print("Transposing the datasets...")
model_chl = ds_model['ModData_interp'].transpose('time', 'lat', 'lon')
sat_chl = ds_sat['SatData_complete'].transpose('time', 'lat', 'lon')
print("The datasets have been transposed!")
print('-'*45)

# ----- ADDIING CORRECT DATETIME -----
print("Adding a datetime to aid with the resampling...")
time_origin = pd.Timestamp("2000-01-01")
model_chl['time'] = time_origin + pd.to_timedelta(model_chl.time.values, unit="D")
sat_chl['time'] = time_origin + pd.to_timedelta(sat_chl.time.values, unit="D")
print("Daily datetime index added!")
print('-'*45)

# ----- WARNING AND RESAMPLING -----
print("\n--- Data Resampling Notice ---")
print("This dataset must be resampled to obtain *monthly averages*, which are")
print("required for the subsequent analyses (tailored to monthly/yearly data).")
print("⚠️  Resampling is computationally intensive, especially on the full dataset.")
print("\nTwo options are available:")
print("1. Perform in-code resampling using parallel Dask processes.")
print("2. Save the dataset and use an external tool like CDO for resampling.\n")
print("3. Save the dataset to resample using another program.")
print("The resampling using a CDO is the current fastest method")
print("But in this test case the already resampled file is already provided!")

# ----- GET RESPONSES FROM USER ------
resample = input("→ Proceed with in-code resampling? (yes/no): ").strip().lower()
do_resample = resample in ["yes", "y"]

cdo = input("→ Run the CDO directly? (yes/no): ").strip().lower()
do_cdo = cdo in ["yes", "y"]

save = input("→ Save the data to resample in another way? (yes/no): ").strip().lower()
do_save = save in ["yes", "y"]

# ----- EXTREME SCENARIO, EITHER EXAMPLE DATA OR KILL -----
if not do_resample and not do_cdo:
    check = input("Are you using the monthly resampled datasets provided in the /Data folder? (Yes/No): ")
    if check in ["yes", "y"]:
        print("Good! We can move on then!")
    else:
        print("\n❌ No action selected. The following analysis cannot progress without resampling.")
        exit()

# ----- 1ST CHECK -----
if do_resample:
    print("\n✅ Starting parallel resampling using Dask...")
    
    # ----- CUTTING DATA IN CHUNKS TO SPEED UP -----
    model_chl_chunked = model_chl.chunk({'time': 100})
    sat_chl_chunked = sat_chl.chunk({'time': 100})

    # ----- PERFORM THE RESAMPLING, TAKES A WHILE -----
    model_monthly, sat_monthly = resample_and_compute(model_chl_chunked, sat_chl_chunked)

    print("✅ Resampling completed.\n")
    
# ----- 2ND CHECK -----
if do_save:
    
    # ----- ONLY SAVE -----
    print("Saving daily SST data for external resampling via CDO...")
    model_chl.to_netcdf(Path(IDIR, "model_chl_daily.nc"))
    sat_chl.to_netcdf(Path(IDIR, "sat_chl_daily.nc"))
    print("✅ Files saved to:", IDIR, "\n")

# ----- 3RD CHECK -----
if do_cdo:
    # About the CDO usage (this is also in the test case README)
    # The CDO is a native linux program and as such it needs to be installed
    # in either a linux subsistem or a WSL in Windows.
    # This code will not run without it!
    # About the paths: since the IDIR path to access the PROCESSING_INPUT
    # folder is build dynamically starting from the cwd (in the future this
    # will be changed to use the __file__ location as basis for the path
    # construction) the "c" path will be built according to the system 
    # architecture. Meaning it will use C: for windows systems and C for 
    # Linux. This means that even though the CDO may be installed this section
    # of the code does not currently work when run from Windows because the 
    # path will be wrong!
    # Future updates will attempt to change this.
    print("Saving daily SST data for external resampling via CDO...")
    model_chl.to_netcdf(Path(IDIR, "model_chl_daily.nc"))
    sat_chl.to_netcdf(Path(IDIR, "sat_chl_daily.nc"))
    print("✅ Files saved to:", IDIR, "\n")

    print("Running the CDO...")
    print("Firstly the model data...")
    
    # ----- BUILD THE PATHS AS LINUX -----
    input_file = Path("/mnt") / IDIR / "model_chl_daily.nc"
    output_file = Path('/mnt/', IDIR, "model_chl_monthly.nc")

    # ----- RUN THE CDO -----
    subprocess.run(["/usr/bin/cdo", "-v", "monmean", input_file, output_file], check=True)
    print("The model data has been resampled!")
    
    # ----- REDO THE SAME FOR THE SATELLITE
    print("Onto the satellite data...")
    input_file_sat = os.path.join(IDIR, "sat_chl_daily.nc")
    output_file_sat = os.path.join(IDIR, "sat_chl_monthly.nc")

    subprocess.run(["cdo", "-v", "monmean", input_file_sat, output_file_sat], check=True)
    print("The satellite data has been resampled!")
    
# ----- GET BACK TO THE ANALYSIS -----
# ----- LOAD NEW DATASETS -----    
print("Loading the monthly datasets...")
model_chl_monthly = xr.open_dataset(IDIR / "model_chl_monthly.nc")
sat_chl_monthly = xr.open_dataset(IDIR / "sat_chl_monthly.nc")
print("The datasets have been loaded!")
print('-'*45)

print("Fetching the data...")
model_chl_data = model_chl_monthly["ModData_interp"]
sat_chl_data = sat_chl_monthly["SatData_complete"]

# ----- APPLY THE MASK -----
# ----- REUSES THE SAME FROM THE DATA SETUPPING -----
print("Masking...")
mask_da = xr.DataArray(ocean_mask)
mask_expanded = mask_da.expand_dims(time=model_chl_data.time)
model_chl_masked = model_chl_data.where(mask_expanded)
sat_chl_masked = sat_chl_data.where(mask_expanded)

# ----- DROP THE BOUNDS DIMENSION IF IT'S THERE -----
print("Dropping the time bounds dimension...")
print("(This is added by the cdo)")
model_chl_data = model_chl_data.drop_vars("time_bnds", errors="ignore")
sat_chl_data = sat_chl_data.drop_vars("time_bnds", errors="ignore")

# ----- ALIGN AS A FAILSAFE -----
print("Aligning the data...")
model_chl_data, sat_chl_data = xr.align(model_chl_data, sat_chl_data, join="inner")
print("The monthly data has been prepared!")
print('-'*45)

# ----- 1ST METRICS, MONTHLY AVG, RAW DATA -----
print("Computing the metrics from the raw data...")
mb_raw, sde_raw, cc_raw, rm_raw, ro_raw, urmse_raw = compute_spatial_efficiency(model_chl_masked, sat_chl_masked)

# Create a dict with metrics as keys, and each value is a list of 12 2D arrays (one per month)
metrics_dict_raw = {
    'MB_raw': [mb_raw.isel(month=month) for month in range(12)],
    'SDE_raw': [sde_raw.isel(month=month) for month in range(12)],
    'CC_raw': [cc_raw.isel(month=month) for month in range(12)],
    'RM_raw': [rm_raw.isel(month=month) for month in range(12)],
    'RO_raw': [ro_raw.isel(month=month) for month in range(12)],
    'URMSE_raw': [urmse_raw.isel(month=month) for month in range(12)],
}

# Convert to DataFrame: rows=months, columns=metrics, cells=2D DataArrays
metrics_df_raw = pd.DataFrame(metrics_dict_raw)

# Set index to months
metrics_df_raw.index = [f'Month_{i+1}' for i in range(12)]
print("Metrics computed!")
print('-'*45)

# ----- DETREND -----
print("Detrending the data...")
model_detrended = detrend_dim(model_chl_data, dim='time', mask=mask_expanded)
sat_detrended = detrend_dim(sat_chl_data, dim='time', mask=mask_expanded)
print("Data detrended!")

# ----- 2ND COMPUTATION, ALSO MONTHLY AVG, DETRENDED DATA -----
print("Computing the metrics from the detrended data...")
mb_detr, sde_detr, cc_detr, rm_detr, ro_detr, urmse_detr = compute_spatial_efficiency(model_detrended, sat_detrended)

# Create a dict with metrics as keys, and each value is a list of 12 2D arrays (one per month)
metrics_dict_detr = {
    'MB_detr': [mb_detr.isel(month=month) for month in range(12)],
    'SDE_detr': [sde_detr.isel(month=month) for month in range(12)],
    'CC_detr': [cc_detr.isel(month=month) for month in range(12)],
    'RM_detr': [rm_detr.isel(month=month) for month in range(12)],
    'RO_detr': [ro_detr.isel(month=month) for month in range(12)],
    'URMSE_detr': [urmse_detr.isel(month=month) for month in range(12)],
}

# Convert to DataFrame: rows=months, columns=metrics, cells=2D DataArrays
metrics_df_detr = pd.DataFrame(metrics_dict_detr)

# Set index to months
metrics_df_detr.index = [f'Month_{i+1}' for i in range(12)]
print("Detrended data metrcis computed!")
print('-'*45)

# ----- SET UP THE SAVE FOLDER -----
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BaseDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "CHL", "l3", timestamp)
os.makedirs(output_path, exist_ok=True)

# ----- BEGIN PLOTTING AND SAVING -----
print("Plotting the results...")

print("Plotting the mean bias...")
plot_spatial_efficiency(mb_raw,
                       geo_coords,
                       output_path,
                       "Mean Bias",
                       unit="mg/m3",
                       cmap="RdBu_r",
                       vmin=-2,
                       vmax=2)
plot_spatial_efficiency(mb_detr,
                       geo_coords,
                       output_path,
                       "Mean Bias",
                       unit="mg/m3",
                       cmap="RdBu_r",
                       vmin=-2,
                       vmax=2,
                       detrended=True)
print("Mean bias plotted!")

print("Plotting the standard deviation error...")
plot_spatial_efficiency(sde_raw,
                       geo_coords,
                       output_path,
                       "Standard Deviation Error",
                       unit="mg/m3",
                       cmap="viridis",
                       vmin=0,
                       vmax=3)
plot_spatial_efficiency(sde_detr,
                       geo_coords,
                       output_path,
                       "Standard Deviation Error",
                       unit="mg/m3",
                       cmap="viridis",
                       vmin=0,
                       vmax=3,
                       detrended=True)
print("Standard deviation error plotted!")

print("Plotting the cross correlation...")
plot_spatial_efficiency(cc_raw,
                       geo_coords,
                       output_path,
                       "Cross Correlation",
                       cmap="OrangeGreen",
                       vmin=-1,
                       vmax=1)
plot_spatial_efficiency(cc_detr,
                       geo_coords,
                       output_path,
                       "Cross Correlation",
                       cmap="OrangeGreen",
                       vmin=-1,
                       vmax=1,
                       detrended=True)
print("Cross correlation plotted!")

print("Plotting the std...")
plot_spatial_efficiency(rm_raw,
                       geo_coords,
                       output_path,
                       "Model Std Dev",
                       unit="mg/m3",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       suffix="(Model)")
plot_spatial_efficiency(rm_detr,
                       geo_coords,
                       output_path,
                       "Model Std Dev",
                       unit="mg/m3",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       detrended=True,
                       suffix="(Model)")

plot_spatial_efficiency(ro_raw,
                       geo_coords,
                       output_path,
                       "Satellite Std Dev",
                       unit="mg/m3",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       suffix="(Satellite)")
plot_spatial_efficiency(ro_detr,
                       geo_coords,
                       output_path,
                       "Satellite Std Dev",
                       unit="mg/m3",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       detrended=True,
                       suffix="(Satellite)")
print("Std plotted!")

print("Plotting the uRMSE...")
plot_spatial_efficiency(urmse_raw,
                       geo_coords,
                       output_path,
                       "Unbiased RMSE",
                       unit="mg/m3",
                       cmap="inferno",
                       vmin=0,
                       vmax=3)
plot_spatial_efficiency(urmse_detr,
                       geo_coords,
                       output_path,
                       "Unbiased RMSE",
                       unit="mg/m3",
                       cmap="inferno",
                       vmin=0,
                       vmax=3,
                       detrended=True)
print("uRMSE plotted!")
print('-'*45)

# ----- COMPUTING THE YEARLY METRCIS -----
# ----- 1ST METRICS, YEARLY AVG, RAW DATA -----
print("Computing the metrics from the raw data...")
mb_raw, sde_raw, cc_raw, rm_raw, ro_raw, urmse_raw = compute_spatial_efficiency(model_chl_masked, sat_chl_masked, time_group="year")

# Create a dict with metrics as keys, and each value is a list of 2D arrays, one per year
metrics_dict_raw = {
    'MB_raw': [mb_raw.isel(year=i) for i in range(len(mb_raw.year))],
    'SDE_raw': [sde_raw.isel(year=i) for i in range(len(sde_raw.year))],
    'CC_raw': [cc_raw.isel(year=i) for i in range(len(cc_raw.year))],
    'RM_raw': [rm_raw.isel(year=i) for i in range(len(rm_raw.year))],
    'RO_raw': [ro_raw.isel(year=i) for i in range(len(ro_raw.year))],
    'URMSE_raw': [urmse_raw.isel(year=i) for i in range(len(urmse_raw.year))],
}

# Convert to DataFrame: rows=years, columns=metrics, cells=2D DataArrays
metrics_df_raw = pd.DataFrame(metrics_dict_raw)

# Set index to years (as strings)
metrics_df_raw.index = [f'Year_{year}' for year in mb_raw.year.values]
print("Metrics computed!")
print('-'*45)

# ----- DETREND -----
print("Detrending the data...")
model_detrended = detrend_dim(model_chl_data, dim='time', mask=mask_expanded)
sat_detrended = detrend_dim(sat_chl_data, dim='time', mask=mask_expanded)
print("Data detrended!")

# ----- 2ND COMPUTATION, ALSO YEARLY AVG, DETRENDED DATA -----
print("Computing the metrics from the detrended data...")
mb_detr, sde_detr, cc_detr, rm_detr, ro_detr, urmse_detr = compute_spatial_efficiency(model_detrended, sat_detrended, time_group="year")

# Create a dict with metrics as keys, and each value is a list of 2D arrays, one per year
metrics_dict_detr = {
    'MB_detr': [mb_detr.isel(year=i) for i in range(len(mb_detr.year))],
    'SDE_detr': [sde_detr.isel(year=i) for i in range(len(sde_detr.year))],
    'CC_detr': [cc_detr.isel(year=i) for i in range(len(cc_detr.year))],
    'RM_detr': [rm_detr.isel(year=i) for i in range(len(rm_detr.year))],
    'RO_detr': [ro_detr.isel(year=i) for i in range(len(ro_detr.year))],
    'URMSE_detr': [urmse_detr.isel(year=i) for i in range(len(urmse_detr.year))],
}

# Convert to DataFrame: rows=years, columns=metrics, cells=2D DataArrays
metrics_df_detr = pd.DataFrame(metrics_dict_detr)

# Set index to years (as strings)
metrics_df_detr.index = [f'Year_{year}' for year in mb_detr.year.values]
print("Detrended data metrics computed!")
print('-'*45)

# ----- BEGIN PLOTTING AND SAVING -----
print("Plotting the results...")
print("Plotting the mean bias...")
plot_spatial_efficiency(mb_raw,
                       geo_coords,
                       output_path,
                       "Mean Bias",
                       unit="mg/m3",
                       cmap="RdBu_r",
                       vmin=-2,
                       vmax=2)
plot_spatial_efficiency(mb_detr,
                       geo_coords,
                       output_path,
                       "Mean Bias",
                       unit="mg/m3",
                       cmap="RdBu_r",
                       vmin=-2,
                       vmax=2,
                       detrended=True)
print("Mean bias plotted!")

print("Plotting the standard deviation error...")
plot_spatial_efficiency(sde_raw,
                       geo_coords,
                       output_path,
                       "Standard Deviation Error",
                       unit="mg/m3",
                       cmap="viridis",
                       vmin=0,
                       vmax=3)
plot_spatial_efficiency(sde_detr,
                       geo_coords,
                       output_path,
                       "Standard Deviation Error",
                       unit="mg/m3",
                       cmap="viridis",
                       vmin=0,
                       vmax=3,
                       detrended=True)
print("Standard deviation error plotted!")

print("Plotting the cross correlation...")
plot_spatial_efficiency(cc_raw,
                       geo_coords,
                       output_path,
                       "Cross Correlation",
                       cmap="OrangeGreen",
                       vmin=-1,
                       vmax=1)
plot_spatial_efficiency(cc_detr,
                       geo_coords,
                       output_path,
                       "Cross Correlation",
                       cmap="OrangeGreen",
                       vmin=-1,
                       vmax=1,
                       detrended=True)
print("Cross correlation plotted!")

print("Plotting the std...")
plot_spatial_efficiency(rm_raw,
                       geo_coords,
                       output_path,
                       "Model Std Dev",
                       unit="mg/m3",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       suffix="(Model)")
plot_spatial_efficiency(rm_detr,
                       geo_coords,
                       output_path,
                       "Model Std Dev",
                       unit="mg/m3",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       detrended=True,
                       suffix="(Model)")

plot_spatial_efficiency(ro_raw,
                       geo_coords,
                       output_path,
                       "Satellite Std Dev",
                       unit="mg/m3",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       suffix="(Satellite)")
plot_spatial_efficiency(ro_detr,
                       geo_coords,
                       output_path,
                       "Satellite Std Dev",
                       unit="mg/m3",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       detrended=True,
                       suffix="(Satellite)")
print("Std plotted!")

print("Plotting the uRMSE...")
plot_spatial_efficiency(urmse_raw,
                       geo_coords,
                       output_path,
                       "Unbiased RMSE",
                       unit="mg/m3",
                       cmap="inferno",
                       vmin=0,
                       vmax=3)
plot_spatial_efficiency(urmse_detr,
                       geo_coords,
                       output_path,
                       "Unbiased RMSE",
                       unit="mg/m3",
                       cmap="inferno",
                       vmin=0,
                       vmax=3,
                       detrended=True)
print("uRMSE plotted!")
print('-'*45)
