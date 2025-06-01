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
import subprocess

# Utility libraries
import numpy as np
import pandas as pd
import xarray as xr
import calendar
from datetime import datetime
import re
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor

###############################################################################
##                                                                           ##
##                                MODULES                                    ##
##                                                                           ##
###############################################################################

print("Loading the necessary modules...")

print("Loading the Pre-Processing functions...")
from Hydrological_model_validator.Processing.time_utils import (split_to_monthly, 
                                                                split_to_yearly)
from Hydrological_model_validator.Plotting.formatting import compute_geolocalized_coords
from Hydrological_model_validator.Processing.file_io import mask_reader
print("\033[92m✅ Pre-processing functions have been loaded!\033[0m")
print("-"*45)

print("Loading the plotting functions...")
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
print("\033[92m✅ The plotting functions have been loaded!\033[0m")
print('-'*45)

print("Loading the statistics functions...")
from Hydrological_model_validator.Processing.stats_math_utils import (detrend_dim)
print("\033[92m✅ The statistics functions have been loaded!\033[0m")
print('-'*45)

print("Loading the efficiency functions...")
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
                                                                        compute_spatial_efficiency,
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

# ----- IMPORTING BASIN AVERAGES -----

print("Importing the Basin Average SST timeseries...")
idir_path = Path(IDIR)
BASSTmod = xr.open_dataset(idir_path / 'BA_sst_mod_l3.nc')['BAmod_L3'].values
BASSTsat = xr.open_dataset(idir_path / 'BA_sst_sat_l3.nc')['BAsat_L3'].values
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

# Define the number of years in which to split the dataset
Ybeg, Yend = dates[0].year, dates[-1].year
ysec = list(range(Ybeg, Yend + 1))

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

###############################################################################
##                                                                           ##
##                           SPATIAL PERFORMANCE                             ##
##                                                                           ##
###############################################################################

# ----- RETRIEVING THE MASK -----
print("Retrieving the mask...")
Mfsm = mask_reader(BDIR)
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
ds_model = xr.open_dataset(IDIR / "ModData_sst_interp_l3.nc")
ds_sat = xr.open_dataset(IDIR / "SatData_sst_interp_l3.nc")
print("The datasets have been loaded!")
print('-'*45)

# ----- TRANSPOSING -----
print("Due to the necessity to resample the data the datasets need to be")
print("Transposed so that the 1st dimension is the time")
print("Transposing the datasets...")
model_sst = ds_model['ModData_interp'].transpose('time', 'lat', 'lon')
sat_sst = ds_sat['SatData_complete'].transpose('time', 'lat', 'lon')
print("The datasets have been transposed!")
print('-'*45)

# ----- ADDIING CORRECT DATETIME -----
print("Adding a datetime to aid with the resampling...")
time_origin = pd.Timestamp("2000-01-01")
model_sst['time'] = time_origin + pd.to_timedelta(model_sst.time.values, unit="D")
sat_sst['time'] = time_origin + pd.to_timedelta(sat_sst.time.values, unit="D")
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
    model_sst_chunked = model_sst.chunk({'time': 100})
    sat_sst_chunked = sat_sst.chunk({'time': 100})

    # ----- PERFORM THE RESAMPLING, TAKES A WHILE -----
    model_sst_monthly_lazy = model_sst_chunked.resample(time='1MS').mean()
    sat_sst_monthly_lazy = sat_sst_chunked.resample(time='1MS').mean()

    with ProgressBar(), ThreadPoolExecutor(max_workers=2) as executor:
        future_model = executor.submit(model_sst_monthly_lazy.compute, scheduler='threads')
        future_sat = executor.submit(sat_sst_monthly_lazy.compute, scheduler='threads')

        model_sst_monthly = future_model.result()
        sat_sst_monthly = future_sat.result()

    print("✅ Resampling completed.\n")
    
# ----- 2ND CHECK -----
if do_save:
    
    # ----- ONLY SAVE -----
    print("Saving daily SST data for external resampling via CDO...")
    model_sst.to_netcdf(Path(IDIR, "model_sst_daily.nc"))
    sat_sst.to_netcdf(Path(IDIR, "sat_sst_daily.nc"))
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
    model_sst.to_netcdf(Path(IDIR, "model_sst_daily.nc"))
    sat_sst.to_netcdf(Path(IDIR, "sat_sst_daily.nc"))
    print("✅ Files saved to:", IDIR, "\n")

    print("Running the CDO...")
    print("Firstly the model data...")
    
    # ----- BUILD THE PATHS AS LINUX -----
    input_file = Path("/mnt") / IDIR / "model_sst_daily.nc"
    output_file = Path('/mnt/', IDIR, "model_sst_monthly.nc")

    # ----- RUN THE CDO -----
    subprocess.run(["/usr/bin/cdo", "-v", "monmean", input_file, output_file], check=True)
    print("The model data has been resampled!")
    
    # ----- REDO THE SAME FOR THE SATELLITE
    print("Onto the satellite data...")
    input_file_sat = os.path.join(IDIR, "sat_sst_daily.nc")
    output_file_sat = os.path.join(IDIR, "sat_sst_monthly.nc")

    subprocess.run(["cdo", "-v", "monmean", input_file_sat, output_file_sat], check=True)
    print("The satellite data has been resampled!")
    
# ----- GET BACK TO THE ANALYSIS -----
# ----- LOAD NEW DATASETS -----    
print("Loading the monthly datasets...")
model_sst_monthly = xr.open_dataset(IDIR / "model_sst_monthly.nc")
sat_sst_monthly = xr.open_dataset(IDIR / "sat_sst_monthly.nc")
print("The datasets have been loaded!")
print('-'*45)

print("Fetching the data...")
model_sst_data = model_sst_monthly["ModData_interp"]
sat_sst_data = sat_sst_monthly["SatData_complete"]

# ----- APPLY THE MASK -----
# ----- REUSES THE SAME FROM THE DATA SETUPPING -----
print("Masking...")
mask_da = xr.DataArray(ocean_mask)
mask_expanded = mask_da.expand_dims(time=model_sst_data.time)
model_sst_masked = model_sst_data.where(mask_expanded)
sat_sst_masked = sat_sst_data.where(mask_expanded)

# ----- DROP THE BOUNDS DIMENSION IF IT'S THERE -----
print("Dropping the time bounds dimension...")
print("(This is added by the cdo)")
model_sst_data = model_sst_data.drop_vars("time_bnds", errors="ignore")
sat_sst_data = sat_sst_data.drop_vars("time_bnds", errors="ignore")

# ----- ALIGN AS A FAILSAFE -----
print("Aligning the data...")
model_sst_data, sat_sst_data = xr.align(model_sst_data, sat_sst_data, join="inner")
print("The monthly data has been prepared!")
print('-'*45)

# ----- 1ST METRICS, MONTHLY AVG, RAW DATA -----
print("Computing the metrics from the raw data...")
mb_raw, sde_raw, cc_raw, rm_raw, ro_raw, urmse_raw = compute_spatial_efficiency(model_sst_masked, sat_sst_masked)
print("Metrics computed!")
print('-'*45)

# ----- DETREND -----
print("Detrending the data...")
model_detrended = detrend_dim(model_sst_data, dim='time', mask=mask_expanded)
sat_detrended = detrend_dim(sat_sst_data, dim='time', mask=mask_expanded)
print("Data detrended!")

# ----- 2ND COMPUTATION, ALSO MONTHLY AVG, DETRENDED DATA -----
print("Computing the metrics from the detrended data...")
mb_detr, sde_detr, cc_detr, rm_detr, ro_detr, urmse_detr = compute_spatial_efficiency(model_detrended, sat_detrended)
print("Detrended data metrcis computed!")
print('-'*45)

# ----- SET UP THE SAVE FOLDER -----
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "SST", timestamp)
os.makedirs(output_path, exist_ok=True)

# ----- BEGIN PLOTTING AND SAVING -----
print("Plotting the results...")
print("Plotting the mean bias...")
plot_spatial_efficiency(mb_raw, geo_coords, output_path, "Mean Bias (°C)", "RdBu_r", -2, 2, )
plot_spatial_efficiency(mb_detr, geo_coords, output_path, "Mean Bias (°C)", "RdBu_r", -2, 2, detrended=True)
print("Mean bias plotted!")

print("Plotting the standard deviation error...")
plot_spatial_efficiency(sde_raw, geo_coords, output_path, "Standard Deviation Error (°C)", "viridis", 0, 3)
plot_spatial_efficiency(sde_detr, geo_coords, output_path, "Standard Deviation Error (°C)", "viridis", 0, 3, detrended=True)
print("Standard deviation error plotted!")

print("Plotting the cross correlation...")
plot_spatial_efficiency(cc_raw, geo_coords, output_path, "Cross Correlation", "OrangeGreen", -1, 1)
plot_spatial_efficiency(cc_detr, geo_coords, output_path, "Cross Correlation", "OrangeGreen", -1, 1, detrended=True)
print("Cross correlation plotted!")

print("Plotting the std...")
plot_spatial_efficiency(rm_raw, geo_coords, output_path, "Model Std Dev (°C)", "plasma", 0, 3, suffix="(Model)")
plot_spatial_efficiency(rm_detr, geo_coords, output_path, "Model Std Dev (°C)", "plasma", 0, 3, detrended=True, suffix="(Model)")

plot_spatial_efficiency(ro_raw, geo_coords, output_path, "Satellite Std Dev (°C)", "plasma", 0, 3, suffix="(Satellite)")
plot_spatial_efficiency(ro_detr, geo_coords, output_path, "Satellite Std Dev (°C)", "plasma", 0, 3, detrended=True, suffix="(Satellite)")
print("Std plotted!")

print("Plotting the uRMSE...")
plot_spatial_efficiency(urmse_raw, geo_coords, output_path, "Unbiased RMSE (°C)", "inferno", 0, 3)
plot_spatial_efficiency(urmse_detr, geo_coords, output_path, "Unbiased RMSE (°C)", "inferno", 0, 3, detrended=True)
print("uRMSE plotted!")
print('-'*45)