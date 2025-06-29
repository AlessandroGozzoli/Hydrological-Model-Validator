#!/usr/bin/python
# -*- coding: utf-8 -*-

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
##                               EXPLANATION                                 ##
###############################################################################
###############################################################################

    title = "WELCOME TO THE SST DATA ANALYSIS TEST CASE SCRIPT"
    border = "#" * 60
    print(border)
    print(title.center(60))
    print(border)


    print("""\nThis is a test case which illustrated the possible analysis that
can be done using the tools provided by the package.
This code retrieve the test case data obtained by processing the
raw input data (usually cmems satellite data and nemo simulations
datasets) using the processes illustrated in the Data_reader_setupper
test case script.
         
If you've not done so we suggest takig a look at the computations
done in that script to better understand how the data is prepared.
          
Here the analysis will be focused on the Sea Surface Temperature
(SST) field and it will revolve around some basic plotting, some
more advanced efficiency evaluation and finally an error
decomposition""")

    input("Please press any key to confirm and move on: \n")

###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

    # Libraries for paths/os interaction
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

    print(border)
    print("\nLoading the necessary modules provided by the package...")

    print("Loading the Pre-Processing functions...")
    from Hydrological_model_validator.Processing.time_utils import (split_to_monthly, 
                                                                    split_to_yearly,
                                                                    resample_and_compute)
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
                                                           plot_spatial_efficiency,
                                                           error_components_timeseries,
                                                           plot_spectral)
    from Hydrological_model_validator.Plotting.Taylor_diagrams import (comprehensive_taylor_diagram,
                                                                       monthly_taylor_diagram)
    from Hydrological_model_validator.Plotting.Target_plots import (comprehensive_target_diagram,
                                                                target_diagram_by_month)
    print("\033[92m✅ The plotting functions have been loaded!\033[0m")
    print('-'*45)

    print("Loading the statistics functions...")
    from Hydrological_model_validator.Processing.stats_math_utils import (detrend_dim,
                                                                      corr_no_nan,
                                                                      compute_fft)
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
                                                                        compute_error_timeseries)
    print("\033[92m✅ The validation modules have been loaded!\033[0m\n")
    print(border)

###############################################################################
##                                                                           ##
##                             DATA LOADING                                  ##
##                                                                           ##
###############################################################################

    # ----- SETTING UP THE WORKING DIRECTORY -----
    print("\nResetting the working directory...")
    WDIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(WDIR)  # Set the working directory to the script location
    print('*' * 45)

    # ----- BASE DATA DIRECTORY -----
    BDIR = Path(WDIR, "Data")

    # ----- INPUT DATA DIRECTORY -----
    IDIR = Path(BDIR, "PROCESSING_INPUT/")
    print("Loading the input data...")
    print(f"""\033[91m⚠️ The input data needs to be located in the {IDIR} folder ⚠️
⚠️ Make sure that it contains all of the necessary datasets ⚠️\033[0m
An example dataset is also provided alongside the repository and it 
contains both the data and a mask...
Make sure to download it!""")
    
    print("-"*45)

    print("The folder contains the following datasets")
    # List the contents of the folder
    contents = os.listdir(IDIR)
    # Print the contents
    print(contents)
    print("-"*45)
    
    # ----- RETRIEVING THE MASK -----
    print("Retrieving the mask...")
    Mfsm = mask_reader(BDIR)
    ocean_mask = Mfsm[0]  # This returns a NumPy array
    print("\033[92m✅ Mask succesfully imported! \033[0m\n")
    print(border)

###############################################################################
##                                                                           ##
##                              DICTIONARIES                                 ##
##                                                                           ##
###############################################################################

    print("""\nNow that the functions have been imported and the paths have been
initialized the code will retrieve the data from the folder and build
the necessary dictionaries...""")

    print("Setting up the SST dictionary...")
    print('-'*45) 

    # ----- IMPORTING 2D DAILY DATASETS -----

    # ----- LOADING THE DAILY DATASETS, OUTPUT OF THE INTERPOLATOR.M -----
    print("Loading the datasets...")
    ds_model = xr.open_dataset(IDIR / "ModData_sst_interp_l3.nc")
    ds_sat = xr.open_dataset(IDIR / "SatData_sst_interp_l3.nc")
    print("The datasets have been loaded!")
    print('-'*45)

    # ----- TRANSPOSING -----
    print("""Due to the necessity to resample the data further down the line
          the datasets need to be transposed so that the 1st dimension is the time""")
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

    # ----- IMPORTING BASIN AVERAGES -----

    print("Importing the Basin Average SST timeseries...")
    idir_path = Path(IDIR)
    BASSTmod_raw = xr.open_dataset(idir_path / 'BA_sst_mod_l3.nc')['BAmod_L3']
    BASSTsat_raw = xr.open_dataset(idir_path / 'BA_sst_sat_l3.nc')['BAsat_L3']
    BASSTmod = BASSTmod_raw.values
    BASSTsat = BASSTsat_raw.values
    print("\033[92m✅ Basin Average Timeseries obtained!\033[0m")

    # Generate datetime index (Daily from 2000 to 2009 (used only with the full
    # dataset, not provided as test case data due to its size))
    dates = pd.date_range(start='2000-01-01', end='2000-12-31', freq='D')

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
    
    print("""\nNow that the dicstionaries have been created and splitted to 
better handle them, the analysis can begin...
          
\033[91m⚠️ HEADS UP ⚠️
To ensure that the code runs smoothly the plots will not
be displayed. After the plot's window closes it will be
saved in the appropriate folder for further analysis.\033[0m
                                                                    
This is done to keep the test case running smoothly and to avoid
memory overload""")

    input("Please press any key to confirm and move on: \n")
    print(border)

###############################################################################
##                                                                           ##
##                              OTHER PLOTS                                  ##
##                                                                           ##
###############################################################################

    print("\nThe plotting will commence...")

    print("Plotting basic plots...")

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
    # Default options are being used
    timeseries(
        BASST,
        BIAS,
        variable_name='SST',
        BA=False,
        output_path=output_path
        )
    print("\033[92m✅ Time-series plotted succesfully!\033[0m")

    # ----- SCATTERPLOTS -----

    print("Plotting the scatter plot...")
    # Default options
    scatter_plot(BASST, variable_name='SST', BA=False, output_path=output_path)
    print("\033[92m✅ Scatter plot plotted succesfully!\033[0m")

    print("Plotting the seasonal data as scatterplots...")
    # Default options
    seasonal_scatter_plot(BASST, variable_name='SST', BA=False, output_path=output_path)
    print("\033[92m✅ Seasonal scatterplots plotted succesfully!\033[0m")

    # ----- WHISKERBOX PLOTS -----

    print("Plotting the whisker-box plots...")
    # Default options
    whiskerbox(BASST_monthly, variable_name='SST', output_path=output_path)
    print("\033[92m✅ Whisker-box plotted succesfully!\033[0m")

    # ----- VIOLIN PLOTS -----

    print("Plotting the violinplots...")
    # Default options
    violinplot(BASST_monthly, variable_name='SST', output_path=output_path)
    print("\033[92m✅ Violinplots plotted succesfully!\033[0m\n")
    print(border)

###############################################################################
##                                                                           ##
##                            TAYLOR DIAGRAMS                                ##
##                                                                           ##
###############################################################################

    print("""\nNow the plots will be tailor toward the evaluation of the performance
of the dataset...""")

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

    print("""Now the results of the efficiency metrics will be computaed
and plotted...""")

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
                        metric_name=metric_key,
                        output_path=output_path)
    
        # Remove any parentheses and their contents for the print message
        clean_title = re.sub(r'\s*\([^)]*\)', '', title)
        print(f"\033[92m✅ {clean_title} plotted!\033[0m")

    print("\033[92m✅ All efficiency metric plots have been successfully created!\033[0m\n")
    print(border)

###############################################################################
##                                                                           ##
##                           SPATIAL PERFORMANCE                             ##
##                                                                           ##
###############################################################################

    print("""\nNow the error decomposition analysis will begin...
          
In this section 2D maps illustrating the spatial behaviour of the
error components will be analysed to better illustrate critical
areas.
          
Before doing the analysis some pre-processing needs to be done""")
    input("Please press any key to confirm and move on: ")

    # ----- GEOLOCALIZE THE DATASET -----
    print("\nComputing the geolocalization...")
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
    
    # ----- WARNING AND RESAMPLING -----
    print("""\n--- Data Resampling Notice ---
This dataset must be resampled to obtain *monthly averages*, which are
required for the subsequent analyses (tailored to monthly/yearly data).

⚠️  Resampling is computationally intensive, especially on the full dataset.
Three options are available:
1. Perform in-code resampling using parallel Dask processes.
2. Save the dataset and use an external tool like CDO for resampling.
3. Save the dataset to resample using another program.

The resampling using CDO is the current fastest method,
but in this test case the already resampled file is already provided!
""")

    # ----- GET INITIAL USER DECISION -----
    choice = input("→ Choose a resampling option (in-code / cdo / save / skip): ").strip().lower()

    if choice in ["in-code", "code", "1"]:
        # ----- 1ST CHECK -----
        print("\n✅ Starting parallel resampling using Dask...")
        
        # ----- CUTTING DATA IN CHUNKS TO SPEED UP -----
        model_sst_chunked = model_sst.chunk({'time': 100})
        sat_sst_chunked = sat_sst.chunk({'time': 100})
        
        # ----- PERFORM THE RESAMPLING, TAKES A WHILE -----
        model_sst_data, sat_sst_data = resample_and_compute(model_sst_chunked, sat_sst_chunked)
        print("✅ Resampling completed.\n")

    elif choice in ["cdo", "2"]:
        # ----- 2ND CHECK -----
        print("Saving daily SST data for external resampling via CDO...")
        model_sst.to_netcdf(Path(IDIR, "model_sst_daily.nc"))
        sat_sst.to_netcdf(Path(IDIR, "sat_sst_daily.nc"))
        print("✅ Files saved to:", IDIR, "\n")
        
        print("Running the CDO...")
        print("Firstly the model data...")
        
        # ----- BUILD THE PATHS AS LINUX -----
        input_file = Path("/mnt") / IDIR / "model_sst_daily.nc"
        output_file = Path("/mnt") / IDIR / "model_sst_monthly.nc"
        
        # ----- RUN THE CDO -----
        subprocess.run(["/usr/bin/cdo", "-v", "monmean", input_file, output_file], check=True)
        print("The model data has been resampled!")
        
        # ----- REDO THE SAME FOR THE SATELLITE -----
        print("Onto the satellite data...")
        input_file_sat = os.path.join(IDIR, "sat_sst_daily.nc")
        output_file_sat = os.path.join(IDIR, "sat_sst_monthly.nc")
        
        subprocess.run(["cdo", "-v", "monmean", input_file_sat, output_file_sat], check=True)
        print("The satellite data has been resampled!")

    elif choice in ["save", "3"]:
        # ----- 3RD CHECK -----
        print("Saving daily SST data for external resampling via another tool...")
        # ----- ONLY SAVE -----
        model_sst.to_netcdf(Path(IDIR, "model_sst_daily.nc"))
        sat_sst.to_netcdf(Path(IDIR, "sat_sst_daily.nc"))
        print("✅ Files saved to:", IDIR, "\n")

    # ----- EXTREME SCENARIO, EITHER EXAMPLE DATA OR KILL -----
    elif choice in ["skip", "4"]:
        check = input("Are we using the already resampled datasets? (Yes/No): ").strip().lower()
        if check in ["yes", "y"]:
            print("Good! We can move on then!")
            print("Loading the monthly datasets...")

            # ----- LOAD NEW DATASETS -----   
            model_sst_monthly = xr.open_dataset(IDIR / "model_sst_monthly.nc")
            sat_sst_monthly = xr.open_dataset(IDIR / "sat_sst_monthly.nc")
            print("✅ Datasets loaded!")

            print("Fetching the data...")
            model_sst_data = model_sst_monthly["ModData_interp"]
            sat_sst_data = sat_sst_monthly["SatData_complete"]

        else:
            print("\n❌ No valid resampling path selected. Cannot proceed. ❌")
            exit()

    else:
        print("\n❌ Invalid input. Please choose one of the listed options. ❌")
        exit()
    
    # ----- GET BACK TO THE ANALYSIS -----

    # ----- APPLY THE MASK -----
    # ----- REUSES THE SAME FROM THE DATA SETUPPING -----
    print("Masking the datasets...")
    mask_da = xr.DataArray(Mfsm)
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
    model_sst_data, sat_sst_data = xr.align(model_sst_masked, sat_sst_masked, join="inner")

    # ----- ADD MONTH/YEAR -----
    model_sst_masked = model_sst_masked.assign_coords(
        month=model_sst_masked.time.dt.month,
        year=model_sst_masked.time.dt.year,
        )
    sat_sst_masked = sat_sst_masked.assign_coords(
        month=sat_sst_masked.time.dt.month,
        year=sat_sst_masked.time.dt.year,
        )
    print("The monthly data has been prepared!")
    print('-'*45)

    print("Now the computation can begin")

    # ----- 1ST METRICS, MONTHLY AVG, RAW DATA -----
    print("Computing the metrics from the raw data...")
    mb_raw, sde_raw, cc_raw, rm_raw, ro_raw, urmse_raw = compute_spatial_efficiency(model_sst_masked, sat_sst_masked, time_group="month")

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
    model_detrended = detrend_dim(model_sst_data, dim='time', mask=mask_expanded)
    sat_detrended = detrend_dim(sat_sst_data, dim='time', mask=mask_expanded)
    print("Data detrended!")

    # ----- 2ND COMPUTATION, ALSO MONTHLY AVG, DETRENDED DATA -----
    print("Computing the metrics from the detrended data...")
    mb_detr, sde_detr, cc_detr, rm_detr, ro_detr, urmse_detr = compute_spatial_efficiency(model_detrended, sat_detrended, time_group="month")

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
    output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "SST", timestamp)
    os.makedirs(output_path, exist_ok=True)

    # ----- BEGIN PLOTTING AND SAVING -----
    print("Plotting the results...")
    print("Plotting the mean bias...")
    plot_spatial_efficiency(mb_raw,
                        geo_coords,
                        output_path,
                        "Mean Bias",
                        unit="°C",
                        cmap="RdBu_r",
                        vmin=-2,
                        vmax=2)
    plot_spatial_efficiency(mb_detr,
                        geo_coords,
                        output_path,
                        "Mean Bias",
                        unit="°C",
                        cmap="RdBu_r",
                        vmin=-2,
                        vmax=2,
                        detrended=True)
    print("Mean bias plotted!")

    print("Plotting the standard deviation error...")
    plot_spatial_efficiency(sde_raw, geo_coords, output_path,
                        "Standard Deviation Error",
                        unit="°C",
                        cmap="viridis",
                        vmin=0,
                        vmax=3)
    plot_spatial_efficiency(sde_detr, geo_coords, output_path,
                        "Standard Deviation Error",
                        unit="°C",
                        cmap="viridis",
                        vmin=0,
                        vmax=3,
                        detrended=True)
    print("Standard deviation error plotted!")

    print("Plotting the cross correlation...")
    plot_spatial_efficiency(cc_raw, geo_coords, output_path,
                        "Cross Correlation",
                        cmap="OrangeGreen",
                        vmin=-1,
                        vmax=1)
    plot_spatial_efficiency(cc_detr, geo_coords, output_path,
                        "Cross Correlation",
                        cmap="OrangeGreen",
                        vmin=-1,
                        vmax=1,
                        detrended=True)
    print("Cross correlation plotted!")

    print("Plotting the std...")
    plot_spatial_efficiency(rm_raw, geo_coords, output_path,
                        "Model Std Dev",
                        unit="°C",
                        cmap="plasma",
                        vmin=0,
                        vmax=8,
                        suffix="(Model)")
    plot_spatial_efficiency(rm_detr, geo_coords, output_path,
                        "Model Std Dev",
                        unit="°C",
                        cmap="plasma",
                        vmin=0,
                        vmax=8,
                        detrended=True,
                        suffix="(Model)")

    plot_spatial_efficiency(ro_raw, geo_coords, output_path,
                        "Satellite Std Dev",
                        unit="°C",
                        cmap="plasma",
                        vmin=0,
                        vmax=8,
                        suffix="(Satellite)")
    plot_spatial_efficiency(ro_detr, geo_coords, output_path,
                        "Satellite Std Dev",
                        unit="°C",
                        cmap="plasma",
                        vmin=0,
                        vmax=8,
                        detrended=True,
                        suffix="(Satellite)")
    print("Std plotted!")

    print("Plotting the uRMSE...")
    plot_spatial_efficiency(urmse_raw, geo_coords, output_path,
                        "Unbiased RMSE",
                        unit="°C",
                        cmap="inferno",
                        vmin=0,
                        vmax=3)
    plot_spatial_efficiency(urmse_detr, geo_coords, output_path,
                        "Unbiased RMSE",
                        unit="°C",
                        cmap="inferno",
                        vmin=0,
                        vmax=3,
                        detrended=True)
    print("uRMSE plotted!")
    print('-'*45)

    # ----- COMPUTING THE YEARLY METRCIS -----
    # ----- 1ST METRICS, YEARLY AVG, RAW DATA -----
    print("Moving onto the yearly data...")
    print("Computing the metrics from the raw data...")
    mb_raw, sde_raw, cc_raw, rm_raw, ro_raw, urmse_raw = compute_spatial_efficiency(model_sst_masked, sat_sst_masked, time_group="year")

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
    model_detrended = detrend_dim(model_sst_data, dim='time', mask=mask_expanded)
    sat_detrended = detrend_dim(sat_sst_data, dim='time', mask=mask_expanded)
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
                       unit="°C",
                       cmap="RdBu_r",
                       vmin=-2,
                       vmax=2)
    plot_spatial_efficiency(mb_detr,
                       geo_coords,
                       output_path,
                       "Mean Bias",
                       unit="°C",
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
                       unit="°C",
                       cmap="viridis",
                       vmin=0,
                       vmax=3)
    plot_spatial_efficiency(sde_detr,
                        geo_coords,
                       output_path,
                       "Standard Deviation Error",
                       unit="°C",
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
                       unit="°C",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       suffix="(Model)")
    plot_spatial_efficiency(rm_detr,
                       geo_coords,
                       output_path,
                       "Model Std Dev",
                       unit="°C",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       detrended=True,
                       suffix="(Model)")

    plot_spatial_efficiency(ro_raw,
                       geo_coords,
                       output_path,
                       "Satellite Std Dev",
                       unit="°C",
                       cmap="plasma",
                       vmin=0,
                       vmax=8,
                       suffix="(Satellite)")
    plot_spatial_efficiency(ro_detr,
                       geo_coords,
                       output_path,
                       "Satellite Std Dev",
                       unit="°C",
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
                       unit="°C",
                       cmap="inferno",
                       vmin=0,
                       vmax=3)
    plot_spatial_efficiency(urmse_detr,
                       geo_coords,
                       output_path,
                       "Unbiased RMSE",
                       unit="°C",
                       cmap="inferno",
                       vmin=0,
                       vmax=3,
                       detrended=True)
    print("uRMSE plotted!\n")
    print('*'*45)

###############################################################################
##                                                                           ##
##                         CLOUD COVER CORRELATION                           ##
##                                                                           ##
###############################################################################

    print("""\nIn this section the cloud cover % will be compared to the timseries
of the error components (Mean BIAS, SDE, uRMSE and CC)
To see their behaviour both as timesries data and using a spectral 
analysis.

The dataset that needs to be used is the 2D daily value one already
Imported in the previous sections of this test case, but in this case
it must remain composed of daily values!")

The data contains Nans so it can be either masked or they can be skipped...")
Since we are using already masked data the problem is not relevant for
our case but for future knowledge this is crucial
          
We can start with the processing...\n""")
    input("Please press any key to confirm and move on: ")

    print("Computing the error components timeseries...")
    error_comp_stats_df = compute_error_timeseries(model_sst, sat_sst, ocean_mask)
    print("Error components timeseries obtaines!")

    print('-'*45)

    print("Retrieving the cloud cover...")
    cloud_cover = xr.open_dataset(idir_path / "cloud_cover_sst.nc")['cloud_cover_sst']
    cloud_cover = pd.Series(cloud_cover, index=dates)
    print("Cloud cover timeseries obtained!")

    print("Computing the smoothed 30 day windows...")
    cloud_cover_30d = cloud_cover.rolling(window=30, center=True).mean()
    print("Smoothed timeseries computed!")

    print('-'*45)

    # ----- SET UP THE SAVE FOLDER -----
    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "EFFICIENCY", "SST", timestamp)
    os.makedirs(output_path, exist_ok=True)
    
    print("Plotting the results...")
    error_components_timeseries(error_comp_stats_df, output_path, 
                                cloud_cover, variable_name='SST')
    print("Results plotted!")

    correlations = {
        'raw_cloud_cover': {},
        'smoothed_cloud_cover_30d': {}
        }

    for metric in error_comp_stats_df.columns:
        corr_raw = corr_no_nan(error_comp_stats_df[metric], cloud_cover)
        corr_smooth = corr_no_nan(error_comp_stats_df[metric], cloud_cover_30d)
        correlations['raw_cloud_cover'][metric] = corr_raw
        correlations['smoothed_cloud_cover_30d'][metric] = corr_smooth

    print("The correlation between cloud cover and error components are...")
    print("Correlation between stats metrics and RAW cloud cover:")
    for metric, corr_val in correlations['raw_cloud_cover'].items():
        print(f"  {metric}: {corr_val:.4f}")
    
    print("\nCorrelation between stats metrics and SMOOTHED (30d) cloud cover:")
    for metric, corr_val in correlations['smoothed_cloud_cover_30d'].items():
        print(f"  {metric}: {corr_val:.4f}")
    print('-'*45)

    print("""Before finishing the plots for the spectral analysis the data
needs to be cleanded and detrended""")
    # ----- CLEAN DATA -----
    combined_df = error_comp_stats_df.copy()
    combined_df['cloud_cover'] = cloud_cover
    combined_df['cloud_cover_30d'] = cloud_cover_30d

    combined_df = combined_df.dropna()

    # Separate cleaned data
    error_comp_clean = combined_df[error_comp_stats_df.columns]
    cloud_cover_clean = combined_df['cloud_cover']
    cloud_cover_30d_clean = combined_df['cloud_cover_30d']

    # ----- DETREND ------
    # It is not necessary to use the detrend function as it is designe for 2D data
    error_comp_detrended = error_comp_clean - error_comp_clean.mean()
    cloud_cover_detrended = cloud_cover_clean - cloud_cover_clean.mean()
    cloud_cover_30d_detrended = cloud_cover_30d_clean - cloud_cover_30d_clean.mean()
    print("Data cleaned and deterended, proceeding to plot...")

    # ----- MAKE THE FFT -----
    
    print("Computing the fft components...")
    # For dict of error components
    freqs, fft_components = compute_fft(error_comp_detrended.to_dict('series'))
    
    # For cloud cover series
    _, fft_cloud = compute_fft(cloud_cover_detrended.values)
    
    # For 30-day cloud cover
    _, fft_cloud_30d = compute_fft(cloud_cover_30d_detrended.values)
    
    print("Plotting the PSD...")
    # PSD plot
    plot_spectral(plot_type='PSD', freqs=freqs, fft_components=fft_components, output_path=output_path)
    print("PSD plotted!")

    print("Plotting the CSD...")
    # CSD vs cloud_cover
    plot_spectral(
        plot_type='CSD',
        error_comp=error_comp_clean,
        output_path=output_path,
        cloud_covers=[
            (cloud_cover_clean, 'cloud_cover'),
            (cloud_cover_30d_clean, 'cloud_cover_30d')
            ])
    print("CSD plotted!\n")
    print(border)
    
if __name__ == "__main__":
    main()