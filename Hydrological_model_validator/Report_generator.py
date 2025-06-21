#!/usr/bin/python
# -*- coding: utf-8 -*-

def generate_full_report(
    data_folder,
    output_dir = None,
    check_only = False,
    generate_pdf = True,
    verbose = False,
    variable = None,
    unit = None,
    open_report = False
):
    # ===== LIBRARIES =====
    # General utility libraries
    import pandas as pd
    import numpy as np
    import xarray as xr
    import rasterio
    from pathlib import Path
    import os
    from datetime import datetime
    import calendar
    import re
    from reportlab.lib.units import inch
    
    # Module specific libraries
    from Hydrological_model_validator.Processing.utils import convert_dataarrays_in_df
    from Hydrological_model_validator.Processing.file_io import (mask_reader,
                                                                 find_file_with_keywords,
                                                                 select_3d_variable)
    from Hydrological_model_validator.Processing.Data_saver import save_variable_to_json
    from Hydrological_model_validator.Processing.time_utils import (split_to_monthly,
                                                                    split_to_yearly,
                                                                    is_invalid_time_index,
                                                                    prompt_for_datetime_index,
                                                                    ensure_datetime_index)
    from Hydrological_model_validator.Plotting.Plots import (timeseries,
                                                             seasonal_scatter_plot,
                                                             whiskerbox,
                                                             violinplot,
                                                             efficiency_plot,
                                                             error_components_timeseries,
                                                             plot_spectral,
                                                             plot_spatial_efficiency)
    from Hydrological_model_validator.Plotting.Taylor_diagrams import (comprehensive_taylor_diagram,
                                                                       monthly_taylor_diagram)
    from Hydrological_model_validator.Plotting.Target_plots import (comprehensive_target_diagram,
                                                                    target_diagram_by_month)
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
    from Hydrological_model_validator.Processing.stats_math_utils import (compute_coverage_stats,
                                                                          corr_no_nan,
                                                                          compute_fft,
                                                                          detrend_dim)
    from Hydrological_model_validator.Plotting.formatting import compute_geolocalized_coords
    from Hydrological_model_validator.Report.report_utils import (PDFReportBuilder,
                                                                  add_rotated_image_page,
                                                                  add_seasonal_scatter_page,
                                                                  add_multiple_rotated_images_grid,
                                                                  add_multiple_images_grid,
                                                                  add_efficiency_pages,
                                                                  add_tables_page,
                                                                  add_plot_to_pdf)
    def vprint(*args, verbose=False, **kwargs):
        if verbose:
            print(*args, **kwargs)
    # Border for terminal prints
    border = "#" * 60
    
    vprint("Starting to retrieve the data to make the report...", verbose=verbose)

    # ===== FILE RETRIEVAL =====
    # Detect if input is a folder or dict of paths
    if isinstance(data_folder, str):
        # Convert string path to a pathlib.Path object for convenient filesystem operations
        data_folder = Path(data_folder)
        if not data_folder.is_dir():
            raise ValueError(f"❌ Provided path '{data_folder}' is not a valid directory. ❌")

        vprint(f"Reading input files from folder: {data_folder}", verbose=verbose)
        # Get a list of all files in the provided folder
        files_in_folder = list(data_folder.iterdir())

        # Attempt to find the spatial data file based on keyword hints
        obs_spatial_path = find_file_with_keywords(files_in_folder, ['obs', 'observed', 'sat', 'satellite'], "observed spatial data")
        sim_spatial_path = find_file_with_keywords(files_in_folder, ['sim', 'simulated', 'model', 'mod'], "simulated spatial data")
        
        # Attempt to find the timeseries data files based on hints
        obs_ts_path = find_file_with_keywords(files_in_folder, ['obs', 'observed', 'sat', 'satellite'], "observed timeseries data")
        sim_ts_path = find_file_with_keywords(files_in_folder, ['sim', 'simulated', 'model', 'mod'], "simulated timeseries data")
        
        # Attempt to find the mask
        mask_path = find_file_with_keywords(files_in_folder, ['mask'], "mask data")

    elif isinstance(data_folder, dict):
        required_keys = ['obs_spatial', 'sim_spatial', 'obs_ts', 'sim_ts', 'mask']
        missing_keys = [k for k in required_keys if k not in data_folder]
        if missing_keys:
            raise ValueError(f"❌ Missing keys in data dictionary: {missing_keys}")

        vprint("Using explicit file paths provided via dictionary.", verbose=verbose)
        # If user provides a dictionary of paths explicitly, extract and convert each one to a Path object
        obs_spatial_path = Path(data_folder['obs_spatial'])
        sim_spatial_path = Path(data_folder['sim_spatial'])
        obs_ts_path = Path(data_folder['obs_ts'])
        sim_ts_path = Path(data_folder['sim_ts'])
        mask_path = Path(data_folder['mask'])
    else:
        raise ValueError("❌ Input must be a folder path string or a dict of file paths ❌")
        
    vprint("Inputs paths validated!", verbose=verbose)
    
    if check_only:
        return
    
    vprint("\n" + border + "\n", verbose=verbose)

    vprint("Beginning to read the files...", verbose=verbose)

    # ===== MASK RETRIEVAL =====
    vprint("Getting the mask...", verbose=verbose)
    # Extract the necessary mask elements
    Mmask, Mfsm, _, _, _ = mask_reader(mask_path)
    # Assign it to the ocean mask to be used later
    ocean_mask=Mmask
    
    vprint("Mask obtained!", verbose=verbose)
    vprint("\n" + border + "\n", verbose=verbose)

    # ===== LOAD THE SPATIAL DATASETS =====
    for label, path in [('obs_spatial', obs_spatial_path), ('sim_spatial', sim_spatial_path)]:
        suffix = Path(path).suffix.lower() # Get the file extension, lowecase for consistency

        # Handle NetCDF/HDF5 file types
        if suffix in ['.nc', '.netcdf', '.h5', '.hdf5']:
            ds = xr.open_dataset(path) # Use Xarray to open

            dataarray = select_3d_variable(ds, label) # Extract the main variable in the dataset

            # Transpose dims if needed to ensure (time, lat, lon)
            expected_dims = ('time', 'lat', 'lon')
            if dataarray.dims != expected_dims:
                vprint("⚠️ Dataarray dimensions do not match the expected one!", verbose=verbose)
                vprint("Attempting to transpose the data...", verbose=verbose)
                try:
                    dataarray = dataarray.transpose(*expected_dims)
                    vprint("Data transposed, new dimensions are (time, lat, lon)", verbose=verbose)
                except ValueError as e:
                    raise ValueError(f"❌ Cannot transpose {dataarray.dims} to {expected_dims} in {label}: {e} ❌")

            # Handle time index validity
            if 'time' in dataarray.coords:
                time_vals = dataarray['time'].values
                if is_invalid_time_index(time_vals): # Handle cases like constant or dummy time values
                    vprint(f"⚠️ Detected invalid trivial time index in {label}, asking for manual input. ⚠️", verbose=verbose)
                    length = dataarray.sizes['time']
                    # Ask for the starting time/frquency and apply it
                    time_index = prompt_for_datetime_index(length)
                    dataarray = dataarray.assign_coords(time=time_index)
                else:
                    # Try to decode time values if they are encoded as CF-compliant attributes
                    try:
                        dataarray = xr.decode_cf(dataarray)
                    except Exception:
                        pass
            else:
                # If no time coordinate is present, ask for it
                vprint(f"⚠️ No 'time' coordinate found in {label}. Asking for manual input. ⚠️", verbose=verbose)
                length = dataarray.sizes['time']
                time_index = prompt_for_datetime_index(length)
                dataarray = dataarray.assign_coords(time=time_index)

        # Handle geospatial raster images (e.g., GeoTIFFs)
        elif suffix in ['.tif', '.tiff']:
            with rasterio.open(path) as src:
                arr = src.read()  # shape: (bands, y, x)

                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]  # Convert to (1, y, x) by expanding
                elif arr.ndim != 3:
                    raise ValueError(f"❌ Unexpected TIFF shape {arr.shape} in {Path(path).name} ❌")
                # Convert numpy array to xarray DataArray (no time dimension in TIFF)
                import xarray as xr
                dataarray = xr.DataArray(arr, dims=('band', 'y', 'x'))
        else:
            raise ValueError(f"❌ Unsupported spatial data extension for {label}: {suffix} ❌")

        # Store the datasets
        if label == 'obs_spatial':
            obs_spatial = dataarray
        else:
            sim_spatial = dataarray

    # ===== LOAD THE TIMESERIES =====
    for label, path in [('obs_ts', obs_ts_path), ('sim_ts', sim_ts_path)]:
        suffix = path.suffix.lower()
        if suffix in ['.csv', '.txt']:
            # Read assuming 1st column is the datetime
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        elif suffix in ['.xls', '.xlsx']:
            # Read assuming the 1st column is the datetime
            df = pd.read_excel(path, index_col=0, parse_dates=True)
        elif suffix in ['.nc', '.netcdf']:
            ds = xr.open_dataset(path)
            # Attempt to auto-detect 1D time series variable
            time_dim = [v for v in ds.data_vars if 'time' in ds[v].dims]
            if len(time_dim) != 1:
                raise ValueError(f"❌ Unable to determine unique timeseries variable in {path.name} ❌")
            var = time_dim[0] # Extract the variable
            df = ds[[var]].to_dataframe()
        else:
            raise ValueError(f"❌ Unsupported timeseries data extension for {label}: {suffix} ❌")

        # Store the dataframes 
        if label == 'obs_ts':
            obs_ts_df = df
        else:
            sim_ts_df = df

    # Convert the dataframes into series
    obs_ts = obs_ts_df.squeeze()
    sim_ts = sim_ts_df.squeeze()

    # Ensure that the series have a datetime index and it is valid
    obs_ts = ensure_datetime_index(obs_ts, "Observed timeseries")
    sim_ts = ensure_datetime_index(sim_ts, "Simulated timeseries")

    if not isinstance(obs_ts.index, pd.DatetimeIndex) or not isinstance(sim_ts.index, pd.DatetimeIndex):
        raise TypeError("❌ Timeseries must have a DatetimeIndex after processing ❌")

    # Align the timeseries
    common_index = obs_ts.index.intersection(sim_ts.index)
    if len(common_index) == 0:
        raise ValueError("❌ No overlapping dates between observed and simulated timeseries ❌")
    obs_aligned = obs_ts.loc[common_index]
    sim_aligned = sim_ts.loc[common_index]
    
    # Ask for a couple of info, used for plots and reporto
    # If both are provided, just optionally print and return
    if variable is not None and unit is not None:
        vprint("Variable and unit provided via CLI:", verbose=verbose)
        vprint(f"Variable: {variable}", verbose=verbose)
        vprint(f"Unit: {unit}", verbose=verbose)
        vprint("\n" + border + "\n", verbose=verbose)
    else:
        vprint("""What kind of variable/units are going to be displayed on the axis/title?
(NOTE: The units provided will be converted in latex format)""", verbose=verbose)
        if variable is None:
            variable = input("What kind of variable are we using? ").strip()
        if unit is None:
            unit = input("What is the units of measurements? ").strip()
        vprint("\n" + border + "\n", verbose=verbose)

    # ===== SETUP OF THE DICTIONARIES =====
    Basin_Average_Timeseries = {
        'observed': obs_aligned,
        'model': sim_aligned,
    }

    # Get the year range to split the dataframe
    Ybeg, Yend = common_index[0].year, common_index[-1].year
    years = list(range(Ybeg, Yend + 1))
    
    vprint("Creating the yearly datasets...", verbose=verbose)
    Basin_Average_Yearly = {}
    for key in Basin_Average_Timeseries:
        Basin_Average_Yearly[key] = split_to_yearly(Basin_Average_Timeseries[key], years)
    vprint("Yearly dataset created!")

    vprint("Creating the monthly datasets...", verbose=verbose)
    Basin_Average_Monthly = {}
    for key in Basin_Average_Yearly:
        Basin_Average_Monthly[key] = split_to_monthly(Basin_Average_Yearly[key])
    vprint("Monthly dataset created!", verbose=verbose)
    vprint("\n" + border + "\n", verbose=verbose)

    # ===== SETUP OF THE OUTPUT FOLDERS =====
    if output_dir is None:
        if isinstance(data_folder, str):
            default_outdir = os.path.join(str(data_folder), "REPORT")
        else:
            default_outdir = os.path.join(str(obs_spatial_path.parent), "REPORT")

        use_default = input(
            f"Save output in default REPORT folder?\n  {default_outdir}\nEnter Y for yes, N to specify another path (Y/n): "
            ).strip().lower()

        if use_default in ('', 'y', 'yes'):
            outdir = default_outdir
        else:
            outdir = input("Enter base output directory path: ").strip()
    else:
        outdir = Path(output_dir)

    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    output_path = os.path.join(outdir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    vprint(f"Output folder created at: {output_path}", verbose=verbose)
    vprint("\n" + border + "\n", verbose=verbose)
    
    plots_path = Path(output_path) / "plots"
    plots_path.mkdir(exist_ok=True)
    
    dataframe_path = Path(output_path) / "dataframes"
    dataframe_path.mkdir(exist_ok=True)
    
    pdf_path = Path(output_path) / f"Report_{timestamp}.pdf"
    if 'variable' in locals():
        pdf_path = Path(output_path) / f"Report_{variable}_{timestamp}.pdf"
    
    # ===== INITIALIZE THE PDF =====
    if generate_pdf:
        vprint("Generating the PDF report...", verbose=verbose)
    
        pdf = PDFReportBuilder(str(pdf_path))
        pdf.build_title_page()
        pdf.build_toc()
        
    else:
        vprint("PDF generation skipped, saving only the plots and the dataframes", verbose=verbose)
    
    # ===== BEGIN THE PLOTTING =====
    
    BIAS = Basin_Average_Timeseries['model'] - Basin_Average_Timeseries['observed']
    
    # ===== TIMESERIES =====
    vprint("Plotting the time-series...", verbose=verbose)
    timeseries(
        Basin_Average_Timeseries,
        BIAS,
        variable=variable,
        unit=unit,
        BA=False,
        output_path=plots_path
    )
    vprint("\033[92m✅ Time-series plotted succesfully!\033[0m", verbose=verbose)
    
    # Add the page
    ts_img = Path(plots_path) / f'{variable}_timeseries.png'
    if generate_pdf:
        add_rotated_image_page(pdf, ts_img, section_title="Time Series Plot")
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    # ===== SCATTERPLOT =====
    vprint("Plotting the seasonal data as scatterplots...", verbose=verbose)
    seasonal_scatter_plot(Basin_Average_Timeseries, variable=variable, unit=unit, BA=False, output_path=plots_path)
    vprint("\033[92m✅ Seasonal scatterplots plotted succesfully!\033[0m", verbose=verbose)
    
    seasonal_img = Path(plots_path) / f"{variable}_all_seasons_scatterplot.png"
    DJF_img = Path(plots_path) / f"{variable}_DJF_scatterplot.png"
    MAM_img = Path(plots_path) / f"{variable}_MAM_scatterplot.png"
    JJA_img = Path(plots_path) / f"{variable}_JJA_scatterplot.png"
    SON_img = Path(plots_path) / f"{variable}_SON_scatterplot.png"
    
    if generate_pdf:
        # Add the page
        add_seasonal_scatter_page(
            pdf,
            seasonal_img,  # main plot path
            [DJF_img, MAM_img, JJA_img, SON_img],  # list of 4 small seasonal plots
            "Seasonal Scatterplots"
            )
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    # ===== VIOLIN AND BOX =====
    vprint("Plotting the whisker-box plots...", verbose=verbose)
    whiskerbox(Basin_Average_Monthly, variable=variable, unit=unit, output_path=plots_path)
    vprint("\033[92m✅ Whisker-box plotted succesfully!\033[0m", verbose=verbose)
    
    whisker_img = Path(plots_path) / f'{variable}_boxplot.png'
    
    vprint("\n" + border + "\n", verbose=verbose)

    vprint("Plotting the violinplots...", verbose=verbose)
    violinplot(Basin_Average_Monthly, variable=variable, unit=unit, output_path=plots_path)
    vprint("\033[92m✅ Violinplots plotted succesfully!\033[0m", verbose=verbose)
    
    violin_img = Path(plots_path) / f'{variable}_violinplot.png'
    
    if generate_pdf:
        # Add the page
        add_multiple_rotated_images_grid(
            pdf,
            img_paths=[whisker_img, violin_img],
            cols=2,
            section_title="Box and Violin Plots"
            )
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    # ===== TAYLOR DIAGRAMS =====
    vprint("Plotting the SST Taylor diagram for yearly data...", verbose=verbose)
    comprehensive_taylor_diagram(Basin_Average_Yearly, output_path=plots_path, variable=variable, unit=unit)
    vprint("\033[92m✅ Yearly data Taylor diagram has been plotted!\033[0m", verbose=verbose)
    
    vprint("-"*45, verbose=verbose)
    
    vprint("Plotting the monthly data diagrams...", verbose=verbose)
    monthly_taylor_diagram(Basin_Average_Monthly, output_path=plots_path, variable=variable, unit=unit)
    vprint("\033[92m✅ Monthly Taylor diagrams have been plotted!\033[0m", verbose=verbose)
    
    yearly_taylor_img = Path(plots_path) / 'Taylor_diagram_summary.png'
    monthly_taylor_img = Path(plots_path) / "Unified_Taylor_Diagram.png"

    if generate_pdf:
        # Add the page
        add_multiple_images_grid(
            pdf,
            img_paths=[yearly_taylor_img, monthly_taylor_img],
            section_title="Taylor Diagrams",
            columns=1,
            rows=2,
            max_width=4.85 * inch,     
            spacing=0     
            )
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    # ===== TARGET PLOTS =====
    vprint("Plotting the Target plot for the yearly data...", verbose=verbose)
    comprehensive_target_diagram(Basin_Average_Yearly, output_path=plots_path, variable=variable, unit=unit)
    vprint("\033[92m✅ Yearly data Target plot has been plotted!\033[0m", verbose=verbose)

    vprint("-"*45, verbose=verbose)
    
    vprint("Plotting the monthly data plots...", verbose=verbose)
    target_diagram_by_month(Basin_Average_Monthly, output_path=plots_path, variable=variable, unit=unit)
    vprint("\033[92m✅ All of the Target plots has been plotted!\033[0m", verbose=verbose)
    
    yearly_target_img = Path(plots_path) / "Unified_Target_Diagram.png"
    monthly_target_img = Path(plots_path) / "Monthly_Target_Diagram.png"

    if generate_pdf:
        # Add to page
        add_multiple_images_grid(
            pdf,
            img_paths=[yearly_target_img, monthly_target_img],
            section_title="Target Plots",
            columns=1,
            rows=2,
            max_width=4.55 * inch,
            spacing=0
            )
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    # ===== EFFICINECY METRICS =====
    vprint("Computing the Efficiency Metrics...", verbose=verbose)
    vprint("-" * 45, verbose=verbose)

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
        vprint(f"\033[93mComputing {name}...\033[0m", verbose=verbose)

        if name in ['ln NSE', 'E_rel', 'd_rel']:
            mask = ~np.isnan(Basin_Average_Timeseries['observed']) & ~np.isnan(Basin_Average_Timeseries['model'])

            if name == 'ln NSE':
                mask &= (Basin_Average_Timeseries['observed'] > 0) & (Basin_Average_Timeseries['model'] > 0)
            if name in ['E_rel', 'd_rel']:
                mask &= Basin_Average_Timeseries['observed'] != 0

            x = Basin_Average_Timeseries['observed'][mask]
            y = Basin_Average_Timeseries['model'][mask]
        else:
            x = Basin_Average_Timeseries['observed']
            y = Basin_Average_Timeseries['model']

        total_val = func(x, y)
        monthly_vals = monthly_func(Basin_Average_Monthly)

        # Store in DataFrame
        efficiency_df.loc[name, 'Total'] = total_val
        efficiency_df.loc[name, months] = monthly_vals

        # Print values
        vprint(f"{name} (Total) = {total_val:.4f}", verbose=verbose)
        for month, val in zip(months, monthly_vals):
            vprint(f"{month}: {name} = {val:.4f}", verbose=verbose)

        vprint("-" * 45, verbose=verbose)

    vprint("\033[92m✅ All of the metrics have been computed!\033[0m", verbose=verbose)
    
    vprint("Saving the dataframe...")
    # Save the dataframe
    save_variable_to_json(efficiency_df, Path(dataframe_path) / "efficiency_df.json")
    vprint("Dataframe saved!", verbose=verbose)
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    vprint("Plotting the efficiency metrics results...", verbose=verbose)

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

    # Plotting all metrics
    for metric_key, title in plot_titles.items():
        total_value = efficiency_df.loc[metric_key, 'Total']
        monthly_values = efficiency_df.loc[metric_key, efficiency_df.columns[1:]].values.astype(float)
    
        efficiency_plot(total_value, monthly_values, 
                        title=f'{title}', 
                        y_label=f'{metric_key}', 
                        metric_name=metric_key,
                        output_path=plots_path)
    
        # Remove any parentheses and their contents for the print message
        clean_title = re.sub(r'\s*\([^)]*\)', '', title)
        vprint(f"\033[92m✅ {clean_title} plotted!\033[0m", verbose=verbose)
        
    vprint("\033[92m✅ All efficiency metric plots have been successfully created!\033[0m", verbose=verbose)
    vprint("\n" + border + "\n", verbose=verbose)
    
    if generate_pdf:
        # Add page
        add_efficiency_pages(pdf, efficiency_df, plot_titles, plots_path)

    # ===== ERROR COMPONENTS + TIMESERIES =====
    vprint("Computing the error components timeseries...", verbose=verbose)
    error_comp_stats_df = compute_error_timeseries(sim_spatial,
                                                   obs_spatial,
                                                   ocean_mask)
    vprint("Error components timeseries obtaines!", verbose=verbose)

    vprint("Computing the cloud cover...", verbose=verbose)
    _, cloud_cover = compute_coverage_stats(obs_spatial.values, Mmask)
    cloud_cover = pd.Series(cloud_cover)  # Convert numpy array to Series
    cloud_cover = ensure_datetime_index(cloud_cover, "Cloud Cover")  # Ensure datetime index   
    vprint("Cloud cover timeseries obtained!", verbose=verbose)
    
    vprint("Plotting the results...", verbose=verbose)
    # Use cloud cover index for the error components df
    correct_index = cloud_cover.index
    error_comp_stats_df.index = correct_index
    # Make the plot
    error_components_timeseries(error_comp_stats_df, plots_path, 
                                cloud_cover, variable=variable)
    err_ts_img = Path(plots_path) / 'Error_Decomposition_Timeseries_.png'  
    if generate_pdf:
        # Add the page
        add_rotated_image_page(pdf, err_ts_img, 
                               section_title="Error Components Timeseries",
                               custom_offset_y=200)
    vprint("Results plotted!", verbose=verbose)
    
    vprint('-'*45, verbose=verbose)
    
    # Compute the correlations
    vprint("Computing the correlation statistics", verbose=verbose)
    cloud_cover_30d = cloud_cover.rolling(window=30, center=True).mean()
    
    correlations = {
        'raw_cloud_cover': {},
        'smoothed_cloud_cover_30d': {}
        }

    for metric in error_comp_stats_df.columns:
        corr_raw = corr_no_nan(error_comp_stats_df[metric], cloud_cover)
        corr_smooth = corr_no_nan(error_comp_stats_df[metric], cloud_cover_30d)
        correlations['raw_cloud_cover'][metric] = corr_raw
        correlations['smoothed_cloud_cover_30d'][metric] = corr_smooth

    vprint("The correlation between cloud cover and error components are...", verbose=verbose)
    vprint("Correlation between stats metrics and RAW cloud cover:", verbose=verbose)
    for metric, corr_val in correlations['raw_cloud_cover'].items():
        vprint(f"  {metric}: {corr_val:.4f}", verbose=verbose)
    
    vprint("\nCorrelation between stats metrics and SMOOTHED (30d) cloud cover:", verbose=verbose)
    for metric, corr_val in correlations['smoothed_cloud_cover_30d'].items():
        vprint(f"  {metric}: {corr_val:.4f}", verbose=verbose)
        
    # Prepare tables for PDF
    correlation_tables = {
        "Raw Cloud Cover vs. Error Components": {
            metric: corr_val for metric, corr_val in correlations['raw_cloud_cover'].items()
            },
        "Smoothed (30d) Cloud Cover vs. Error Components": {
            metric: corr_val for metric, corr_val in correlations['smoothed_cloud_cover_30d'].items()
            }
        }

    if generate_pdf:
        # Add a page with both tables to the PDF    
        add_tables_page(
            pdf,
            tables_dict=correlation_tables,
            section_title="Correlation Between Cloud Cover and Error Components",
            columns=1,
            rows=2
            )
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    # ===== SPECTRAL =====
    vprint("Making the spectral analysis...", verbose=verbose)
    
    # Add the cloud cover to the dataframe
    combined_df = error_comp_stats_df.copy()
    combined_df['cloud_cover'] = cloud_cover
    combined_df['cloud_cover_30d'] = cloud_cover_30d

    combined_df = combined_df.dropna()
    
    vprint("Saving the dataframe...", verbose=verbose)
    # Save the dataframe
    save_variable_to_json(combined_df, Path(dataframe_path) / "error_ts_df.json")
    vprint("Dataframe saved!", verbose=verbose)

    # Separate cleaned data
    error_comp_clean = combined_df[error_comp_stats_df.columns]
    cloud_cover_clean = combined_df['cloud_cover']
    cloud_cover_30d_clean = combined_df['cloud_cover_30d']

    # Make a quick detrend
    error_comp_detrended = error_comp_clean - error_comp_clean.mean()
    cloud_cover_detrended = cloud_cover_clean - cloud_cover_clean.mean()
    cloud_cover_30d_detrended = cloud_cover_30d_clean - cloud_cover_30d_clean.mean()
    vprint("Data cleaned and deterended, proceeding to plot...", verbose=verbose)
    
    vprint("Computing the fft components...", verbose=verbose)
    # Compute the fft components for all of the data in the dataframe
    freqs, fft_components = compute_fft({
        k: v.values for k, v in error_comp_detrended.items()
        })
    
    # For cloud cover series
    _, fft_cloud = compute_fft(cloud_cover_detrended.values)
    
    # For 30-day cloud cover
    _, fft_cloud_30d = compute_fft(cloud_cover_30d_detrended.values)
    
    vprint("Plotting the PSD...", verbose=verbose)
    plot_type='PSD'
    plot_spectral(plot_type=plot_type,
                  freqs=freqs,
                  fft_components=fft_components,
                  output_path=plots_path,
                  variable_name=variable)
    vprint("PSD plotted!", verbose=verbose)
    
    psd_img = Path(plots_path) / f"Spectral_Plot_{plot_type}_{variable}.png"

    vprint("Plotting the CSD...", verbose=verbose)
    plot_type='CSD'
    plot_spectral(
        plot_type=plot_type,
        error_comp=error_comp_clean,
        output_path=plots_path,
        variable_name=variable,
        cloud_covers=[
            (cloud_cover_clean, 'cloud_cover'),
            (cloud_cover_30d_clean, 'cloud_cover_30d')
            ])
    vprint("CSD plotted!", verbose=verbose)
    
    csd_img = Path(plots_path) / f"Spectral_Plot_{plot_type}_{variable}.png"
    
    if generate_pdf:
        # Add the plot's pages
        add_multiple_images_grid(
            pdf,
            img_paths=[psd_img, csd_img],
            section_title="Spectral Analysis",
            columns=1,
            rows=2,
            max_width=8 * inch,
            spacing=0
            )
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    vprint("Beginning with the spatial analysis...", verbose=verbose)
    
    # ===== RESAMPLING =====
    vprint("""The resampling can be done with either:
1. The built-in resampler (might take a while)
2. The CDO (requires previous installation)""", verbose=verbose)
    method = input("Enter the method of your choice for the resampling (1 or 2): ")
    
    if method == '1':
        
        from Hydrological_model_validator.Processing.time_utils import resample_and_compute
        
        # Chunk to better handle the data
        obs_spatial_chunked = obs_spatial.chunk({'time': 100})
        sim_spatial_chunked = sim_spatial.chunk({'time': 100})
        
        vprint("Resampling...", verbose=verbose)
        obs_monthly, sim_monthly = resample_and_compute(obs_spatial_chunked, sim_spatial_chunked)
        vprint("Data resampled!", verbose=verbose)
        
        # Mask
        obs_monthly = obs_monthly.where(ocean_mask)
        sim_monthly = sim_monthly.where(ocean_mask)
        
        vprint("Finishing setting up the data...", verbose=verbose)
        # Add month and year dimensions
        sim_monthly = sim_monthly.assign_coords(
            month=sim_monthly.time.dt.month,
            year=sim_monthly.time.dt.year,
            )
        obs_monthly = obs_monthly.assign_coords(
            month=obs_monthly.time.dt.month,
            year=obs_monthly.time.dt.year,
            )
        vprint("Data ready!", verbose=verbose)
        
        vprint("\n" + border + "\n", verbose=verbose)
        
    elif method == '2':
        
        import subprocess
        
        # Save CDO-ready datasets if not already saved 
        obs_output_path = Path(data_folder) / f"{obs_spatial_path.stem}_CDO.nc"
        sim_output_path = Path(data_folder) / f"{sim_spatial_path.stem}_CDO.nc"

        vprint("Checking if CDO-ready datasets exist...", verbose=verbose)
        if not obs_output_path.exists():
            vprint("Saving observed spatial dataset...", verbose=verbose)
            obs_spatial.to_netcdf(obs_output_path, mode="w")
        else:
            vprint("Observed spatial dataset already exists, skipping save.", verbose=verbose)

        if not sim_output_path.exists():
            vprint("Saving simulated spatial dataset...")
            sim_spatial.to_netcdf(sim_output_path, mode="w")
        else:
            vprint("Simulated spatial dataset already exists, skipping save.", verbose=verbose)

        vprint("CDO-ready datasets prepared!", verbose=verbose)

        #  Run CDO on observed spatial dataset 
        vprint("Running CDO on observed data...", verbose=verbose)
        obs_spatial_path = obs_output_path  # update path
        obs_monthly_path = obs_spatial_path.with_name("obs_spatial_monthly.nc")
        
        if not obs_monthly_path.exists():
            vprint("Running CDO on observed data...", verbose=verbose)
            subprocess.run(["cdo", "-v", "monmean", str(obs_spatial_path), str(obs_monthly_path)], check=True)
            vprint("The observed data has been resampled!", verbose=verbose)
        else:
            vprint("Monthly observed file already exists, skipping CDO.", verbose=verbose)
        
        #  Run CDO on simulated spatial dataset 
        vprint("Onto the satellite data...", verbose=verbose)
        sim_spatial_path = sim_output_path  # update path
        sim_monthly_path = sim_spatial_path.with_name("sim_spatial_monthly.nc")
       
        if not sim_monthly_path.exists():
            vprint("Running CDO on satellite data...", verbose=verbose)
            subprocess.run(["cdo", "-v", "monmean", str(sim_spatial_path), str(sim_monthly_path)], check=True)
            vprint("The satellite data has been resampled!", verbose=verbose)
        else:
            vprint("Monthly satellite file already exists, skipping CDO.", verbose=verbose)
            
        # Open the new datasets
        obs_monthly = xr.open_dataset(Path(data_folder) / "obs_spatial_monthly.nc")
        sim_monthly = xr.open_dataset(Path(data_folder) / "sim_spatial_monthly.nc")
        
        vprint("Finishing setting up the data...", verbose=verbose)
        # Extract the data
        obs_monthly_da = select_3d_variable(obs_monthly, "obs_monthly")
        sim_monthly_da = select_3d_variable(sim_monthly, "sim_monthly")

        # Drop the bounds added by the CDO
        obs_monthly = obs_monthly_da.drop_vars("time_bnds", errors="ignore")
        sim_monthly = sim_monthly_da.drop_vars("time_bnds", errors="ignore")        
              
        # Mask using an expanded mask, the regular one does not work for the 2D plotting
        mask_da = xr.DataArray(Mfsm)
        mask_expanded = mask_da.expand_dims(time=sim_monthly.time)
        obs_monthly = obs_monthly.where(mask_expanded)
        sim_monthly = sim_monthly.where(mask_expanded)
        
        vprint("Data ready!", verbose=verbose)
        
        vprint("\n" + border + "\n", verbose=verbose)
        
    else:
        vprint("Invalid choice", verbose=verbose)
        exit()
        
    # ===== SPATIAL ERROR / EFFICIENCY =====
    vprint("Computing the metrics from the raw data...", verbose=verbose)
    mb_raw, sde_raw, cc_raw, rm_raw, ro_raw, urmse_raw = compute_spatial_efficiency(sim_monthly, obs_monthly, time_group="month")

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
    vprint("Metrics computed!", verbose=verbose)

    # Use before saving:
    metrics_df_raw = convert_dataarrays_in_df(metrics_df_raw)
    
    vprint("Saving the dataframe...", verbose=verbose)
    save_variable_to_json(metrics_df_raw, Path(dataframe_path) / "metrics_df_raw_month.json")
    vprint("Dataframe saved!", verbose=verbose)
    
    vprint('-'*45, verbose=verbose)

    # Detrend the monthly data
    vprint("Detrending the data...", verbose=verbose)
    sim_monthly_detrend = detrend_dim(sim_monthly, dim='time', mask=mask_expanded)
    obs_monthly_detrend = detrend_dim(obs_monthly, dim='time', mask=mask_expanded)
    vprint("Data detrended!", verbose=verbose)

    # Repeat the computations for the detrended data
    vprint("Computing the metrics from the detrended data...", verbose=verbose)
    mb_detr, sde_detr, cc_detr, rm_detr, ro_detr, urmse_detr = compute_spatial_efficiency(sim_monthly_detrend, obs_monthly_detrend, time_group="month")

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
    vprint("Detrended data metrcis computed!", verbose=verbose)
    vprint('-'*45, verbose=verbose)
    
    vprint("Saving the dataframe...", verbose=verbose)
    metrics_df_detr = convert_dataarrays_in_df(metrics_df_detr)
    save_variable_to_json(metrics_df_detr, Path(dataframe_path) / "metrics_df_detrended_month.json")
    vprint("Dataframe saved!", verbose=verbose)
    
    vprint("\n" + border + "\n", verbose=verbose)
    
    vprint("The geolocalization specifics are needed to plot the data, please provide them:", verbose=verbose)
    # Known values from the dataset, need to be changed if the area of analysis is changed
    import ast

    grid_shape_str = input("Provide the grid shape (lat, lon): ")
    grid_shape = ast.literal_eval(grid_shape_str)

    epsilon_str = input("Provide a correction factor if needed: ")
    epsilon = float(epsilon_str)

    x_start_str = input("Provide the starting longitude coordinate: ")
    x_start = float(x_start_str)

    x_step_str = input("Provide the x-step discretization: ")
    x_step = float(x_step_str)

    y_start_str = input("Provide the starting latitude coordinate: ")
    y_start = float(y_start_str)

    y_step_str = input("Provide the y-step discretization: ")
    y_step = float(y_step_str)

    geo_coords = compute_geolocalized_coords(grid_shape, epsilon, x_start, x_step, y_start, y_step)
    vprint("\033[92m✅ Geolocalization complete! \033[0m", verbose=verbose)
    vprint("\n" + border + "\n", verbose=verbose)

    # Plot the results
    vprint("Plotting the results...", verbose=verbose)
    
    # Variable used to store all of the paths to append the pages
    plot_paths = {}

    # Used to dentify the saved plots
    period = "Monthly"  

    # Mapping of titles to short keys for your variables
    key_map = {
        "Mean Bias": "mb",
        "Standard Deviation Error": "sde",
        "Cross Correlation": "cc",
        "Model Std Dev": "rm",
        "Satellite Std Dev": "ro",
        "Unbiased RMSE": "urmse",
        }

    # All of the possibilities for the plots are collected in a dictionary
    plots_info = [
        {
            "raw": mb_raw, "detr": mb_detr, "title": "Mean Bias", "unit": "°C",
            "cmap": "RdBu_r", "vmin": -2, "vmax": 2, "suffix": "(Model-Satellite)",
        },
        {
            "raw": sde_raw, "detr": sde_detr, "title": "Standard Deviation Error", "unit": "°C",
            "cmap": "viridis", "vmin": 0, "vmax": 3, "suffix": "(Model-Satellite)",
        },
        {
            "raw": cc_raw, "detr": cc_detr, "title": "Cross Correlation",
            "cmap": "OrangeGreen", "vmin": -1, "vmax": 1, "suffix": "(Model-Satellite)",
        },
        {
            "raw": rm_raw, "detr": rm_detr, "title": "Model Std Dev", "unit": "°C",
            "cmap": "plasma", "vmin": 0, "vmax": 8, "suffix": "(Model)",
        },
        {
            "raw": ro_raw, "detr": ro_detr, "title": "Satellite Std Dev", "unit": "°C",
            "cmap": "plasma", "vmin": 0, "vmax": 8, "suffix": "(Satellite)",
        },
        {
            "raw": urmse_raw, "detr": urmse_detr, "title": "Unbiased RMSE", "unit": "°C",
            "cmap": "inferno", "vmin": 0, "vmax": 3, "suffix": "(Model-Satellite)",
        },
        ]

    # Begin the loop to plot the data
    for info in plots_info:
        key_prefix = key_map.get(info["title"], info["title"].lower().replace(" ", "_"))
        
        # Starting with raw data
        vprint(f"Plotting the {info['title'].lower()}...", verbose=verbose)
        plot_spatial_efficiency(info["raw"], geo_coords, plots_path,
                                info["title"],
                                unit=info.get("unit"),
                                cmap=info["cmap"],
                                vmin=info["vmin"],
                                vmax=info["vmax"],
                                suffix=info.get("suffix"))
        
        suffix = info.get("suffix", "")
        safe_title = info["title"].replace("/", "_").replace("\\", "_")
        raw_filename = f"Monthly {safe_title} (Raw) {suffix}".strip() + ".png"  # Hardcoded monthly/yearly
        raw_path = Path(plots_path) / raw_filename

        # Append the path to be retrieved later
        plot_paths[f"{key_prefix}_img_{period}"] = raw_path
        
        # Plot the detrended results
        plot_spatial_efficiency(info["detr"], geo_coords, plots_path,
                                info["title"],
                                unit=info.get("unit"),
                                cmap=info["cmap"],
                                vmin=info["vmin"],
                                vmax=info["vmax"],
                                detrended=True,
                                suffix=info.get("suffix"))
        
        detrended_filename = f"Monthly {safe_title} (Detrended) {suffix}".strip() + ".png"
        detrended_path = Path(plots_path) / detrended_filename

        # Append the path to be retrieved later
        plot_paths[f"{key_prefix}_img_detr_{period}"] = detrended_path
        
        vprint(f"\033[92m✅ {info['title']} plotted! \033[0m", verbose=verbose)
    
    vprint("Monthly data done!", verbose=verbose)
    
    vprint("\n" + border + "\n", verbose=verbose)

    # Repeat the computations for the yearly data
    vprint("Moving onto the yearly data...", verbose=verbose)
    vprint("Computing the metrics from the raw data...", verbose=verbose)
    mb_raw, sde_raw, cc_raw, rm_raw, ro_raw, urmse_raw = compute_spatial_efficiency(sim_monthly_detrend, obs_monthly_detrend, time_group="year")

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
    vprint("Metrics computed!", verbose=verbose)
    
    vprint("Saving the dataframe...", verbose=verbose)
    metrics_df_raw = convert_dataarrays_in_df(metrics_df_raw)
    save_variable_to_json(metrics_df_raw, Path(dataframe_path) / "metrics_df_raw_year.json")
    vprint("Dataframe saved!", verbose=verbose)
    
    vprint('-'*45, verbose=verbose)

    # Detrending the data
    vprint("Detrending the data...", verbose=verbose)
    sim_monthly_detrend = detrend_dim(sim_monthly, dim='time', mask=mask_expanded)
    obs_monthly_detrend = detrend_dim(obs_monthly, dim='time', mask=mask_expanded)
    vprint("Data detrended!", verbose=verbose)

    # Remake the computations for the detrended data
    vprint("Computing the metrics from the detrended data...", verbose=verbose)
    mb_detr, sde_detr, cc_detr, rm_detr, ro_detr, urmse_detr = compute_spatial_efficiency(sim_monthly_detrend, obs_monthly_detrend, time_group="year")

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
    vprint("Detrended data metrics computed!", verbose=verbose)
    
    vprint("Saving the dataframe...", verbose=verbose)
    metrics_df_detr = convert_dataarrays_in_df(metrics_df_detr)
    save_variable_to_json(metrics_df_detr, Path(dataframe_path) / "metrics_df_detrended_year.json")
    vprint("Dataframe saved!", verbose=verbose)
    
    vprint('-'*45, verbose=verbose)

    # Start with the plotting
    vprint("Plotting the results...", verbose=verbose)
    
    # New period definition for the Yearly data
    period = "Yearly"

    plots_info = [
        {
            "raw": mb_raw, "detr": mb_detr, "title": "Mean Bias", "unit": "°C",
            "cmap": "RdBu_r", "vmin": -2, "vmax": 2, "suffix": "(Model-Satellite)",
        },
        {
            "raw": sde_raw, "detr": sde_detr, "title": "Standard Deviation Error", "unit": "°C",
            "cmap": "viridis", "vmin": 0, "vmax": 3, "suffix": "(Model-Satellite)",
        },
        {
            "raw": cc_raw, "detr": cc_detr, "title": "Cross Correlation",
            "cmap": "OrangeGreen", "vmin": -1, "vmax": 1, "suffix": "(Model-Satellite)",
        },
        {
            "raw": rm_raw, "detr": rm_detr, "title": "Model Std Dev", "unit": "°C",
            "cmap": "plasma", "vmin": 0, "vmax": 8, "suffix": "(Model)",
        },
        {
            "raw": ro_raw, "detr": ro_detr, "title": "Satellite Std Dev", "unit": "°C",
            "cmap": "plasma", "vmin": 0, "vmax": 8, "suffix": "(Satellite)",
        },
        {
            "raw": urmse_raw, "detr": urmse_detr, "title": "Unbiased RMSE", "unit": "°C",
            "cmap": "inferno", "vmin": 0, "vmax": 3, "suffix": "(Model-Satellite)",
        },
        ]

    # Begin the loop to plot and append the paths
    for info in plots_info:
        key_prefix = key_map.get(info["title"], info["title"].lower().replace(" ", "_"))
        
        # Plotting of the raw data
        vprint(f"Plotting the {info['title'].lower()}...", verbose=verbose)
        plot_spatial_efficiency(
            info["raw"], geo_coords, plots_path,
            info["title"],
            unit=info.get("unit"),
            cmap=info["cmap"],
            vmin=info["vmin"],
            vmax=info["vmax"],
            suffix=info.get("suffix")
            )
        
        suffix = info.get("suffix", "")
        safe_title = info["title"].replace("/", "_").replace("\\", "_")
        raw_filename = f"Yearly {safe_title} (Raw) {suffix}".strip() + ".png"  
        raw_path = Path(plots_path) / raw_filename

        # Save path and append it
        plot_paths[f"{key_prefix}_img_{period}"] = raw_path
        
        # Plots and save the detrended results
        plot_spatial_efficiency(
            info["detr"], geo_coords, plots_path,
            info["title"],
            unit=info.get("unit"),
            cmap=info["cmap"],
            vmin=info["vmin"],
            vmax=info["vmax"],
            detrended=True,
            suffix=info.get("suffix")
            )
        
        detrended_filename = f"Yearly {safe_title} (Detrended) {suffix}".strip() + ".png"
        detrended_path = Path(plots_path) / detrended_filename
        
        # Save the paths and append them
        plot_paths[f"{key_prefix}_img_detr_{period}"] = detrended_path
        
        vprint(f"\033[92m✅ {info['title']} plotted!\033[0m", verbose=verbose) 
    
    vprint("Yearly data done!", verbose=verbose)
    
    reverse_key_map = {v: k for k, v in key_map.items()}  # Reverse lookup to retrieve the paths

    if generate_pdf:
        # Begin the loop to append the pages
        for key, img_path in plot_paths.items():
        
            # Path keys to retrieve the correct file
            parts = key.split('_')  # e.g. ['mb', 'img', 'detr', 'Monthly']

            plot_type = parts[0].lower()
            detrended = 'detr' in parts
            time_type = parts[-1].capitalize()  # e.g. 'Monthly' or 'Yearly'
            
            # Get full title
            title_base = reverse_key_map.get(plot_type, plot_type.upper())
            
            # Use the keys to build the section title
            section_title = f"{title_base} - {'Detrended' if detrended else 'Raw'} - {time_type}"
            
            # Append the page
            add_plot_to_pdf(pdf, img_path, section_title, width=6 * inch)
            
    vprint("\n" + border + "\n", verbose=verbose)  
    
    # ==== SAVE =====
    if generate_pdf:
        vprint("Saving the report...", verbose=verbose)
        pdf.save()
        vprint(f"\033[92m✅ PDF report saved at {output_path}\033[0m", verbose=verbose)
    
    if open_report and generate_pdf:
        import subprocess
        import platform
        
        if platform.system() == "Darwin":   # For macOS
            subprocess.run(["open", str(pdf_path)])
        elif platform.system() == "Windows":    # Windows
            os.startfile(str(pdf_path))
        else:                                   # Linux variants
            subprocess.run(["xdg-open", str(pdf_path)])
    
    else:
        vprint(f"\033[92m✅ The data has been saved at {output_path}\033[0m", verbose=verbose)