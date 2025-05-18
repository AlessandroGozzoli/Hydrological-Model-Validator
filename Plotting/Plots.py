###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Ignoring a depracation warning to ensure a better console run
import warnings
# Suppress FutureWarning from Seaborn regarding 'use_inf_as_na'
warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is deprecated")

# General Libraries
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import calendar
import itertools

# Plotting Libraries
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns

# Cartopy (for map projections and geospatial features)
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Custom Imports
from Corollary import format_unit, extract_mod_sat_keys

from Auxilliary import (get_min_max_for_identity_line, 
                        get_variable_label_unit,
                        fit_huber,
                        fit_lowess,
                        get_season_mask)

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################

def timeseries_basin_average(output_path, daily_means_dict, variable_name, BIAS, BA=False):
    """
    Plots the daily mean values for each dataset in the dictionary along with BIAS.

    Parameters:
    - output_path: Path to save the output plot.
    - daily_means_dict: Dictionary where the keys are dataset names, and the values are 1D arrays 
      containing the daily mean values.
    - variable_name: Name of the variable being plotted (e.g., 'SST').
    - BIAS: 1D array of bias values (model - satellite) over time.
    - BA: Boolean indicating if it's Basin Average. If True, title will include "(Basin Average)".
    """

    sns.set(style="whitegrid")
    sns.set_style("ticks")
    color_palette = itertools.cycle(['#BF636B', '#5976A2', '#70A494', '#D98B5F', '#D3A4BD', '#7294D4'])

    title = f'Daily Mean Values for {variable_name} Datasets'
    if BA:
        title += ' (Basin Average)'

    mod_key, sat_key = extract_mod_sat_keys(daily_means_dict)
    label_lookup = {
        mod_key: "Model Output",
        sat_key: "Satellite Observations"
        }
    
    variable, unit = get_variable_label_unit(variable_name)

    fig = plt.figure(figsize=(20, 10), dpi=300)
    gs = GridSpec(2, 1, height_ratios=[8, 4])

    # First subplot: daily means
    ax1 = fig.add_subplot(gs[0])
    for key, daily_mean in daily_means_dict.items():
        label = label_lookup.get(key, key)
        color = next(color_palette)

        # Ensure data is a Series
        if not isinstance(daily_mean, pd.Series):
            daily_mean = pd.Series(daily_mean)

        sns.lineplot(data=daily_mean, label=label, ax=ax1, lw=2, color=color)

    ax1.set_title(title, fontsize=20, fontweight='bold')
    ax1.set_ylabel(f'{variable} {unit}', fontsize=14)
    ax1.tick_params(width=2)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, linestyle='--')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Second subplot: BIAS
    ax2 = fig.add_subplot(gs[1])
    if not isinstance(BIAS, pd.Series):
        BIAS = pd.Series(BIAS)
    sns.lineplot(data=BIAS, color='k', ax=ax2)
    ax2.set_title(f'BIAS ({variable_name})', fontsize=18, fontweight='bold')
    ax2.set_ylabel(f'BIAS {unit}', fontsize=14)
    ax2.tick_params(width=2)
    ax2.grid(True, linestyle='--')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    plt.tight_layout()
    output_path = Path(output_path)
    filename = f'{variable_name}_timeseries.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.close()
    
def scatter_plot(output_path, daily_means_dict, variable_name, BA=False):
    """
    Creates a scatter plot comparing the model and satellite data for each dataset
    in the daily_means_dict. Automatically identifies which is which based on key names.

    Parameters:
    - output_path: Path where the plot will be saved.
    - daily_means_dict: Dictionary with keys as dataset names and values as 1D arrays or Series.
    - variable_name: Short variable code (e.g., 'SST', 'CHL').
    - BA: If True, adds 'Basin Average' to the title.
    """

    # Extract model and satellite keys and data
    mod_key, sat_key = extract_mod_sat_keys(daily_means_dict)
    BAmod = pd.Series(daily_means_dict[mod_key])
    BAsat = pd.Series(daily_means_dict[sat_key])

    # Handle variable naming and units
    variable, unit = get_variable_label_unit(variable_name)

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        'Model': BAmod,
        'Satellite': BAsat
    })

    # Set style
    sns.set(style="whitegrid", context='notebook')
    sns.set_style("ticks")

    # Create figure
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])

    # Plot
    sns.scatterplot(x='Model', y='Satellite', data=df, color='#5976A2', alpha=0.7, ax=ax1, s=50)

    # Title and labels
    title = f'Scatter Plot of {variable} (Model vs. Satellite)'
    if BA:
        title += ' (Basin Average)'
    ax1.set_title(title, fontsize=20, fontweight='bold')
    ax1.set_xlabel(f'{variable} (Model) {unit}', fontsize=15)
    ax1.set_ylabel(f'{variable} (Satellite) {unit}', fontsize=15)

    # y = x line
    min_val, max_val = get_min_max_for_identity_line(df['Model'], df['Satellite'])
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (Ideal)', linewidth=2)

    ax1.tick_params(width=2, labelsize=13)
    ax1.grid(True, linestyle='--')

    for spine in ax1.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    ax1.legend(fontsize=12)
    plt.tight_layout()

    # Save and display
    output_path = Path(output_path)
    filename = f'{variable_name}_scatterplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
    
def scatter_plot_by_season(output_path, daily_means_dict, variable_name, BA=False):
    """
    Creates seasonal scatter plots (DJF, MAM, JJA, SON) comparing model vs satellite values,
    assuming the data starts from 01/01/2000 and is daily and continuous.
    Additionally, generates a comprehensive plot with all seasons combined.

    Parameters:
    - output_path: Path to save the plots.
    - daily_means_dict: Dictionary with 'mod' and 'sat' keys and 1D daily mean value arrays.
    - variable_name: Name of the variable (e.g., 'SST').
    - BA: Boolean for Basin Average (adds to title).
    """

    # Assign dates starting from Jan 1, 2000
    sample_array = next(iter(daily_means_dict.values()))
    dates = pd.date_range(start="2000-01-01", periods=len(sample_array), freq='D')

    # Extract model and satellite keys and arrays
    mod_key, sat_key = extract_mod_sat_keys(daily_means_dict)
    BAmod = np.array(daily_means_dict[mod_key])
    BAsat = np.array(daily_means_dict[sat_key])

    # Define seasons and colors
    seasons = {
        'DJF': 'gray',
        'MAM': 'green',
        'JJA': 'red',
        'SON': 'gold'
    }
    palette = ['#808080', '#008000', '#FF0000', '#FFD700']

    # Get variable descriptive name and unit
    variable, unit = get_variable_label_unit(variable_name)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_mod_points, all_sat_points, all_colors = [], [], []

    sns.set(style="whitegrid")
    sns.set_style("ticks")

    for season_name, color in seasons.items():
        mask = get_season_mask(dates, season_name)

        mod_season = BAmod[mask]
        sat_season = BAsat[mask]

        valid_mask = ~np.isnan(mod_season) & ~np.isnan(sat_season)
        mod_season = mod_season[valid_mask]
        sat_season = sat_season[valid_mask]

        if len(mod_season) == 0 or len(sat_season) == 0:
            print(f"Skipping {season_name}: no data found.")
            continue

        df = pd.DataFrame({
            'Model': mod_season,
            'Satellite': sat_season,
            'Season': [season_name] * len(mod_season)
        })

        plt.figure(figsize=(10, 8), dpi=300)
        ax = sns.scatterplot(x='Model', y='Satellite', data=df, color=color, alpha=0.7, label=f'{variable_name} {season_name}', s=50)

        # Regression fits using helper functions
        x_vals, y_vals = fit_huber(mod_season, sat_season)
        ax.plot(x_vals, y_vals, color='black', linestyle='-', linewidth=2, label='Linear Fit (Huber)')

        smoothed = fit_lowess(mod_season, sat_season)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='magenta', linestyle='-.', linewidth=2, label='Smoothed Fit (LOWESS)')

        min_val, max_val = get_min_max_for_identity_line(mod_season, sat_season)
        ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Ideal Fit')

        title = f'{variable_name} Scatter Plot (Model vs Satellite) - {season_name}'
        if BA:
            title += ' (Basin Average)'
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlabel(f'{variable} (Model - {season_name}) {unit}', fontsize=15)
        ax.set_ylabel(f'{variable} (Satellite - {season_name}) {unit}', fontsize=15)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=13, width=2)

        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        ax.grid(True, linestyle='--')
        plt.tight_layout()

        filename = f"{variable_name}_{season_name}_scatterplot.png"
        plt.savefig(output_path / filename)
        plt.show(block=False)
        plt.draw()
        plt.pause(2)
        plt.close()

        all_mod_points.extend(mod_season)
        all_sat_points.extend(sat_season)
        all_colors.extend([color] * len(mod_season))

    # Plot all seasons combined
    all_mod_points = np.array(all_mod_points)
    all_sat_points = np.array(all_sat_points)

    if len(all_mod_points) > 0:
        plt.figure(figsize=(10, 8), dpi=300)
        scatter_df = pd.DataFrame({
            'Model': all_mod_points,
            'Satellite': all_sat_points,
            'Season': all_colors
        })

        ax = sns.scatterplot(x='Model', y='Satellite', data=scatter_df, hue='Season', palette=palette, alpha=0.7, s=50)

        min_val_all, max_val_all = get_min_max_for_identity_line(all_mod_points, all_sat_points)
        ax.plot([min_val_all, max_val_all], [min_val_all, max_val_all], 'b--', linewidth=2, label='Ideal Fit')

        x_vals_all, y_vals_all = fit_huber(all_mod_points, all_sat_points)
        ax.plot(x_vals_all, y_vals_all, color='black', linestyle='-', linewidth=2, label='Linear Fit (Huber)')

        smoothed_all = fit_lowess(all_mod_points, all_sat_points)
        ax.plot(smoothed_all[:, 0], smoothed_all[:, 1], color='magenta', linestyle='-.', linewidth=2, label='Smoothed Fit (LOWESS)')

        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='DJF'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='MAM'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='JJA'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10, label='SON'),
            Line2D([0], [0], color='b', linestyle='--', linewidth=2, label='Ideal Fit'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Linear Fit (Huber)'),
            Line2D([0], [0], color='magenta', linestyle='-.', linewidth=2, label='Smoothed Fit (LOWESS)')
        ]
        ax.legend(handles=handles, loc='upper left', fontsize=12)

        ax.set_title(f'{variable} Scatter Plot (Model vs Satellite) - All Seasons', fontsize=20, fontweight='bold')
        ax.set_xlabel(f'{variable} (Model - All Seasons) {unit}', fontsize=15)
        ax.set_ylabel(f'{variable} (Satellite - All Seasons) {unit}', fontsize=15)
        ax.tick_params(axis='both', labelsize=13, width=2)

        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        ax.grid(True, linestyle='--')
        plt.tight_layout()

        comprehensive_filename = f"{variable_name}_all_seasons_scatterplot.png"
        plt.savefig(output_path / comprehensive_filename)
        plt.show(block=False)
        plt.draw()
        plt.pause(2)
        plt.close()
        
def plot_monthly_comparison_boxplot(output_path, data_dict, variable_name='Variable', unit=''):
    """
    Plots a monthly boxplot comparison for two data sources (e.g., model vs satellite),
    automatically identifying the sources based on key names.

    Parameters:
        data_dict (dict): Dictionary with two sub-dictionaries, each containing:
                          {year: [month_1_array, ..., month_12_array]}
        variable_name (str): Name of the variable to plot (e.g., 'SST')
        unit (str): Unit of the variable (e.g., '°C', 'mg/m³')
    """
    
    # --- Dynamic key detection based on naming conventions ---
    model_keys = [key for key in data_dict if 'mod' in key.lower()]
    sat_keys = [key for key in data_dict if 'sat' in key.lower()]

    if len(model_keys) != 1 or len(sat_keys) != 1:
        raise ValueError("Could not uniquely identify model and satellite keys based on 'mod' and 'sat' keywords.")

    model_key = model_keys[0]
    sat_key = sat_keys[0]

    # --- Setup for plotting ---
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Prepare data for seaborn
    plot_data = []

    for month_idx in range(12):
        # --- Model data ---
        model_values = []
        for year in data_dict[model_key]:
            array = np.asarray(data_dict[model_key][year][month_idx])
            model_values.append(array.flatten())
        model_all = np.concatenate(model_values)
        model_all = model_all[~np.isnan(model_all)]
        
        plot_data.extend([(model_val, f'{month_names[month_idx]} Model') for model_val in model_all])

        # --- Satellite data ---
        sat_values = []
        for year in data_dict[sat_key]:
            array = np.asarray(data_dict[sat_key][year][month_idx])
            sat_values.append(array.flatten())
        sat_all = np.concatenate(sat_values)
        sat_all = sat_all[~np.isnan(sat_all)]
        
        plot_data.extend([(sat_val, f'{month_names[month_idx]} Satellite') for sat_val in sat_all])

    # Convert to DataFrame for Seaborn
    plot_df = pd.DataFrame(plot_data, columns=['Value', 'Label'])

    # --- Plotting ---
    plt.figure(figsize=(16, 6), dpi=300)
    ax = sns.boxplot(x='Label', y='Value', data=plot_df, palette=['#5976A2', '#BF636B'] * 12, showfliers=True)
    
    # Styling
    ax.set_title(f'Monthly {variable_name} Comparison: Model vs Satellite', fontsize=16, fontweight='bold')
    ylabel = f'{variable_name} ({unit})' if unit else variable_name
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    ax.tick_params(width=2)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    plt.tight_layout()

    # Save and show the plot
    output_path = Path(output_path)
    filename = f'{variable_name}_boxplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()  # <-- Force rendering
    plt.pause(3)
    plt.close()
    
def plot_monthly_comparison_violinplot(output_path, data_dict, variable_name='Variable', unit=''):
    """
    Plots a monthly boxplot comparison for two data sources (e.g., model vs satellite),
    automatically identifying the sources based on key names.

    Parameters:
        data_dict (dict): Dictionary with two sub-dictionaries, each containing:
                          {year: [month_1_array, ..., month_12_array]}
        variable_name (str): Name of the variable to plot (e.g., 'SST')
        unit (str): Unit of the variable (e.g., '°C', 'mg/m³')
    """
    
    # --- Dynamic key detection based on naming conventions ---
    model_keys = [key for key in data_dict if 'mod' in key.lower()]
    sat_keys = [key for key in data_dict if 'sat' in key.lower()]

    if len(model_keys) != 1 or len(sat_keys) != 1:
        raise ValueError("Could not uniquely identify model and satellite keys based on 'mod' and 'sat' keywords.")

    model_key = model_keys[0]
    sat_key = sat_keys[0]

    # --- Setup for plotting ---
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Prepare data for seaborn
    plot_data = []

    for month_idx in range(12):
        # --- Model data ---
        model_values = []
        for year in data_dict[model_key]:
            array = np.asarray(data_dict[model_key][year][month_idx])
            model_values.append(array.flatten())
        model_all = np.concatenate(model_values)
        model_all = model_all[~np.isnan(model_all)]
        
        plot_data.extend([(model_val, f'{month_names[month_idx]} Model') for model_val in model_all])

        # --- Satellite data ---
        sat_values = []
        for year in data_dict[sat_key]:
            array = np.asarray(data_dict[sat_key][year][month_idx])
            sat_values.append(array.flatten())
        sat_all = np.concatenate(sat_values)
        sat_all = sat_all[~np.isnan(sat_all)]
        
        plot_data.extend([(sat_val, f'{month_names[month_idx]} Satellite') for sat_val in sat_all])

    # Convert to DataFrame for Seaborn
    plot_df = pd.DataFrame(plot_data, columns=['Value', 'Label'])

    # --- Plotting ---
    plt.figure(figsize=(16, 6), dpi=300)
    ax = sns.violinplot(x='Label', y='Value', data=plot_df, palette=['#5976A2', '#BF636B'] * 12, showfliers=True)
    
    # Styling
    ax.set_title(f'Monthly {variable_name} Comparison: Model vs Satellite', fontsize=16, fontweight='bold')
    ylabel = f'{variable_name} ({unit})' if unit else variable_name
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    ax.tick_params(width=2)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    plt.tight_layout()

    # Save and show the plot
    output_path = Path(output_path)
    filename = f'{variable_name}_violinplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()  # <-- Force rendering
    plt.pause(3)
    plt.close()
    
# Function to plot the overall metric and the monthly metrics
def plot_metric(metric_name, overall_value, monthly_values, y_label, output_path):
    months = [calendar.month_name[i + 1] for i in range(12)]
    df = pd.DataFrame({'Month': months, 'Value': monthly_values})

    # Normalize color map and assign marker colors
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    marker_colors = []
    for val in monthly_values:
        if val < 0:
            marker_colors.append('red')
        elif val > 1:
            marker_colors.append('green')
        else:
            marker_colors.append(cmap(norm(val)))

    # Seaborn styling
    sns.set(style="whitegrid")
    sns.set_style("ticks")

    # Create the plot
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.lineplot(x='Month', y='Value', data=df, color='blue', lw=1.75, label="Monthly Trend")
    
    # Add reference lines
    if metric_name in [
        "Nash-Sutcliffe Efficiency",
        "Nash-Sutcliffe Efficiency (Logarithmic)",
        "Modified NSE ($E_1$, j=1)",
        "Relative NSE ($E_{rel}$)"
    ]:
        ax.axhline(0, linestyle='-.', lw=3, color='red', label='Zero Reference')

    ax.axhline(overall_value, linestyle='--', lw=2, color='black', label='Overall')

    # Plot colored markers
    for x, y, color in zip(months, monthly_values, marker_colors):
        ax.plot(x, y, marker='o', markersize=10, color=color, markeredgecolor='black', markeredgewidth=1.2)

    # Labels and formatting
    ax.set_title(metric_name, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel(f'${y_label}$', fontsize=12)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45)
    ax.tick_params(width=2)
    ax.legend(loc='lower right')
    
    ax.grid(True, linestyle='--')
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    plt.tight_layout()

    # Save and show
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f'{metric_name}.png'
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
        
def Benthic_depth(Bmost, output_path):
    """
    Plots the benthic layer depth from the Bmost 2D array on a map using Cartopy,
    equivalent to MATLAB's 'equidistant cylindrical' projection.
    """
    if not isinstance(Bmost, np.ndarray) or Bmost.ndim != 2:
        raise ValueError("Bmost must be a 2D NumPy array")
        
    # Set all 0 values to NaN
    Bmost = np.where(Bmost == 0, np.nan, (Bmost) * 2)  # Convert layer index to depth in meters

    # Constants (grid origin and resolution)
    x_origin = 12.200
    x_step = 0.0100
    y_origin = 43.774
    y_step = 0.0073
    epsilon = 0.06

    Yrow, Xcol = Bmost.shape

    # Generate latitude and longitude arrays
    lats = y_origin + np.arange(Yrow) * y_step + .2*epsilon
    lons = x_origin + np.arange(Xcol) * x_step + .4*epsilon  # Shift longitude by epsilon for visual consistency

    # Define map bounds (same as MATLAB m_proj 'lon'/'lat' range)
    min_lat, max_lat = lats.min() + epsilon, lats.max() + epsilon
    min_lon, max_lon = lons.min() - epsilon, lons.max() + epsilon

    # Create map with Plate Carrée projection (equidistant cylindrical)
    plt.figure(figsize=(10, 10))  # Square figure
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Plot the data as filled contours
    contour_levels = np.linspace(np.nanmin(Bmost), np.nanmax(Bmost), 26)  # Define contour levels
    contour = ax.contourf(lons, lats, Bmost, levels=contour_levels, cmap='jet', extend='both', transform=ccrs.PlateCarree())

    # Add coastlines and borders with custom styling
    ax.coastlines(linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Set the title for the plot
    ax.set_title("Benthic Layer Depth", fontsize=16, fontweight='bold')

    # Add gridlines with labels
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='gray', linestyle='--')

    # Create the colorbar with custom position and ticks
    colorbar_width = 0.65  # 70% of the figure width
    colorbar_height = 0.025  # 1.2% of the figure height
    left_position = (1 - colorbar_width) / 2  # Center horizontally by adjusting left position
    bottom_position = 0.175  # 15% from the bottom of the figure for the colorbar

    # Create the colorbar axes and place it at the desired position
    cbar_ax = plt.gcf().add_axes([left_position, bottom_position, colorbar_width, colorbar_height])

    # Create a new BoundaryNorm for the colorbar
    norm = mcolors.BoundaryNorm(np.linspace(np.nanmin(Bmost), np.nanmax(Bmost), 11), contour.cmap.N)
    cbar = plt.colorbar(contour, cax=cbar_ax, orientation='horizontal', norm=norm, extend='both')
    cbar.set_label('[m]', fontsize=12)
    cbar.set_ticks(np.linspace(np.nanmin(Bmost), np.nanmax(Bmost), 6).astype(int))  # Set ticks every 6th value

    # Increase the font size of the tick labels
    cbar.ax.tick_params(direction='in', length=18, labelsize=10)

    # Thicken the borders of the subplot (axes)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Show the plot with custom styling
    filename = "NA - Benthic Depth.png"
    save_path = Path(output_path, filename)
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(2)
    plt.close()
    
def Benthic_chemical_plot(MinLambda, MaxLambda, MinPhi, MaxPhi, P_2d, t, lonp, latp, bfm2plot, Mname, ystr, selected_unit, selected_description, output_path):
    epsilon = 0.06
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([MinLambda, MaxLambda, MinPhi, MaxPhi], crs=ccrs.PlateCarree())

    TbP = P_2d[t, :, :]

    # Set vmin/vmax and contour levels
    if bfm2plot == 'O2o':
        vmin, vmax = 0, 350
        num_ticks = 8
    elif bfm2plot == 'Chla':
        vmin, vmax = 0, 16
        num_ticks = 5
    elif bfm2plot == 'N1p':
        vmin, vmax = 0, 0.4
        num_ticks = 5
    elif bfm2plot == 'N3n':
        vmin, vmax = 0, 30
        num_ticks = 6
    elif bfm2plot == 'N4n':
        vmin, vmax = 0, 10
        num_ticks = 6
    elif bfm2plot == 'P1c':
        vmin,vmax = 0, 300
        num_ticks = 6
    elif bfm2plot == 'P2c':
        vmin, vmax = 0, 200
        num_ticks = 6
    elif bfm2plot == 'P3c':
        vmin, vmax = 0, 30
        num_ticks = 6
    elif bfm2plot == 'P4c':
        vmin, vmax = 0, 300
        num_ticks = 6
    elif bfm2plot == 'Z3c':
        vmin, vmax = 0, 5
        num_ticks = 6
    elif bfm2plot == 'Z4c':
        vmin, vmax = 0, 20
        num_ticks = 6
    elif bfm2plot == 'Z5c':
        vmin, vmax = 0, 100
        num_ticks = 6
    elif bfm2plot == 'Z6c':
        vmin, vmax = 0, 60
        num_ticks = 8
    elif bfm2plot == 'R6c':
        vmin, vmax = 0, 500
        num_ticks = 6
    elif bfm2plot == 'votemper':
        vmin, vmax = 5, 25
        num_ticks = 5
    elif bfm2plot == 'vosaline':
        vmin, vmax = 36, 39
        num_ticks = 7
    elif bfm2plot == 'vodensity_EOS' or bfm2plot == 'vodensity_EOS80' or bfm2plot == 'vodensity_TEOS10':
        vmin, vmax = 1025, 1030
        num_ticks = 6
    else:
        # Used to find out the fixed range to use in the colomap
        # Insert a new fixed range to improve display of phenomena
        vmin, vmax = float(np.nanmin(TbP)), float(np.nanmax(TbP))
        num_ticks = 6  # default fallback
        
    if bfm2plot == 'vosaline' or bfm2plot == 'votemper' or bfm2plot == 'vodensity_EOS' or bfm2plot == 'vodensity_EOS80' or bfm2plot == 'vodensity_TEOS10':
        extended = 'both'
    else:
        extended = 'max'

    levels = np.linspace(vmin, vmax, 41)

    # Plot using regular 'jet' colormap
    ax.contourf(
        lonp + 0.4 * epsilon,
        latp + 0.2 * epsilon,
        TbP,
        levels=levels,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        extend=extended,
        transform=ccrs.PlateCarree()
    )
    
    if bfm2plot == 'vodensity_EOS' or bfm2plot == 'vodensity_EOS80' or bfm2plot == 'vodensity_TEOS10':
        ax.contour(
            lonp + 0.4 * epsilon,
            latp + 0.2 * epsilon,
            TbP,
            levels=[1025, 1026, 1027, 1028, 1029, 1030],
            colors='black',  # draw black contour lines
            linewidths=1,   # optional: control line thickness
            transform=ccrs.PlateCarree()
            )

    ax.coastlines(linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.set_title(f"{selected_description} | {Mname[t]} - {ystr}", fontsize=16, fontweight='bold')
    ax.gridlines(draw_labels=True, dms=True, color='gray', linestyle='--')

    # --- Colorbar setup ---
    colorbar_width = 0.65
    colorbar_height = 0.025
    left_position = (1 - colorbar_width) / 2
    bottom_position = 0.175
    cbar_ax = plt.gcf().add_axes([left_position, bottom_position, colorbar_width, colorbar_height])

    colorbar_cmap = plt.get_cmap('jet')
    colorbar_norm = BoundaryNorm(levels, ncolors=colorbar_cmap.N)
    mappable = ScalarMappable(norm=colorbar_norm, cmap=colorbar_cmap)

    field_units = format_unit(selected_unit)
    cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='horizontal', extend=extended)
    cbar.set_label(rf'$\left[{field_units[1:-1]}\right]$', fontsize=14)
    cbar.set_ticks(np.linspace(vmin, vmax, num_ticks))
    cbar.ax.tick_params(direction='in', length=18, labelsize=10)

    # Thicken subplot borders
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # --- Output and save ---
    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    chem_output_path = os.path.join(output_path, timestamp, ystr)
    os.makedirs(chem_output_path, exist_ok=True)

    filename = f"NAmod - {bfm2plot} - {ystr} - {Mname[t]}"
    save_path = Path(chem_output_path, filename)
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(2)
    plt.close()