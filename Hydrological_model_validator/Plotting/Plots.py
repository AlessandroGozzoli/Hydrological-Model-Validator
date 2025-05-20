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
from typing import Union, Dict, Any
from types import SimpleNamespace

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
from Hydrological_model_validator.Plotting.formatting import (
                        format_unit,
                        get_min_max_for_identity_line,
                        get_variable_label_unit)

from Hydrological_model_validator.Processing.data_alignment import (extract_mod_sat_keys,
                                                                    gather_monthly_data_across_years)

from Hydrological_model_validator.Processing.stats_math_utils import (fit_huber,
                                                                      fit_lowess)

from Hydrological_model_validator.Processing.time_utils import get_season_mask

from Hydrological_model_validator.Plotting.default_plot_options import (default_plot_options_ts,
                                                                        default_plot_options_scatter,
                                                                        default_scatter_by_season_options,
                                                                        default_boxplot_options,
                                                                        default_violinplot_options)

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################

###############################################################################
def timeseries_basin_average(
    data_dict: Dict[str, Union[pd.Series, list]],
    BIAS: Union[pd.Series, list],
    **kwargs: Any
) -> None:
    """
    Plots daily means and BIAS.
    
    All configurable parameters (including output_path, variable_name, etc.) are passed via kwargs.
    If a parameter is not specified, it falls back to default_plot_options.
    """

    # Merge defaults and kwargs; kwargs overrides defaults
    options = SimpleNamespace(**{**default_plot_options_ts, **kwargs})

    # Validate critical options
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")
    if options.variable_name is None:
        raise ValueError("variable_name must be specified either in kwargs or default options.")

    # Extract variable and unit if missing
    if options.variable is None or options.unit is None:
        var, u = get_variable_label_unit(options.variable_name)
        if options.variable is None:
            options.variable = var
        if options.unit is None:
            options.unit = u

    title = f'Daily Mean Values for {options.variable_name} Datasets'
    if options.BA:
        title += ' (Basin Average)'

    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    label_lookup = {
        mod_key: "Model Output",
        sat_key: "Satellite Observations"
    }

    fig = plt.figure(figsize=options.figsize, dpi=options.dpi)
    gs = GridSpec(2, 1, height_ratios=[8, 4])

    # First subplot
    ax1 = fig.add_subplot(gs[0])
    for key, daily_mean in data_dict.items():
        label = label_lookup.get(key, key)
        color = next(options.color_palette)

        if not isinstance(daily_mean, pd.Series):
            daily_mean = pd.Series(daily_mean)

        sns.lineplot(data=daily_mean, label=label, ax=ax1, lw=options.line_width, color=color)

    ax1.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
    ax1.set_ylabel(f'{options.variable} {options.unit}', fontsize=options.label_fontsize)
    ax1.tick_params(width=2)
    ax1.legend(loc='upper left', fontsize=options.legend_fontsize)
    ax1.grid(True, linestyle='--')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Second subplot
    ax2 = fig.add_subplot(gs[1])
    if not isinstance(BIAS, pd.Series):
        BIAS = pd.Series(BIAS)
    sns.lineplot(data=BIAS, color='k', ax=ax2)
    ax2.set_title(f'BIAS ({options.variable_name})', fontsize=options.bias_title_fontsize, fontweight='bold')
    ax2.set_ylabel(f'BIAS {options.unit}', fontsize=options.label_fontsize)
    ax2.tick_params(width=2)
    ax2.grid(True, linestyle='--')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    plt.tight_layout()
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f'{options.variable_name}_timeseries.png'
    save_path = output_path / filename
    plt.savefig(save_path, **options.savefig_kwargs)
    plt.show(block=False)
    plt.draw()
    plt.close()
###############################################################################
    
###############################################################################
def scatter_plot(
    data_dict: Dict[str, Union[pd.Series, list]],
    **kwargs: Any
) -> None:
    """
    Creates a scatter plot comparing the model and satellite data for each dataset
    in daily_means_dict. Most parameters are passed as **kwargs with defaults.

    Required:
    - daily_means_dict: Dict with keys as dataset names and values as 1D arrays or Series.

    Optional (via kwargs):
    - output_path (str or Path): Where to save plot.
    - variable_name (str): Variable short name e.g. 'SST'.
    - BA (bool): Whether to add 'Basin Average' to title.
    - figsize (tuple): Figure size.
    - dpi (int): Figure DPI.
    - color (str): Scatter plot color.
    - alpha (float): Scatter plot transparency.
    - marker_size (int): Scatter plot marker size.
    - title_fontsize (int)
    - label_fontsize (int)
    - line_width (int): Width of y=x line.
    - tick_labelsize (int)
    - legend_fontsize (int)
    - pause_time (int or float): Seconds to pause plot display.

    Returns:
    None
    """

    # Merge defaults and kwargs; kwargs overrides defaults
    options = SimpleNamespace(**{**default_plot_options_scatter, **kwargs})

    # Validate critical options
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")
    if options.variable_name is None:
        raise ValueError("variable_name must be specified either in kwargs or default options.")

    # Extract variable and unit if missing
    if not hasattr(options, 'variable') or options.variable is None or \
       not hasattr(options, 'unit') or options.unit is None:
        var, u = get_variable_label_unit(options.variable_name)
        if not hasattr(options, 'variable') or options.variable is None:
            options.variable = var
        if not hasattr(options, 'unit') or options.unit is None:
            options.unit = u

    # Extract model and satellite keys
    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    BAmod = pd.Series(data_dict[mod_key])
    BAsat = pd.Series(data_dict[sat_key])

    # Prepare DataFrame for plotting
    df = pd.DataFrame({'Model': BAmod, 'Satellite': BAsat})

    sns.set(style="whitegrid", context='notebook')
    sns.set_style("ticks")

    fig = plt.figure(figsize=options.figsize, dpi=options.dpi)
    ax1 = fig.add_subplot(1, 1, 1)

    sns.scatterplot(
        x='Model',
        y='Satellite',
        data=df,
        color=options.color,
        alpha=options.alpha,
        s=options.marker_size,
        ax=ax1
    )

    title = f'Scatter Plot of {options.variable} (Model vs. Satellite)'
    if options.BA:
        title += ' (Basin Average)'

    ax1.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
    ax1.set_xlabel(f'{options.variable} (Model) {options.unit}', fontsize=options.label_fontsize)
    ax1.set_ylabel(f'{options.variable} (Satellite) {options.unit}', fontsize=options.label_fontsize)

    min_val, max_val = get_min_max_for_identity_line(df['Model'], df['Satellite'])
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=options.line_width, label='y = x (Ideal)')

    ax1.tick_params(width=2, labelsize=options.tick_labelsize)
    ax1.grid(True, linestyle='--')

    for spine in ax1.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    ax1.legend(fontsize=options.legend_fontsize)

    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f'{options.variable_name}_scatterplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)

    plt.show(block=False)
    plt.draw()
    plt.pause(options.pause_time)
    plt.close()
###############################################################################

###############################################################################    
def scatter_plot_by_season(daily_means_dict, **kwargs):
    """
    Creates seasonal scatter plots (DJF, MAM, JJA, SON) comparing model vs satellite values,
    plus a combined scatter plot for all seasons.

    Inputs:
    - daily_means_dict: dict with 'mod' and 'sat' keys and 1D daily mean arrays or Series
    - kwargs: options for customization, merged with defaults
    
    Options in kwargs:
    - output_path: str or Path (required)
    - variable_name: str (required)
    - BA: bool, add "(Basin Average)" to title
    - figsize, dpi, season_colors, alpha, marker_size, font sizes, line_width, tick_labelsize, pause_time
    """

    # Merge defaults and kwargs; kwargs overrides defaults
    options = SimpleNamespace(**{**default_scatter_by_season_options, **kwargs})

    # Validate critical options
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")
    if options.variable_name is None:
        raise ValueError("variable_name must be specified either in kwargs or default options.")

    # Extract variable and unit if missing
    if not hasattr(options, 'variable') or not hasattr(options, 'unit') or options.variable is None or options.unit is None:
        var, u = get_variable_label_unit(options.variable_name)
        if not hasattr(options, 'variable') or options.variable is None:
            options.variable = var
        if not hasattr(options, 'unit') or options.unit is None:
            options.unit = u

    # Assign dates assuming continuous daily from 2000-01-01
    sample_array = next(iter(daily_means_dict.values()))
    dates = pd.date_range(start="2000-01-01", periods=len(sample_array), freq='D')

    # Extract model and satellite keys and arrays
    mod_key, sat_key = extract_mod_sat_keys(daily_means_dict)
    BAmod = np.array(daily_means_dict[mod_key])
    BAsat = np.array(daily_means_dict[sat_key])

    # Define seasons and colors (allow override)
    seasons = options.season_colors

    # Prepare output directory
    output_path = Path(options.output_path)
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

        plt.figure(figsize=options.figsize, dpi=options.dpi)
        ax = sns.scatterplot(
            x='Model', y='Satellite', data=df,
            color=color, alpha=options.alpha,
            label=f'{options.variable} {season_name}', s=options.marker_size
        )

        # Regression fits
        x_vals, y_vals = fit_huber(mod_season, sat_season)
        ax.plot(x_vals, y_vals, color='black', linestyle='-', linewidth=options.line_width, label='Linear Fit (Huber)')

        smoothed = fit_lowess(mod_season, sat_season)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='magenta', linestyle='-.', linewidth=options.line_width, label='Smoothed Fit (LOWESS)')

        min_val, max_val = get_min_max_for_identity_line(mod_season, sat_season)
        ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=options.line_width, label='Ideal Fit')

        title = f'{options.variable} Scatter Plot (Model vs Satellite) - {season_name}'
        if options.BA:
            title += ' (Basin Average)'

        ax.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
        ax.set_xlabel(f'{options.variable} (Model - {season_name}) {options.unit}', fontsize=options.label_fontsize)
        ax.set_ylabel(f'{options.variable} (Satellite - {season_name}) {options.unit}', fontsize=options.label_fontsize)
        ax.legend(fontsize=options.legend_fontsize)
        ax.tick_params(axis='both', labelsize=options.tick_labelsize, width=options.line_width)

        for spine in ax.spines.values():
            spine.set_linewidth(options.line_width)
            spine.set_edgecolor('black')

        ax.grid(True, linestyle='--')
        plt.tight_layout()

        filename = f"{options.variable_name}_{season_name}_scatterplot.png"
        plt.savefig(output_path / filename)
        plt.show(block=False)
        plt.draw()
        plt.pause(options.pause_time)
        plt.close()

        all_mod_points.extend(mod_season)
        all_sat_points.extend(sat_season)
        all_colors.extend([color] * len(mod_season))

    # Combined plot
    all_mod_points = np.array(all_mod_points)
    all_sat_points = np.array(all_sat_points)

    if len(all_mod_points) > 0:
        plt.figure(figsize=options.figsize, dpi=options.dpi)
        scatter_df = pd.DataFrame({
            'Model': all_mod_points,
            'Satellite': all_sat_points,
            'Season': all_colors
        })

        ax = sns.scatterplot(
            x='Model', y='Satellite', data=scatter_df,
            hue='Season', palette=list(seasons.values()),
            alpha=options.alpha, s=options.marker_size
        )

        min_val_all, max_val_all = get_min_max_for_identity_line(all_mod_points, all_sat_points)
        ax.plot([min_val_all, max_val_all], [min_val_all, max_val_all], 'b--', linewidth=options.line_width, label='Ideal Fit')

        x_vals_all, y_vals_all = fit_huber(all_mod_points, all_sat_points)
        ax.plot(x_vals_all, y_vals_all, color='black', linestyle='-', linewidth=options.line_width, label='Linear Fit (Huber)')

        smoothed_all = fit_lowess(all_mod_points, all_sat_points)
        ax.plot(smoothed_all[:, 0], smoothed_all[:, 1], color='magenta', linestyle='-.', linewidth=options.line_width, label='Smoothed Fit (LOWESS)')

        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=clr, markersize=10, label=season)
            for season, clr in seasons.items()
        ] + [
            Line2D([0], [0], color='b', linestyle='--', linewidth=options.line_width, label='Ideal Fit'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=options.line_width, label='Linear Fit (Huber)'),
            Line2D([0], [0], color='magenta', linestyle='-.', linewidth=options.line_width, label='Smoothed Fit (LOWESS)')
        ]
        ax.legend(handles=handles, loc='upper left', fontsize=options.legend_fontsize)

        ax.set_title(f'{options.variable} Scatter Plot (Model vs Satellite) - All Seasons', fontsize=options.title_fontsize, fontweight='bold')
        ax.set_xlabel(f'{options.variable} (Model - All Seasons) {options.unit}', fontsize=options.label_fontsize)
        ax.set_ylabel(f'{options.variable} (Satellite - All Seasons) {options.unit}', fontsize=options.label_fontsize)
        ax.tick_params(axis='both', labelsize=options.tick_labelsize, width=options.line_width)

        for spine in ax.spines.values():
            spine.set_linewidth(options.line_width)
            spine.set_edgecolor('black')

        ax.grid(True, linestyle='--')
        plt.tight_layout()

        comprehensive_filename = f"{options.variable_name}_all_seasons_scatterplot.png"
        plt.savefig(output_path / comprehensive_filename)
        plt.show(block=False)
        plt.draw()
        plt.pause(options.pause_time)
        plt.close()
###############################################################################
        
###############################################################################
def plot_monthly_comparison_boxplot(data_dict, **kwargs):
    # Merge defaults and kwargs; kwargs overrides defaults
    options = SimpleNamespace(**{**default_boxplot_options, **kwargs})

    # Validate critical options
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")
    if options.variable_name is None:
        raise ValueError("variable_name must be specified either in kwargs or default options.")

    # Identify model and satellite keys
    model_key, sat_key = extract_mod_sat_keys(data_dict)

    # Get variable label and unit
    var_label, var_unit = get_variable_label_unit(options.variable_name)

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Prepare data for seaborn boxplot
    plot_data = []
    for month_idx in range(12):
        model_all = gather_monthly_data_across_years(data_dict, model_key, month_idx)
        plot_data.extend([(val, f'{month_names[month_idx]} Model') for val in model_all])

        sat_all = gather_monthly_data_across_years(data_dict, sat_key, month_idx)
        plot_data.extend([(val, f'{month_names[month_idx]} Satellite') for val in sat_all])

    plot_df = pd.DataFrame(plot_data, columns=['Value', 'Label'])

    # Plotting
    plt.figure(figsize=options.figsize, dpi=options.dpi)
    ax = sns.boxplot(
        x='Label', y='Value', data=plot_df,
        palette=options.palette,
        showfliers=options.showfliers
    )

    # Styling
    ax.set_title(f'Monthly {var_label} Comparison: Model vs Satellite',
                 fontsize=options.title_fontsize,
                 fontweight=options.title_fontweight)
    ylabel = f'{var_label} {var_unit}'
    ax.set_ylabel(ylabel, fontsize=options.ylabel_fontsize)
    ax.set_xlabel(options.xlabel)
    ax.grid(True, linestyle='--', alpha=options.grid_alpha)
    plt.xticks(rotation=options.xtick_rotation)
    ax.tick_params(width=options.tick_width)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    plt.tight_layout()

    # Save and show plot
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f'{var_label}_boxplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(options.pause_time)
    plt.close()
###############################################################################
    
###############################################################################
def plot_monthly_comparison_violinplot(data_dict, **kwargs):
    # Merge defaults and kwargs
    options = SimpleNamespace(**{**default_violinplot_options, **kwargs})

    # Validate critical options
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")
    if options.variable_name is None:
        raise ValueError("variable_name must be specified either in kwargs or default options.")

    # Extract variable label and unit if missing
    variable, unit = get_variable_label_unit(options.variable_name)

    # Identify model and satellite keys
    model_key, sat_key = extract_mod_sat_keys(data_dict)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plot_data = []
    for month_idx in range(12):
        model_all = gather_monthly_data_across_years(data_dict, model_key, month_idx)
        plot_data.extend([(val, f'{month_names[month_idx]} Model') for val in model_all])

        sat_all = gather_monthly_data_across_years(data_dict, sat_key, month_idx)
        plot_data.extend([(val, f'{month_names[month_idx]} Satellite') for val in sat_all])

    plot_df = pd.DataFrame(plot_data, columns=['Value', 'Label'])

    plt.figure(figsize=options.figsize, dpi=options.dpi)
    ax = sns.violinplot(x='Label', y='Value', data=plot_df,
                        palette=options.palette, cut=options.cut)

    ax.set_title(f'Monthly {variable} Comparison: Model vs Satellite',
                 fontsize=options.title_fontsize, fontweight=options.title_fontweight)
    ax.set_ylabel(f'{variable} {unit}', fontsize=options.ylabel_fontsize)
    ax.set_xlabel('', fontsize=options.xlabel_fontsize)
    ax.grid(True, linestyle='--', alpha=options.grid_alpha)
    plt.xticks(rotation=options.xtick_rotation)
    ax.tick_params(width=options.tick_width)

    for spine in ax.spines.values():
        spine.set_linewidth(options.spine_linewidth)
        spine.set_edgecolor('black')

    plt.tight_layout()

    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f'{variable}_violinplot.png'
    plt.savefig(output_path / filename)

    plt.show(block=False)
    plt.draw()
    plt.pause(options.pause_time)
    plt.close()
###############################################################################    

###############################################################################
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
###############################################################################      

###############################################################################
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
###############################################################################

###############################################################################
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