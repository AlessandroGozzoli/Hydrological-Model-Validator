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
import numpy as np
import pandas as pd
from pathlib import Path
import calendar
from typing import Union, Dict, Any
from types import SimpleNamespace
import itertools
from itertools import starmap, chain
from functools import partial

# Plotting Libraries
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Custom Imports
from .formatting import (plot_line,
                        get_min_max_for_identity_line,
                        get_variable_label_unit,
                        style_axes_spines)

from ..Processing.data_alignment import (extract_mod_sat_keys,
                                         gather_monthly_data_across_years)

from ..Processing.stats_math_utils import (fit_huber,
                                           fit_lowess)

from ..Processing.time_utils import get_season_mask

from .default_plot_options import (default_plot_options_ts,
                                   default_plot_options_scatter,
                                   default_scatter_by_season_options,
                                   default_boxplot_options,
                                   default_violinplot_options,
                                   default_efficiency_plot_options)

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################

###############################################################################
def timeseries(data_dict: Dict[str, Union[pd.Series, list]], BIAS: Union[pd.Series, list], **kwargs: Any) -> None:
    """
    Plot time series of daily mean values from multiple datasets along with BIAS.

    This function generates a two-panel time series plot:
      1. The upper subplot displays daily mean values of each dataset (typically model and satellite data).
      2. The lower subplot shows the BIAS (model - satellite) as a time series.

    The figure is saved to a specified output directory as a PNG file and displayed using matplotlib.

    Parameters
    ----------
    data_dict : Dict[str, Union[pd.Series, list]]
        Dictionary containing daily mean values for different sources (e.g., model and satellite).
        Keys are strings identifying the data source.
        Values should be `pandas.Series` with datetime indices or lists that can be converted to Series.

    BIAS : Union[pd.Series, list]
        Series (or list) representing the BIAS time series (typically model - satellite).
        Should be time-aligned with the values in `data_dict`.

    Accepted kwargs include:
    -------------------------
    Keyword arguments overriding default plotting options. Include:
        - output_path (str or Path)       : Required. Path where the figure should be saved.
        - variable_name (str)             : Required. Variable code name (used to infer full name and unit).
        - variable (str)                  : Full variable name (e.g., "Chlorophyll"). Used in titles and axis.
        - unit (str)                     : Unit of measurement (e.g., "mg Chl/m³"). Displayed on axis.
        - BA (bool)                     : If True, appends " (Basin Average)" to the title.
        - figsize (tuple of float)        : Size of figure in inches (default typically (12, 8)).
        - dpi (int)                     : Resolution of the figure (default 100).
        - color_palette (iterator)        : Iterator of colors (e.g., `itertools.cycle(sns.color_palette("tab10"))`).
        - line_width (float)             : Width of plotted lines (default 2.0).
        - title_fontsize (int)          : Font size of the main title.
        - bias_title_fontsize (int)     : Font size of the BIAS subplot title.
        - label_fontsize (int)          : Font size of axis labels.
        - legend_fontsize (int)         : Font size of the legend.
        - savefig_kwargs (dict)           : Additional args for `plt.savefig()`, e.g., `bbox_inches`, `transparent`.
        
    Example
    -------
    >>> timeseries(
    ...     data_dict={'model': model_series, 'satellite': sat_series},
    ...     BIAS=model_series - sat_series,
    ...     variable_name='Chl',
    ...     output_path='figures/',
    ...     BA=True
    ... )

    Notes
    -----
    - If `variable` or `unit` is not provided, the function attempts to resolve them using
      `get_variable_label_unit(variable_name)`.

    - The data series are auto-converted to `pandas.Series` if passed as lists.
    
    - The order and coloring of plotted datasets depend on the order in `data_dict` and `color_palette`.
    """

    # ----- RETRIEVE DEFAUL OPTIONS -----
    options = SimpleNamespace(**{**default_plot_options_ts, **kwargs})

    # ----- RETRIEVE NECESSARY OUTPUT PATH AND VARIABLE/UNIT INFO -----
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    if options.variable_name is not None:
        # Infer full variable name and unit from short name
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    else:
        # variable_name not given — require both variable and unit
        if options.variable is None or options.unit is None:
            raise ValueError(
                "If 'variable_name' is not provided, both 'variable' and 'unit' must be specified in kwargs or defaults."
                )

    # ----- ADD BASIN AVERAGE LABEL, STRONGLY SUGGESTED IF DATA USED IS -----
    # ----- L4 BUT NOT NECESSARY -----
    title = f'Daily Mean Values for {options.variable_name} Datasets'
    if options.BA:
        title += ' (Basin Average)'

    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    label_lookup = {
        mod_key: "Model Output",
        sat_key: "Satellite Observations"
    }

    # ----- PRE-CONVERT DATA_DICT AND BIAS TO PANDAS SERIES IF -----
    # ----- NOT ALREADY IN THIS FORMAT WHEN GIVE IN INPUT -----
    # ----- DONE TO PASS THEM MORE EASILY TO SEABORN FOR PLOTTING -----
    data_dict = {k: pd.Series(v) if not isinstance(v, pd.Series) else v for k, v in data_dict.items()}
    BIAS = pd.Series(BIAS) if not isinstance(BIAS, pd.Series) else BIAS

    # ----- CREATE FIGURE -----
    fig = plt.figure(figsize=options.figsize, dpi=options.dpi)
    gs = GridSpec(2, 1, height_ratios=[8, 4])

    # ----- FIRST SUBPLOT: TIMESERIES -----
    ax1 = fig.add_subplot(gs[0])
    plotter = partial(
        plot_line,
        ax=ax1,
        label_lookup=label_lookup,
        color_palette=options.color_palette,
        line_width=options.line_width
    )
    list(starmap(plotter, data_dict.items()))

    # ----- PLOTTING OPTIONS -----
    ax1.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
    ax1.set_ylabel(f'{options.variable} {options.unit}', fontsize=options.label_fontsize)
    ax1.tick_params(width=2)
    ax1.legend(loc='upper left', fontsize=options.legend_fontsize)
    ax1.grid(True, linestyle='--')
    style_axes_spines(ax1)

    # ----- BIAS TIMESERIES -----
    ax2 = fig.add_subplot(gs[1])
    sns.lineplot(data=BIAS, color='k', ax=ax2)
    
    # ----- PLOTTING OPTIONS -----
    ax2.set_title(f'BIAS ({options.variable_name})', fontsize=options.bias_title_fontsize, fontweight='bold')
    ax2.set_ylabel(f'BIAS {options.unit}', fontsize=options.label_fontsize)
    ax2.tick_params(width=2)
    ax2.grid(True, linestyle='--')
    style_axes_spines(ax2)

    plt.tight_layout()
    
    # ----- CHECK IF FOLDER IS AVAILABLE -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ----- SAVE AND PRINT THE PLOT/S -----
    filename = f'{options.variable_name}_timeseries.png'
    save_path = output_path / filename
    plt.savefig(save_path, **options.savefig_kwargs)
    plt.show(block=False)
    plt.draw()
    plt.close()
###############################################################################
    
###############################################################################
def scatter_plot(data_dict: Dict[str, Union[pd.Series, list]], **kwargs: Any) -> None:
    """
    Generate a scatter plot comparing daily mean values between model and satellite datasets.

    This function creates a single scatter plot showing the relationship between model outputs
    and satellite observations. It also includes an identity line (`y = x`) for reference,
    allowing visual evaluation of model accuracy.

    Parameters
    ----------
    data_dict : Dict[str, Union[pd.Series, list]]
        Dictionary containing model and satellite data.
        Keys should correspond to model and satellite dataset names.
        Values must be 1D arrays or pandas Series.

    Accepted kwargs include:
    -------------------------
        - output_path (str or Path)     : Required. Directory to save figures.
        - variable_name (str)           : Required. Variable code (e.g., 'SST').
        - BA (bool)                     : Whether to append "(Basin Average)" to titles.
        - figsize (tuple of float)      : Figure size (width, height).
        - dpi (int)                     : Figure resolution.
        - color (str)                   : Scatter point color.
        - season_colors (dict)          : Map of season names to colors (for seasonal plots).
        - alpha (float)                 : Transparency of scatter points.
        - marker_size (int)             : Size of scatter markers.
        - title_fontsize (int)          : Font size for plot titles.
        - label_fontsize (int)          : Font size for axis labels.
        - tick_labelsize (int)          : Size of tick labels.
        - line_width (float)            : Width of lines (identity, fits, axes).
        - legend_fontsize (int)         : Size of legend text.
        - pause_time (float)            : Time to pause for each plot (for interactive viewing).
        - variable (str)                : Long name of variable (used in title).
        - unit (str)                    : Unit of variable (used in labels).

    Example
    -------
    >>> scatter_plot(
    ...     data_dict={'model': model_series, 'satellite': sat_series},
    ...     variable_name='SST',
    ...     output_path='figures/',
    ...     BA=True
    ... )
    """

    # ----- RETRIEVE DEFAULT OPTIONS -----
    options = SimpleNamespace(**{**default_plot_options_scatter, **kwargs})

    # ----- RETRIEVE NECESSARY OUTPUT PATH AND VARIABLE/UNIT INFO -----
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    if options.variable_name is not None:
        # Infer full variable name and unit from short name
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    else:
        # variable_name not given — require both variable and unit
        if options.variable is None or options.unit is None:
            raise ValueError(
                "If 'variable_name' is not provided, both 'variable' and 'unit' must be specified in kwargs or defaults."
                )

    # ----- EXTRACT MODEL AND SATELLITE KEYS FROM DATASET FOR PLOTTING -----
    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    BAmod = pd.Series(data_dict[mod_key])
    BAsat = pd.Series(data_dict[sat_key])

    # ----- BUILD FULL DATAFRAME -----
    df = pd.DataFrame({'Model': BAmod, 'Satellite': BAsat})

    # ----- BASIC SEABORN SETTINGS -----
    sns.set(style="whitegrid", context='notebook')
    sns.set_style("ticks")

    # ----- CREATE THE FIGURE -----
    fig = plt.figure(figsize=options.figsize, dpi=options.dpi)
    ax1 = fig.add_subplot(1, 1, 1)

    # ----- CREATE THE MARKERS -----
    sns.scatterplot(
        x='Model',
        y='Satellite',
        data=df,
        color=options.color,
        alpha=options.alpha,
        s=options.marker_size,
        ax=ax1
    )

    # ----- SET THE TITLE AND BA TAG IF NECESSARY -----
    title = f'Scatter Plot of {options.variable} (Model vs. Satellite)'
    if options.BA:
        title += ' (Basin Average)'

    # ----- PLOTTING OPTIONS -----
    ax1.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
    ax1.set_xlabel(f'{options.variable} (Model) {options.unit}', fontsize=options.label_fontsize)
    ax1.set_ylabel(f'{options.variable} (Satellite) {options.unit}', fontsize=options.label_fontsize)

    # ----- EXTRACT AND PLOT IDEAL IDENTITY LINE -----
    min_val, max_val = get_min_max_for_identity_line(df['Model'], df['Satellite'])
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=options.line_width, label='y = x (Ideal)')

    # ----- OTHER FORMATTING OPTIONS -----
    ax1.tick_params(width=2, labelsize=options.tick_labelsize)
    ax1.grid(True, linestyle='--')
    list(starmap(lambda _, spine: (spine.set_linewidth(2), spine.set_edgecolor('black')),
                 enumerate(ax1.spines.values())))
    ax1.legend(fontsize=options.legend_fontsize)

    plt.tight_layout()
    
    # ----- CHECK IF FOLDER IS AVAILABLE -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ----- SAVE AND PRINT PLOT -----
    filename = f'{options.variable_name}_scatterplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(options.pause_time)
    plt.close()
###############################################################################

###############################################################################    
def seasonal_scatter_plot(daily_means_dict: Dict[str, Union[np.ndarray, pd.Series]], **kwargs: Any) -> None:
    """
    Generates seasonal scatter plots (DJF, MAM, JJA, SON) and a combined plot of model vs satellite daily means.

    Each seasonal subplot shows a scatter comparison of model and satellite values, with:
    - Identity line (y = x)
    - Robust linear regression fit (Huber)
    - Nonparametric LOWESS fit

    A final composite plot displays all seasons together, color-coded and with a shared legend.

    Parameters:
    ----------
    daily_means_dict : dict
        Dictionary with keys typically "mod" and "sat", each containing a 1D array or pandas Series of daily means.
        Assumes data starts from 2000-01-01 and is daily.

    Accepted kwargs include:
    -------------------------
        Keyword arguments overriding default plotting options. Include:
        - output_path (str or Path)     : Required. Directory to save figures.
        - variable_name (str)           : Required. Variable code (e.g., 'SST').
        - BA (bool)                     : Whether to append "(Basin Average)" to titles.
        - figsize (tuple of int)        : Figure size (width, height).
        - dpi (int)                     : Figure resolution.
        - season_colors (dict)          : Map of season names to colors. Default is DJF/MAM/JJA/SON.
        - alpha (float)                 : Transparency of scatter points.
        - marker_size (int)             : Size of scatter markers.
        - title_fontsize (int)          : Font size for plot titles.
        - label_fontsize (int)          : Font size for axis labels.
        - line_width (int)              : Width of lines (identity, fits, axes).
        - tick_labelsize (int)          : Size of tick labels.
        - legend_fontsize (int)         : Size of legend text.
        - pause_time (float)            : Time to pause for each plot (for interactive viewing).
        - variable (str)                : Long name of variable (used in title).
        - unit (str)                    : Unit of variable (used in labels).

    Returns:
    -------
    None
        Saves each seasonal plot and one combined plot to the output path.

    Notes:
    -----
    - Assumes data is aligned and continuous from 2000-01-01 onward.
    - Handles NaNs automatically during fitting and plotting.
    - Ideal for visualizing agreement across seasons in long-term climate or model outputs.
    
    Example
    -------
    >>> seasonal_scatter_plot(
    ...     daily_means_dict={'mod': model_series, 'sat': sat_series},
    ...     variable_name='SST',
    ...     output_path='figures/',
    ...     BA=True
    ... )
    """

    # ----- RETRIEVE DEFAULT OPTIONS -----
    options = SimpleNamespace(**{**default_scatter_by_season_options, **kwargs})

    # ----- RETRIEVE NECESSARY OUTPUT PATH AND VARIABLE LABEL/UNIT -----
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    if options.variable_name is not None:
        # Infer full variable name and unit from short name
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    else:
        # variable_name not given — require both variable and unit
        if options.variable is None or options.unit is None:
            raise ValueError(
                "If 'variable_name' is not provided, both 'variable' and 'unit' must be specified in kwargs or defaults."
                )
            
    # ----- ASSIGN DATE RANGE IF NOT AVAILABLE -----
    sample_array = next(iter(daily_means_dict.values()))
    dates = pd.date_range(start="2000-01-01", periods=len(sample_array), freq='D')

    # ----- EXTRACT MODEL AND SATELLITE KEYS -----
    mod_key, sat_key = extract_mod_sat_keys(daily_means_dict)
    # ----- BUILD MODEL/SATELLITE DATASETS -----
    BAmod = np.asarray(daily_means_dict[mod_key])
    BAsat = np.asarray(daily_means_dict[sat_key])

    # ----- RETRIEVE SEASON LIST -----
    seasons = options.season_colors
    
    # ----- CHECK IF THE DIRECTORY IS AVAILABLE, DONE ONCE FOR ALL -----
    # ----- PLOTS IN THE FUNCTION -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ----- INITIALIZE EMPTY ARRAYS -----
    all_mod_points, all_sat_points, all_colors = [], [], []

    # ----- BASIC SEABORN OPTIONS -----
    sns.set(style="whitegrid", context='notebook')
    sns.set_style("ticks")

    # ----- BEGIN PLOTTING THE SEASONAL SCATTERPLOTS -----
    for season_name, color in seasons.items():
        
        # ----- MASK TO RETRIEVE THE SEASON -----
        mask = get_season_mask(dates, season_name)

        # ----- MASK TO USE ONLY VALID POINTS -----
        # ----- BOTH MUST NOT BE NANS -----
        mod_season = BAmod[mask]
        sat_season = BAsat[mask]
        valid = ~np.isnan(mod_season) & ~np.isnan(sat_season)

        # ----- ASSIGN VALID POINTS TO SEASONAL ARRAYS -----
        mod_season = mod_season[valid]
        sat_season = sat_season[valid]

        # ----- CHECK FOR SEASON NOT BEING EMPTY -----
        # ----- OTHERWISE SKIP THE SEASON -----
        if mod_season.size == 0:
            print(f"Skipping {season_name}: no valid data.")
            continue

        # USE MODEL/SATELLITE/SEASON INFO TO BUILD THE DATAFRAME -----
        df = pd.DataFrame({
            'Model': mod_season,
            'Satellite': sat_season,
            'Season': [season_name] * len(mod_season)
        })

        # ----- INITIALIZE FIGURE -----
        plt.figure(figsize=options.figsize, dpi=options.dpi)
        
        # ----- PLOT -----
        ax = sns.scatterplot(
            x='Model', y='Satellite', data=df,
            color=color, alpha=options.alpha, s=options.marker_size,
            label=f'{options.variable} {season_name}'
        )

        # ----- PERFORM THE HUBER REGRESSION FIT -----
        x_vals, y_vals = fit_huber(mod_season, sat_season)
        ax.plot(x_vals, y_vals, color='black', linestyle='-', linewidth=options.line_width, label='Linear Fit (Huber)')

        # ----- PERFORM THE LOWESS REGRESSION FIT -----
        smoothed = fit_lowess(mod_season, sat_season)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='magenta', linestyle='-.', linewidth=options.line_width, label='Smoothed Fit (LOWESS)')

        # ----- PERFORM THE IDEAL FIT -----
        min_val, max_val = get_min_max_for_identity_line(mod_season, sat_season)
        ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=options.line_width, label='Ideal Fit')

        # ----- TITLE AND BASIN AVERAGE TAG IF USER SPECIFIED IT -----
        # ----- BASIN AVERAGE STRONGLY SUGGESTED TO MARK L4 SATELLITE DATA -----
        title = f'{options.variable} Scatter Plot (Model vs Satellite) - {season_name}'
        if options.BA:
            title += ' (Basin Average)'

        # ----- PLOT FORMATTING -----
        ax.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
        ax.set_xlabel(f'{options.variable} (Model - {season_name}) {options.unit}', fontsize=options.label_fontsize)
        ax.set_ylabel(f'{options.variable} (Satellite - {season_name}) {options.unit}', fontsize=options.label_fontsize)

        ax.legend(fontsize=options.legend_fontsize)
        ax.tick_params(axis='both', labelsize=options.tick_labelsize, width=options.line_width)

        style_axes_spines(ax, linewidth=options.line_width, edgecolor='black')

        ax.grid(True, linestyle='--')
        plt.tight_layout()

        # ----- SAVING AND PRINTING THE PLOT -----
        filename = f"{options.variable_name}_{season_name}_scatterplot.png"
        plt.savefig(output_path / filename)
        plt.show(block=False)
        plt.draw()
        plt.pause(options.pause_time)
        plt.close()

        # ----- APPEND THE DATA TO FINAL PLOT DATAFRAME -----
        all_mod_points.extend(mod_season)
        all_sat_points.extend(sat_season)
        all_colors.extend([season_name] * len(mod_season))
        
    # ----- CONVERT POINTS INTO NP.ARRAY TO BE USED FOR REGRESSION LINES -----
    all_mod_points = np.array(all_mod_points)
    all_sat_points = np.array(all_sat_points)

    # ----- SWICTH TO SUMMARY PLOT -----
    # ----- DONE ONLY IF THE CONTENT OF THE ALL_POINT IS NOT NONE -----
    if len(all_mod_points) > 0:
        
        # ----- INITIALIZE FIGURE -----
        plt.figure(figsize=options.figsize, dpi=options.dpi)
        
        # ----- BUILD COMBINED DATAFRAME -----
        df_combined = pd.DataFrame({
            'Model': all_mod_points,
            'Satellite': all_sat_points,
            'Season': all_colors
        })

        # ----- PLOT MARKERS -----
        ax = sns.scatterplot(
            x='Model', y='Satellite', data=df_combined,
            hue='Season', palette=seasons,
            alpha=options.alpha, s=options.marker_size
        )

        # ----- PERFORM IDEAL REGRESSION FIT -----
        min_val, max_val = get_min_max_for_identity_line(all_mod_points, all_sat_points)
        ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=options.line_width, label='Ideal Fit')

        # ----- PERFORM HUBER REGRESSION FIT -----
        x_vals, y_vals = fit_huber(all_mod_points, all_sat_points)
        ax.plot(x_vals, y_vals, color='black', linestyle='-', linewidth=options.line_width, label='Linear Fit (Huber)')

        # ----- PERFORM LOWESS FIT -----
        smoothed = fit_lowess(all_mod_points, all_sat_points)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='magenta', linestyle='-.', linewidth=options.line_width, label='Smoothed Fit (LOWESS)')

        # ----- BUILD HANDLES FOR LEGEND -----
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=season)
            for season, color in seasons.items()
        ] + [
            Line2D([0], [0], color='b', linestyle='--', linewidth=options.line_width, label='Ideal Fit'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=options.line_width, label='Linear Fit (Huber)'),
            Line2D([0], [0], color='magenta', linestyle='-.', linewidth=options.line_width, label='Smoothed Fit (LOWESS)')
        ]
        ax.legend(handles=handles, fontsize=options.legend_fontsize, loc='upper left')

        # ----- FORMATTING OPTIONS -----
        ax.set_title(f'{options.variable} Scatter Plot (Model vs Satellite) - All Seasons', fontsize=options.title_fontsize, fontweight='bold')
        ax.set_xlabel(f'{options.variable} (Model - All Seasons) {options.unit}', fontsize=options.label_fontsize)
        ax.set_ylabel(f'{options.variable} (Satellite - All Seasons) {options.unit}', fontsize=options.label_fontsize)
        ax.tick_params(axis='both', labelsize=options.tick_labelsize, width=options.line_width)

        style_axes_spines(ax, linewidth=options.line_width, edgecolor='black')

        ax.grid(True, linestyle='--')
        plt.tight_layout()

        # ----- SAVE AND PLOT -----
        filename = f"{options.variable_name}_all_seasons_scatterplot.png"
        plt.savefig(output_path / filename)
        plt.show(block=False)
        plt.draw()
        plt.pause(options.pause_time)
        plt.close()
###############################################################################
        
###############################################################################
def whiskerbox(data_dict, **kwargs):
    """
    Create a boxplot comparing monthly values of model and satellite data.

    This function plots side-by-side boxplots for each month, showing model vs. satellite distributions.
    It's useful for visualizing variability and central tendency over time.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing time-series data for model and satellite.
        Keys must distinguish between model and satellite (e.g., 'model', 'satellite').

    **kwargs : keyword arguments
        Keyword arguments overriding default plotting options. Include:
        - output_path (str or Path)       : Required. Directory to save the resulting PNG plot.
        - variable_name (str)             : Required. Variable short name (e.g., 'SST').
        - variable (str)                  : Full variable name (e.g., 'Sea Surface Temperature').
        - unit (str)                      : Unit of the variable (e.g., '°C').
        - figsize (tuple of float)        : Figure size in inches, e.g., (14, 8).
        - dpi (int)                       : Plot resolution in dots per inch.
        - palette (str or list)           : Seaborn-compatible color palette.
        - showfliers (bool)               : Whether to show outlier points in the boxplot.
        - title_fontsize (int)            : Font size of the plot title.
        - title_fontweight (str)          : Font weight of the plot title (e.g., 'bold').
        - ylabel_fontsize (int)           : Font size of the y-axis label.
        - xlabel (str)                    : Label for the x-axis (default: '').
        - grid_alpha (float)              : Transparency for grid lines.
        - xtick_rotation (int or float)   : Rotation angle for x-axis tick labels.
        - tick_width (float)              : Width of axis ticks.
        - pause_time (float)              : Time (seconds) to pause the plot after showing it.

    Example
    -------
    >>> whiskerbox(
    ...     data_dict={'model': model_series, 'satellite': sat_series},
    ...     variable_name='Chl',
    ...     output_path='figures/',
    ...     figsize=(14, 8),
    ...     palette='Set2',
    ...     showfliers=False
    ... )
    """
    
    # ----- FETCH DEFAULT OPTIONS -----
    options = SimpleNamespace(**{**default_boxplot_options, **kwargs})

    # ----- RETRIEVE NECESSARY OUTPUT PATH AND VARIABLE/UNIT INFO -----
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    if options.variable_name is not None:
        # Infer full variable name and unit from short name
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    else:
        # variable_name not given — require both variable and unit
        if options.variable is None or options.unit is None:
            raise ValueError(
                "If 'variable_name' is not provided, both 'variable' and 'unit' must be specified in kwargs or defaults."
                )
            
    # ----- OBRAIN VARIABLE LABEL AND UNITS FROM THE VARIABLE NAME -----
    variable, unit = get_variable_label_unit(options.variable_name)

    # ----- EXTRACT MODEL AND SATELLITE KEYS FROM THE DATASET -----
    model_key, sat_key = extract_mod_sat_keys(data_dict)

    # ----- DEFINE MONTH NAMES -----
    months = [calendar.month_abbr[i] for i in range(1, 13)]


    # ----- INITIALIZE ARRAY FOR DATA -----
    plot_data = []
    
    # ----- GATHER THE DATA BASED ON MONTHS -----
    plot_data = list(chain.from_iterable(
        chain(
            ((val, f"{months[month_idx]} Model") for val in gather_monthly_data_across_years(data_dict, model_key, month_idx)),
            ((val, f"{months[month_idx]} Satellite") for val in gather_monthly_data_across_years(data_dict, sat_key, month_idx))
        )
        for month_idx in range(12)
    ))

    # ----- BUILD THE DATAFRAME -----
    plot_df = pd.DataFrame(plot_data, columns=['Value', 'Label'])

    # ----- INITIALIZE FIGURE -----
    plt.figure(figsize=options.figsize, dpi=options.dpi)
    
    # ----- PLOT DATA -----
    ax = sns.boxplot(
        x='Label', y='Value', data=plot_df,
        palette=options.palette,
        showfliers=options.showfliers
    )

    # ----- FORMATTING AND PLOT OPTIONS
    ax.set_title(f'Monthly {variable} Comparison: Model vs Satellite',
                 fontsize=options.title_fontsize,
                 fontweight=options.title_fontweight)
    ylabel = f'{variable} {unit}'
    ax.set_ylabel(ylabel, fontsize=options.ylabel_fontsize)
    ax.set_xlabel(options.xlabel)
    ax.grid(True, linestyle='--', alpha=options.grid_alpha)
    plt.xticks(rotation=options.xtick_rotation)
    ax.tick_params(width=options.tick_width)

    style_axes_spines(ax, linewidth=2, edgecolor='black')

    plt.tight_layout()

    # ----- CHECK IF FOLDER EXISTS -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ----- PRINT AND SAVE PLOT ------
    filename = f'{variable}_boxplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(options.pause_time)
    plt.close()
###############################################################################
    
###############################################################################
def violinplot(data_dict, **kwargs):
    """
    Plot a violin plot comparing monthly model and satellite values.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data indexed by datetime.

    Keyword Arguments
    -----------------
    - output_path (str or Path)       : Required. Directory where the figure is saved.
    - variable_name (str)             : Short name used to infer full variable name and unit.
    - variable (str)                  : Full variable name (e.g., "Chlorophyll"). Used in the title.
    - unit (str)                      : Unit of measurement (e.g., "mg Chl/m³"). Shown on the y-axis.
    - figsize (tuple of float)        : Size of the figure in inches.
    - dpi (int)                       : Resolution of the plot.
    - palette (list or dict)          : Colors for the violin plots.
    - cut (float)                     : Defines how far the violin extends past extreme datapoints.
    - title_fontsize (int)           : Font size of the title.
    - title_fontweight (str or int)   : Font weight of the title (e.g., 'bold').
    - ylabel_fontsize (int)           : Font size of the y-axis label.
    - xlabel_fontsize (int)           : Font size of the x-axis label.
    - xtick_rotation (int)            : Degree of x-tick label rotation.
    - tick_width (float)              : Width of axis ticks.
    - spine_linewidth (float)         : Line width for axis spines.
    - grid_alpha (float)              : Transparency of grid lines.
    - pause_time (float)              : Time in seconds to pause after plotting.

    Example
    -------
    >>> violinplot(
    ...     data_dict={'model': model_series, 'satellite': sat_series},
    ...     variable_name='SST',
    ...     output_path='figures/',
    ...     figsize=(12, 6)
    ... )
    """
    
    # ----- FETCH DEFAULT OPTIONS -----
    options = SimpleNamespace(**{**default_violinplot_options, **kwargs})

    # ----- OBTAIN NECESSARY VALUES -----
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    if options.variable_name is not None:
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    elif options.variable is None or options.unit is None:
        raise ValueError("If 'variable_name' is not provided, both 'variable' and 'unit' must be specified.")

    # ----- FETCH MOD AND SAT KEYS FROM DICTIONARY -----
    model_key, sat_key = extract_mod_sat_keys(data_dict)
    
    # ----- DEFINE MONTH NAMES -----
    months = [calendar.month_abbr[i] for i in range(1, 13)]

    # ----- GATHER DATA -----
    plot_data = list(chain.from_iterable(
        chain(
            ((val, f"{months[month_idx]} Model") for val in gather_monthly_data_across_years(data_dict, model_key, month_idx)),
            ((val, f"{months[month_idx]} Satellite") for val in gather_monthly_data_across_years(data_dict, sat_key, month_idx))
        )
        for month_idx in range(12)
    ))

    # ----- SWITCH TO PANDAS DATAFRAME -----
    plot_df = pd.DataFrame(plot_data, columns=['Value', 'Label'])

    # ----- DEFINE FIGURE ------
    plt.figure(figsize=options.figsize, dpi=options.dpi)
    
    # ----- PLOT DATA -----
    ax = sns.violinplot(x='Label', y='Value', data=plot_df,
                        palette=options.palette, cut=options.cut)

    # ----- FORMATTING THE PLOT -----
    ax.set_title(f'Monthly {options.variable} Comparison: Model vs Satellite',
                 fontsize=options.title_fontsize, fontweight=options.title_fontweight)
    ax.set_ylabel(f'{options.variable} {options.unit}', fontsize=options.ylabel_fontsize)
    ax.set_xlabel('', fontsize=options.xlabel_fontsize)
    ax.grid(True, linestyle='--', alpha=options.grid_alpha)
    plt.xticks(rotation=options.xtick_rotation)
    ax.tick_params(width=options.tick_width)

    style_axes_spines(ax, linewidth=2, edgecolor='black')

    plt.tight_layout()

    # ----- CHECK IF PROVIDED FOLDER EXISTS -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ----- PRINT THE PLOT AND SAVE -----
    filename = f'{options.variable}_violinplot.png'
    plt.savefig(output_path / filename)
    plt.show(block=False)
    plt.draw()
    plt.pause(options.pause_time)
    plt.close()
###############################################################################    

###############################################################################
def efficiency_plot(total_value, monthly_values, **kwargs):
    """
    Plot efficiency metric (e.g., NSE) for each month with color-coded markers.

    Parameters
    ----------
    total_value : float
        The efficiency value computed over the full time period (used as reference line).

    monthly_values : list of float
        Efficiency values for each month (12 total).

    Keyword Arguments
    -----------------
    - output_path (str or Path)        : Required. Directory where the figure is saved.
    - metric_name (str)                : Required. Used for the filename (e.g., "NSE").
    - title (str)                      : Title of the plot (e.g., "Nash-Sutcliffe Efficiency").
    - y_label (str)                    : LaTeX-formatted y-axis label (e.g., "$E_{rel}$").
    - figsize (tuple of float)         : Size of the figure in inches.
    - dpi (int)                        : Resolution of the plot.
    - line_color (str)                 : Color of the line connecting monthly points.
    - line_width (float)               : Width of the connecting line.
    - marker_size (float)              : Size of circular markers.
    - marker_edge_color (str)          : Color of marker edge.
    - marker_edge_width (float)        : Width of marker edge.
    - xtick_rotation (int)             : Degree of x-tick label rotation.
    - tick_width (float)              : Width of axis ticks.
    - spine_width (float)              : Width of axis spines.
    - legend_loc (str)                 : Location of the legend.
    - grid_style (str)                 : Style of grid lines (e.g., "--", ":").
    - pause_time (float)               : Time in seconds to pause after plotting.
    - zero_line (dict)                 : Zero reference line options:
                                         {
                                             "show": bool,
                                             "style": str,
                                             "width": float,
                                             "color": str,
                                             "label": str
                                         }
    - overall_line (dict)              : Overall mean line options:
                                         {
                                             "style": str,
                                             "width": float,
                                             "color": str,
                                             "label": str
                                         }
    """
    
    # ----- FETCH DEFAULT OPTIONS -----
    options = SimpleNamespace(**{**default_efficiency_plot_options, **kwargs})

    # ----- RETREIVE NECESSARY OUTPUT PATH -----
    if options.output_path is None:
        raise ValueError("output_path must be specified.")

    # ----- GET MONTHS -----
    months = list(calendar.month_name[1:13])
    
    # ----- BUILD THE DATAFRAME -----
    df = pd.DataFrame({'Month': months, 'Value': monthly_values})

    # ----- BUILD COLORMAP -----
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # ----- COLORMAP THRESHOLD CASES -----
    marker_colors = [
        'gray' if not isinstance(val, (int, float)) else
        'red' if val < 0 else
        'green' if val > 1 else
        cmap(norm(val))
        for val in monthly_values
    ]

    # ----- SEABORN SETUP -----
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    
    # ----- INITIALIZE FIGURE -----
    plt.figure(figsize=options.figsize, dpi=options.dpi)

    # ----- PLOT PERFORMANCE POINT -----
    ax = sns.lineplot(x='Month', y='Value', data=df,
                      color=options.line_color,
                      lw=options.line_width)

    # ----- LIST OF METRICS USING ZERO THRESHOLD -----
    if options.zero_line.get("show", False) and options.title in {
        "Nash-Sutcliffe Efficiency",
        "Nash-Sutcliffe Efficiency (Logarithmic)",
        "Modified NSE ($E_1$, j=1)",
        "Relative NSE ($E_{rel}$)"
    }:
        # ----- SET ZERO THRESHOLD LINE -----
        ax.axhline(0,
                   linestyle=options.zero_line["style"],
                   lw=options.zero_line["width"],
                   color=options.zero_line["color"],
                   label=options.zero_line["label"])

    # ----- PLOT OVERALL VALUE -----
    ax.axhline(total_value,
               linestyle=options.overall_line["style"],
               lw=options.overall_line["width"],
               color=options.overall_line["color"],
               label=options.overall_line["label"])

    # ----- PLOT THE MARKER -----
    for month, value, color in itertools.zip_longest(months, monthly_values, marker_colors):
        ax.plot(month, value, marker='o',
                markersize=options.marker_size,
                color=color,
                markeredgecolor=options.marker_edge_color,
                markeredgewidth=options.marker_edge_width)

    # ----- PLOT FORMATTING -----
    ax.set_title(options.title, fontsize=options.title_fontsize)
    ax.set_xlabel('')
    ax.set_ylabel(f'${options.y_label}$', fontsize=options.ylabel_fontsize)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=options.xtick_rotation)
    ax.tick_params(width=options.tick_width)
    ax.legend(loc=options.legend_loc)
    ax.grid(True, linestyle=options.grid_style)

    style_axes_spines(ax, linewidth=2, edgecolor='black')

    plt.tight_layout()
    
    # ----- CHECK EXISTENCE OF THE SAVE FOLDER -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ----- PRINT AND SAVE -----
    plt.savefig(output_path / f'{options.metric_name}.png')
    plt.show(block=False)
    plt.draw()
    plt.pause(options.pause_time)
    plt.close()
###############################################################################   

###############################################################################   
def plot_spatial_efficiency(data_array, geo_coords, output_path,
                            title_prefix, cmap, vmin, vmax,
                            detrended=False, suffix="(Model - Satellite)"):
    """
    Plot monthly or yearly spatial efficiency maps with shared color scale and Cartopy projection.

    Parameters
    ----------
    data_array : xarray.DataArray
        3D array with dims (month/year, y, x) to plot.
    geo_coords : dict
        Dictionary with keys: 'latp', 'lonp', 'MinLambda', 'MaxLambda',
        'MinPhi', 'MaxPhi', and optional 'Epsilon'.
    output_path : str or Path
        Directory to save the plot.
    title_prefix : str
        Prefix for subplot titles.
    cmap : str or Colormap
        Colormap for plotting.
    vmin, vmax : float
        Global colorbar limits.
    detrended : bool, optional
        Whether data is detrended.
    suffix : str, optional
        Suffix in the overall plot title.

    Returns
    -------
    None
    """

    latp = geo_coords['latp']
    lonp = geo_coords['lonp']
    epsilon = geo_coords.get('Epsilon', 0.06)
    min_lon, max_lon = geo_coords['MinLambda'], geo_coords['MaxLambda']
    min_lat, max_lat = geo_coords['MinPhi'], geo_coords['MaxPhi']
    lat_offset = epsilon + 0.2702044

    # Determine if data is monthly or yearly
    if 'month' in data_array.dims:
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        dim_name = 'month'
        n_plots = data_array.sizes['month']
        suptitle_prefix = "Monthly"
        filename_prefix = "Monthly"
    elif 'year' in data_array.dims:
        labels = data_array.year.values.astype(str)
        dim_name = 'year'
        n_plots = data_array.sizes['year']
        suptitle_prefix = "Yearly"
        filename_prefix = "Yearly"
    else:
        raise ValueError("data_array must have either 'month' or 'year' dimension.")

    max_cols = 3
    nrows = int(np.ceil(n_plots / max_cols))
    remainder = n_plots % max_cols
    full_rows = n_plots // max_cols

    fig = plt.figure(figsize=(5 * max_cols, 4 * nrows), constrained_layout=True)
    gs = GridSpec(nrows, max_cols, figure=fig)
    axes = []

    for row in range(nrows):
        if row < full_rows:
            for col in range(max_cols):
                ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
                axes.append(ax)
        else:  # Last row with fewer plots
            if remainder == 1:
                ax = fig.add_subplot(gs[row, 1], projection=ccrs.PlateCarree())
                axes.append(ax)
            elif remainder == 2:
                ax1 = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree())
                ax2 = fig.add_subplot(gs[row, 2], projection=ccrs.PlateCarree())
                axes.extend([ax1, ax2])

    # Custom diverging colormap
    if cmap == 'OrangeGreen':
        colors = ['#086e04', 'white', '#ff6700']
        base_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        white_start = int(norm(-0.2) * 255)
        white_end = int(norm(0.2) * 255)
        new_colors = base_cmap(np.linspace(0, 1, 256))
        new_colors[white_start:white_end, :] = np.array([1, 1, 1, 1])
        cmap = mcolors.ListedColormap(new_colors)

    contour_levels = np.linspace(vmin, vmax, 11)

    for i, ax in enumerate(axes):
        data_slice = data_array.isel({dim_name: i})
        ax.set_extent([min_lon, max_lon, min_lat + lat_offset, max_lat], crs=ccrs.PlateCarree())
        contour = ax.contourf(
            lonp + (0.35 * epsilon), latp + (0.1 * epsilon), data_slice,
            levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax,
            extend='both', transform=ccrs.PlateCarree()
        )
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        label = labels[i] if i < len(labels) else f"{dim_name.capitalize()} {i+1}"
        ax.set_title(f"{title_prefix} - {label}", fontsize=16, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")
        gl = ax.gridlines(draw_labels=True, dms=True, color='gray', linestyle='--', alpha=0.7)
        gl.top_labels = True
        gl.right_labels = True

    cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', shrink=0.6,
                        ticks=np.linspace(vmin, vmax, 11))
    cbar.ax.tick_params(direction='in', length=32, labelsize=12)
    cbar.set_label(f"{title_prefix}", fontsize=18, labelpad=10)

    det_text = "Detrended" if detrended else "Raw"
    plt.suptitle(f"{suptitle_prefix} {title_prefix} ({det_text}) {suffix}", fontsize=20, fontweight='bold')

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_title = title_prefix.replace("/", "_").replace("\\", "_")
    plt.savefig(output_path / f'{filename_prefix} {safe_title} ({det_text}) {suffix}.png')
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
###############################################################################   