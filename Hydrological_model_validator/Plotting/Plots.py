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
from itertools import starmap, chain, cycle
from functools import partial

# Plotting Libraries
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.signal import csd

# Module imports
from .formatting import (plot_line,
                        get_min_max_for_identity_line,
                        get_variable_label_unit,
                        style_axes_spines,
                        format_unit)

from ..Processing.data_alignment import (extract_mod_sat_keys,
                                         gather_monthly_data_across_years)

from ..Processing.stats_math_utils import (fit_huber,
                                           fit_lowess,
                                           )

from ..Processing.time_utils import get_season_mask
from ..Processing.utils import extract_options

from .default_plot_options import (default_plot_options_ts,
                                   default_plot_options_scatter,
                                   default_scatter_by_season_options,
                                   default_boxplot_options,
                                   default_violinplot_options,
                                   default_efficiency_plot_options,
                                   spatial_efficiency_defaults,
                                   default_error_timeseries_options,
                                   default_spectral)

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def timeseries(data_dict: Dict[str, Union[pd.Series, list]], BIAS: Union[pd.Series, list, None], **kwargs: Any) -> None:
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
        - unit (str)                      : Unit of measurement (e.g., "mg Chl/m³"). Displayed on axis.
        - BA (bool)                       : If True, appends " (Basin Average)" to the title.
        - figsize (tuple of float)        : Size of figure in inches (default typically (12, 8)).
        - dpi (int)                       : Resolution of the figure (default 100).
        - color_palette (iterator)        : Iterator of colors (e.g., `itertools.cycle(sns.color_palette("tab10"))`).
        - line_width (float)              : Width of plotted lines (default 2.0).
        - title_fontsize (int)            : Font size of the main title.
        - bias_title_fontsize (int)       : Font size of the BIAS subplot title.
        - label_fontsize (int)            : Font size of axis labels.
        - legend_fontsize (int)           : Font size of the legend.
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
    # ----- RETRIEVE DEFAULT OPTIONS -----
    options = SimpleNamespace(**{**default_plot_options_ts, **kwargs})

    # ----- REQUIRED PARAMS CHECK -----
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    if options.variable_name is not None:
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    else:
        if options.variable is None or options.unit is None:
            raise ValueError("If 'variable_name' is not provided, both 'variable' and 'unit' must be specified.")

    # ----- BASIN AVERAGE LABEL -----
    title = f'Daily Mean Values for {options.variable} Datasets'
    if options.BA:
        title += ' (Basin Average)'

    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    label_lookup = {mod_key: "Model Output", sat_key: "Satellite Observations"}

    # ----- CONVERT INPUTS TO SERIES -----
    data_dict = {k: pd.Series(v) if not isinstance(v, pd.Series) else v for k, v in data_dict.items()}
    if BIAS is not None:
        BIAS = pd.Series(BIAS) if not isinstance(BIAS, pd.Series) else BIAS

    # ----- FIGURE SETUP -----
    if BIAS is not None:
        fig = plt.figure(figsize=options.figsize, dpi=options.dpi)
        gs = GridSpec(2, 1, height_ratios=[8, 4])
        ax1 = fig.add_subplot(gs[0])
    else:
        fig = plt.figure(figsize=(options.figsize[0], options.figsize[1] * 0.6), dpi=options.dpi)
        gs = GridSpec(1, 1)
        ax1 = fig.add_subplot(gs[0])

    # ----- MAIN TIMESERIES PLOT -----
    plotter = partial(
        plot_line,
        ax=ax1,
        label_lookup=label_lookup,
        color_palette=options.color_palette,
        line_width=options.line_width,
        library='plt',
    )
    list(starmap(plotter, data_dict.items()))

    ax1.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
    ax1.set_ylabel(f'{options.variable} {options.unit}', fontsize=options.label_fontsize)
    ax1.tick_params(width=2)
    ax1.legend(loc='upper left', fontsize=options.legend_fontsize)
    ax1.grid(True, linestyle='--')
    style_axes_spines(ax1)

    # ----- OPTIONAL BIAS PLOT -----
    if BIAS is not None:
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(BIAS.index, BIAS.values, color='k')

        ax2.set_title(f'BIAS ({options.variable})', fontsize=options.bias_title_fontsize, fontweight='bold')
        ax2.set_ylabel(f'BIAS {options.unit}', fontsize=options.label_fontsize)
        ax2.tick_params(width=2)
        ax2.grid(True, linestyle='--')
        style_axes_spines(ax2)

    plt.tight_layout()

    # ----- SAVE FIGURE -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f'{options.variable}_timeseries.png'
    save_path = output_path / filename
    plt.savefig(save_path, **options.savefig_kwargs)
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
    if getattr(options, 'output_path', None) is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    variable_name = getattr(options, 'variable_name', None)
    variable = getattr(options, 'variable', None)
    unit = getattr(options, 'unit', None)

    if variable_name is not None:
        # Infer full variable name and unit from short name
        variable_label, unit_label = get_variable_label_unit(variable_name)
        options.variable = variable or variable_label
        options.unit = unit or unit_label
    else:
        # variable_name not given — require both variable and unit
        if variable is None or unit is None:
            raise ValueError(
                "If 'variable_name' is not provided, both 'variable' and 'unit' must be specified in kwargs or defaults."
            )
        options.variable = variable
        options.unit = unit

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
    if getattr(options, 'BA', False):
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
    filename = f'{variable_name or options.variable or "scatterplot"}_scatterplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
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
        filename = f"{options.variable}_{season_name}_scatterplot.png"
        plt.savefig(output_path / filename)
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
        filename = f"{options.variable}_all_seasons_scatterplot.png"
        plt.savefig(output_path / filename)
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

    variable_name = getattr(options, 'variable_name', None)
    variable = getattr(options, 'variable', None)
    unit = getattr(options, 'unit', None)

    if variable_name is not None:
        var, un = get_variable_label_unit(variable_name)
        variable = variable or var
        unit = unit or un
    else:
        if variable is None or unit is None:
            raise ValueError(
                "If 'variable_name' is not provided, both 'variable' and 'unit' must be specified in kwargs or defaults."
            )

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

    # ----- PRINT AND SAVE PLOT -----
    filename = f'{variable}_boxplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
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
    - title_fontsize (int)            : Font size of the title.
    - title_fontweight (str or int)   : Font weight of the title (e.g., 'bold').
    - ylabel_fontsize (int)           : Font size of the y-axis label.
    - xlabel_fontsize (int)           : Font size of the x-axis label.
    - xtick_rotation (int)            : Degree of x-tick label rotation.
    - tick_width (float)              : Width of axis ticks.
    - spine_linewidth (float)         : Line width for axis spines.
    - grid_alpha (float)              : Transparency of grid lines.

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
    - tick_width (float)               : Width of axis ticks.
    - spine_width (float)              : Width of axis spines.
    - legend_loc (str)                 : Location of the legend.
    - grid_style (str)                 : Style of grid lines (e.g., "--", ":").
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

    # --- OPTIONS VALIDATION ---
    if options.output_path is None:
        raise ValueError("output_path must be specified.")
    if options.metric_name is None:
        raise KeyError("metric_name must be specified.")
    if options.y_label is None:
        raise KeyError("y_label must be specified.")

    # --- PREPARE DATA ---
    months = list(calendar.month_name[1:13])
    df = pd.DataFrame({'Month': months, 'Value': monthly_values})

    # --- CMAP ---
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    marker_colors = [
        'gray' if not isinstance(val, (int, float)) else
        'red' if val < 0 else
        'green' if val > 1 else
        cmap(norm(val))
        for val in monthly_values
    ]

    # --- SNS/FIGURE SETUP  ---
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    plt.figure(figsize=options.figsize, dpi=options.dpi)

    ax = sns.lineplot(x='Month', y='Value', data=df,
                      color=options.line_color,
                      lw=options.line_width)

    # --- XERO LINE PLOT ---
    if options.zero_line.get("show", False) and options.title in {
        "Nash-Sutcliffe Efficiency",
        "Nash-Sutcliffe Efficiency (Logarithmic)",
        "Modified NSE ($E_1$, j=1)",
        "Relative NSE ($E_{rel}$)"
    }:
        ax.axhline(0,
                   linestyle=options.zero_line["style"],
                   lw=options.zero_line["width"],
                   color=options.zero_line["color"],
                   label=options.zero_line["label"])

    # --- OVERALL LINE PLOT ---
    ax.axhline(total_value,
               linestyle=options.overall_line["style"],
               lw=options.overall_line["width"],
               color=options.overall_line["color"],
               label=options.overall_line["label"])

    # --- MARKERS ---
    for month, value, color in itertools.zip_longest(months, monthly_values, marker_colors):
        if value is not None:
            ax.plot(month, value, marker='o',
                    markersize=options.marker_size,
                    color=color,
                    markeredgecolor=options.marker_edge_color,
                    markeredgewidth=options.marker_edge_width)

    # --- FORMATTING ---
    ax.set_title(options.title, fontsize=options.title_fontsize)
    ax.set_xlabel('')
    ax.set_ylabel(f'${options.y_label}$', fontsize=options.ylabel_fontsize)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=options.xtick_rotation)
    ax.tick_params(width=options.tick_width)
    ax.legend(loc=options.legend_loc)
    ax.grid(True, linestyle=options.grid_style)

    style_axes_spines(ax, linewidth=options.spine_width, edgecolor='black')

    plt.tight_layout()

    # ----- PRINT AND SAVE -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path / f'{options.metric_name}.png')
    plt.close()
    
###############################################################################   

###############################################################################
   
def plot_spatial_efficiency(data_array, geo_coords, output_path, title_prefix, **kwargs):
    """
    Plot spatial efficiency metric maps (e.g., correlation, NSE) by month or year with Cartopy projection.

    This function generates a grid of spatial maps (e.g., for each month or year) showing the
    spatial distribution of a performance metric such as correlation or NSE between model and
    satellite data. It supports extensive customization via keyword arguments and a centralized
    defaults dictionary.

    Parameters
    ----------
    data_array : xarray.DataArray
        3D data with shape (month/year, lat, lon). Must contain either a 'month' or 'year' dimension.

    geo_coords : dict
        Dictionary containing:
            - 'latp' (2D array): Latitude grid.
            - 'lonp' (2D array): Longitude grid.
            - 'MinLambda', 'MaxLambda' (float): Longitude bounds.
            - 'MinPhi', 'MaxPhi' (float): Latitude bounds.
            - 'Epsilon' (float, optional): Spatial padding offset (used for label adjustment).

    output_path : str or Path
        Directory where the figure will be saved.

    title_prefix : str
        Title prefix for the colorbar and each subplot (e.g., "Correlation").

    Keyword Arguments
    -----------------
    - cmap (str or Colormap)           : Colormap to use (e.g., "coolwarm").
    - vmin, vmax (float)               : Min and max values for colorbar.
    - suffix (str)                     : Suffix for plot title and filename.
    - suptitle_fontsize (int)          : Font size of the super title (if the 
                                            resulting number of columns is 1
                                            the value will be reduced by 6).
    - suptitle_fontweight (str)        : Font weight of the super title.
    - suptitle_y (float)               : Vertical position of the super title.
    - title_fontsize (int)             : Font size of subplot titles.
    - title_fontweight (str)           : Font weight of subplot titles.
    - cbar_labelsize (int)             : Font size of colorbar tick labels.
    - cbar_labelpad (int)              : Padding between colorbar and label.
    - cbar_shrink (float)              : Shrink factor for horizontal colorbar.
    - cbar_ticks (int)                 : Number of colorbar ticks.
    - figsize_per_plot (tuple)         : Size per subplot (width, height).
    - max_cols (int)                   : Max number of columns in subplot grid.
    - epsilon (float)                  : Padding fallback if not in geo_coords.
    - lat_offset_base (float)          : Extra lat offset if needed.
    - gridline_color (str)             : Color of gridlines.
    - gridline_style (str)             : Line style of gridlines (e.g., "--").
    - gridline_alpha (float)           : Gridline transparency.
    - gridline_dms (bool)              : Format labels in DMS (deg:min:sec).
    - gridline_labels_top (bool)       : Show labels on top axis.
    - gridline_labels_right (bool)     : Show labels on right axis.
    - projection (str)                 : Cartopy projection class name.
    - resolution (str)                 : Resolution of coastlines (e.g., "10m").
    - land_color (str)                 : Color for landmasses.
    - show (bool)                      : Display the plot interactively.
    - block (bool)                     : Block execution on plt.show().
    - dpi (int)                        : Resolution of the output figure.
    
    Raises
    ------
    ValueError
        If the `data_array` does not contain a 'month' or 'year' dimension.

    Examples
    --------
    >>> plot_spatial_efficiency(data_array, geo_coords, "figures", "Correlation",
    ...                         cmap="coolwarm", vmax=1.0, vmin=-1.0, show=True)
    """
    
    # ----- GET DEFAULT OPTIONS -----
    options = extract_options(kwargs, spatial_efficiency_defaults)

    # ----- SET GEOMETRY -----
    latp = geo_coords['latp']
    lonp = geo_coords['lonp']
    epsilon = geo_coords.get("Epsilon", options["epsilon"])
    lat_offset = epsilon + options["lat_offset_base"]
    min_lon = geo_coords['MinLambda']
    max_lon = geo_coords['MaxLambda']
    min_lat = geo_coords['MinPhi']
    max_lat = geo_coords['MaxPhi']

    # ----- FETCH THE TIME LABELS -----
    if 'month' in data_array.dims:
        labels = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
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

    # ----- COMPUTE OPTIMAL GRID LAYOUT -----
    nrows = int(np.ceil(np.sqrt(n_plots)))
    ncols = int(np.ceil(n_plots / nrows))


    # ----- SETUP THE FIGURE AND SIZE -----
    figsize = ((options["figsize_per_plot"][0] if ncols !=1 else options["figsize_per_plot"][0] + 3) * ncols,
               (options["figsize_per_plot"][1] if ncols !=1 else options["figsize_per_plot"][1] + 3) * nrows)
    fig = plt.figure(figsize=figsize, dpi=options["dpi"], constrained_layout=True)
    gs = GridSpec(nrows, ncols, figure=fig)
    axes = []
    
    # ----- ADD SUBPLOTS DYNAMICALLY, CENTER LAST ROW IF NEEDED -----
    axes = []
    if n_plots > 6 and (n_plots % ncols != 0):
        last_row_filled = n_plots % ncols
        pad_left = (ncols - last_row_filled) // 2

        plot_index = 0
        for row in range(nrows):
            for col in range(ncols):
                # If it's the last row and we're before the pad_left, skip to pad
                if row == nrows - 1 and col < pad_left:
                    continue
                # If all plots are placed, break
                if plot_index >= n_plots:
                    break
                ax = fig.add_subplot(gs[row, col], projection=getattr(ccrs, options["projection"])())
                axes.append(ax)
                plot_index += 1
    else:
        for i in range(n_plots):
            row = i // ncols
            col = i % ncols
            ax = fig.add_subplot(gs[row, col], projection=getattr(ccrs, options["projection"])())
            axes.append(ax)

    # ---- BUILD ORANGE - GREEEN COLORMAP -----
    cmap = options["cmap"]
    vmin = options["vmin"]
    vmax = options["vmax"]
    if cmap == "OrangeGreen":
        colors = ['#086e04', 'white', '#ff6700']
        base_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        white_start = int(norm(-0.2) * 255)
        white_end = int(norm(0.2) * 255)
        new_colors = base_cmap(np.linspace(0, 1, 256))
        new_colors[white_start:white_end, :] = [1, 1, 1, 1]
        cmap = mcolors.ListedColormap(new_colors)

    contour_levels = np.linspace(vmin, vmax, 11)

    # ----- BEGIN PLOTTING -----
    for i, ax in enumerate(axes):
        data_slice = data_array.isel({dim_name: i})
        ax.set_extent([min_lon, max_lon, min_lat + lat_offset, max_lat], crs=ccrs.PlateCarree())
        contour = ax.contourf(
            lonp + (0.35 * epsilon), latp + (0.1 * epsilon), data_slice,
            levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax,
            extend="both", transform=ccrs.PlateCarree()
        )
        ax.coastlines(resolution=options["resolution"])
        ax.add_feature(cfeature.LAND, facecolor=options["land_color"])
        label = labels[i] if i < len(labels) else f"{dim_name.capitalize()} {i+1}"
        ax.set_title(label, fontsize=options["title_fontsize"], fontweight=options["title_fontweight"])
        gl = ax.gridlines(draw_labels=True, dms=options["gridline_dms"],
                          color=options["gridline_color"], linestyle=options["gridline_style"],
                          alpha=options["gridline_alpha"])
        gl.top_labels = options["gridline_labels_top"]
        gl.right_labels = options["gridline_labels_right"]

    # ----- ADD A COLORBAR -----
    unit = options["unit"]
    
    # Only format if unit is not None, else use empty string
    if unit is not None:
        formatted_unit = format_unit(unit)[1:-1]
    else:
        formatted_unit = ""
        
    if ncols == 1:
        cbar = fig.colorbar(contour, ax=axes, orientation="horizontal",
                    shrink=options["cbar_shrink"]+0.1,
                    ticks=np.linspace(vmin, vmax, options["cbar_ticks"]))

        if formatted_unit:
            cbar.set_label(rf'$\left[{formatted_unit}\right]$', fontsize=16, labelpad=options["cbar_labelpad"])
            cbar.ax.tick_params(labelsize=14)
        else:
            cbar.set_label("", fontsize=16, labelpad=options["cbar_labelpad"])
            cbar.ax.tick_params(labelsize=14)
    
    else:
        cbar = fig.colorbar(contour, ax=axes, orientation="horizontal",
                    shrink=(options["cbar_shrink"]),
                    ticks=np.linspace(vmin, vmax, options["cbar_ticks"]))

        if formatted_unit:
            cbar.set_label(rf'$\left[{formatted_unit}\right]$', fontsize=22, labelpad=options["cbar_labelpad"])
            cbar.ax.tick_params(labelsize=16)
        else:
            cbar.set_label("", fontsize=22, labelpad=options["cbar_labelpad"])
            cbar.ax.tick_params(labelsize=16)

    # ----- MAKE THE TITLE -----
    detrended = kwargs.get("detrended", False)
    det_text = "Detrended" if detrended else "Raw"

    # Prepare unit string for title safely:
    unit_title = unit if unit is not None else ""

    if ncols == 1:
        plt.suptitle(
            f"{suptitle_prefix} {title_prefix} {unit_title} \n ({det_text}) \n {options['suffix']}",
            fontsize=(options["suptitle_fontsize"]-6),
            fontweight=options["suptitle_fontweight"],
            )
    else :
        plt.suptitle(
            f"{suptitle_prefix} {title_prefix} {unit_title} ({det_text}) \n {options['suffix']}",
            fontsize=options["suptitle_fontsize"],
            fontweight=options["suptitle_fontweight"],
            )

    # ----- SET OUTPUT PATH -----
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_title = title_prefix.replace("/", "_").replace("\\", "_")
    filename = f"{filename_prefix} {safe_title} ({det_text}) {options['suffix']}.png"
    plt.savefig(output_path / filename, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
###############################################################################   

###############################################################################
   
def error_components_timeseries(
    stats_df,
    output_path,
    cloud_cover=None,
    variable_name='',
    **kwargs
):
    """
    Plot time series of error components and optional cloud cover.

    This function generates a multi-panel plot showing:
      - Mean Bias
      - Unbiased RMSE
      - Standard Deviation of Error
      - Correlation
      - (Optional) Cloud Cover with smoothed version

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame containing time series of statistical error components with columns:
        ['mean_bias', 'unbiased_rmse', 'std_error', 'correlation'].

    output_path : str or Path
        Directory path where the resulting figure should be saved.

    cloud_cover : pd.Series, optional
        Time series of cloud cover data (percentage). If provided, an extra subplot will be shown.

    variable_name : str, optional
        Name of the variable (e.g., "SST", "Chlorophyll") for labeling purposes.

    kwargs : dict
        Keyword arguments to override default plotting options.

    Accepted kwargs include:
    -------------------------
    Keyword arguments overriding default plotting options. Include:
        - fig_width (float)                  : Width of the full figure (default varies).
        - fig_height_per_plot (float)        : Height allocated per subplot row.
        - sharex (bool)                      : If True, subplots share the same x-axis.
        - title_fontsize (int)               : Font size of the figure title.
        - title_fontweight (str)             : Font weight of the figure title.
        - label_fontsize (int)               : Font size of y-axis labels.
        - grid_color (str)                   : Color of grid lines.
        - grid_linestyle (str)               : Linestyle for grid (e.g., '--').
        - grid_alpha (float)                 : Alpha transparency of the grid lines.
        - mean_bias_color (str)              : Line color for Mean Bias subplot.
        - unbiased_rmse_color (str)          : Line color for Unbiased RMSE subplot.
        - std_error_color (str)              : Line color for Std Error subplot.
        - correlation_color (str)            : Line color for Correlation subplot.
        - cloud_cover_color (str)            : Line color for raw Cloud Cover.
        - cloud_cover_smoothed_color (str)   : Line color for smoothed Cloud Cover.
        - cloud_cover_rolling_window (int)   : Rolling window size for smoothing cloud cover.
        - spine_linewidth (float)            : Width of axes spines.
        - spine_edgecolor (str)              : Color of axes spines.
        - filename_template (str)            : Template for saved filename (e.g., '{}_errors.png').

    Example
    -------
    >>> error_components_timeseries(
    ...     stats_df=error_df,
    ...     cloud_cover=cloud_series,
    ...     output_path="figures/",
    ...     variable_name="SST"
    ... )

    Notes
    -----
    - Plots are styled with Seaborn and Matplotlib.
    - Default style and colors are controlled via `default_error_timeseries_options`.
    """
    # ----- OPTIONS -----
    options = extract_options(kwargs, default_error_timeseries_options)

    # ----- STYLE -----
    sns.set(style="whitegrid", context='notebook')
    sns.set_style("ticks")

    # ----- SETUP -----
    n_plots = 5 if cloud_cover is not None else 4
    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(options['fig_width'], options['fig_height_per_plot'] * n_plots),
        sharex=options['sharex']
    )

    # ----- TITLE -----
    title = "Comparison between error components timeseries"
    if cloud_cover is not None:
        title += " and cloud cover"
    if variable_name:
        title += f" ({variable_name})"
    fig.suptitle(title, fontsize=options['title_fontsize'], fontweight=options['title_fontweight'])
    fig.subplots_adjust(top=0.85)

    # ----- GRID STYLE -----
    grid_style = {
        'color': options['grid_color'],
        'linestyle': options['grid_linestyle'],
        'alpha': options['grid_alpha']
    }

    # ----- MEAN BIAS -----
    stats_df['mean_bias'].plot(ax=axes[0], color=options['mean_bias_color'], legend=False)
    axes[0].set_ylabel('Mean Bias', fontsize=options['label_fontsize'])
    axes[0].grid(**grid_style)

    # ----- UNBIASED RMSE -----
    stats_df['unbiased_rmse'].plot(ax=axes[1], color=options['unbiased_rmse_color'], legend=False)
    axes[1].set_ylabel('Unbiased RMSE', fontsize=options['label_fontsize'])
    axes[1].grid(**grid_style)

    # ----- STD ERROR -----
    stats_df['std_error'].plot(ax=axes[2], color=options['std_error_color'], legend=False)
    axes[2].set_ylabel('Std Error', fontsize=options['label_fontsize'])
    axes[2].grid(**grid_style)

    # ----- CORRELATION -----
    stats_df['cross_correlation'].plot(ax=axes[3], color=options['correlation_color'], legend=False)
    axes[3].set_ylabel('Correlation', fontsize=options['label_fontsize'])
    axes[3].grid(**grid_style)

    # ----- CLOUD COVER -----
    if cloud_cover is not None:
        cloud_cover_30d = cloud_cover.rolling(window=options['cloud_cover_rolling_window'], center=True).mean()
        axes[4].plot(cloud_cover.index, cloud_cover, color=options['cloud_cover_color'], label='Cloud Cover')
        axes[4].plot(cloud_cover_30d.index, cloud_cover_30d, color=options['cloud_cover_smoothed_color'], label='30-day Smoothed')
        axes[4].set_ylabel('Cloud Cover (%)', fontsize=options['label_fontsize'])
        axes[4].grid(**grid_style)
        axes[4].legend()
    else:
        axes[3].set_xlabel('')
        axes[3].grid(**grid_style)

    # ----- STYLE AXES -----
    for ax in axes:
        ax.label_outer()
        style_axes_spines(
            ax,
            linewidth=options['spine_linewidth'],
            edgecolor=options['spine_edgecolor']
        )

    # ----- LAYOUT -----
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ----- SAVE FIGURE -----
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = options['filename_template'].format(variable_name=variable_name)
    plt.savefig(output_path / filename)
    plt.close()
    
###############################################################################   

###############################################################################    
  
def plot_spectral(
    data=None,
    plot_type='PSD',
    freqs=None,
    fft_components=None,
    error_comp=None,
    cloud_covers=None,
    output_path=None,
    variable_name=None,
    fs=1.0,
    nperseg=256,
    **kwargs
):
    """
    Plot spectral analysis of time series data using either PSD or CSD.

    Parameters
    ----------
    data : pd.Series or dict, optional
        Optional time series input (not used directly in current implementation).

    plot_type : str
        Type of spectral plot to generate: 'PSD' (Power Spectral Density) or 'CSD' (Cross Spectral Density).

    freqs : array-like, optional
        Frequency values used for PSD plotting.

    fft_components : dict of arrays, optional
        Dictionary mapping labels to FFT-transformed series for PSD plotting.

    error_comp : pd.DataFrame or dict, optional
        Error component data used in CSD plotting.

    cloud_covers : list of (Series, str), optional
        List of tuples containing cloud cover time series and their labels (used in CSD).

    output_path : str or Path
        Path to save the resulting spectral plot.

    variable_name : str
        Short code name of the variable for use in the output filename.

    fs : float
        Sampling frequency (default is 1.0).

    nperseg : int
        Segment length for computing CSD (default is 256).

    kwargs : dict
        Additional keyword arguments for customization.

    Accepted kwargs include:
    -------------------------
    Keyword arguments overriding default plotting options. Include:
        - figsize (tuple)                    : Figure size (e.g., (12, 6)).
        - xlabel_fontsize (int)              : Font size of x-axis label.
        - ylabel_fontsize (int)              : Font size of y-axis label.
        - title_fontsize (int)               : Font size of plot title.
        - title_fontweight (str)             : Font weight of plot title (e.g., 'bold').
        - tick_labelsize (int)               : Font size of tick labels.
        - grid_color (str)                   : Grid color.
        - grid_alpha (float)                 : Grid line transparency.
        - grid_linestyle (str)               : Grid line style (e.g., '--').
        - freq_xlim (tuple)                  : Limits for frequency axis (e.g., (0.0, 0.5)).
        - additional_linestyles (list)       : Linestyles for multiple cloud cover series (e.g., ['--', '-.', ':']).
        - spine_linewidth (float)            : Width of plot spines.
        - spine_edgecolor (str)              : Color of plot spines.

    Raises
    ------
    ValueError
        If required inputs are missing or an unknown plot type is provided.

    """
    # ----- OPTIONS -----
    options = extract_options(kwargs, default_spectral)

    # ----- STYLE -----
    sns.set(style="whitegrid", context="notebook")
    sns.set_style("ticks")

    # ----- FIGURE -----
    plt.figure(figsize=options['figsize'])

    # ----- PSD -----
    if plot_type == 'PSD':
        if freqs is None or fft_components is None:
            raise ValueError("freqs and fft_components must be provided for PSD plot")

        for col, fft_vals in fft_components.items():
            if np.all(np.abs(fft_vals) == 0):
                continue
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                plt.plot(freqs, np.abs(fft_vals), label=col)

        plt.xlabel('Frequency (1/day)', fontsize=options['xlabel_fontsize'])
        plt.ylabel('Aplitude', fontsize=options['ylabel_fontsize'])
        plt.title('Power Spectral Density (PSD)', fontsize=options['title_fontsize'], fontweight=options['title_fontweight'])

    # ----- CSD -----
    elif plot_type == 'CSD':
        if error_comp is None:
            raise ValueError("error_comp must be provided for CSD plot")
        if cloud_covers is None or len(cloud_covers) == 0:
            raise ValueError("At least one cloud_cover tuple (data,label) must be provided in cloud_covers")

        additional_styles = cycle(options['additional_linestyles'])
        columns = error_comp.columns if hasattr(error_comp, 'columns') else list(error_comp.keys())
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        var_colors = {col: colors[i % len(colors)] for i, col in enumerate(columns)}

        for i, (cloud_cover, label) in enumerate(cloud_covers):
            linestyle = '-' if i == 0 else next(additional_styles)
            for col in columns:
                try:
                    f, Pxy = csd(error_comp[col], cloud_cover, fs=fs, nperseg=nperseg)
                    if np.all(np.abs(Pxy) == 0):
                        continue
                    plt.semilogy(f, np.abs(Pxy), linestyle=linestyle, color=var_colors[col], label=f'{col} vs {label}')
                except ZeroDivisionError:
                    continue

        plt.xlabel('Frequency (1/day)', fontsize=options['xlabel_fontsize'])
        plt.ylabel('Cross Power', fontsize=options['ylabel_fontsize'])
        labels = ', '.join([label for _, label in cloud_covers])
        plt.title(f'Cross-Spectral Density with {labels}', fontsize=options['title_fontsize'], fontweight=options['title_fontweight'])

    # ----- UNKNOWN TYPE -----
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
        
    # ----- FINAL FORMATTING -----
    plt.legend()
    plt.grid(True, color=options['grid_color'], alpha=options['grid_alpha'], linestyle=options['grid_linestyle'])
    plt.xlim(*options['freq_xlim'])
    plt.tick_params(axis='both', which='major', labelsize=options['tick_labelsize'])

    # ----- SPINES -----
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # ----- STYLE SPINES -----
    style_axes_spines(ax, linewidth=options['spine_linewidth'], edgecolor=options['spine_edgecolor'])

    # ----- SAVE FIGURE -----
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"Spectral_Plot_{plot_type}_{variable_name}"
    plt.savefig(output_path / filename)
    plt.close()