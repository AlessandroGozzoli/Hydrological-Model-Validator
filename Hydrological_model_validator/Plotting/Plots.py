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
                        style_axes_spines,
                        format_unit)

from ..Processing.data_alignment import (extract_mod_sat_keys,
                                         gather_monthly_data_across_years)

from ..Processing.stats_math_utils import (fit_huber,
                                           fit_lowess,
                                           corr_no_nan)

from ..Processing.time_utils import get_season_mask
from ..Processing.utils import extract_options

from .default_plot_options import (default_plot_options_ts,
                                   default_plot_options_scatter,
                                   default_scatter_by_season_options,
                                   default_boxplot_options,
                                   default_violinplot_options,
                                   default_efficiency_plot_options,
                                   spatial_efficiency_defaults )

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################

###############################################################################
def timeseries(
    data_dict: Dict[str, Union[pd.Series, list]],
    BIAS: Union[pd.Series, list, None] = None,
    cloud_coverage: Union[pd.Series, list, None] = None,
    **kwargs: Any
) -> None:
    """
    Plot time series of daily mean values from multiple datasets along with BIAS and optional cloud coverage.
    
    This function generates up to two-panel time series plots:
        1. The first figure shows daily mean values of each dataset (typically model and satellite data).
        If BIAS and cloud coverage are provided, a second figure is generated with:
            2. An upper subplot displaying the BIAS (model - satellite) time series.
            3. A lower subplot showing cloud coverage time series, optionally smoothed by 7- or 30-day running means.
            
    Figures are saved to a specified output directory as PNG files and displayed using matplotlib.
            
    Parameters
    ----------
    data_dict : Dict[str, Union[pd.Series, list]]
        Dictionary containing daily mean values for different sources (e.g., model and satellite).
        Keys are strings identifying the data source.
        Values should be `pandas.Series` with datetime indices or lists convertible to Series.

    BIAS : Union[pd.Series, list, None], optional
        Series (or list) representing the BIAS time series (typically model - satellite).
        Should be time-aligned with the values in `data_dict`. If None, only the time series plot is generated.

    cloud_coverage : Union[pd.Series, list, None], optional
        Series (or list) of daily cloud coverage percentages aligned with `data_dict` and `BIAS`.
        If provided alongside BIAS, a second figure with BIAS and cloud coverage plots is generated.

    Accepted kwargs include:
    -----------------------
    Keyword arguments overriding default plotting options. Include:
        - output_path (str or Path)        : **Required.** Directory path to save figures.
        - variable_name (str)              : Variable code name to infer full name and unit.
        - variable (str)                   : Full variable name (e.g., "Chlorophyll"). Used in titles and axis labels.
        - unit (str)                      : Unit of measurement (e.g., "mg Chl/m³"). Displayed on axis.
        - BA (bool)                       : If True, appends " (Basin Average)" to the title.
        - figsize (tuple of float)        : Figure size in inches (default e.g., (20, 8)).
        - dpi (int)                      : Figure resolution in dots per inch (default 300).
        - color_palette (iterator)        : Iterator of colors (e.g., `itertools.cycle(sns.color_palette("tab10"))`).
        - line_width (float)              : Width of plotted lines (default 1.0).
        - title_fontsize (int)            : Font size of the main title.
        - bias_title_fontsize (int)       : Font size of the BIAS subplot title.
        - label_fontsize (int)            : Font size of axis labels.
        - legend_fontsize (int)           : Font size of the legend.
        - savefig_kwargs (dict)            : Additional keyword arguments passed to `plt.savefig()`, e.g., `bbox_inches`, `transparent`.
        - smooth7 (bool)                  : If True, adds a 7-day running mean smoothing line to cloud coverage plot.
        - smooth30 (bool)                 : If True, adds a 30-day running mean smoothing line to cloud coverage plot.

    Example
    -------
    >>> timeseries(
    ...     data_dict={'model': model_series, 'satellite': sat_series},
    ...     BIAS=model_series - sat_series,
    ...     cloud_coverage=cloud_series,
    ...     variable_name='Chl',
    ...     output_path='figures/',
    ...     BA=True,
    ...     smooth7=True
    ... )
    """
    # ----- RETRIEVE DEFAULT OPTIONS AND OVERRIDE WITH kwargs -----
    options = SimpleNamespace(**{**default_plot_options_ts, **kwargs})

    # Check mandatory output path
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    # Handle variable and unit extraction
    if options.variable_name is not None:
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    else:
        if options.variable is None or options.unit is None:
            raise ValueError("If 'variable_name' is not provided, both 'variable' and 'unit' must be specified.")

    # Title construction
    title = f'Daily Mean Values for {options.variable_name or options.variable} Datasets'
    if options.BA:
        title += ' (Basin Average)'

    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    label_lookup = {
        mod_key: "Model Output",
        sat_key: "Satellite Observations"
    }

    # Convert to pandas.Series if not already
    data_dict = {k: pd.Series(v) if not isinstance(v, pd.Series) else v for k, v in data_dict.items()}
    if BIAS is not None:
        BIAS = pd.Series(BIAS) if not isinstance(BIAS, pd.Series) else BIAS
    if cloud_coverage is not None:
        cloud_coverage = pd.Series(cloud_coverage) if not isinstance(cloud_coverage, pd.Series) else cloud_coverage

    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # SINGLE PAGE: only timeseries (no BIAS or cloud_coverage)
    if BIAS is None or cloud_coverage is None:
        fig = plt.figure(figsize=options.figsize, dpi=options.dpi)
        ax1 = fig.add_subplot(1, 1, 1)

        plotter = partial(
            plot_line,
            ax=ax1,
            label_lookup=label_lookup,
            color_palette=options.color_palette,
            line_width=options.line_width
        )
        list(starmap(plotter, data_dict.items()))

        ax1.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
        ax1.set_ylabel(f'{options.variable} {options.unit}', fontsize=options.label_fontsize)
        ax1.tick_params(width=2)
        ax1.legend(loc='upper left', fontsize=options.legend_fontsize)
        ax1.grid(True, linestyle='--')
        style_axes_spines(ax1)

        plt.tight_layout()
        filename = f'{options.variable_name or options.variable}_timeseries.png'
        plt.savefig(output_path / filename, **options.savefig_kwargs)
        plt.show(block=False)
        plt.draw()
        plt.close()
        return

    # TWO PAGE PLOTS: Page 1 is timeseries + BIAS, Page 2 is BIAS + cloud coverage plots
    # --- PAGE 1: timeseries + BIAS ---
    fig1 = plt.figure(figsize=options.figsize, dpi=options.dpi)
    gs1 = GridSpec(2, 1, height_ratios=[8, 4])
    ax1a = fig1.add_subplot(gs1[0])

    plotter = partial(
        plot_line,
        ax=ax1a,
        label_lookup=label_lookup,
        color_palette=options.color_palette,
        line_width=options.line_width
    )
    list(starmap(plotter, data_dict.items()))

    ax1a.set_title(title, fontsize=options.title_fontsize, fontweight='bold')
    ax1a.set_ylabel(f'{options.variable} {options.unit}', fontsize=options.label_fontsize)
    ax1a.tick_params(width=2)
    ax1a.legend(loc='upper left', fontsize=options.legend_fontsize)
    ax1a.grid(True, linestyle='--')
    style_axes_spines(ax1a)

    plt.tight_layout()
    filename1 = f'{options.variable_name or options.variable}_timeseries_bias.png'
    plt.savefig(output_path / filename1, **options.savefig_kwargs)
    plt.show(block=False)
    plt.draw()
    plt.close()

    # --- PAGE 2: BIAS + Cloud Coverage (with smoothing if requested) ---

    if BIAS is not None or cloud_coverage is not None:
        fig2 = plt.figure(figsize=options.figsize, dpi=options.dpi)
        gs2 = GridSpec(2, 1, height_ratios=[8, 4])
        ax2a = fig2.add_subplot(gs2[0])
        ax2b = fig2.add_subplot(gs2[1])

        # Plot BIAS if present, else hide upper subplot
        if BIAS is not None:
            sns.lineplot(data=BIAS, color='k', ax=ax2a)
            ax2a.set_title(f'BIAS ({options.variable_name or options.variable})', fontsize=options.bias_title_fontsize, fontweight='bold')
            ax2a.set_ylabel(f'BIAS {options.unit}', fontsize=options.label_fontsize)
            ax2a.tick_params(width=2)
            ax2a.grid(True, linestyle='--')
            style_axes_spines(ax2a)
        else:
            ax2a.set_visible(False)

        # Plot cloud coverage if present, else hide lower subplot
        if cloud_coverage is not None:
            sns.lineplot(data=cloud_coverage, color='gray', ax=ax2b, label='Daily Cloud Coverage')
            if getattr(options, 'smooth7', False):
                cc_smooth7 = cloud_coverage.rolling(window=7, center=True, min_periods=1).mean()
                sns.lineplot(data=cc_smooth7, color='r', linestyle='--', ax=ax2b, label='7-day Smooth')
            if getattr(options, 'smooth30', False):
                cc_smooth30 = cloud_coverage.rolling(window=30, center=True, min_periods=1).mean()
                sns.lineplot(data=cc_smooth30, color='blue', linestyle='--', ax=ax2b, label='30-day Smooth')

            ax2b.set_title('Cloud Coverage (%)', fontsize=options.bias_title_fontsize, fontweight='bold')
            ax2b.set_ylabel('Cloud Coverage (%)', fontsize=options.label_fontsize)
            ax2b.tick_params(width=2)
            ax2b.grid(True, linestyle='--')
            ax2b.legend(fontsize=options.legend_fontsize)
            style_axes_spines(ax2b)
        else:
            ax2b.set_visible(False)
        
    # Compute and print correlations only if both BIAS and cloud coverage are present
    if BIAS is not None and cloud_coverage is not None:
        corr_raw = corr_no_nan(BIAS, cloud_coverage)
        corr_smooth7_val = corr_no_nan(BIAS, cc_smooth7) if cc_smooth7 is not None else float('nan')
        corr_smooth30_val = corr_no_nan(BIAS, cc_smooth30) if cc_smooth30 is not None else float('nan')

        print(f"Correlation between BIAS and raw cloud coverage: {corr_raw:.3f}")
        print(f"Correlation between BIAS and 7-day smoothed cloud coverage: {corr_smooth7_val:.3f}")
        print(f"Correlation between BIAS and 30-day smoothed cloud coverage: {corr_smooth30_val:.3f}")

    plt.tight_layout()
    filename2 = f'{options.variable_name or options.variable}_bias_cloudcoverage.png'
    plt.savefig(output_path / filename2, **options.savefig_kwargs)
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
    - suptitle_fontsize (int)         : Font size of the super title.
    - suptitle_fontweight (str)       : Font weight of the super title.
    - suptitle_y (float)              : Vertical position of the super title.
    - title_fontsize (int)            : Font size of subplot titles.
    - title_fontweight (str)          : Font weight of subplot titles.
    - cbar_labelsize (int)            : Font size of colorbar tick labels.
    - cbar_labelpad (int)             : Padding between colorbar and label.
    - cbar_shrink (float)             : Shrink factor for horizontal colorbar.
    - cbar_ticks (int)                : Number of colorbar ticks.
    - figsize_per_plot (tuple)        : Size per subplot (width, height).
    - max_cols (int)                  : Max number of columns in subplot grid.
    - epsilon (float)                 : Padding fallback if not in geo_coords.
    - lat_offset_base (float)         : Latitude offset for label placement.
    - gridline_color (str)            : Color of gridlines.
    - gridline_style (str)            : Line style of gridlines (e.g., "--").
    - gridline_alpha (float)          : Gridline transparency.
    - gridline_dms (bool)             : Format labels in DMS (deg:min:sec).
    - gridline_labels_top (bool)      : Show labels on top axis.
    - gridline_labels_right (bool)    : Show labels on right axis.
    - projection (str)                : Cartopy projection class name.
    - resolution (str)                : Resolution of coastlines (e.g., "10m").
    - land_color (str)                : Color for landmasses.
    - show (bool)                     : Display the plot interactively.
    - block (bool)                    : Block execution on plt.show().
    - pause_duration (float)          : Time (sec) to pause after plotting.
    - dpi (int)                       : Resolution of the output figure.
    
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

    # ----- SETUP THE PAGE ------
    max_cols = options["max_cols"]
    nrows = int(np.ceil(n_plots / max_cols))
    remainder = n_plots % max_cols
    full_rows = n_plots // max_cols

    # ----- SETUP THE FIGURE AND SIZE -----
    figsize = (options["figsize_per_plot"][0] * max_cols,
               options["figsize_per_plot"][1] * nrows)
    fig = plt.figure(figsize=figsize, dpi=options["dpi"], constrained_layout=True)
    gs = GridSpec(nrows, max_cols, figure=fig)
    axes = []

    # ----- DEFINE CORNER CASES FOR NUMBER NOT CELLS /2 OR /3 -----
    for row in range(nrows):
        if row < full_rows:
            for col in range(max_cols):
                ax = fig.add_subplot(gs[row, col], projection=getattr(ccrs, options["projection"])())
                axes.append(ax)
        else:
            if remainder == 1:
                ax = fig.add_subplot(gs[row, 1], projection=getattr(ccrs, options["projection"])())
                axes.append(ax)
            elif remainder == 2:
                ax1 = fig.add_subplot(gs[row, 0], projection=getattr(ccrs, options["projection"])())
                ax2 = fig.add_subplot(gs[row, 2], projection=getattr(ccrs, options["projection"])())
                axes.extend([ax1, ax2])

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

    cbar = fig.colorbar(contour, ax=axes, orientation="horizontal",
                    shrink=options["cbar_shrink"],
                    ticks=np.linspace(vmin, vmax, options["cbar_ticks"]))
    cbar.ax.tick_params(direction='in', length=32, labelsize=options["cbar_labelsize"])

    if formatted_unit:
        cbar.set_label(rf'$\left[{formatted_unit}\right]$', fontsize=18, labelpad=options["cbar_labelpad"])
    else:
        cbar.set_label("", fontsize=18, labelpad=options["cbar_labelpad"])

    # ----- MAKE THE TITLE -----
    detrended = kwargs.get("detrended", False)
    det_text = "Detrended" if detrended else "Raw"

    # Prepare unit string for title safely:
    unit_title = unit if unit is not None else ""

    plt.suptitle(
        f"{suptitle_prefix} {title_prefix} {unit_title} ({det_text}) {options['suffix']}",
        fontsize=options["suptitle_fontsize"],
        fontweight=options["suptitle_fontweight"],
        y=options["suptitle_y"]
        )

    # ----- SET OUTPUT PATH -----
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_title = title_prefix.replace("/", "_").replace("\\", "_")
    filename = f"{filename_prefix} {safe_title} ({det_text}) {options['suffix']}.png"
    plt.savefig(output_path / filename)

    if options["show"]:
        plt.show(block=options["block"])
        plt.draw()
        plt.pause(options["pause_duration"])
    plt.close()
###############################################################################   