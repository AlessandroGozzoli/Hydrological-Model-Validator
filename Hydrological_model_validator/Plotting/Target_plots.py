###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Standard library imports
import itertools
from itertools import cycle
from pathlib import Path

# Data handling and plotting libraries
import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm

# Module formatting and plotting utilities
from .formatting import fill_annular_region, get_variable_label_unit
from .default_target_options import (
    default_target_base_options,
    default_target_overlay_options,
    default_target_data_marker_options,
    default_target_plt_options,
    default_month_markers,
    default_target_monthly_plt_options,
    default_target_monthly_base_options,
    default_target_monthly_data_marker_options,
)

# Local processing modules
from ..Processing.Target_computations import (
    compute_normalised_target_stats,
    compute_normalised_target_stats_by_month,
    compute_target_extent_monthly,
    compute_target_extent_yearly,
)
from ..Processing.data_alignment import extract_mod_sat_keys
from ..Processing.utils import extract_options

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def comprehensive_target_diagram(data_dict: dict, **kwargs) -> None:
    """
    Generate a comprehensive yearly target diagram using normalized statistics (bias, CRMSD, RMSD).

    Parameters
    ----------
    data_dict : dict
        Dictionary of model and reference time series indexed by datetime, e.g.,
        {
            "Ref": pd.Series,
            2000: pd.Series,
            2001: pd.Series,
            ...
        }

    Keyword Arguments
    -----------------
    - output_path (str or Path)         : Required. Directory where the figure is saved.
    - variable_name (str)               : Short name used to infer full variable name and unit.
    - variable (str)                    : Full variable name (e.g., "Chlorophyll").
    - unit (str)                        : Unit of measurement (e.g., "mg Chl/m³").
    - filename (str)                    : Name of the output image file.
    - title (str)                       : Custom title for the plot.
    - zone_bounds (tuple)               : Tuple of two floats defining zone radii (e.g., (0.5, 0.7)).
    - marker_shapes (list)              : List of marker shapes for the data points.
    - base_* (various types)            : Options for the base target diagram (prefix: "base_").
    - overlay_* (various types)         : Options for the overlay circles (prefix: "overlay_").
    - data_* (various types)            : Options for the data point markers (prefix: "data_").
    - (no prefix) (various types)       : General plot options (e.g., figsize, dpi, title_pad).

    Raises
    ------
    ValueError
        If required variable info or output path is missing.

    Example
    -------
    comprehensive_target_diagram(
        data_dict,
        variable_name="Chl",
        output_path="./plots",
        base_alpha=0.3,
        overlay_circles=[0.5, 1.0, 1.5],
        data_markersize=16,
        title="Target Plot"
    )
    """

    # ----- VALIDATE AND PREPARE INPUT OPTIONS -----
    output_path_value = kwargs.pop("output_path", None)
    if output_path_value is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")
    output_path = Path(output_path_value)

    variable_name = kwargs.pop("variable_name", None)
    variable = kwargs.pop("variable", None)
    unit = kwargs.pop("unit", None)

    if variable_name is not None:
        variable, unit = get_variable_label_unit(variable_name)
        variable = variable or kwargs.get("variable")
        unit = unit or kwargs.get("unit")
    elif variable is None or unit is None:
        raise ValueError("You must provide either 'variable' and 'unit' or 'variable_name' in kwargs.")

    # ----- EXTRACT PLOTTING OPTIONS FROM KWARGS -----
    base_opts = extract_options(kwargs, default_target_base_options, prefix="base_")
    overlay_opts = extract_options(kwargs, default_target_overlay_options, prefix="overlay_")
    data_marker_opts = extract_options(kwargs, default_target_data_marker_options, prefix="data_")
    plt_opts = extract_options(kwargs, default_target_plt_options)

    # ----- COMPUTE STATISTICS AND LAYOUT -----
    bias, crmsd, rmsd, labels = compute_normalised_target_stats(data_dict)
    marker_shapes = cycle(kwargs.pop("marker_shapes", ["P", "o", "X", "s", "D", "^", "v", "p", "h", "*"]))
    extent = compute_target_extent_yearly(data_dict)

    # ----- INITIALIZE THE FIGURE -----
    fig, ax = plt.subplots(figsize=plt_opts.get("figsize"), dpi=plt_opts.get("dpi"))
    
    # ----- DRAW PERFORMANCE SHADED REGIONS -----
    zone_bounds = kwargs.get("zone_bounds", (0.5, 0.7))
    if not (isinstance(zone_bounds, tuple) and len(zone_bounds) == 2 and all(isinstance(x, (int, float)) for x in zone_bounds)):
        raise ValueError("'zone_bounds' must be a tuple of two numbers, e.g., (0.5, 0.7)")
    bound1, bound2 = zone_bounds

    for r_start, r_end, color in [
        (0, bound1, 'lightgreen'),
        (bound1, bound2, 'khaki'),
        (bound2, extent, 'lightcoral')
    ]:
        fill_annular_region(ax, r_start, r_end, color=color, alpha=0.4)

    ax.set_axisbelow(True)

    # ----- DRAW BASE DIAGRAM -----
    sm.target_diagram(1.0, extent, extent,
                      markerLabelColor=base_opts.get('markerLabelColor'),
                      alpha=base_opts.get('alpha'),
                      markersize=base_opts.get('markersize'),
                      circlelinespec=base_opts.get('circlelinespec'))
    
    # ----- DRAW OVERLAY -----
    sm.target_diagram(1.0, extent, extent,
                      markerLabelColor=overlay_opts.get('markerLabelColor'),
                      alpha=overlay_opts.get('alpha'),
                      markersize=overlay_opts.get('markersize'),
                      circlelinespec=overlay_opts.get('circlelinespec'),
                      circles=overlay_opts.get('circles'))

    # ----- PLOT YEARLY DATA POINTS -----
    for b, c, r, label in zip(bias, crmsd, rmsd, labels):
        sm.target_diagram(b, c, r,
                          markerLabelColor=data_marker_opts.get('markerLabelColor'),
                          markersymbol=next(marker_shapes),
                          markersize=data_marker_opts.get('markersize'),
                          alpha=data_marker_opts.get('alpha'),
                          circles=data_marker_opts.get('circles'),
                          overlay=data_marker_opts.get('overlay'))

    # ----- ADD THE TITLE -----
    plt.title(kwargs.get("title", f"Yearly Target Plot (Normalized Stats) | {variable}"),
              pad=plt_opts.get("title_pad"),
              fontweight=plt_opts.get("title_fontweight"))

    # ----- CHECK FOR DIRECTORY EXISTENCE -----
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / kwargs.get("filename", "Unified_Target_Diagram.png")

    # ----- PRINT THE PLOT AND SAVE -----
    plt.tight_layout()
    plt.savefig(save_path, dpi=plt_opts.get("dpi"))
    plt.close()
###############################################################################

###############################################################################
def target_diagram_by_month(data_dict: dict, **kwargs) -> None:
    """
    Generate a monthly target diagram with normalized statistics (bias, CRMSD, RMSD).

    Parameters
    ----------
    data_dict : dict
        Dictionary of model and reference monthly time series by year.

    Keyword Arguments
    -----------------
    - output_path (str or Path)         : Required. Directory where the figure is saved.
    - variable_name (str)               : Short name used to infer full variable name and unit.
    - variable (str)                    : Full variable name (e.g., "Chlorophyll").
    - unit (str)                        : Unit of measurement (e.g., "mg Chl/m³").
    - filename (str)                    : Name of the output image file.
    - title (str)                       : Custom title for the plot.
    - zone_bounds (tuple)               : Tuple of two floats defining zone radii (e.g., (0.5, 0.7)).
    - markers (list)                    : List of marker shapes per year.
    - month_colors (list)               : List of colors corresponding to each month.
    - base_* (various types)            : Options for the base diagram (prefix: "base_").
    - overlay_* (various types)         : Options for overlay diagram (prefix: "overlay_").
    - data_* (various types)            : Options for data point markers (prefix: "data_").
    - (no prefix) (various types)       : General plot options (e.g., figsize, dpi, fontsize).

    Raises
    ------
    ValueError
        If required variable info or output path is missing.

    Example
    -------
    target_diagram_by_month(
        data_dict,
        variable_name="Chl",
        output_path="./monthly_plots",
        data_markersize=16,
        title="Monthly Target Diagram"
    )
    """

    # ---- VALIDATE AND PREPARE INPUT OPTIONS ----
    output_path = kwargs.pop("output_path", None)
    if output_path is None:
        raise ValueError("output_path must be specified in kwargs.")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    variable_name = kwargs.pop("variable_name", None)
    variable = kwargs.pop("variable", None)
    unit = kwargs.pop("unit", None)

    if variable_name:
        variable, unit = get_variable_label_unit(variable_name)
        variable = variable or kwargs.get("variable")
        unit = unit or kwargs.get("unit")
    elif variable is None or unit is None:
        raise ValueError("You must provide either 'variable' and 'unit' or 'variable_name' in kwargs.")

    # ---- EXTRACT PLOTTING OPTIONS ----
    markers = itertools.cycle(kwargs.pop("markers", default_month_markers["markers"]))
    month_colors = itertools.cycle(kwargs.pop("month_colors", default_month_markers["month_colors"]))
    plt_opts = extract_options(kwargs, default_target_monthly_plt_options)
    base_opts = extract_options(kwargs, default_target_monthly_base_options, prefix="base_")
    overlay_opts = extract_options(kwargs, default_target_overlay_options, prefix="overlay_")
    data_marker_opts = extract_options(kwargs, default_target_monthly_data_marker_options, prefix="data_")

    # ---- COMPUTE EXTENT TO CORRECTLY ENCOMPASS ALL MARKERS ----
    extent = compute_target_extent_monthly(data_dict)
    
    # ----- INITIALIZE FIGURE -----
    fig, ax = plt.subplots(figsize=plt_opts.get("figsize"))

    # ----- CREATE TITLE -----
    plt.title(
        kwargs.get("title", f"Monthly Target Plot (Normalized Stats) | {variable}"),
        fontsize=plt_opts.get("title_fontsize"),
        pad=plt_opts.get("title_pad"),
        fontweight=plt_opts.get("title_fontweight")
    )

    # ---- DRAW BASE DIAGRAM ----
    sm.target_diagram(
        np.array([1.0]), np.array([extent]), np.array([1.0]),
        markerLabelColor=base_opts.get("markerLabelColor"),
        markersize=base_opts.get("markersize"),
        alpha=base_opts.get("alpha"),
        circles=base_opts.get("overlay_circles", [0.7, 1.0] + list(np.arange(2.0, extent + 1e-6, 1.0))),
        circlelinespec=base_opts.get("circlelinespec")
    )

    # ----- DRAW OVERLAY -----
    sm.target_diagram(
        np.array([1.0]), np.array([extent]), np.array([1.0]),
        markerLabelColor=overlay_opts.get("markerLabelColor"),
        markersize=overlay_opts.get("markersize"),
        alpha=overlay_opts.get("alpha"),
        circles=overlay_opts.get("circles"),
        circlelinespec=overlay_opts.get("circlelinespec")
    )

    # ---- DRAW SHADED ZONES ----
    zone_bounds = kwargs.get("zone_bounds", (0.5, 0.7))
    if not (isinstance(zone_bounds, tuple) and len(zone_bounds) == 2 and all(isinstance(x, (int, float)) for x in zone_bounds)):
        raise ValueError("'zone_bounds' must be a tuple of two numbers, e.g., (0.5, 0.7)")
    bound1, bound2 = zone_bounds

    for r_start, r_end, color in [(0, bound1, 'lightgreen'), (bound1, bound2, 'khaki'), (bound2, extent, 'lightcoral')]:
        fill_annular_region(ax, r_start, r_end, color=color, alpha=0.4)

    # ---- PLOT MONTHLY DATA POINTS ----
    mod_key, _ = extract_mod_sat_keys(data_dict)
    years = list(data_dict[mod_key].keys())

    # ----- LOOP TO COMPUTE ALL DATA -----
    for month_index in range(12):
        try:
            bias, crmsd, rmsd, labels = compute_normalised_target_stats_by_month(data_dict, month_index)
        except ValueError:
            continue

        # ----- LOOP FOR ALL YEARS TO FETCH SHAPE/COLOR -----
        color = next(month_colors)
        for i, (b, c, r, label) in enumerate(zip(bias, crmsd, rmsd, labels)):
            year = str(label)
            year_index = years.index(year) if year in years else i
            marker = next(markers)

            # ----- PLOT THE MARKERS -----
            sm.target_diagram(
                np.array([b]), np.array([c]), np.array([r]),
                markercolors={"face": color, "edge": data_marker_opts.get("edge_color")},
                markersymbol=marker,
                markersize=data_marker_opts.get("markersize"),
                alpha=data_marker_opts.get("alpha"),
                overlay=data_marker_opts.get("overlay"),
                circles=data_marker_opts.get("circles")
            )

    # ---- FORMATTING OPTIONS ----
    ax.tick_params(axis='both', labelsize=plt_opts.get("tick_labelsize"))
    ax.xaxis.label.set_size(plt_opts.get("axis_labelsize"))
    ax.yaxis.label.set_size(plt_opts.get("axis_labelsize"))

    # ----- CHECK EXISTENCE OF SAVE FOLDER -----
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / kwargs.get("filename", "Monthly_Target_Diagram.png")
    
    # ----- PRINT AND SAVE THE PLOT -----
    plt.savefig(save_path, dpi=plt_opts.get("dpi"))
    plt.close()
###############################################################################