import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm
from pathlib import Path
import random
from matplotlib.lines import Line2D
import itertools
from types import SimpleNamespace


from ..Processing.Taylor_computations import (compute_yearly_taylor_stats, 
                                              build_all_points)
from ..Processing.utils import extract_options

from .formatting import get_variable_label_unit

from .default_taylor_options import (default_taylor_base_options,
                                   default_taylor_ref_marker_options,
                                   default_taylor_data_marker_options,
                                   default_taylor_plt_options,
                                   default_marker_shapes,
                                   default_monthly_taylor_base_options,
                                   default_monthly_ref_marker_options,
                                   default_monthly_data_marker_options,
                                   default_month_colors,
                                   default_monthly_plt_options)

###############################################################################
def comprehensive_taylor_diagram(data_dict, **kwargs):
    """
    Plot a comprehensive yearly Taylor diagram comparing model and satellite statistics.

    Parameters
    ----------
    data_dict : dict
        Dictionary with model and satellite data organized by year.

    Keyword Arguments
    -----------------
    - output_path (str or Path)         : Required. Directory where the figure is saved.
    - variable_name (str)               : Short name used to infer full variable name and unit.
    - variable (str)                    : Full variable name (e.g., "Chlorophyll").
    - unit (str)                        : Unit of measurement (e.g., "mg Chl/m³").
    - marker_shapes (list of str)       : Marker symbols used for each year (e.g., ['o', 's', 'D']).

    Plot Options
    ------------
    - figsize (tuple of float)          : Figure size in inches.
    - dpi (int)                         : Plot resolution.
    - title (str)                       : Plot title.
    - title_fontsize (int)              : Font size of the title.
    - title_fontweight (str or int)     : Font weight of the title.
    - title_pad (float)                 : Padding between title and plot.

    Base Taylor Options
    -------------------
    - tickrms (list of float)           : RMSD contours for base diagram.
    - titleRMS (str)                    : Turn RMS title on/off.

    Reference Marker Options (Prefix with 'Ref_')
    ---------------------------------------------
    - Ref_markersymbol (str)            : Marker symbol for reference overlay.
    - Ref_markercolor (str or tuple)    : Marker face or edge color.
    - Ref_markersize (float)            : Marker size.

    Data Marker Options
    -------------------
    - markersymbol (str)                : Marker symbol for yearly points.
    - markercolor (str or tuple)        : Marker color.
    - markersize (float)                : Marker size.

    Raises
    ------
    ValueError
        If neither 'variable_name' nor both 'variable' and 'unit' are provided.

    Saves
    -----
    Taylor_diagram_summary.png : Saved in output_path
    """
    # ----- CONVERT KWARGS TO NAMESPACE -----
    options = SimpleNamespace(**kwargs)
    options.output_path = getattr(options, "output_path", None)
    options.variable = getattr(options, "variable", None)
    options.unit = getattr(options, "unit", None)
    options.variable_name = getattr(options, "variable_name", None)

    # ----- RETRIEVE NECESSARY OUTPUT PATH AND VARIABLE/UNIT INFO -----
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    if options.variable_name is not None:
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    else:
        if options.variable is None or options.unit is None:
            raise ValueError(
                "If 'variable_name' is not provided, both 'variable' and 'unit' must be specified in kwargs or defaults."
            )

    stats_by_year, std_ref = compute_yearly_taylor_stats(data_dict)

    # ----- ASSIGN SHAPES FOR EACH DATA ENTRY -----
    user_shapes = getattr(options, "marker_shapes", default_marker_shapes)
    labels = ["Ref"] + [entry[0] for entry in stats_by_year]

    if len(user_shapes) < len(labels) - 1:
        all_markers = [m for m in Line2D.markers.keys() if isinstance(m, str) and m not in user_shapes and m != ' ']
        user_shapes += random.sample(all_markers, len(labels) - 1 - len(user_shapes))

    # ----- NORMALIZE THE DATA -----
    sdev = np.array([std_ref] + [e[1] for e in stats_by_year]) / std_ref
    crmsd = np.array([0.0] + [e[2] for e in stats_by_year]) / std_ref
    ccoef = np.array([1.0] + [e[3] for e in stats_by_year])

    # ----- FETCH DEFAULT OPTIONS -----
    base_opts = extract_options(kwargs, default_taylor_base_options)
    ref_opts = extract_options(kwargs, default_taylor_ref_marker_options, prefix="Ref_")
    data_opts = extract_options(kwargs, default_taylor_data_marker_options)
    plt_opts = extract_options(kwargs, default_taylor_plt_options)

    # ----- SET REFERENCE MARKER VALUES -----
    sdev_ref = kwargs.pop("sdevRef", 1.0)
    crmsd_ref = kwargs.pop("crmsdRef", 0.0)
    ccoef_ref = kwargs.pop("ccoefRef", 1.0)

    # ----- CREATE FIGURE -----
    plt.figure(figsize=plt_opts.get('figsize'), dpi=plt_opts.get('dpi'))
    
    # ----- SET TITLE -----
    plt.title(
        plt_opts.get('title', f"Yearly Taylor Diagram (Normalized Stats) | {options.variable}"),
        pad=plt_opts.get('title_pad'),
        fontsize=plt_opts.get('title_fontsize'),
        fontweight=plt_opts.get('title_fontweight')
    )

    # ----- BASE DIAGRAM -----
    sm.taylor_diagram(
        sdev, crmsd, ccoef,
        markersymbol='none',
        markercolors={"face": "none", "edge": "none"},
        markersize=0,
        alpha=0,
        **base_opts
    )
    
    # ----- CREATE LABELS -----
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    label_y = y_min - 0.11 * (y_max - y_min)

    tickrms = base_opts.get("tickrms", [0.5])
    first_tickrms = tickrms[0] if tickrms else 0.5

    plt.text(sdev[0], label_y, "Ref", ha="center", va="center", fontsize=16, fontweight='bold', color='r')
    if base_opts.get('titleRMS', 'off') == 'off':
        plt.text(sdev[0] + first_tickrms, label_y, 'RMSD', fontsize=12,
                 ha='center', va='center', fontweight='bold', color=(0.0, 0.6, 0.0))

    # ----- OVERLAY REFERENCE -----
    sm.taylor_diagram(
        np.array([sdev_ref] * 2),
        np.array([crmsd_ref] * 2),
        np.array([ccoef_ref] * 2),
        tickrms=[0.0],
        overlay='on',
        **ref_opts
    )

    # ----- OVERLAY DATA POINTS -----
    for shape, (x, y, c) in zip(itertools.cycle(user_shapes), zip(sdev[1:], crmsd[1:], ccoef[1:])):
        sm.taylor_diagram(
            np.array([x, x]),
            np.array([y, y]),
            np.array([c, c]),
            showlabelsrms='off',
            overlay='on',
            markersymbol=shape,
            **data_opts
        )

    # ----- CHECK DIRECTORY EXISTENCE -----
    Path(options.output_path).mkdir(parents=True, exist_ok=True)
    
    # ----- PRINT AND SAVE PLOT -----
    plt.savefig(Path(options.output_path) / 'Taylor_diagram_summary.png')
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
###############################################################################

###############################################################################
def monthly_taylor_diagram(data_dict, **kwargs):
    """
    Plot a unified Taylor diagram with points for each month and year.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data for each month and year.

    Keyword Arguments
    -----------------
    - output_path (str or Path)         : Required. Directory where the figure is saved.
    - variable_name (str)               : Short name used to infer full variable name and unit.
    - variable (str)                    : Full variable name (e.g., "Chlorophyll").
    - unit (str)                        : Unit of measurement (e.g., "mg Chl/m³").

    Plot Options
    ------------
    - title (str)                       : Plot title.
    - figsize (tuple of float)          : Figure size in inches.
    - dpi (int)                         : Plot resolution.
    - title_pad (float)                 : Padding between title and plot.
    - title_fontsize (int)              : Font size of the title.
    - title_fontweight (str or int)     : Font weight of the title.

    Raises
    ------
    ValueError
        If neither 'variable_name' nor both 'variable' and 'unit' are provided.

    Saves
    -----
    Unified_Taylor_Diagram.png : Saved in output_path
    """
    # ----- CONVERT KWARGS TO NAMESPACE -----
    options = SimpleNamespace(**kwargs)
    options.output_path = getattr(options, "output_path", None)
    options.variable = getattr(options, "variable", None)
    options.unit = getattr(options, "unit", None)
    options.variable_name = getattr(options, "variable_name", None)

    # ----- RETRIEVE NECESSARY OUTPUT PATH AND VARIABLE/UNIT INFO -----
    if options.output_path is None:
        raise ValueError("output_path must be specified either in kwargs or default options.")

    if options.variable_name is not None:
        variable, unit = get_variable_label_unit(options.variable_name)
        options.variable = options.variable or variable
        options.unit = options.unit or unit
    else:
        if options.variable is None or options.unit is None:
            raise ValueError(
                "If 'variable_name' is not provided, both 'variable' and 'unit' must be specified in kwargs or defaults."
            )

    # ----- PREPARE OUTPUT DIRECTORY -----
    output_path = Path(options.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ----- BUILD DATAFRAME AND YEARS -----
    df, years = build_all_points(data_dict)

    # ----- EXTRACT PLOTTING OPTIONS -----
    plt_opts = extract_options(kwargs, default_monthly_plt_options)

    # ----- CREATE FIGURE -----
    plt.figure(figsize=plt_opts["figsize"], dpi=plt_opts["dpi"])
    
    # ----- SET TITLE -----
    plt.title(
        kwargs.get("title", f"Monthly Taylor Diagram (Normalized Stats) | {options.variable}"),
        pad=plt_opts["title_pad"],
        fontsize=plt_opts["title_fontsize"],
        fontweight=plt_opts["title_fontweight"],
    )

    # ----- BASE DIAGRAM -----
    sm.taylor_diagram(
        df["sdev"].values,
        df["crmsd"].values,
        df["ccoef"].values,
        **default_monthly_taylor_base_options
    )

    # ----- REFERENCE POINT AND LABEL -----
    ref = df[df["year"] == "Ref"].iloc[0]
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    label_y = y_min - 0.11 * (y_max - y_min)

    tickrms = default_monthly_taylor_base_options.get("tickrms")
    first_tickrms = tickrms[0] if isinstance(tickrms, (list, tuple)) and tickrms else 0.5

    plt.text(ref["sdev"], label_y, "Ref", ha="center", va="center", fontsize=16, fontweight='bold', color='r')
    if default_monthly_taylor_base_options.get('titleRMS', 'off') == 'off':
        plt.text(ref["sdev"] + first_tickrms, label_y, 'RMSD', fontsize=12,
                 ha='center', va='center', fontweight='bold', color=(0.0, 0.6, 0.0))

    # ----- OVERLAY REFERENCE MARKER -----
    sm.taylor_diagram(
        np.array([ref["sdev"], ref["sdev"]]),
        np.array([ref["crmsd"], ref["crmsd"]]),
        np.array([ref["ccoef"], ref["ccoef"]]),
        **default_monthly_ref_marker_options
    )

    # ----- PLOT MONTHLY POINTS -----
    non_ref = df[df["year"] != "Ref"]
    for _, row in non_ref.iterrows():
        month_color = default_month_colors[row["month"]]
        year_index = years.index(row["year"]) if row["year"] in years else -1
        marker_shape = default_marker_shapes[year_index % len(default_marker_shapes)]

        sm.taylor_diagram(
            np.array([row["sdev"], row["sdev"]]),
            np.array([row["crmsd"], row["crmsd"]]),
            np.array([row["ccoef"], row["ccoef"]]),
            markersymbol=marker_shape,
            markercolors={
                "face": month_color,
                "edge": default_monthly_data_marker_options["markercolors_edge"],
            },
            markersize=default_monthly_data_marker_options["markersize"],
            alpha=default_monthly_data_marker_options["alpha"],
            overlay=default_monthly_data_marker_options["overlay"],
        )

    # ----- SAVE AND PRINT FIGURE -----
    save_path = output_path / "Unified_Taylor_Diagram.png"
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
###############################################################################