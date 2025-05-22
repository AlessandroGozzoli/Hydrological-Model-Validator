import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm
from pathlib import Path
import random
from matplotlib.lines import Line2D


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
    output_path = kwargs.pop('output_path', './')

    # Extract variable info
    variable = kwargs.pop('variable', None)
    unit = kwargs.pop('unit', None)
    variable_name = kwargs.pop('variable_name', None)

    if variable is None or unit is None:
        if variable_name is not None:
            variable, unit = get_variable_label_unit(variable_name)
        else:
            raise ValueError(
                "You must provide either 'variable' and 'unit' or 'variable_name' in kwargs"
            )

    stats_by_year, std_ref = compute_yearly_taylor_stats(data_dict)

    # Marker shapes with random filling if needed
    user_shapes = kwargs.pop("marker_shapes", default_marker_shapes)
    labels = ["Ref"] + [entry[0] for entry in stats_by_year]
    if len(user_shapes) < len(labels) - 1:
        all_markers = [m for m in Line2D.markers.keys() if isinstance(m, str) and m not in user_shapes and m != ' ']
        user_shapes += random.sample(all_markers, len(labels) - 1 - len(user_shapes))

    sdev = np.array([std_ref] + [entry[1] for entry in stats_by_year]) / std_ref
    crmsd = np.array([0.0] + [entry[2] for entry in stats_by_year]) / std_ref
    ccoef = np.array([1.0] + [entry[3] for entry in stats_by_year])

    base_opts = extract_options(kwargs, default_taylor_base_options)
    ref_opts = extract_options(kwargs, default_taylor_ref_marker_options, prefix="Ref_")
    data_opts = extract_options(kwargs, default_taylor_data_marker_options)
    plt_opts = extract_options(kwargs, default_taylor_plt_options)

    sdev_ref_override = kwargs.pop("sdevRef", 1.0)
    crmsd_ref_override = kwargs.pop("crmsdRef", 0.0)
    ccoef_ref_override = kwargs.pop("ccoefRef", 1.0)

    plt.figure(figsize=plt_opts.get('figsize'), dpi=plt_opts.get('dpi'))
    plt.title(
        plt_opts.get('title', f"Yearly Taylor Diagram (Normalized Stats) | {variable}"),
        pad=plt_opts.get('title_pad'),
        fontsize=plt_opts.get('title_fontsize'),
        fontweight=plt_opts.get('title_fontweight')
    )

    sm.taylor_diagram(
        sdev, crmsd, ccoef,
        markersymbol='none',
        markercolors={"face": "none", "edge": "none"},
        markersize=0,
        alpha=0,
        **base_opts
    )

    ax = plt.gca()
    label_y = ax.get_ylim()[0] - 0.11 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    
    tickrms = base_opts.get("tickrms")
    first_tickrms = tickrms[0] if isinstance(tickrms, (list, tuple)) and tickrms else 0.5

    plt.text(sdev[0], label_y, "Ref", ha="center", va="center", fontsize=16, fontweight='bold', color='r')
    if base_opts.get('titleRMS', 'off') == 'off':
        plt.text(sdev[0] + first_tickrms, label_y, 'RMSD', fontsize=12,
                 ha='center', va='center', fontweight='bold', color=(0.0, 0.6, 0.0))

    sm.taylor_diagram(
        np.array([sdev_ref_override] * 2),
        np.array([crmsd_ref_override] * 2),
        np.array([ccoef_ref_override] * 2),
        tickrms=[0.0],
        overlay='on',
        **ref_opts
    )

    for i, (x, y, c) in enumerate(zip(sdev[1:], crmsd[1:], ccoef[1:])):
        sm.taylor_diagram(
            np.array([x, x]),
            np.array([y, y]),
            np.array([c, c]),
            showlabelsrms='off',
            overlay='on',
            markersymbol=user_shapes[i],
            **data_opts
        )

    Path(output_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_path) / 'Taylor_diagram_summary.png')
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
###############################################################################

###############################################################################
def monthly_taylor_diagram(data_dict, **kwargs):
    output_path = Path(kwargs.pop("output_path", "./"))
    variable = kwargs.pop("variable", None)
    unit = kwargs.pop("unit", None)
    variable_name = kwargs.pop("variable_name", None)

    if variable is None or unit is None:
        if variable_name is not None:
            variable, unit = get_variable_label_unit(variable_name)
        else:
            raise ValueError(
                "You must provide either 'variable' and 'unit' or 'variable_name' in kwargs"
            )

    output_path.mkdir(parents=True, exist_ok=True)

    df, years = build_all_points(data_dict)

    # Extract plotting options from kwargs or fallback to defaults
    plt_opts = extract_options(kwargs, default_monthly_plt_options)

    plt.figure(figsize=plt_opts["figsize"], dpi=plt_opts["dpi"])
    plt.title(
        kwargs.get("title", f"Monthly Taylor Diagram (Normalized Stats) | {variable}"),
        pad=plt_opts["title_pad"],
        fontsize=plt_opts["title_fontsize"],
        fontweight=plt_opts["title_fontweight"],
    )

    sm.taylor_diagram(
        df["sdev"].values,
        df["crmsd"].values,
        df["ccoef"].values,
        **default_monthly_taylor_base_options
    )

    ref = df[df["year"] == "Ref"].iloc[0]
    ax = plt.gca()
    label_y = ax.get_ylim()[0] - 0.11 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    
    tickrms = default_monthly_taylor_base_options.get("tickrms")
    first_tickrms = tickrms[0] if isinstance(tickrms, (list, tuple)) and tickrms else 0.5

    plt.text(ref["sdev"], label_y, "Ref", ha="center", va="center", fontsize=16, fontweight='bold', color='r')
    if default_monthly_taylor_base_options.get('titleRMS', 'off') == 'off':
        plt.text(ref["sdev"] + first_tickrms, label_y, 'RMSD', fontsize=12,
                 ha='center', va='center', fontweight='bold', color=(0.0, 0.6, 0.0))

    sm.taylor_diagram(
        np.array([ref["sdev"], ref["sdev"]]),
        np.array([ref["crmsd"], ref["crmsd"]]),
        np.array([ref["ccoef"], ref["ccoef"]]),
        **default_monthly_ref_marker_options
    )

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

    save_path = output_path / "Unified_Taylor_Diagram.png"
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
###############################################################################