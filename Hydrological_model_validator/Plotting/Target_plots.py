import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm
from pathlib import Path

from .formatting import fill_annular_region, get_variable_label_unit

from ..Processing.Target_computations import (compute_normalised_target_stats, 
                                              compute_normalised_target_stats_by_month,
                                              compute_target_extent_monthly,
                                              compute_target_extent_yearly)

from ..Processing.data_alignment import extract_mod_sat_keys

from ..Processing.utils import extract_options

from .default_target_options import (default_target_base_options,
                                     default_target_overlay_options,
                                     default_target_data_marker_options,
                                     default_target_plt_options,
                                     default_month_markers,
                                     default_target_monthly_plt_options,
                                     default_target_monthly_base_options,
                                     default_target_monthly_data_marker_options)

###############################################################################
def comprehensive_target_diagram(data_dict, **kwargs):
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

    # Extract default options (you'll define these dicts externally)
    base_opts = extract_options(kwargs, default_target_base_options, prefix="base_")
    overlay_opts = extract_options(kwargs, default_target_overlay_options, prefix="overlay_")
    data_marker_opts = extract_options(kwargs, default_target_data_marker_options, prefix="data_")
    plt_opts = extract_options(kwargs, default_target_plt_options)

    # Compute normalized statistics
    bias, crmsd, rmsd, labels = compute_normalised_target_stats(data_dict)

    marker_shapes = kwargs.pop("marker_shapes", ["P", "o", "X", "s", "D", "^", "v", "p", "h", "*"])

    extent = compute_target_extent_yearly(data_dict)

    fig, ax = plt.subplots(figsize=plt_opts.get("figsize"), dpi=plt_opts.get("dpi"))

    # Allow user to define threshold boundaries
    zone_bounds = kwargs.get("zone_bounds", (0.5, 0.7))  # Default (0.5, 0.7)
    assert len(zone_bounds) == 2 and all(isinstance(x, (int, float)) for x in zone_bounds), \
        "'zone_bounds' must be a tuple of two numbers, e.g., (0.5, 0.7)"

    bound1, bound2 = zone_bounds

    # Draw shaded areas with customizable boundaries
    ax = plt.gca()
    for r_start, r_end, color in [
        (0, bound1, 'lightgreen'),
        (bound1, bound2, 'khaki'),
        (bound2, extent, 'lightcoral')
    ]:
        fill_annular_region(ax, r_start, r_end, color=color, alpha=0.4)
        
    ax.set_axisbelow(True)

    # Base transparent target diagram with circlelines
    sm.target_diagram(1.0, extent, extent,
                      markerLabelColor=base_opts.get('markerLabelColor'),
                      alpha=base_opts.get('alpha'),
                      markersize=base_opts.get('markersize'),
                      circlelinespec=base_opts.get('circlelinespec'))

    # Overlay circles
    sm.target_diagram(1.0, extent, extent,
                      markerLabelColor=overlay_opts.get('markerLabelColor'),
                      alpha=overlay_opts.get('alpha'),
                      markersize=overlay_opts.get('markersize'),
                      circlelinespec=overlay_opts.get('circlelinespec'),
                      circles=overlay_opts.get('circles'))

    # Plot data points
    for i, (b, c, r, label) in enumerate(zip(bias, crmsd, rmsd, labels)):
        sm.target_diagram(b, c, r,
                          markerLabelColor=data_marker_opts.get('markerLabelColor'),
                          markersymbol=marker_shapes[i % len(marker_shapes)],
                          markersize=data_marker_opts.get('markersize'),
                          alpha=data_marker_opts.get('alpha'),
                          circles=data_marker_opts.get('circles'),
                          overlay=data_marker_opts.get('overlay'))

    plt.title(kwargs.get("title", f"Yearly Target Plot (Normalized Stats) | {variable}"),
              pad=plt_opts.get("title_pad"),
              fontweight=plt_opts.get("title_fontweight"))

    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / kwargs.get("filename", "Unified_Target_Diagram.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=plt_opts.get("dpi"))
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
###############################################################################

###############################################################################
def target_diagram_by_month(data_dict, **kwargs):
    plt.close('all')

    output_path = Path(kwargs.pop("output_path", "./"))
    output_path.mkdir(parents=True, exist_ok=True)

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

    # Extract all relevant options from kwargs or use defaults
    markers = kwargs.pop("markers", default_month_markers["markers"])
    month_colors = kwargs.pop("month_colors", default_month_markers["month_colors"])

    plt_opts = extract_options(kwargs, default_target_monthly_plt_options)
    base_opts = extract_options(kwargs, default_target_monthly_base_options, prefix="base_")
    overlay_opts = extract_options(kwargs, default_target_overlay_options, prefix="overlay_")
    data_marker_opts = extract_options(kwargs, default_target_monthly_data_marker_options, prefix="data_")

    extent = compute_target_extent_monthly(data_dict)

    plt.figure(figsize=plt_opts.get("figsize"))

    plt.title(
        kwargs.get("title", f"Monthly Target Plot (Normalized Stats) | {variable}"),
        fontsize=plt_opts.get("title_fontsize"),
        pad=plt_opts.get("title_pad"),
        fontweight=plt_opts.get("title_fontweight")
    )

    # Base target diagram circles
    sm.target_diagram(
        np.array([1.0]), np.array([extent]), np.array([1.0]),
        markerLabelColor=base_opts.get("markerLabelColor"),
        markersize=base_opts.get("markersize"),
        alpha=base_opts.get("alpha"),
        circles=base_opts.get("overlay_circles", [0.7, 1.0] + list(np.arange(2.0, extent + 1e-6, 1.0))),
        circlelinespec=base_opts.get("circlelinespec")
    )
    
    # Overlay
    sm.target_diagram(
        np.array([1.0]), np.array([extent]), np.array([1.0]),
        markerLabelColor=overlay_opts.get("markerLabelColor"),
        markersize=overlay_opts.get("markersize"),
        alpha=overlay_opts.get("alpha"),
        circles=overlay_opts.get("circles"),
        circlelinespec=overlay_opts.get("circlelinespec")
    )

    # Allow user to define threshold boundaries
    zone_bounds = kwargs.get("zone_bounds", (0.5, 0.7))
    assert len(zone_bounds) == 2 and all(isinstance(x, (int, float)) for x in zone_bounds), \
        "'zone_bounds' must be a tuple of two numbers, e.g., (0.5, 0.7)"

    bound1, bound2 = zone_bounds

    # Draw shaded areas with customizable boundaries
    ax = plt.gca()
    for r_start, r_end, color in [
        (0, bound1, 'lightgreen'),
        (bound1, bound2, 'khaki'),
        (bound2, extent, 'lightcoral')
    ]:
        fill_annular_region(ax, r_start, r_end, color=color, alpha=0.4)

    # Extract years from data_dict
    mod_key, _ = extract_mod_sat_keys(data_dict)
    years = list(data_dict[mod_key].keys())

    # Plot data points month by month
    for month_index in range(12):
        try:
            bias, crmsd, rmsd, labels = compute_normalised_target_stats_by_month(data_dict, month_index)
        except ValueError:
            continue

        color = month_colors[month_index % len(month_colors)]

        for i, (b, c, r, label) in enumerate(zip(bias, crmsd, rmsd, labels)):
            year = label  # Assuming label is year as string or int
            year_index = years.index(str(year)) if str(year) in years else i
            marker = markers[year_index % len(markers)]

            sm.target_diagram(
                np.array([b]), np.array([c]), np.array([r]),
                markercolors={"face": color, "edge": data_marker_opts.get("edge_color")},
                markersymbol=marker,
                markersize=data_marker_opts.get("markersize"),
                alpha=data_marker_opts.get("alpha"),
                overlay=data_marker_opts.get("overlay"),
                circles=data_marker_opts.get("circles")
            )

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=plt_opts.get("tick_labelsize"))
    ax.xaxis.label.set_size(plt_opts.get("axis_labelsize"))
    ax.yaxis.label.set_size(plt_opts.get("axis_labelsize"))

    save_path = output_path / kwargs.get("filename", "Unified_Target_Diagram.png")
    plt.savefig(save_path, dpi=plt_opts.get("dpi"))
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()
###############################################################################