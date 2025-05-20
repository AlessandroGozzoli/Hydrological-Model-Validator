import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm
from pathlib import Path
import sys
import os

from Auxilliary import fill_annular_region, get_variable_label_unit

WDIR = os.getcwd()
ProcessingDIR = Path(WDIR, "Processing/")
sys.path.append(str(ProcessingDIR))  # Add the folder to the system path
from Target_computations import (compute_normalised_target_stats, 
                                 compute_normalised_target_stats_by_month,
                                 compute_target_extent_monthly,
                                 compute_target_extent_yearly)
from Corollary import extract_mod_sat_keys

def comprehensive_target_diagram(data_dict, output_path, variable_name):
    """
    Generate and save a normalised Target diagram using preprocessed statistics.

    Parameters:
        taylor_dict (dict): Dictionary with model and satellite time series per year.
        output_path (str or Path): Directory to save the output plot.
    """

    plt.close('all')
    
    variable, unit = get_variable_label_unit(variable_name)

    # Compute normalised statistics
    bias, crmsd, rmsd, labels = compute_normalised_target_stats(data_dict)
    
    marker_shapes = ["P", "o", "X", "s", "D", "^", "v", "p", "h", "*"]
    
    extent = compute_target_extent_yearly(data_dict)
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Fill the three performance zones
    fill_annular_region(ax, 0.0, 0.5, color='lightgreen', alpha=0.4)   # Excellent zone
    fill_annular_region(ax, 0.5, 0.7, color='khaki', alpha=0.4)        # Good zone
    fill_annular_region(ax, 0.7, extent, color='lightcoral', alpha=0.3)   # Acceptable zone

    # Set axes below other plot elements
    ax.set_axisbelow(True)
    
    # Plot the base target diagram with transparent markers and specific circle lines
    sm.target_diagram(1.0, extent, extent,
                markerLabelColor=(0.0, 0.0, 0.0),
                  alpha=0.0,
                  markersize=0,
                  circlelinespec='-.r')

    # Overlay additional circles if needed
    sm.target_diagram(1.0, extent, extent,
                  markerLabelColor=(0.0, 0.0, 0.0),
                  alpha=0.0,
                  markersize=0,
                  circlelinespec=':b',
                  circles=[0.5])

    # Plot your data points with markers
    for i, (b, c, r, label) in enumerate(zip(bias, crmsd, rmsd, labels)):
        sm.target_diagram(b, c, r,
                      markerLabelColor='r',
                      markersymbol=marker_shapes[i % len(marker_shapes)],
                      markersize=10,
                      alpha=0.8,
                      circles=[0.0],
                      overlay='on')

    plt.title(f"Normalised Target Plot (Yearly Performance) | {variable}", pad=40, fontweight='bold')

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = Path(output_path, "Target_plot_summary_norm.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()

def target_diagram_by_month(data_dict, output_path, variable_name):
    MARKERS = ["P", "o", "X", "s", "D", "^", "v", "p", "h", "*"]
    MONTH_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
        "#17becf", "#393b79", "#637939", "#8c6d31"
    ]

    variable, unit = get_variable_label_unit(variable_name)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    extent = compute_target_extent_monthly(data_dict)

    plt.figure(figsize=(11, 10))
    plt.title(f"Monthly Target Diagram (Normalized Stats) | {variable}", fontsize=18, pad=55, fontweight='bold')

    # Base target diagram circles
    sm.target_diagram(
        np.array([1.0]), np.array([extent]), np.array([1.0]),
        markerLabelColor='none', markersize=0, alpha=0.0,
        circles=[0.5], circlelinespec='b:'
    )
    sm.target_diagram(
        np.array([1.0]), np.array([extent]), np.array([1.0]),
        markerLabelColor='none', markersize=0, alpha=0.0,
        circles=[0.7, 1.0] + list(np.arange(2.0, extent + 1e-6, 1.0)),
        circlelinespec='r-.'
    )

    # Shaded areas
    theta = np.linspace(0, 2 * np.pi, 300)
    for r_start, r_end, color in [(0, 0.5, 'lightgreen'), (0.5, 0.7, 'khaki'), (0.7, extent, 'lightcoral')]:
        x_outer = r_end * np.cos(theta)
        y_outer = r_end * np.sin(theta)
        x_inner = r_start * np.cos(theta)
        y_inner = r_start * np.sin(theta)
        plt.fill(
            np.concatenate([x_outer, x_inner[::-1]]),
            np.concatenate([y_outer, y_inner[::-1]]),
            color=color, alpha=0.4, zorder=0
        )

    # Extract year list
    mod_key, _ = extract_mod_sat_keys(data_dict)
    years = list(data_dict[mod_key].keys())

    # Plot data points month by month
    for month_index in range(12):
        try:
            bias, crmsd, rmsd, labels = compute_normalised_target_stats_by_month(data_dict, month_index)
        except ValueError:
            continue

        color = MONTH_COLORS[month_index % len(MONTH_COLORS)]

        for i, (b, c, r, label) in enumerate(zip(bias, crmsd, rmsd, labels)):
            # Match label to year, cycle through marker shapes
            year = label  # Assuming label is year as string or int
            year_index = years.index(str(year)) if str(year) in years else i
            marker = MARKERS[year_index % len(MARKERS)]

            sm.target_diagram(
                np.array([b]), np.array([c]), np.array([r]),
                markercolors={"face": color, "edge": "k"},
                markersymbol=marker,
                markersize=10,
                alpha=0.8,
                overlay='on',
                circles=[0.0]
            )

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    save_path = output_path / "Unified_Target_Diagram.png"
    plt.savefig(save_path, dpi=300)
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()

    print("\033[92m✅ Unified Target Diagram has been plotted!\033[0m")
    print("*" * 45)