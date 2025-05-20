import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm
from pathlib import Path
import sys
import os

WDIR = os.getcwd()
ProcessingDIR = Path(WDIR, "Processing/")
sys.path.append(str(ProcessingDIR))  # Add the folder to the system path
from Taylor_computations import (compute_yearly_taylor_stats, 
                                 build_all_points)

from Auxilliary import get_variable_label_unit

def comprehensive_taylor_diagram(data_dict, output_path, variable_name):
    """
    Generate and plot a Taylor diagram for model vs reference data in the provided taylor_dict.
    """
    stats_by_year, std_ref = compute_yearly_taylor_stats(data_dict)
    
    variable, unit = get_variable_label_unit(variable_name)

    # Add reference point
    labels = ["Ref"] + [entry[0] for entry in stats_by_year]
    sdev = np.array([std_ref] + [entry[1] for entry in stats_by_year])
    crmsd = np.array([0.0] + [entry[2] for entry in stats_by_year])
    ccoef = np.array([1.0] + [entry[3] for entry in stats_by_year])
    
    sdev = sdev / std_ref
    crmsd = crmsd / std_ref

    marker_shapes = ["P", "o", "X", "s", "D", "^", "v", "p", "h", "*"]

    plt.figure(figsize=(7, 7), dpi=300)
    plt.title(f"Taylor Diagram (Yearly Performance) | {variable}", pad=45, fontsize=16, fontweight='bold')

    # Base diagram
    sm.taylor_diagram(
        sdev, crmsd, ccoef,
        markersymbol='none',
        markercolors={"face": "none", "edge": "none"},
        markersize=0,
        alpha=0,
        styleobs='-',
        colobs='r',
        titleSTD='on',
        titleRMS='off',
        colrms=(0.0, 0.6, 0.0),
        widthcor=1.5,
        widthstd=1.5,
        widthobs=2.5,
        showlabelsrms='on',
        rmslabelformat='0',
        tickrmsangle=120,
        labelweight='bold'
    )
    
    ax = plt.gca()
    ylim = ax.get_ylim()

    # Put labels 5% below the bottom y-limit
    label_y = ylim[0] - 0.11 * (ylim[1] - ylim[0])

    # RMSD axis label
    plt.text(sdev[0], label_y, "Ref", ha="center", va="center", fontsize=16, fontweight='bold', color='r')
    plt.text(sdev[0]+0.5, label_y, 'RMSD', fontsize=12, ha='center', va='center', fontweight='bold', color=(0.0, 0.6, 0.0))

    # Reference marker
    sm.taylor_diagram(
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        markersymbol='.',
        markercolors={"face": 'r', "edge": 'r'},
        markersize=25,
        alpha=1.0,
        overlay='on',
        tickrms=[0.0]
    )

    # Data markers
    for i, (x, y, c, label) in enumerate(zip(sdev[1:], crmsd[1:], ccoef[1:], labels[1:])):
        sm.taylor_diagram(
            np.array([x, x]),
            np.array([y, y]),
            np.array([c, c]),
            markersymbol=marker_shapes[i % len(marker_shapes)],
            markercolors={"face": "#BF636B", "edge": "#BF636B"},
            markersize=10,
            alpha=0.8,
            styleobs='-',
            colobs='r',
            widthobs=2.5,
            showlabelsrms='off',
            overlay='on'
        )

    # Save and show
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / 'Taylor_diagram_summary.png'
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()

def monthly_taylor_diagram(data_dict, output_path, variable_name):
    df, years = build_all_points(data_dict)
    """
    Plot the Taylor diagram given the DataFrame of stats and years list.
    Saves figure to output_path.
    """
    MARKERS = [
        "P", "o", "X", "s", "D", "^", "v", "p", "h", "*"
    ]
    MONTH_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
        "#17becf", "#393b79", "#637939", "#8c6d31"
    ]
    
    variable, unit = get_variable_label_unit(variable_name)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10), dpi=300)
    plt.title(f"Monthly Taylor Diagram (Normalized Stats) | {variable}", pad=65, fontsize=18, fontweight='bold')

    # Draw full layout without markers
    sm.taylor_diagram(
        df["sdev"].values, df["crmsd"].values, df["ccoef"].values,
        markersymbol='none',
        markercolors={"face": "none", "edge": "none"},
        markersize=0,
        alpha=0,
        styleobs='-',
        colobs='r',
        titleSTD='on',
        titleRMS='off',
        colrms=(0.0, 0.6, 0.0),
        widthcor=1.5,
        widthstd=1.5,
        widthobs=2.5,
        widthrms=2,
        showlabelsrms='on',
        rmslabelformat='0',
        tickrmsangle=120,
        labelweight='bold',
        tickrms=[0.5, 1.0, 1.5]
    )
    
    ax = plt.gca()
    ylim = ax.get_ylim()

    # Put labels 5% below the bottom y-limit
    label_y = ylim[0] - 0.11 * (ylim[1] - ylim[0])
    
    ref = df[df["year"] == "Ref"].iloc[0]
    plt.text(ref["sdev"], label_y, "Ref", ha="center", va="bottom", fontsize=13, fontweight='bold', color='r')
    plt.text(ref["sdev"]+0.5, label_y, 'RMSD', fontsize=13, ha='center', va='bottom', fontweight='bold', color=(0.0, 0.6, 0.0))

    # Reference marker
    sm.taylor_diagram(
        np.array([ref["sdev"], ref["sdev"]]),
        np.array([ref["crmsd"], ref["crmsd"]]),
        np.array([ref["ccoef"], ref["ccoef"]]),
        markersymbol='.',
        markercolors={"face": "r", "edge": "r"},
        markersize=25,
        alpha=1.0,
        overlay='on',
        tickrms=[0.0]
    )

    # Plot all other data points
    non_ref = df[df["year"] != "Ref"]
    for _, row in non_ref.iterrows():
        month_color = MONTH_COLORS[row["month"]]
        year_index = years.index(row["year"]) if row["year"] in years else -1
        marker_shape = MARKERS[year_index % len(MARKERS)]

        sm.taylor_diagram(
            np.array([row["sdev"], row["sdev"]]),
            np.array([row["crmsd"], row["crmsd"]]),
            np.array([row["ccoef"], row["ccoef"]]),
            markersymbol=marker_shape,
            markercolors={"face": month_color, "edge": "black"},
            markersize=12.5,
            alpha=0.7,
            overlay='on'
        )

    save_path = output_path / "Unified_Taylor_Diagram.png"
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(3)
    plt.close()