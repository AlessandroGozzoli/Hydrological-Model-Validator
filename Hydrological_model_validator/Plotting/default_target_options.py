default_target_base_options = {
    "markerLabelColor": (0.0, 0.0, 0.0),
    "alpha": 0.0,
    "markersize": 0,
    "circlelinespec": "r-.",
}

default_target_overlay_options = {
    "markerLabelColor": (0.0, 0.0, 0.0),
    "alpha": 0.0,
    "markersize": 0,
    "circlelinespec": ":b",
    "circles": [0.5],
}

default_target_data_marker_options = {
    "markerLabelColor": "r",
    "markersize": 10,
    "alpha": 0.8,
    "circles": [0.0],
    "overlay": "on",
}

default_target_plt_options = {
    "figsize": (7, 6),
    "dpi": 300,
    "title_pad": 40,
    "title_fontweight": "bold",
    "filename": "Target_plot_summary_norm.png",
}

default_month_markers = {
    "markers": ["P", "o", "X", "s", "D", "^", "v", "p", "h", "*"],
    "month_colors": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
        "#17becf", "#393b79", "#637939", "#8c6d31"
    ],
}

default_target_monthly_plt_options = {
    "figsize": (11, 10),
    "title_fontsize": 18,
    "title_pad": 55,
    "title_fontweight": "bold",
    "tick_labelsize": 14,
    "axis_labelsize": 16,
    "dpi": 300,
    "filename": "Unified_Target_Diagram.png",
}

default_target_monthly_base_options = {
    "markerLabelColor": "none",
    "markersize": 0,
    "alpha": 0.0,
    "circles": [0.5],
    "circlelinespec": "r-.",
}

default_target_monthly_data_marker_options = {
    "edge_color": "k",
    "markersize": 10,
    "alpha": 0.8,
    "overlay": "on",
    "circles": [0.0],
}
