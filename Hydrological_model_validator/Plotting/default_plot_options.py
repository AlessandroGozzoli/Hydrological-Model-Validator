import itertools

###############################################################################
default_plot_options_ts = {
    "BA": False,
    "variable": None,
    "unit": None,
    "variable_name": None,          # added
    "output_path": None,            # added, must be specified
    "figsize": (20, 8),
    "dpi": 300,
    "color_palette": itertools.cycle(['#BF636B', '#5976A2']),
    "title_fontsize": 20,
    "label_fontsize": 14,
    "legend_fontsize": 12,
    "bias_title_fontsize": 18,
    "line_width": 1,
    "savefig_kwargs": {},
    "smooth7": False,               # added default
    "smooth30": False,              # added default
}
###############################################################################

###############################################################################
default_plot_options_scatter = {
    "output_path": "./plots",
    "variable": None,
    "unit": None,
    "BA": False,
    "figsize": (10, 8),
    "dpi": 300,
    "color": '#5976A2',
    "alpha": 0.7,
    "marker_size": 50,
    "title_fontsize": 20,
    "label_fontsize": 15,
    "line_width": 2,
    "tick_labelsize": 13,
    "legend_fontsize": 12,
}
###############################################################################

###############################################################################
default_scatter_by_season_options = {
    'output_path': None,
    "variable": None,
    "unit": None,
    'BA': False,
    'figsize': (10, 8),
    'dpi': 300,
    'season_colors': {
        'DJF': '#808080',
        'MAM': '#008000',
        'JJA': '#FF0000',
        'SON': '#FFD700'
    },
    'alpha': 0.7,
    'marker_size': 50,
    'title_fontsize': 20,
    'label_fontsize': 15,
    'legend_fontsize': 12,
    'line_width': 2,
    'tick_labelsize': 13,
}
###############################################################################

###############################################################################
default_boxplot_options = {
    'output_path': None,
    "variable": None,
    "unit": None,
    'figsize': (16, 6),
    'dpi': 300,
    'palette': ['#5976A2', '#BF636B'] * 12,
    'showfliers': True,
    'title_fontsize': 16,
    'title_fontweight': 'bold',
    'ylabel_fontsize': 14,
    'xlabel': '',
    'grid_alpha': 0.5,
    'xtick_rotation': 45,
    'tick_width': 2,
}
###############################################################################

###############################################################################
default_violinplot_options = {
    'output_path': None,
    "variable": None,
    "variable_name": None,
    "unit": None,
    'figsize': (16, 6),
    'dpi': 300,
    'palette': ['#5976A2', '#BF636B'] * 12,
    'cut': 0,
    'title_fontsize': 16,
    'title_fontweight': 'bold',
    'ylabel_fontsize': 14,
    'xlabel_fontsize': 12,
    'xtick_rotation': 45,
    'grid_alpha': 0.5,
    'tick_width': 2,
    'spine_linewidth': 2,
}
###############################################################################

###############################################################################
default_efficiency_plot_options = {
    'metric_name': None,
    'title': '',
    'overall_value': 0.0,
    'y_label': '',
    'output_path': None,
    'figsize': (10, 6),
    'dpi': 300,
    'line_color': 'blue',
    'line_width': 1.75,
    'zero_line': {
        'show': True,
        'style': '-.',
        'width': 3,
        'color': 'red',
        'label': 'Zero Reference'
    },
    'overall_line': {
        'style': '--',
        'width': 2,
        'color': 'black',
        'label': 'Overall'
    },
    'marker_size': 10,
    'marker_edge_color': 'black',
    'marker_edge_width': 1.2,
    'title_fontsize': 14,
    'ylabel_fontsize': 12,
    'xtick_rotation': 45,
    'tick_width': 2,
    'legend_loc': 'lower right',
    'grid_style': '--',
    'spine_width': 2,
}
###############################################################################

###############################################################################
spatial_efficiency_defaults = {
    "cmap": "OrangeGreen",
    "vmin": -1.0,
    "vmax": 1.0,
    "suffix": "(Model - Satellite)",
    "suptitle_fontsize": 20,
    "suptitle_fontweight": "bold",
    "suptitle_y": 1.03,
    "title_fontsize": 16,
    "title_fontweight": "bold",
    "cbar_labelsize": 12,
    "cbar_labelpad": 10,
    "cbar_shrink": 0.6,
    "cbar_ticks": 11,
    "figsize_per_plot": (5, 4),
    "max_cols": 3,
    "epsilon": 0.06,
    "lat_offset_base": 0.2702044,
    "gridline_color": "gray",
    "gridline_style": "--",
    "gridline_alpha": 0.7,
    "gridline_dms": True,
    "gridline_labels_top": True,
    "gridline_labels_right": True,
    "projection": "PlateCarree",
    "resolution": "10m",
    "land_color": "lightgray",
    "show": True,
    "block": False,
    "dpi": 100,
    "unit" : None,
}
###############################################################################

###############################################################################
default_error_timeseries_options = {
    'fig_width': 25,
    'fig_height_per_plot': 3,
    'mean_bias_color': 'tab:blue',
    'unbiased_rmse_color': 'tab:orange',
    'std_error_color': 'tab:green',
    'correlation_color': 'tab:red',
    'cloud_cover_color': 'tab:gray',
    'cloud_cover_smoothed_color': 'black',
    'title_fontsize': 24,
    'title_fontweight': 'bold',
    'label_fontsize': 16,
    'grid_color': 'gray',
    'grid_linestyle': '--',
    'grid_alpha': 0.7,
    'cloud_cover_rolling_window': 30,
    'filename_template': "Error_Decomposition_Timeseries_{variable_name}.png",
    'sharex': True,
    'spine_linewidth': 2,
    'spine_edgecolor': 'black',
}
###############################################################################

###############################################################################
default_spectral = {
    # Figure and axes
    'figsize': (14, 6),

    # Fontsizes
    'xlabel_fontsize': 14,
    'ylabel_fontsize': 14,
    'title_fontsize': 16,
    'title_fontweight': 'bold',
    'tick_labelsize': 12,

    # Grid style
    'grid_color': 'gray',
    'grid_alpha': 0.7,
    'grid_linestyle': '--',

    # Frequency x-axis limits
    'freq_xlim': (0, 0.1),

    # Spine styling
    'spine_linewidth': 1.5,
    'spine_edgecolor': 'black',

    # Linestyles for additional cloud covers in CSD plot
    'additional_linestyles': ['--', ':', '-.'],
}