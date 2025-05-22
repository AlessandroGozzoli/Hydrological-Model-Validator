import itertools

###############################################################################
default_plot_options_ts = {
    "BA": False,
    "variable": None,
    "unit": None,
    "figsize": (20, 10),
    "dpi": 300,
    "color_palette": itertools.cycle(['#BF636B', '#5976A2']),
    "title_fontsize": 20,
    "label_fontsize": 14,
    "legend_fontsize": 12,
    "bias_title_fontsize": 18,
    "line_width": 1,
    "savefig_kwargs": {},
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
    "pause_time": 3,
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
    'pause_time': 2,
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
    'pause_time': 3,
}
###############################################################################

###############################################################################
default_violinplot_options = {
    'output_path': None,
    "variable": None,
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
    'pause_time': 3,
}
###############################################################################

###############################################################################
default_efficiency_plot_options = {
    'metric_name': '',
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
    'pause_time': 3,
}
###############################################################################