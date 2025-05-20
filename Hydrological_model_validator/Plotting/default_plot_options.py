import itertools
from pathlib import Path

default_plot_options_ts = {
    "output_path": Path("./plots"),        # default output folder
    "variable_name": "UnknownVar",         # default variable name if not provided
    "BA": False,
    "variable": None,
    "unit": None,
    "figsize": (20, 10),
    "dpi": 300,
    "color_palette": itertools.cycle(['#BF636B', '#5976A2', '#70A494', '#D98B5F', '#D3A4BD', '#7294D4']),
    "title_fontsize": 20,
    "label_fontsize": 14,
    "legend_fontsize": 12,
    "bias_title_fontsize": 18,
    "line_width": 2,
    "savefig_kwargs": {},
}

default_plot_options_scatter = {
    "output_path": "./plots",
    "variable_name": "SST",
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

default_scatter_by_season_options = {
    'output_path': None,
    'variable_name': None,
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

default_boxplot_options = {
    'output_path': None,
    'variable_name': None,
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

default_violinplot_options = {
    'output_path': None,
    'variable_name': None,
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