import numpy as np
import re
from typing import Tuple, Union, Iterator, Dict, Any
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import cmocean

###############################################################################
def format_unit(unit: str) -> str:
    """
    Formats a chemical unit string into LaTeX math mode with subscripts and exponents.

    Parameters
    ----------
    unit : str
        The unit string to format. Expected formats include:
        - "O2/m3"
        - "mg/L"
        - "PO4"
        - "kg/m2"

    Returns
    -------
    str
        A LaTeX-formatted string with appropriate subscripts and exponents.
        Examples:
            - "$\\frac{O_2}{m^{3}}$"
            - "$mg \\cdot L^{-1}$"
            - "$PO_4$"

    Raises
    ------
    ValueError
        If input is not a non-empty string.

    Examples
    --------
    >>> format_unit("O2/m3")
    '$\\frac{O_2}{m^{3}}$'

    >>> format_unit("mg/L")
    '$\\frac{mg}{L}$'

    >>> format_unit("PO4")
    '$PO_4$'

    >>> format_unit("kg/m2")
    '$\\frac{kg}{m^{2}}$'
    """
    
    if not isinstance(unit, str):
        raise ValueError("Input 'unit' must be a string.")
    if not unit.strip():
        raise ValueError("Input 'unit' cannot be an empty string.")

    def add_subscripts(text: str) -> str:
        return re.sub(r'([A-Za-z]{1,2})(\d+)', r'\1_{\2}', text)

    def add_exponents(text: str) -> str:
        return re.sub(r'([a-zA-Z])(\d+)', r'\1^{\2}', text)

    if '/' in unit:
        numerator, denominator = map(str.strip, unit.split('/', 1))
        return f'$\\frac{{{add_subscripts(numerator)}}}{{{add_exponents(denominator)}}}$'
    
    formatted = add_subscripts(unit)
    formatted = add_exponents(formatted)
    return f'${formatted}$'
###############################################################################

###############################################################################
def get_variable_label_unit(variable_name: str) -> Tuple[str, str]:
    """
    Returns a descriptive label and a LaTeX-formatted unit string for a given variable name.

    Parameters
    ----------
    variable_name : str
        Short variable code or name. Example values include:
        - "SST"
        - "CHL_L3"
        - "CHL_L4"

    Returns
    -------
    tuple of str
        A tuple containing:
        - label : str — A descriptive name for the variable.
        - unit : str — The corresponding LaTeX-formatted unit string.
        If the variable is not recognized, the function returns:
        - (variable_name, '')

    Raises
    ------
    ValueError
        If `variable_name` is not a non-empty string.

    Examples
    --------
    >>> get_variable_label_unit("SST")
    ('Sea Surface Temperature', '[$°C$]')

    >>> get_variable_label_unit("CHL_L3")
    ('Chlorophyll (Level 3)', '[$mg/m^3$]')

    >>> get_variable_label_unit("XYZ")
    ('XYZ', '')
    """
    if not isinstance(variable_name, str):
        raise ValueError("Input 'variable_name' must be a string.")
    if not variable_name.strip():
        raise ValueError("Input 'variable_name' cannot be an empty string.")

    mapping = {
        'SST': ('Sea Surface Temperature', '[$°C$]'),
        'CHL_L3': ('Chlorophyll (Level 3)', '[$mg/m^3$]'),
        'CHL_L4': ('Chlorophyll (Level 4)', '[$mg/m^3$]'),
    }

    return mapping.get(variable_name, (variable_name, ''))

###############################################################################

###############################################################################
def fill_annular_region(ax: Axes, 
                        r_in: float, 
                        r_out: float, 
                        color: str, 
                        alpha: float = 0.3) -> None:
    """
    Fill an annular (ring-shaped) region between two radii on a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object to draw the annular region on.

    r_in : float
        Inner radius of the annulus. Must be non-negative and less than or equal to `r_out`.

    r_out : float
        Outer radius of the annulus. Must be greater than or equal to `r_in`.

    color : str
        Fill color (e.g., 'blue', '#1f77b4').

    alpha : float, optional
        Transparency level of the fill. Must be between 0 and 1.
        Default is 0.3.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If radius values are invalid or alpha is out of bounds.

    Examples
    --------
    >>> fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    >>> fill_annular_region(ax, r_in=1.0, r_out=2.0, color='red', alpha=0.2)
    """
    if not isinstance(ax, Axes):
        raise ValueError("Input 'ax' must be a matplotlib.axes.Axes instance.")
    if not isinstance(r_in, (int, float)) or r_in < 0:
        raise ValueError("Input 'r_in' must be a non-negative number.")
    if not isinstance(r_out, (int, float)) or r_out < r_in:
        raise ValueError("Input 'r_out' must be a number greater than or equal to 'r_in'.")
    if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
        raise ValueError("Input 'alpha' must be a number between 0 and 1.")
    if not isinstance(color, str):
        raise ValueError("Input 'color' must be a string.")

    theta = np.linspace(0, 2 * np.pi, 500)
    x_outer, y_outer = r_out * np.cos(theta), r_out * np.sin(theta)
    x_inner, y_inner = r_in * np.cos(theta[::-1]), r_in * np.sin(theta[::-1])

    x = np.concatenate((x_outer, x_inner))
    y = np.concatenate((y_outer, y_inner))

    ax.fill(x, y, color=color, alpha=alpha, zorder=0)
###############################################################################

###############################################################################    
def get_min_max_for_identity_line(x: Union[np.ndarray, list, tuple],
                                  y: Union[np.ndarray, list, tuple]) -> Tuple[float, float]:
    """
    Compute the global minimum and maximum from two numeric sequences,
    suitable for setting axis limits for an identity line (y = x).

    Parameters
    ----------
    x : array-like
        First numeric sequence (list, tuple, or NumPy array). NaNs are ignored.

    y : array-like
        Second numeric sequence (list, tuple, or NumPy array). NaNs are ignored.

    Returns
    -------
    tuple of float
        (min_value, max_value) — The combined minimum and maximum of `x` and `y`,
        excluding NaNs.

    Raises
    ------
    ValueError
        If either input is empty or contains no finite values.

    Examples
    --------
    >>> get_min_max_for_identity_line([1, 2, 3], [4, 5, 6])
    (1.0, 6.0)

    >>> get_min_max_for_identity_line([np.nan, 2], [1, np.nan])
    (1.0, 2.0)
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.size == 0 or np.all(np.isnan(x_arr)):
        raise ValueError("Input 'x' cannot be empty or all NaNs.")
    if y_arr.size == 0 or np.all(np.isnan(y_arr)):
        raise ValueError("Input 'y' cannot be empty or all NaNs.")

    min_val = min(np.nanmin(x_arr), np.nanmin(y_arr))
    max_val = max(np.nanmax(x_arr), np.nanmax(y_arr))

    return min_val, max_val
###############################################################################

###############################################################################
def style_axes_spines(ax: Axes,
                      linewidth: float = 2,
                      edgecolor: str = 'black') -> None:
    """
    Style the spines of a matplotlib Axes object by setting their linewidth and edge color.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes whose spines will be styled.

    linewidth : float, optional
        The width of the spines' lines. Must be positive. Default is 2.

    edgecolor : str, optional
        The color to apply to the spines. Default is 'black'.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If 'ax' is not a matplotlib.axes.Axes instance.
        If 'linewidth' is not a positive number.
        If 'edgecolor' is not a string.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> style_axes_spines(ax, linewidth=1.5, edgecolor='gray')
    """
    if not isinstance(ax, Axes):
        raise ValueError("Input 'ax' must be a matplotlib.axes.Axes instance.")
    if not isinstance(linewidth, (int, float)) or linewidth <= 0:
        raise ValueError("Input 'linewidth' must be a positive number.")
    if not isinstance(edgecolor, str):
        raise ValueError("Input 'edgecolor' must be a string.")

    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
        spine.set_edgecolor(edgecolor)
###############################################################################

###############################################################################
def plot_line(key: str,
              daily_mean: Union[pd.Series, list],
              ax: Axes,
              label_lookup: dict,
              color_palette: Iterator[str],
              line_width: float) -> None:
    """
    Plot a single line on a matplotlib Axes using seaborn, with label and color customization.

    Parameters
    ----------
    key : str
        Key used to look up a human-readable label in `label_lookup`.

    daily_mean : Union[pd.Series, list]
        Time series data to be plotted. If a list is provided, it is converted to a pandas Series.

    ax : matplotlib.axes.Axes
        Axes object on which the line will be drawn.

    label_lookup : dict
        Dictionary mapping keys to human-readable labels.

    color_palette : Iterator[str]
        Iterator yielding color values to assign to each plotted line.

    line_width : float
        Width of the plotted line. Must be positive.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If inputs are of incorrect type or line width is not positive.

    Examples
    --------
    >>> from itertools import cycle
    >>> fig, ax = plt.subplots()
    >>> color_iter = cycle(['blue', 'green'])
    >>> plot_line(
    ...     key='model',
    ...     daily_mean=[1, 2, 3],
    ...     ax=ax,
    ...     label_lookup={'model': 'Model Output'},
    ...     color_palette=color_iter,
    ...     line_width=2
    ... )
    """
    if not isinstance(key, str):
        raise ValueError("Input 'key' must be a string.")
    if not isinstance(ax, Axes):
        raise ValueError("Input 'ax' must be a matplotlib.axes.Axes instance.")
    if not isinstance(label_lookup, dict):
        raise ValueError("Input 'label_lookup' must be a dictionary.")
    if not hasattr(color_palette, '__iter__'):
        raise ValueError("Input 'color_palette' must be an iterator.")
    if not isinstance(line_width, (int, float)) or line_width <= 0:
        raise ValueError("Input 'line_width' must be a positive number.")

    if not isinstance(daily_mean, pd.Series):
        daily_mean = pd.Series(daily_mean)

    label = label_lookup.get(key, key)
    color = next(color_palette)

    sns.lineplot(data=daily_mean, label=label, ax=ax, lw=line_width, color=color)
###############################################################################

###############################################################################
def compute_geolocalized_coords(grid_shape: Tuple[int, int],
                                epsilon: float,
                                x_start: float,
                                x_step: float,
                                y_start: float,
                                y_step: float) -> Dict[str, Any]:
    """
    Compute geolocalized latitude and longitude 1D and 2D arrays, and bounding box extents.

    Parameters
    ----------
    grid_shape : tuple of int
        The shape of the grid as (Yrow, Xcol), i.e., number of latitude and longitude points.
    epsilon : float
        Margin to add/subtract to min/max lat/lon for plotting extent.
    x_start : float
        Starting longitude coordinate.
    x_step : float
        Longitude grid step size.
    y_start : float
        Starting latitude coordinate.
    y_step : float
        Latitude grid step size.

    Returns
    -------
    geo_coords : dict
        Dictionary with keys:
        - 'latp': 2D numpy array of latitude coordinates (Yrow x Xcol)
        - 'lonp': 2D numpy array of longitude coordinates (Yrow x Xcol)
        - 'lat_1d': 1D numpy array of latitude coordinates
        - 'lon_1d': 1D numpy array of longitude coordinates
        - 'MinLambda': float, minimum longitude extent (with epsilon margin)
        - 'MaxLambda': float, maximum longitude extent (with epsilon margin)
        - 'MinPhi': float, minimum latitude extent (with epsilon margin)
        - 'MaxPhi': float, maximum latitude extent (with epsilon margin)

    Raises
    ------
    ValueError
        If grid_shape is not a tuple of two positive integers.
        If epsilon or step sizes are non-positive.
    """

    # ----- INPUT VALIDATION -----
    if not (isinstance(grid_shape, tuple) and len(grid_shape) == 2):
        raise ValueError("grid_shape must be a tuple of two integers (Yrow, Xcol).")
    if not all(isinstance(dim, int) and dim > 0 for dim in grid_shape):
        raise ValueError("Both dimensions in grid_shape must be positive integers.")
    if not (isinstance(epsilon, (float, int)) and epsilon >= 0):
        raise ValueError("epsilon must be a non-negative float.")
    if not all(isinstance(step, (float, int)) and step > 0 for step in (x_step, y_step)):
        raise ValueError("x_step and y_step must be positive floats.")

    # ----- COORDINATE GENERATION-----
    Yrow, Xcol = grid_shape
    lat_1d = np.fromiter((y_start + j * y_step for j in range(Yrow)), dtype=float, count=Yrow)
    lon_1d = np.fromiter((x_start + i * x_step for i in range(Xcol)), dtype=float, count=Xcol)

    latp = np.tile(lat_1d[:, np.newaxis], (1, Xcol))
    lonp = np.tile(lon_1d[np.newaxis, :], (Yrow, 1))

    # ----- BUILDING THE DOMAIN -----
    MinPhi = np.nanmin(lat_1d) - epsilon
    MaxPhi = np.nanmax(lat_1d) + epsilon
    MinLambda = np.nanmin(lon_1d) - epsilon
    MaxLambda = np.nanmax(lon_1d) + epsilon

    # ----- SETTING UP THE DICTIONARY -----
    geo_coords = {
        'latp': latp,
        'lonp': lonp,
        'lat_1d': lat_1d,
        'lon_1d': lon_1d,
        'MinLambda': MinLambda,
        'MaxLambda': MaxLambda,
        'MinPhi': MinPhi,
        'MaxPhi': MaxPhi,
        'Epsilon': epsilon
    }

    return geo_coords
###############################################################################

###############################################################################
def swifs_colormap(data_in, variable_name):
    """
    Applies a custom logarithmic chlorophyll colormap to the input data.

    Parameters
    ----------
    data_in : 2D np.ndarray
        Input chlorophyll concentration data (in mg Chl/m^3).

    Returns
    -------
    norm_data : 2D np.ndarray
        Data normalized to the data scale for use with BoundaryNorm.
    cmap : ListedColormap
        Custom discrete colormap.
    norm : BoundaryNorm
        Boundary normalization for color levels.
    ticks : list
        Tick values for colorbar (bin edges).
    tick_labels : list
        Corresponding string labels for colorbar.
    """
    
    chla_vars = {"Chla"}
    n_vars = {"N1p", "N3n", "N4n"}
    p_vars = {"P1c", "P2c", "P3c", "P4c"}
    z_vars = {"Z3c", "Z4c", "Z5c", "Z6c"}
    r_vars = {"R6c"}
    
    # Logarithmic colorbar ticks (bin edges)
    if variable_name in chla_vars:
        Lticks = np.array([
            0.04, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50,
            0.80, 1.30, 2.00, 3.00, 4.00, 9.00, 12.0
            ])
    if variable_name in n_vars:
        if variable_name == 'N1p':
            Lticks = np.array([0.01, 0.015, 0.02, 0.03, 0.04,
                               0.05, 0.06, 0.08, 0.10, 0.12,
                               0.15, 0.18, 0.20])
        if variable_name == 'N3n':
            Lticks = np.array([0.01, 0.02, 0.05, 0.1, 0.2,
                               0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
        if variable_name == 'N4n':
            Lticks = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0,
                               5.0, 8.0, 10.0, 15.0, 20.0])
    if variable_name in p_vars:
        if variable_name == 'P1c':
            Lticks = np.array([1,  2,  3,  5, 10, 15, 20, 30,
                               50, 75, 100, 150, 200, 250, 300])
        if variable_name == 'P2c':
            Lticks = np.array([1,  2,  3,  5, 10, 15, 20, 30,
                               50, 75, 100, 150, 200])
        if variable_name == 'P3c':
            Lticks = np.array([1,  2,  3,  5, 10, 15, 20, 30])
        if variable_name == 'P4c':
            Lticks = np.array([1,  2,  3,  5, 10, 15, 20, 30,
                               50, 75, 100, 150, 200, 250, 300])
    if variable_name in z_vars:
        if variable_name == 'Z3c':
            Lticks = np.array([0.1, 0.15, 0.2, 0.3, 0.4,
                               0.5, 0.7, 1.0, 1.5, 2.0,
                               3.0, 4.0, 5.0])
        if variable_name == 'Z4c':
            Lticks = np.array([1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20])
        if variable_name == 'Z5c':
            Lticks = np.array([1, 1.5, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100])
        if variable_name == 'Z6c':
            Lticks = np.array([1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50])
    if variable_name in r_vars:
        if variable_name == 'R6c':
            Lticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500])
    nc = len(Lticks) - 1  # Number of color bins

    # Clip data to range defined by Lticks
    data_clipped = np.clip(data_in, Lticks[0], Lticks[-1])
    data_clipped[np.isnan(data_in)] = np.nan  # Preserve NaNs

    # Colormap with nc discrete colors

    # === Assign colormap by family ===
    if variable_name in chla_vars:
        cmap = get_cmap("viridis", nc)       # Blue to green hues
    elif variable_name in n_vars:
        cmap = get_cmap("YlGnBu", nc)           # Nutrients: Yellow-Green-Blue
    elif variable_name in p_vars:
        cmap = get_cmap(getattr(cmocean.cm, 'algae'), nc)           # Primary producers: Green
    elif variable_name in z_vars:
        cmap = get_cmap(getattr(cmocean.cm, 'turbid'), nc)          # Secondary producers: Orange
    elif variable_name in r_vars:
        cmap = get_cmap(getattr(cmocean.cm, 'matter'), nc)             # Organic matter: Purple-Red
    else:
        cmap = get_cmap("viridis", nc)          # Default: Viridis

    # Use Lticks as boundaries directly (data values)
    bounds = Lticks

    # BoundaryNorm maps data to colormap indices using bounds
    norm = BoundaryNorm(boundaries=bounds, ncolors=nc, clip=False)

    # Ticks are exactly the boundaries (for colorbar)
    ticks = Lticks
    tick_labels = [f"{val}" for val in Lticks]

    # No normalization of data needed here — contourf will use norm and cmap

    return data_clipped, cmap, norm, ticks, tick_labels
###############################################################################

###############################################################################
def get_benthic_plot_parameters(bfm2plot: str, var_dataframe: dict, opts: dict):   
    """
    Return vmin, vmax, levels, num_ticks, colormap, and a flag for custom cmap based on variable type and family.

    Parameters
    ----------
    bfm2plot : str
        Variable to plot (e.g., "votemper", "vosaline", "density", "dense_water", or others).
    var_dataframe : dict
        Nested dict of {year: [monthly 2D arrays]}.
    opts : dict
        Dictionary of plotting options containing vmin, vmax, levels for known variables.

    Returns
    -------
    vmin : float
    vmax : float
    levels : np.ndarray
    num_ticks : int or None
    cmap : matplotlib colormap or str
    use_custom_cmap : bool
    hypoxia_threshold : float or None
        Hypoxia threshold value in mmol/m3 for O2o, otherwise None.
    """

    # Variable families
    chla_vars = {"Chla"}
    n_vars = {"N1p", "N3n", "N4n"}
    p_vars = {"P1c", "P2c", "P3c", "P4c"}
    z_vars = {"Z3c", "Z4c", "Z5c", "Z6c"}
    r_vars = {"R6c"}
    
    use_custom_cmap = False
    hypoxia_threshold = None  # default None, only set for O2o

    if bfm2plot == 'O2o':
        vmin, vmax = 0, 350
        levels = np.linspace(vmin, vmax, 29)
        num_ticks = None  # or your preferred tick count
        hypoxia_threshold = 62.5  # mmol/m3 hypoxia threshold
        hyperoxia_threshold = 312.5  # mmol/m3 hyperoxia threshold
    
        # Create the custom colormap with adapted vmax and thresholds
        cmap = custom_oxy(vmin=0, vmax=350, low=62.5, high=312.5)
    
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    elif bfm2plot == "votemper":
       vmin, vmax = opts["vmin_votemper"], opts["vmax_votemper"]
       levels = np.linspace(vmin, vmax, opts["levels_votemper"])
       num_ticks = opts.get("num_ticks_votemper", 5)
       cmap = getattr(cmocean.cm, 'thermal')
       return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold

    elif bfm2plot == "vosaline":
       vmin, vmax = opts["vmin_vosaline"], opts["vmax_vosaline"]
       levels = np.linspace(vmin, vmax, opts["levels_vosaline"])
       num_ticks = opts.get("num_ticks_vosaline", 5)
       cmap = getattr(cmocean.cm, 'haline')
       return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold

    elif bfm2plot in ("density", "dense_water"):
       vmin, vmax = opts["vmin_density"], opts["vmax_density"]
       levels = np.linspace(vmin, vmax, opts["levels_density"])
       num_ticks = opts.get("num_ticks_density", 5)
       cmap = getattr(cmocean.cm, 'dense')
       return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold

    elif any(bfm2plot in family for family in (chla_vars, n_vars, p_vars, z_vars, r_vars)):
       return None, None, None, None, None, True, hypoxia_threshold

    else:
       all_data = np.concatenate([
           data2D.ravel()
           for monthly_data in var_dataframe.values()
           for data2D in monthly_data if data2D is not None
       ])
       all_data = all_data[~np.isnan(all_data)]
       if all_data.size == 0:
           raise ValueError("No valid data found to determine vmin and vmax.")
       vmin, vmax = float(all_data.min()), float(all_data.max())
       levels = np.linspace(vmin, vmax, 20)
       num_ticks = 5
       cmap = 'jet'
       
       return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold
###############################################################################

############################################################################### 
def cmocean_to_plotly(cmap_name, n=256):
    """
    Convert a cmocean colormap to a Plotly colorscale.
    """
    cmap = getattr(cmocean.cm, cmap_name)  # get colormap directly by attribute
    colorscale = []
    for i in range(n):
        color = cmap(i / (n - 1))  # input normalized [0,1]
        r, g, b, _ = [int(255*c) for c in color]
        colorscale.append([i/(n-1), f'rgb({r},{g},{b})'])
    return colorscale
###############################################################################

############################################################################### 
def invert_colorscale(colorscale):
    """
    Invert a Plotly colorscale list.

    Parameters
    ----------
    colorscale : list of [float, color] pairs
        Original colorscale.

    Returns
    -------
    list
        Inverted colorscale.
    """
    # colorscale is a list like [[0.0, 'rgb(...)'], [0.1, 'rgb(...)'], ..., [1.0, 'rgb(...)']]
    return [[1 - t, color] for t, color in reversed(colorscale)]
############################################################################### 

############################################################################### 
def custom_oxy(vmin=0, vmax=350, low=62.5, high=312.5, name='oxy_custom'):
    """
    Creates a custom segmented colormap with three gradients:
    - Red gradient from vmin to low
    - Gray gradient from low to high
    - Yellow gradient from high to vmax

    Parameters
    ----------
    vmin : float
        Minimum value of the data range.
    vmax : float
        Maximum value of the data range.
    low : float
        Value at which red gradient ends and gray gradient starts.
    high : float
        Value at which gray gradient ends and yellow gradient starts.
    name : str
        Name of the colormap.

    Returns
    -------
    cmap : LinearSegmentedColormap
        The resulting custom colormap.
    norm : Normalize
        Normalization instance for mapping data to 0-1 for the colormap.
    """
    # Normalize threshold positions from data range to 0-1
    x_low = (low - vmin) / (vmax - vmin)
    x_high = (high - vmin) / (vmax - vmin)
    
    if not (0 <= x_low < x_high <= 1):
        raise ValueError(f"Thresholds normalized must be in ascending order between 0 and 1, got low={x_low}, high={x_high}")

    # Colors in RGB normalized [0,1]
    red_start = (0x41/255, 0x05/255, 0x05/255)   # #410505
    red_end = (0x8B/255, 0x12/255, 0x07/255)     # #8B1207

    gray_start = (0x50/255, 0x50/255, 0x50/255)  # #505050
    gray_end = (0xF1/255, 0xF1/255, 0xEF/255)    # #F1F1EF

    yellow_start = (0xF2/255, 0xF9/255, 0x5D/255) # #F2F95D
    yellow_end = (0xDE/255, 0xB2/255, 0x1B/255)   # #DEB21B

    cdict = {
        'red': [
            (0.0, red_start[0], red_start[0]),
            (x_low, red_end[0], red_end[0]),
            (x_low, gray_start[0], gray_start[0]),
            (x_high, gray_end[0], gray_end[0]),
            (x_high, yellow_start[0], yellow_start[0]),
            (1.0, yellow_end[0], yellow_end[0])
        ],
        'green': [
            (0.0, red_start[1], red_start[1]),
            (x_low, red_end[1], red_end[1]),
            (x_low, gray_start[1], gray_start[1]),
            (x_high, gray_end[1], gray_end[1]),
            (x_high, yellow_start[1], yellow_start[1]),
            (1.0, yellow_end[1], yellow_end[1])
        ],
        'blue': [
            (0.0, red_start[2], red_start[2]),
            (x_low, red_end[2], red_end[2]),
            (x_low, gray_start[2], gray_start[2]),
            (x_high, gray_end[2], gray_end[2]),
            (x_high, yellow_start[2], yellow_start[2]),
            (1.0, yellow_end[2], yellow_end[2])
        ],
    }

    cmap = mcolors.LinearSegmentedColormap('CustomRedGrayYellow', segmentdata=cdict, N=256)
    return cmap