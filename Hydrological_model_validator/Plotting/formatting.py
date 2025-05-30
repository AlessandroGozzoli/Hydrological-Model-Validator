import numpy as np
import re
from typing import Tuple, Union, Iterator, Dict, Any, Optional, List
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
from matplotlib.cm import get_cmap
import cmocean
from matplotlib.patches import Polygon

###############################################################################
def format_unit(unit: str) -> str:
    """
    Formats a chemical or physical unit string into LaTeX math mode with subscripts and exponents.

    Parameters
    ----------
    unit : str
        The unit string to format. Examples: "O2/m3", "mg/L", "PO4", "kg/m2".

    Returns
    -------
    str
        A LaTeX-formatted string wrapped in math mode.

    Raises
    ------
    ValueError
        If the input is not a non-empty string.

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
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Input 'unit' must be a non-empty string.")

    def add_subscripts(text: str) -> str:
        return re.sub(r'(?<=[A-Za-z])(\d+)', r'_{\1}', text)

    def add_exponents(text: str) -> str:
        return re.sub(r'([a-zA-Z])(\d+)', r'\1^{\2}', text)

    unit = unit.strip()

    if '/' in unit:
        numerator, denominator = map(str.strip, unit.split('/', 1))
        return f'$\\frac{{{add_subscripts(numerator)}}}{{{add_exponents(denominator)}}}$'

    formatted = add_subscripts(unit)
    formatted = add_exponents(formatted)
    return f'${formatted}$'
###############################################################################

###############################################################################
def get_variable_label_unit(variable_name: str) -> Tuple[str, str]:
    if not isinstance(variable_name, str) or not variable_name.strip():
        raise ValueError("Input 'variable_name' must be a non-empty string.")

    mapping = {
        'SST': ('Sea Surface Temperature', '°C'),
        'CHL_L3': ('Chlorophyll (Level 3)', 'mg/m3'),
        'CHL_L4': ('Chlorophyll (Level 4)', 'mg/m3'),
    }

    label, raw_unit = mapping.get(variable_name, (variable_name, ''))
    return label, f"[{format_unit(raw_unit)}]" if raw_unit else ''
###############################################################################

###############################################################################
def fill_annular_region(
    ax: Axes,
    r_in: float,
    r_out: float,
    color: str,
    alpha: float = 0.3,
    zorder: int = 0
) -> None:
    """
    Fill an annular (ring-shaped) region between two radii on a matplotlib polar Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object to draw the annular region on. Must be polar.

    r_in : float
        Inner radius of the annulus. Must be non-negative and less than or equal to `r_out`.

    r_out : float
        Outer radius of the annulus. Must be greater than or equal to `r_in`.

    color : str
        Fill color (e.g., 'blue', '#1f77b4').

    alpha : float, optional
        Transparency level of the fill. Must be between 0 and 1. Default is 0.3.

    zorder : int, optional
        Drawing order for the patch. Default is 0.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any inputs are invalid.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    >>> fill_annular_region(ax, r_in=1.0, r_out=2.0, color='red', alpha=0.2)
    >>> plt.show()
    """
    # --- VALIDATION ---
    if not isinstance(ax, Axes) or ax.name != "polar":
        raise ValueError("Input 'ax' must be a matplotlib polar Axes instance.")
    if not isinstance(r_in, (int, float)) or r_in < 0:
        raise ValueError("Input 'r_in' must be a non-negative number.")
    if not isinstance(r_out, (int, float)) or r_out < r_in:
        raise ValueError("Input 'r_out' must be greater than or equal to 'r_in'.")
    if not isinstance(color, str):
        raise ValueError("Input 'color' must be a string.")
    if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
        raise ValueError("Input 'alpha' must be between 0 and 1.")

    # --- COMPUTE ANNULAR REGION ---
    theta = np.linspace(0, 2 * np.pi, 500)
    r_outer = np.full_like(theta, r_out)
    r_inner = np.full_like(theta, r_in)

    # Polar to Cartesian (Outer arc + reversed inner arc)
    x = np.concatenate([r_outer * np.cos(theta), r_inner[::-1] * np.cos(theta[::-1])])
    y = np.concatenate([r_outer * np.sin(theta), r_inner[::-1] * np.sin(theta[::-1])])
    coords = np.column_stack((x, y))

    # Draw polygon patch
    polygon = Polygon(coords, closed=True, color=color, alpha=alpha, zorder=zorder,
                      transform=ax.transData)
    ax.add_patch(polygon)
###############################################################################

###############################################################################    
def get_min_max_for_identity_line(
    x: Union[np.ndarray, list, tuple],
    y: Union[np.ndarray, list, tuple]
) -> Tuple[float, float]:
    """
    Compute the combined minimum and maximum from two numeric sequences,
    suitable for setting axis limits for an identity line (y = x).

    Parameters
    ----------
    x : array-like
        First numeric sequence (list, tuple, or NumPy array). NaNs and infinite values are ignored.

    y : array-like
        Second numeric sequence (list, tuple, or NumPy array). NaNs and infinite values are ignored.

    Returns
    -------
    tuple of float
        (min_value, max_value) — The global minimum and maximum of `x` and `y`,
        excluding NaNs and infinite values.

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

    x_finite = x_arr[np.isfinite(x_arr)]
    y_finite = y_arr[np.isfinite(y_arr)]

    combined = np.concatenate((x_finite, y_finite))
    if combined.size == 0:
        raise ValueError("Both inputs contain no finite values.")

    return float(combined.min()), float(combined.max())
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
def compute_geolocalized_coords(
    grid_shape: Tuple[int, int],
    epsilon: float,
    x_start: float,
    x_step: float,
    y_start: float,
    y_step: float
) -> Dict[str, Any]:
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
        - 'Epsilon': float, the epsilon margin used for extents

    Raises
    ------
    ValueError
        If grid_shape is not a tuple of two positive integers.
        If epsilon or step sizes are invalid.

    Examples
    --------
    >>> compute_geolocalized_coords((3, 4), 0.1, -10, 0.5, 50, 0.25)
    {
        'latp': array([[50.  , 50.  , 50.  , 50.  ],
                       [50.25, 50.25, 50.25, 50.25],
                       [50.5 , 50.5 , 50.5 , 50.5 ]]),
        'lonp': array([[-10.  , -9.5 , -9.  , -8.5 ],
                       [-10.  , -9.5 , -9.  , -8.5 ],
                       [-10.  , -9.5 , -9.  , -8.5 ]]),
        'lat_1d': array([50.  , 50.25, 50.5 ]),
        'lon_1d': array([-10. ,  -9.5,  -9. ,  -8.5]),
        'MinLambda': -10.1,
        'MaxLambda': -8.4,
        'MinPhi': 49.9,
        'MaxPhi': 50.6,
        'Epsilon': 0.1
    }
    """
    # Validate inputs
    if not (isinstance(grid_shape, tuple) and len(grid_shape) == 2):
        raise ValueError("grid_shape must be a tuple of two integers (Yrow, Xcol).")
    if not all(isinstance(dim, int) and dim > 0 for dim in grid_shape):
        raise ValueError("Both dimensions in grid_shape must be positive integers.")
    if not (isinstance(epsilon, (float, int)) and epsilon >= 0):
        raise ValueError("epsilon must be a non-negative float.")
    if not all(isinstance(step, (float, int)) and step > 0 for step in (x_step, y_step)):
        raise ValueError("x_step and y_step must be positive floats.")

    epsilon = float(epsilon)  # normalize

    Yrow, Xcol = grid_shape

    lat_1d = y_start + y_step * np.arange(Yrow, dtype=float)
    lon_1d = x_start + x_step * np.arange(Xcol, dtype=float)

    lonp, latp = np.meshgrid(lon_1d, lat_1d)

    return {
        'latp': latp,
        'lonp': lonp,
        'lat_1d': lat_1d,
        'lon_1d': lon_1d,
        'MinLambda': float(lon_1d.min() - epsilon),
        'MaxLambda': float(lon_1d.max() + epsilon),
        'MinPhi': float(lat_1d.min() - epsilon),
        'MaxPhi': float(lat_1d.max() + epsilon),
        'Epsilon': epsilon
    }
###############################################################################

###############################################################################
def swifs_colormap(
    data_in: np.ndarray,
    variable_name: str
) -> Tuple[np.ndarray, ListedColormap, BoundaryNorm, np.ndarray, List[str]]:
    """
    Apply a variable-specific discrete colormap with custom bin edges for oceanographic data.

    Parameters
    ----------
    data_in : np.ndarray
        Input concentration data (e.g., chlorophyll in mg Chl/m^3).
    variable_name : str
        Variable name to select appropriate colormap and bins.

    Returns
    -------
    data_clipped : np.ndarray
        Data clipped to the colormap bin range, NaNs preserved.
    cmap : ListedColormap
        Discrete ListedColormap instance corresponding to the variable.
    norm : BoundaryNorm
        Normalization object for color boundaries.
    ticks : np.ndarray
        Array of bin edge values for colorbar ticks.
    tick_labels : List[str]
        String labels for the colorbar ticks.

    Raises
    ------
    ValueError
        If variable_name is unknown or data_in contains only NaNs.

    Examples
    --------
    >>> data = np.array([[0.1, 0.5], [2.0, 5.0]])
    >>> norm_data, cmap, norm, ticks, labels = swifs_colormap(data, "Chla")
    """

    # Define variable groups
    chla_vars = {"Chla"}
    n_vars = {"N1p", "N3n", "N4n"}
    p_vars = {"P1c", "P2c", "P3c", "P4c"}
    z_vars = {"Z3c", "Z4c", "Z5c", "Z6c"}
    r_vars = {"R6c"}

    # Assign bin edges and colormap per variable
    if variable_name in chla_vars:
        Lticks = np.array([0.04, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50,
                           0.80, 1.30, 2.00, 3.00, 4.00, 9.00, 12.0])
        base_cmap = get_cmap("viridis")

    elif variable_name in n_vars:
        if variable_name == 'N1p':
            Lticks = np.array([0.01, 0.015, 0.02, 0.03, 0.04,
                               0.05, 0.06, 0.08, 0.10, 0.12,
                               0.15, 0.18, 0.20])
        elif variable_name == 'N3n':
            Lticks = np.array([0.01, 0.02, 0.05, 0.1, 0.2,
                               0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
        elif variable_name == 'N4n':
            Lticks = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0,
                               5.0, 8.0, 10.0, 15.0, 20.0])
        else:
            raise ValueError(f"Unknown nitrogen variable '{variable_name}'. Allowed: {n_vars}")
        base_cmap = get_cmap("YlGnBu")

    elif variable_name in p_vars:
        if variable_name == 'P1c':
            Lticks = np.array([1, 2, 3, 5, 10, 15, 20, 30,
                               50, 75, 100, 150, 200, 250, 300])
        elif variable_name == 'P2c':
            Lticks = np.array([1, 2, 3, 5, 10, 15, 20, 30,
                               50, 75, 100, 150, 200])
        elif variable_name == 'P3c':
            Lticks = np.array([1, 2, 3, 5, 10, 15, 20, 30])
        elif variable_name == 'P4c':
            Lticks = np.array([1, 2, 3, 5, 10, 15, 20, 30,
                               50, 75, 100, 150, 200, 250, 300])
        else:
            raise ValueError(f"Unknown primary producer variable '{variable_name}'. Allowed: {p_vars}")
        base_cmap = cmocean.cm.algae

    elif variable_name in z_vars:
        if variable_name == 'Z3c':
            Lticks = np.array([0.1, 0.15, 0.2, 0.3, 0.4,
                               0.5, 0.7, 1.0, 1.5, 2.0,
                               3.0, 4.0, 5.0])
        elif variable_name == 'Z4c':
            Lticks = np.array([1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20])
        elif variable_name == 'Z5c':
            Lticks = np.array([1, 1.5, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100])
        elif variable_name == 'Z6c':
            Lticks = np.array([1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50])
        else:
            raise ValueError(f"Unknown secondary producer variable '{variable_name}'. Allowed: {z_vars}")
        base_cmap = cmocean.cm.turbid

    elif variable_name in r_vars:
        if variable_name == 'R6c':
            Lticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500])
        else:
            raise ValueError(f"Unknown organic matter variable '{variable_name}'. Allowed: {r_vars}")
        base_cmap = cmocean.cm.matter

    else:
        # Default linear bins between data min and max
        min_val = np.nanmin(data_in)
        max_val = np.nanmax(data_in)
        if np.isnan(min_val) or np.isnan(max_val):
            raise ValueError("data_in contains only NaNs, cannot determine default bins.")
        Lticks = np.linspace(min_val, max_val, 11)
        base_cmap = get_cmap("viridis")

    n_colors = len(Lticks) - 1

    # Convert continuous colormap to discrete ListedColormap with n_colors
    # Safe fallback if base_cmap.colors attribute doesn't exist
    try:
        colors = base_cmap(np.linspace(0, 1, n_colors))
    except Exception:
        colors = base_cmap.colors if hasattr(base_cmap, "colors") else base_cmap(np.linspace(0, 1, n_colors))

    cmap = ListedColormap(colors)

    # Clip data to bins (preserves NaNs)
    data_clipped = np.clip(data_in, Lticks[0], Lticks[-1])

    norm = BoundaryNorm(boundaries=Lticks, ncolors=n_colors, clip=False)

    tick_labels = [f"{val:g}" for val in Lticks]

    return data_clipped, cmap, norm, Lticks, tick_labels
###############################################################################

###############################################################################
def get_benthic_plot_parameters(
    bfm2plot: str,
    var_dataframe: Dict[int, List[np.ndarray]],
    opts: Dict[str, Union[int, float]]
) -> Tuple[
    Optional[float], Optional[float], Optional[np.ndarray], Optional[int],
    Optional[Union[str, LinearSegmentedColormap]], bool,
    Optional[float], Optional[float]
]:
    """
    Return plotting parameters for benthic variables, including color limits,
    contour levels, colormap, and hypoxia/hyperoxia thresholds.

    Parameters
    ----------
    bfm2plot : str
        Variable name for plotting.
    var_dataframe : dict[int, list[np.ndarray]]
        Dictionary mapping keys (years) to lists of monthly 2D data arrays.
    opts : dict[str, int|float]
        Options dictionary for min/max values, levels, and ticks.

    Returns
    -------
    vmin : float or None
        Minimum color scale value.
    vmax : float or None
        Maximum color scale value.
    levels : np.ndarray or None
        Contour levels for plotting.
    num_ticks : int or None
        Number of colorbar ticks.
    cmap : str, LinearSegmentedColormap or None
        Colormap or its name.
    use_custom_cmap : bool
        True if a special/custom colormap should be used.
    hypoxia_threshold : float or None
        Threshold below which hypoxia occurs.
    hyperoxia_threshold : float or None
        Threshold above which hyperoxia occurs.

    Raises
    ------
    ValueError
        If required options are missing or no valid data found.
    """

    chla_vars = {"Chla"}
    n_vars = {"N1p", "N3n", "N4n"}
    p_vars = {"P1c", "P2c", "P3c", "P4c"}
    z_vars = {"Z3c", "Z4c", "Z5c", "Z6c"}
    r_vars = {"R6c"}

    use_custom_cmap = False
    hypoxia_threshold = None
    hyperoxia_threshold = None

    if bfm2plot == 'O2o':
        vmin, vmax = 0.0, 350.0
        levels = np.linspace(vmin, vmax, 29, endpoint=True)
        num_ticks = None
        hypoxia_threshold = 62.5
        hyperoxia_threshold = 312.5
        try:
            # Assume custom_oxy is defined elsewhere
            cmap = custom_oxy(vmin=vmin, vmax=vmax,
                              low=hypoxia_threshold, high=hyperoxia_threshold)
        except NameError:
            cmap = get_cmap('coolwarm')
        use_custom_cmap = True
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    elif bfm2plot == "votemper":
        vmin = opts.get("vmin_votemper")
        vmax = opts.get("vmax_votemper")
        if vmin is None or vmax is None:
            raise ValueError("vmin_votemper and vmax_votemper must be specified in opts")
        levels = np.linspace(vmin, vmax, opts.get("levels_votemper", 20), endpoint=True)
        num_ticks = opts.get("num_ticks_votemper", 5)
        cmap = cmocean.cm.thermal
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    elif bfm2plot == "vosaline":
        vmin = opts.get("vmin_vosaline")
        vmax = opts.get("vmax_vosaline")
        if vmin is None or vmax is None:
            raise ValueError("vmin_vosaline and vmax_vosaline must be specified in opts")
        levels = np.linspace(vmin, vmax, opts.get("levels_vosaline", 20), endpoint=True)
        num_ticks = opts.get("num_ticks_vosaline", 5)
        cmap = cmocean.cm.haline
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    elif bfm2plot in ("density", "dense_water"):
        vmin = opts.get("vmin_density")
        vmax = opts.get("vmax_density")
        if vmin is None or vmax is None:
            raise ValueError("vmin_density and vmax_density must be specified in opts")
        levels = np.linspace(vmin, vmax, opts.get("levels_density", 20), endpoint=True)
        num_ticks = opts.get("num_ticks_density", 5)
        cmap = cmocean.cm.dense
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    elif any(bfm2plot in family for family in (chla_vars, n_vars, p_vars, z_vars, r_vars)):
        # For these variables, custom colormap handled elsewhere
        return None, None, None, None, None, True, hypoxia_threshold, hyperoxia_threshold

    else:
        # Default: flatten data, remove NaNs, infer vmin/vmax
        data_arrays = [
            arr for monthly_list in var_dataframe.values()
            for arr in monthly_list if arr is not None
        ]
        if not data_arrays:
            raise ValueError("No data arrays found in var_dataframe.")
        all_data = np.concatenate([arr.ravel() for arr in data_arrays])
        all_data = all_data[~np.isnan(all_data)]
        if all_data.size == 0:
            raise ValueError("No valid data found to determine vmin and vmax.")

        vmin = float(all_data.min())
        vmax = float(all_data.max())
        levels = np.linspace(vmin, vmax, 20, endpoint=True)
        num_ticks = 5
        cmap = 'jet'  # default fallback
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold
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