import numpy as np
import re
from typing import Tuple, Union, Iterator, Dict, Any, Optional, List
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
import cmocean
import collections.abc
import matplotlib.pyplot as plt

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
    # ===== INPUT VALIDATION =====
    # Validate that input is a non-empty string to avoid processing invalid types
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("❌ Input 'unit' must be a non-empty string. ❌")

    # ===== HELPER FUNCTIONS =====
    # Add LaTeX subscripts by converting digits following letters into _{digit}
    def add_subscripts(text: str) -> str:
        # Lookbehind for letters, then digits, wrap digits in _{ }
        return re.sub(r'(?<=[A-Za-z])(\d+)', r'_{\1}', text)

    # Add LaTeX exponents by converting digits following letters into ^{digit}
    def add_exponents(text: str) -> str:
        # Match letters followed by digits, wrap digits in ^{ }
        return re.sub(r'([a-zA-Z])(\d+)', r'\1^{\2}', text)

    # Strip whitespace from input for consistent processing
    unit = unit.strip()

    # ===== FRACTION HANDLING =====
    # Check if unit contains a division, to format as a LaTeX fraction
    if '/' in unit:
        # Split numerator and denominator around first '/'
        numerator, denominator = map(str.strip, unit.split('/', 1))

        # Format numerator with subscripts (e.g., O2 -> O_2)
        # Format denominator with exponents (e.g., m3 -> m^{3})
        return f'$\\frac{{{add_subscripts(numerator)}}}{{{add_exponents(denominator)}}}$'

    # ===== SIMPLE UNIT FORMATTING =====
    # For units without '/', just add subscripts and exponents as needed
    formatted = add_subscripts(unit)
    formatted = add_exponents(formatted)

    # Wrap result in LaTeX math mode delimiters
    return f'${formatted}$'
###############################################################################

###############################################################################
def get_variable_label_unit(variable_name: str) -> tuple[str, str]:
    """
    Returns a descriptive label and formatted unit string for a given variable name.

    Parameters
    ----------
    variable_name : str
        The name of the variable. Examples: "SST", "CHL_L3", "CHL_L4".

    Returns
    -------
    tuple of (str, str)
        A tuple containing:
        - The descriptive label for the variable.
        - The unit string formatted in LaTeX math mode wrapped in square brackets,
          or an empty string if the unit is unknown or not provided.

    Raises
    ------
    ValueError
        If the input `variable_name` is not a non-empty string.

    Examples
    --------
    >>> get_variable_label_unit("SST")
    ('Sea Surface Temperature', '[°C]')
    >>> get_variable_label_unit("CHL_L3")
    ('Chlorophyll (Level 3)', '[$\\frac{mg}{m^{3}}$]')
    >>> get_variable_label_unit("UNKNOWN")
    ('UNKNOWN', '')
    """
    # ===== INPUT VALIDATION =====
    # Ensure the input is a non-empty string to prevent incorrect lookups or errors
    if not isinstance(variable_name, str) or not variable_name.strip():
        # ❌ Add red x emoji for input validation error messages ❌
        raise ValueError("❌ Input 'variable_name' must be a non-empty string. ❌")

    # ===== MAPPING DICTIONARY =====
    # Define a mapping from known variable names to their descriptive labels and raw units
    mapping = {
        'SST': ('Sea Surface Temperature', '°C'),
        'CHL_L3': ('Chlorophyll (Level 3)', 'mg/m3'),
        'CHL_L4': ('Chlorophyll (Level 4)', 'mg/m3'),
    }

    # ===== LOOKUP AND FORMAT =====
    # Get the label and raw unit from the mapping, or default to variable name and empty unit
    label, raw_unit = mapping.get(variable_name, (variable_name, ''))

    # If unit is known, format it using format_unit and wrap with square brackets
    # Otherwise, return an empty string for unit
    return label, f"[{format_unit(raw_unit)}]" if raw_unit else ''
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
    # ===== INPUT VALIDATION =====
    # Validate that ax is an instance of matplotlib Axes for correct drawing context
    if not isinstance(ax, Axes):
        raise ValueError("❌ Input 'ax' must be a matplotlib.axes.Axes instance. ❌")

    # Validate r_in is non-negative numeric value to define valid inner radius
    if not isinstance(r_in, (int, float)) or r_in < 0:
        raise ValueError("❌ Input 'r_in' must be a non-negative number. ❌")

    # Validate r_out is numeric and greater than or equal to r_in to ensure valid annulus
    if not isinstance(r_out, (int, float)) or r_out < r_in:
        raise ValueError("❌ Input 'r_out' must be a number greater than or equal to 'r_in'. ❌")

    # Validate alpha is a float between 0 and 1 for proper transparency level
    if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
        raise ValueError("❌ Input 'alpha' must be a number between 0 and 1. ❌")

    # Validate color is a string to ensure matplotlib can interpret the color correctly
    if not isinstance(color, str):
        raise ValueError("❌ Input 'color' must be a string. ❌")

    # ===== COMPUTE CIRCLE COORDINATES =====
    # Create an array of angles from 0 to 2*pi for full circle representation
    theta = np.linspace(0, 2 * np.pi, 500)

    # Calculate x, y coordinates for outer radius circle (counterclockwise)
    x_outer, y_outer = r_out * np.cos(theta), r_out * np.sin(theta)

    # Calculate x, y coordinates for inner radius circle (clockwise to close polygon correctly)
    x_inner, y_inner = r_in * np.cos(theta[::-1]), r_in * np.sin(theta[::-1])

    # ===== CONCATENATE COORDINATES =====
    # Combine outer circle points with inner circle points to form closed annulus path
    x = np.concatenate((x_outer, x_inner))
    y = np.concatenate((y_outer, y_inner))

    # ===== FILL ANNUALR REGION =====
    # Use matplotlib's fill to color the annulus with specified transparency and layering
    ax.fill(x, y, color=color, alpha=alpha, zorder=0)
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
    # ===== CONVERT INPUTS TO ARRAYS =====
    # Convert inputs to NumPy arrays to leverage vectorized operations and filtering
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # ===== FILTER FINITE VALUES =====
    # Remove NaNs and infinite values from both arrays to ensure meaningful min/max
    x_finite = x_arr[np.isfinite(x_arr)]
    y_finite = y_arr[np.isfinite(y_arr)]

    # ===== COMBINE AND VALIDATE =====
    # Concatenate filtered arrays to get combined range from both inputs
    combined = np.concatenate((x_finite, y_finite))

    # Raise error if no finite values remain to avoid invalid axis limits
    if combined.size == 0:
        raise ValueError("❌ Both inputs contain no finite values. ❌")

    # ===== COMPUTE MIN AND MAX =====
    # Return the global minimum and maximum as floats for use in plot axis limits
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
    # ===== INPUT VALIDATION =====
    # Ensure ax is a matplotlib Axes instance for spine styling
    if not isinstance(ax, Axes):
        raise ValueError("❌ Input 'ax' must be a matplotlib.axes.Axes instance. ❌")

    # Ensure linewidth is a positive number for visible spine lines
    if not isinstance(linewidth, (int, float)) or linewidth <= 0:
        raise ValueError("❌ Input 'linewidth' must be a positive number. ❌")

    # Ensure edgecolor is a string representing a valid color
    if not isinstance(edgecolor, str):
        raise ValueError("❌ Input 'edgecolor' must be a string. ❌")

    # ===== STYLE SPINES =====
    # Loop through all spines of the axes and apply linewidth and edgecolor
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
              line_width: float,
              *,
              library: str) -> None:
    """
    Plot a single line on a matplotlib Axes using seaborn, with label and color customization.

    Parameters
    ----------
    key : str
        Key used to look up a human-readable label in `label_lookup`.
    
    library : str
        Used to decide with which plotting library (sns or plt) to plot

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
    # ===== INPUT VALIDATION =====
    # Check that key is string to access label lookup correctly
    if not isinstance(key, str):
        raise ValueError("❌ Input 'key' must be a string. ❌")

    # Ensure ax is a matplotlib Axes instance to plot on
    if not isinstance(ax, Axes):
        raise ValueError("❌ Input 'ax' must be a matplotlib.axes.Axes instance. ❌")

    # Verify label_lookup is a dictionary for label retrieval
    if not isinstance(label_lookup, dict):
        raise ValueError("❌ Input 'label_lookup' must be a dictionary. ❌")

    # Confirm color_palette is an iterator (like itertools.cycle) for sequential colors
    if not isinstance(color_palette, collections.abc.Iterator):
        raise ValueError("❌ Input 'color_palette' must be an iterator (e.g., from itertools.cycle). ❌")

    # Check line_width is positive for visible plotting
    if not isinstance(line_width, (int, float)) or line_width <= 0:
        raise ValueError("❌ Input 'line_width' must be a positive number. ❌")

    # ===== PREPARE DATA =====
    # Convert daily_mean to pandas Series if it is a list for seaborn compatibility
    if not isinstance(daily_mean, pd.Series):
        daily_mean = pd.Series(daily_mean)

    # ===== PLOT LINE =====
    # Get the label for the line or default to the key if not found
    label = label_lookup.get(key, key)

    # Get the next color from the color palette iterator for consistent coloring
    color = next(color_palette)

    if library == 'sns':
        # Use seaborn lineplot to plot daily_mean on ax with specified line width, color, and label
        sns.lineplot(data=daily_mean, label=label, ax=ax, lw=line_width, color=color)
    elif library == 'plt':
        # matplotlib alternative
        ax.plot(daily_mean.index, daily_mean.values, label=label, linewidth=line_width, color=color)
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
    # ===== INPUT VALIDATION =====
    # Check grid_shape is tuple of length 2 representing (rows, columns)
    if not (isinstance(grid_shape, tuple) and len(grid_shape) == 2):
        raise ValueError("❌ grid_shape must be a tuple of two integers (Yrow, Xcol). ❌")
    
    # Ensure each dimension is a positive integer
    if not all(isinstance(dim, int) and dim > 0 for dim in grid_shape):
        raise ValueError("❌ Both dimensions in grid_shape must be positive integers. ❌")
    
    # epsilon must be a non-negative float or int (margin for plotting)
    if not (isinstance(epsilon, (float, int)) and epsilon >= 0):
        raise ValueError("❌ epsilon must be a non-negative float. ❌")
    
    # Step sizes for longitude and latitude must be positive floats or ints
    if not all(isinstance(step, (float, int)) and step > 0 for step in (x_step, y_step)):
        raise ValueError("❌ x_step and y_step must be positive floats. ❌")
    
    # Normalize epsilon to float
    epsilon = float(epsilon)

    # ===== COMPUTE 1D COORDINATES =====
    # Create 1D arrays for latitude and longitude by stepping from start values
    Yrow, Xcol = grid_shape
    lat_1d = y_start + y_step * np.arange(Yrow, dtype=float)
    lon_1d = x_start + x_step * np.arange(Xcol, dtype=float)

    # ===== CREATE 2D GRID COORDINATES =====
    # Meshgrid combines 1D lat and lon arrays into 2D grids matching grid shape
    lonp, latp = np.meshgrid(lon_1d, lat_1d)

    # ===== COMPUTE EXTENTS WITH EPSILON MARGIN =====
    # These extents can be used for setting plot limits with a margin
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
    # ===== VARIABLE GROUPS =====
    # Group variables by type to assign specific colormaps and bin edges
    chla_vars = {"Chla"}
    n_vars = {"N1p", "N3n", "N4n"}
    p_vars = {"P1c", "P2c", "P3c", "P4c"}
    z_vars = {"Z3c", "Z4c", "Z5c", "Z6c"}
    r_vars = {"R6c"}

    all_vars = chla_vars | n_vars | p_vars | z_vars | r_vars  # Combine all groups

    # ===== INPUT VALIDATION =====
    # Ensure the variable_name provided is recognized, otherwise raise error with red x emoji marks
    if variable_name not in all_vars:
        raise ValueError(f"❌ Unknown variable '{variable_name}'. Allowed variables are: {sorted(all_vars)} ❌")

    # ===== ASSIGN BIN EDGES & COLORMAP =====
    # Select bin edges (Lticks) and base colormap depending on the variable group
    if variable_name in chla_vars:
        # Chlorophyll uses viridis with fine bins for concentration ranges
        Lticks = np.array([0.04, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50,
                           0.80, 1.30, 2.00, 3.00, 4.00, 9.00, 12.0])
        base_cmap = plt.colormaps["viridis"]

    elif variable_name in n_vars:
        # Nitrogen-related variables with different bin edges per subtype, use YlGnBu colormap
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
        base_cmap = plt.colormaps["YlGnBu"]

    elif variable_name in p_vars:
        # Phosphorus-related variables use algae colormap from cmocean with specific bins
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
        base_cmap = cmocean.cm.algae

    elif variable_name in z_vars:
        # Zooplankton groups with different bin edges and turbid colormap from cmocean
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
        base_cmap = cmocean.cm.turbid

    elif variable_name in r_vars:
        # Rare group example (R6c) uses matter colormap and wider bins
        Lticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500])
        base_cmap = cmocean.cm.matter

    # ===== DATA VALIDATION =====
    # Ensure data contains at least some numeric values (not all NaN), else raise error with red x emoji marks
    if np.isnan(np.nanmin(data_in)) or np.isnan(np.nanmax(data_in)):
        raise ValueError("❌ data_in contains only NaNs, cannot determine bins. ❌")

    # Number of discrete colors = number of bins - 1
    n_colors = len(Lticks) - 1

    # ===== COLORMAP GENERATION =====
    # Sample discrete colors evenly from the continuous base colormap
    try:
        colors = base_cmap(np.linspace(0, 1, n_colors))
    except Exception:
        # Fallback: if base_cmap is already discrete, use its colors attribute
        colors = base_cmap.colors if hasattr(base_cmap, "colors") else base_cmap(np.linspace(0, 1, n_colors))

    # Create a discrete ListedColormap from sampled colors
    cmap = ListedColormap(colors)

    # ===== DATA CLIPPING =====
    # Clip data to fall within the min and max bins for proper coloring, but keep NaNs intact
    data_clipped = np.clip(data_in, Lticks[0], Lticks[-1])

    # ===== NORMALIZATION =====
    # Create a BoundaryNorm that maps data values to colormap bins by bin edges (Lticks)
    norm = BoundaryNorm(boundaries=Lticks, ncolors=n_colors, clip=False)

    # Generate tick labels as string representations of bin edges for colorbar display
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
    # ===== VARIABLE GROUPS =====
    # Define groups of benthic variables for specialized handling
    chla_vars = {"Chla"}
    n_vars = {"N1p", "N3n", "N4n"}
    p_vars = {"P1c", "P2c", "P3c", "P4c"}
    z_vars = {"Z3c", "Z4c", "Z5c", "Z6c"}
    r_vars = {"R6c"}

    # Initialize flags and thresholds for hypoxia/hyperoxia (oxygen levels)
    use_custom_cmap = False
    hypoxia_threshold = None
    hyperoxia_threshold = None

    # ===== OXYGEN PLOTTING =====
    if bfm2plot == 'O2o':
        # Use fixed color limits relevant for oxygen concentration (0-350)
        vmin, vmax = 0.0, 350.0
        # Define contour levels evenly spaced across range (29 levels)
        levels = np.linspace(vmin, vmax, 29, endpoint=True)
        num_ticks = None  # colorbar ticks handled by custom cmap or outside
        hypoxia_threshold = 62.5    # Oxygen below this is hypoxia
        hyperoxia_threshold = 312.5 # Oxygen above this is hyperoxia
        try:
            # Use specialized colormap for oxygen if available
            cmap = custom_oxy(vmin=vmin, vmax=vmax,
                              low=hypoxia_threshold, high=hyperoxia_threshold)
        except NameError:
            # Fallback colormap if custom_oxy not defined
            cmap = plt.colormaps['coolwarm']
        use_custom_cmap = False  # Although a special cmap, flag is False here for external use
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    # ===== TEMPERATURE PLOTTING =====
    elif bfm2plot == "votemper":
        vmin = opts.get("vmin_votemper")
        vmax = opts.get("vmax_votemper")
        # Validate that vmin and vmax options are provided
        if vmin is None or vmax is None:
            raise ValueError("❌ vmin_votemper and vmax_votemper must be specified in opts ❌")
        # Create evenly spaced contour levels from options, default to 20 if not specified
        levels = np.linspace(vmin, vmax, opts.get("levels_votemper", 20), endpoint=True)
        # Number of ticks for colorbar, default 5 if missing
        num_ticks = opts.get("num_ticks_votemper", 5)
        # Use thermal colormap suitable for temperature
        cmap = cmocean.cm.thermal
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    # ===== SALINITY PLOTTING =====
    elif bfm2plot == "vosaline":
        vmin = opts.get("vmin_vosaline")
        vmax = opts.get("vmax_vosaline")
        # Ensure limits specified in options
        if vmin is None or vmax is None:
            raise ValueError("❌ vmin_vosaline and vmax_vosaline must be specified in opts ❌")
        levels = np.linspace(vmin, vmax, opts.get("levels_vosaline", 20), endpoint=True)
        num_ticks = opts.get("num_ticks_vosaline", 5)
        cmap = cmocean.cm.haline  # Salinity colormap
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    # ===== DENSITY PLOTTING =====
    elif bfm2plot in ("density", "dense_water"):
        vmin = opts.get("vmin_density")
        vmax = opts.get("vmax_density")
        # Validate presence of min/max density values
        if vmin is None or vmax is None:
            raise ValueError("❌ vmin_density and vmax_density must be specified in opts ❌")
        levels = np.linspace(vmin, vmax, opts.get("levels_density", 20), endpoint=True)
        num_ticks = opts.get("num_ticks_density", 5)
        cmap = cmocean.cm.dense  # Dense water colormap
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold

    # ===== BIOLOGICAL VARIABLES =====
    elif any(bfm2plot in family for family in (chla_vars, n_vars, p_vars, z_vars, r_vars)):
        # For biological groups, a custom colormap is expected to be applied elsewhere
        # Return None for limits and levels but set use_custom_cmap flag True
        return None, None, None, None, None, True, hypoxia_threshold, hyperoxia_threshold

    # ===== DEFAULT FALLBACK =====
    else:
        # Flatten all available data arrays ignoring None entries
        data_arrays = [
            arr for monthly_list in var_dataframe.values()
            for arr in monthly_list if arr is not None
        ]
        # If no data found, raise error
        if not data_arrays:
            raise ValueError("❌ No data arrays found in var_dataframe. ❌")

        # Concatenate all data values into one array and remove NaNs
        all_data = np.concatenate([arr.ravel() for arr in data_arrays])
        all_data = all_data[~np.isnan(all_data)]
        if all_data.size == 0:
            raise ValueError("❌ No valid data found to determine vmin and vmax. ❌")

        # Determine min and max for color scale from data
        vmin = float(all_data.min())
        vmax = float(all_data.max())
        # Create 20 evenly spaced contour levels between min and max
        levels = np.linspace(vmin, vmax, 20, endpoint=True)
        num_ticks = 5  # Default number of ticks
        cmap = 'jet'   # Default fallback colormap string
        return vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypoxia_threshold, hyperoxia_threshold
###############################################################################

############################################################################### 
def cmocean_to_plotly(cmap_name, n=256):
    """
    Convert a cmocean colormap to a Plotly colorscale.
    
    Parameters
    ----------
    cmap_name : str
        Name of the cmocean colormap to convert.
    n : int, optional
        Number of discrete colors in the output colorscale (default is 256).
    
    Returns
    -------
    colorscale : list
        Plotly-compatible colorscale as list of [position, 'rgb(r,g,b)'] pairs.
    """

    # ===== COLORMAP RETRIEVAL =====
    # Retrieve the cmocean colormap object by attribute name
    cmap = getattr(cmocean.cm, cmap_name)  # e.g. cmocean.cm.thermal

    colorscale = []  # Initialize list to hold Plotly colorscale entries

    # ===== COLOR SAMPLING LOOP =====
    for i in range(n):
        # Normalize index to [0,1] range for colormap input
        normalized_val = i / (n - 1)

        # Get the RGBA color tuple at normalized_val position in cmap
        color = cmap(normalized_val)

        # Convert each RGB channel from float [0,1] to int [0,255]
        r, g, b, _ = [int(255 * c) for c in color]

        # Append position and rgb string to colorscale list
        colorscale.append([normalized_val, f'rgb({r},{g},{b})'])

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

    # ===== INVERT COLORSCALE =====
    # colorscale is a list of pairs [position (float), color string]
    # Example: [[0.0, 'rgb(...)'], [0.1, 'rgb(...)'], ..., [1.0, 'rgb(...)']]

    # Reverse the order of colorscale entries to flip the gradient,
    # and invert each position t -> 1 - t to maintain positions in [0,1]
    inverted = [[1 - t, color] for t, color in reversed(colorscale)]

    return inverted
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
    """

    # ===== NORMALIZE THRESHOLDS =====
    # Normalize 'low' and 'high' thresholds to a 0-1 scale relative to data range
    # This lets us specify color stops as fractions of the colormap length.
    x_low = (low - vmin) / (vmax - vmin)
    x_high = (high - vmin) / (vmax - vmin)

    # Validate normalized thresholds: must be between 0 and 1 and low < high
    if not (0 <= x_low < x_high <= 1):
        raise ValueError(f"❌ Thresholds normalized must be in ascending order between 0 and 1, got low={x_low}, high={x_high} ❌")

    # ===== DEFINE COLORS =====
    # Define start and end RGB colors for each gradient segment, normalized [0,1]
    red_start = (0x41/255, 0x05/255, 0x05/255)   # dark red (#410505)
    red_end = (0x8B/255, 0x12/255, 0x07/255)     # brighter red (#8B1207)

    gray_start = (0x50/255, 0x50/255, 0x50/255)  # dark gray (#505050)
    gray_end = (0xF1/255, 0xF1/255, 0xEF/255)    # light gray (#F1F1EF)

    yellow_start = (0xF2/255, 0xF9/255, 0x5D/255) # pale yellow (#F2F95D)
    yellow_end = (0xDE/255, 0xB2/255, 0x1B/255)   # darker yellow (#DEB21B)

    # ===== CONSTRUCT COLOR DICTIONARY =====
    # Map the colors to red, green, blue channels with corresponding stops:
    # Each tuple: (position, start_value, end_value)
    # Positions where color changes happen: 0.0, x_low, x_high, 1.0
    cdict = {
        'red': [
            (0.0, red_start[0], red_start[0]),       # start red gradient
            (x_low, red_end[0], red_end[0]),         # end red gradient / start gray gradient
            (x_low, gray_start[0], gray_start[0]),   # start gray gradient
            (x_high, gray_end[0], gray_end[0]),      # end gray gradient / start yellow gradient
            (x_high, yellow_start[0], yellow_start[0]), # start yellow gradient
            (1.0, yellow_end[0], yellow_end[0])      # end yellow gradient
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

    # ===== CREATE COLORMAP OBJECT =====
    # Create the LinearSegmentedColormap with 256 discrete colors by default
    cmap = mcolors.LinearSegmentedColormap(name, segmentdata=cdict, N=256)

    return cmap