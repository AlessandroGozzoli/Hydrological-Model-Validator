import numpy as np
import re
from typing import Tuple, Union, Iterator
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns

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