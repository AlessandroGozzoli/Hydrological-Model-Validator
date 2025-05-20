import numpy as np
import re
from typing import Tuple, Union
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

###############################################################################
def format_unit(unit: str) -> str:
    """
    Formats a chemical unit string into LaTeX math mode with subscripts and exponents.

    Args:
        unit (str): Unit string, e.g. "O2/m3", "mg/L"

    Returns:
        str: LaTeX formatted unit string, e.g. "$\\frac{O_2}{m^{3}}$"
    """

    def add_subscripts(s: str) -> str:
        # Convert chemical notation like O2 -> O_2
        return re.sub(r'([A-Za-z]{1,2})(\d+)', r'\1_{\2}', s)

    def handle_exponents(s: str) -> str:
        # Convert units like m3 -> m^{3}
        return re.sub(r'([a-zA-Z])(\d+)', r'\1^{\2}', s)

    if '/' in unit:
        numerator, denominator = map(str.strip, unit.split('/', 1))
        numerator_fmt = add_subscripts(numerator)
        denominator_fmt = handle_exponents(denominator)
        return f'$\\frac{{{numerator_fmt}}}{{{denominator_fmt}}}$'
    else:
        unit_fmt = add_subscripts(unit)
        unit_fmt = handle_exponents(unit_fmt)
        return f'${unit_fmt}$'
###############################################################################

###############################################################################
def get_variable_label_unit(variable_name: str) -> Tuple[str, str]:
    """
    Get a descriptive label and unit string for a given variable name.

    Args:
        variable_name (str): Short variable code/name.

    Returns:
        Tuple[str, str]: A tuple with (label, unit). If variable not found,
                         returns (variable_name, '').
    """
    mapping = {
        'SST': ('Sea Surface Temperature', '[$°C$]'),
        'CHL_L3': ('Chlorophyll (Level 3)', '[$mg/m^3$]'),
        'CHL_L4': ('Chlorophyll (Level 4)', '[$mg/m^3$]'),
    }
    return mapping.get(variable_name, (variable_name, ''))
###############################################################################

###############################################################################
def fill_annular_region(
    ax: Axes,
    r_in: float,
    r_out: float,
    color: str,
    alpha: float = 0.3
) -> None:
    """
    Fill an annular (ring-shaped) region between two radii on a given matplotlib Axes.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes to draw on.
        r_in (float): Inner radius of the annulus (must be >= 0 and <= r_out).
        r_out (float): Outer radius of the annulus (must be > r_in).
        color (str): Fill color.
        alpha (float): Transparency level of the fill (default 0.3).

    Returns:
        None
    """
    assert 0 <= r_in <= r_out, "r_in must be non-negative and less than or equal to r_out"
    assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"

    theta = np.linspace(0, 2 * np.pi, 500)
    x_outer = r_out * np.cos(theta)
    y_outer = r_out * np.sin(theta)
    x_inner = r_in * np.cos(theta[::-1])
    y_inner = r_in * np.sin(theta[::-1])

    x = np.concatenate([x_outer, x_inner])
    y = np.concatenate([y_outer, y_inner])

    ax.fill(x, y, color=color, alpha=alpha, zorder=0)
###############################################################################

###############################################################################    
def get_min_max_for_identity_line(
    x: Union[np.ndarray, list, tuple],
    y: Union[np.ndarray, list, tuple]
) -> Tuple[float, float]:
    """
    Compute the minimum and maximum values from two numeric sequences
    to define the limits of an identity line (y = x).

    Args:
        x (array-like): First numeric sequence.
        y (array-like): Second numeric sequence.

    Returns:
        Tuple[float, float]: (min_value, max_value) covering both sequences,
                             ignoring NaNs.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    assert x_arr.size > 0, "Input x cannot be empty"
    assert y_arr.size > 0, "Input y cannot be empty"

    min_val = min(np.nanmin(x_arr), np.nanmin(y_arr))
    max_val = max(np.nanmax(x_arr), np.nanmax(y_arr))
    return min_val, max_val
###############################################################################

###############################################################################
def _style_axis_custom(ax: plt.Axes, grid_style: str = '--') -> None:
    """Apply consistent styling to axes with custom grid style."""
    ax.tick_params(width=2)
    ax.grid(True, linestyle=grid_style)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
###############################################################################