import numpy as np
from typing import Tuple, Union
from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess

###############################################################################
def fit_huber(mod_data: np.ndarray, sat_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a robust linear regression (Huber) model to mod_data and sat_data.

    Args:
        mod_data (np.ndarray): 1D array of model data (predictor).
        sat_data (np.ndarray): 1D array of satellite data (response).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (x_vals, y_vals) representing
                                       points on the fitted regression line.
    """
    # Validate inputs
    assert isinstance(mod_data, np.ndarray), "mod_data must be a numpy array"
    assert isinstance(sat_data, np.ndarray), "sat_data must be a numpy array"
    assert mod_data.ndim == 1, "mod_data must be 1D"
    assert sat_data.ndim == 1, "sat_data must be 1D"
    assert mod_data.size == sat_data.size, "mod_data and sat_data must have the same length"
    assert mod_data.size > 0, "Input arrays must not be empty"

    model = HuberRegressor().fit(mod_data.reshape(-1, 1), sat_data)
    x_vals = np.linspace(mod_data.min(), mod_data.max(), 100)
    y_vals = model.predict(x_vals.reshape(-1, 1))
    return x_vals, y_vals
###############################################################################

###############################################################################
def fit_lowess(
    mod_data: np.ndarray, 
    sat_data: np.ndarray, 
    frac: float = 0.3
) -> np.ndarray:
    """
    Compute LOWESS (Locally Weighted Scatterplot Smoothing) of sat_data vs mod_data.

    Args:
        mod_data (np.ndarray): 1D array of model data (predictor).
        sat_data (np.ndarray): 1D array of satellite data (response).
        frac (float, optional): The fraction of data used when estimating each y-value.
                                Default is 0.3.

    Returns:
        np.ndarray: Array of shape (N, 2) with columns [mod_data_sorted, smoothed_sat_data].
                    Used for plotting the smoothed curve.
    """
    # Input validation
    assert isinstance(mod_data, np.ndarray), "mod_data must be a numpy array"
    assert isinstance(sat_data, np.ndarray), "sat_data must be a numpy array"
    assert mod_data.ndim == 1, "mod_data must be 1D"
    assert sat_data.ndim == 1, "sat_data must be 1D"
    assert mod_data.size == sat_data.size, "mod_data and sat_data must have the same length"
    assert 0 < frac <= 1, "frac must be between 0 and 1"

    sorted_idx = np.argsort(mod_data)
    smoothed = lowess(sat_data[sorted_idx], mod_data[sorted_idx], frac=frac)
    return smoothed
###############################################################################
    
###############################################################################    
def round_up_to_nearest(x: Union[float, int], base: float = 1.0) -> float:
    """
    Round up the given number to the nearest multiple of the specified base.

    Args:
        x (float or int): Number to round up.
        base (float): The base multiple to round up to (default is 1.0).

    Returns:
        float: The rounded up value.
    """
    assert base > 0, "Base must be a positive number"
    return base * np.ceil(x / base)
###############################################################################