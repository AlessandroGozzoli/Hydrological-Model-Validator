import numpy as np
from typing import Tuple, Union
from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess

###############################################################################
def fit_huber(mod_data: np.ndarray,
              sat_data: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a robust linear regression (Huber) model to 1D predictor and response arrays.

    Parameters
    ----------
    mod_data : np.ndarray
        1D array of model data (predictor variable). Must be the same length as `sat_data`.

    sat_data : np.ndarray
        1D array of satellite data (response variable). Must be the same length as `mod_data`.

    Returns
    -------
    tuple of np.ndarray
        Tuple (x_vals, y_vals) containing 100 points on the fitted Huber regression line.

    Raises
    ------
    ValueError
        If inputs are not 1D NumPy arrays of the same length, or are empty.

    Examples
    --------
    >>> mod = np.array([1, 2, 3, 4])
    >>> sat = np.array([1.2, 1.9, 3.1, 3.9])
    >>> x, y = fit_huber(mod, sat)
    >>> len(x), len(y)
    (100, 100)
    """
    if not isinstance(mod_data, np.ndarray):
        raise ValueError("Input 'mod_data' must be a NumPy array.")
    if not isinstance(sat_data, np.ndarray):
        raise ValueError("Input 'sat_data' must be a NumPy array.")
    if mod_data.ndim != 1 or sat_data.ndim != 1:
        raise ValueError("Both 'mod_data' and 'sat_data' must be 1D arrays.")
    if mod_data.size != sat_data.size:
        raise ValueError("'mod_data' and 'sat_data' must have the same length.")
    if mod_data.size == 0:
        raise ValueError("Input arrays must not be empty.")

    model = HuberRegressor().fit(mod_data.reshape(-1, 1), sat_data)
    x_vals = np.linspace(mod_data.min(), mod_data.max(), 100)
    y_vals = model.predict(x_vals.reshape(-1, 1))

    return x_vals, y_vals
###############################################################################

###############################################################################
def fit_lowess(mod_data: np.ndarray,
               sat_data: np.ndarray,
               frac: float = 0.3) -> np.ndarray:
    """
    Compute LOWESS (Locally Weighted Scatterplot Smoothing) of sat_data vs mod_data.

    Parameters
    ----------
    mod_data : np.ndarray
        1D array of model data (predictor). Must be same length as `sat_data`.

    sat_data : np.ndarray
        1D array of satellite data (response). Must be same length as `mod_data`.

    frac : float, optional
        The fraction of data used when estimating each y-value.
        Must be between 0 and 1 (default is 0.3).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with columns [mod_data_sorted, smoothed_sat_data].
        Suitable for plotting the smoothed curve.

    Raises
    ------
    ValueError
        If input arrays are not 1D NumPy arrays of equal length, or
        if frac is not in (0, 1].

    Examples
    --------
    >>> mod = np.array([1, 2, 3, 4])
    >>> sat = np.array([1.2, 1.8, 2.5, 3.9])
    >>> smooth = fit_lowess(mod, sat, frac=0.5)
    >>> smooth.shape
    (4, 2)
    """
    if not isinstance(mod_data, np.ndarray):
        raise ValueError("Input 'mod_data' must be a NumPy array.")
    if not isinstance(sat_data, np.ndarray):
        raise ValueError("Input 'sat_data' must be a NumPy array.")
    if mod_data.ndim != 1 or sat_data.ndim != 1:
        raise ValueError("Both 'mod_data' and 'sat_data' must be 1D arrays.")
    if mod_data.size != sat_data.size:
        raise ValueError("'mod_data' and 'sat_data' must have the same length.")
    if not (0 < frac <= 1):
        raise ValueError("Input 'frac' must be between 0 (exclusive) and 1 (inclusive).")

    sorted_idx = np.argsort(mod_data)
    smoothed = lowess(sat_data[sorted_idx], mod_data[sorted_idx], frac=frac)
    return smoothed
###############################################################################
    
###############################################################################    
def round_up_to_nearest(x: Union[float, int], base: float = 1.0) -> float:
    """
    Round up the given number to the nearest multiple of the specified base.

    Parameters
    ----------
    x : float or int
        Number to round up.

    base : float, optional
        The base multiple to round up to. Must be positive (default is 1.0).

    Returns
    -------
    float
        The rounded up value as a multiple of the base.

    Raises
    ------
    ValueError
        If `base` is not positive.

    Examples
    --------
    >>> round_up_to_nearest(5.3, base=2)
    6.0
    >>> round_up_to_nearest(7, base=0.5)
    7.0
    """
    if base <= 0:
        raise ValueError("Base must be a positive number.")
    return base * np.ceil(x / base)
###############################################################################

###############################################################################
def compute_coverage_stats(
    data: np.ndarray,
    Mmask: np.ndarray,
    plot: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the percentage of data availability and cloud coverage within a basin mask
    for each time step of a 3D dataset.

    Parameters
    ----------
    data : np.ndarray
        3D data array of shape (time, y, x), e.g., Schl_complete.
    Mmask : np.ndarray
        2D boolean mask array (True = ocean, False = land) with shape (y, x).
    plot : bool, optional
        If True, plots the % data available and % cloud coverage (default is True).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - data_available_percent : np.ndarray
            Percentage of non-NaN data inside the basin at each time step.
        - cloud_coverage_percent : np.ndarray
            Percentage of NaN (cloud-covered) data inside the basin at each time step.

    Raises
    ------
    ValueError
        If input dimensions do not match expected shapes.
    """

    # VALIDATE INPUT SHAPES
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array (time, y, x)")
    if Mmask.shape != data.shape[1:]:
        raise ValueError(f"Shape of Mmask {Mmask.shape} does not match spatial dimensions of data {data.shape[1:]}")

    # Ensure mask is boolean
    Mmask = Mmask.astype(bool)

    T = data.shape[0]
    data_available_percent = np.empty(T)
    cloud_coverage_percent = np.empty(T)

    for t in range(T):
        basin_data = data[t][Mmask]
        total = basin_data.size
        valid = np.count_nonzero(~np.isnan(basin_data))
        cloud = np.count_nonzero(np.isnan(basin_data))

        data_available_percent[t] = 100 * valid / total
        cloud_coverage_percent[t] = 100 * cloud / total
    
    return data_available_percent, cloud_coverage_percent