import numpy as np
from typing import Tuple, Union
from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess

###############################################################################
def fit_huber(mod_data: np.ndarray, sat_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    Tuple[np.ndarray, np.ndarray]
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
    # === INPUT VALIDATION ===
    if not isinstance(mod_data, np.ndarray) or not isinstance(sat_data, np.ndarray):
        raise ValueError("Both inputs must be NumPy arrays.")
    if mod_data.ndim != 1 or sat_data.ndim != 1:
        raise ValueError("Both arrays must be 1D.")
    if mod_data.size != sat_data.size:
        raise ValueError("Arrays must have the same length.")
    if mod_data.size == 0:
        raise ValueError("Input arrays must not be empty.")

    # === FITTING HUBER REGRESSOR ===
    huber = HuberRegressor()
    huber.fit(mod_data[:, None], sat_data)

    # === GENERATING PREDICTED LINE ===
    x_vals = np.linspace(mod_data.min(), mod_data.max(), 100)
    y_vals = huber.predict(x_vals[:, None])

    return x_vals, y_vals
###############################################################################

###############################################################################
def fit_lowess(
    mod_data: np.ndarray,
    sat_data: np.ndarray,
    frac: float = 0.3
) -> np.ndarray:
    """
    Fit a LOWESS (Locally Weighted Scatterplot Smoothing) curve to satellite vs. model data.

    Parameters
    ----------
    mod_data : np.ndarray
        1D array of model data (predictor). Must be the same length as `sat_data`.

    sat_data : np.ndarray
        1D array of satellite data (response). Must be the same length as `mod_data`.

    frac : float, optional
        Fraction of the data used when estimating each y-value.
        Must be in the interval (0, 1]. Default is 0.3.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with columns [sorted_mod_data, smoothed_sat_data],
        suitable for plotting.

    Raises
    ------
    ValueError
        If inputs are not 1D NumPy arrays of equal length,
        or if `frac` is not in (0, 1].

    Examples
    --------
    >>> mod = np.array([1, 2, 3, 4])
    >>> sat = np.array([1.2, 1.8, 2.5, 3.9])
    >>> smooth = fit_lowess(mod, sat, frac=0.5)
    >>> smooth.shape
    (4, 2)
    """
    # === VALIDATION ===
    if not isinstance(mod_data, np.ndarray) or not isinstance(sat_data, np.ndarray):
        raise ValueError("Both inputs must be NumPy arrays.")
    if mod_data.ndim != 1 or sat_data.ndim != 1:
        raise ValueError("Both arrays must be 1D.")
    if mod_data.size != sat_data.size or mod_data.size == 0:
        raise ValueError("Arrays must be non-empty and of the same length.")
    if not (0 < frac <= 1):
        raise ValueError("Parameter 'frac' must be in the interval (0, 1].")

    # === SORT AND APPLY LOWESS ===
    order = np.argsort(mod_data)
    mod_sorted = mod_data[order]
    sat_sorted = sat_data[order]

    smoothed = lowess(sat_sorted, mod_sorted, frac=frac, return_sorted=True)

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
        raise ValueError("Base must be positive.")
    return base * np.ceil(x / base)
###############################################################################

###############################################################################
def compute_coverage_stats(
    data: np.ndarray,
    Mmask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute percentage of data availability and cloud coverage within a basin mask
    for each time step of a 3D dataset.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (time, y, x), e.g., Schl_complete.
    Mmask : np.ndarray
        2D boolean mask (True = ocean, False = land), shape (y, x).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - data_available_percent : % of non-NaN data inside mask per time step.
        - cloud_coverage_percent : % of NaN data (cloud coverage) inside mask per time step.

    Raises
    ------
    ValueError
        If input dimensions do not match expected shapes.
    """
    if data.ndim != 3:
        raise ValueError("Input 'data' must be 3D (time, y, x).")
    if Mmask.shape != data.shape[1:]:
        raise ValueError(f"Shape of Mmask {Mmask.shape} does not match data spatial dims {data.shape[1:]}.")

    Mmask = Mmask.astype(bool, copy=False)

    masked_data = data[:, Mmask]  # shape: (time, masked_points)

    total_points = masked_data.shape[1]
    if total_points == 0:
        n_time = data.shape[0]
        return np.full(n_time, np.nan), np.full(n_time, np.nan)

    valid_counts = np.count_nonzero(~np.isnan(masked_data), axis=1)
    cloud_counts = total_points - valid_counts

    data_available_percent = 100 * valid_counts / total_points
    cloud_coverage_percent = 100 * cloud_counts / total_points

    return data_available_percent, cloud_coverage_percent