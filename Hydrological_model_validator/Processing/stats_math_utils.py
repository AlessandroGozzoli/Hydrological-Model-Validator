import numpy as np
from typing import Tuple, Union
from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
import xarray as xr
from scipy.signal import detrend
import pandas as pd

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
###############################################################################

###############################################################################
def detrend_dim(
    da: xr.DataArray,
    dim: str,
    mask: xr.DataArray = None,
    min_valid_points: int = 5
) -> xr.DataArray:
    """
    Detrend a DataArray along a dimension using scipy.signal.detrend,
    optionally masking out points and skipping those with insufficient valid data.

    Parameters
    ----------
    da : xr.DataArray
        Input data with dimension `dim`.
    dim : str
        Dimension name along which to detrend (e.g., 'time').
    mask : xr.DataArray, optional
        Boolean mask to apply before detrending. True = valid points.
    min_valid_points : int, default=5
        Minimum number of valid (non-NaN) points along `dim` required to perform detrending.

    Returns
    -------
    xr.DataArray
        Detrended data array.
    """
    if mask is not None:
        da = da.where(mask)

    def detrend_1d(x):
        # Ensure float dtype to support NaNs
        x = x.astype(float)
        nans = np.isnan(x)

        if np.sum(~nans) < min_valid_points:
            # Not enough valid points: return all NaNs of same shape & dtype float
            return np.full_like(x, np.nan, dtype=float)

        if nans.any():
            # Indices of valid and invalid points
            valid_idx = np.flatnonzero(~nans)
            invalid_idx = np.flatnonzero(nans)

            # Ensure valid_idx is sorted for np.interp (usually true)
            sorted_valid_idx = np.sort(valid_idx)

            # Corresponding values for valid indices
            valid_vals = x[sorted_valid_idx]

            # Interpolate linearly over missing values
            x_interp = x.copy()
            x_interp[invalid_idx] = np.interp(invalid_idx, sorted_valid_idx, valid_vals)
        else:
            x_interp = x

        # Perform linear detrending
        detrended = detrend(x_interp, type='linear')

        # Restore NaNs in original positions
        detrended[nans] = np.nan

        return detrended

    detrended = xr.apply_ufunc(
        detrend_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype],
    )
    return detrended
###############################################################################

###############################################################################
def mean_bias(m, o, time_dim='time'):
    """Compute the mean bias between model and observations."""
    m_mean = m.mean(dim=time_dim)
    o_mean = o.mean(dim=time_dim)
    return m_mean - o_mean
###############################################################################

###############################################################################
def standard_deviation_error(m, o, time_dim='time'):
    """Compute standard deviation error (CRMSD) between m and o."""
    m_mean = m.mean(dim=time_dim)
    o_mean = o.mean(dim=time_dim)
    m_anom = m - m_mean
    o_anom = o - o_mean
    var_m = (m_anom ** 2).mean(dim=time_dim)
    var_o = (o_anom ** 2).mean(dim=time_dim)
    cov_mo = (m_anom * o_anom).mean(dim='time')
    return (var_m + var_o - 2 * cov_mo) ** 0.5
###############################################################################

###############################################################################
def cross_correlation(m, o, time_dim='time'):
    """Compute Pearson correlation coefficient between m and o."""
    m_mean = m.mean(dim=time_dim)
    o_mean = o.mean(dim=time_dim)
    m_anom = m - m_mean
    o_anom = o - o_mean
    cov = (m_anom * o_anom).mean(dim=time_dim)
    std_m = (m_anom ** 2).mean(dim=time_dim) ** 0.5
    std_o = (o_anom ** 2).mean(dim=time_dim) ** 0.5
    return cov * ((1 / std_m) * (1 / std_o))
###############################################################################

###############################################################################
def corr_no_nan(series1, series2):
    combined = pd.concat([series1, series2], axis=1).dropna()
    return combined.iloc[:,0].corr(combined.iloc[:,1])
###############################################################################

###############################################################################
def std_dev(da, time_dim='time'):
    """Compute standard deviation of a DataArray along time dimension."""
    mean = da.mean(dim=time_dim)
    return ((da - mean) ** 2).mean(dim=time_dim) ** 0.5
###############################################################################

###############################################################################
def unbiased_rmse(m, o, time_dim='time'):
    """Compute unbiased RMSE (centered RMSE) between m and o."""
    m_mean = m.mean(dim=time_dim)
    o_mean = o.mean(dim=time_dim)
    m_anom = m - m_mean
    o_anom = o - o_mean
    return ((m_anom - o_anom) ** 2).mean(dim=time_dim) ** 0.5
###############################################################################