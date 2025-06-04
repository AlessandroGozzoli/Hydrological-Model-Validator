import numpy as np
from typing import Tuple, Union
from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
import xarray as xr
from scipy.signal import detrend
import pandas as pd
from scipy.fft import fft, fftfreq

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
    if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in m.dims:
        return np.nanmean(m) - np.nanmean(o)
    return m.mean(dim=time_dim) - o.mean(dim=time_dim)
###############################################################################

###############################################################################
def standard_deviation_error(m, o, time_dim='time'):
    """Compute difference between standard deviations of m and o.

    Raises ValueError if inputs have different lengths along time_dim.
    """
    # Check length compatibility
    if isinstance(m, (pd.Series, np.ndarray)) and isinstance(o, (pd.Series, np.ndarray)):
        if len(m) != len(o):
            raise ValueError("Inputs 'm' and 'o' must have the same length.")
    elif time_dim in getattr(m, 'dims', []) and time_dim in getattr(o, 'dims', []):
        if m.sizes[time_dim] != o.sizes[time_dim]:
            raise ValueError(f"Inputs 'm' and 'o' must have the same length along dimension '{time_dim}'.")
    else:
        # If dims are missing, no check is performed (optional: raise warning or error)
        pass

    # Compute difference of std devs
    if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in getattr(m, 'dims', []):
        sdm = np.nanstd(m)
        sdo = np.nanstd(o)
    else:
        sdm = m.std(dim=time_dim)
        sdo = o.std(dim=time_dim)
    return sdm - sdo
###############################################################################

###############################################################################
def cross_correlation(m, o, time_dim='time'):
    """Compute Pearson correlation coefficient between m and o."""
    if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in m.dims:
        m_mean = np.nanmean(m)
        o_mean = np.nanmean(o)
        m_anom = m - m_mean
        o_anom = o - o_mean
        cov = np.nanmean(m_anom * o_anom)
        std_m = np.sqrt(np.nanmean(m_anom ** 2))
        std_o = np.sqrt(np.nanmean(o_anom ** 2))
    else:
        m_mean = m.mean(dim=time_dim)
        o_mean = o.mean(dim=time_dim)
        m_anom = m - m_mean
        o_anom = o - o_mean
        cov = (m_anom * o_anom).mean(dim=time_dim)
        std_m = np.sqrt((m_anom ** 2).mean(dim=time_dim))
        std_o = np.sqrt((o_anom ** 2).mean(dim=time_dim))
    return cov / (std_m * std_o)
###############################################################################

###############################################################################
def corr_no_nan(series1, series2):
    """Pandas-only quick Pearson correlation ignoring NaNs."""
    combined = pd.concat([series1, series2], axis=1).dropna()
    return combined.iloc[:, 0].corr(combined.iloc[:, 1])
###############################################################################

###############################################################################
def std_dev(da, time_dim='time'):
    """Compute standard deviation along time dimension."""
    if isinstance(da, (pd.Series, np.ndarray)) or time_dim not in da.dims:
        mean = np.nanmean(da)
        return np.sqrt(np.nanmean((da - mean) ** 2))
    mean = da.mean(dim=time_dim)
    return ((da - mean) ** 2).mean(dim=time_dim) ** 0.5
###############################################################################

###############################################################################
def unbiased_rmse(m, o, time_dim='time'):
    """Compute unbiased RMSE (centered RMSE) between m and o."""
    if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in m.dims:
        m_mean = np.nanmean(m)
        o_mean = np.nanmean(o)
        m_anom = m - m_mean
        o_anom = o - o_mean
        return np.sqrt(np.nanmean((m_anom - o_anom) ** 2))
    m_mean = m.mean(dim=time_dim)
    o_mean = o.mean(dim=time_dim)
    m_anom = m - m_mean
    o_anom = o - o_mean
    return ((m_anom - o_anom) ** 2).mean(dim=time_dim) ** 0.5
###############################################################################

###############################################################################
def spatial_mean(data_array, mask):
    # Mask is 2D boolean (lat, lon)
    return data_array.where(mask).mean(dim=['lat', 'lon'], skipna=True)
###############################################################################

###############################################################################
def compute_lagged_correlations(series1, series2, max_lag=30):
    lags = range(-max_lag, max_lag + 1)
    results = {}
    for lag in lags:
        shifted = series2.shift(lag)
        results[lag] = corr_no_nan(series1, shifted)
    return pd.Series(results)
###############################################################################

###############################################################################
def compute_fft(data, dt=1):
    """
    Compute FFT and positive frequencies for input data.

    Parameters:
    - data: dict of 1D arrays/Series or a single 1D array/Series
    - dt: sampling interval (default=1)

    Returns:
    - freqs: array of positive FFT frequencies
    - fft_result: dict of FFT arrays if input is dict, else single FFT array
    """
    if isinstance(data, dict):
        # Assume all series/arrays have same length
        N = len(next(iter(data.values())))
        freqs = fftfreq(N, dt)[:N//2]
        fft_result = {
            key: fft(arr)[:N//2]
            for key, arr in data.items()
        }
        return freqs, fft_result

    else:
        # Single array/series input
        N = len(data)
        freqs = fftfreq(N, dt)[:N//2]
        fft_result = fft(data)[:N//2]
        return freqs, fft_result