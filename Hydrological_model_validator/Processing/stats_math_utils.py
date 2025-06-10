import numpy as np
from typing import Tuple, Union, Dict
from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
import xarray as xr
from scipy.signal import detrend
import pandas as pd
from scipy.fft import fft, fftfreq

import logging
from eliot import start_action, log_message

from .time_utils import Timer

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
    # ===== INPUT VALIDATION =====
    if base <= 0:
        # Prevent invalid operation — rounding to a non-positive base is undefined
        raise ValueError("Base must be positive.")
    
    # Divide x by base to scale the problem, apply ceiling to ensure rounding *up*, 
    # then scale back by multiplying with base to get the nearest upper multiple
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

    Examples
    --------
    >>> data = np.random.rand(10, 3, 3)
    >>> data[:, 1:, 1:] = np.nan  # simulate cloud-covered region
    >>> mask = np.array([[1, 1, 1],
    ...                  [1, 1, 1],
    ...                  [1, 1, 1]], dtype=bool)
    >>> compute_coverage_stats(data, mask)
    (array([...]), array([...]))
    """
    # ===== INPUT VALIDATION =====
    if data.ndim != 3:
        # Ensure input data is 3D (time, y, x)
        raise ValueError("Input 'data' must be 3D (time, y, x).")
    if Mmask.shape != data.shape[1:]:
        # Spatial dimensions of the mask must match those of the data
        raise ValueError(f"Shape of Mmask {Mmask.shape} does not match data spatial dims {data.shape[1:]}.")

    # Ensure Mmask is boolean without copying unless necessary
    Mmask = Mmask.astype(bool, copy=False)

    # Apply mask to spatial dimensions, reduces each 2D frame to a 1D vector of masked values
    masked_data = data[:, Mmask]  # shape: (time, masked_points)

    total_points = masked_data.shape[1]
    if total_points == 0:
        # If the mask selects zero points, return NaNs for all time steps
        n_time = data.shape[0]
        return np.full(n_time, np.nan), np.full(n_time, np.nan)

    # ===== COUNTING =====
    # Count valid (non-NaN) values per time step
    valid_counts = np.count_nonzero(~np.isnan(masked_data), axis=1)

    # Remaining points are assumed to be cloud-covered (i.e., NaNs)
    cloud_counts = total_points - valid_counts

    # Convert counts to percentage relative to total masked points
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
    # ===== INPUT VALIDATION =====
    if mask is not None:
        da = da.where(mask)

    # ===== FUNCTION TO DETREND 1D =====
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

            # ===== FIXING =====
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

    # ===== DETRENDING PROCESS =====
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
def mean_bias(
    m: Union[np.ndarray, pd.Series, xr.DataArray],
    o: Union[np.ndarray, pd.Series, xr.DataArray],
    time_dim: str = 'time'
) -> Union[float, xr.DataArray]:
    """
    Compute the mean bias between model and observations over the specified time dimension.

    Parameters
    ----------
    m : np.ndarray, pd.Series, or xr.DataArray
        Model data.
    o : np.ndarray, pd.Series, or xr.DataArray
        Observation data.
    time_dim : str, optional
        Name of the time dimension to compute mean over (default is 'time').

    Returns
    -------
    float or xr.DataArray
        Mean bias: model minus observation, averaged over time.

    Examples
    --------
    >>> m = np.array([1.1, 2.2, 3.3])
    >>> o = np.array([1.0, 2.0, 3.0])
    >>> mean_bias(m, o)
    0.19999999999999973
    """
    # If input is not an xarray with the specified time dimension,
    # compute global mean bias using np.nanmean to ignore NaNs
    if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in m.dims:
        return np.nanmean(m) - np.nanmean(o)

    # Compute bias along the specified time dimension (works for xarray DataArray)
    return m.mean(dim=time_dim) - o.mean(dim=time_dim)
###############################################################################

###############################################################################
def standard_deviation_error(
    m: Union[np.ndarray, pd.Series, xr.DataArray],
    o: Union[np.ndarray, pd.Series, xr.DataArray],
    time_dim: str = 'time'
) -> Union[float, xr.DataArray]:
    """
    Compute the difference between the standard deviations of model and observation data.

    Parameters
    ----------
    m : np.ndarray, pd.Series, or xr.DataArray
        Model data.
    o : np.ndarray, pd.Series, or xr.DataArray
        Observation data.
    time_dim : str, optional
        Name of the time dimension to compute std over (default is 'time').

    Returns
    -------
    float or xr.DataArray
        The difference between model and observation standard deviations (model - obs).

    Raises
    ------
    ValueError
        If m and o have different lengths along the time dimension.

    Examples
    --------
    >>> m = np.array([1.0, 2.0, 3.0])
    >>> o = np.array([1.5, 2.5, 3.5])
    >>> standard_deviation_error(m, o)
    0.0
    """
    # --- Validate input lengths based on input type ---
    if isinstance(m, (pd.Series, np.ndarray)) and isinstance(o, (pd.Series, np.ndarray)):
        # For array-like inputs, check length directly
        if len(m) != len(o):
            raise ValueError("Inputs 'm' and 'o' must have the same length.")
    elif time_dim in getattr(m, 'dims', []) and time_dim in getattr(o, 'dims', []):
        # For xarray DataArrays, check size along time_dim
        if m.sizes[time_dim] != o.sizes[time_dim]:
            raise ValueError(f"Inputs 'm' and 'o' must have the same length along dimension '{time_dim}'.")

    # --- Compute std dev difference based on type ---
    if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in getattr(m, 'dims', []):
        # Fallback to global std dev using np.nanstd to ignore NaNs
        sdm = np.nanstd(m)
        sdo = np.nanstd(o)
    else:
        # Compute std dev along specified dimension for xarray
        sdm = m.std(dim=time_dim)
        sdo = o.std(dim=time_dim)

    return sdm - sdo
###############################################################################

###############################################################################
def cross_correlation(
    m: Union[np.ndarray, pd.Series, xr.DataArray],
    o: Union[np.ndarray, pd.Series, xr.DataArray],
    time_dim: str = 'time'
) -> Union[float, xr.DataArray]:
    """
    Compute the Pearson correlation coefficient between model and observation data.

    Parameters
    ----------
    m : np.ndarray, pd.Series, or xr.DataArray
        Model data.
    o : np.ndarray, pd.Series, or xr.DataArray
        Observation data.
    time_dim : str, optional
        Name of the time dimension over which to compute the correlation (default is 'time').

    Returns
    -------
    float or xr.DataArray
        Pearson correlation coefficient.

    Examples
    --------
    >>> m = np.array([1.0, 2.0, 3.0])
    >>> o = np.array([1.1, 1.9, 3.2])
    >>> cross_correlation(m, o)
    0.9981908926857269
    """
    if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in getattr(m, 'dims', []):
        # Fallback for array-like input or when time_dim is missing: compute global correlation
        m_mean = np.nanmean(m)
        o_mean = np.nanmean(o)

        # Compute anomalies by removing the mean
        m_anom = m - m_mean
        o_anom = o - o_mean

        # Covariance is mean of product of anomalies
        cov = np.nanmean(m_anom * o_anom)

        # Standard deviations from variance of anomalies
        std_m = np.sqrt(np.nanmean(m_anom ** 2))
        std_o = np.sqrt(np.nanmean(o_anom ** 2))
    else:
        # For xarray, compute along the specified dimension
        m_mean = m.mean(dim=time_dim)
        o_mean = o.mean(dim=time_dim)

        m_anom = m - m_mean
        o_anom = o - o_mean

        cov = (m_anom * o_anom).mean(dim=time_dim)
        std_m = np.sqrt((m_anom ** 2).mean(dim=time_dim))
        std_o = np.sqrt((o_anom ** 2).mean(dim=time_dim))

    # Pearson correlation: covariance divided by product of std devs
    return cov / (std_m * std_o)
###############################################################################

###############################################################################
def corr_no_nan(
    series1: pd.Series,
    series2: pd.Series
) -> float:
    """
    Compute the Pearson correlation coefficient between two pandas Series, ignoring NaNs.

    Parameters
    ----------
    series1 : pd.Series
        First time series.
    series2 : pd.Series
        Second time series.

    Returns
    -------
    float
        Pearson correlation coefficient computed over overlapping non-NaN values.

    Examples
    --------
    >>> s1 = pd.Series([1, 2, 3, None, 5])
    >>> s2 = pd.Series([2, 2, 3, 4, None])
    >>> corr_no_nan(s1, s2)
    0.9819805060619657
    """
    # Concatenate series column-wise and drop rows with any NaNs to get valid pairs
    combined = pd.concat([series1, series2], axis=1).dropna()

    # Compute correlation between the two columns of the cleaned DataFrame
    return combined.iloc[:, 0].corr(combined.iloc[:, 1])
###############################################################################

###############################################################################
def std_dev(
    da: Union[np.ndarray, pd.Series, xr.DataArray],
    time_dim: str = 'time'
) -> Union[float, xr.DataArray]:
    """
    Compute the standard deviation along the specified time dimension.

    Parameters
    ----------
    da : np.ndarray, pd.Series, or xr.DataArray
        Input data array or series.
    time_dim : str, optional
        Name of the time dimension to compute std over (default is 'time').

    Returns
    -------
    float or xr.DataArray
        Standard deviation along the time dimension.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> std_dev(data)
    1.118033988749895
    """
    if isinstance(da, (pd.Series, np.ndarray)) or time_dim not in getattr(da, 'dims', []):
        # For array-like input or missing time_dim, compute global std ignoring NaNs
        mean = np.nanmean(da)
        return np.sqrt(np.nanmean((da - mean) ** 2))
    
    # For xarray DataArray, compute mean along time_dim
    mean = da.mean(dim=time_dim)

    # Compute sqrt of mean squared deviations along time_dim (std dev)
    return ((da - mean) ** 2).mean(dim=time_dim) ** 0.5
###############################################################################

###############################################################################
def unbiased_rmse(
    m: Union[np.ndarray, pd.Series, xr.DataArray],
    o: Union[np.ndarray, pd.Series, xr.DataArray],
    time_dim: str = 'time'
) -> Union[float, xr.DataArray]:
    """
    Compute the unbiased Root Mean Square Error (centered RMSE) between model and observations.

    Parameters
    ----------
    m : np.ndarray, pd.Series, or xr.DataArray
        Model data.
    o : np.ndarray, pd.Series, or xr.DataArray
        Observation data.
    time_dim : str, optional
        Name of the time dimension to compute RMSE over (default is 'time').

    Returns
    -------
    float or xr.DataArray
        Unbiased RMSE value.

    Examples
    --------
    >>> m = np.array([1.0, 2.0, 3.0])
    >>> o = np.array([1.1, 1.9, 3.1])
    >>> unbiased_rmse(m, o)
    0.10000000000000009
    """
    if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in getattr(m, 'dims', []):
        # For array-like or missing time dimension, compute global unbiased RMSE ignoring NaNs
        m_mean = np.nanmean(m)
        o_mean = np.nanmean(o)
        m_anom = m - m_mean
        o_anom = o - o_mean
        return np.sqrt(np.nanmean((m_anom - o_anom) ** 2))

    # For xarray DataArray, compute means along time_dim
    m_mean = m.mean(dim=time_dim)
    o_mean = o.mean(dim=time_dim)
    m_anom = m - m_mean
    o_anom = o - o_mean

    # Compute unbiased RMSE along the time dimension
    return ((m_anom - o_anom) ** 2).mean(dim=time_dim) ** 0.5
###############################################################################

###############################################################################
def spatial_mean(
    data_array: xr.DataArray,
    mask: Union[xr.DataArray, np.ndarray]
) -> xr.DataArray:
    """
    Compute the spatial mean of a DataArray over latitude and longitude, applying a boolean mask.

    Parameters
    ----------
    data_array : xr.DataArray
        Input data array with dimensions including 'lat' and 'lon'.
    mask : xr.DataArray or np.ndarray
        2D boolean mask with shape matching (lat, lon). True indicates valid points.

    Returns
    -------
    xr.DataArray
        Spatial mean over lat/lon of data within the mask, ignoring NaNs.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> data = xr.DataArray(np.random.rand(3,4,5), dims=('time', 'lat', 'lon'))
    >>> mask = xr.DataArray(np.array([[True, False, True, True, False],
    ...                               [True, True, False, False, True],
    ...                               [False, True, True, False, True],
    ...                               [True, False, True, True, True]]),
    ...                     dims=('lat', 'lon'))
    >>> spatial_mean(data.isel(time=0), mask)
    <xarray.DataArray ()>
    array(0.5267)
    """
    # Apply mask to keep only valid spatial points
    masked_data = data_array.where(mask)

    # Compute mean over spatial dims lat and lon, skipping NaNs caused by masking
    return masked_data.mean(dim=['lat', 'lon'], skipna=True)
###############################################################################

###############################################################################
def compute_lagged_correlations(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 30
) -> pd.Series:
    """
    Compute Pearson correlation coefficients between two pandas Series over a range of time lags.

    Parameters
    ----------
    series1 : pd.Series
        Reference time series.
    series2 : pd.Series
        Time series to be shifted and correlated with series1.
    max_lag : int, optional
        Maximum lag (positive and negative) to compute correlations for (default is 30).

    Returns
    -------
    pd.Series
        Correlation coefficients indexed by lag values (from -max_lag to +max_lag).

    Examples
    --------
    >>> s1 = pd.Series([1, 2, 3, 4, 5])
    >>> s2 = pd.Series([5, 4, 3, 2, 1])
    >>> compute_lagged_correlations(s1, s2, max_lag=2)
    -2   -0.5
    -1   -0.5
     0   -1.0
     1   -0.5
     2   -0.5
    dtype: float64
    """
    lags = range(-max_lag, max_lag + 1)
    results = {}

    for lag in lags:
        # Shift series2 by lag (positive lag shifts forward, negative backward)
        shifted = series2.shift(lag)

        # Compute correlation ignoring NaNs between series1 and shifted series2
        results[lag] = corr_no_nan(series1, shifted)

    return pd.Series(results)
###############################################################################

###############################################################################
def compute_fft(
    data: Union[np.ndarray, np.ndarray, Dict[str, Union[np.ndarray, np.generic]]],
    dt: float = 1.0
) -> Tuple[np.ndarray, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Compute the Fast Fourier Transform (FFT) and corresponding positive frequencies.

    Parameters
    ----------
    data : np.ndarray, 1D array-like, or dict of 1D arrays
        Input data to transform. Can be a single 1D array or a dictionary of 1D arrays.
    dt : float, optional
        Sampling interval between data points (default is 1).

    Returns
    -------
    freqs : np.ndarray
        Array of positive FFT frequencies.
    fft_result : np.ndarray or dict of np.ndarray
        FFT results for each input array; if input was dict, returns dict of FFT arrays.

    Raises
    ------
    ValueError
        If input dict is empty.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> freqs, fft_vals = compute_fft(data, dt=1)
    >>> freqs
    array([0.        , 0.16666667, 0.33333333])
    >>> list(fft_vals)
    [21.+0.j, -3.+5.19615242j, -3.+1.73205081j]

    >>> data_dict = {'a': np.array([1,2,3,4]), 'b': np.array([4,3,2,1])}
    >>> freqs, fft_dict = compute_fft(data_dict)
    >>> freqs
    array([0. , 0.25])
    >>> fft_dict['a']
    array([10.+0.j, -2.+2.j])
    """
    if isinstance(data, dict):
        if len(data) == 0:
            raise ValueError("Input dict is empty; cannot compute FFT.")

        # Assume all arrays have the same length; get length from first array
        N = len(next(iter(data.values())))
        freqs = fftfreq(N, dt)[:N // 2]

        # Compute FFT for each array, keeping only positive frequencies
        fft_result = {
            key: fft(arr)[:N // 2]
            for key, arr in data.items()
        }
        return freqs, fft_result

    else:
        N = len(data)
        freqs = fftfreq(N, dt)[:N // 2]

        # Compute FFT and return positive frequency components
        fft_result = fft(data)[:N // 2]
        return freqs, fft_result