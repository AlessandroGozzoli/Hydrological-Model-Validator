###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Data handling libraries
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, Union, Dict, Optional, List

# Statistical and signal processing libraries
from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import detrend
from scipy.fft import fft, fftfreq
from scipy.stats import linregress

# Logging and tracing
import logging
from eliot import start_action, log_message

# Module utilities
from .time_utils import Timer

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
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

    with Timer('fit_huber function'):
        with start_action(action_type='fit_huber', mod_data_size=mod_data.size, sat_data_size=sat_data.size):
            log_message('Starting Huber regression fitting')
            logging.info('Starting Huber regression fitting')

            # === FITTING HUBER REGRESSOR ===
            huber = HuberRegressor()
            huber.fit(mod_data[:, None], sat_data)
            log_message('Huber regression fitted')
            logging.info('Huber regression fitted')

            # === GENERATING PREDICTED LINE ===
            x_vals = np.linspace(mod_data.min(), mod_data.max(), 100)
            y_vals = huber.predict(x_vals[:, None])
            log_message('Predicted Huber regression line generated')
            logging.info('Predicted Huber regression line generated')

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

    with Timer('fit_lowess function'):
        with start_action(action_type='fit_lowess', mod_data_size=mod_data.size, sat_data_size=sat_data.size, frac=frac):
            log_message('Starting LOWESS smoothing')
            logging.info('Starting LOWESS smoothing')

            # === SORT AND APPLY LOWESS ===
            order = np.argsort(mod_data)
            mod_sorted = mod_data[order]
            sat_sorted = sat_data[order]

            smoothed = lowess(sat_sorted, mod_sorted, frac=frac, return_sorted=True)

            log_message('LOWESS smoothing completed', points=len(smoothed))
            logging.info(f'LOWESS smoothing completed, points={len(smoothed)}')

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
    
    with Timer('round_up_to_nearest function'):
        with start_action(action_type='round_up_to_nearest', x=x, base=base):
            # Divide x by base to scale the problem, apply ceiling to ensure rounding *up*, 
            # then scale back by multiplying with base to get the nearest upper multiple
            result = base * np.ceil(x / base)
            log_message('Rounded value computed', result=result)
            logging.info(f'Rounded value computed: {result}')
            return result
        
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

    with Timer('compute_coverage_stats function'):
        with start_action(action_type='compute_coverage_stats', data_shape=data.shape, Mmask_shape=Mmask.shape):
            # Apply mask to spatial dimensions, reduces each 2D frame to a 1D vector of masked values
            masked_data = data[:, Mmask]  # shape: (time, masked_points)

            total_points = masked_data.shape[1]
            if total_points == 0:
                # If the mask selects zero points, return NaNs for all time steps
                n_time = data.shape[0]
                log_message('Mask has zero selected points, returning NaNs', n_time=n_time)
                logging.info(f'Mask has zero selected points, returning NaNs for {n_time} time steps.')
                return np.full(n_time, np.nan), np.full(n_time, np.nan)

            # ===== COUNTING =====
            # Count valid (non-NaN) values per time step
            valid_counts = np.count_nonzero(~np.isnan(masked_data), axis=1)

            # Remaining points are assumed to be cloud-covered (i.e., NaNs)
            cloud_counts = total_points - valid_counts

            # Convert counts to percentage relative to total masked points
            data_available_percent = 100 * valid_counts / total_points
            cloud_coverage_percent = 100 * cloud_counts / total_points

            log_message('Coverage stats computed', total_points=total_points)
            logging.info(f'Coverage stats computed: total masked points = {total_points}')
            
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

    with Timer('detrend_dim function'):
        with start_action(action_type='detrend_dim', dim=dim, min_valid_points=min_valid_points):
            log_message('Starting detrending process')
            logging.info(f'Starting detrending along dimension "{dim}" with min_valid_points={min_valid_points}')

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

            log_message('Detrending completed')
            logging.info(f'Detrending completed along dimension "{dim}"')

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
    with Timer('mean_bias function'):
        with start_action(action_type='mean_bias', time_dim=time_dim):
            # If input is not an xarray with the specified time dimension,
            # compute global mean bias using np.nanmean to ignore NaNs
            if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in getattr(m, 'dims', []):
                result = np.nanmean(m) - np.nanmean(o)
                log_message('Computed global mean bias', result=result)
                logging.info(f'Global mean bias computed: {result}')
                return result

            # Compute bias along the specified time dimension (works for xarray DataArray)
            result = m.mean(dim=time_dim) - o.mean(dim=time_dim)
            log_message('Computed mean bias over time_dim', time_dim=time_dim)
            logging.info(f'Mean bias computed over dimension "{time_dim}"')
            return result
        
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
    # ===== INPUT VALIDATION =====
    if isinstance(m, (pd.Series, np.ndarray)) and isinstance(o, (pd.Series, np.ndarray)):
        # For array-like inputs, check length directly
        if len(m) != len(o):
            raise ValueError("Inputs 'm' and 'o' must have the same length.")
    elif time_dim in getattr(m, 'dims', []) and time_dim in getattr(o, 'dims', []):
        # For xarray DataArrays, check size along time_dim
        if m.sizes[time_dim] != o.sizes[time_dim]:
            raise ValueError(f"Inputs 'm' and 'o' must have the same length along dimension '{time_dim}'.")

    with Timer('standard_deviation_error function'):
        with start_action(action_type='standard_deviation_error', time_dim=time_dim):
            # ===== COMPUTE STD =====
            if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in getattr(m, 'dims', []):
                # Fallback to global std dev using np.nanstd to ignore NaNs
                sdm = np.nanstd(m)
                sdo = np.nanstd(o)
            else:
                # Compute std dev along specified dimension for xarray
                sdm = m.std(dim=time_dim)
                sdo = o.std(dim=time_dim)

            result = sdm - sdo
            log_message('Computed std deviation error (model - obs)', result=result)
            logging.info(f'Standard deviation error computed: {result}')
            return result
        
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
    with Timer('cross_correlation function'):
        with start_action(action_type='cross_correlation', time_dim=time_dim):
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

            result = cov / (std_m * std_o)
            log_message('Computed Pearson correlation coefficient', result=result)
            logging.info(f'Pearson correlation coefficient computed: {result}')
            return result
        
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
    with Timer('corr_no_nan function'):
        with start_action(action_type='corr_no_nan'):
            # Concatenate series column-wise and drop rows with any NaNs to get valid pairs
            combined = pd.concat([series1, series2], axis=1).dropna()

            # Compute correlation between the two columns of the cleaned DataFrame
            result = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            log_message('Computed Pearson correlation ignoring NaNs', result=result)
            logging.info(f'Correlation computed ignoring NaNs: {result}')
            return result
        
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
    with Timer('std_dev function'):
        with start_action(action_type='std_dev'):
            if isinstance(da, (pd.Series, np.ndarray)) or time_dim not in getattr(da, 'dims', []):
                # For array-like input or missing time_dim, compute global std ignoring NaNs
                mean = np.nanmean(da)
                result = np.sqrt(np.nanmean((da - mean) ** 2))
                log_message('Computed global std dev ignoring NaNs', result=result)
                logging.info(f'Global std dev computed: {result}')
                return result
            
            # For xarray DataArray, compute mean along time_dim
            mean = da.mean(dim=time_dim)

            # Compute sqrt of mean squared deviations along time_dim (std dev)
            result = ((da - mean) ** 2).mean(dim=time_dim) ** 0.5
            log_message('Computed std dev along dimension', dim=time_dim)
            logging.info(f'Std dev computed along dimension {time_dim}')
            return result
        
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
    with Timer('unbiased_rmse function'):
        with start_action(action_type='unbiased_rmse', time_dim=time_dim):
            if isinstance(m, (pd.Series, np.ndarray)) or time_dim not in getattr(m, 'dims', []):
                # For array-like or missing time dimension, compute global unbiased RMSE ignoring NaNs
                m_mean = np.nanmean(m)
                o_mean = np.nanmean(o)
                m_anom = m - m_mean
                o_anom = o - o_mean
                result = np.sqrt(np.nanmean((m_anom - o_anom) ** 2))
                log_message('Computed global unbiased RMSE ignoring NaNs', result=result)
                logging.info(f'Global unbiased RMSE computed: {result}')
                return result

            # For xarray DataArray, compute means along time_dim
            m_mean = m.mean(dim=time_dim)
            o_mean = o.mean(dim=time_dim)
            m_anom = m - m_mean
            o_anom = o - o_mean

            # Compute unbiased RMSE along the time dimension
            result = ((m_anom - o_anom) ** 2).mean(dim=time_dim) ** 0.5
            log_message('Computed unbiased RMSE along dimension', dim=time_dim)
            logging.info(f'Unbiased RMSE computed along dimension {time_dim}')
            return result
        
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
    with Timer('spatial_mean function'):
        with start_action(action_type='spatial_mean', dims=data_array.dims):

            # Apply mask to keep only valid spatial points
            masked_data = data_array.where(mask)
            log_message('Mask applied to data_array', mask_shape=mask.shape, data_shape=data_array.shape)
            logging.info(f"Mask applied: mask shape {mask.shape}, data shape {data_array.shape}")

            # Compute mean over spatial dims lat and lon, skipping NaNs caused by masking
            result = masked_data.mean(dim=['lat', 'lon'], skipna=True)
            log_message('Computed spatial mean with mask applied', result=result.values if hasattr(result, 'values') else result)
            logging.info(f"Computed spatial mean result: {result.values if hasattr(result, 'values') else result}")

            return result
        
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
    with Timer('compute_lagged_correlations function'):
        with start_action(action_type='compute_lagged_correlations'):
            lags = range(-max_lag, max_lag + 1)
            results = {}

            for lag in lags:
                # Shift series2 by lag (positive lag shifts forward, negative backward)
                shifted = series2.shift(lag)

                # Compute correlation ignoring NaNs between series1 and shifted series2
                results[lag] = corr_no_nan(series1, shifted)

            log_message(f'Computed lagged correlations for lags -{max_lag} to {max_lag}')
            logging.info(f'Computed lagged correlations for lags -{max_lag} to {max_lag}')
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
    with Timer('compute_fft function'):
        with start_action(action_type='compute_fft'):
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

                log_message(f'Computed FFT for dict with keys: {list(data.keys())}')
                logging.info(f'Computed FFT for dict with keys: {list(data.keys())}')
                return freqs, fft_result

            else:
                N = len(data)
                freqs = fftfreq(N, dt)[:N // 2]

                # Compute FFT and return positive frequency components
                fft_result = fft(data)[:N // 2]
                log_message('Computed FFT for single array input')
                logging.info('Computed FFT for single array input')
                return freqs, fft_result
            
###############################################################################

###############################################################################

def detrend_poly_dim(data_array: xr.DataArray, dim: str, degree: int = 1) -> xr.DataArray:
    """
    Remove a polynomial trend of specified degree along a given dimension in an xarray.DataArray.

    Parameters
    ----------
    data_array : xarray.DataArray
        Input data array containing the data to detrend.
    dim : str
        Dimension name along which the polynomial trend will be fitted and removed.
    degree : int, optional
        Degree of the polynomial to fit and remove (default is 1, linear detrend).

    Returns
    -------
    xarray.DataArray
        Detrended data array with the polynomial trend removed along the specified dimension.

    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.DataArray(np.random.rand(10, 5), dims=['time', 'space'])
    >>> detrended = detrend_dim(data, dim='time', degree=1)
    """
    with Timer('detrend_dim'):
        with start_action(action_type='detrend_dim', dim=dim, degree=degree):
            # Fit polynomial coefficients along the given dimension (e.g., time)
            polyfit_params = data_array.polyfit(dim=dim, deg=degree)
            log_message('polyfit_params computed', params=polyfit_params.polyfit_coefficients.values.tolist())
            logging.info(f'detrend_dim: polyfit_params computed with coefficients {polyfit_params.polyfit_coefficients.values.tolist()}')

            # Evaluate the polynomial trend at each coordinate along the dimension
            trend_fit = xr.polyval(data_array[dim], polyfit_params.polyfit_coefficients)
            log_message('trend_fit calculated')
            logging.info('detrend_dim: trend_fit calculated')

            # Subtract the polynomial trend from the original data to get detrended data
            detrended = data_array - trend_fit
            log_message('detrended data computed')
            logging.info('detrend_dim: detrended data computed')

            return detrended
        
###############################################################################

###############################################################################

def detrend_linear(data: Union[np.ndarray, List[float], pd.Series]) -> np.ndarray:
    """
    Remove a linear trend from 1D numeric data using least squares linear regression.

    Parameters
    ----------
    data : np.ndarray or list or pd.Series
        1D numeric input data from which the linear trend will be removed.

    Returns
    -------
    np.ndarray
        Detrended 1D array with the linear trend removed.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(10) + np.random.rand(10)
    >>> detrended = detrend(data)
    """
    with Timer('detrend'):
        with start_action(action_type='detrend', length=len(data)):
            # Generate time indices as independent variable for regression
            time_indices = np.arange(len(data))

            # Perform linear regression to find slope and intercept
            slope, intercept, _, _, _ = linregress(time_indices, data)
            log_message('linear regression result', slope=slope, intercept=intercept)
            logging.info(f'detrend_linear: linear regression slope={slope}, intercept={intercept}')

            # Calculate linear trend using slope and intercept
            linear_trend = slope * time_indices + intercept

            # Subtract the linear trend from the original data
            detrended_data = np.asarray(data) - linear_trend
            log_message('detrended data computed', detrended_mean=float(np.mean(detrended_data)))
            logging.info(f'detrend_linear: detrended data computed, mean={np.mean(detrended_data)}')

            return detrended_data
        
###############################################################################

###############################################################################

def monthly_anomaly(data_array: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate monthly mean anomalies without detrending by removing the monthly climatology.

    Parameters
    ----------
    data_array : xarray.DataArray
        Time series data with a 'time' coordinate, must be monthly or higher frequency.

    Returns
    -------
    Tuple[xarray.DataArray, xarray.DataArray]
        Tuple containing:
        - anomalies: Monthly anomalies computed by subtracting monthly means.
        - monthly_climatology: Mean values for each month.

    Examples
    --------
    >>> anomalies, climatology = monthly_anomaly(data_array)
    """
    with Timer('monthly_anomaly'):
        with start_action(action_type='monthly_anomaly'):
            # Compute monthly climatology (mean for each month across all years)
            monthly_climatology = data_array.groupby('time.month').mean('time')
            log_message('monthly climatology computed', months=int(monthly_climatology.month.size))
            logging.info(f'monthly_anomaly: monthly climatology computed for {int(monthly_climatology.month.size)} months')

            # Compute anomalies by subtracting monthly climatology from data
            anomalies = data_array.groupby('time.month') - monthly_climatology
            log_message('monthly anomalies computed')
            logging.info('monthly_anomaly: monthly anomalies computed')

            return anomalies, monthly_climatology
        
###############################################################################

###############################################################################

def yearly_anomaly(data_array: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate yearly mean anomalies without detrending by removing the yearly climatology.

    Parameters
    ----------
    data_array : xarray.DataArray
        Time series data with a 'time' coordinate, typically annual or higher frequency.

    Returns
    -------
    Tuple[xarray.DataArray, xarray.DataArray]
        Tuple containing:
        - anomalies: Yearly anomalies computed by subtracting yearly means.
        - yearly_climatology: Mean values for each year.

    Examples
    --------
    >>> anomalies, climatology = yearly_anomaly(data_array)
    """
    with Timer('yearly_anomaly'):
        with start_action(action_type='yearly_anomaly'):
            # Compute yearly climatology (mean for each year across all months/days)
            yearly_climatology = data_array.groupby('time.year').mean('time')
            log_message('yearly climatology computed', years=int(yearly_climatology.year.size))
            logging.info(f'yearly_anomaly: yearly climatology computed for {int(yearly_climatology.year.size)} years')

            # Compute anomalies by subtracting yearly climatology from data
            anomalies = data_array.groupby('time.year') - yearly_climatology
            log_message('yearly anomalies computed')
            logging.info('yearly_anomaly: yearly anomalies computed')

            return anomalies, yearly_climatology
        
###############################################################################

###############################################################################

def detrended_monthly_anomaly(data_array: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate detrended monthly anomalies by removing the linear trend first, then removing the monthly climatology.

    Parameters
    ----------
    data_array : xarray.DataArray
        Time series data with a 'time' coordinate, typically monthly.

    Returns
    -------
    Tuple[xarray.DataArray, xarray.DataArray]
        Tuple containing:
        - anomalies: Detrended monthly anomalies.
        - monthly_climatology: Monthly climatology of the detrended data.

    Examples
    --------
    >>> anomalies, climatology = detrended_monthly_anomaly(data_array)
    """
    with Timer('detrended_monthly_anomaly'):
        with start_action(action_type='detrended_monthly_anomaly'):
            # Remove linear trend along the time dimension
            detrended_data = detrend_poly_dim(data_array, 'time', degree=1)
            log_message('data detrended')
            logging.info('detrended_monthly_anomaly: data detrended')

            # Calculate monthly climatology from detrended data
            monthly_climatology = detrended_data.groupby('time.month').mean('time')
            log_message('monthly climatology computed')
            logging.info('detrended_monthly_anomaly: monthly climatology computed')

            # Calculate monthly anomalies from detrended data
            anomalies = detrended_data.groupby('time.month') - monthly_climatology
            log_message('detrended monthly anomalies computed')
            logging.info('detrended_monthly_anomaly: detrended monthly anomalies computed')

            return anomalies, monthly_climatology
        
###############################################################################

###############################################################################

def np_covariance(field_time_xy: np.ndarray, index_time: np.ndarray) -> np.ndarray:
    """
    Calculate covariance between a 3D spatial-temporal field and a 1D index time series.

    Parameters
    ----------
    field_time_xy : np.ndarray
        3D array with shape (time, y, x), spatial field over time.
    index_time : np.ndarray
        1D array with length equal to time dimension in field_time_xy.

    Returns
    -------
    np.ndarray
        2D covariance map with shape (y, x).

    Examples
    --------
    >>> cov = np_covariance(field, index)
    """
    with Timer('np_covariance'):
        with start_action(action_type='np_covariance'):
            # Calculate anomalies by removing mean along time for spatial field
            field_anom = field_time_xy - np.mean(field_time_xy, axis=0)

            # Calculate anomalies by removing mean for index time series
            index_anom = index_time - np.mean(index_time)

            # Compute covariance map as mean of pointwise product of anomalies over time
            covariance = np.einsum('tij,t->ij', field_anom, index_anom) / len(index_time)

            log_message('covariance computed', shape=covariance.shape)
            logging.info(f'np_covariance: covariance computed with shape {covariance.shape}')
            return covariance
        
###############################################################################

###############################################################################

def np_correlation(field_time_xy: np.ndarray, index_time: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson correlation coefficient map between a 3D spatial-temporal field and a 1D index time series.

    Parameters
    ----------
    field_time_xy : np.ndarray
        3D array (time, y, x) representing spatial field over time.
    index_time : np.ndarray
        1D array representing time series index.

    Returns
    -------
    np.ndarray
        2D correlation map with shape (y, x).

    Examples
    --------
    >>> corr = np_correlation(field, index)
    """
    with Timer('np_correlation'):
        with start_action(action_type='np_correlation'):
            # Calculate covariance map
            covariance = np_covariance(field_time_xy, index_time)

            # Calculate correlation by normalizing covariance with std deviations
            correlation = covariance / (np.std(field_time_xy, axis=0) * np.std(index_time))

            log_message('correlation computed', shape=correlation.shape)
            logging.info(f'np_correlation: correlation computed with shape {correlation.shape}')
            return correlation
        
###############################################################################

###############################################################################

def np_regression(field_time_xy: np.ndarray, index_time: np.ndarray, std_units: str = 'yes') -> np.ndarray:
    """
    Compute regression coefficients between a 3D spatial-temporal field and a 1D index time series.

    Parameters
    ----------
    field_time_xy : np.ndarray
        3D spatial-temporal field array (time, y, x).
    index_time : np.ndarray
        1D index time series.
    std_units : str, optional
        If 'yes', normalize regression by standard deviation of index_time.

    Returns
    -------
    np.ndarray
        2D regression coefficients map (y, x).

    Examples
    --------
    >>> regression = np_regression(field, index)
    """
    with Timer('np_regression'):
        with start_action(action_type='np_regression', std_units=std_units):
            # Calculate covariance between field and index time series
            covariance = np_covariance(field_time_xy, index_time)

            # Calculate variance of index time series
            variance_index = np.var(index_time)

            # Calculate regression coefficients by dividing covariance by variance
            regression = covariance / variance_index

            # Optionally normalize regression by std of index time series
            if std_units == 'yes':
                regression /= np.std(index_time)

            log_message('regression computed', shape=regression.shape)
            logging.info(f'np_regression: regression computed with shape {regression.shape} and std_units={std_units}')
            return regression
        
###############################################################################

###############################################################################

def extract_multidecadal_peak(
    freqs: np.ndarray,
    amps: np.ndarray,
    frequency_threshold: float = 1/10
) -> Optional[Dict[str, float]]:
    """
    Extract amplitude and frequency/period of the largest peak within a frequency threshold region.

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequencies from power spectrum.
    amps : np.ndarray
        Corresponding amplitude array from power spectrum.
    frequency_threshold : float, optional
        Frequency threshold defining multidecadal region (default 1/10 cycles per year).

    Returns
    -------
    dict or None
        Dictionary with keys 'Peak Amplitude', 'Peak Frequency (cycles/year)', 'Peak Period (years)'
        for the largest amplitude peak in the multidecadal region,
        or None if no frequencies below threshold.

    Examples
    --------
    >>> peak = extract_multidecadal_peak(freqs, amps)
    """
    with Timer('extract_multidecadal_peak'):
        with start_action(action_type='extract_multidecadal_peak', freq_threshold=frequency_threshold):
            # Identify frequencies less than the threshold (multidecadal region)
            multidecadal_indices = np.where(freqs < frequency_threshold)[0]

            if len(multidecadal_indices) == 0:
                log_message('no multidecadal frequencies found')
                logging.info('no multidecadal frequencies found')
                return None

            # Extract the frequencies and amplitudes in the multidecadal region
            freqs_sub = freqs[multidecadal_indices]
            amps_sub = amps[multidecadal_indices]

            # Find index of the largest amplitude peak
            peak_idx = np.argmax(amps_sub)

            # Extract peak frequency and amplitude
            peak_frequency = freqs_sub[peak_idx]
            peak_amplitude = amps_sub[peak_idx]

            # Convert frequency to period in years (avoid division by zero)
            peak_period = 1 / peak_frequency if peak_frequency > 0 else np.inf

            peak_info = {
                "Peak Amplitude": peak_amplitude,
                "Peak Frequency (cycles/year)": peak_frequency,
                "Peak Period (years)": peak_period
            }
            log_message('peak extracted', peak=peak_info)
            logging.info(f"peak extracted: {peak_info}")
            return peak_info
        
###############################################################################

###############################################################################

def extract_multidecadal_peaks_from_spectra(
        power_spectra: Dict[str, Tuple[np.ndarray, ...]],
    frequency_threshold: float = 1/10
) -> pd.DataFrame:
    """
    Extract multidecadal peak amplitude and frequency information from multiple regions' power spectra.

    Parameters
    ----------
    power_spectra : dict
        Dictionary mapping region names to tuples containing frequency array, amplitude array, and others.
    frequency_threshold : float, optional
        Threshold frequency below which frequencies are considered multidecadal (default 1/10 cycles/year).

    Returns
    -------
    pd.DataFrame
        DataFrame with peak amplitude, frequency, and period for each region.

    Examples
    --------
    >>> df = extract_multidecadal_peaks_from_spectra(power_spectra)
    """
    with Timer('extract_multidecadal_peaks_from_spectra'):
        with start_action(action_type='extract_multidecadal_peaks_from_spectra', freq_threshold=frequency_threshold):
            results = {}
            for region, (freqs, amps, *_) in power_spectra.items():
                # Extract peak info for each region using helper function
                peak_info = extract_multidecadal_peak(freqs, amps, frequency_threshold)
                if peak_info:
                    results[region] = peak_info
                    log_message('peak info extracted for region', region=region, peak=peak_info)
                    logging.info(f"peak info extracted for region: {region}, peak: {peak_info}")
            df = pd.DataFrame.from_dict(results, orient='index')
            log_message('all peaks extracted', num_regions=len(results))
            logging.info(f"all peaks extracted, num_regions: {len(results)}")
            return df
        
###############################################################################

###############################################################################

def identify_extreme_events(
    time_series: Union[np.ndarray, pd.Series, xr.DataArray],
    threshold_multiplier: float = 1.5,
    step: Optional[int] = None,
    comparison_index: Optional[int] = None
) -> Dict[str, Union[np.ndarray, List[int]]]:
    """
    Identify extreme positive and negative events in a time series using a threshold based on standard deviation.
    Optionally, sample these extreme events at fixed intervals starting at a given index.

    Parameters
    ----------
    time_series : np.ndarray or pd.Series or xarray.DataArray
        1D time series data.
    threshold_multiplier : float, optional
        Multiplier for standard deviation to define extreme events (default 1.5).
    step : int, optional
        Step interval to sample the time series for extreme events (e.g., 12 for monthly December events).
    comparison_index : int, optional
        Starting index for sampling when step is provided (e.g., 12 to start from December).

    Returns
    -------
    dict
        Dictionary containing:
        - 'positive_events_mask': Boolean mask array for extreme positive events.
        - 'negative_events_mask': Boolean mask array for extreme negative events.
        - 'sampled_positive_indices': Indices of sampled positive extreme events (if step provided).
        - 'sampled_negative_indices': Indices of sampled negative extreme events (if step provided).

    Examples
    --------
    >>> events = identify_extreme_events(n34_array, threshold_multiplier=1.5, step=12, comparison_index=12)
    """
    with Timer('identify_extreme_events'):
        with start_action(action_type='identify_extreme_events', threshold_multiplier=threshold_multiplier):
            # Extract raw numeric values if input is xarray or pandas object
            values = time_series.values if hasattr(time_series, 'values') else np.asarray(time_series)

            # Calculate standard deviation of the time series
            std_dev = np.std(values)
            log_message('standard deviation computed', std_dev=float(std_dev))
            logging.info(f'standard deviation computed: {std_dev}')

            # Identify where values exceed positive threshold (Niño-like events)
            positive_events_mask = values > std_dev * threshold_multiplier
            # Identify where values are below negative threshold (Niña-like events)
            negative_events_mask = values < -std_dev * threshold_multiplier
            log_message('extreme event masks computed')
            logging.info('extreme event masks computed')

            if step is not None:
                if comparison_index is None:
                    error_msg = "If 'step' is provided, 'comparison_index' must be specified."
                    log_message('error', message=error_msg)
                    logging.info(f'error: {error_msg}')
                    raise ValueError(error_msg)

                # Sample the time series starting at comparison_index every 'step' points
                sampled_values = values[comparison_index::step]

                # Find indices within the sampled data where extreme positive events occur
                sampled_positive_indices = np.where(sampled_values >= std_dev * threshold_multiplier)[0]
                # Find indices within the sampled data where extreme negative events occur
                sampled_negative_indices = np.where(sampled_values <= -std_dev * threshold_multiplier)[0]

                log_message('sampled extreme events identified',
                            sampled_positive_count=len(sampled_positive_indices),
                            sampled_negative_count=len(sampled_negative_indices))
                logging.info(f'sampled extreme events identified: sampled_positive_count={len(sampled_positive_indices)}, sampled_negative_count={len(sampled_negative_indices)}')

                return {
                    'positive_events_mask': positive_events_mask,
                    'negative_events_mask': negative_events_mask,
                    'sampled_positive_indices': sampled_positive_indices,
                    'sampled_negative_indices': sampled_negative_indices
                }
            else:
                # If no sampling, just return boolean masks for extremes
                return {
                    'positive_events_mask': positive_events_mask,
                    'negative_events_mask': negative_events_mask
                }