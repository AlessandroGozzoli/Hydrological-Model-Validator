###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Data handling libraries
import numpy as np
import xarray as xr
import pandas as pd
from typing import Literal, Tuple, Dict, Union, List, Sequence

# Logging and tracing
import logging
from eliot import start_action, log_message

# Module utilities and stats functions
from .time_utils import Timer
from .stats_math_utils import (
    mean_bias,
    standard_deviation_error,
    std_dev,
    cross_correlation,
    unbiased_rmse,
)

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def r_squared(obs: Union[np.ndarray, Sequence[float]], pred: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Calculate the coefficient of determination (r²) between observed and predicted data.

    Parameters
    ----------
    obs : np.ndarray
        Array of observed values.
    pred : np.ndarray
        Array of predicted values.

    Returns
    -------
    float
        The coefficient of determination (r²), which quantifies how well predictions
        approximate the observed values. Returns np.nan if fewer than 2 valid (non-NaN) pairs.

    Notes
    -----
    - NaN values in either input array are ignored.
    - r² is the square of the Pearson correlation coefficient between obs and pred.
    - r² ranges from 0 (no correlation) to 1 (perfect linear correlation).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([3.0, 4.5, 5.2, np.nan, 6.1])
    >>> pred = np.array([2.8, 4.7, 5.0, 5.9, np.nan])
    >>> r_squared(obs, pred)
    0.9911...  # Very high correlation with missing data ignored

    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> pred = np.array([1.1, 1.9, 3.1])
    >>> r_squared(obs, pred)
    0.9983...

    >>> obs = np.array([np.nan, np.nan])
    >>> pred = np.array([1.0, 2.0])
    >>> r_squared(obs, pred)
    nan
    """
    # ===== INPUT VALIDATION =====
    if obs is None or pred is None:
        raise ValueError("❌ Input arrays 'obs' and 'pred' must not be None. ❌")
        
    obs = np.asarray(obs)
    pred = np.asarray(pred)

    if obs.shape != pred.shape:
        raise ValueError(f"❌ Shape mismatch: 'obs' has shape {obs.shape}, but 'pred' has shape {pred.shape}. They must match.")
    
    if obs.ndim != 1:
        raise ValueError("❌ Inputs must be one-dimensional arrays.")

    # ===== COMPUTATIONS AND LOGGING =====
    with Timer("r_squared function"):
        with start_action(action_type="r_squared") as action:
            log_message("Entered r_squared", obs_shape=np.shape(obs), pred_shape=np.shape(pred))
            logging.info("[Start] r_squared calculation")

            # Create a mask to ignore any NaN values in either array
            mask = ~np.isnan(obs) & ~np.isnan(pred)

            if np.sum(mask) < 2:
                logging.info("[Info] Not enough valid data points, returning np.nan")
                log_message("Insufficient valid data for r_squared", valid_points=int(np.sum(mask)))
                return np.nan

            # ===== COMPUTATIONS =====
            corr = np.corrcoef(obs[mask], pred[mask])[0, 1]

            r2 = corr ** 2

            log_message("Computed r_squared", r_squared=r2)
            logging.info(f"[Done] r_squared computed: {r2}")

            return r2
        
###############################################################################

###############################################################################

def monthly_r_squared(data_dict: Dict[str, Dict[int, List[Union[np.ndarray, List[float]]]]]) -> List[float]:
    """
    Compute monthly R² values between model and satellite datasets over multiple years.

    Parameters
    ----------
    data_dict : dict
        Dictionary with structure:
        {
            'BASSTmod': {year1: [12 arrays], year2: [...], ...},
            'BASSTsat': {year1: [12 arrays], year2: [...], ...}
        }
        Keys should contain 'mod' for model and 'sat' for satellite data. Each value is
        a dictionary mapping years to lists of 12 monthly 2D arrays.

    Returns
    -------
    list of float
        List of 12 R² values (one for each month from January to December).

    Notes
    -----
    - This function concatenates data across all years for each month and calculates
      the R² between the model and satellite data for that month.
    - NaN values are excluded from the computation.
    - If no valid data exists for a month, the R² for that month is set to np.nan.

    Examples
    --------
    >>> mod_data = {
    ...     2000: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
    ...     2001: [np.array([[2, 3], [4, 5]]) for _ in range(12)]
    ... }
    >>> sat_data = {
    ...     2000: [np.array([[1, 2.1], [3.1, 4]]) for _ in range(12)],
    ...     2001: [np.array([[2.2, 3], [4.1, 5.1]]) for _ in range(12)]
    ... }
    >>> data_dict = {'BASSTmod': mod_data, 'BASSTsat': sat_data}
    >>> monthly_r_squared(data_dict)
    [0.999..., 0.999..., ..., 0.999...]  # 12 values
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(data_dict, dict):
        raise TypeError("❌ Input must be a dictionary.")

    # Find keys for model and satellite data by searching for 'mod' and 'sat' substrings
    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in data_dict if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in data_dict if any(kw in k.lower() for kw in sat_keywords)), None)

    # Raise error if keys are not found to avoid silent failures
    if model_key is None or sat_key is None:
        raise KeyError("❌ Model or satellite key not found in the dictionary. Expected keys containing 'mod' and 'sat'.")

    mod_monthly = data_dict[model_key]
    sat_monthly = data_dict[sat_key]

    # Ensure model and satellite data are dictionaries
    if not isinstance(mod_monthly, dict) or not isinstance(sat_monthly, dict):
        raise TypeError("❌ Model and satellite values must be dictionaries mapping years to lists of arrays.")

    # Ensure both datasets have the same years
    mod_years = set(mod_monthly.keys())
    sat_years = set(sat_monthly.keys())
    if mod_years != sat_years:
        raise ValueError(f"❌ Mismatched years between model and satellite data: {mod_years ^ sat_years}")

    years = sorted(mod_years)  # List all available years from model data

    # Determine number of months dynamically from the first year in model data
    first_year = years[0]
    n_months = len(mod_monthly[first_year])

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_r_squared function"):
        with start_action(action_type="monthly_r_squared") as action:
            log_message("Entered monthly_r_squared", years=years, n_months=n_months)
            logging.info("[Start] monthly_r_squared computation")

            r2_monthly = []  # Initialize list to store R² for each month

            # ===== LOOPING THE COMPUTATIONS =====
            for month in range(n_months):
                # Extract monthly data arrays from all years and flatten to 1D for comparison
                mod_arrays = []
                sat_arrays = []

                for year in years:
                    mod_list = mod_monthly[year]
                    sat_list = sat_monthly[year]

                    mod_arrays.append(np.asarray(mod_list[month]).ravel())
                    sat_arrays.append(np.asarray(sat_list[month]).ravel())

                # Concatenate monthly arrays from all years into single long arrays
                mod_concat = np.concatenate(mod_arrays)
                sat_concat = np.concatenate(sat_arrays)

                # Create mask to ignore NaNs in either dataset
                valid_mask = ~np.isnan(mod_concat) & ~np.isnan(sat_concat)

                if np.any(valid_mask):
                    # Compute R² using valid data points
                    r2 = r_squared(mod_concat[valid_mask], sat_concat[valid_mask])
                else:
                    # If no valid data, assign NaN to indicate missing correlation
                    r2 = np.nan

                log_message("Computed monthly r_squared", month=month + 1, r_squared=r2)
                logging.info(f"Month {month + 1}: R² = {r2}")

                r2_monthly.append(r2)  # Append monthly R² to list

            logging.info("[Done] monthly_r_squared computation completed")
            log_message("Completed monthly_r_squared", total_months=len(r2_monthly))

            return r2_monthly
        
###############################################################################

###############################################################################

def weighted_r_squared(obs: Union[np.ndarray, list], pred: Union[np.ndarray, list]) -> float:
    """
    Compute weighted coefficient of determination (weighted R²) between observed and predicted data.

    The weighting accounts for the slope of the regression line between predicted and observed values,
    emphasizing cases where the slope is close to 1 (ideal 1:1 relation).

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        Weighted R² value or np.nan if insufficient data.

    Notes
    -----
    - The function first computes the standard R² between obs and pred.
    - It then fits a linear regression line pred = slope * obs + intercept.
    - The absolute value of the slope is used to weight the R²:
      - If slope is close to 1, weight ≈ 1 (no change).
      - If slope deviates far from 1, weight is reduced, clipped between 0.1 and 1.
    - This penalizes cases where prediction trends differ substantially from observations,
      even if correlation is high.
    - A small floor of 0.1 prevents the weighted R² from becoming zero or negligible
      when the slope is near zero.

    Examples
    --------
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    >>> weighted_r_squared(obs, pred)
    0.98  # example output (actual value depends on data)
    """
    # ===== INPUT VALIDATION =====
    if obs is None or pred is None:
        raise ValueError("❌ Input arrays 'obs' and 'pred' must not be None. ❌")

    obs = np.asarray(obs)
    pred = np.asarray(pred)

    if obs.shape != pred.shape:
        raise ValueError(f"❌ Input arrays must have the same shape, got {obs.shape} and {pred.shape}. ❌")

    # Create mask to ignore NaNs in either obs or pred
    mask = ~np.isnan(obs) & ~np.isnan(pred)

    # Return NaN if fewer than 2 valid data points (insufficient for regression)
    if np.sum(mask) < 2:
        return np.nan

    x = obs[mask]
    y = pred[mask]

    # ===== COMPUTATION AND LOGGING =====
    with Timer("weighted_r_squared function"):
        with start_action(action_type="weighted_r_squared") as action:
            log_message("Entered weighted_r_squared", valid_points=len(x))
            logging.info("[Start] weighted_r_squared computation")

            # Compute standard R² between observed and predicted for valid data
            r2 = r_squared(x, y)

            # Fit a linear regression line: pred = slope * obs + intercept
            slope, intercept = np.polyfit(x, y, 1)

            # Calculate weight based on how close slope is to 1:
            # If slope > 1, invert it to keep weight <= 1,
            # otherwise use slope directly (absolute value)
            slope_abs = abs(slope)
            weight = slope_abs if slope_abs <= 1 else 1 / slope_abs
            # Set minimum weight to 0.1 to avoid zero or near-zero weights
            weight = max(weight, 0.1)

            weighted_r2 = weight * r2

            log_message(
                "Computed weighted_r_squared",
                slope=slope,
                weight=weight,
                r_squared=r2,
                weighted_r_squared=weighted_r2,
            )
            logging.info(
                f"Weighted R²: {weighted_r2:.4f} (Slope: {slope:.4f}, Weight: {weight:.4f})"
            )

            logging.info("[Done] weighted_r_squared computation completed")
            log_message("Completed weighted_r_squared")

            return weighted_r2
        
###############################################################################

###############################################################################   
 
def monthly_weighted_r_squared(dictionary: Dict[str, Dict[int, List[Union[np.ndarray, List[float]]]]]) -> List[float]:
    """
    Compute weighted coefficient of determination (weighted R²) for each calendar month across multiple years,
    using paired model and satellite datasets.

    The weighting adjusts the R² based on the slope of the linear regression between predicted (model) and
    observed (satellite) values, emphasizing months where the relationship is closer to a 1:1 correspondence.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys with substrings 'mod' and 'sat' representing model and satellite data, respectively.
        Each key maps to a dictionary of years, where each year corresponds to a list or array of 12 monthly arrays
        of data points.

    Returns
    -------
    list of float
        List of 12 weighted R² values, each representing one calendar month (January to December).

    Raises
    ------
    KeyError
        Raised if no keys containing 'mod' or 'sat' are found in the input dictionary.

    Notes
    -----
    - For each month, data from all years are concatenated to form a single paired dataset of model and satellite values.
    - NaN values in either dataset are excluded from the calculations.
    - The weighted R² combines the classical coefficient of determination with a weighting factor derived from
      the slope of the regression line between satellite (observed) and model (predicted) data.
    - This method penalizes cases where the prediction slope deviates significantly from unity, even if correlation is high.

    Examples
    --------
    >>> data = {
    ...     'model': {
    ...         2000: [np.array([1, 2]), np.array([3, 4])] * 6,
    ...         2001: [np.array([2, 3]), np.array([4, 5])] * 6,
    ...     },
    ...     'satellite': {
    ...         2000: [np.array([1.1, 2.1]), np.array([2.9, 4.1])] * 6,
    ...         2001: [np.array([2.2, 2.9]), np.array([3.8, 5.2])] * 6,
    ...     }
    ... }
    >>> monthly_weighted_r_squared(data)
    [0.95, 0.92, ..., 0.90]
    """
    from .Efficiency_metrics import weighted_r_squared

    # ===== INPUT VALIDATIONS =====
    if not isinstance(dictionary, dict):
        raise ValueError("❌ Input must be a dictionary. ❌")

    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in dictionary if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in dictionary if any(kw in k.lower() for kw in sat_keywords)), None)

    if not model_key or not sat_key:
        raise KeyError("❌ Model or satellite key not found in the dictionary. ❌")

    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    if not isinstance(mod_monthly, dict) or not isinstance(sat_monthly, dict):
        raise ValueError("❌ Model and satellite data must be dictionaries keyed by year. ❌")

    years = list(mod_monthly.keys())

    if not years:
        raise ValueError("❌ No year data found in the model dataset. ❌")

    first_year = years[0]
    n_months = len(mod_monthly[first_year])

    wr2_monthly = []

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_weighted_r_squared function"):
        with start_action(action_type="monthly_weighted_r_squared") as action:
            log_message("Entered monthly_weighted_r_squared", years=years, n_months=n_months)
            logging.info("[Start] monthly_weighted_r_squared computation")

            # ===== LOOPING THE COMPUTATIONS =====
            for month in range(n_months):
                try:
                    mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
                    sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])
                except IndexError:
                    raise ValueError(f"❌ Month index {month} is out of bounds in one of the datasets. ❌")

                valid_mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

                if np.any(valid_mask):
                    wr2 = weighted_r_squared(sat_all[valid_mask], mod_all[valid_mask])
                else:
                    wr2 = np.nan

                wr2_monthly.append(wr2)

                log_message("Computed monthly weighted_r_squared", month=month + 1, weighted_r_squared=wr2)
                logging.info(f"Month {month + 1}: Weighted R² = {wr2}")

            logging.info("[Done] monthly_weighted_r_squared computation completed")
            log_message("Completed monthly_weighted_r_squared", total_months=len(wr2_monthly))

            return wr2_monthly
        
###############################################################################

###############################################################################
 
def nse(obs: Union[np.ndarray, Sequence[float]], pred: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Compute Nash–Sutcliffe Efficiency (NSE) between observed and predicted data.

    NSE is a normalized statistic that determines the relative magnitude of the residual variance
    ("noise") compared to the variance of the observed data ("signal"). It is widely used to assess
    the predictive skill of hydrological models.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        NSE value, ranging from -∞ to 1:
        - NSE = 1 indicates a perfect match between observed and predicted data.
        - NSE = 0 indicates that the model predictions are as accurate as the mean of the observations.
        - NSE < 0 indicates that the observed mean is a better predictor than the model.
        Returns np.nan if insufficient valid data or if variance of observed data is zero.

    Notes
    -----
    - The function ignores pairs where either observation or prediction is NaN.
    - At least two valid data points are required to compute NSE.
    - NSE is sensitive to extreme values and assumes that observations are error-free.

    Examples
    --------
    >>> obs = np.array([3, -0.5, 2, 7])
    >>> pred = np.array([2.5, 0.0, 2, 8])
    >>> nse(obs, pred)
    0.8571428571428571
    """
    obs = np.asarray(obs)  # Convert to numpy array for vectorized operations
    pred = np.asarray(pred)

    # ==== INPUT VALIDATION =====
    if obs.shape != pred.shape:
        raise ValueError("❌ Input arrays must have the same shape. ❌")

    # Create a boolean mask to filter out any pairs where obs or pred is NaN,
    # since these invalid pairs would distort the NSE calculation
    mask = ~np.isnan(obs) & ~np.isnan(pred)

    # Require at least two valid pairs to compute a meaningful NSE;
    # otherwise return NaN since statistic can't be computed reliably
    if np.sum(mask) < 2:
        return np.nan

    obs_masked = obs[mask]
    pred_masked = pred[mask]

    # ===== COMPUTATION AND LOGGING =====
    with Timer("nse function"):
        with start_action(action_type="nse") as action:
            log_message("Entered nse function", valid_data_points=len(obs_masked))
            logging.info("[Start] NSE computation")

            # Calculate the sum of squared residuals (difference between observed and predicted),
            # representing the "noise" or error variance of the model predictions
            numerator = np.sum((obs_masked - pred_masked) ** 2)

            # Calculate the variance of the observed data relative to its mean,
            # representing the "signal" or natural variance in observations
            denominator = np.sum((obs_masked - np.mean(obs_masked)) ** 2)

            if denominator == 0:
                logging.warning("Observed variance is zero; NSE is undefined (NaN returned).")
                log_message("NSE undefined due to zero variance in observations")
                return np.nan

            nse_value = 1 - numerator / denominator

            log_message("Computed NSE", nse=nse_value)
            logging.info(f"[Done] NSE computation: {nse_value}")

            return nse_value
        
###############################################################################

############################################################################### 

def monthly_nse(dictionary: Dict[str, Dict[int, List[Union[np.ndarray, List[float]]]]]) -> List[float]:
    """
    Compute monthly Nash–Sutcliffe Efficiency (NSE) between model and satellite datasets.

    This function aggregates paired model and satellite data over multiple years,
    calculates the NSE for each calendar month, and returns a list of monthly NSE values.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys with 'mod' and 'sat' for model and satellite data.
        Each key maps to a dictionary of years, where each year is a list or array of 12 monthly data arrays.

    Returns
    -------
    list of float
        NSE values for each month (length 12). Each value represents the NSE aggregated over all years for that month.

    Raises
    ------
    KeyError
        If no model or satellite keys are found in the input dictionary.

    Notes
    -----
    - The function concatenates monthly data across all years before computing NSE.
    - Pairs with NaN values in either dataset are excluded.
    - Returns np.nan for months where valid paired data is insufficient.

    Examples
    --------
    >>> data = {
    ...     'model_data': {
    ...         2020: [np.array([1, 2]), np.array([2, 3]), ...],  # 12 monthly arrays
    ...         2021: [np.array([1.1, 1.9]), np.array([2.1, 3.1]), ...]
    ...     },
    ...     'satellite_data': {
    ...         2020: [np.array([1, 2]), np.array([2, 2.9]), ...],
    ...         2021: [np.array([1.0, 2.0]), np.array([2.0, 3.0]), ...]
    ...     }
    ... }
    >>> monthly_nse(data)
    [0.95, 0.89, ..., 0.92]  # Example output, one value per month
    """
    from .Efficiency_metrics import nse

    # ===== INPUT VALIDATION =====
    if not isinstance(dictionary, dict):
        raise TypeError("❌ Input must be a dictionary containing model and satellite data keys. ❌")

    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in dictionary if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in dictionary if any(kw in k.lower() for kw in sat_keywords)), None)

    if not model_key or not sat_key:
        raise KeyError("❌ Model or satellite key not found in the dictionary. ❌")

    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    # List all years present in the data (assumed same for model and satellite)
    years = list(mod_monthly.keys())

    if not years:
        raise ValueError("❌ No year data found in the model dataset. ❌")

    nse_monthly = []

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_nse function"):
        with start_action(action_type="monthly_nse") as action:
            log_message("Entered monthly_nse", years=years, n_months=len(mod_monthly[years[0]]))
            logging.info("[Start] monthly_nse computation")

            for month in range(len(mod_monthly[years[0]])):
                try:
                    # Concatenate monthly data from all years into one flat array for model
                    mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
                    sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])
                except IndexError:
                    raise ValueError(f"❌ Month index {month} is out of bounds in one of the datasets. ❌")

                # Create a mask that selects only pairs where both model and satellite data are valid (not NaN)
                valid_mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

                # If any valid pairs exist, compute NSE for that month; else assign NaN
                if np.any(valid_mask):
                    nse_val = nse(sat_all[valid_mask], mod_all[valid_mask])
                else:
                    nse_val = np.nan

                # Store the monthly NSE result
                nse_monthly.append(nse_val)

                log_message("Computed monthly NSE", month=month + 1, nse=nse_val)
                logging.info(f"Month {month + 1}: NSE = {nse_val}")

            logging.info("[Done] monthly_nse computation completed")
            log_message("Completed monthly_nse", total_months=len(nse_monthly))

            return nse_monthly
        
###############################################################################

############################################################################### 

def index_of_agreement(obs: Union[np.ndarray, Sequence[float]], 
                       pred: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Calculate the Index of Agreement (d) between observed and predicted values.

    The Index of Agreement is a standardized measure of the degree of model prediction error,
    which varies between 0 (no agreement) and 1 (perfect agreement).

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        Index of Agreement (d), or np.nan if insufficient valid data or denominator is zero.

    Notes
    -----
    - Excludes any pairs where either observed or predicted values are NaN.
    - Requires at least two valid data points to compute.
    - Denominator involves sums of absolute deviations from the observed mean.
    - The metric penalizes both under- and over-prediction differently than simple correlation.

    Examples
    --------
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    >>> index_of_agreement(obs, pred)
    0.97  # example output (actual value depends on data)
    """
    # ===== INPUT VALIDATION =====
    if not (hasattr(obs, "__iter__") and hasattr(pred, "__iter__")):
        raise TypeError("❌ Inputs obs and pred must be array-like sequences. ❌")

    obs = np.asarray(obs)
    pred = np.asarray(pred)

    if obs.shape != pred.shape:
        raise ValueError(f"❌ Input arrays must have the same shape, got {obs.shape} and {pred.shape}. ❌")

    # Create mask to ignore pairs where either value is NaN
    mask = ~np.isnan(obs) & ~np.isnan(pred)

    # Require at least two valid data points for meaningful calculation
    if np.sum(mask) < 2:
        return np.nan

    # Extract valid observed and predicted values
    obs_masked = obs[mask]
    pred_masked = pred[mask]

    # ===== COMPUTATION AND LOGGING =====
    with Timer("index_of_agreement function"):
        with start_action(action_type="index_of_agreement") as action:
            log_message("Entered index_of_agreement function", valid_data_points=len(obs_masked))
            logging.info("[Start] Index of Agreement computation")

            # Numerator: sum of squared differences between observed and predicted values (prediction error)
            numerator = np.sum((obs_masked - pred_masked) ** 2)

            # Denominator: sum of squared sums of absolute deviations from the observed mean
            # This measures the total potential error, considering both over- and under-predictions
            denominator = np.sum((np.abs(pred_masked - np.mean(obs_masked)) + np.abs(obs_masked - np.mean(obs_masked))) ** 2)

            # If denominator is zero (no variation in observations), index is undefined
            if denominator == 0:
                logging.warning("Observed variance is zero; Index of Agreement is undefined (NaN returned).")
                log_message("Index of Agreement undefined due to zero variance in observations")
                return np.nan

            # Calculate Index of Agreement as 1 minus the ratio of error to potential error
            index_value = 1 - numerator / denominator

            log_message("Computed Index of Agreement", index_of_agreement=index_value)
            logging.info(f"[Done] Index of Agreement computation: {index_value}")

            return index_value
        
###############################################################################

############################################################################### 

def monthly_index_of_agreement(
    dictionary: Dict[str, Dict[int, List[Union[np.ndarray, List[float]]]]]
) -> List[float]:
    """
    Compute the monthly Index of Agreement (d) between model and satellite datasets.

    This function aggregates paired model and satellite data across multiple years,
    then calculates the Index of Agreement for each month to evaluate the agreement
    between predicted and observed values.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys with 'mod' and 'sat' indicating model and satellite data.
        Each key maps to a dict where each year contains a list or array of 12 monthly data arrays.

    Returns
    -------
    list of float
        List of 12 Index of Agreement values, one per month.

    Raises
    ------
    KeyError
        If no keys containing 'mod' or 'sat' are found in the dictionary.

    Notes
    -----
    - The function concatenates monthly data across all years before computing the metric.
    - Handles NaNs by excluding them pairwise in the calculation.
    - Requires at least two valid data points per month to return a numeric result.
    - Returns np.nan for months with insufficient data.

    Examples
    --------
    >>> data = {
    ...     'mod_data': {2020: [np.array([1,2]), np.array([3,4]), ...], ...},
    ...     'sat_data': {2020: [np.array([1.1,1.9]), np.array([2.9,4.1]), ...], ...}
    ... }
    >>> monthly_index_of_agreement(data)
    [0.98, 0.95, ..., 0.97]  # 12 values for each month
    """
    from .Efficiency_metrics import index_of_agreement
    
    # Input type validation
    if not isinstance(dictionary, dict):
        raise TypeError("❌ Input must be a dictionary. ❌")

    # Find keys that likely correspond to model and satellite data (case-insensitive)
    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in dictionary if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in dictionary if any(kw in k.lower() for kw in sat_keywords)), None)

    # ===== INPUT VALIDATION =====
    # Raise error if either model or satellite keys are missing
    if not model_key or not sat_key:
        raise KeyError("❌ Model or satellite key not found in the dictionary ❌")

    # Extract dictionaries of monthly data by year for model and satellite
    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    # Check that both have the same years
    years_mod = set(mod_monthly.keys())
    years_sat = set(sat_monthly.keys())
    if years_mod != years_sat:
        raise ValueError("❌ Mismatch in years between model and satellite data ❌")

    years = list(years_mod)
    if not years:
        raise ValueError("❌ No yearly data found in model and satellite datasets ❌")

    # Determine number of months dynamically from first available year/model data
    first_year = years[0]
    n_months = len(mod_monthly[first_year])

    # Validate structure for all years (same number of months)
    for year in years:
        if len(mod_monthly[year]) != n_months or len(sat_monthly[year]) != n_months:
            raise ValueError(f"❌ Inconsistent number of months for year {year} in model or satellite data ❌")

    d_monthly = []

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_index_of_agreement function"):
        with start_action(action_type="monthly_index_of_agreement") as action:
            log_message("Entered monthly_index_of_agreement", years=years, n_months=n_months)
            logging.info("[Start] monthly_index_of_agreement computation")

            # Loop over months dynamically
            for month in range(n_months):
                # Concatenate monthly data across all years for model and satellite
                mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
                sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

                # Create a mask to keep only pairs without NaNs in either dataset
                valid_mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

                if np.any(valid_mask):
                    # Compute the Index of Agreement on valid paired data
                    d_val = index_of_agreement(sat_all[valid_mask], mod_all[valid_mask])
                else:
                    # Not enough valid data, assign NaN for this month
                    d_val = np.nan

                d_monthly.append(d_val)

                log_message("Computed monthly Index of Agreement", month=month + 1, index_of_agreement=d_val)
                logging.info(f"Month {month + 1}: Index of Agreement = {d_val}")

            logging.info("[Done] monthly_index_of_agreement computation completed")
            log_message("Completed monthly_index_of_agreement", total_months=len(d_monthly))

            return d_monthly
        
###############################################################################

############################################################################### 

def ln_nse(
    obs: Union[Sequence[float], np.ndarray], 
    pred: Union[Sequence[float], np.ndarray]
) -> float:
    """
    Compute the Nash–Sutcliffe Efficiency (NSE) on the natural logarithms of observed and predicted data.

    This metric evaluates model performance emphasizing relative differences by 
    transforming data with the natural logarithm. It is useful when data span 
    several orders of magnitude or when multiplicative errors are more meaningful.

    Parameters
    ----------
    obs : array-like
        Observed values (must be strictly positive).
    pred : array-like
        Predicted values (must be strictly positive).

    Returns
    -------
    float
        Logarithmic NSE value, or np.nan if there is insufficient valid data or
        if input contains non-positive values.

    Notes
    -----
    - Both obs and pred must contain strictly positive values; zeros or negatives
      will be excluded from the calculation.
    - The function computes NSE on ln(obs) and ln(pred).
    - Requires at least two valid paired observations.
    - Returns np.nan if denominator in NSE calculation is zero or data are insufficient.

    Examples
    --------
    >>> obs = np.array([1.0, 10.0, 100.0, 1000.0])
    >>> pred = np.array([1.1, 9.5, 110.0, 950.0])
    >>> ln_nse(obs, pred)
    0.95  # example output (actual value depends on data)
    """
    # ===== INPUT VALIDATION =====
    if not hasattr(obs, "__iter__") or not hasattr(pred, "__iter__"):
        raise TypeError("❌ Inputs obs and pred must be array-like. ❌")

    obs = np.asarray(obs)
    pred = np.asarray(pred)

    if obs.shape != pred.shape:
        raise ValueError("❌ Inputs obs and pred must have the same shape. ❌")

    if obs.size == 0:
        raise ValueError("❌ Input arrays must not be empty. ❌")

    # Create mask to filter out NaNs and non-positive values,
    # since log is undefined for <= 0, and we need paired valid data
    mask = (~np.isnan(obs)) & (~np.isnan(pred)) & (obs > 0) & (pred > 0)

    # Require at least two valid data points to compute meaningful NSE
    if np.sum(mask) < 2:
        return np.nan

    # ===== COMPUTATION AND LOGGING =====
    with Timer("ln_nse function"):
        with start_action(action_type="ln_nse") as action:
            log_message("Entered ln_nse function", valid_data_points=np.sum(mask))
            logging.info("[Start] Logarithmic NSE computation")

            # Apply natural logarithm transformation to emphasize relative errors
            log_obs = np.log(obs[mask])
            log_pred = np.log(pred[mask])

            # Calculate sum of squared differences (residual variance)
            numerator = np.sum((log_obs - log_pred) ** 2)

            # Calculate total variance of log-observed values (signal variance)
            denominator = np.sum((log_obs - np.mean(log_obs)) ** 2)

            # If denominator is zero, variance is zero and NSE is undefined
            if denominator == 0:
                logging.warning("Log-observed variance is zero; ln_NSE is undefined (NaN returned).")
                log_message("ln_NSE undefined due to zero variance in log-observations")
                return np.nan

            # NSE formula: 1 minus ratio of residual variance to signal variance
            ln_nse_value = 1 - numerator / denominator

            log_message("Computed ln_NSE", ln_nse=ln_nse_value)
            logging.info(f"[Done] Logarithmic NSE computation: {ln_nse_value}")

            return ln_nse_value
        
###############################################################################

############################################################################### 

def monthly_ln_nse(
    dictionary: Dict[str, Dict[int, List[Union[np.ndarray, list]]]]
) -> List[float]:
    """
    Compute monthly logarithmic Nash–Sutcliffe Efficiency (ln NSE) from paired model and satellite data.

    This metric evaluates model performance on the natural logarithm scale,
    emphasizing relative differences and multiplicative errors.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each key maps to a dict of years, each year a list/array of 12 monthly arrays.

    Returns
    -------
    list of float
        Logarithmic NSE values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing.

    Notes
    -----
    - Only pairs of positive values (both observed and predicted) are considered for each month.
    - Months without sufficient valid data yield np.nan.
    - Relies on the ln_nse function to compute the metric.

    Examples
    --------
    >>> data = {
    ...     'model': {2020: [np.array([1,2]), np.array([3,4]), ..., np.array([11,12])]},
    ...     'satellite': {2020: [np.array([1.1,2.1]), np.array([2.9,3.8]), ..., np.array([10.9,11.8])]}
    ... }
    >>> monthly_ln_nse(data)
    [0.95, 0.87, ..., 0.93]  # example output (actual values depend on data)
    """
    from .Efficiency_metrics import ln_nse

    # Identify keys in the dictionary corresponding to model and satellite data (case-insensitive)
    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in dictionary if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in dictionary if any(kw in k.lower() for kw in sat_keywords)), None)

    if not model_key or not sat_key:
        raise KeyError("❌ Model or satellite key not found in the dictionary. ❌")

    # Extract dictionaries of monthly data by year for model and satellite
    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    # Get years present in both model and satellite dictionaries
    years = list(set(mod_monthly.keys()) & set(sat_monthly.keys()))
    if not years:
        raise ValueError("❌ No overlapping years found between model and satellite data. ❌")

    # Determine the number of months from the first year and key (validate consistent month counts)
    first_year = years[0]
    n_months = len(mod_monthly[first_year])
    for y in years:
        if len(mod_monthly[y]) != n_months or len(sat_monthly[y]) != n_months:
            raise ValueError("❌ Inconsistent number of months per year in data arrays. ❌")

    ln_nse_monthly = []

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_ln_nse function"):
        with start_action(action_type="monthly_ln_nse") as action:
            log_message("Entered monthly_ln_nse", years=years, n_months=n_months)
            logging.info("[Start] monthly_ln_nse computation")

            # Loop through all months dynamically (no hardcoded 12)
            for month in range(n_months):
                # Concatenate monthly data arrays across all years into single 1D arrays
                mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
                sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

                # Create a mask to select paired values that are not NaN and strictly positive,
                # because logarithms require positive inputs
                valid_mask = (~np.isnan(mod_all) & ~np.isnan(sat_all) & (mod_all > 0) & (sat_all > 0))

                # Compute ln_nse only if there is at least two valid paired data points
                if np.sum(valid_mask) >= 2:
                    ln_nse_val = ln_nse(sat_all[valid_mask], mod_all[valid_mask])
                else:
                    # If no or insufficient valid data, set result as NaN for this month
                    ln_nse_val = np.nan

                # Append the monthly ln NSE value to the result list
                ln_nse_monthly.append(ln_nse_val)

                log_message("Computed monthly ln NSE", month=month + 1, ln_nse=ln_nse_val)
                logging.info(f"Month {month + 1}: ln NSE = {ln_nse_val}")

            logging.info("[Done] monthly_ln_nse computation completed")
            log_message("Completed monthly_ln_nse", total_months=len(ln_nse_monthly))

            return ln_nse_monthly
        
###############################################################################

############################################################################### 

def nse_j(
    obs: Union[Sequence[float], np.ndarray], 
    pred: Union[Sequence[float], np.ndarray], 
    j: float = 1
) -> float:
    """
    Compute modified Nash–Sutcliffe Efficiency (E_j) for an arbitrary exponent j.

    This generalizes the Nash–Sutcliffe Efficiency by raising the absolute differences 
    between observed and predicted values to the power j, allowing flexible weighting 
    of deviations.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.
    j : float, optional
        Exponent applied to absolute differences (default is 1).

    Returns
    -------
    float
        Modified NSE value (E_j), or np.nan if insufficient valid data or zero denominator.

    Notes
    -----
    - The modified NSE is defined as: 
      E_j = 1 - (sum(|obs - pred|^j) / sum(|obs - mean(obs)|^j))
    - Requires at least two paired valid values.
    - If the denominator is zero (no variability in obs), returns np.nan.
    - Increasing j increases sensitivity to larger errors.

    Examples
    --------
    >>> obs = np.array([1, 2, 3, 4])
    >>> pred = np.array([1.1, 1.9, 2.8, 4.2])
    >>> nse_j(obs, pred, j=2)
    0.95  # Example value, actual depends on data
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)

    # ===== INPUT VALIDATION =====
    # Ensure obs and pred have the same shape
    if obs.shape != pred.shape:
        raise ValueError("❌ Observed and predicted arrays must have the same shape. ❌")

    # Mask to ignore pairs where either value is NaN
    mask = ~np.isnan(obs) & ~np.isnan(pred)

    # Require at least two valid paired values to compute metric
    if np.sum(mask) < 2:
        return np.nan

    obs_masked = obs[mask]
    pred_masked = pred[mask]

    # ===== COMPUTATION AND LOGGING =====
    with Timer("nse_j function"):
        with start_action(action_type="nse_j") as action:
            log_message("Entered nse_j function", valid_data_points=len(obs_masked), exponent=j)
            logging.info("[Start] Modified NSE computation")

            numerator = np.sum(np.abs(obs_masked - pred_masked) ** j)
            denominator = np.sum(np.abs(obs_masked - np.mean(obs_masked)) ** j)

            # If denominator is zero, no variability in obs, metric undefined
            if denominator == 0:
                logging.warning("Observed variance is zero; modified NSE is undefined (NaN returned).")
                log_message("Modified NSE undefined due to zero variance in observations")
                return np.nan

            modified_nse = 1 - numerator / denominator

            log_message("Computed modified NSE", modified_nse=modified_nse, exponent=j)
            logging.info(f"[Done] Modified NSE computation: {modified_nse}")

            return modified_nse
        
###############################################################################

############################################################################### 

def monthly_nse_j(
    dictionary: Dict[str, Dict[int, List[Union[np.ndarray, list]]]],
    j: float = 1
) -> List[float]:
    """
    Compute monthly modified Nash–Sutcliffe Efficiency (E_j) for arbitrary exponent j
    from paired model and satellite data.

    This generalizes the NSE by raising absolute differences to the power j,
    allowing flexible emphasis on deviations.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each key maps to a dict of years, each year a list/array of 12 monthly arrays.
    j : float, optional
        Exponent for the absolute difference (default is 1).

    Returns
    -------
    list of float
        Modified NSE values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing.

    Notes
    -----
    - The function calculates E_j = 1 - sum(|obs - pred|^j) / sum(|obs - mean(obs)|^j) for each month.
    - Requires at least two valid paired values per month, else returns np.nan.
    - Higher values of j increase sensitivity to large errors.

    Examples
    --------
    >>> data = {
    ...     'model': {2020: [np.array([1, 2]), ..., np.array([11, 12])]},
    ...     'satellite': {2020: [np.array([1.1, 2.1]), ..., np.array([10.9, 11.8])]}
    ... }
    >>> monthly_nse_j(data, j=2)
    [0.90, 0.85, ..., 0.88]  # example output (depends on data)
    """
    from .Efficiency_metrics import nse_j

    # Find keys corresponding to model and satellite data (case-insensitive)
    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in dictionary if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in dictionary if any(kw in k.lower() for kw in sat_keywords)), None)

    if not model_key or not sat_key:
        raise KeyError("❌ Model or satellite key not found in the dictionary. ❌")

    # Extract dictionaries of monthly data by year for model and satellite
    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    years = list(mod_monthly.keys())

    # Dynamically determine number of months from the first year
    num_months = len(mod_monthly[years[0]])

    nse_j_monthly = []

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_nse_j function"):
        with start_action(action_type="monthly_nse_j") as action:
            log_message("Entered monthly_nse_j", years=years, num_months=num_months, exponent=j)
            logging.info("[Start] monthly_nse_j computation")

            # ===== LOOPING =====
            for month in range(num_months):
                mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
                sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

                valid_mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

                if np.sum(valid_mask) >= 2:
                    val = nse_j(sat_all[valid_mask], mod_all[valid_mask], j=j)
                else:
                    val = np.nan

                nse_j_monthly.append(val)

                log_message("Computed monthly nse_j", month=month + 1, nse_j=val)
                logging.info(f"Month {month + 1}: nse_j = {val}")

            logging.info("[Done] monthly_nse_j computation completed")
            log_message("Completed monthly_nse_j", total_months=len(nse_j_monthly))

            return nse_j_monthly
        
###############################################################################

###############################################################################
 
def index_of_agreement_j(
    obs: Union[Sequence[float], np.ndarray], 
    pred: Union[Sequence[float], np.ndarray], 
    j: float = 1
) -> float:
    """
    Compute modified Index of Agreement (d_j) with arbitrary exponent j.

    This generalizes the Index of Agreement by raising absolute deviations 
    to the power j, allowing flexible emphasis on prediction errors.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.
    j : float, optional
        Exponent parameter applied to absolute deviations (default is 1).

    Returns
    -------
    float
        Modified Index of Agreement (d_j), or np.nan if insufficient valid data or zero denominator.

    Notes
    -----
    - The modified index is defined as:
      d_j = 1 - (sum(|obs - pred|^j) / sum((|pred - mean(obs)| + |obs - mean(obs)|)^j))
    - Requires at least two paired valid values.
    - If denominator is zero (lack of variability), returns np.nan.
    - Larger j values penalize larger deviations more heavily.

    Examples
    --------
    >>> obs = np.array([2, 3, 4])
    >>> pred = np.array([2.1, 2.9, 3.8])
    >>> index_of_agreement_j(obs, pred, j=2)
    0.92  # example output, actual depends on data
    """
    obs = np.asarray(obs)  # Convert inputs to numpy arrays
    pred = np.asarray(pred)

    # ===== INPUT VALIDATION =====
    # Validate shapes are equal to avoid silent errors
    if obs.shape != pred.shape:
        raise ValueError("❌ Observed and predicted arrays must have the same shape. ❌")

    mask = ~np.isnan(obs) & ~np.isnan(pred)  # Ignore NaNs in obs or pred

    if np.sum(mask) < 2:  # Need at least two valid pairs
        return np.nan

    obs_masked = obs[mask]
    pred_masked = pred[mask]

    # ===== COMPUTATION AND LOGGING =====
    with Timer("index_of_agreement_j function"):
        with start_action(action_type="index_of_agreement_j") as action:
            log_message("Entered index_of_agreement_j function", valid_data_points=len(obs_masked), exponent=j)
            logging.info("[Start] Modified Index of Agreement computation")

            # Sum of powered absolute errors between observed and predicted
            numerator = np.sum(np.abs(obs_masked - pred_masked) ** j)

            # Sum of powered combined deviations from mean of observed data
            denominator = np.sum((np.abs(pred_masked - np.mean(obs_masked)) + np.abs(obs_masked - np.mean(obs_masked))) ** j)

            if denominator == 0:  # Avoid division by zero if no variability in obs
                logging.warning("Observed variance is zero; modified Index of Agreement is undefined (NaN returned).")
                log_message("Modified Index of Agreement undefined due to zero variance in observations")
                return np.nan

            modified_index = 1 - numerator / denominator

            log_message("Computed modified Index of Agreement", modified_index=modified_index, exponent=j)
            logging.info(f"[Done] Modified Index of Agreement computation: {modified_index}")

            return modified_index
        
############################################################################### 

############################################################################### 

def monthly_index_of_agreement_j(
    dictionary: Dict[str, Dict[int, List[Union[np.ndarray, list]]]],
    j: float = 1
) -> List[float]:
    """
    Compute monthly modified Index of Agreement (d_j) with exponent j from paired model and satellite data.

    Calculates the modified Index of Agreement for each calendar month by aggregating
    all paired model and satellite data across years. The exponent j controls the sensitivity
    of the metric to deviations, with j=1 corresponding to the standard index.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each key maps to a dict of years, where each year is a list or array of 12 monthly arrays.
    j : float, optional
        Exponent parameter for the modified Index of Agreement (default is 1).

    Returns
    -------
    list of float
        Modified Index of Agreement (d_j) values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing from the dictionary.

    Notes
    -----
    - Requires at least two paired valid observations per month.
    - Returns np.nan for months with insufficient data or zero variability.
    - The metric generalizes the traditional Index of Agreement by raising deviations to the power j,
      allowing emphasis on different scales of error.
    
    Examples
    --------
    >>> dictionary = {
    ...     'mod_data': {2020: [np.array([1,2]), np.array([3,4]), ...], 2021: [...], ...},
    ...     'sat_data': {2020: [np.array([1.1,2.1]), np.array([3.1,3.9]), ...], 2021: [...], ...}
    ... }
    >>> monthly_index_of_agreement_j(dictionary, j=2)
    [0.85, 0.88, ..., 0.90]  # list of 12 floats, one per month
    """
    from .Efficiency_metrics import index_of_agreement_j

    # Identify keys containing model and satellite data
    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in dictionary if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in dictionary if any(kw in k.lower() for kw in sat_keywords)), None)

    if not model_key or not sat_key:
        raise KeyError("❌ Model or satellite key not found in the dictionary. ❌")

    # Extract dictionaries of monthly data by year for model and satellite
    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    # List all years available in the dataset
    years = list(mod_monthly.keys())

    # Determine number of months from first year available
    first_year = years[0]
    n_months = len(mod_monthly[first_year])

    d_j_monthly = []

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_index_of_agreement_j function"):
        with start_action(action_type="monthly_index_of_agreement_j") as action:
            log_message("Entered monthly_index_of_agreement_j", years=years, n_months=n_months, exponent=j)
            logging.info("[Start] monthly_index_of_agreement_j computation")

            # ===== LOOPING =====
            for month in range(n_months):
                # Aggregate all model data for the current month across years
                mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
                # Aggregate all satellite data for the current month across years
                sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

                # Mask to select valid paired (non-NaN) observations
                mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

                if np.any(mask):
                    # Compute the modified index of agreement for valid data only
                    val = index_of_agreement_j(sat_all[mask], mod_all[mask], j=j)
                else:
                    # Return NaN if no valid data points available
                    val = np.nan

                d_j_monthly.append(val)

                log_message("Computed monthly index_of_agreement_j", month=month + 1, value=val)
                logging.info(f"Month {month + 1}: index_of_agreement_j = {val}")

            logging.info("[Done] monthly_index_of_agreement_j computation completed")
            log_message("Completed monthly_index_of_agreement_j", total_months=len(d_j_monthly))

            return d_j_monthly
        
###############################################################################

############################################################################### 

def relative_nse(
    obs: Union[Sequence[float], np.ndarray], 
    pred: Union[Sequence[float], np.ndarray]
) -> float:
    """
    Compute the Relative Nash–Sutcliffe Efficiency (Relative NSE) between observations and predictions.

    This metric evaluates model performance by comparing relative deviations 
    (normalized by observations) rather than absolute deviations, making it 
    sensitive to proportional errors.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        Relative NSE value, or np.nan if insufficient data, division by zero,
        or invalid calculation occurs.

    Notes
    -----
    - Observations with zero values are excluded to avoid division by zero.
    - Requires at least two valid paired observations.
    - Relative NSE close to 1 indicates good model performance on relative scale.
    - A small denominator (zero variance in relative observations) returns np.nan.

    Examples
    --------
    >>> obs = np.array([10, 20, 30, 40])
    >>> pred = np.array([11, 18, 33, 39])
    >>> relative_nse(obs, pred)
    0.95  # example output
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)

    # Mask to exclude NaNs and zeros in observations (to avoid division by zero)
    mask = ~np.isnan(obs) & ~np.isnan(pred) & (obs != 0)

    # Require at least two valid pairs
    if np.sum(mask) < 2:
        return np.nan

    obs_masked = obs[mask]
    pred_masked = pred[mask]

    obs_mean = np.mean(obs_masked)

    # ===== COMPUTATION AND LOGGING =====
    with Timer("relative_nse function"):
        with start_action(action_type="relative_nse") as action:
            log_message("Entered relative_nse function", valid_data_points=len(obs_masked))
            logging.info("[Start] Relative NSE computation")

            # Numerator: sum of squared relative errors
            numerator = np.sum(((obs_masked - pred_masked) / obs_masked) ** 2)
            # Denominator: sum of squared relative deviations from mean
            denominator = np.sum(((obs_masked - obs_mean) / obs_mean) ** 2)

            if denominator == 0:
                logging.warning("Relative variance denominator is zero; Relative NSE undefined (NaN returned).")
                log_message("Relative NSE undefined due to zero relative variance in observations")
                return np.nan

            rel_nse_value = 1 - numerator / denominator

            log_message("Computed Relative NSE", relative_nse=rel_nse_value)
            logging.info(f"[Done] Relative NSE computation: {rel_nse_value}")

            return rel_nse_value
        
###############################################################################

############################################################################### 

def monthly_relative_nse(
    dictionary: Dict[str, Dict[int, List[Union[np.ndarray, list]]]]
) -> List[float]:
    """
    Compute monthly Relative Nash–Sutcliffe Efficiency (Relative NSE) from paired model and satellite data.

    This function calculates the Relative NSE metric for each calendar month by aggregating
    data across all available years. It compares relative deviations normalized by observations,
    emphasizing proportional accuracy of model predictions compared to satellite observations.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys with 'mod' and 'sat' identifying model and satellite datasets.
        Each key maps to a dict of years, with each year containing a list or array of 12 monthly data arrays.

    Returns
    -------
    list of float
        Relative NSE values computed for each month (length 12). Returns np.nan for months with insufficient or invalid data.

    Raises
    ------
    KeyError
        If either model ('mod') or satellite ('sat') keys are missing in the dictionary.

    Notes
    -----
    - Observations with zero values are excluded to avoid division by zero.
    - Requires at least two valid paired observations per month to compute the metric.
    - Relative NSE close to 1 indicates good proportional agreement between model and observations.
    - Months with zero variance in relative observations or insufficient data return np.nan.

    Examples
    --------
    >>> dictionary = {
    ...     'mod': {
    ...         2020: [np.array([10, 15]), np.array([20, 25]), ..., np.array([30, 35])],  # 12 arrays for each month
    ...         2021: [np.array([12, 16]), np.array([22, 26]), ..., np.array([32, 37])]
    ...     },
    ...     'sat': {
    ...         2020: [np.array([9, 14]), np.array([19, 24]), ..., np.array([29, 34])],
    ...         2021: [np.array([11, 15]), np.array([21, 25]), ..., np.array([31, 36])]
    ...     }
    ... }
    >>> monthly_relative_nse(dictionary)
    [0.95, 0.97, ..., 0.93]  # example output for each month
    """
    from .Efficiency_metrics import relative_nse

    # ===== INPUT VALIDATION =====
    # Input validation: dictionary type
    if not isinstance(dictionary, dict):
        raise TypeError("❌ Input must be a dictionary ❌")

    # Identify model and satellite keys (case-insensitive)
    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in dictionary if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in dictionary if any(kw in k.lower() for kw in sat_keywords)), None)

    if not model_key or not sat_key:
        raise KeyError("❌ Model or satellite key not found in the dictionary. ❌")

    # Extract dictionaries of monthly data by year for model and satellite
    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    # Validate that mod_monthly and sat_monthly are dicts
    if not isinstance(mod_monthly, dict):
        raise ValueError("❌ Model data must be a dictionary of years mapping to monthly arrays ❌")
    if not isinstance(sat_monthly, dict):
        raise ValueError("❌ Satellite data must be a dictionary of years mapping to monthly arrays ❌")

    years = list(mod_monthly.keys())
    if not years:
        raise ValueError("❌ No yearly data found in model dataset ❌")

    # Check first year has monthly data, and get number of months dynamically
    first_year = years[0]
    if first_year not in sat_monthly:
        raise ValueError(f"❌ Year {first_year} missing from satellite data ❌")

    n_months = len(mod_monthly[first_year])

    # Validate the monthly data structures for both model and satellite for the first year
    if not isinstance(mod_monthly[first_year], (list, tuple)) or not isinstance(sat_monthly[first_year], (list, tuple)):
        raise ValueError("❌ Monthly data for the first year must be a list or tuple of monthly arrays ❌")

    for month_data in mod_monthly[first_year]:
        if not hasattr(month_data, "__iter__"):
            raise ValueError("❌ Each monthly entry in model data must be iterable (like a list or numpy array) ❌")
    for month_data in sat_monthly[first_year]:
        if not hasattr(month_data, "__iter__"):
            raise ValueError("❌ Each monthly entry in satellite data must be iterable (like a list or numpy array) ❌")

    e_rel_monthly = []

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_relative_nse function"):
        with start_action(action_type="monthly_relative_nse") as action:
            log_message("Entered monthly_relative_nse", years=years, n_months=n_months)
            logging.info("[Start] monthly_relative_nse computation")

            # ===== LOOPING =====
            for month in range(n_months):
                try:
                    # Aggregate monthly data across all years
                    mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
                    sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])
                except Exception as e:
                    raise ValueError(f"❌ Error concatenating monthly data for month {month}: {e} ❌")

                # Mask: exclude NaNs and zero satellite observations to avoid division by zero
                mask = (~np.isnan(mod_all)) & (~np.isnan(sat_all)) & (sat_all != 0)

                if np.sum(mask) >= 2:  # Require at least two valid pairs
                    val = relative_nse(sat_all[mask], mod_all[mask])
                else:
                    val = np.nan

                e_rel_monthly.append(val)

                log_message("Computed monthly_relative_nse", month=month + 1, value=val)
                logging.info(f"Month {month + 1}: relative_nse = {val}")

            logging.info("[Done] monthly_relative_nse computation completed")
            log_message("Completed monthly_relative_nse", total_months=len(e_rel_monthly))

            return e_rel_monthly
        
###############################################################################

############################################################################### 

def relative_index_of_agreement(
    obs: Union[Sequence[float], np.ndarray], 
    pred: Union[Sequence[float], np.ndarray]
) -> float:
    """
    Compute the Relative Index of Agreement (d_rel) between observed and predicted values.

    This metric assesses the agreement between predictions and observations by evaluating
    relative errors normalized by the observations, making it sensitive to proportional differences.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        Relative Index of Agreement value ranging typically between 0 and 1, where values closer to 1
        indicate better agreement. Returns np.nan if the calculation is invalid due to insufficient
        data, zero variance, or division by zero.

    Notes
    -----
    - Observations with zero values are excluded to avoid division by zero errors.
    - Requires at least two valid paired observations.
    - Returns np.nan if the observations have zero variance (all equal).
    - Sensitive to relative rather than absolute errors.

    Examples
    --------
    >>> obs = np.array([10, 20, 30, 40])
    >>> pred = np.array([11, 19, 28, 39])
    >>> relative_index_of_agreement(obs, pred)
    0.92  # example output
    """
    # ===== INPUT VALIDATION =====
    if obs is None or pred is None:
        raise ValueError("❌ Observed and predicted inputs must not be None ❌")

    obs = np.asarray(obs)
    pred = np.asarray(pred)

    if obs.shape != pred.shape:
        raise ValueError("❌ Observed and predicted arrays must have the same shape ❌")

    # Mask to exclude NaNs and zero observations (avoid division by zero)
    mask = ~np.isnan(obs) & ~np.isnan(pred) & (obs != 0)

    # Require at least two valid pairs
    if np.sum(mask) < 2:
        return np.nan  # Insufficient data

    obs_masked = obs[mask]
    pred_masked = pred[mask]

    obs_mean = np.mean(obs_masked)

    # Check for zero variance in observations
    if np.allclose(obs_masked, obs_mean):
        return np.nan

    # ===== COMPUTATION AND LOGGING =====
    with Timer("relative_index_of_agreement function"):
        with start_action(action_type="relative_index_of_agreement") as action:
            log_message("Entered relative_index_of_agreement function", valid_data_points=len(obs_masked))
            logging.info("[Start] Relative Index of Agreement computation")

            numerator = np.sum(((obs_masked - pred_masked) / obs_masked) ** 2)

            denominator = np.sum(
                ((np.abs(pred_masked - obs_mean) + np.abs(obs_masked - obs_mean)) / obs_mean) ** 2
            )

            if denominator == 0:
                logging.warning("Relative Index of Agreement denominator is zero; undefined (NaN returned).")
                log_message("Relative Index of Agreement undefined due to zero denominator")
                return np.nan

            ria_value = 1 - numerator / denominator

            log_message("Computed Relative Index of Agreement", relative_index_of_agreement=ria_value)
            logging.info(f"[Done] Relative Index of Agreement computation: {ria_value}")

            return ria_value
        
###############################################################################

############################################################################### 

def monthly_relative_index_of_agreement(
    dictionary: Dict[str, Dict[int, List[Union[np.ndarray, list]]]]
) -> List[float]:
    """
    Compute the Relative Index of Agreement (d_rel) for each calendar month by aggregating 
    paired observed (satellite) and predicted (model) data across multiple years.

    This metric assesses proportional agreement between observations and predictions on 
    a monthly basis, by evaluating relative errors normalized by observations. It is 
    sensitive to proportional differences rather than absolute errors.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing paired model and satellite monthly data with keys containing
        'mod' and 'sat' respectively. Each key maps to a dictionary of years, where each
        year contains a list or array of 12 elements representing monthly data:
        {
            'mod...': {year1: [month_0_data, ..., month_11_data], year2: [...], ...},
            'sat...': {year1: [month_0_data, ..., month_11_data], year2: [...], ...}
        }

    Returns
    -------
    list of float
        List of 12 Relative Index of Agreement values, one for each month (January=0,...,December=11).
        Returns np.nan for months with insufficient or invalid data.

    Notes
    -----
    - Observations (satellite data) with zero values are excluded to avoid division by zero errors.
    - Requires at least two valid paired observations per month.
    - Returns np.nan if the observations have zero variance (all equal) or denominator is zero.
    - The metric ranges typically between 0 and 1, with values closer to 1 indicating better agreement.

    Examples
    --------
    >>> dictionary = {
    ...     'mod_data': {
    ...         2020: [np.array([1,2]), np.array([3,4]), ...],  # 12 months of data per year
    ...         2021: [np.array([2,3]), np.array([4,5]), ...]
    ...     },
    ...     'sat_data': {
    ...         2020: [np.array([1.1,1.9]), np.array([2.9,4.1]), ...],
    ...         2021: [np.array([2.1,2.8]), np.array([3.8,5.2]), ...]
    ...     }
    ... }
    >>> monthly_relative_index_of_agreement(dictionary)
    [0.95, 0.91, ..., 0.89]  # example output list with 12 values
    """
    from .Efficiency_metrics import relative_index_of_agreement

    # Find dictionary keys corresponding to model and satellite data
    # Define keyword groups for matching
    model_keywords = ['sim', 'simulated', 'model', 'mod']
    sat_keywords = ['obs', 'observed', 'sat', 'satellite']

    # Try to find keys that match each group
    model_key = next((k for k in dictionary if any(kw in k.lower() for kw in model_keywords)), None)
    sat_key = next((k for k in dictionary if any(kw in k.lower() for kw in sat_keywords)), None)

    if not model_key or not sat_key:
        raise KeyError("❌ Model or satellite key not found in the dictionary. ❌")

    # Extract dictionaries of monthly data by year for model and satellite
    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    # Get all years available in the model data (assumed same in satellite)
    years = list(mod_monthly.keys())
    if not years:
        return []

    # Determine number of months from the first available year (avoid hardcoded 12)
    first_year = years[0]
    num_months = len(mod_monthly[first_year])

    d_rel_monthly = []

    # ===== COMPUTATION AND LOGGING =====
    with Timer("monthly_relative_index_of_agreement function"):
        with start_action(action_type="monthly_relative_index_of_agreement") as action:
            log_message("Entered monthly_relative_index_of_agreement", years=years, num_months=num_months)
            logging.info("[Start] monthly_relative_index_of_agreement computation")

            # ===== LOOPING =====
            for month in range(num_months):
                # Concatenate monthly data for all years into single arrays
                mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
                sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

                # Mask to exclude NaNs and satellite zeros (avoid division by zero)
                mask = (~np.isnan(mod_all)) & (~np.isnan(sat_all)) & (sat_all != 0)

                # Compute relative index of agreement if data is valid; else NaN
                if np.any(mask):
                    d_rel = relative_index_of_agreement(sat_all[mask], mod_all[mask])
                else:
                    d_rel = np.nan

                d_rel_monthly.append(d_rel)

                log_message("Computed monthly_relative_index_of_agreement", month=month + 1, value=d_rel)
                logging.info(f"Month {month + 1}: relative_index_of_agreement = {d_rel}")

            logging.info("[Done] monthly_relative_index_of_agreement computation completed")
            log_message("Completed monthly_relative_index_of_agreement", total_months=len(d_rel_monthly))

            return d_rel_monthly
        
###############################################################################

###############################################################################

def compute_spatial_efficiency(
    model_da: xr.DataArray, 
    sat_da: xr.DataArray, 
    time_group: Literal["month", "year"] = "month"
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute spatial efficiency metrics between model and satellite data aggregated over time groups.

    This function calculates multiple performance metrics spatially across the domain by aggregating
    the input datasets over calendar months or years. It returns time-resolved maps for each metric.

    Parameters
    ----------
    model_da : xarray.DataArray
        Model data with a 'time' coordinate.
    sat_da : xarray.DataArray
        Satellite (observed) data with a 'time' coordinate, matching the model in space and time.
    time_group : {'month', 'year'}, optional
        Temporal aggregation level:
            - 'month': groups data by calendar month (1–12),
            - 'year': groups data by unique years in the time dimension.

    Returns
    -------
    tuple of xarray.DataArray
        Six DataArrays with spatial metrics computed for each time group (month or year), with dimensions:
        (time_group, lat, lon). The returned metrics are:
            - mb_all : Mean Bias
            - sde_all : Standard Deviation of the Error
            - cc_all : Pearson Cross-Correlation
            - rm_all : Standard Deviation of the Model
            - ro_all : Standard Deviation of the Observation
            - urmse_all : Unbiased Root Mean Squared Error

    Raises
    ------
    ValueError
        If `time_group` is not 'month' or 'year'.

    Notes
    -----
        - Input DataArrays must have a 'time' coordinate with datetime-like values.
        - Each metric is computed for all available times in the group (month or year).
        - The function assumes spatial alignment between model and satellite datasets.

    Examples
    --------
    >>> mb, sde, cc, rm, ro, urmse = compute_spatial_efficiency(model_da, sat_da, time_group="month")
    >>> mb.sel(month=1).plot()  # Plot Mean Bias for January
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(model_da, xr.DataArray):
        raise TypeError("❌ 'model_da' must be an xarray.DataArray ❌")
    if not isinstance(sat_da, xr.DataArray):
        raise TypeError("❌ 'sat_da' must be an xarray.DataArray ❌")
    if "time" not in model_da.coords:
        raise ValueError("❌ 'model_da' must have a 'time' coordinate ❌")
    if "time" not in sat_da.coords:
        raise ValueError("❌ 'sat_da' must have a 'time' coordinate ❌")
    if not np.array_equal(model_da['time'], sat_da['time']):
        raise ValueError("❌ 'model_da' and 'sat_da' must have the same 'time' coordinate values ❌")
    if time_group not in {"month", "year"}:
        raise ValueError(f"❌ Invalid time_group '{time_group}', must be 'month' or 'year' ❌")

    # ===== TIME CLASSIFICATION =====
    # Determine grouping based on selected time group
    if time_group == "month":
        groups = range(1, 13)  # Months 1 to 12
        time_sel = 'month'
    elif time_group == "year":
        groups = sorted(np.unique(model_da['time.year'].values))  # Unique years in the data
        time_sel = 'year'

    # ==== INNER FUNCTION TO COMPUTE GROUP =====
    # Define function to compute all metrics for a single time group
    def compute_metrics_for_group(group):
        # Select model and satellite data for the given group
        m_sel = model_da.sel(time=model_da['time.' + time_sel] == group)
        o_sel = sat_da.sel(time=sat_da['time.' + time_sel] == group)

        # Compute and return all six metrics for the selected group
        return (
            mean_bias(m_sel, o_sel),
            standard_deviation_error(m_sel, o_sel),
            cross_correlation(m_sel, o_sel),
            std_dev(m_sel),
            std_dev(o_sel),
            unbiased_rmse(m_sel, o_sel),
        )

    # ===== COMPUTATION AND LOGGING =====
    with Timer("compute_spatial_efficiency function"):
        with start_action(action_type="compute_spatial_efficiency") as action:
            log_message("Started compute_spatial_efficiency", time_group=time_group, groups=list(groups))
            logging.info(f"[Start] Computing spatial efficiency metrics grouped by {time_group}")

            # Apply the metrics computation for each time group
            results = list(map(compute_metrics_for_group, groups))

            # Unpack the results into separate metric lists
            mb_maps, sde_maps, cc_maps, rm_maps, ro_maps, urmse_maps = zip(*results)

            # Set dimension name and coordinates based on the time grouping
            dim_name = time_group
            coord_vals = groups

            # Concatenate results into DataArrays with the correct dimension and coordinate assignment
            mb_all = xr.concat(mb_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
            sde_all = xr.concat(sde_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
            cc_all = xr.concat(cc_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
            rm_all = xr.concat(rm_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
            ro_all = xr.concat(ro_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
            urmse_all = xr.concat(urmse_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})

            logging.info(f"[Done] Computed spatial efficiency metrics for {len(groups)} {time_group} groups")
            log_message("Completed compute_spatial_efficiency", groups_computed=len(groups))

            # Return the full set of spatial efficiency metrics
            return mb_all, sde_all, cc_all, rm_all, ro_all, urmse_all
        
###############################################################################

###############################################################################

def compute_error_timeseries(model_sst_data: xr.DataArray, sat_sst_data: xr.DataArray, ocean_mask: xr.DataArray) -> pd.DataFrame:
    """
    Compute daily error statistics between model and satellite SST data within a specified basin mask.

    For each time step, this function applies the spatial mask to both model and satellite data,
    computes a suite of statistical metrics on the valid values, and returns a time-indexed
    DataFrame containing these metrics.

    Parameters
    ----------
    model_sst_data : xarray.DataArray
        Sea Surface Temperature (SST) data from the model, with dimensions including 'time', 'lat', and 'lon'.
    sat_sst_data : xarray.DataArray
        Observed satellite SST data, aligned in space and time with the model data.
    basin_mask : xarray.DataArray
        Boolean mask (with dimensions 'lat' and 'lon') indicating the spatial domain (e.g., a basin)
        over which statistics should be computed. True (or 1) values indicate inclusion.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by time (daily), where each row contains statistics computed between
        model and satellite SST for the corresponding day. Columns may include metrics such as:
        - Mean Bias
        - Standard Deviation of Error
        - Correlation Coefficient
        - RMSE
        - Relative metrics, etc.
        (as defined by `compute_stats_single_time`)

    Notes
    -----
    - Assumes input DataArrays are spatially aligned and share a common time coordinate.
    - Invalid (NaN or masked) values are excluded before metric computation.
    - The function `compute_stats_single_time` must return a dictionary or similar structure
      convertible to a DataFrame row.

    Examples
    --------
    >>> df = compute_error_timeseries(model_sst_data, sat_sst_data, basin_mask)
    >>> df.head()
                         mean_bias  rmse   corr
    2000-01-01             -0.12    0.45   0.88
    2000-01-02             -0.15    0.51   0.85
    ...
    """
    # ===== INPUT VALIDATION =====
    for da, name in zip([model_sst_data, sat_sst_data], ["model_sst_data", "sat_sst_data"]):
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"❌ {name} must be an xarray.DataArray ❌")
     
    # Switching mask to DataArray
    basin_mask = xr.DataArray(
        ocean_mask,
        coords={"lat": sat_sst_data["lat"], "lon": sat_sst_data["lon"]},
        dims=("lat", "lon")
    )
    
    # Check spatial dimensions presence and alignment
    if not all(dim in model_sst_data.dims for dim in ('lat', 'lon')):
        raise ValueError("❌ model_sst_data must have 'lat' and 'lon' dimensions ❌")
    if not all(dim in sat_sst_data.dims for dim in ('lat', 'lon')):
        raise ValueError("❌ sat_sst_data must have 'lat' and 'lon' dimensions ❌")
    if not isinstance(basin_mask, xr.DataArray) or basin_mask.shape != (len(sat_sst_data.lat), len(sat_sst_data.lon)):
        raise ValueError("❌ basin_mask must match the shape of sat_sst_data (lat x lon) ❌")
    
    # Check spatial coordinates align
    if not (np.array_equal(model_sst_data['lat'], basin_mask['lat']) and
            np.array_equal(model_sst_data['lon'], basin_mask['lon']) and
            np.array_equal(sat_sst_data['lat'], basin_mask['lat']) and
            np.array_equal(sat_sst_data['lon'], basin_mask['lon'])):
        raise ValueError("❌ Spatial coordinates (lat, lon) of inputs do not align ❌")
    
    # Check time alignment
    if not np.array_equal(model_sst_data['time'], sat_sst_data['time']):
        raise ValueError("❌ Time coordinates of model_sst_data and sat_sst_data must be identical ❌")

    # Apply the basin mask to the model and satellite SST data (mask shape broadcasts over time)
    model_masked = model_sst_data.where(basin_mask)
    sat_masked = sat_sst_data.where(basin_mask)

    # ===== TIME SETUP =====
    # Number of time steps
    n_time = model_masked.sizes['time']

    stats_list = []
    dates = model_sst_data['time'].values

    # ===== COMPUTATION AND LOGGING =====
    with Timer("compute_error_timeseries function"):
        with start_action(action_type="compute_error_timeseries") as action:
            logging.info(f"[Start] Computing error timeseries for {n_time} time steps")
            log_message("Started compute_error_timeseries", n_time=n_time)

            # ===== LOOP THROUGH TIMESTEPS =====
            for t in range(n_time):
                m = model_masked.isel(time=t).values.flatten()
                o = sat_masked.isel(time=t).values.flatten()

                # Remove pairs where either is nan
                valid_mask = ~np.isnan(m) & ~np.isnan(o)
                m_valid = m[valid_mask]
                o_valid = o[valid_mask]

                # If no valid data, fill stats with NaNs or defaults
                if len(m_valid) == 0:
                    stats = {k: np.nan for k in compute_stats_single_time(np.array([0]), np.array([0])).keys()}
                else:
                    stats = compute_stats_single_time(m_valid, o_valid)

                stats_list.append(stats)

            logging.info(f"[Done] Computed error timeseries for all {n_time} time steps")
            log_message("Completed compute_error_timeseries", n_time=n_time)

    # Construct DataFrame indexed by time
    stats_df = pd.DataFrame(stats_list, index=pd.to_datetime(dates))

    return stats_df

###############################################################################

###############################################################################

def compute_stats_single_time(model_slice: np.ndarray, sat_slice: np.ndarray) -> dict:
    """
    Compute error statistics between model and satellite data for a single time slice.

    This function evaluates a set of core statistical metrics comparing model output and satellite
    observations for one timestep, using only valid (non-NaN) paired values.

    Parameters
    ----------
    model_slice : np.ndarray
        1D array of model data values at a single timestep, typically flattened from 2D (lat/lon).
    sat_slice : np.ndarray
        1D array of satellite observation values at the same timestep and spatial extent.

    Returns
    -------
    dict
        Dictionary containing the following metrics:
        - 'mean_bias' : float
            Mean difference between model and satellite values.
        - 'unbiased_rmse' : float
            Root Mean Square Error after removing mean bias.
        - 'std_error' : float
            Standard deviation of the model-satellite difference.
        - 'cross_correlation' : float
            Pearson correlation coefficient between model and satellite.

        If no valid data pairs exist, all values are returned as np.nan.

    Notes
    -----
    - Only pairs where both model and satellite values are finite (non-NaN) are used.
    - This function assumes input arrays are already aligned in space.

    Examples
    --------
    >>> m = np.array([20.1, 19.5, np.nan, 21.0])
    >>> o = np.array([19.8, 19.7, 20.0, 21.1])
    >>> compute_stats_single_time(m, o)
    {'mean_bias': 0.0X, 'unbiased_rmse': 0.0Y, 'std_error': 0.0Z, 'correlation': 0.99}
    """
    # ===== INPUT VALIDATION =====
    if not (isinstance(model_slice, np.ndarray) and isinstance(sat_slice, np.ndarray)):
        raise TypeError("Both model_slice and sat_slice must be numpy arrays.")
    if model_slice.ndim != 1 or sat_slice.ndim != 1:
        raise ValueError("Both model_slice and sat_slice must be 1-dimensional arrays.")
    if model_slice.shape[0] != sat_slice.shape[0]:
        raise ValueError("model_slice and sat_slice must have the same length.")

    with Timer("compute_stats_single_time"):
        # Create a boolean mask where both model and satellite values are not NaN
        valid = ~np.isnan(model_slice) & ~np.isnan(sat_slice)

        # ===== CORNER CASE HANDLING =====
        # If there are no valid paired values, return NaN for all statistics
        if valid.sum() == 0:
            return dict(
                mean_bias=np.nan,
                unbiased_rmse=np.nan,
                std_error=np.nan,
                cross_correlation=np.nan
            )

        # Extract only the valid model and satellite values
        m_valid = model_slice[valid]
        o_valid = sat_slice[valid]

        # Compute and return the statistics using the valid data
        return dict(
            mean_bias=mean_bias(m_valid, o_valid),
            unbiased_rmse=unbiased_rmse(m_valid, o_valid),
            std_error=standard_deviation_error(m_valid, o_valid),
            cross_correlation=cross_correlation(m_valid, o_valid)
        )