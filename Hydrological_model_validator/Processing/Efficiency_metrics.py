import numpy as np
import xarray as xr
import pandas as pd

from .stats_math_utils import (mean_bias, 
                               standard_deviation_error, 
                               std_dev,
                               cross_correlation,
                               unbiased_rmse)

###############################################################################
def r_squared(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate coefficient of determination (r²) between observed and predicted data.

    Parameters
    ----------
    obs : np.ndarray
        Observed values.
    pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        r² value, or np.nan if insufficient data.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)

    if np.sum(mask) < 2:
        return np.nan

    corr = np.corrcoef(obs[mask], pred[mask])[0, 1]
    return corr ** 2
###############################################################################

###############################################################################
def monthly_r_squared(data_dict: dict) -> list[float]:
    """
    Compute monthly r² values between model and satellite datasets over multiple years.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data with keys containing 'mod' and 'sat'.

    Returns
    -------
    list of float
        List of 12 r² values, one per month.
    """
    model_key = next((k for k in data_dict if 'mod' in k.lower()), None)
    sat_key = next((k for k in data_dict if 'sat' in k.lower()), None)

    if model_key is None or sat_key is None:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = data_dict[model_key]
    sat_monthly = data_dict[sat_key]

    years = list(mod_monthly.keys())
    r2_monthly = []

    for month in range(12):
        mod_arrays = [np.asarray(mod_monthly[year][month]).ravel() for year in years]
        sat_arrays = [np.asarray(sat_monthly[year][month]).ravel() for year in years]

        mod_concat = np.concatenate(mod_arrays)
        sat_concat = np.concatenate(sat_arrays)

        valid_mask = ~np.isnan(mod_concat) & ~np.isnan(sat_concat)

        if np.any(valid_mask):
            r2 = r_squared(mod_concat[valid_mask], sat_concat[valid_mask])
        else:
            r2 = np.nan

        print(f"Month {month+1}: R² = {r2}")
        r2_monthly.append(r2)

    return r2_monthly
###############################################################################

###############################################################################
def weighted_r_squared(obs, pred):
    """
    Compute weighted coefficient of determination (wr²).

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        Weighted R-squared value or np.nan if insufficient data.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    
    if np.sum(mask) < 2:
        return np.nan
    
    x = obs[mask]
    y = pred[mask]

    # Compute standard R² on valid data
    r2 = r_squared(x, y)
    
    # Fit slope of pred vs obs
    slope, intercept = np.polyfit(x, y, 1)

    # Use absolute value of slope, clipped to [0, 1]
    slope_abs = abs(slope)
    weight = slope_abs if slope_abs <= 1 else 1 / slope_abs

    # To avoid zero weight when slope is close to zero, add a small floor
    weight = max(weight, 0.1)

    weighted_r2 = weight * r2

    return weighted_r2
###############################################################################

###############################################################################    
def monthly_weighted_r_squared(dictionary):
    """
    Compute weighted R² (wr²) for each month using paired model and satellite data.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys with 'mod' and 'sat' for model and satellite data.
        Each key maps to a dict of years, each year is a list/array of 12 monthly arrays.

    Returns
    -------
    list of float
        Weighted R² values for each month (length 12).

    Raises
    ------
    KeyError
        If no model or satellite keys found in dictionary.
    """
    from .Efficiency_metrics import weighted_r_squared

    model_keys = [k for k in dictionary if 'mod' in k.lower()]
    sat_keys = [k for k in dictionary if 'sat' in k.lower()]

    if not model_keys or not sat_keys:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_keys[0]]
    sat_monthly = dictionary[sat_keys[0]]

    years = list(mod_monthly.keys())

    wr2_monthly = []

    for month in range(12):
        # Gather all model and satellite data for this month across years
        mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
        sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

        valid_mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

        if np.any(valid_mask):
            wr2 = weighted_r_squared(sat_all[valid_mask], mod_all[valid_mask])
        else:
            wr2 = np.nan

        wr2_monthly.append(wr2)

    return wr2_monthly
###############################################################################

############################################################################### 
def nse(obs, pred):
    """
    Compute Nash–Sutcliffe Efficiency (NSE) between observations and predictions.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        NSE value, or np.nan if insufficient valid data.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    
    if np.sum(mask) < 2:
        return np.nan
    
    obs_masked = obs[mask]
    pred_masked = pred[mask]
    
    numerator = np.sum((obs_masked - pred_masked) ** 2)
    denominator = np.sum((obs_masked - np.mean(obs_masked)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    return 1 - numerator / denominator
###############################################################################

############################################################################### 
def monthly_nse(dictionary):
    """
    Compute monthly Nash–Sutcliffe Efficiency (NSE) from paired model and satellite data.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each maps to a dict of years, each year a list/array of 12 monthly arrays.

    Returns
    -------
    list of float
        NSE values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing.
    """
    from .Efficiency_metrics import nse

    model_keys = [k for k in dictionary if 'mod' in k.lower()]
    sat_keys = [k for k in dictionary if 'sat' in k.lower()]

    if not model_keys or not sat_keys:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_keys[0]]
    sat_monthly = dictionary[sat_keys[0]]

    years = list(mod_monthly.keys())
    nse_monthly = []

    for month in range(12):
        # Concatenate all data for this month across all years
        mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
        sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

        # Mask NaNs in both arrays
        valid_mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

        if np.any(valid_mask):
            nse_val = nse(sat_all[valid_mask], mod_all[valid_mask])
        else:
            nse_val = np.nan

        nse_monthly.append(nse_val)

    return nse_monthly
###############################################################################

############################################################################### 
def index_of_agreement(obs, pred):
    """
    Calculate the Index of Agreement (d) between observations and predictions.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        Index of Agreement, or np.nan if insufficient valid data or division by zero.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)

    if np.sum(mask) < 2:
        return np.nan

    obs_masked = obs[mask]
    pred_masked = pred[mask]

    numerator = np.sum((obs_masked - pred_masked) ** 2)
    denominator = np.sum((np.abs(pred_masked - np.mean(obs_masked)) + np.abs(obs_masked - np.mean(obs_masked))) ** 2)

    if denominator == 0:
        return np.nan
    
    return 1 - numerator / denominator
###############################################################################

############################################################################### 
def monthly_index_of_agreement(dictionary):
    """
    Compute monthly Index of Agreement (d) from paired model and satellite data.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each maps to a dict of years, each year a list/array of 12 monthly arrays.

    Returns
    -------
    list of float
        Index of Agreement values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing.
    """
    from .Efficiency_metrics import index_of_agreement

    model_keys = [k for k in dictionary if 'mod' in k.lower()]
    sat_keys = [k for k in dictionary if 'sat' in k.lower()]

    if not model_keys or not sat_keys:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_keys[0]]
    sat_monthly = dictionary[sat_keys[0]]

    years = list(mod_monthly.keys())
    d_monthly = []

    for month in range(12):
        # Concatenate all data for this month across all years, flattening arrays
        mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
        sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

        # Mask out NaNs in both arrays
        valid_mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

        if np.any(valid_mask):
            d_val = index_of_agreement(sat_all[valid_mask], mod_all[valid_mask])
        else:
            d_val = np.nan

        d_monthly.append(d_val)

    return d_monthly
###############################################################################

############################################################################### 
def ln_nse(obs, pred):
    """
    Compute Nash–Sutcliffe Efficiency (NSE) on logarithmic values.

    Note: obs and pred must contain only positive values.

    Parameters
    ----------
    obs : array-like
        Observed values (positive).
    pred : array-like
        Predicted values (positive).

    Returns
    -------
    float
        Logarithmic NSE value, or np.nan if insufficient valid data or invalid values.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)

    # Valid mask: non-NaN and positive values only
    mask = (~np.isnan(obs)) & (~np.isnan(pred)) & (obs > 0) & (pred > 0)

    if np.sum(mask) < 2:
        return np.nan

    log_obs = np.log(obs[mask])
    log_pred = np.log(pred[mask])

    numerator = np.sum((log_obs - log_pred) ** 2)
    denominator = np.sum((log_obs - np.mean(log_obs)) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - numerator / denominator
###############################################################################

############################################################################### 
def monthly_ln_nse(dictionary):
    """
    Compute monthly logarithmic NSE (ln NSE) from paired model and satellite data.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each maps to a dict of years, each year a list/array of 12 monthly arrays.

    Returns
    -------
    list of float
        Logarithmic NSE values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing.
    """
    from .Efficiency_metrics import ln_nse

    model_keys = [k for k in dictionary if 'mod' in k.lower()]
    sat_keys = [k for k in dictionary if 'sat' in k.lower()]

    if not model_keys or not sat_keys:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_keys[0]]
    sat_monthly = dictionary[sat_keys[0]]

    years = list(mod_monthly.keys())
    ln_nse_monthly = []

    for month in range(12):
        mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
        sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

        # Mask for valid data: non-NaN and strictly positive
        valid_mask = (~np.isnan(mod_all) & ~np.isnan(sat_all) & (mod_all > 0) & (sat_all > 0))

        if np.any(valid_mask):
            ln_nse_val = ln_nse(sat_all[valid_mask], mod_all[valid_mask])
        else:
            ln_nse_val = np.nan

        ln_nse_monthly.append(ln_nse_val)

    return ln_nse_monthly
###############################################################################

############################################################################### 
def nse_j(obs, pred, j=1):
    """
    Compute modified Nash–Sutcliffe Efficiency (E_j) for arbitrary exponent j.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.
    j : float, optional
        Exponent for the absolute difference (default is 1).

    Returns
    -------
    float
        Modified NSE value, or np.nan if insufficient valid data or zero denominator.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)

    if np.sum(mask) < 2:
        return np.nan

    obs_masked = obs[mask]
    pred_masked = pred[mask]

    numerator = np.sum(np.abs(obs_masked - pred_masked) ** j)
    denominator = np.sum(np.abs(obs_masked - np.mean(obs_masked)) ** j)

    if denominator == 0:
        return np.nan

    return 1 - numerator / denominator
###############################################################################

############################################################################### 
def monthly_nse_j(dictionary, j=1):
    """
    Compute monthly modified NSE (E_j) from paired model and satellite data.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each maps to a dict of years, each year a list/array of 12 monthly arrays.
    j : float, optional
        Exponent parameter for modified NSE (default 1).

    Returns
    -------
    list of float
        Modified NSE values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing.
    """
    from .Efficiency_metrics import nse_j

    model_keys = [k for k in dictionary if 'mod' in k.lower()]
    sat_keys = [k for k in dictionary if 'sat' in k.lower()]

    if not model_keys or not sat_keys:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_keys[0]]
    sat_monthly = dictionary[sat_keys[0]]

    nse_j_monthly = []
    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data)
            mod_vals.extend(mod_data[mask])
            sat_vals.extend(sat_data[mask])

        nse_j_monthly.append(nse_j(sat_vals, mod_vals, j=j))

    return nse_j_monthly
###############################################################################

############################################################################### 
def index_of_agreement_j(obs, pred, j=1):
    """
    Compute modified Index of Agreement (d_j) with exponent j.

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.
    j : float, optional
        Exponent parameter (default is 1).

    Returns
    -------
    float
        Modified index of agreement value, or np.nan if invalid input or zero denominator.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)

    if np.sum(mask) < 2:
        return np.nan

    obs_masked = obs[mask]
    pred_masked = pred[mask]

    numerator = np.sum(np.abs(obs_masked - pred_masked) ** j)
    denominator = np.sum((np.abs(pred_masked - np.mean(obs_masked)) + np.abs(obs_masked - np.mean(obs_masked))) ** j)

    if denominator == 0:
        return np.nan

    return 1 - numerator / denominator
###############################################################################

############################################################################### 
def monthly_index_of_agreement_j(dictionary, j=1):
    """
    Compute monthly modified Index of Agreement (d_j) with exponent j from paired model and satellite data.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each maps to a dict of years, each year a list/array of 12 monthly arrays.
    j : float, optional
        Exponent parameter for the modified Index of Agreement (default 1).

    Returns
    -------
    list of float
        Modified Index of Agreement values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing.
    """
    from .Efficiency_metrics import index_of_agreement_j

    model_keys = [k for k in dictionary if 'mod' in k.lower()]
    sat_keys = [k for k in dictionary if 'sat' in k.lower()]

    if not model_keys or not sat_keys:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_keys[0]]
    sat_monthly = dictionary[sat_keys[0]]

    years = list(mod_monthly.keys())
    d_j_monthly = []

    for month in range(12):
        mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
        sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

        mask = ~np.isnan(mod_all) & ~np.isnan(sat_all)

        if np.any(mask):
            val = index_of_agreement_j(sat_all[mask], mod_all[mask], j=j)
        else:
            val = np.nan

        d_j_monthly.append(val)

    return d_j_monthly
###############################################################################

############################################################################### 
def relative_nse(obs, pred):
    """
    Compute the Relative Nash–Sutcliffe Efficiency (Relative NSE).

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        Relative NSE value, or np.nan if calculation is invalid.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    
    # Mask valid data and avoid division by zero in obs
    mask = ~np.isnan(obs) & ~np.isnan(pred) & (obs != 0)
    
    if np.sum(mask) < 2:
        return np.nan
    
    obs_masked = obs[mask]
    pred_masked = pred[mask]
    obs_mean = np.mean(obs_masked)
    
    numerator = np.sum(((obs_masked - pred_masked) / obs_masked) ** 2)
    denominator = np.sum(((obs_masked - obs_mean) / obs_mean) ** 2)
    
    if denominator == 0:
        return np.nan
    
    return 1 - numerator / denominator
###############################################################################

############################################################################### 
def monthly_relative_nse(dictionary):
    """
    Compute monthly Relative Nash–Sutcliffe Efficiency (relative NSE) from paired model and satellite data.

    Parameters
    ----------
    dictionary : dict
        Dictionary with keys containing 'mod' and 'sat' for model and satellite data.
        Each maps to a dict of years, each year a list/array of 12 monthly arrays.

    Returns
    -------
    list of float
        Relative NSE values for each month (length 12).

    Raises
    ------
    KeyError
        If model or satellite keys are missing.
    """
    from .Efficiency_metrics import relative_nse

    model_keys = [k for k in dictionary if 'mod' in k.lower()]
    sat_keys = [k for k in dictionary if 'sat' in k.lower()]

    if not model_keys or not sat_keys:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_keys[0]]
    sat_monthly = dictionary[sat_keys[0]]

    years = list(mod_monthly.keys())
    e_rel_monthly = []

    for month in range(12):
        mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
        sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

        mask = (~np.isnan(mod_all)) & (~np.isnan(sat_all)) & (sat_all != 0)

        if np.any(mask):
            val = relative_nse(sat_all[mask], mod_all[mask])
        else:
            val = np.nan

        e_rel_monthly.append(val)

    return e_rel_monthly
###############################################################################

############################################################################### 
def relative_index_of_agreement(obs, pred):
    """
    Compute the Relative Index of Agreement (d_rel).

    Parameters
    ----------
    obs : array-like
        Observed values.
    pred : array-like
        Predicted values.

    Returns
    -------
    float
        Relative Index of Agreement value, or np.nan if calculation is invalid.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)

    mask = ~np.isnan(obs) & ~np.isnan(pred) & (obs != 0)
    if np.sum(mask) < 2:
        return np.nan

    obs_masked = obs[mask]
    pred_masked = pred[mask]
    obs_mean = np.mean(obs_masked)

    numerator = np.sum(((obs_masked - pred_masked) / obs_masked) ** 2)
    denominator = np.sum(
        ((np.abs(pred_masked - obs_mean) + np.abs(obs_masked - obs_mean)) / obs_mean) ** 2
    )

    if np.allclose(obs_masked, obs_mean):
        return np.nan

    return 1 - numerator / denominator
###############################################################################

############################################################################### 
def monthly_relative_index_of_agreement(dictionary):
    """
    Calculate the Relative Index of Agreement (d_rel) for each month across multiple years.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing model and satellite monthly data structured as:
        {
            'mod...': {year: [month_0_data, ..., month_11_data], ...},
            'sat...': {year: [month_0_data, ..., month_11_data], ...}
        }

    Returns
    -------
    list of float
        List of 12 relative index of agreement values, one for each month.
    """
    from .Efficiency_metrics import relative_index_of_agreement
    import numpy as np

    model_key = next((k for k in dictionary if 'mod' in k.lower()), None)
    sat_key = next((k for k in dictionary if 'sat' in k.lower()), None)

    if model_key is None or sat_key is None:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key]
    sat_monthly = dictionary[sat_key]

    years = list(mod_monthly.keys())
    d_rel_monthly = []

    for month in range(12):
        mod_all = np.concatenate([np.asarray(mod_monthly[year][month]).ravel() for year in years])
        sat_all = np.concatenate([np.asarray(sat_monthly[year][month]).ravel() for year in years])

        mask = (~np.isnan(mod_all)) & (~np.isnan(sat_all)) & (sat_all != 0)

        if np.any(mask):
            d_rel = relative_index_of_agreement(mod_all[mask], sat_all[mask])
        else:
            d_rel = np.nan

        d_rel_monthly.append(d_rel)

    return d_rel_monthly
###############################################################################

###############################################################################
def compute_spatial_efficiency(model_da, sat_da, time_group="month"):
    if time_group == "month":
        groups = range(1, 13)
        time_sel = 'month'
    elif time_group == "year":
        groups = sorted(np.unique(model_da['time.year'].values))
        time_sel = 'year'
    else:
        raise ValueError(f"Invalid time_group '{time_group}', must be 'month' or 'year'")
    
    def compute_metrics_for_group(group):
        m_sel = model_da.sel(time=model_da['time.' + time_sel] == group)
        o_sel = sat_da.sel(time=sat_da['time.' + time_sel] == group)
        
        return (
            mean_bias(m_sel, o_sel),
            standard_deviation_error(m_sel, o_sel),
            cross_correlation(m_sel, o_sel),
            std_dev(m_sel),
            std_dev(o_sel),
            unbiased_rmse(m_sel, o_sel),
        )
    
    results = list(map(compute_metrics_for_group, groups))
    mb_maps, sde_maps, cc_maps, rm_maps, ro_maps, urmse_maps = zip(*results)
    
    dim_name = time_group
    coord_vals = groups
    
    mb_all = xr.concat(mb_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
    sde_all = xr.concat(sde_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
    cc_all = xr.concat(cc_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
    rm_all = xr.concat(rm_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
    ro_all = xr.concat(ro_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
    urmse_all = xr.concat(urmse_maps, dim=dim_name).assign_coords(**{dim_name: coord_vals})
    
    return mb_all, sde_all, cc_all, rm_all, ro_all, urmse_all
###############################################################################

###############################################################################
def compute_error_timeseries(model_sst_data: xr.DataArray, sat_sst_data: xr.DataArray, basin_mask: xr.DataArray) -> pd.DataFrame:
    """
    Compute daily statistics between model and satellite SST over the basin mask.

    Returns a DataFrame indexed by time with columns for each metric.
    """
    # Expand mask to 3D (time, lat, lon)

    # Mask the datasets
    model_masked = model_sst_data.where(basin_mask)
    sat_masked = sat_sst_data.where(basin_mask)

    n_time = model_masked.sizes['time']

    stats_list = []
    for t in range(n_time):
        m = model_masked.isel(time=t).values.flatten()
        o = sat_masked.isel(time=t).values.flatten()
        stats_list.append(compute_stats_single_time(m, o))

    dates = model_sst_data.time.values
    stats_df = pd.DataFrame(stats_list, index=pd.to_datetime(dates))
    return stats_df

###############################################################################

###############################################################################
def compute_stats_single_time(model_slice: np.ndarray, sat_slice: np.ndarray) -> dict:
    """
    Compute statistics between model and satellite for a single time slice.

    Returns a dict with mean_bias, unbiased_rmse, std_error, correlation.
    """
    valid = ~np.isnan(model_slice) & ~np.isnan(sat_slice)
    if valid.sum() == 0:
        return dict(mean_bias=np.nan, unbiased_rmse=np.nan, std_error=np.nan, correlation=np.nan)

    m_valid = model_slice[valid]
    o_valid = sat_slice[valid]

    return dict(
        mean_bias=mean_bias(m_valid, o_valid),
        unbiased_rmse=unbiased_rmse(m_valid, o_valid),
        std_error=standard_deviation_error(m_valid, o_valid),
        correlation=cross_correlation(m_valid, o_valid)
    )