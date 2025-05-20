import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union

from .utils import find_key

###############################################################################
def get_valid_mask(mod_vals: np.ndarray, sat_vals: np.ndarray) -> np.ndarray:
    """
    Return boolean mask where both model and satellite values are not NaN.

    Args:
        mod_vals (np.ndarray): Model values.
        sat_vals (np.ndarray): Satellite values.

    Returns:
        np.ndarray: Boolean mask array.
    """
    assert isinstance(mod_vals, np.ndarray)
    assert isinstance(sat_vals, np.ndarray)
    assert mod_vals.shape == sat_vals.shape, "mod_vals and sat_vals must have the same shape"
    return ~np.isnan(mod_vals) & ~np.isnan(sat_vals)
###############################################################################

###############################################################################
def get_valid_mask_pandas(
    mod_series: pd.Series, 
    sat_series: pd.Series
) -> pd.Series:
    """
    Return a boolean pandas Series mask indicating positions where both Series have non-NaN values
    aligned by index.

    Args:
        mod_series (pd.Series): Model data series.
        sat_series (pd.Series): Satellite data series.

    Returns:
        pd.Series: Boolean mask with index equal to the intersection of input indexes.
    """
    # Align indexes with intersection
    common_index = mod_series.index.intersection(sat_series.index)
    
    mod_aligned = mod_series.loc[common_index]
    sat_aligned = sat_series.loc[common_index]
    
    mask = (~mod_aligned.isna()) & (~sat_aligned.isna())
    return mask
###############################################################################

###############################################################################
def align_pandas_series(
    mod_series: pd.Series, 
    sat_series: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two pandas Series by index, keeping only overlapping non-NaN data.

    Returns:
        Tuple of np.ndarrays: (mod_values, sat_values)
    """
    mask = get_valid_mask_pandas(mod_series, sat_series)
    mod_aligned = mod_series.loc[mask.index][mask].values
    sat_aligned = sat_series.loc[mask.index][mask].values
    return mod_aligned, sat_aligned
###############################################################################

###############################################################################
def align_numpy_arrays(
    mod_vals: np.ndarray, 
    sat_vals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two numpy arrays by filtering out indices where either array is NaN.

    Returns:
        Tuple of np.ndarrays: (mod_values, sat_values)
    """
    mask = get_valid_mask(mod_vals, sat_vals)
    return mod_vals[mask], sat_vals[mask]
###############################################################################

###############################################################################
def get_common_series_by_year(
    data_dict: dict
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Extract and align model and satellite data by year.

    Args:
        data_dict (dict): Dictionary with model and satellite data keyed by year.

    Returns:
        List of tuples: (year as string, mod_values, sat_values)
    """
    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    common_series = []

    for year in sorted(data_dict[mod_key].keys()):
        mod_series = data_dict[mod_key][year].dropna()
        sat_series = data_dict[sat_key][year].dropna()

        combined = mod_series.to_frame('mod').join(sat_series.to_frame('sat'), how='inner').dropna()
        if combined.empty:
            print(f"Warning: No overlapping data for year {year}. Skipping.")
            continue

        common_series.append((str(year), combined['mod'].values, combined['sat'].values))

    return common_series
###############################################################################

###############################################################################
def get_common_series_by_year_month(
    data_dict: dict
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Extract aligned model and satellite values by year and month.

    Args:
        data_dict (dict): Dictionary with model and satellite data keyed by year and month.

    Returns:
        List of tuples: (year, month, mod_values, sat_values)
    """
    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    result = []

    years = sorted(data_dict[mod_key].keys())
    for year in years:
        for month_index in range(12):
            try:
                mod_vals = np.asarray(data_dict[mod_key][year][month_index])
                sat_vals = np.asarray(data_dict[sat_key][year][month_index])
            except (IndexError, KeyError):
                continue

            valid = get_valid_mask(mod_vals, sat_vals)
            if not np.any(valid):
                continue

            result.append((year, month_index, mod_vals[valid], sat_vals[valid]))

    return result
###############################################################################

###############################################################################
def extract_mod_sat_keys(taylor_dict: dict) -> Tuple[str, str]:
    """
    Extract model and satellite keys from a dictionary using common substrings.

    Args:
        taylor_dict (dict): Dictionary containing model and satellite data.

    Returns:
        Tuple[str, str]: (model_key, satellite_key)

    Raises:
        ValueError: If keys cannot be found.
    """
    mod_key = find_key(taylor_dict, ['mod', 'model', 'predicted'])
    sat_key = find_key(taylor_dict, ['sat', 'satellite', 'observed'])

    if mod_key is None or sat_key is None:
        raise ValueError("taylor_dict must contain keys for 'mod' and 'sat' data.")

    return mod_key, sat_key
###############################################################################

###############################################################################     
def gather_monthly_data_across_years(
    data_dict: Dict[int, List[Union[np.ndarray, list]]],
    key: str,
    month_idx: int
) -> np.ndarray:
    """
    Gathers, flattens, concatenates, and removes NaNs from all years' data
    for a specific key and month index.

    Args:
        data_dict (dict): Dictionary with {year: [12-month arrays]}
        key (str): Dictionary key for model or satellite
        month_idx (int): Month index (0 = January, 11 = December)

    Returns:
        np.ndarray: Cleaned, 1D array of all valid data for that month

    Raises:
        AssertionError: If inputs are invalid or data is missing.
    """
    assert isinstance(data_dict, dict), "data_dict must be a dictionary"
    assert key in data_dict, f"Key '{key}' not found in data_dict"
    assert isinstance(month_idx, int) and 0 <= month_idx <= 11, "month_idx must be an integer between 0 and 11"

    values = []
    for year in data_dict[key]:
        year_data = data_dict[key][year]
        assert len(year_data) == 12, f"Year {year} does not have 12 months of data"
        month_data = np.asarray(year_data[month_idx]).flatten()
        values.append(month_data)

    all_data = np.concatenate(values) if values else np.array([])
    return all_data[~np.isnan(all_data)]
###############################################################################