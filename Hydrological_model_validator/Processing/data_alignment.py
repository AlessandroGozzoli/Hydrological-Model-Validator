import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union

from .utils import find_key

###############################################################################
def get_valid_mask(mod_vals: np.ndarray, sat_vals: np.ndarray) -> np.ndarray:
    """
    Generate a boolean mask where both model and satellite values are not NaN.

    Parameters
    ----------
    mod_vals : np.ndarray
        Array of model data values.
    sat_vals : np.ndarray
        Array of satellite data values, same shape as mod_vals.

    Returns
    -------
    np.ndarray
        Boolean mask array where True indicates valid (non-NaN) data in both inputs.

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If input arrays do not have the same shape.
    """
    if not isinstance(mod_vals, np.ndarray):
        raise TypeError("mod_vals must be a numpy array")
    if not isinstance(sat_vals, np.ndarray):
        raise TypeError("sat_vals must be a numpy array")
    if mod_vals.shape != sat_vals.shape:
        raise ValueError("mod_vals and sat_vals must have the same shape")

    return ~np.isnan(mod_vals) & ~np.isnan(sat_vals)
###############################################################################

###############################################################################
def get_valid_mask_pandas(mod_series: pd.Series, 
                          sat_series: pd.Series) -> pd.Series:
    """
    Return a boolean pandas Series mask indicating positions where both Series have non-NaN values
    aligned by index.

    Parameters
    ----------
    mod_series : pd.Series
        Model data series.
    sat_series : pd.Series
        Satellite data series.

    Returns
    -------
    pd.Series
        Boolean mask with index equal to the intersection of input indexes.

    Raises
    ------
    TypeError
        If inputs are not pandas Series.
    """
    if not isinstance(mod_series, pd.Series):
        raise TypeError("mod_series must be a pandas Series")
    if not isinstance(sat_series, pd.Series):
        raise TypeError("sat_series must be a pandas Series")

    # Align indexes with intersection
    common_index = mod_series.index.intersection(sat_series.index)
    
    mod_aligned = mod_series.loc[common_index]
    sat_aligned = sat_series.loc[common_index]
    
    mask = (~mod_aligned.isna()) & (~sat_aligned.isna())
    return mask
###############################################################################

###############################################################################
def align_pandas_series(mod_series: pd.Series, 
                        sat_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two pandas Series by index, returning only overlapping non-NaN values.

    Parameters
    ----------
    mod_series : pd.Series
        Model data series.
    sat_series : pd.Series
        Satellite data series.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of aligned numpy arrays (mod_values, sat_values).

    Raises
    ------
    TypeError
        If inputs are not pandas Series.
    """
    if not isinstance(mod_series, pd.Series):
        raise TypeError("mod_series must be a pandas Series")
    if not isinstance(sat_series, pd.Series):
        raise TypeError("sat_series must be a pandas Series")

    mask = get_valid_mask_pandas(mod_series, sat_series)
    mod_aligned = mod_series.loc[mask.index][mask].values
    sat_aligned = sat_series.loc[mask.index][mask].values
    return mod_aligned, sat_aligned
###############################################################################

###############################################################################
def align_numpy_arrays(mod_vals: np.ndarray, 
                       sat_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two numpy arrays by filtering out indices where either array has NaN values.

    Parameters
    ----------
    mod_vals : np.ndarray
        Array of model values.
    sat_vals : np.ndarray
        Array of satellite values, same shape as mod_vals.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of filtered numpy arrays (mod_vals_filtered, sat_vals_filtered) containing only valid data points.

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If input arrays do not have the same shape.
    """
    if not isinstance(mod_vals, np.ndarray):
        raise TypeError("mod_vals must be a numpy array")
    if not isinstance(sat_vals, np.ndarray):
        raise TypeError("sat_vals must be a numpy array")
    if mod_vals.shape != sat_vals.shape:
        raise ValueError("mod_vals and sat_vals must have the same shape")

    mask = get_valid_mask(mod_vals, sat_vals)
    return mod_vals[mask], sat_vals[mask]
###############################################################################

###############################################################################
def get_common_series_by_year(data_dict: Dict[str, Dict[int, pd.Series]]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Extract and align model and satellite data by year.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data, keyed by strings (e.g., 'model', 'satellite'),
        each value is a dict keyed by year containing pandas Series.

    Returns
    -------
    List[Tuple[str, np.ndarray, np.ndarray]]
        List of tuples, each with:
        - year as string,
        - numpy array of model values aligned by time,
        - numpy array of satellite values aligned by time.

    Notes
    -----
    Relies on a helper function `extract_mod_sat_keys(data_dict)` which returns
    (model_key, satellite_key) strings to access data_dict.
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
def get_common_series_by_year_month(data_dict: Dict[str, Dict[Union[int, str], List[np.ndarray]]]) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Extract aligned model and satellite values by year and month.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data keyed by strings (e.g., 'model', 'satellite').
        Each value is a dict keyed by year (int or str), each year containing a list of 12 arrays (months).

    Returns
    -------
    List[Tuple[int, int, np.ndarray, np.ndarray]]
        List of tuples, each with:
        - year (int)
        - month index (0-based int)
        - numpy array of aligned model values for that month
        - numpy array of aligned satellite values for that month
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
                # If month or year key missing, skip
                continue

            valid = get_valid_mask(mod_vals, sat_vals)
            if not np.any(valid):
                # Skip if no valid data points for that month
                continue

            result.append((int(year), month_index, mod_vals[valid], sat_vals[valid]))

    return result
###############################################################################

###############################################################################
def extract_mod_sat_keys(taylor_dict: Dict) -> Tuple[str, str]:
    """
    Extract model and satellite keys from a dictionary using common substrings.

    Parameters
    ----------
    taylor_dict : dict
        Dictionary containing model and satellite data keys.

    Returns
    -------
    Tuple[str, str]
        Tuple containing (model_key, satellite_key).

    Raises
    ------
    ValueError
        If model or satellite keys cannot be found in the dictionary.
    """
    mod_key = find_key(taylor_dict, ['mod', 'model', 'predicted'])
    sat_key = find_key(taylor_dict, ['sat', 'satellite', 'observed'])

    if mod_key is None or sat_key is None:
        raise ValueError("taylor_dict must contain keys for model and satellite data.")

    return mod_key, sat_key
###############################################################################

###############################################################################     
def gather_monthly_data_across_years(data_dict: Dict[str, Dict[int, List[Union[np.ndarray, list]]]],
                                     key: str,
                                     month_idx: int) -> np.ndarray:
    """
    Gather, flatten, concatenate, and remove NaNs from all years' data for a given key and month index.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys (e.g. 'mod', 'sat'), each mapping to another dict keyed by year,
        where each year maps to a list of 12 monthly arrays/lists.
    key : str
        Key in data_dict to select model or satellite data.
    month_idx : int
        Month index (0=January, ..., 11=December).

    Returns
    -------
    np.ndarray
        1D array of valid (non-NaN) data concatenated across all years for the specified month.

    Raises
    ------
    ValueError
        If inputs are invalid or data for a year is missing or incomplete.
    """
    if not isinstance(data_dict, dict):
        raise ValueError("data_dict must be a dictionary")
    if key not in data_dict:
        raise ValueError(f"Key '{key}' not found in data_dict")
    if not (isinstance(month_idx, int) and 0 <= month_idx <= 11):
        raise ValueError("month_idx must be an integer between 0 and 11")

    values = []
    for year, year_data in data_dict[key].items():
        if not isinstance(year_data, (list, tuple)) or len(year_data) != 12:
            raise ValueError(f"Year {year} does not have 12 months of data")
        month_data = np.asarray(year_data[month_idx]).flatten()
        values.append(month_data)

    if values:
        all_data = np.concatenate(values)
        return all_data[~np.isnan(all_data)]
    else:
        return np.array([])
###############################################################################