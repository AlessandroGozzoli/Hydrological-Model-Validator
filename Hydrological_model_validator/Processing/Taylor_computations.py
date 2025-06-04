import skill_metrics as sm
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional

from .data_alignment import get_common_series_by_year, get_valid_mask, extract_mod_sat_keys

###############################################################################
def compute_taylor_stat_tuple(mod_values: np.ndarray, 
                              sat_values: np.ndarray, 
                              label: str) -> Tuple[str, float, float, float]:
    """
    Compute Taylor statistics (standard deviation, centered RMSD, correlation coefficient) 
    for model and satellite data arrays.

    Parameters
    ----------
    mod_values : np.ndarray
        Array of model data values.
    sat_values : np.ndarray
        Array of satellite data values.
    label : str
        Label associated with these data (e.g., year or month).

    Returns
    -------
    tuple
        A tuple containing:
        - label (str)
        - model standard deviation (float)
        - centered RMSD (float)
        - correlation coefficient (float)
    """
    if mod_values.size == 0 or sat_values.size == 0:
        raise ValueError("Input arrays must not be empty.")

    # Mask to keep only finite pairs
    valid_mask = np.isfinite(mod_values) & np.isfinite(sat_values)
    if not np.any(valid_mask):
        raise ValueError("No valid finite data pairs found in input arrays.")

    mod_valid = mod_values[valid_mask]
    sat_valid = sat_values[valid_mask]

    stats = sm.taylor_statistics(mod_valid, sat_valid, 'data')
    return (label, stats['sdev'][1], stats['crmsd'][1], stats['ccoef'][1])
###############################################################################

###############################################################################
def compute_std_reference(sat_data_by_year: Dict[Union[int, str], List[Union[np.ndarray, list]]],
                          years: List[Union[int, str]],
                          month_index: int) -> float:
    """
    Compute reference standard deviation for satellite data of a given month over all years.

    Parameters
    ----------
    sat_data_by_year : dict
        Dictionary keyed by year containing lists of monthly satellite data arrays.
    years : list
        List of years to include.
    month_index : int
        Month index (0-based).

    Returns
    -------
    float
        Standard deviation of concatenated satellite data for the specified month across all years.

    Raises
    ------
    ValueError
        If month_index is not between 0 and 11.
        If no valid data found for the specified month across the years.
    """
    if not (0 <= month_index <= 11):
        raise ValueError(f"month_index must be between 0 and 11, got {month_index}")

    monthly_data = []
    for year in years:
        if year in sat_data_by_year and month_index < len(sat_data_by_year[year]):
            arr = np.asarray(sat_data_by_year[year][month_index]).flatten()
            if arr.size == 0:
                raise ValueError(f"Empty data array for year {year}, month {month_index}")
            monthly_data.append(arr)

    if not monthly_data:
        raise ValueError(f"No valid satellite data found for month index {month_index} across given years.")

    all_monthly_sat = np.concatenate(monthly_data)
    return np.nanstd(all_monthly_sat)
###############################################################################

###############################################################################
def compute_norm_taylor_stats(mod_vals: np.ndarray, 
                              sat_vals: np.ndarray, 
                              std_ref: float) -> Optional[Dict[str, float]]:
    """
    Compute normalized Taylor statistics for one month-year pair.

    Parameters
    ----------
    mod_vals : np.ndarray
        Model data values.
    sat_vals : np.ndarray
        Satellite data values.
    std_ref : float
        Reference standard deviation to normalize the statistics.

    Returns
    -------
    dict or None
        Dictionary with normalized 'sdev', 'crmsd', and 'ccoef' if valid data exists, else None.

    Raises
    ------
    ValueError
        If std_ref is zero or negative.
    """
    if std_ref <= 0:
        raise ValueError(f"std_ref must be positive, got {std_ref}")

    valid = get_valid_mask(mod_vals, sat_vals)
    if not np.any(valid):
        return None

    stats = sm.taylor_statistics(mod_vals[valid], sat_vals[valid], 'data')

    return {
        "sdev": stats['sdev'][1] / std_ref,
        "crmsd": stats['crmsd'][1] / std_ref,
        "ccoef": stats['ccoef'][1],
    }
###############################################################################

###############################################################################
def build_all_points(data_dict: Dict[Union[str, int], Dict[int, List[Union[np.ndarray, list]]]]) -> Tuple[pd.DataFrame, List[Union[str, int]]]:
    """
    Build a DataFrame of normalized Taylor statistics points for all months and years,
    including reference points per month.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data by year and month.

    Returns
    -------
    Tuple[pandas.DataFrame, list]
        DataFrame with columns ['sdev', 'crmsd', 'ccoef', 'month', 'year'] and
        list of years found in the data.
    """
    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    model_data_by_year = data_dict[mod_key]
    sat_data_by_year = data_dict[sat_key]
    years = sorted(sat_data_by_year.keys())

    # Precompute std_ref per month
    std_refs = {
        month_idx: compute_std_reference(sat_data_by_year, years, month_idx)
        for month_idx in range(12)
    }

    all_points = []
    for month_idx in range(12):
        std_ref = std_refs[month_idx]
        if std_ref <= 0 or np.isnan(std_ref):
            # Skip months with invalid std reference
            continue

        # Reference point normalized
        all_points.append({
            "sdev": 1.0,
            "crmsd": 0.0,
            "ccoef": 1.0,
            "month": month_idx,
            "year": "Ref"
        })

        for year in years:
            try:
                mod_vals = np.asarray(model_data_by_year[year][month_idx])
                sat_vals = np.asarray(sat_data_by_year[year][month_idx])
            except (IndexError, KeyError):
                continue

            norm_stats = compute_norm_taylor_stats(mod_vals, sat_vals, std_ref)
            if norm_stats is None:
                continue

            all_points.append({
                **norm_stats,
                "month": month_idx,
                "year": year
            })

    return pd.DataFrame(all_points), years
###############################################################################

###############################################################################
def compute_yearly_taylor_stats(data_dict: Dict[Union[str, int], Dict[int, List[Union[np.ndarray, list]]]]) -> Tuple[List[Tuple[str, float, float, float]], float]:
    """
    Compute Taylor statistics for each year using model and satellite data from data_dict.
    Also computes the global standard deviation of all satellite data.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data organized by year and month.

    Returns
    -------
    tuple
        (yearly_stats, std_ref) where
        yearly_stats is a list of tuples (year, sdev, crmsd, ccoef),
        std_ref is the global satellite standard deviation (float).
    """
    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    sat_data_by_year = data_dict[sat_key]

    # Flatten and concatenate all satellite data across years and months
    all_sat_data = np.concatenate([
        np.asarray(month_array).flatten()
        for year_data in sat_data_by_year.values()
        for month_array in year_data
        if month_array is not None
    ])

    std_ref = np.nanstd(all_sat_data)
    if np.isnan(std_ref) or std_ref == 0:
        raise ValueError("Global satellite standard deviation is zero or NaN, invalid data.")

    aligned_data = get_common_series_by_year(data_dict)
    yearly_stats = [
        compute_taylor_stat_tuple(mod_values, sat_values, year)
        for year, mod_values, sat_values in aligned_data
    ]

    return yearly_stats, std_ref
###############################################################################