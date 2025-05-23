from typing import Tuple, List, Dict, Optional
from itertools import starmap, chain
import numpy as np
import skill_metrcis as sm

from ..Processing.data_alignment import (get_common_series_by_year, 
                                         get_common_series_by_year_month)
from ..Processing.stats_math_utils import round_up_to_nearest 

###############################################################################
def compute_single_target_stat(year: str, 
                               mod: np.ndarray, 
                               sat: np.ndarray) -> Optional[Tuple[float, float, float, str]]:
    """
    Compute normalized bias, CRMSD, and RMSD for a single year of data.

    Parameters
    ----------
    year : str
        Year label.
    mod : np.ndarray
        Model values.
    sat : np.ndarray
        Satellite values.

    Returns
    -------
    Optional[Tuple[float, float, float, str]]
        Normalized bias, CRMSD, RMSD, and year label. Returns None if reference std is zero.
    """
    ref_std = np.std(sat, ddof=1)
    if ref_std == 0:
        print(f"Warning: Zero standard deviation in satellite data for {year}. Skipping.")
        return None

    stats = sm.target_statistics(mod, sat, 'data')
    return (
        stats['bias'] / ref_std,
        stats['crmsd'] / ref_std,
        stats['rmsd'] / ref_std,
        year
    )
###############################################################################

###############################################################################
def compute_single_month_target_stat(year: int,
                                     month: int,
                                     mod: np.ndarray,
                                     sat: np.ndarray) -> Optional[Tuple[float, float, float, str]]:
    """
    Compute normalized bias, CRMSD, and RMSD for a single year-month.

    Parameters
    ----------
    year : int
        Year label.
    month : int
        Month index (0–11).
    mod : np.ndarray
        Model values.
    sat : np.ndarray
        Satellite values.

    Returns
    -------
    Optional[Tuple[float, float, float, str]]
        Normalized bias, CRMSD, RMSD, and label (e.g., "2001"). Returns None if std = 0.
    """
    ref_std = np.std(sat, ddof=1)
    if ref_std == 0:
        print(f"Warning: Zero standard deviation in satellite data for {year}, month {month}. Skipping.")
        return None

    stats = sm.target_statistics(mod, sat, 'data')
    return (
        stats['bias'] / ref_std,
        stats['crmsd'] / ref_std,
        stats['rmsd'] / ref_std,
        str(year)
    )

###############################################################################
def compute_normalised_target_stats(data_dict: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Compute normalized target statistics (bias, CRMSD, RMSD) for each year
    in the provided data dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing yearly model and satellite data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]
        Arrays of normalized bias, CRMSD, RMSD, and corresponding year labels.

    Raises
    ------
    ValueError
        If no overlapping or valid model/satellite data is found.
    """
    yearly_data = get_common_series_by_year(data_dict)
    if not yearly_data:
        raise ValueError("No overlapping model/satellite data found.")

    results = list(filter(None, starmap(compute_single_target_stat, yearly_data)))

    if not results:
        raise ValueError("No valid data available to compute statistics.")

    bias_norm, crmsd_norm, rmsd_norm, labels = zip(*results)
    return np.array(bias_norm), np.array(crmsd_norm), np.array(rmsd_norm), list(labels)
###############################################################################

###############################################################################
def compute_normalised_target_stats_by_month(data_dict: Dict,
                                             month_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Compute normalized target statistics (bias, CRMSD, RMSD) for a specified month
    across all years in the provided data dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data.
    month_index : int
        Month index (0 = January, 11 = December).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]
        Arrays of normalized bias, CRMSD, RMSD, and year labels.

    Raises
    ------
    ValueError
        If no overlapping or valid data is found for the specified month.
    """
    if not isinstance(month_index, int) or not (0 <= month_index <= 11):
        raise ValueError("month_index must be an integer between 0 and 11.")

    monthly_data = get_common_series_by_year_month(data_dict)
    filtered_data = [(year, month, mod, sat) for (year, month, mod, sat) in monthly_data if month == month_index]

    if not filtered_data:
        raise ValueError(f"No overlapping model/satellite data found for month {month_index}.")

    results = list(filter(None, starmap(compute_single_month_target_stat, filtered_data)))

    if not results:
        raise ValueError("No valid data available to compute statistics.")

    bias_norm, crmsd_norm, rmsd_norm, labels = zip(*results)
    return np.array(bias_norm), np.array(crmsd_norm), np.array(rmsd_norm), list(labels)
###############################################################################

###############################################################################
def compute_target_extent_monthly(taylor_dict: Dict) -> float:
    """
    Compute the plotting extent (max RMSD rounded up) for a monthly target diagram.

    Parameters
    ----------
    taylor_dict : dict
        Dictionary containing model and satellite data.

    Returns
    -------
    float
        Rounded-up maximum normalized RMSD across all months.

    Raises
    ------
    ValueError
        If no valid RMSD data is found across months.
    """
    monthly_rmsds = (
        compute_normalised_target_stats_by_month(taylor_dict, month)[2]
        for month in range(12)
    )

    all_rmsds = list(chain.from_iterable(
        rmsd for rmsd in monthly_rmsds if rmsd.size > 0
    ))

    if not all_rmsds:
        raise ValueError("No valid RMSD data to determine extent.")

    return round_up_to_nearest(max(all_rmsds))
###############################################################################

###############################################################################
def compute_target_extent_yearly(data_dict: Dict) -> float:
    """
    Compute the plotting extent (max RMSD rounded up) for a yearly target diagram.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data.

    Returns
    -------
    float
        Rounded-up maximum normalized RMSD, or 1.0 if all RMSDs are below that threshold.

    Raises
    ------
    ValueError
        If no valid RMSD data is found.
    """
    _, _, rmsd, _ = compute_normalised_target_stats(data_dict)

    if rmsd.size == 0:
        raise ValueError("No valid RMSD data to determine extent.")

    max_rmsd = np.max(rmsd)
    return round_up_to_nearest(max_rmsd) if max_rmsd > 1.0 else 1.0
###############################################################################