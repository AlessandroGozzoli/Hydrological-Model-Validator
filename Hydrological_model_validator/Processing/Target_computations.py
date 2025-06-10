from typing import Tuple, List, Dict, Optional
from itertools import starmap, chain
import numpy as np
import skill_metrics as sm

import logging
from eliot import start_action, log_message

from .time_utils import Timer

from ..Processing.data_alignment import (get_common_series_by_year, 
                                         get_common_series_by_year_month)
from ..Processing.stats_math_utils import round_up_to_nearest 
from .utils import check_numeric_data

###############################################################################
def compute_single_target_stat(year: str, 
                               mod: np.ndarray, 
                               sat: np.ndarray) -> Optional[Tuple[float, float, float, str]]:
    """
    Compute normalized bias, centered RMSD (CRMSD), and RMSD for a single year of data.

    Parameters
    ----------
    year : str
        Year label as a string.
    mod : np.ndarray
        Model data values.
    sat : np.ndarray
        Satellite data values.

    Returns
    -------
    Optional[Tuple[float, float, float, str]]
        Tuple of normalized bias, CRMSD, RMSD, and the year label.
        Returns None if the reference standard deviation of satellite data is zero.

    Raises
    ------
    ValueError
        If 'mod' and 'sat' arrays have different shapes.
        If 'mod' or 'sat' contain non-numeric data.
        If 'year' is not a string.

    Example
    -------
    >>> compute_single_target_stat('2020', np.array([1, 2, 3]), np.array([1, 2, 4]))
    (normalized_bias, normalized_crmsd, normalized_rmsd, '2020')
    """
    # =====VALIDATION=====
    # Check year is a string
    if not isinstance(year, str):
        raise ValueError("❌ 'year' must be a string. ❌")
    # Check inputs are numpy arrays
    if not isinstance(mod, np.ndarray) or not isinstance(sat, np.ndarray):
        raise ValueError("❌ 'mod' and 'sat' must be numpy arrays. ❌")
    # Check arrays have the same shape
    if mod.shape != sat.shape:
        raise ValueError("❌ 'mod' and 'sat' must have the same shape. ❌")
    # Ensure data types are numeric
    if not (np.issubdtype(mod.dtype, np.number) and np.issubdtype(sat.dtype, np.number)):
        raise ValueError("❌ 'mod' and 'sat' must contain numeric data. ❌")

    with Timer("compute_single_target_stat"):
        with start_action(action_type="compute_single_target_stat", year=year):
            logging.info(f"Computing target statistics for year {year}")
            log_message("Starting compute_single_target_stat", year=year)

            # =====STD CHECK=====
            # Compute standard deviation of satellite data for normalization
            ref_std = np.std(sat, ddof=1)
            # If std is zero, return None (no variability to compare against)
            if ref_std == 0:
                logging.warning(f"Zero standard deviation in satellite data for {year}. Skipping.")
                print(f"Warning: Zero standard deviation in satellite data for {year}. Skipping.")
                return None

            # =====COMPUTATION=====
            # Compute statistical metrics using target diagram method
            stats = sm.target_statistics(mod, sat, 'data')
            # Normalize metrics by satellite std and return with year label
            result = (
                stats['bias'] / ref_std,
                stats['crmsd'] / ref_std,
                stats['rmsd'] / ref_std,
                year
            )

            logging.info(f"Computed stats for year {year}: {result}")
            log_message("Completed compute_single_target_stat", year=year, result=result)

            return result
###############################################################################

###############################################################################
def compute_single_month_target_stat(year: int,
                                    month: int,
                                    mod: np.ndarray,
                                    sat: np.ndarray) -> Optional[Tuple[float, float, float, str]]:
    """
    Compute normalized bias, centered RMSD (CRMSD), and RMSD for a single year-month.

    Parameters
    ----------
    year : int
        Year label as an integer.
    month : int
        Month index (0–11).
    mod : np.ndarray
        Model data values.
    sat : np.ndarray
        Satellite data values.

    Returns
    -------
    Optional[Tuple[float, float, float, str]]
        Tuple of normalized bias, CRMSD, RMSD, and year label as string.
        Returns None if the reference standard deviation of satellite data is zero.

    Raises
    ------
    ValueError
        If 'year' is not an integer.
        If 'month' is not an integer or not in the range 0–11.
        If 'mod' and 'sat' arrays have different shapes.
        If 'mod' or 'sat' contain non-numeric data.

    Example
    -------
    >>> compute_single_month_target_stat(2021, 5, np.array([1, 2, 3]), np.array([1, 2, 4]))
    (normalized_bias, normalized_crmsd, normalized_rmsd, '2021')
    """    
    # =====VALIDATION=====
    # Ensure year is an integer
    if not isinstance(year, int):
        raise ValueError("❌ 'year' must be an integer. ❌")
    # Ensure month is an integer in [0, 11]
    if not (isinstance(month, int) and 0 <= month <= 11):
        raise ValueError("❌ 'month' must be an integer between 0 and 11. ❌")
    # Ensure inputs are numpy arrays
    if not isinstance(mod, np.ndarray) or not isinstance(sat, np.ndarray):
        raise ValueError("❌ 'mod' and 'sat' must be numpy arrays. ❌")
    # Ensure arrays have same shape
    if mod.shape != sat.shape:
        raise ValueError("❌ 'mod' and 'sat' must have the same shape. ❌")
    # Ensure both arrays contain numeric types
    if not (np.issubdtype(mod.dtype, np.number) and np.issubdtype(sat.dtype, np.number)):
        raise ValueError("❌ 'mod' and 'sat' must contain numeric data. ❌")

    with Timer("compute_single_month_target_stat"):
        with start_action(action_type="compute_single_month_target_stat", year=year, month=month):
            logging.info(f"Computing target statistics for year {year}, month {month}")
            log_message("Starting compute_single_month_target_stat", year=year, month=month)

            # =====STD CHECK=====
            # Calculate standard deviation of satellite data
            ref_std = np.std(sat, ddof=1)
            # Skip if std is zero (no variability)
            if ref_std == 0:
                logging.warning(f"Zero standard deviation in satellite data for {year}, month {month}. Skipping.")
                print(f"Warning: Zero standard deviation in satellite data for {year}, month {month}. Skipping.")
                return None

            # =====COMPUTATION=====
            # Compute statistics using target diagram metrics
            stats = sm.target_statistics(mod, sat, 'data')
            # Return normalized metrics and year as string
            result = (
                stats['bias'] / ref_std,
                stats['crmsd'] / ref_std,
                stats['rmsd'] / ref_std,
                str(year)
            )

            logging.info(f"Computed stats for year {year}, month {month}: {result}")
            log_message("Completed compute_single_month_target_stat", year=year, month=month, result=result)

            return result
###############################################################################

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
        If 'data_dict' is not a dictionary.
        If no overlapping or valid model/satellite data is found.

    Example
    -------
    >>> data_dict = {
    ...     '2000': (np.array([1, 2, 3]), np.array([1, 2, 4])),
    ...     '2001': (np.array([2, 3, 4]), np.array([2, 3, 5]))
    ... }
    >>> bias, crmsd, rmsd, labels = compute_normalised_target_stats(data_dict)
    >>> labels
    ['2000', '2001']
    """    
    # =====VALIDATION=====
    # Ensure input is a dictionary
    if not isinstance(data_dict, dict):
        raise ValueError("❌ 'data_dict' must be a dictionary. ❌")

    # =====DATA ALIGNMENT=====
    # Get list of (year, mod, sat) tuples with aligned data
    yearly_data = get_common_series_by_year(data_dict)
    if not yearly_data:
        raise ValueError("❌ No overlapping model/satellite data found. ❌")

    with Timer("compute_normalised_target_stats"):
        with start_action(action_type="compute_normalised_target_stats"):
            logging.info("Starting computation of normalized target statistics")
            log_message("Starting compute_normalised_target_stats")

            # =====STATISTICS COMPUTATION=====
            # Compute statistics for each valid year, ignoring any that return None
            results = list(filter(None, starmap(compute_single_target_stat, yearly_data)))
            if not results:
                raise ValueError("❌ No valid data available to compute statistics. ❌")

            # =====OUTPUT FORMATTING=====
            # Unpack computed stats and return as arrays
            bias_norm, crmsd_norm, rmsd_norm, labels = zip(*results)

            logging.info(f"Computed normalized target stats for years: {labels}")
            log_message("Completed compute_normalised_target_stats", labels=list(labels))

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
        Month index (0-based).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]
        Arrays of normalized bias, CRMSD, RMSD, and year labels.

    Raises
    ------
    ValueError
        If 'month_index' is not found in the data.
        If no overlapping or valid data is found for the specified month.
        If any monthly data contains non-numeric entries.

    Example
    -------
    >>> data_dict = {
    ...     'model': {2000: [np.array([1.0]), ...]},
    ...     'satellite': {2000: [np.array([1.1]), ...]}
    ... }
    >>> bias, crmsd, rmsd, labels = compute_normalised_target_stats_by_month(data_dict, 0)
    >>> labels
    ['2000']
    """
    # ===== DATA TYPE VALIDATION =====
    check_numeric_data(data_dict)

    # ===== DATA EXTRACTION =====
    monthly_data = get_common_series_by_year_month(data_dict)
    available_months = set(month for _, month, _, _ in monthly_data)

    # ===== MONTH VALIDATION =====
    if month_index not in available_months:
        raise ValueError(f"❌ 'month_index' {month_index} not found in data. Available months: {sorted(available_months)} ❌")

    # ===== FILTER MONTH =====
    filtered_data = [(year, month, mod, sat) for (year, month, mod, sat) in monthly_data if month == month_index]
    if not filtered_data:
        raise ValueError(f"❌ No overlapping model/satellite data found for month {month_index}. ❌")

    with Timer("compute_normalised_target_stats_by_month"):
        with start_action(action_type="compute_normalised_target_stats_by_month"):
            logging.info(f"Starting computation of normalized target stats for month {month_index}")
            log_message(f"Starting compute_normalised_target_stats_by_month for month {month_index}")

            # ===== STATISTICS COMPUTATION =====
            results = list(filter(None, starmap(compute_single_month_target_stat, filtered_data)))
            if not results:
                raise ValueError("❌ No valid data available to compute statistics. ❌")

            # ===== OUTPUT FORMATTING =====
            bias_norm, crmsd_norm, rmsd_norm, labels = zip(*results)

            logging.info(f"Computed normalized target stats for month {month_index}, years: {labels}")
            log_message(f"Completed compute_normalised_target_stats_by_month for month {month_index}", labels=list(labels))

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

    Example
    -------
    >>> taylor_dict = {
    ...     '2000': [...],  # monthly data
    ...     '2001': [...]
    ... }
    >>> extent = compute_target_extent_monthly(taylor_dict)
    >>> isinstance(extent, float)
    True
    """
    # =====MONTH EXTRACTION=====
    # Get all unique months present in the data
    monthly_indices = set()
    monthly_data = get_common_series_by_year_month(taylor_dict)
    for _, month, _, _ in monthly_data:
        monthly_indices.add(month)
    monthly_indices = sorted(monthly_indices)

    with Timer("compute_target_extent_monthly"):
        with start_action(action_type="compute_target_extent_monthly"):
            logging.info("Starting RMSD extraction for all months")
            log_message("Starting compute_target_extent_monthly")

            # =====RMSD COLLECTION=====
            # Compute normalized RMSD values for each month
            monthly_rmsds = (
                compute_normalised_target_stats_by_month(taylor_dict, month)[2]  # index 2 = RMSD
                for month in monthly_indices
            )

            # =====FLATTEN & FILTER=====
            # Combine all monthly RMSD arrays into a single flat list
            all_rmsds = list(chain.from_iterable(
                rmsd for rmsd in monthly_rmsds if rmsd.size > 0  # skip empty arrays
            ))

            # =====VALIDATION=====
            # Raise error if no RMSD data was found
            if not all_rmsds:
                raise ValueError("❌ No valid RMSD data to determine extent. ❌")

            # =====MAX EXTENT=====
            # Return maximum RMSD, rounded up
            extent = round_up_to_nearest(max(all_rmsds))

            logging.info(f"Computed extent (max RMSD rounded): {extent}")
            log_message(f"Completed compute_target_extent_monthly with extent {extent}")

            return extent
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
        Rounded-up maximum normalized RMSD across all years.
        Returns 1.0 if all RMSDs are below or equal to 1.0.

    Raises
    ------
    ValueError
        If no valid RMSD data is found.

    Example
    -------
    >>> data_dict = {
    ...     '2000': [...],  # yearly data
    ...     '2001': [...]
    ... }
    >>> extent = compute_target_extent_yearly(data_dict)
    >>> isinstance(extent, float)
    True
    """
    # =====INPUT VALIDATION=====
    # Ensure input is a dictionary
    if not isinstance(data_dict, dict):
        raise ValueError("❌ 'data_dict' must be a dictionary containing model and satellite data. ❌")

    with Timer("compute_target_extent_yearly"):
        with start_action(action_type="compute_target_extent_yearly"):
            logging.info("Starting RMSD extraction for all years")
            log_message("Starting compute_target_extent_yearly")

            # =====STATISTICS COMPUTATION=====
            # Get normalized RMSD values for each year (ignore bias and crmsd)
            _, _, rmsd, _ = compute_normalised_target_stats(data_dict)

            # =====VALIDATION=====
            # Ensure RMSD array contains data
            if rmsd.size == 0:
                raise ValueError("❌ No valid RMSD data to determine extent. ❌")

            # =====MAX EXTENT=====
            # Return rounded-up max RMSD if > 1.0, else default to 1.0
            max_rmsd = np.max(rmsd)
            extent = round_up_to_nearest(max_rmsd) if max_rmsd > 1.0 else 1.0

            logging.info(f"Computed extent (max RMSD rounded or default 1.0): {extent}")
            log_message(f"Completed compute_target_extent_yearly with extent {extent}")

            return extent
###############################################################################