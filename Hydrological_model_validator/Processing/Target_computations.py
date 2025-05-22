import numpy as np
import skill_metrics as sm

from Hydrological_model_validator.Processing.data_alignment import (get_common_series_by_year, 
                                                                    get_common_series_by_year_month)
from Hydrological_model_validator.Processing.stats_math_utils import round_up_to_nearest

def compute_normalised_target_stats(data_dict):
    """
    Compute normalised target statistics (bias, CRMSD, RMSD) for each year
    in the provided taylor_dict.

    Returns:
        Tuple of (bias_array, crmsd_array, rmsd_array, labels)
    """
    import numpy as np
    import skill_metrics as sm

    yearly_data = get_common_series_by_year(data_dict)
    if not yearly_data:
        raise ValueError("No overlapping model/satellite data found.")

    bias_norm, crmsd_norm, rmsd_norm, labels = [], [], [], []

    for year, mod_values, sat_values in yearly_data:
        stats = sm.target_statistics(mod_values, sat_values, 'data')

        ref_std = np.std(sat_values, ddof=1)
        if ref_std == 0:
            print(f"Warning: Zero standard deviation in satellite data for {year}. Skipping.")
            continue

        bias_norm.append(stats['bias'] / ref_std)
        crmsd_norm.append(stats['crmsd'] / ref_std)
        rmsd_norm.append(stats['rmsd'] / ref_std)
        labels.append(year)

    if not bias_norm:
        raise ValueError("No valid data available to compute statistics.")

    return np.array(bias_norm), np.array(crmsd_norm), np.array(rmsd_norm), labels

def compute_normalised_target_stats_by_month(data_dict, month_index):
    """
    Compute normalized target statistics (bias, CRMSD, RMSD) for a specified month
    across all years in the provided taylor_dict.

    Returns:
        Tuple of (bias_array, crmsd_array, rmsd_array, labels)
    """

    # Get all (year, month, mod_values, sat_values) tuples
    monthly_data = get_common_series_by_year_month(data_dict)

    # Filter only the specified month
    filtered_data = [(year, mod, sat) for (year, month, mod, sat) in monthly_data if month == month_index]

    if not filtered_data:
        raise ValueError(f"No overlapping model/satellite data found for month {month_index}.")

    bias_norm, crmsd_norm, rmsd_norm, labels = [], [], [], []

    for year, mod_values, sat_values in filtered_data:
        stats = sm.target_statistics(mod_values, sat_values, 'data')
        ref_std = np.std(sat_values, ddof=1)

        if ref_std == 0:
            print(f"Warning: Zero standard deviation in satellite data for {year}, month {month_index}. Skipping.")
            continue

        bias_norm.append(stats['bias'] / ref_std)
        crmsd_norm.append(stats['crmsd'] / ref_std)
        rmsd_norm.append(stats['rmsd'] / ref_std)
        labels.append(str(year))

    if not bias_norm:
        raise ValueError("No valid data available to compute statistics.")

    return np.array(bias_norm), np.array(crmsd_norm), np.array(rmsd_norm), labels

def compute_target_extent_monthly(taylor_dict):

    all_rmsds = []
    for month_index in range(12):
        try:
            _, _, rmsd, _ = compute_normalised_target_stats_by_month(taylor_dict, month_index)
            all_rmsds.extend(rmsd)
        except ValueError:
            continue

    if not all_rmsds:
        raise ValueError("No valid RMSD data to determine extent.")

    max_rmsd = max(all_rmsds)
    extent = round_up_to_nearest(max_rmsd)
    return extent

def compute_target_extent_yearly(data_dict):
    
    compute_normalised_target_stats(data_dict)
    _, _, rmsd, _ = compute_normalised_target_stats(data_dict)
    
    if np.max(rmsd) > 1.0:
        extent = round_up_to_nearest(np.max(rmsd))
    else:
        extent = 1.0
        
    return extent