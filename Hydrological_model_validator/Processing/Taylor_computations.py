import skill_metrics as sm
import numpy as np
import pandas as pd

from .data_alignment import get_common_series_by_year, get_valid_mask, extract_mod_sat_keys

def compute_taylor_stat_tuple(mod_values, sat_values, label):
    """Compute Taylor statistics for two arrays of values."""
    stats = sm.taylor_statistics(mod_values, sat_values, 'data')
    return (label, stats['sdev'][1], stats['crmsd'][1], stats['ccoef'][1])

def compute_std_reference(sat_data_by_year, years, month_index):
    """Compute reference std deviation for satellite data of a given month over all years."""
    all_monthly_sat = np.concatenate([
        np.asarray(sat_data_by_year[year][month_index]).flatten()
        for year in years
        if month_index < len(sat_data_by_year[year])
    ])
    return np.nanstd(all_monthly_sat)

def compute_norm_taylor_stats(mod_vals, sat_vals, std_ref):
    """
    Compute normalized Taylor statistics for one month-year pair.
    Returns None if no valid data.
    """
    valid = get_valid_mask(mod_vals, sat_vals)
    if not np.any(valid):
        return None

    stats = sm.taylor_statistics(mod_vals[valid], sat_vals[valid], 'data')
    return {
        "sdev": stats['sdev'][1] / std_ref,
        "crmsd": stats['crmsd'][1] / std_ref,
        "ccoef": stats['ccoef'][1],
    }

def build_all_points(data_dict):
    """
    Build list of all Taylor diagram points with normalized stats for all months and years.
    Includes the reference points.
    """
    mod_key, sat_key = extract_mod_sat_keys(data_dict)
    model_data_by_year = data_dict[mod_key]
    sat_data_by_year = data_dict[sat_key]
    years = sorted(sat_data_by_year.keys())

    all_points = []

    # Precompute std_ref per month once
    std_refs = {
        month_idx: compute_std_reference(sat_data_by_year, years, month_idx)
        for month_idx in range(12)
    }

    for month_idx in range(12):
        std_ref = std_refs[month_idx]

        # Add reference point (normalized)
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

def compute_yearly_taylor_stats(data_dict):
    """
    Compute Taylor statistics for each year using model and satellite data from data_dict.
    Also computes the global standard deviation of all satellite data.

    Returns:
        tuple: (list of yearly stats tuples, global satellite std_ref)
    """
    _, sat_key = extract_mod_sat_keys(data_dict)
    sat_data_by_year = data_dict[sat_key]

    # Flatten and concatenate all satellite data across years and months
    all_sat_data = np.concatenate([
        np.asarray(month_array).flatten()
        for year_data in sat_data_by_year.values()
        for month_array in year_data
    ])

    std_ref = np.nanstd(all_sat_data)

    aligned_data = get_common_series_by_year(data_dict)
    yearly_stats = [compute_taylor_stat_tuple(mod, sat, year) for year, mod, sat in aligned_data]

    return yearly_stats, std_ref