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
    Compute Taylor statistics (standard deviation, centered RMSD, and correlation coefficient)
    for a given pair of model and satellite data arrays.

    Parameters
    ----------
    mod_values : np.ndarray
        Array of model data values.
    sat_values : np.ndarray
        Array of satellite data values.
    label : str
        Identifier associated with these data (e.g., year or month string).

    Returns
    -------
    Tuple[str, float, float, float]
        A tuple containing:
        - label (str): The input label
        - model standard deviation (float)
        - centered RMSD (float)
        - correlation coefficient (float)

    Raises
    ------
    ValueError
        If input arrays are empty.
        If no finite data pairs exist between model and satellite arrays.

    Example
    -------
    >>> mod = np.array([1.1, 2.0, 3.2])
    >>> sat = np.array([1.0, 2.1, 3.0])
    >>> compute_taylor_stat_tuple(mod, sat, '2001')
    ('2001', ..., ..., ...)
    """
    # =====INPUT VALIDATION=====
    # Ensure inputs are numpy arrays
    if not isinstance(mod_values, np.ndarray) or not isinstance(sat_values, np.ndarray):
        raise ValueError("❌ Both 'mod_values' and 'sat_values' must be NumPy arrays. ❌")

    # Ensure inputs are not empty
    if mod_values.size == 0 or sat_values.size == 0:
        raise ValueError("❌ Input arrays must not be empty. ❌")

    # =====VALID DATA MASK=====
    # Find valid finite data points for comparison
    valid_mask = np.isfinite(mod_values) & np.isfinite(sat_values)
    if not np.any(valid_mask):
        raise ValueError("❌ No valid finite data pairs found in input arrays. ❌")

    # =====FILTER DATA=====
    # Extract only valid data points
    mod_valid = mod_values[valid_mask]
    sat_valid = sat_values[valid_mask]

    # =====COMPUTE STATISTICS=====
    # Calculate Taylor stats for filtered data
    stats = sm.taylor_statistics(mod_valid, sat_valid, 'data')

    # =====RETURN=====
    # Return tuple: (label, model std dev, centered RMSD, correlation coef)
    return (label, stats['sdev'][1], stats['crmsd'][1], stats['ccoef'][1])
###############################################################################

###############################################################################
def compute_std_reference(sat_data_by_year: Dict[Union[int, str], List[Union[np.ndarray, list]]],
                          years: List[Union[int, str]],
                          month_index: int) -> float:
    """
    Compute the reference standard deviation for satellite data of a specific month
    across multiple years.

    Parameters
    ----------
    sat_data_by_year : dict
        Dictionary keyed by year (int or str), with each value being a list of monthly data arrays.
    years : list
        List of years (int or str) to include in the computation.
    month_index : int
        Index of the month (0 = January). Can be any valid index that exists in the dataset.

    Returns
    -------
    float
        Standard deviation of concatenated satellite values for the specified month across all given years.

    Raises
    ------
    ValueError
        If 'month_index' is not a non-negative integer.
        If no valid data is found for the specified month across the selected years.
        If any matched monthly array is empty.

    Example
    -------
    >>> sat_data_by_year = {
    ...     2000: [np.random.rand(10) for _ in range(6)],  # up to June
    ...     2001: [np.random.rand(10) for _ in range(6)]
    ... }
    >>> std = compute_std_reference(sat_data_by_year, [2000, 2001], 2)  # March
    >>> isinstance(std, float)
    True
    """
    # =====INPUT VALIDATION=====
    # Ensure month_index is a non-negative integer
    if not isinstance(month_index, int) or month_index < 0:
        raise ValueError(f"❌ 'month_index' must be a non-negative integer. Got {month_index}. ❌")

    # =====COLLECT MONTHLY DATA=====
    # Initialize list to hold monthly satellite data across years
    monthly_data = []
    for year in years:
        # Skip year if not present in satellite data dictionary
        if year not in sat_data_by_year:
            continue

        monthly_series = sat_data_by_year[year]
        # Skip if month_index out of range for that year
        if month_index >= len(monthly_series):
            continue

        # Flatten monthly data array to 1D for concatenation
        arr = np.asarray(monthly_series[month_index]).flatten()

        # Ensure monthly data is not empty
        if arr.size == 0:
            raise ValueError(f"❌ Empty data array for year {year}, month index {month_index}. ❌")

        # Append valid data array to collection
        monthly_data.append(arr)

    # =====VALIDATION=====
    # Check if any valid monthly data was collected
    if not monthly_data:
        raise ValueError(f"❌ No valid satellite data found for month index {month_index} across given years. ❌")

    # =====CONCATENATE & COMPUTE STD=====
    # Concatenate all monthly arrays into one
    all_monthly_sat = np.concatenate(monthly_data)

    # Return standard deviation ignoring NaNs
    return np.nanstd(all_monthly_sat)
###############################################################################

###############################################################################
def compute_norm_taylor_stats(mod_vals: np.ndarray, 
                               sat_vals: np.ndarray, 
                               std_ref: float) -> Optional[Dict[str, float]]:
    """
    Compute normalized Taylor statistics for a given pair of model and satellite data arrays.

    Parameters
    ----------
    mod_vals : np.ndarray
        Array of model data values.
    sat_vals : np.ndarray
        Array of satellite data values.
    std_ref : float
        Reference standard deviation to normalize the statistics.

    Returns
    -------
    dict or None
        Dictionary containing:
        - 'sdev' : Normalized model standard deviation
        - 'crmsd': Normalized centered root-mean-square difference
        - 'ccoef': Correlation coefficient
        Returns None if there are no valid overlapping values.

    Raises
    ------
    ValueError
        If `std_ref` is not a positive number.

    Example
    -------
    >>> mod = np.array([1.0, 2.0, 3.0])
    >>> sat = np.array([1.1, 2.1, 3.1])
    >>> std_ref = 0.5
    >>> stats = compute_norm_taylor_stats(mod, sat, std_ref)
    >>> stats.keys()
    dict_keys(['sdev', 'crmsd', 'ccoef'])
    """
    # =====INPUT VALIDATION=====
    # std_ref must be a positive number for normalization
    if not isinstance(std_ref, (int, float)) or std_ref <= 0:
        raise ValueError(f"❌ 'std_ref' must be a positive number. Got {std_ref}. ❌")

    # =====VALID DATA MASK=====
    # Determine valid (finite) overlapping data points in both arrays
    valid = get_valid_mask(mod_vals, sat_vals)
    if not np.any(valid):
        # Return None if no valid data points
        return None

    # =====COMPUTE TAYLOR STATISTICS=====
    # Compute Taylor stats on valid data subset
    stats = sm.taylor_statistics(mod_vals[valid], sat_vals[valid], 'data')

    # =====NORMALIZE & RETURN=====
    # Normalize standard deviation and CRMSD by std_ref, keep correlation coefficient unchanged
    return {
        "sdev": stats['sdev'][1] / std_ref,
        "crmsd": stats['crmsd'][1] / std_ref,
        "ccoef": stats['ccoef'][1],
    }
###############################################################################

###############################################################################
def build_all_points(
    data_dict: Dict[Union[str, int], Dict[int, List[Union[np.ndarray, list]]]]
) -> Tuple[pd.DataFrame, List[Union[str, int]]]:
    """
    Build a DataFrame of normalized Taylor statistics points for all months and years,
    including reference points per month.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data structured as:
        {
            'model_key': { year: [monthly data arrays/lists] },
            'satellite_key': { year: [monthly data arrays/lists] }
        }
        Years can be strings or integers, months are indexed 0-based.

    Returns
    -------
    tuple of (pandas.DataFrame, list)
        - DataFrame with columns ['sdev', 'crmsd', 'ccoef', 'month', 'year'] containing
          normalized Taylor statistics for each year and month, plus monthly reference points.
        - List of years found in the satellite data.

    Raises
    ------
    KeyError
        If expected model or satellite keys are missing in `data_dict`.

    Notes
    -----
    - Months with invalid or zero reference standard deviation are skipped.
    - The reference point per month has sdev=1, crmsd=0, ccoef=1, labeled year='Ref'.

    Example
    -------
    >>> data_dict = {
    ...     'model': {
    ...         2000: [np.array([...]), np.array([...]), ...],
    ...         2001: [np.array([...]), np.array([...]), ...],
    ...     },
    ...     'satellite': {
    ...         2000: [np.array([...]), np.array([...]), ...],
    ...         2001: [np.array([...]), np.array([...]), ...],
    ...     }
    ... }
    >>> df, years = build_all_points(data_dict)
    >>> df.head()
       sdev  crmsd  ccoef  month  year
    0  1.00   0.00   1.00      0   Ref
    1  0.85   0.12   0.95      0  2000
    2  0.88   0.10   0.96      0  2001
    ...

    """
    # =====EXTRACT MODEL AND SATELLITE KEYS=====
    mod_key, sat_key = extract_mod_sat_keys(data_dict)

    # =====VALIDATE PRESENCE OF REQUIRED KEYS=====
    if mod_key not in data_dict or sat_key not in data_dict:
        raise KeyError(f"❌ Expected keys '{mod_key}' and '{sat_key}' not found in data_dict. ❌")

    model_data_by_year = data_dict[mod_key]
    sat_data_by_year = data_dict[sat_key]

    # =====SORT YEARS FROM SATELLITE DATA=====
    years = sorted(sat_data_by_year.keys())

    # =====DETERMINE MAXIMUM MONTHS AVAILABLE=====
    max_months = max(len(sat_data_by_year[year]) for year in years if year in sat_data_by_year)

    std_refs = {}

    # =====COMPUTE REFERENCE STANDARD DEVIATION PER MONTH=====
    for month_idx in range(max_months):
        try:
            std_refs[month_idx] = compute_std_reference(sat_data_by_year, years, month_idx)
        except ValueError:
            # Skip months without valid reference standard deviation
            continue

    all_points = []

    # =====BUILD DATA POINTS INCLUDING REFERENCE AND NORMALIZED STATS=====
    for month_idx, std_ref in std_refs.items():
        # Skip invalid or NaN reference std deviation months
        if std_ref <= 0 or np.isnan(std_ref):
            continue

        # Add reference point for perfect agreement in this month
        all_points.append({
            "sdev": 1.0,
            "crmsd": 0.0,
            "ccoef": 1.0,
            "month": month_idx,
            "year": "Ref"
        })

        # For each year, compute normalized Taylor stats if data available
        for year in years:
            try:
                mod_vals = np.asarray(model_data_by_year[year][month_idx])
                sat_vals = np.asarray(sat_data_by_year[year][month_idx])
            except (IndexError, KeyError):
                # Skip if data is missing for this month/year
                continue

            norm_stats = compute_norm_taylor_stats(mod_vals, sat_vals, std_ref)
            if norm_stats is None:
                # Skip if no valid overlapping data points
                continue

            # Append computed normalized stats with month and year info
            all_points.append({
                **norm_stats,
                "month": month_idx,
                "year": year
            })

    # =====RETURN DATAFRAME AND YEARS LIST=====
    return pd.DataFrame(all_points), years
###############################################################################

###############################################################################
def compute_yearly_taylor_stats(
    data_dict: Dict[Union[str, int], Dict[int, List[Union[np.ndarray, list]]]]
) -> Tuple[List[Tuple[str, float, float, float]], float]:
    """
    Compute Taylor statistics for each year using model and satellite data from the data dictionary.
    Also computes the global standard deviation of all satellite data.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model and satellite data organized by year and month.
        Expected structure:
        {
            'model_key': { year: [monthly data arrays/lists] },
            'satellite_key': { year: [monthly data arrays/lists] }
        }

    Returns
    -------
    tuple
        - yearly_stats : list of tuples
          List of (year, sdev, crmsd, ccoef) tuples representing Taylor statistics for each year.
        - std_ref : float
          Global satellite standard deviation across all years and months, used as a normalization reference.

    Raises
    ------
    KeyError
        If expected model or satellite keys are missing in data_dict.
    ValueError
        If global satellite standard deviation is zero or NaN (indicating invalid data).

    Example
    -------
    >>> yearly_stats, std_ref = compute_yearly_taylor_stats(data_dict)
    >>> for year, sdev, crmsd, ccoef in yearly_stats:
    ...     print(f"{year}: sdev={sdev:.2f}, crmsd={crmsd:.2f}, ccoef={ccoef:.2f}")
    ...
    2000: sdev=0.85, crmsd=0.12, ccoef=0.95
    2001: sdev=0.88, crmsd=0.10, ccoef=0.96
    ...
    """
    # =====EXTRACT MODEL AND SATELLITE KEYS=====
    mod_key, sat_key = extract_mod_sat_keys(data_dict)

    # =====CHECK THAT KEYS EXIST IN INPUT DICTIONARY=====
    if mod_key not in data_dict or sat_key not in data_dict:
        raise KeyError(f"❌ Expected keys '{mod_key}' and '{sat_key}' not found in data_dict. ❌")

    sat_data_by_year = data_dict[sat_key]

    # =====FLATTEN ALL SATELLITE DATA ACROSS ALL YEARS AND MONTHS=====
    # Filter out None or empty arrays, flatten each monthly array
    all_sat_data = np.concatenate([
        np.asarray(month_array).flatten()
        for year_data in sat_data_by_year.values()
        for month_array in year_data
        if month_array is not None and np.asarray(month_array).size > 0
    ])

    # =====COMPUTE GLOBAL STANDARD DEVIATION AS REFERENCE=====
    std_ref = np.nanstd(all_sat_data)

    # =====VALIDATE STANDARD DEVIATION=====
    if np.isnan(std_ref) or std_ref == 0:
        raise ValueError("Global satellite standard deviation is zero or NaN, indicating invalid or missing data.")

    # =====ALIGN DATA BY YEAR TO OBTAIN COMMON SERIES FOR MODEL AND SATELLITE=====
    aligned_data = get_common_series_by_year(data_dict)

    # =====COMPUTE TAYLOR STATISTICS FOR EACH YEAR=====
    yearly_stats = [
        compute_taylor_stat_tuple(mod_values, sat_values, str(year))
        for year, mod_values, sat_values in aligned_data
    ]

    # =====RETURN THE LIST OF YEARLY STATS AND GLOBAL STD REF=====
    return yearly_stats, std_ref
###############################################################################