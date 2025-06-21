###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Data handling libraries
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union

# Logging and tracing
import logging
from eliot import start_action, log_message

# Module utilities
from .time_utils import Timer

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def get_valid_mask(mod_vals: np.ndarray, sat_vals: np.ndarray) -> np.ndarray:
    """
    Generate a boolean mask identifying elements where both model and satellite data are valid (non-NaN).

    This function compares two numpy arrays element-wise and returns a boolean array that is True
    only at positions where neither array has NaN values, effectively marking data points valid
    for both datasets. This mask can be used for paired analysis or filtering.

    Parameters
    ----------
    mod_vals : np.ndarray
        Array of model data values, can be of any shape.
    sat_vals : np.ndarray
        Array of satellite data values, must have the same shape as `mod_vals`.

    Returns
    -------
    np.ndarray
        Boolean numpy array of the same shape as inputs, where True indicates positions with valid
        (non-NaN) data in both `mod_vals` and `sat_vals`, and False otherwise.

    Raises
    ------
    TypeError
        If either `mod_vals` or `sat_vals` is not a numpy ndarray.
    ValueError
        If the shapes of `mod_vals` and `sat_vals` do not match.

    Example
    -------
    >>> import numpy as np
    >>> model_data = np.array([1.0, np.nan, 3.0, 4.0])
    >>> satellite_data = np.array([1.5, 2.0, np.nan, 4.5])
    >>> mask = get_valid_mask(model_data, satellite_data)
    >>> print(mask)
    [ True False False  True]
    """
    # ===== INPUT VALIDATIONS =====
    if not isinstance(mod_vals, np.ndarray):
        raise TypeError("❌ mod_vals must be a numpy array ❌")
    if not isinstance(sat_vals, np.ndarray):
        raise TypeError("❌ sat_vals must be a numpy array ❌")

    if mod_vals.shape != sat_vals.shape:
        raise ValueError("❌ mod_vals and sat_vals must have the same shape ❌")

    with Timer("get_valid_mask function"):
        with start_action(action_type="get_valid_mask function"):

            log_message("Input types validated",
                        mod_vals_type=str(type(mod_vals)),
                        sat_vals_type=str(type(sat_vals)))
            logging.info(f"Input arrays validated: mod_vals type={type(mod_vals)}, sat_vals type={type(sat_vals)}")

            log_message("Input shapes validated", shape=mod_vals.shape)
            logging.info(f"Input arrays shapes matched: shape={mod_vals.shape}")

            # Create a boolean mask indicating True where mod_vals is not NaN AND sat_vals is not NaN
            valid_mask = ~np.isnan(mod_vals) & ~np.isnan(sat_vals)

            log_message("Computed valid mask", valid_count=np.sum(valid_mask), total_elements=valid_mask.size)
            logging.info(f"Valid mask computed: {np.sum(valid_mask)} valid points out of {valid_mask.size}")

            return valid_mask
        
###############################################################################

###############################################################################
def get_valid_mask_pandas(mod_series: pd.Series, 
                          sat_series: pd.Series) -> pd.Series:
    """
    Generate a boolean pandas Series mask indicating positions where both model and satellite data
    Series have valid (non-NaN) values, aligned by their common index.

    This function takes two pandas Series, aligns them on the intersection of their indices, and
    returns a boolean Series that is True where both input Series have non-missing data. This mask
    can be used to filter or compare paired time series or other indexed data.

    Parameters
    ----------
    mod_series : pd.Series
        Pandas Series containing model data values.
    sat_series : pd.Series
        Pandas Series containing satellite data values.

    Returns
    -------
    pd.Series
        Boolean Series indexed by the intersection of input Series indices, where True indicates
        valid data points (non-NaN) in both inputs.

    Raises
    ------
    TypeError
        If either input is not a pandas Series.

    Example
    -------
    >>> import pandas as pd
    >>> model_s = pd.Series([1.0, None, 3.0, 4.0], index=pd.date_range("2023-01-01", periods=4))
    >>> sat_s = pd.Series([1.5, 2.0, None, 4.5], index=pd.date_range("2023-01-01", periods=4))
    >>> mask = get_valid_mask_pandas(model_s, sat_s)
    >>> print(mask)
    2023-01-01     True
    2023-01-02    False
    2023-01-03    False
    2023-01-04     True
    Freq: D, dtype: bool
    """
    # ===== INPUT VALIDATION =====
    # Validate that mod_series is a pandas Series for correct operations and alignment
    if not isinstance(mod_series, pd.Series):
        raise TypeError("❌ mod_series must be a pandas Series ❌")
    
    # Validate that sat_series is a pandas Series as well
    if not isinstance(sat_series, pd.Series):
        raise TypeError("❌ sat_series must be a pandas Series ❌")

    with Timer("get_valid_mask_pandas function"):
        with start_action(action_type="get_valid_mask_pandas function"):

            log_message("Input types validated",
                        mod_series_type=str(type(mod_series)),
                        sat_series_type=str(type(sat_series)))
            logging.info(f"Input Series validated: mod_series type={type(mod_series)}, sat_series type={type(sat_series)}")

            # ===== ALIGN THE DATA =====
            # Find the intersection of indices between both Series to compare aligned data points
            common_index = mod_series.index.intersection(sat_series.index)

            log_message("Computed common index",
                        common_index_length=len(common_index))
            logging.info(f"Common index computed with length: {len(common_index)}")

            # Align both Series to the common index so the mask compares matching data points
            mod_aligned = mod_series.loc[common_index]
            sat_aligned = sat_series.loc[common_index]

            log_message("Series aligned to common index",
                        aligned_length=len(mod_aligned))
            logging.info(f"Series aligned to common index length: {len(mod_aligned)}")

            # ===== MASK =====
            # Create mask: True where neither Series has NaN (isna() checks for NaNs, ~ negates it)
            mask = (~mod_aligned.isna()) & (~sat_aligned.isna())

            valid_count = mask.sum()
            log_message("Computed valid mask",
                        valid_count=int(valid_count),
                        total=len(mask))
            logging.info(f"Valid mask computed: {valid_count} valid points out of {len(mask)}")

            # Return the boolean mask Series indexed by the common index
            return mask
        
###############################################################################

###############################################################################
def align_pandas_series(mod_series: pd.Series, 
                        sat_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two pandas Series by their index, returning numpy arrays of values where both Series have
    overlapping indices and non-NaN data.

    This function finds the intersection of the indices from the two Series, filters out any entries
    where either Series has NaN, and returns two numpy arrays containing only the valid paired data,
    ready for further analysis or comparison.

    Parameters
    ----------
    mod_series : pd.Series
        Pandas Series containing model data.
    sat_series : pd.Series
        Pandas Series containing satellite data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of two numpy arrays: (aligned model values, aligned satellite values), where each array
        contains only values at indices where both Series have non-NaN data.

    Raises
    ------
    TypeError
        If either input is not a pandas Series.

    Example
    -------
    >>> import pandas as pd
    >>> model_s = pd.Series([1.0, None, 3.0, 4.0], index=pd.date_range("2023-01-01", periods=4))
    >>> sat_s = pd.Series([1.5, 2.0, None, 4.5], index=pd.date_range("2023-01-01", periods=4))
    >>> mod_vals, sat_vals = align_pandas_series(model_s, sat_s)
    >>> print(mod_vals)
    [1. 4.]
    >>> print(sat_vals)
    [1.5 4.5]
    """
    # ===== INPUT VALIDATION =====
    # Ensure mod_series is a pandas Series to support index alignment and selection
    if not isinstance(mod_series, pd.Series):
        raise TypeError("❌ mod_series must be a pandas Series ❌")
    
    # Ensure sat_series is also a pandas Series
    if not isinstance(sat_series, pd.Series):
        raise TypeError("❌ sat_series must be a pandas Series ❌")

    with Timer("align_pandas_series function"):
        with start_action(action_type="align_pandas_series function"):

            log_message("Input types validated",
                        mod_series_type=str(type(mod_series)),
                        sat_series_type=str(type(sat_series)))
            logging.info(f"Input Series validated: mod_series type={type(mod_series)}, sat_series type={type(sat_series)}")

            # ===== MASK =====
            # Generate a boolean mask identifying positions where both Series have valid (non-NaN) data,
            # aligned on their common indices using the helper function.
            mask = get_valid_mask_pandas(mod_series, sat_series)

            log_message("Valid mask generated", valid_mask_sum=int(mask.sum()))
            logging.info(f"Valid mask generated with {mask.sum()} valid points")

            # ===== ALIGN THE DATA =====
            # Use the boolean mask's index to select the overlapping indices, then filter mod_series 
            # to keep only the valid values (where mask is True). Extract values as a numpy array.
            mod_aligned = mod_series.loc[mask.index][mask].values

            # Similarly, filter sat_series for the same indices and valid mask positions,
            # extracting corresponding numpy array values.
            sat_aligned = sat_series.loc[mask.index][mask].values

            log_message("Aligned data extracted",
                        mod_aligned_len=len(mod_aligned),
                        sat_aligned_len=len(sat_aligned))
            logging.info(f"Aligned data extracted: mod_aligned length={len(mod_aligned)}, sat_aligned length={len(sat_aligned)}")

            # Return the two numpy arrays containing aligned and valid data pairs for analysis.
            return mod_aligned, sat_aligned
        
###############################################################################

###############################################################################
def align_numpy_arrays(mod_vals: np.ndarray, 
                       sat_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two numpy arrays by removing elements where either array contains NaN.

    This function creates a boolean mask identifying indices where both input arrays have valid
    (non-NaN) data, then returns filtered arrays containing only those valid data points.

    Parameters
    ----------
    mod_vals : np.ndarray
        Array of model values.
    sat_vals : np.ndarray
        Array of satellite values, must be the same shape as mod_vals.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of two numpy arrays (mod_vals_filtered, sat_vals_filtered) containing only elements
        where both inputs have valid (non-NaN) data.

    Raises
    ------
    TypeError
        If either input is not a numpy ndarray.
    ValueError
        If input arrays do not have the same shape.

    Example
    -------
    >>> import numpy as np
    >>> mod = np.array([1.0, np.nan, 3.0, 4.0])
    >>> sat = np.array([1.5, 2.0, np.nan, 4.5])
    >>> mod_filt, sat_filt = align_numpy_arrays(mod, sat)
    >>> print(mod_filt)
    [1. 4.]
    >>> print(sat_filt)
    [1.5 4.5]
    """
    # ===== INPUT VALIDATION =====
    # Ensure both inputs are numpy arrays to support element-wise operations
    if not isinstance(mod_vals, np.ndarray):
        raise TypeError("❌ mod_vals must be a numpy array ❌")
    if not isinstance(sat_vals, np.ndarray):
        raise TypeError("❌ sat_vals must be a numpy array ❌")

    # Check that both arrays have the same shape to allow element-wise comparison
    if mod_vals.shape != sat_vals.shape:
        raise ValueError("❌ mod_vals and sat_vals must have the same shape ❌")

    with Timer("align_numpy_arrays function"):
        with start_action(action_type="align_numpy_arrays function"):

            log_message("Input validation passed",
                        mod_vals_type=str(type(mod_vals)),
                        sat_vals_type=str(type(sat_vals)),
                        mod_vals_shape=mod_vals.shape,
                        sat_vals_shape=sat_vals.shape)
            logging.info(f"Input validation passed: mod_vals type={type(mod_vals)}, sat_vals type={type(sat_vals)}, "
                         f"mod_vals shape={mod_vals.shape}, sat_vals shape={sat_vals.shape}")

            # ===== MAKE MASK =====
            # Generate a boolean mask where both arrays have valid (non-NaN) entries
            # This allows filtering out positions with missing data in either input
            mask = get_valid_mask(mod_vals, sat_vals)

            log_message("Valid mask created", valid_mask_sum=int(mask.sum()))
            logging.info(f"Valid mask created with {mask.sum()} valid elements")

            # Apply the mask to both arrays to extract only valid (paired) data points
            return mod_vals[mask], sat_vals[mask]
        
###############################################################################

###############################################################################

def get_common_series_by_year(data_dict: Dict[str, Dict[int, pd.Series]]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Extract and align model and satellite time series data by year, returning only overlapping data points.

    This function takes a dictionary containing yearly model and satellite data as pandas Series,
    aligns them on their time indices for each year, and returns numpy arrays of paired values
    where both datasets have valid (non-NaN) data.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys for model and satellite data (e.g., 'model', 'satellite'),
        each mapping to a dictionary keyed by year (int), with pandas Series as values.

    Returns
    -------
    List[Tuple[str, np.ndarray, np.ndarray]]
        List of tuples, each containing:
        - year as a string,
        - numpy array of aligned model values for that year,
        - numpy array of aligned satellite values for that year.
        Only years with overlapping valid data are included.

    Raises
    ------
    TypeError
        If input is not a dictionary or if the model/satellite data are not dictionaries keyed by years.

    Notes
    -----
    This function depends on `extract_mod_sat_keys(data_dict)` to determine the model and satellite keys.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...     'model': {
    ...         2000: pd.Series([1.0, 2.0, np.nan], index=pd.date_range('2000-01-01', periods=3)),
    ...         2001: pd.Series([4.0, 5.0], index=pd.date_range('2001-01-01', periods=2)),
    ...     },
    ...     'satellite': {
    ...         2000: pd.Series([1.1, 2.1, 3.1], index=pd.date_range('2000-01-01', periods=3)),
    ...         2001: pd.Series([4.1, np.nan], index=pd.date_range('2001-01-01', periods=2)),
    ...     }
    ... }
    >>> get_common_series_by_year(data)
    [('2000', array([1., 2.]), array([1.1, 2.1])), ('2001', array([4.]), array([4.1]))]
    """
    # ===== INPUT VALIDATION =====
    # Confirm input is a dictionary to avoid errors when accessing keys
    if not isinstance(data_dict, dict):
        raise TypeError(f"❌ Expected input data to be dict, got {type(data_dict)} ❌")

    # Use helper function to find keys for model and satellite data, making this function flexible
    mod_key, sat_key = extract_mod_sat_keys(data_dict)

    # Validate that the data for model and satellite are dictionaries keyed by year
    if not isinstance(data_dict[mod_key], dict):
        raise TypeError(f"❌ Expected '{mod_key}' data to be a dict keyed by years, got {type(data_dict[mod_key])} ❌")
    if not isinstance(data_dict[sat_key], dict):
        raise TypeError(f"❌ Expected '{sat_key}' data to be a dict keyed by years, got {type(data_dict[sat_key])} ❌")

    with Timer("get_common_series_by_year function"):
        with start_action(action_type="get_common_series_by_year function"):

            log_message("Input validation passed",
                        data_type=str(type(data_dict)),
                        model_key=mod_key,
                        satellite_key=sat_key)
            logging.info(f"Input validation passed with data type {type(data_dict)}; model_key={mod_key}, satellite_key={sat_key}")

            common_series = []

            # ===== ALIGN AND SORT =====
            # Iterate over each year in model data to align with satellite data
            # Sorting ensures consistent order for downstream processing
            for year in sorted(data_dict[mod_key].keys()):
                mod_series = data_dict[mod_key][year].dropna()  # Remove NaNs from both series to avoid false matches on missing data
                sat_series = data_dict[sat_key][year].dropna()

                log_message("Processing year",
                            year=year,
                            mod_series_length=len(mod_series),
                            sat_series_length=len(sat_series))
                logging.info(f"Processing year {year}: mod_series length={len(mod_series)}, sat_series length={len(sat_series)}")

                # Join the two series on their datetime indices (inner join keeps only overlapping dates)
                # Drop any remaining NaNs after joining to ensure paired valid data
                combined = mod_series.to_frame('mod').join(sat_series.to_frame('sat'), how='inner').dropna()

                log_message("Joined series",
                            year=year,
                            combined_length=len(combined))
                logging.info(f"Year {year}: combined series length after join and dropna = {len(combined)}")

                # If no overlapping valid data points exist, skip this year to avoid empty results
                if combined.empty:
                    print(f"⚠️ Warning: No overlapping data for year {year}. Skipping. ⚠️")
                    log_message("Skipping year due to no overlapping data", year=year)
                    logging.warning(f"No overlapping data for year {year}, skipping.")
                    continue

                # Append a tuple containing the year (as string), and the numpy arrays of aligned values
                # Using .values extracts the raw numeric arrays for further numerical processing
                common_series.append((str(year), combined['mod'].values, combined['sat'].values))

                log_message("Appended aligned data",
                            year=year,
                            mod_values_len=len(combined['mod'].values),
                            sat_values_len=len(combined['sat'].values))
                logging.info(f"Appended data for year {year} with {len(combined['mod'].values)} model and {len(combined['sat'].values)} satellite points")

            return common_series
        
###############################################################################

###############################################################################

def get_common_series_by_year_month(
    data_dict: Dict[str, Dict[Union[int, str], List[np.ndarray]]]
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Extract and align monthly model and satellite data by year.

    This function iterates over all available years and months in the input data,
    aligning model and satellite arrays for each month. It filters out elements where 
    either array contains NaN values, returning only valid data pairs.

    Parameters
    ----------
    data_dict : dict
        Dictionary with two top-level keys (e.g., 'model' and 'satellite'). Each key maps
        to a dictionary where each year (int or str) maps to a list of 12 numpy arrays,
        one per month, representing time-resolved spatial or summary data.

    Returns
    -------
    List[Tuple[int, int, np.ndarray, np.ndarray]]
        A list of tuples, each containing:
        - year as an integer,
        - month index (0-based, 0 = January, 11 = December),
        - NumPy array of model values with valid data,
        - NumPy array of satellite values with valid data.

    Raises
    ------
    TypeError
        If input is not structured as expected (e.g., dicts or lists missing or incorrect types).
    ValueError
        If a year does not contain 12 monthly entries.

    Example
    -------
    >>> import numpy as np
    >>> data = {
    ...     'model': {
    ...         2000: [np.array([1.0, np.nan]), np.array([2.0]), *[np.array([])]*10]
    ...     },
    ...     'satellite': {
    ...         2000: [np.array([1.1, 2.2]), np.array([2.1]), *[np.array([])]*10]
    ...     }
    ... }
    >>> get_common_series_by_year_month(data)
    [(2000, 0, array([1.]), array([1.1])), (2000, 1, array([2.]), array([2.1]))]
    """
    # ===== INPU VALIDATION AND GET KEYS =====
    if not isinstance(data_dict, dict):
        raise TypeError("❌ data_dict must be a dictionary ❌")

    # Identify model and satellite keys from the dictionary keys
    mod_key, sat_key = extract_mod_sat_keys(data_dict)

    model_data = data_dict.get(mod_key, {})
    satellite_data = data_dict.get(sat_key, {})

    if not isinstance(model_data, dict) or not isinstance(satellite_data, dict):
        raise TypeError("❌ Sub-values of data_dict must be dictionaries of lists of numpy arrays ❌")

    with Timer("get_common_series_by_year_month function"):
        with start_action(action_type="get_common_series_by_year_month function"):

            log_message("Input validation passed",
                        data_type=str(type(data_dict)),
                        model_key=mod_key,
                        satellite_key=sat_key)
            logging.info(f"Input validation passed; model_key={mod_key}, satellite_key={sat_key}")

            results = []

            # Find common years
            common_years = sorted(set(model_data.keys()) & set(satellite_data.keys()))

            log_message("Common years found", years=common_years)
            logging.info(f"Common years found: {common_years}")

            # ===== GET COMMON YEARS =====
            for year in common_years:
                mod_monthly = model_data[year]
                sat_monthly = satellite_data[year]

                # ===== FURTHER VALIDATION =====
                if not isinstance(mod_monthly, (list, tuple)) or not isinstance(sat_monthly, (list, tuple)):
                    raise TypeError(f"❌ Data for year {year} must be a list or tuple of numpy arrays ❌")
                if len(mod_monthly) != len(sat_monthly):
                    raise ValueError(f"❌ Year {year} does not contain the same number of monthly entries for model and satellite ❌")

                month_count = len(mod_monthly)
                log_message("Processing year",
                            year=year,
                            month_count=month_count)
                logging.info(f"Processing year {year} with {month_count} months")

                # ===== MONTHS LOOP =====
                for month in range(month_count):
                    mod_vals = mod_monthly[month]
                    sat_vals = sat_monthly[month]

                    mod_vals = np.asarray(mod_vals)
                    sat_vals = np.asarray(sat_vals)

                    # Skip if shapes differ
                    if mod_vals.shape != sat_vals.shape:
                        log_message("Skipping month due to shape mismatch",
                                    year=year,
                                    month=month,
                                    mod_shape=mod_vals.shape,
                                    sat_shape=sat_vals.shape)
                        logging.warning(f"Skipping year {year} month {month} due to shape mismatch: mod_shape={mod_vals.shape}, sat_shape={sat_vals.shape}")
                        continue

                    # Align arrays by removing entries where either is NaN
                    mod_filtered, sat_filtered = align_numpy_arrays(mod_vals, sat_vals)

                    log_message("Aligned month data",
                                year=year,
                                month=month,
                                mod_filtered_len=len(mod_filtered),
                                sat_filtered_len=len(sat_filtered))
                    logging.info(f"Year {year} month {month}: aligned data lengths: mod={len(mod_filtered)}, sat={len(sat_filtered)}")

                    if len(mod_filtered) > 0:
                        results.append((int(year), month, mod_filtered, sat_filtered))

            return results
        
###############################################################################

###############################################################################

def extract_mod_sat_keys(taylor_dict: Dict) -> Tuple[str, str]:
    """
    Identify and return the keys corresponding to model and satellite data within a dictionary.

    This function searches for keys commonly associated with model data (e.g., 'mod', 'model', 'predicted')
    and satellite data (e.g., 'sat', 'satellite', 'observed') within the provided dictionary.
    It returns a tuple containing the identified model and satellite keys.

    Parameters
    ----------
    taylor_dict : dict
        Dictionary expected to contain keys for model and satellite datasets.

    Returns
    -------
    Tuple[str, str]
        Tuple with two strings:
        - model_key: Key associated with model data in the dictionary.
        - satellite_key: Key associated with satellite data in the dictionary.

    Raises
    ------
    TypeError
        If the input is not a dictionary.
    ValueError
        If suitable keys for model or satellite data cannot be found in the dictionary.

    Example
    -------
    >>> data = {'model': ..., 'satellite': ...}
    >>> extract_mod_sat_keys(data)
    ('model', 'satellite')
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(taylor_dict, dict):
        raise TypeError("❌ Input must be a dictionary ❌")

    with Timer("extract_mod_sat_keys function"):
        with start_action(action_type="extract_mod_sat_keys function"):

            log_message("Input validation passed", input_type=str(type(taylor_dict)))
            logging.info(f"Input validation passed; type: {type(taylor_dict)}")

            # ===== POSSIBLE CANDIDATES =====
            model_candidates = ['mod', 'model', 'predicted', 'model_data']
            satellite_candidates = ['sat', 'satellite', 'observed', 'obs', 'sat_data']

            model_key = None
            satellite_key = None

            # Iterate keys and lower them once for efficiency
            lowered_keys = {k: k.lower() for k in taylor_dict.keys()}

            # ===== START LOOKING =====
            # Find model key by substring matching candidates in keys
            for candidate in model_candidates:
                for orig_key, lowered_key in lowered_keys.items():
                    if candidate in lowered_key:
                        model_key = orig_key
                        log_message("Found model key", candidate=candidate, model_key=model_key)
                        logging.info(f"Found model key '{model_key}' matching candidate '{candidate}'")
                        break
                if model_key is not None:
                    break

            # Find satellite key similarly
            for candidate in satellite_candidates:
                for orig_key, lowered_key in lowered_keys.items():
                    if candidate in lowered_key:
                        satellite_key = orig_key
                        log_message("Found satellite key", candidate=candidate, satellite_key=satellite_key)
                        logging.info(f"Found satellite key '{satellite_key}' matching candidate '{candidate}'")
                        break
                if satellite_key is not None:
                    break

            # ===== FAILURE ERRORS =====
            if model_key is None:
                raise ValueError("❌ No suitable model key found in the dictionary ❌")
            if satellite_key is None:
                raise ValueError("❌ No suitable satellite key found in the dictionary ❌")

            return model_key, satellite_key
        
###############################################################################

############################################################################### 
    
def gather_monthly_data_across_years(data_dict: Dict[str, Dict[int, List[Union[np.ndarray, list]]]],
                                     key: str,
                                     month_idx: int) -> np.ndarray:
    """
    Collect and concatenate data for a specified month across all years for a given dataset key.

    This function extracts monthly data arrays or lists for the specified key (e.g., model or satellite)
    from each year in the provided nested dictionary. It flattens each month's data, concatenates all
    years' data for that month into a single 1D numpy array, and removes any NaN values.

    Parameters
    ----------
    data_dict : dict
        Nested dictionary containing data arrays/lists keyed first by dataset keys (e.g., 'mod', 'sat'),
        then by year (int), where each year maps to a list of monthly arrays or lists.
    key : str
        Dataset key to select data from `data_dict` (e.g., 'mod' or 'sat').
    month_idx : int
        Zero-based month index to select (0 = January, ..., 11 = December).

    Returns
    -------
    np.ndarray
        One-dimensional numpy array of concatenated valid (non-NaN) data for the specified month
        across all years.

    Raises
    ------
    ValueError
        If `data_dict` is not a dictionary or `key` is not found in it, or data for a year/month is invalid.
    IndexError
        If `month_idx` is not in the range 0 to 11 or if any year's data does not have enough monthly entries.

    Example
    -------
    >>> data = {
    ...     'mod': {
    ...         2020: [np.array([1, 2, np.nan]), np.array([3, 4]), *[np.array([])]*10],
    ...         2021: [np.array([5, np.nan]), np.array([6, 7]), *[np.array([])]*10]
    ...     }
    ... }
    >>> gather_monthly_data_across_years(data, 'mod', 0)
    array([1., 2., 5.])
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(data_dict, dict):
        raise ValueError("❌ data_dict must be a dictionary ❌")
    if key not in data_dict:
        raise ValueError(f"❌ Key '{key}' not found in data_dict ❌")
    if not isinstance(month_idx, int) or not (0 <= month_idx <= 11):
        raise IndexError("❌ month_idx must be an integer between 0 and 11 ❌")

    with Timer("gather_monthly_data_across_years function"):
        with start_action(action_type="gather_monthly_data_across_years function", key=key, month_idx=month_idx):
            
            logging.info(f"Starting data gathering for key '{key}', month index {month_idx}")
            log_message("Starting data gathering", key=key, month_idx=month_idx)

            year_data = data_dict[key]
            all_data = []

            # ===== GATHER THE DATA =====
            for year, monthly_list in year_data.items():
                if not isinstance(monthly_list, (list, tuple)):
                    raise ValueError(f"❌ Year {year} data must be a list or tuple ❌")

                if month_idx >= len(monthly_list):
                    raise IndexError(f"❌ Year {year} does not contain month index {month_idx} ❌")

                month_data = monthly_list[month_idx]

                # Use np.asarray to handle any array-like input flexibly
                month_array = np.asarray(month_data)

                if month_array.ndim == 0:
                    # For scalar values, convert to 1D array explicitly
                    month_array = month_array.reshape(1)

                flat_data = month_array.ravel()
                all_data.append(flat_data)

                logging.info(f"Collected data for year {year}, month {month_idx}, length: {len(flat_data)}")
                log_message("Collected monthly data", year=year, month_idx=month_idx, length=len(flat_data))

            if not all_data:
                logging.info("No data found to concatenate; returning empty array")
                log_message("No data found to concatenate")
                return np.array([])

            concatenated = np.concatenate(all_data)
            valid_data = concatenated[~np.isnan(concatenated)]

            logging.info(f"Concatenated data length: {len(concatenated)}; valid data length (no NaNs): {len(valid_data)}")
            log_message("Completed data gathering", concatenated_length=len(concatenated), valid_length=len(valid_data))

            return valid_data
        
###############################################################################

###############################################################################

def apply_3d_mask(data: np.ndarray, mask3d: np.ndarray) -> np.ndarray:
    """
    Apply a 3D mask to a data array, setting masked elements to NaN where the mask is zero.

    This function takes a 3D mask array with shape (depth, lat, lon) and applies it to the input
    data array, which must have the last three dimensions matching (or broadcast-compatible with)
    the mask shape. Any element in `data` corresponding to a zero in the mask will be replaced
    by `np.nan`. The mask is broadcasted first to match the mask shape exactly, then broadcasted
    again to match the full `data` shape.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (..., depth, lat, lon) or exactly (depth, lat, lon).
    mask3d : np.ndarray
        3D mask array of shape (depth, lat, lon), where zero values indicate masked regions.

    Returns
    -------
    np.ndarray
        Data array of the same shape as the input, with masked elements set to `np.nan`.

    Raises
    ------
    TypeError
        If either `data` or `mask3d` is not a numpy ndarray.
    ValueError
        If `mask3d` is not 3-dimensional or cannot be broadcast to the last three dimensions of `data`.

    Example
    -------
    >>> import numpy as np
    >>> data = np.ones((2, 3, 4, 5))
    >>> mask = np.ones((3, 4, 5))
    >>> mask[1, 2, 3] = 0
    >>> masked_data = apply_3d_mask(data, mask)
    >>> np.isnan(masked_data[:, 1, 2, 3]).all()
    True
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(data, np.ndarray) or not isinstance(mask3d, np.ndarray):
        raise TypeError("❌ Both data and mask3d must be numpy arrays ❌")

    # Ensure the mask has exactly 3 dimensions representing (depth, lat, lon)
    if mask3d.ndim != 3:
        raise ValueError("❌ mask3d must be a 3D array ❌")

    with Timer("apply_3d_mask function"):
        with start_action(action_type="apply_3d_mask_function") as action:
            # Log the shapes inside Eliot action
            action.log("shapes_logged", fields={
                "data_shape": str(data.shape),
                "mask_shape": str(mask3d.shape),
            })

            logging.info(f"Applying 3D mask with shape {mask3d.shape} to data with shape {data.shape}")
            log_message("Starting mask application", data_shape=str(data.shape), mask_shape=str(mask3d.shape))

            try:
                # Broadcast the mask to the shape of the last three dimensions of data
                # This allows applying the same mask pattern to all preceding dimensions (e.g., time)
                broadcast_mask = np.broadcast_to(mask3d, data.shape[-3:])
                # Then broadcast this mask to the full data shape so indexing can be done directly
                full_mask = np.broadcast_to(broadcast_mask, data.shape)
            except ValueError:
                # Raise clear error if broadcasting is not possible, helping user debug shape mismatches
                err_msg = f"❌ mask3d shape {mask3d.shape} is not broadcast-compatible with data shape {data.shape} ❌"
                logging.error(err_msg)
                log_message("Mask broadcasting error", error=err_msg)
                raise ValueError(err_msg)

            # Use np.where to replace data values where the mask is zero with np.nan,
            # preserving other values. This is an efficient, vectorized masking operation.
            masked_data = np.where(full_mask == 0, np.nan, data)

            logging.info("Mask applied successfully")
            log_message("Completed mask application")

            return masked_data