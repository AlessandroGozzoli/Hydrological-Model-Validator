from typing import List, Optional, Dict, Any, Union, Tuple, Iterable
from pathlib import Path
import re
import numpy as np
import xarray as xr

###############################################################################
def find_key(
    dictionary: Dict[Any, Any], 
    possible_keys: Iterable[str]
) -> Optional[str]:
    """
    Find the first key in a dictionary containing any of the substrings in possible_keys (case insensitive).

    Parameters
    ----------
    dictionary : dict
        Dictionary to search keys in.

    possible_keys : iterable of str
        Iterable of substrings to look for in the dictionary keys.

    Returns
    -------
    Optional[str]
        The first matching key found that contains any substring from possible_keys (case insensitive),
        or None if no key matches.

    Raises
    ------
    ValueError
        If `dictionary` is not a dict or `possible_keys` is not an iterable of strings.

    Examples
    --------
    >>> d = {'Temperature': 23, 'Salinity': 35}
    >>> find_key(d, ['temp', 'sal'])
    'Temperature'
    >>> find_key(d, ['sal'])
    'Salinity'
    >>> find_key(d, ['pressure'])
    None
    """
    # ===== VALIDATE INPUT TYPES =====
    if not isinstance(dictionary, dict):
        raise ValueError("Input 'dictionary' must be a dictionary.")
    if not (hasattr(possible_keys, '__iter__') and
            not isinstance(possible_keys, str) and
            all(isinstance(k, str) for k in possible_keys)):
        raise ValueError("Input 'possible_keys' must be an iterable of strings, not a string itself.")

    # ===== PREPARE LOWERCASE SUBSTRINGS FOR MATCHING =====
    possible_keys_lower = [sub.lower() for sub in possible_keys]  # Case-insensitive matching

    # ===== SEARCH FOR FIRST MATCHING KEY =====
    for key in dictionary:
        key_str = str(key).lower()  # Convert key to lowercase string for matching
        if any(sub in key_str for sub in possible_keys_lower):
            return key  # Return first matching key found

    return None  # No matching key found
###############################################################################

###############################################################################
def extract_options(
    user_kwargs: Dict[str, Any],
    default_dict: Dict[str, Any],
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Extract options from `user_kwargs` by overriding values in `default_dict` for keys
    optionally prefixed by `prefix`.

    Parameters
    ----------
    user_kwargs : dict
        Dictionary of user-supplied keyword arguments.

    default_dict : dict
        Dictionary of default options to be updated.

    prefix : str, optional
        Prefix to prepend to keys when looking them up in `user_kwargs` (default is "").

    Returns
    -------
    dict
        New dictionary with updated options from `user_kwargs`. 
        For each key in `default_dict`, the function first checks if `prefix + key` exists
        in `user_kwargs` and uses that value; otherwise, it checks for the key without prefix.

    Raises
    ------
    ValueError
        If inputs are not dictionaries or prefix is not a string.

    Examples
    --------
    >>> defaults = {'color': 'blue', 'linewidth': 2}
    >>> user_args = {'plot_color': 'red', 'linewidth': 3}
    >>> extract_options(user_args, defaults, prefix='plot_')
    {'color': 'red', 'linewidth': 3}
    >>> extract_options(user_args, defaults)
    {'color': 'blue', 'linewidth': 3}
    """
    # ===== VALIDATE INPUT TYPES =====
    if not isinstance(user_kwargs, dict):
        raise ValueError("Input 'user_kwargs' must be a dictionary.")
    if not isinstance(default_dict, dict):
        raise ValueError("Input 'default_dict' must be a dictionary.")
    if not isinstance(prefix, str):
        raise ValueError("Input 'prefix' must be a string.")

    # ===== COPY DEFAULTS AND OVERRIDE WITH USER VALUES =====
    result = default_dict.copy()  # Start with default options
    for key in default_dict:
        prefixed_key = f"{prefix}{key}"  # Compose prefixed key
        if prefixed_key in user_kwargs:
            result[key] = user_kwargs[prefixed_key]  # Override with prefixed key if present
        elif key in user_kwargs:
            result[key] = user_kwargs[key]  # Else override with non-prefixed key if present

    return result
###############################################################################

###############################################################################
def infer_years_from_path(
    directory: Union[str, Path],
    *,
    target_type: str = "file",
    pattern: str = r'_(\d{4})\.nc$',
    debug: bool = False
) -> Tuple[int, int, List[int]]:
    """
    Infer available years from directory content by matching a regex pattern on file or folder names.

    Parameters
    ----------
    directory : str or Path
        Directory path to scan.

    target_type : str, optional
        Type of items to scan in the directory: "file" or "folder". Default is "file".

    pattern : str, optional
        Regex pattern to extract year as a capturing group (e.g. r'_(\d{4})\.nc$' or r'output\s*(\d{4})').
        The year must be captured in the first group.

    debug : bool, optional
        If True, prints debug info.

    Returns
    -------
    Ybeg : int
        Earliest year found.

    Yend : int
        Latest year found.

    ysec : List[int]
        List of all years from Ybeg to Yend inclusive.

    Raises
    ------
    ValueError
        If no matching years are found or directory does not exist.
    """
    # ===== VALIDATE DIRECTORY =====
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory '{directory}' does not exist.")

    # ===== LIST TARGET ITEMS =====
    target_type = target_type.lower()
    if target_type == "file":
        items = [f for f in directory.iterdir() if f.is_file()]  # List files
    elif target_type == "folder":
        items = [d for d in directory.iterdir() if d.is_dir()]   # List directories
    else:
        raise ValueError(f"Invalid target_type '{target_type}'. Use 'file' or 'folder'.")

    # ===== COMPILE REGEX AND EXTRACT YEARS =====
    year_re = re.compile(pattern)

    # Use set comprehension with regex search to find all unique years in item names
    years_found = sorted({
        int(match.group(1)) 
        for item in items 
        if (match := year_re.search(item.name))
    })

    if debug:
        print(f"Scanned {len(items)} {target_type}s in {directory}")
        print(f"Found years: {years_found}")

    # ===== VALIDATE YEARS FOUND =====
    if not years_found:
        raise ValueError(f"No {target_type}s with year pattern '{pattern}' found in {directory}")

    # ===== RETURN YEAR RANGE AND SEQUENCE =====
    Ybeg, Yend = years_found[0], years_found[-1]   # Earliest and latest years
    ysec = list(range(Ybeg, Yend + 1))             # Full continuous list of years

    return Ybeg, Yend, ysec
###############################################################################

###############################################################################
def build_bfm_filename(year: int, filename_fragments: Dict[str, str]) -> str:
    """Construct BFM filename with given year and fragments."""
    return f"ADR{year}{filename_fragments['ffrag1']}{year}{filename_fragments['ffrag2']}{year}{filename_fragments['ffrag3']}.nc"
###############################################################################

###############################################################################
def temp_threshold(slice_data: np.ndarray, mask_shallow: np.ndarray, mask_deep: np.ndarray) -> np.ndarray:
    """
    Compute invalid mask for temperature based on shallow and deep thresholds.

    Parameters
    ----------
    slice_data : np.ndarray
        3D array of temperature data (Y, X).
    mask_shallow : np.ndarray
        Boolean mask where True corresponds to shallow depths.
    mask_deep : np.ndarray
        Boolean mask where True corresponds to deep depths.

    Returns
    -------
    np.ndarray
        Boolean mask of invalid temperature points.

    Example
    -------
    >>> import numpy as np
    >>> temp = np.array([[10, 36], [7, 9]])
    >>> shallow_mask = np.array([[True, True], [False, False]])
    >>> deep_mask = np.array([[False, False], [True, True]])
    >>> temp_threshold(temp, shallow_mask, deep_mask)
    array([[False,  True],
           [ True, False]])
    """
    # ===== APPLY SHALLOW THRESHOLD =====
    # Valid shallow temps: 5 < temp < 35; invert and restrict to shallow mask
    invalid_shallow = ~((slice_data > 5) & (slice_data < 35)) & mask_shallow

    # ===== APPLY DEEP THRESHOLD =====
    # Valid deep temps: 8 < temp < 25; invert and restrict to deep mask
    invalid_deep = ~((slice_data > 8) & (slice_data < 25)) & mask_deep

    # ===== COMBINE MASKS =====
    # Mark points invalid if they fail either shallow or deep threshold criteria
    invalid_mask = invalid_shallow | invalid_deep

    return invalid_mask
###############################################################################

###############################################################################
def hal_threshold(slice_data: np.ndarray, mask_shallow: np.ndarray, mask_deep: np.ndarray) -> np.ndarray:
    """
    Compute invalid mask for salinity based on shallow and deep thresholds.

    Parameters
    ----------
    slice_data : np.ndarray
        3D array of salinity data (Y, X).
    mask_shallow : np.ndarray
        Boolean mask where True corresponds to shallow depths.
    mask_deep : np.ndarray
        Boolean mask where True corresponds to deep depths.

    Returns
    -------
    np.ndarray
        Boolean mask of invalid salinity points.

    Example
    -------
    >>> import numpy as np
    >>> salinity = np.array([[26, 41], [37, 39]])
    >>> shallow_mask = np.array([[True, True], [False, False]])
    >>> deep_mask = np.array([[False, False], [True, True]])
    >>> hal_threshold(salinity, shallow_mask, deep_mask)
    array([[False,  True],
           [False, False]])
    """
    # ===== APPLY SHALLOW THRESHOLD =====
    # Valid shallow salinity: 25 < salinity < 40; invert and restrict to shallow mask
    invalid_shallow = ~((slice_data > 25) & (slice_data < 40)) & mask_shallow

    # ===== APPLY DEEP THRESHOLD =====
    # Valid deep salinity: 36 < salinity < 40; invert and restrict to deep mask
    invalid_deep = ~((slice_data > 36) & (slice_data < 40)) & mask_deep

    # ===== COMBINE MASKS =====
    # Mark points invalid if they fail either shallow or deep threshold criteria
    invalid_mask = invalid_shallow | invalid_deep

    return invalid_mask
###############################################################################

###############################################################################
def find_key_variable(nc_vars: Iterable[str], candidates: List[str]) -> str:
    """
    Return the first variable name found in nc_vars from candidates list,
    or raise KeyError if none found.

    Parameters
    ----------
    nc_vars : iterable
        Collection of variable names available (e.g., keys of a NetCDF dataset).
    candidates : list
        List of candidate variable names to search for.

    Returns
    -------
    str
        The first variable name found in nc_vars from the candidates list.

    Raises
    ------
    KeyError
        If none of the candidate variable names are found in nc_vars.

    Example
    -------
    >>> vars_available = ['temp', 'salinity', 'depth']
    >>> candidates = ['chlorophyll', 'salinity', 'temperature']
    >>> find_key_variable(vars_available, candidates)
    'salinity'
    """
    # ===== SEARCH FOR FIRST MATCH =====
    # Iterate over candidates, return first found variable in nc_vars
    found_var = next((v for v in candidates if v in nc_vars), None)

    # ===== HANDLE NO MATCH =====
    # Raise error if no candidate variable found
    if found_var is None:
        raise KeyError(
            f"\033[91m❌ None of the variables {candidates} found in the dataset\033[0m"
        )

    return found_var
###############################################################################

###############################################################################
def _to_dataarray(
    val: Union[float, int, xr.DataArray], 
    reference_da: xr.DataArray
) -> xr.DataArray:
    """
    Ensure output is an xarray.DataArray, broadcast to reference lat/lon shape if needed.

    Parameters
    ----------
    val : scalar or xarray.DataArray
        Value to convert or broadcast. Must be scalar if not already a DataArray.
    reference_da : xarray.DataArray
        Reference DataArray providing target shape and coordinates for broadcasting.

    Returns
    -------
    xarray.DataArray
        DataArray matching the spatial dimensions of reference_da, containing val.

    Raises
    ------
    ValueError
        If val is neither a scalar nor a DataArray.

    Example
    -------
    >>> import xarray as xr
    >>> ref = xr.DataArray(np.zeros((5, 10)), dims=('lat', 'lon'))
    >>> _to_dataarray(3.14, ref)
    <xarray.DataArray (lat: 5, lon: 10)>
    array([[3.14, 3.14, ..., 3.14]])
    Coordinates:
      * lat      (lat) int64 0 1 2 3 4
      * lon      (lon) int64 0 1 2 3 4 5 6 7 8 9
    """
    # ===== RETURN IF ALREADY DATAARRAY =====
    # Return val directly if it is already an xarray.DataArray
    if isinstance(val, xr.DataArray):
        return val

    # ===== CHECK SCALAR =====
    # Only allow scalar values to be broadcasted
    if not np.isscalar(val):
        raise ValueError(f"Expected scalar or DataArray, got {type(val)}")

    # ===== SELECT REFERENCE SLICE =====
    # If 'time' dim exists, use first time slice as shape template
    if 'time' in reference_da.dims:
        ref = reference_da.isel(time=0)
    else:
        ref = reference_da

    # ===== BROADCAST SCALAR TO DATAARRAY =====
    # Create DataArray full of val with shape of ref
    return xr.full_like(ref, val)
###############################################################################

###############################################################################
def check_numeric_data(data_dict: Dict[str, Dict[int, List[np.ndarray]]]) -> None:
    """
    Validate that all monthly data arrays in the input dictionary contain numeric data
    and follow the expected structure (12 numpy arrays per year per key).

    This function checks that the data dictionary for 'model' and 'satellite' contains,
    for each year, exactly 12 monthly numpy arrays. It raises errors if any array is
    non-numeric or if the structure is invalid.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys such as 'model' and 'satellite', each mapping to a dictionary
        where keys are years (int) and values are lists of 12 numpy arrays (one per month).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If data for any year and key is not a list or tuple of length 12.
        If any monthly numpy array contains non-numeric data.

    Example
    -------
    >>> data = {
    ...     'model': {2000: [np.array([1.0, 2.0])] + [np.array([])] * 11},
    ...     'satellite': {2000: [np.array([1.1, 2.1])] + [np.array([])] * 11}
    ... }
    >>> check_numeric_data(data)  # passes silently if valid
    """
    for key in ['model', 'satellite']:
        if key not in data_dict:
            continue
        for year, monthly_arrays in data_dict[key].items():
            if not isinstance(monthly_arrays, (list, tuple)) or len(monthly_arrays) != 12:
                raise ValueError(f"Data for year {year} under '{key}' must be a list or tuple of 12 numpy arrays")
            for month_idx, arr in enumerate(monthly_arrays):
                if arr.size > 0 and not np.issubdtype(arr.dtype, np.number):
                    raise ValueError(f"Data for year {year}, month {month_idx} under '{key}' must contain numeric data")