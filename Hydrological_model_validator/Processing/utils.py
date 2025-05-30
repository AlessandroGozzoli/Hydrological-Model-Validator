from typing import List, Optional, Dict, Any, Union, Tuple, Iterable
from pathlib import Path
import re
import numpy as np

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
    if not isinstance(dictionary, dict):
        raise ValueError("Input 'dictionary' must be a dictionary.")
    if not (hasattr(possible_keys, '__iter__') and all(isinstance(k, str) for k in possible_keys)):
        raise ValueError("Input 'possible_keys' must be an iterable of strings.")

    possible_keys_lower = [sub.lower() for sub in possible_keys]

    for key in dictionary:
        key_str = str(key).lower()
        if any(sub in key_str for sub in possible_keys_lower):
            return key

    return None
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
    if not isinstance(user_kwargs, dict):
        raise ValueError("Input 'user_kwargs' must be a dictionary.")
    if not isinstance(default_dict, dict):
        raise ValueError("Input 'default_dict' must be a dictionary.")
    if not isinstance(prefix, str):
        raise ValueError("Input 'prefix' must be a string.")

    result = default_dict.copy()
    for key in default_dict:
        prefixed_key = f"{prefix}{key}"
        if prefixed_key in user_kwargs:
            result[key] = user_kwargs[prefixed_key]
        elif key in user_kwargs:
            result[key] = user_kwargs[key]

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
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory '{directory}' does not exist.")

    target_type = target_type.lower()
    if target_type == "file":
        items = [f for f in directory.iterdir() if f.is_file()]
    elif target_type == "folder":
        items = [d for d in directory.iterdir() if d.is_dir()]
    else:
        raise ValueError(f"Invalid target_type '{target_type}'. Use 'file' or 'folder'.")

    year_re = re.compile(pattern)
    
    # Extract years by searching regex pattern on each item name
    years_found = sorted({
        int(match.group(1))
        for item in items
        if (match := year_re.search(item.name))
    })

    if debug:
        print(f"Scanned {len(items)} {target_type}s in {directory}")
        print(f"Found years: {years_found}")

    if not years_found:
        raise ValueError(f"No {target_type}s with year pattern '{pattern}' found in {directory}")

    Ybeg, Yend = years_found[0], years_found[-1]
    ysec = list(range(Ybeg, Yend + 1))

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
    """
    invalid_shallow = ~((slice_data > 5) & (slice_data < 35)) & mask_shallow
    invalid_deep = ~((slice_data > 8) & (slice_data < 25)) & mask_deep
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
    """
    invalid_shallow = ~((slice_data > 25) & (slice_data < 40)) & mask_shallow
    invalid_deep = ~((slice_data > 36) & (slice_data < 40)) & mask_deep
    invalid_mask = invalid_shallow | invalid_deep
    return invalid_mask
###############################################################################

###############################################################################
def find_key_variable(nc_vars, candidates):
    """
    Return the first variable name found in nc_vars from candidates list,
    or raise KeyError if none found.
    """
    found_var = next((v for v in candidates if v in nc_vars), None)
    if found_var is None:
        raise KeyError(
            f"\033[91m❌ None of the variables {candidates} found in the dataset\033[0m"
        )
    return found_var