from typing import List, Optional, Dict, Any

###############################################################################
def find_key(dictionary: Dict[Any, Any], 
             possible_keys: List[str]) -> Optional[str]:
    """
    Find the first key in a dictionary containing any of the substrings in possible_keys (case insensitive).

    Parameters
    ----------
    dictionary : dict
        Dictionary to search keys in.

    possible_keys : List[str]
        List of substrings to look for in the dictionary keys.

    Returns
    -------
    Optional[str]
        The first matching key found that contains any substring from possible_keys (case insensitive),
        or None if no key matches.

    Raises
    ------
    ValueError
        If `dictionary` is not a dict or `possible_keys` is not a list of strings.

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
    if not (isinstance(possible_keys, list) and all(isinstance(k, str) for k in possible_keys)):
        raise ValueError("Input 'possible_keys' must be a list of strings.")

    for key in dictionary:
        lowered = str(key).lower()
        if any(sub.lower() in lowered for sub in possible_keys):
            return key
    return None
###############################################################################

###############################################################################
def extract_options(user_kwargs: Dict[str, Any],
                    default_dict: Dict[str, Any],
                    prefix: str = "") -> Dict[str, Any]:
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
        New dictionary with updated options from `user_kwargs` if keys (with prefix) exist.

    Raises
    ------
    ValueError
        If inputs are not dictionaries or prefix is not a string.

    Examples
    --------
    >>> defaults = {'color': 'blue', 'linewidth': 2}
    >>> user_args = {'plot_color': 'red', 'linewidth': 3}
    >>> extract_options(user_args, defaults, prefix='plot_')
    {'color': 'red', 'linewidth': 2}
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
        full_key = f"{prefix}{key}"
        if full_key in user_kwargs:
            result[key] = user_kwargs[full_key]
    return result
###############################################################################