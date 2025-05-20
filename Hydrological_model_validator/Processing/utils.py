from typing import List, Optional

###############################################################################
def find_key(dictionary: dict, possible_keys: List[str]) -> Optional[str]:
    """
    Find the first key in dictionary containing any of the substrings in possible_keys (case insensitive).

    Args:
        dictionary (dict): Dictionary to search keys in.
        possible_keys (List[str]): List of substrings to look for in keys.

    Returns:
        Optional[str]: The first matching key, or None if not found.
    """
    for key in dictionary:
        lowered = key.lower()
        if any(sub in lowered for sub in possible_keys):
            return key
    return None
###############################################################################