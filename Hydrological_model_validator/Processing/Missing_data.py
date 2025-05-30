import numpy as np
from datetime import datetime
from typing import Tuple, List

###############################################################################
def check_missing_days(
    T_orig: np.ndarray, 
    data_orig: np.ndarray,
    desired_start_date: datetime = datetime(2000, 1, 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill missing days in a time series based on expected daily intervals.
    Automatically shifts the time series so the first timestamp corresponds to desired_start_date.

    Parameters
    ----------
    T_orig : np.ndarray
        1D array of timestamps (in seconds since epoch).
    data_orig : np.ndarray
        3D array with shape (time, lat, lon), matching T_orig in the first dimension.
    desired_start_date : datetime, optional
        The desired start date to align the time series to (default is 2000-01-01).

    Returns
    -------
    Ttrue : np.ndarray
        Complete array of timestamps with daily spacing, shifted to desired_start_date.
    data_complete : np.ndarray
        Data array with NaNs inserted where gaps were detected.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input shapes are invalid or time step is non-positive.
    AssertionError
        If time series is out of order or length mismatch occurs.
    """
    # === INPUT VALIDATION ===
    if not isinstance(T_orig, np.ndarray):
        raise TypeError("T_orig must be a NumPy array.")
    if not isinstance(data_orig, np.ndarray):
        raise TypeError("data_orig must be a NumPy array.")
    if T_orig.ndim != 1:
        raise ValueError("T_orig must be 1-dimensional.")
    if data_orig.ndim != 3:
        raise ValueError("data_orig must be 3-dimensional.")
    if len(T_orig) != data_orig.shape[0]:
        raise ValueError("T_orig length must match first dimension of data_orig.")

    print(f"Original time steps: {len(T_orig)}")
    first_date = datetime.utcfromtimestamp(T_orig[0])
    print("First timestamp (original):", T_orig[0], f"({first_date.strftime('%Y-%m-%d')})")

    # === SHIFT TIME SERIES TO DESIRED START DATE ===
    offset_seconds = (desired_start_date - first_date).total_seconds()
    print(f"Offset to add (seconds): {offset_seconds}")

    T_shifted = T_orig + offset_seconds
    shifted_first_date = datetime.utcfromtimestamp(T_shifted[0])
    print("First timestamp (shifted):", T_shifted[0], f"({shifted_first_date.strftime('%Y-%m-%d')})")

    # === DETECT TIME STEP ===
    steps = np.diff(T_shifted)
    if not np.all(steps > 0):
        raise ValueError("Timestamps must be strictly increasing after shifting.")
    
    unique_steps, counts = np.unique(steps, return_counts=True)
    time_step = unique_steps[np.argmax(counts)]

    if time_step <= 0:
        raise ValueError("Detected non-positive time step.")

    print(f"Detected time step: {time_step} seconds ({time_step / 86400:.2f} days)")

    # === GENERATE EXPECTED TIME RANGE ===
    T_start, T_end = T_shifted[0], T_shifted[-1]
    Ttrue = np.arange(T_start, T_end + time_step, time_step)
    missing_count = len(Ttrue) - len(T_shifted)

    if missing_count == 0:
        print("\033[92m✅ The time series is complete!\033[0m")
        print('*' * 45)
        return T_shifted.copy(), data_orig.copy()

    print("\033[91m⚠️ Time series is incomplete!\033[0m")
    print(f"\033[91mMissing {missing_count} time steps detected.\033[0m")
    print("Filling missing time steps...")
    print('-' * 45)

    # === FILL DATA INTO COMPLETE TIME SERIES ===
    time_index = {t: i for i, t in enumerate(T_shifted)}
    data_shape = (len(Ttrue), *data_orig.shape[1:])
    data_complete = np.full(data_shape, np.nan, dtype=data_orig.dtype)

    for i, t in enumerate(Ttrue):
        idx = time_index.get(t)
        if idx is not None:
            data_complete[i] = data_orig[idx]
        else:
            dt = datetime.utcfromtimestamp(t)
            print(f"Missing day filled: {dt.strftime('%Y-%m-%d')}")

    # === FINAL SANITY CHECK ===
    if data_complete.shape[0] != len(Ttrue):
        raise AssertionError("Mismatch in expected time series length after filling.")

    print("\033[92m✅ Time series gaps filled successfully.\033[0m")
    print('*' * 45)

    return Ttrue, data_complete

###############################################################################

###############################################################################
def find_missing_observations(data_complete: np.ndarray) -> Tuple[int, List[int]]:
    """
    Identify days with no satellite observations (all NaN or zero).

    Parameters
    ----------
    data_complete : np.ndarray
        3D array of shape (days, lat, lon) with satellite data.

    Returns
    -------
    cnan : int
        Number of days with no satellite observations.
    satnan : List[int]
        Indices of days with no valid observations.

    Raises
    ------
    TypeError
        If input is not a NumPy array.
    ValueError
        If input array does not have 3 dimensions.
    """
    if not isinstance(data_complete, np.ndarray):
        raise TypeError("data_complete must be a NumPy array.")
    if data_complete.ndim != 3:
        raise ValueError("data_complete must be a 3D array (days, lat, lon).")

    print("Checking for missing satellite observations...")

    # Compute daily sums ignoring NaNs, zero sum means all values are NaN or zero
    daily_sums = np.nansum(data_complete, axis=(1, 2))
    mask_missing = daily_sums == 0

    cnan = int(np.sum(mask_missing))
    satnan = np.flatnonzero(mask_missing).tolist()

    if cnan > 0:
        print(f"\033[91m⚠️ {cnan} daily satellite fields have no observations ⚠️\033[0m")
    else:
        print("\033[92m✅ No missing satellite observation days found\033[0m")

    print('*' * 45)
    return cnan, satnan
###############################################################################

###############################################################################
def eliminate_empty_fields(data_complete: np.ndarray) -> np.ndarray:
    """
    Replace empty chlorophyll fields (days where all values are NaN or zero) with NaNs.

    Parameters
    ----------
    data_complete : np.ndarray
        3D array of chlorophyll data with shape (days, lat, lon).

    Returns
    -------
    np.ndarray
        Modified array with empty daily fields set entirely to NaN.

    Raises
    ------
    TypeError
        If input is not a NumPy array.
    ValueError
        If input array does not have 3 dimensions.

    Examples
    --------
    >>> data = np.random.rand(10, 5, 5)
    >>> data[3] = 0  # simulate empty day
    >>> data[7] = np.nan
    >>> result = eliminate_empty_fields(data)
    """
    if not isinstance(data_complete, np.ndarray):
        raise TypeError("data_complete must be a NumPy array.")
    if data_complete.ndim != 3:
        raise ValueError("data_complete should be a 3D array (days, lat, lon).")

    print("Checking and removing empty fields...")

    # Identify empty days: all NaN or all zero (ignoring NaNs for zero check)
    all_nan = np.isnan(data_complete).all(axis=(1, 2))

    # Suppress the all-NaN slice warning temporarily
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_vals = np.nanmax(data_complete, axis=(1, 2))

    all_zero = (max_vals == 0)
    
    empty_mask = all_nan | all_zero
    cempty = np.sum(empty_mask)

    if cempty > 0:
        # Set entire days flagged as empty to NaN
        data_complete[empty_mask] = np.nan
        for day_idx in np.flatnonzero(empty_mask):
            print(f"\033[91m⚠️ Empty field found at day {day_idx + 1} — replaced with NaNs ⚠️\033[0m")
        print(f"\033[93m{cempty} empty fields were found and corrected\033[0m")
    else:
        print("\033[92m✅ No empty fields found in dataset\033[0m")

    print('*' * 45)
    return data_complete