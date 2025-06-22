###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Data handling libraries
import numpy as np
from datetime import datetime
from typing import Tuple, List

# Logging and tracing
import logging
from eliot import start_action, log_message

# Module utilities
from Hydrological_model_validator.Processing.time_utils import Timer

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
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
    # Ensure T_orig is a NumPy array
    if not isinstance(T_orig, np.ndarray):
        raise TypeError("❌ T_orig must be a NumPy array. ❌")
    # Ensure data_orig is a NumPy array
    if not isinstance(data_orig, np.ndarray):
        raise TypeError("❌ data_orig must be a NumPy array. ❌")
    # Ensure T_orig is 1-dimensional
    if T_orig.ndim != 1:
        raise ValueError("❌ T_orig must be 1-dimensional. ❌")
    # Ensure data_orig is 3-dimensional (time, lat, lon)
    if data_orig.ndim != 3:
        raise ValueError("❌ data_orig must be 3-dimensional. ❌")
    # Ensure the length of T_orig matches the first dimension of data_orig
    if len(T_orig) != data_orig.shape[0]:
        raise ValueError("❌ T_orig length must match first dimension of data_orig. ❌")

    with Timer("check_missing_days function"):
        with start_action(
            action_type="check_missing_days function",
            original_time_steps=len(T_orig),
            desired_start_date=desired_start_date.strftime("%Y-%m-%d")
        ):
            print(f"Original time steps: {len(T_orig)}")
            first_date = datetime.utcfromtimestamp(T_orig[0])
            print("First timestamp (original):", T_orig[0], f"({first_date.strftime('%Y-%m-%d')})")

            # === SHIFT TIME SERIES TO DESIRED START DATE ===
            # Calculate offset in seconds to shift the time series start date
            offset_seconds = (desired_start_date - first_date).total_seconds()
            print(f"Offset to add (seconds): {offset_seconds}")

            # Apply offset to all timestamps to align start date to desired_start_date
            T_shifted = T_orig + offset_seconds
            shifted_first_date = datetime.utcfromtimestamp(T_shifted[0])
            print("First timestamp (shifted):", T_shifted[0], f"({shifted_first_date.strftime('%Y-%m-%d')})")

            log_message(
                "Shifted time series start",
                original_first_date=first_date.strftime("%Y-%m-%d"),
                shifted_first_date=shifted_first_date.strftime("%Y-%m-%d"),
                offset_seconds=offset_seconds
            )
            logging.info(f"Shifted time series start from {first_date} to {shifted_first_date}")

            # === DETECT TIME STEP ===
            # Calculate differences between consecutive timestamps to find sampling interval
            steps = np.diff(T_shifted)
            # Ensure timestamps strictly increase (no duplicates or out-of-order)
            if not np.all(steps > 0):
                raise ValueError("❌ Timestamps must be strictly increasing after shifting. ❌")
            
            # Identify the most frequent time step as expected sampling interval
            unique_steps, counts = np.unique(steps, return_counts=True)
            time_step = unique_steps[np.argmax(counts)]

            # Validate that the detected time step is positive
            if time_step <= 0:
                raise ValueError("❌ Detected non-positive time step. ❌")

            print(f"Detected time step: {time_step} seconds ({time_step / 86400:.2f} days)")
            logging.info(f"Detected time step: {time_step} seconds ({time_step / 86400:.2f} days)")
            log_message("Detected time step", time_step_seconds=time_step)

            # === GENERATE EXPECTED TIME RANGE ===
            # Define full expected time range based on detected time step
            T_start, T_end = T_shifted[0], T_shifted[-1]
            Ttrue = np.arange(T_start, T_end + time_step, time_step)  # Include last day by adding time_step

            # Calculate how many timestamps are missing compared to ideal timeline
            missing_count = len(Ttrue) - len(T_shifted)

            if missing_count == 0:
                print("\033[92m✅ The time series is complete!\033[0m")
                print('*' * 45)
                log_message("Time series complete", time_steps=len(Ttrue))
                logging.info("✅ The time series is complete!")
                # Return the shifted original data since there are no missing entries
                return T_shifted.copy(), data_orig.copy()

            print("\033[91m⚠️ Time series is incomplete!\033[0m")
            print(f"\033[91mMissing {missing_count} time steps detected.\033[0m")
            print("Filling missing time steps...")
            print('-' * 45)
            log_message("Time series incomplete", missing_steps=missing_count)
            logging.warning(f"⚠️ Missing {missing_count} time steps detected. Filling missing data.")

            # === FILL DATA INTO COMPLETE TIME SERIES ===
            # Create a map from timestamp to index for quick lookup
            time_index = {t: i for i, t in enumerate(T_shifted)}
            # Prepare an output array initialized with NaNs for missing entries
            data_shape = (len(Ttrue), *data_orig.shape[1:])
            data_complete = np.full(data_shape, np.nan, dtype=data_orig.dtype)

            # Fill data where available, leave NaNs where timestamps are missing
            missing_days_logged = []
            for i, t in enumerate(Ttrue):
                idx = time_index.get(t)
                if idx is not None:
                    data_complete[i] = data_orig[idx]
                else:
                    dt = datetime.utcfromtimestamp(t)
                    missing_days_logged.append(dt.strftime('%Y-%m-%d'))

            for day in missing_days_logged:
                print(f"Missing day filled: {day}")

            # === FINAL SANITY CHECK ===
            # Confirm that output data length matches the complete timeline length
            if data_complete.shape[0] != len(Ttrue):
                raise AssertionError("❌ Mismatch in expected time series length after filling. ❌")

            print("\033[92m✅ Time series gaps filled successfully.\033[0m")
            print('*' * 45)
            log_message("Time series gaps filled successfully", filled_length=len(Ttrue))
            logging.info("✅ Time series gaps filled successfully.")

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
    # ==== INPUT VALIDATION ====
    # Ensure input is a NumPy array
    if not isinstance(data_complete, np.ndarray):
        raise TypeError("❌ data_complete must be a NumPy array. ❌")
    # Check that input has three dimensions (days, lat, lon)
    if data_complete.ndim != 3:
        raise ValueError("❌ data_complete must be a 3D array (days, lat, lon). ❌")

    with Timer("find_missing_observations function"):
        with start_action(
            action_type="find_missing_observations function",
            data_shape=data_complete.shape
        ):
            print("Checking for missing satellite observations...")

            # Sum over spatial dimensions ignoring NaNs to detect days with no data
            daily_sums = np.nansum(data_complete, axis=(1, 2))
            # Days where sum is zero mean all values are either NaN or zero (no observations)
            mask_missing = daily_sums == 0

            # Count how many days have missing data
            cnan = int(np.sum(mask_missing))
            # Get list of indices for days with missing observations
            satnan = np.flatnonzero(mask_missing).tolist()

            if cnan > 0:
                print(f"\033[91m⚠️ {cnan} daily satellite fields have no observations ⚠️\033[0m")
                logging.warning(f"{cnan} daily satellite fields have no observations")
                log_message("Missing satellite observations found", count=cnan, indices=satnan)
            else:
                print("\033[92m✅ No missing satellite observation days found\033[0m")
                logging.info("No missing satellite observation days found")
                log_message("No missing satellite observations")

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
    # ===== INPUT VALIDATION =====
    # Validate input is a NumPy array
    if not isinstance(data_complete, np.ndarray):
        raise TypeError("❌ data_complete must be a NumPy array. ❌")
    # Validate input is 3D (days, lat, lon)
    if data_complete.ndim != 3:
        raise ValueError("❌ data_complete should be a 3D array (days, lat, lon). ❌")

    with Timer("eliminate_empty_fields function"):
        with start_action(
            action_type="eliminate_empty_fields function",
            data_shape=data_complete.shape
        ):
            print("Checking and removing empty fields...")

            # Identify days where all values are NaN across spatial dimensions
            all_nan = np.isnan(data_complete).all(axis=(1, 2))

            # Suppress warnings for empty slices when computing max
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Get max value per day ignoring NaNs to detect days that might be all zero
                max_vals = np.nanmax(data_complete, axis=(1, 2))

            # Identify days where max value is zero => all values are zero or NaN
            all_zero = (max_vals == 0)

            # Combine masks to find days that are either all NaN or all zero
            empty_mask = all_nan | all_zero
            # Count how many empty days found
            cempty = np.sum(empty_mask)

            if cempty > 0:
                # Replace entire daily fields flagged as empty with NaNs
                data_complete[empty_mask] = np.nan
                # Inform user about which days were corrected (1-based indexing for clarity)
                for day_idx in np.flatnonzero(empty_mask):
                    print(f"\033[91m⚠️ Empty field found at day {day_idx + 1} — replaced with NaNs ⚠️\033[0m")
                print(f"\033[93m{cempty} empty fields were found and corrected\033[0m")
                logging.warning(f"{cempty} empty fields found and replaced with NaNs")
                log_message("Empty fields replaced", count=int(cempty), indices=np.flatnonzero(empty_mask).tolist())
            else:
                print("\033[92m✅ No empty fields found in dataset\033[0m")
                logging.info("No empty fields found in dataset")
                log_message("No empty fields detected")

            print('*' * 45)

            return data_complete