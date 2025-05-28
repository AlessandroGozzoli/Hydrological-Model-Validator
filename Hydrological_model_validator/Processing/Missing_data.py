import numpy as np
from datetime import datetime

###############################################################################
def check_missing_days(T_orig, data_orig):
    """
    Checks and fills missing days in a time series based on expected daily intervals.

    Parameters
    ----------
    T_orig : array-like
        1D array of timestamps (in seconds since epoch).
    data_orig : ndarray
        3D array with shape (time, lat, lon), matching T_orig in first dimension.

    Returns
    -------
    Ttrue : np.ndarray
        Complete array of timestamps with gaps filled.
    data_complete : np.ndarray
        Data array with NaNs filled where gaps were detected.

    Raises
    ------
    AssertionError
        If input validation or gap-filling logic fails.
    """
    SZT_orig = len(T_orig)

    assert SZT_orig > 0, "❌ T_orig is empty — no time data provided"
    assert data_orig.shape[0] == SZT_orig, "❌ Time and data length mismatch"

    # Determine time step (assume most common step)
    steps = np.diff(T_orig)
    unique_steps, counts = np.unique(steps, return_counts=True)
    SinD = unique_steps[np.argmax(counts)]
    assert SinD > 0, "❌ SinD must be a positive interval in seconds"

    # Determine true length
    total_duration = int((T_orig[-1] - T_orig[0]) // SinD + 1)
    Truedays = total_duration

    if Truedays == SZT_orig:
        print("\033[92m✅ The time series is complete!\033[0m")
        print('*'*45)
        return T_orig.copy(), data_orig.copy()

    # Begin gap-filling process
    totdiffD = Truedays - SZT_orig
    print("\033[91m⚠️ The time series is not complete ⚠️\033[0m")
    print(f"\033[91mThere are {totdiffD} missing days!\033[0m")
    print("Proceeding to search...")
    print('-'*45)

    Tcount = 0
    Ttrue = np.full(Truedays, np.nan)
    data_complete = np.full((Truedays, *data_orig.shape[1:]), np.nan)

    Ttrue[Tcount] = T_orig[0]
    data_complete[Tcount, :, :] = data_orig[0, :, :]

    cmiss = 0
    breakloop = False

    for d in range(1, SZT_orig):
        if cmiss == totdiffD:
            TbTData = Truedays - (Tcount + 1)
            dleft = SZT_orig - d
            assert dleft == TbTData, "❌ Remaining days don't match the expected count"
            data_complete[Tcount+1:Truedays, :, :] = data_orig[d:SZT_orig, :, :]
            Ttrue[Tcount+1:Truedays] = T_orig[d:SZT_orig]
            print(f"✅ \033[92mThe remaining {TbTData} have been transferred\033[0m")
            print('-'*45)
            breakloop = True
            break

        dayjump = T_orig[d-1] + SinD
        if T_orig[d] == dayjump:
            Tcount += 1
            Ttrue[Tcount] = T_orig[d]
            data_complete[Tcount, :, :] = data_orig[d, :, :]
        elif T_orig[d] > dayjump:
            dgap = int((T_orig[d] - T_orig[d-1]) // SinD - 1)
            print(f"Found a gap of {dgap} days. The missing dates are:")

            cmiss += dgap
            incTstep = T_orig[d-1]

            for gap in range(dgap):
                incTstep += SinD
                missingdate = datetime.utcfromtimestamp(int(incTstep))
                print(missingdate)
                Tcount += 1
                Ttrue[Tcount] = incTstep
                data_complete[Tcount, :, :] = np.nan

            Tcount += 1
            Ttrue[Tcount] = T_orig[d]
            data_complete[Tcount, :, :] = data_orig[d, :, :]
        else:
            raise AssertionError(
                f"❌ Unexpected date order: diff = {T_orig[d] - T_orig[d-1]}. Check your input data."
            )

        if breakloop:
            break
        
    print("✅ \033[92mThe gaps have been filled\033[0m")
    print('*' * 45)

    return Ttrue, data_complete
###############################################################################

###############################################################################
def find_missing_observations(data_complete):
    """
    Identifies days with no satellite observations (all NaNs or 0s in a day).
    
    Parameters:
    Schl_complete (np.ndarray): 3D array of shape (days, lat, lon) with satellite chlorophyll data.
    Truedays (int): Expected number of total days.

    Returns:
    cnan (int): Count of days with no satellite observations.
    satnan (list): Indices of days with no observations.
    """
    assert isinstance(data_complete, np.ndarray), "❌ Schl_complete must be a NumPy array"
    assert data_complete.ndim == 3, "❌ Schl_complete should be a 3D array (days, lat, lon)"
    
    print("Checking for missing satellite observations...")

    # Create mask for missing days
    daily_sums = np.nansum(data_complete, axis=(1, 2))
    mask_missing = (daily_sums == 0)
    cnan = np.sum(mask_missing)
    satnan = np.where(mask_missing)[0].tolist()

    if cnan > 0:
        print(f"\033[91m⚠️ {cnan} daily satellite fields have no observations ⚠️\033[0m")
    else:
        print("\033[92m✅ No missing satellite observation days found\033[0m")

    print('*' * 45)
    return cnan, satnan
###############################################################################

###############################################################################
def eliminate_empty_fields(data_complete):
    """
    Replaces empty chlorophyll fields (where all values are zero or NaN) with NaNs.
    
    Parameters:
    Schl_complete (np.ndarray): 3D array of chlorophyll data (days, lat, lon).
    Truedays (int): Total number of expected days.
    
    Returns:
    Schl_complete (np.ndarray): Modified array with empty fields set to NaN.
    """
    assert isinstance(data_complete, np.ndarray), "❌ Schl_complete must be a NumPy array"
    assert data_complete.ndim == 3, "❌ Schl_complete should be a 3D array (days, lat, lon)"


    print("Checking and removing empty chlorophyll fields...")

    cempty = 0
    for d in range(data_complete.shape[0]):
        day_slice = data_complete[d, :, :]

        # Check if entire slice is NaN
        if np.isnan(day_slice).all() or np.nanmax(day_slice) == 0:
            cempty += 1
            data_complete[d, :, :] = np.nan
            print(f"\033[91m⚠️ Empty field found at day {d + 1} — replaced with NaNs ⚠️\033[0m")

    if cempty == 0:
        print("\033[92m✅ No empty fields found in dataset\033[0m")
    else:
        print(f"\033[93m{cempty} empty fields were found and corrected\033[0m")
    
    print('*' * 45)
    return data_complete