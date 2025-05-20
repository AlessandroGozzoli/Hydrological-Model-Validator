import numpy as np
from datetime import datetime, timedelta

def check_missing_days(T_orig, Schl_orig, Truedays, SinD, startingyear, startingmonth, startingday):
    SZT_orig = len(T_orig)
    
    assert SZT_orig > 0, "❌ T_orig is empty — no time data provided"
    assert Schl_orig.shape[0] == SZT_orig, "❌ Time and data length mismatch"
    assert isinstance(Truedays, int) and Truedays > 0, "❌ Truedays must be a positive integer"
    assert SinD > 0, "❌ SinD must be a positive interval in seconds"
    
    if Truedays == SZT_orig:
        Ttrue = T_orig.copy()
        Schl_complete = Schl_orig.copy()
        print("\033[92m✅ The time series is complete!\033[0m")
        print('*'*45)
        return T_orig.copy(), Schl_orig.copy()
    
    elif SZT_orig < Truedays:
        # Missing days detected
        totdiffD = Truedays - SZT_orig
        print("\033[91m⚠️ The time series is not complete ⚠️\033[0m")
        print(f"\033[91mThere are {totdiffD} missing days!\033[0m")
        print("Proceeding to search...")
        print('-'*45)
        
        Tcount = 0
        Ttrue = np.full(Truedays, np.nan)
        Schl_complete = np.full((Truedays, *Schl_orig.shape[1:]), np.nan)
        
        Ttrue[Tcount] = T_orig[0]
        Schl_complete[Tcount, :, :] = Schl_orig[0, :, :]
        
        cmiss = 0
        breakloop = False
        
        for d in range(1,SZT_orig):
            if cmiss == totdiffD:
                print("✅ \033[92mGap search is over\033[0m")
                print('-'*45)
                
                TbTData = Truedays - (Tcount + 1)
                print(f"Still {TbTData} data to be transferred")
                print("Transferring remaining days...")
                dleft = SZT_orig - d
                assert dleft == TbTData, "❌ Remaining days don't match the expected count"
                
                Schl_complete[Tcount+1:Truedays, :, :] = Schl_orig[d:SZT_orig, :, :]
                Tcount += TbTData
                
                if dleft == TbTData:
                    print(f"✅ \033[92mThe remaining {TbTData} have been transferred\033[0m")
                    print('-'*45)
                    breakloop = True
                else:
                    assert dleft != TbTData, "❌ Remaining days don't match the expected count"
                break
                
            dayjump = T_orig[d-1] + SinD
            
            if T_orig[d] == dayjump:
                Tcount += 1
                Ttrue[Tcount] = T_orig[d]
                Schl_complete[Tcount, :, :] = Schl_orig[d, :, :]
                
            elif T_orig[d] > dayjump:
                dgap = int(((T_orig[d] - T_orig[d-1]) / SinD) - 1)
                print(f"Found a gap of {dgap} days. The missing dates are:")
                
                cmiss += dgap
                incTstep = T_orig[d-1]
                
                for gap in range(dgap):
                    incTstep += SinD
                    missingdate = datetime(startingyear, startingmonth, startingday) + timedelta(seconds=int(incTstep))
                    print(missingdate)
                    print(f"This is happening at iteration {d}")
                    print('-'*45)
                    
                    Tcount += 1
                    Ttrue[Tcount] = Ttrue[Tcount-1] + SinD
                    Schl_complete[Tcount, :, :] = np.nan
                
                Tcount += 1
                Ttrue[Tcount] = T_orig[d]
                Schl_complete[Tcount, :, :] = Schl_orig[d, :, :]
            
            else:
                raise AssertionError(
                    f"❌ Unexpected date order: diff = {T_orig[d] - T_orig[d-1]}. Check your input data."
                )
            
            if breakloop:
                break
        
        assert Tcount + 1 == Truedays, "❌ \033[91mProblem in gap filling — mismatch in expected day count\033[0m"

        print("✅ \033[92mThe gaps have been filled\033[0m")
        print('*' * 45)
            
    return Ttrue, Schl_complete

def find_missing_observations(Schl_complete, Truedays):
    """
    Identifies days with no satellite observations (all NaNs or 0s in a day).
    
    Parameters:
    Schl_complete (np.ndarray): 3D array of shape (days, lat, lon) with satellite chlorophyll data.
    Truedays (int): Expected number of total days.

    Returns:
    cnan (int): Count of days with no satellite observations.
    satnan (list): Indices of days with no observations.
    """
    assert isinstance(Schl_complete, np.ndarray), "❌ Schl_complete must be a NumPy array"
    assert Schl_complete.ndim == 3, "❌ Schl_complete should be a 3D array (days, lat, lon)"
    assert Schl_complete.shape[0] == Truedays, (
        f"❌ Mismatch: Schl_complete has {Schl_complete.shape[0]} days, expected {Truedays}"
    )
    
    print("Checking for missing satellite observations...")

    # Create mask for missing days
    daily_sums = np.nansum(Schl_complete, axis=(1, 2))
    mask_missing = (daily_sums == 0)
    cnan = np.sum(mask_missing)
    satnan = np.where(mask_missing)[0].tolist()

    if cnan > 0:
        print(f"\033[91m⚠️ {cnan} daily satellite fields have no observations ⚠️\033[0m")
    else:
        print("\033[92m✅ No missing satellite observation days found\033[0m")

    print('*' * 45)
    return cnan, satnan

def eliminate_empty_fields(Schl_complete, Truedays):
    """
    Replaces empty chlorophyll fields (where all values are zero or NaN) with NaNs.
    
    Parameters:
    Schl_complete (np.ndarray): 3D array of chlorophyll data (days, lat, lon).
    Truedays (int): Total number of expected days.
    
    Returns:
    Schl_complete (np.ndarray): Modified array with empty fields set to NaN.
    """
    assert isinstance(Schl_complete, np.ndarray), "❌ Schl_complete must be a NumPy array"
    assert Schl_complete.ndim == 3, "❌ Schl_complete should be a 3D array (days, lat, lon)"
    assert Schl_complete.shape[0] == Truedays, (
        f"❌ Schl_complete has {Schl_complete.shape[0]} days, but Truedays is {Truedays}"
    )

    print("Checking and removing empty chlorophyll fields...")

    cempty = 0
    for d in range(Truedays):
        day_slice = Schl_complete[d, :, :]

        # Check if entire slice is NaN
        if np.isnan(day_slice).all() or np.nanmax(day_slice) == 0:
            cempty += 1
            Schl_complete[d, :, :] = np.nan
            print(f"\033[91m⚠️ Empty field found at day {d + 1} — replaced with NaNs ⚠️\033[0m")

    if cempty == 0:
        print("\033[92m✅ No empty fields found in dataset\033[0m")
    else:
        print(f"\033[93m{cempty} empty fields were found and corrected\033[0m")
    
    print('*' * 45)
    return Schl_complete