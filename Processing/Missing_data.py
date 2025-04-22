import numpy as np
from datetime import datetime, timedelta

def check_missing_days(T_orig, Schl_orig, Truedays, SinD, startingyear, startingmonth, startingday):
    SZT_orig = len(T_orig)
    
    if Truedays == SZT_orig:
        Ttrue = T_orig.copy()
        Schl_complete = Schl_orig.copy()
        print("Time series is complete")
        print('*'*45)
    
    elif SZT_orig < Truedays:
        # Missing days detected
        totdiffD = Truedays - SZT_orig
        print("Time series is not complete...")
        print(f"There are {totdiffD} missing days!")
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
                print("Gap search is over")
                print('-'*45)
                TbTData = Truedays - (Tcount + 1)
                print(f"Still {TbTData} data to be transferred")
                print("Transferring remaining days...")
                dleft = SZT_orig - d
                Schl_complete[Tcount+1:Truedays, :, :] = Schl_orig[d:SZT_orig, :, :]
                Tcount += TbTData
                
                if dleft == TbTData:
                    print(f"The remaining {TbTData} have been transferred")
                    print('-'*45)
                    breakloop = True
                else:
                    print("Something is rotten with the final data transfer!!!!!")
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
                print(f"This should never appear!!!!. Anyway, diffd = {T_orig[d] - T_orig[d-1]}")
                return None, None
            
            if breakloop:
                break
        
        if Tcount + 1 == Truedays:
            print("The gaps have been filled")
            print('*'*45)
        else:
            print("Problem in gap filling")
            print('*'*45)
            return None, None
    
    return Ttrue, Schl_complete

def find_missing_observations(Schl_complete, Truedays):
    """
    Identifies days with no satellite observations.
    
    Parameters:
    Schl_complete (numpy array): 3D array containing satellite data.
    Truedays (int): Total number of days in the dataset.
    
    Returns:
    cnan (int): Count of days with no satellite observations.
    satnan (list): Indices of days with no observations.
    """
    cnan = 0
    tempo1 = np.ones(Truedays)
    
    for t in range(Truedays):
        tempo2 = np.nansum(Schl_complete[t, :, :])
        if tempo2 == 0:
            cnan += 1
            tempo1[t] = np.nan
    
    print("Checking for missing satellite observations...")
    print(f"There are {cnan} daily satellite fields with no observations")
    print('*'*45)
    
    # Find indices where tempo1 is NaN (days with no observations)
    satnan = np.where(np.isnan(tempo1))[0].tolist()
    
    return cnan, satnan

def eliminate_empty_fields(Schl_complete, Truedays):
    """Eliminates empty fields (all-zero fields) from Schl_complete."""
    cempty = 0
    
    for d in range(Truedays):
        tempo = Schl_complete[d, :, :]
        z = np.nanmax(tempo)  # Get the max value while ignoring NaNs
        
        if z == 0:
            cempty += 1
            Schl_complete[d, :, :] = np.nan  # Replace empty field with NaN
            print(f"Found empty field at day = {d+1}")
            print('*'*45)
    
    if cempty == 0:
        print("No empty fields in file")
        print('*'*45)
    
    return Schl_complete