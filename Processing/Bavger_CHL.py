import numpy as np

def Bavg_CHL(MChl_dataset, SChl_dataset, Mfsm, chllev):
    BACHLmod = []
    BACHLsat = []

    var_name_mod = list(MChl_dataset.data_vars)[0]
    var_name_sat = list(SChl_dataset.data_vars)[0]
    ydays = len(MChl_dataset[var_name_mod])

    for d in range(ydays):
        Mchl = MChl_dataset[var_name_mod].isel(time=d).values
        Schl = SChl_dataset[var_name_sat].isel(time=d).values

        # Find NaNs in satellite CHL
        Schlfsm = np.isnan(Schl)

        # Apply NaNs
        Mchl[Mfsm] = np.nan
        Mchl[Schlfsm] = np.nan
        Schl[Mfsm] = np.nan

        # Store daily mean values
        BACHLmod.append(np.nanmean(Mchl))
        BACHLsat.append(np.nanmean(Schl))
    
    return {chllev: BACHLmod}, {chllev: BACHLsat}