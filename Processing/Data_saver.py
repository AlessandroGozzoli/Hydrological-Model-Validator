import scipy.io
import pandas as pd
import xarray as xr
import numpy as np
from Data_setupper import Truedays, Slon, Slat, Schl_complete, satnan

data = {
    'Truedays': Truedays,
    'Slon': Slon,
    'Slat': Slat,
    'Schl_complete': Schl_complete,
    'satnan': satnan
}

def save_as_mat():
    for key, value in data.items():
        scipy.io.savemat(f"{key}.mat", {key: value})

def save_as_csv():
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            df = pd.DataFrame(value)
            df.to_csv(f"{key}.csv", index=False)
        else:
            with open(f"{key}.csv", 'w') as f:
                f.write(str(value))

def save_as_nc():
    for key, value in data.items():
        ds = xr.Dataset({key: (['dim_0', 'dim_1'] if value.ndim == 2 else ['dim_0'], value)})
        ds.to_netcdf(f"{key}.nc")

def save_all():
    save_as_mat()
    save_as_csv()
    save_as_nc()