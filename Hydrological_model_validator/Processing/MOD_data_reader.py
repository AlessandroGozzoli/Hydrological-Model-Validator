import numpy as np
import os
import gzip
import shutil
from netCDF4 import Dataset
from pathlib import Path
import xarray as xr

from ..Plotting.Plots import Benthic_chemical_plot

from .Density import compute_density

from .time_utils import leapyear

def read_model_chl_data(Dmod, Ybeg, Tspan, Truedays, DinY, Mfsm):
    """Reads and processes model CHL data from netCDF files."""
    
    assert isinstance(Dmod, (str, Path)), "Dmod must be a string or Path object"
    assert isinstance(Ybeg, int), "Ybeg must be an integer"
    assert isinstance(Tspan, int) and Tspan > 0, "Tspan must be a positive integer"
    assert isinstance(Truedays, int) and Truedays > 0, "Truedays must be a positive integer"
    assert isinstance(DinY, int) and DinY in [365, 366], "DinY must be 365 or 366"
    assert (
        isinstance(Mfsm, tuple)
        and len(Mfsm) == 2
        and all(isinstance(i, np.ndarray) for i in Mfsm)
        ), "Mfsm must be a tuple of two numpy arrays for indexing"

    fMchl = 'Chlasat_od'
    ib = 0
    ie = 0
    ymod = Ybeg - 1
    Mchl_complete = []  # Initialize dynamically
    
    for y in range(Tspan):
        ymod += 1
        ymod_str = str(ymod)
        YDIR = "output" + ymod_str
        
        ib = ie
        amileap = DinY + leapyear(ymod)
        assert amileap in [365, 366], f"Leap year calculation failed for year {ymod}"
        ie = ib + amileap
        
        Mchlpath = Path(Dmod, YDIR) / f"ADR{ymod_str}new15bb_Chlsat.nc"
        Mchlpathgz = Path(Dmod, YDIR) / f"ADR{ymod_str}new15bb_Chlsat.nc.gz"
        
        print(f"Obtaining the CHL data for the year {ymod_str}...")
        
        if not os.path.exists(Mchlpath):
            if os.path.exists(Mchlpathgz):
                with gzip.open(Mchlpathgz, 'rb') as f_in, open(Mchlpath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        with Dataset(Mchlpath, 'r') as nc_file:
            assert fMchl in nc_file.variables, f"{fMchl} variable not found in {Mchlpath}"
            Mchl_orig = nc_file.variables[fMchl][:]
            assert Mchl_orig.shape[0] == amileap, f"Unexpected number of days in CHL data for year {ymod}"
        
        if os.path.exists(Mchlpathgz):
            print("Zipped file already existing")
            os.remove(Mchlpath)
        else:
            print("Zipping...")
            with open(Mchlpath, 'rb') as f_in, gzip.open(Mchlpathgz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"\033[92m✅ The model CHL data for the year {ymod_str} has been retrieved!\033[0m")
        print('-'*45)
        
        if y == 0:
            Mchlrow, Mchlcol = Mchl_orig.shape[1], Mchl_orig.shape[2]
            Mchl_complete = np.zeros((Truedays, Mchlrow, Mchlcol))
        
        tempo = np.full((Mchlrow, Mchlcol), np.nan)
        for t in range(amileap):
            tempo[:, :] = Mchl_orig[t, :, :]
            tempo[Mfsm] = np.nan
            Mchl_orig[t, :, :] = tempo[:, :]
        
        Mchl_complete[ib:ie, :, :] = Mchl_orig[:amileap, :, :]

    assert ie == Truedays, f"Total days mismatch: expected {Truedays}, got {ie}"

    print("\033[92m✅ Model CHL data fully loaded!\033[0m")
    print('*'*45)

    return Mchl_complete

def read_model_sst(Dmod, ysec, Mfsm):
    """
    Reads model SST data across multiple years and returns structured SST data.
    
    Parameters:
        DSST_mod (str): Path to the model SST data directory.
        ysec (list): List of years to iterate over.
    
    Returns:
        dict: Dictionary containing SST data organized by year.
    """
    
    # Input validations
    assert isinstance(Dmod, (str, Path)), "Dmod must be a string or Path object"
    assert isinstance(ysec, (list, tuple)) and all(isinstance(y, int) for y in ysec), "ysec must be a list or tuple of integers"
    assert (
        isinstance(Mfsm, tuple)
        and len(Mfsm) == 2
        and all(isinstance(i, np.ndarray) for i in Mfsm)
    ), "Mfsm must be a tuple of two numpy arrays for indexing"
    
    sst_data = {}  # Dictionary to store SST data by year
    
    for y in ysec:
        Ynow = str(y)
        print(f"Processing year {Ynow}...")
        current_year = str('output' + Ynow)
        
        DSST_mod = os.path.join(Dmod, current_year)
        
        # Construct the file path
        Msstpath = os.path.join(DSST_mod, 
                                f"ADR{Ynow}new_g100_1d_{Ynow}0101_{Ynow}1231_grid_T.nc")
        
        # Generate zipped file path
        Msstpathgz = f"{Msstpath}.gz"
        
        # Unzip if necessary
        if not os.path.exists(Msstpath) and os.path.exists(Msstpathgz):
            with gzip.open(Msstpathgz, 'rb') as f_in:
                with open(Msstpath, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        # Read SST data
        if os.path.exists(Msstpath):
            ds = xr.open_dataset(Msstpath)
            assert 'sst' in ds.variables, f"'sst' variable not found in file {Msstpath}"

            Msst = ds['sst'].values  # Extract SST variable
            assert Msst.ndim == 3, f"Expected 3D SST data, got shape {Msst.shape}"

            for t in range(Msst.shape[0]):
                # Apply mask to latitude/longitude using Mfsm
                Msst[t, Mfsm[0], Mfsm[1]] = np.nan  # Index mask

            sst_data[Ynow] = Msst
            print(f"\033[92m✅ The model SST data for the year {Ynow} has been retrieved!\033[0m")
            print('-'*45)
        else:
            print(f"\033[91m⚠️ Warning: SST file for {Ynow} not found.\033[0m")
            print('-'*45)

    return sst_data

def Bavg_sst(Tspan, ysec, Dmod, Sat_sst, Mfsm):
    """
    Reads and processes SST model data for multiple years, applying masking 
    consistent with satellite SST data and handling missing values.
    
    Parameters:
    - Tspan: number of years to process
    - ysec: list or array of years (e.g., [1998, 1999, ..., 2020])
    - DSST_mod: base path to model SST files
    - Ssst_orig: full satellite SST data, shape (Truedays, lat, lon)
    - Mfsm: indices (boolean array or mask) where values should be NaN’d
    
    Returns:
    - BASSTmod: list of daily mean SST values from the model
    - BASSTsat: list of corresponding satellite SST values
    """
    
    assert isinstance(Tspan, int) and Tspan > 0, "Tspan must be a positive integer"
    assert isinstance(ysec, (list, np.ndarray)), "ysec must be a list or NumPy array"
    assert len(ysec) == Tspan, "Length of ysec must equal Tspan"

    assert isinstance(Dmod, (str, Path)), "Dmod must be a string or Path object"
    assert os.path.isdir(Dmod), f"'{Dmod}' is not a valid directory"
    
    assert isinstance(Sat_sst, np.ndarray), "Sat_sst must be a NumPy array"
    assert Sat_sst.ndim == 3, "Sat_sst must be a 3D array (days, lat, lon)"
    
    assert isinstance(Mfsm, tuple) and len(Mfsm) == 2, "Mfsm must be a tuple with two index arrays"
    assert all(isinstance(arr, np.ndarray) for arr in Mfsm), "Mfsm must contain NumPy arrays"
    assert all(arr.ndim == 1 for arr in Mfsm), "Mfsm index arrays should be 1D"
    
    DafterD = 0
    BASSTmod = []
    BASSTsat = []

    for y in range(Tspan):
        Ynow = str(ysec[y])
        print(f"Processing year {Ynow}")
        
        YDIR = 'output' + str(Ynow)
        DSST_mod = Path(Dmod, YDIR)

        # File path construction
        fname = f"{DSST_mod}/ADR{Ynow}new_g100_1d_{Ynow}0101_{Ynow}1231_grid_T.nc"
        fname_gz = fname + ".gz"

        # Unzip if not already unzipped
        if not os.path.exists(fname) and os.path.exists(fname_gz):
            print(f"Unzipping {fname_gz}")
            with gzip.open(fname_gz, 'rb') as f_in:
                with open(fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        assert os.path.exists(fname), f"Expected SST file {fname} not found even after unzip"

        # Load SST data
        ds = xr.open_dataset(fname)
        Msst_orig = ds['sst'].values  # Shape: (days, lat, lon)
        ydays = Msst_orig.shape[0]

        for d in range(ydays):
            DafterD += 1
            Msst = Msst_orig[d, :, :]
            Ssst = Sat_sst[DafterD - 1, :, :]

            assert Msst.shape == Ssst.shape, "Model and satellite SST shapes must match"

            # Find NaNs in satellite SST
            Ssstfsm = np.isnan(Ssst)

            # Apply NaNs based on Mfsm and satellite NaNs
            Msst[Mfsm] = np.nan
            Msst[Ssstfsm] = np.nan
            Ssst[Mfsm] = np.nan

            # Store daily mean values
            BASSTmod.append(np.nanmean(Msst))
            BASSTsat.append(np.nanmean(Ssst))

    return BASSTmod, BASSTsat

def read_bfm(BDIR, Ybeg, Yend, ysec, bfm2plot, mask3d, Bmost, output_path, selected_unit, selected_description):
    # ----- FILENAME FRAGMENTS -----
    ffrag1 = "new15_1m_"
    ffrag2 = "0101_"
    ffrag3 = "1231_grid_bfm"
    
    # ----- ACCESSING THE MODEL DATA FOLDER -----
    print("Accessing the Model Data folder...")
    MDIR = Path(BDIR, "MODEL")
    print(f"Model data folder is {MDIR}")

    Epsilon = 0.06  # Plotting domain expansion

    latp, lonp, P_2d = None, None, None
    
    Mname = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] 

    for year in ysec:
        # -----SETUP-----
        ystr = str(year)
        print(f"Retrieving year: {ystr}")

        #-----FILENAME FRAGMENTS-----

        ffrag1="new15_1m_";
        ffrag2="0101_";
        ffrag3="1231_grid_bfm";
        
        file = os.path.join(MDIR, f"output{ystr}", f"ADR{ystr}{ffrag1}{ystr}{ffrag2}{ystr}{ffrag3}.nc")
        filegz = file + ".gz"

        # -----UNZIP-----
        print("Unzipping the file...")
        with gzip.open(filegz, 'rb') as f_in:
            with open(file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("File succesfully unzipped")

        # -----READ LAT, LON, DEPTH-----
        with Dataset(file, 'r') as nc:
            ytemp = nc.variables['nav_lat'][:]
            xtemp = nc.variables['nav_lon'][:]
            d = nc.variables['deptht'][:]

        Yrow, Xcol = ytemp.shape
        x1st = 12.200
        xstep = 0.0100
        y1st = 43.774
        ystep = 0.0073

        # -----GENERATE FIXED LAT/LON GRIDS-----
        ytemp1d = np.array([y1st + j * ystep for j in range(Yrow)])
        xtemp1d = np.array([x1st + i * xstep for i in range(Xcol)])
        ytemp = np.tile(ytemp1d[:, np.newaxis], (1, Xcol))
        xtemp = np.tile(xtemp1d[np.newaxis, :], (Yrow, 1))

        # -----CREATE 3D COORDINATE ARRAYS-----
        lat = np.repeat(ytemp[np.newaxis, :, :], len(d), axis=0)
        lon = np.repeat(xtemp[np.newaxis, :, :], len(d), axis=0)

        print("The geolocalized dataset has been initialized!")

        Epsilon = 0.06
        MinPhi = np.nanmin(ytemp1d) + Epsilon
        MaxPhi = np.nanmax(ytemp1d) + Epsilon
        MinLambda = np.nanmin(xtemp1d) - Epsilon
        MaxLambda = np.nanmax(xtemp1d) + Epsilon

        latp = lat[0, :, :]
        lonp = lon[0, :, :]

        # -----LOAD FIELD-----
        with Dataset(file, 'r') as nc:
            P_orig = nc.variables[bfm2plot][:]

        os.remove(file)

        Tlev = P_orig.shape[0]
        print(f"The {selected_description} field has been found!")

        # -----NaN MASK-----
        P = P_orig.copy()
        P[:, mask3d == 0] = np.nan
        print(f"The mask has been applied to the {selected_description} field!")

        # -----EXTRACT BOTTOM LAYER FIELDS-----
        P_2d = np.zeros((Tlev, mask3d.shape[1], mask3d.shape[2]))
        for i in range(mask3d.shape[2]):
            for j in range(mask3d.shape[1]):
                k = int(Bmost[j, i]) - 1
                P_2d[:, j, i] = P[:, k, j, i]
        
        print(f"The Bentic Layer data for the {selected_description} has been obtained!")
        
        # -----PLOT MONTHLY AVERAGES-----
        print(f"Beginning to plot the monthly mean value for the year {ystr}...")
        for t in range(Tlev):
            Benthic_chemical_plot(MinLambda, MaxLambda, MinPhi, MaxPhi, P_2d, t, lonp, latp, bfm2plot, Mname, ystr, selected_unit, selected_description, output_path)
            
        print("The monthly mean plots have been created!")
        print('-'*45)
    
def read_bentic_sst_sal(BDIR, Ybeg, Yend, ysec, mask3d, Bmost, output_path):

    # ----- FILENAME FRAGMENTS -----
    ffrag1 = "new15_1m_"
    ffrag2 = "0101_"
    ffrag3 = "1231_grid_T"

    # ----- ACCESSING THE MODEL DATA FOLDER -----
    print("Accessing the Model Data folder...")
    MDIR = Path(BDIR, "MODEL")
    print(f"Model data folder is {MDIR}")

    Epsilon = 0.06  # Plotting domain expansion
    Mname = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for year in ysec:
        ystr = str(year)
        print(f"Retrieving year: {ystr}")

        file = os.path.join(MDIR, f"output{ystr}", f"ADR{ystr}{ffrag1}{ystr}{ffrag2}{ystr}{ffrag3}.nc")
        filegz = file + ".gz"

        # -----UNZIP-----
        print("Unzipping the file...")
        with gzip.open(filegz, 'rb') as f_in:
            with open(file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("File successfully unzipped")
        print('-'*45)

        print("Geolocalizing the dataset...")
        # -----READ LAT, LON, DEPTH-----
        with Dataset(file, 'r') as nc:
            ytemp = nc.variables['nav_lat'][:]
            xtemp = nc.variables['nav_lon'][:]
            d = nc.variables['deptht'][:]

        Yrow, Xcol = ytemp.shape
        x1st = 12.200
        xstep = 0.0100
        y1st = 43.774
        ystep = 0.0073

        ytemp1d = np.array([y1st + j * ystep for j in range(Yrow)])
        xtemp1d = np.array([x1st + i * xstep for i in range(Xcol)])
        ytemp = np.tile(ytemp1d[:, np.newaxis], (1, Xcol))
        xtemp = np.tile(xtemp1d[np.newaxis, :], (Yrow, 1))

        lat = np.repeat(ytemp[np.newaxis, :, :], len(d), axis=0)
        lon = np.repeat(xtemp[np.newaxis, :, :], len(d), axis=0)

        print("The geolocalized dataset has been initialized!")
        print('-'*45)

        MinPhi = np.nanmin(ytemp1d) + Epsilon
        MaxPhi = np.nanmax(ytemp1d) + Epsilon
        MinLambda = np.nanmin(xtemp1d) - Epsilon
        MaxLambda = np.nanmax(xtemp1d) + Epsilon

        latp = lat[0, :, :]
        lonp = lon[0, :, :]

        # -----READ TEMPERATURE AND SALINITY-----
        with Dataset(file, 'r') as nc:
            print("Loading Temperature...")
            temp_data = nc.variables['votemper'][:]
            print("Temperature dataset obtained!")
            print("Loading Salinity...")
            sal_data = nc.variables['vosaline'][:]
            print("Salinity dataset obtained!")

        Tlev = temp_data.shape[0]

        # -----APPLY MASK-----
        temp_data[:, mask3d == 0] = np.nan
        sal_data[:, mask3d == 0] = np.nan

        # -----EXTRACT BOTTOM TEMPERATURE AND SALINITY-----
        temp_2d = np.zeros((Tlev, mask3d.shape[1], mask3d.shape[2]))
        sal_2d = np.zeros((Tlev, mask3d.shape[1], mask3d.shape[2]))
        for i in range(mask3d.shape[2]):
            for j in range(mask3d.shape[1]):
                k = int(Bmost[j, i]) - 1
                temp_2d[:, j, i] = temp_data[:, k, j, i]
                sal_2d[:, j, i] = sal_data[:, k, j, i]

        print("Bottom temperature and salinity fields have been extracted!")
        print("-"*45)

        # -----COMPUTE DENSITY-----
        print("Comuting the density field...")
        density_EOS = compute_density(temp_2d, sal_2d, Bmost, method="EOS")
        print("Density field computed using the simplified density formula!")
        density_EOS80 = compute_density(temp_2d, sal_2d, Bmost, method="EOS80")
        print("Density field computed using the EOS-80 density formula!")
        density_TEOS10 = compute_density(temp_2d, sal_2d, Bmost, method="TEOS10")
        print("Density field computed using the TEOS-10 density formula!")
        print("-"*45)

        # -----PLOT TEMPERATURE, SALINITY, AND PRESSURE-----
        plot_vars = {
            'votemper': (temp_2d, '°C', 'Temperature'),
            'vosaline': (sal_2d, 'psu', 'Salinity'),
            'vodensity_EOS': (density_EOS, 'kg/m3', 'Density [Simplified EOS]'),
            'vodensity_EOS80': (density_EOS80, 'kg/m3', 'Density [EOS-80]'),
            'vodensity_TEOS10': (density_TEOS10, 'kg/m3', 'Density [TEOS-10]')
        }

        for key, (P_2d, unit, desc) in plot_vars.items():
            outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", desc)
            os.makedirs(outdir, exist_ok=True)

            print(f"Plotting {desc} for year {ystr}...")
            for t in range(Tlev):
                Benthic_chemical_plot(
                    MinLambda, MaxLambda, MinPhi, MaxPhi,
                    P_2d, t, lonp, latp,
                    key, Mname, ystr,
                    unit, desc,
                    outdir
                )
            print(f"{desc} plots completed.")
            print('-' * 45)

        # Clean up
        os.remove(file)
