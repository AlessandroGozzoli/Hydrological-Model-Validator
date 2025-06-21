###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Type hinting
from typing import Dict, List, Union, Any, Tuple

# Data handling
import pandas as pd
import numpy as np
import xarray as xr
import itertools

# Parallel processing and diagnostics
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor

# Logging and tracing
import logging
from eliot import start_action, log_message
from eliot.stdlib import EliotHandler

# Date and time handling
from datetime import datetime

# Context management
from contextlib import contextmanager

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def leapyear(year: int) -> int:
    """
    Check if a given year is a leap year.

    Parameters
    ----------
    year : int
        Year as a positive integer.

    Returns
    -------
    int
        Returns 1 if the year is a leap year, 0 otherwise.

    Raises
    ------
    ValueError
        If `year` is not a positive integer.

    Examples
    --------
    >>> leapyear(2020)
    1
    >>> leapyear(1900)
    0
    >>> leapyear(2000)
    1
    >>> leapyear(2023)
    0
    """
    # Validate input is a positive integer
    if not (isinstance(year, int) and year > 0):
        raise ValueError("❌ Year must be a positive integer. ❌")

    with Timer("leapyear function"):
        with start_action(action_type="leapyear function", year=year):
            # Leap year if divisible by 4 but not 100, or divisible by 400
            result = int((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0))

            log_message("Computed leap year result", year=year, result=result)
            logging.info(f"Year {year} is {'a leap year' if result else 'not a leap year'}.")

            return result
###############################################################################

###############################################################################
def true_time_series_length(
    chlfstart: List[int],
    chlfend: List[int],
    DinY: int
) -> int:
    """
    Calculate the true time series length in days over multiple files,
    accounting for leap years.

    Parameters
    ----------
    chlfstart : list[int]
        List of start years per file.

    chlfend : list[int]
        List of end years per file.

    DinY : int
        Number of days in a normal year (expected 365).

    Returns
    -------
    int
        Total number of days including leap years.

    Raises
    ------
    ValueError
        If input types or values do not meet expectations.

    Examples
    --------
    >>> true_time_series_length([2000], [2001], 365)
    731
    >>> true_time_series_length([1999, 2001], [2000, 2002], 365)
    1096
    """
    # ===== VALIDATION =====
    if len(chlfstart) != len(chlfend):
        raise ValueError("❌ chlfstart and chlfend must be the same length. ❌")  # Matching start/end lists
    if not all(isinstance(x, int) for x in itertools.chain(chlfstart, chlfend)):
        raise ValueError("❌ chlfstart and chlfend must contain integers only. ❌")  # All years are ints
    if not all(end >= start for start, end in zip(chlfstart, chlfend)):
        raise ValueError("❌ Each chlfend must be >= corresponding chlfstart. ❌")  # Valid year ranges
    if not (isinstance(DinY, int) and DinY == 365):
        raise ValueError("❌ DinY must be 365. ❌")  # Only standard year length allowed

    with Timer("true_time_series_length computation"):
        logging.info(f"Inputs received - chlfstart: {chlfstart}, chlfend: {chlfend}, DinY: {DinY}")
        log_message("Inputs received", chlfstart=chlfstart, chlfend=chlfend, DinY=DinY)

        # ===== CALCULATION =====
        # Generate all years covered by the file ranges inclusive
        years = itertools.chain.from_iterable(range(start, end + 1) for start, end in zip(chlfstart, chlfend))
        
        # Sum days per year adding 1 day if leap year, else 0
        total_days = 0
        for year in years:
            days = DinY + leapyear(year)
            total_days += days
            logging.info(f"Year: {year}, days added: {days}")
            log_message("Year days added", year=year, days=days)
        
        logging.info(f"Computed total_days: {total_days}")
        log_message("Computed total_days", total_days=total_days)

        return total_days
###############################################################################

###############################################################################
def split_to_monthly(
    yearly_data: Dict[int, Union[pd.Series, pd.DataFrame]]
) -> Dict[int, List[Union[pd.Series, pd.DataFrame]]]:
    """
    Split yearly pandas Series or DataFrames with datetime index into monthly segments.

    Parameters
    ----------
    yearly_data : dict[int, pd.Series or pd.DataFrame]
        Dictionary keyed by year, with values being pandas Series or DataFrames
        indexed by datetime.

    Returns
    -------
    dict[int, list[pd.Series or pd.DataFrame]]
        Dictionary keyed by year, each containing a list of 12 elements corresponding
        to monthly slices of the data (January to December). Months with no data
        will have empty Series or DataFrames of the same type.

    Raises
    ------
    ValueError
        If yearly_data is not a dictionary or values are not pandas Series/DataFrames
        with datetime-like indexes.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range('2020-01-01', '2020-12-31')
    >>> data = pd.Series(np.random.rand(len(dates)), index=dates)
    >>> yearly = {2020: data}
    >>> monthly = split_to_monthly(yearly)
    >>> len(monthly[2020])
    12
    >>> monthly[2020][0].index.month.unique()
    Int64Index([1], dtype='int64')
    """
    # ===== VALIDATION =====
    if not isinstance(yearly_data, dict):
        raise ValueError("❌ Input yearly_data must be a dictionary. ❌")

    for year, data in yearly_data.items():
        if not isinstance(year, int):
            raise ValueError(f"❌ Year keys must be int, got {type(year)}. ❌")  # Ensure year keys are ints
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError(f"❌ Values must be pandas Series or DataFrame, got {type(data)}. ❌")  # Validate types
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError(f"❌ Data index for year {year} must be datetime-like. ❌")  # Confirm datetime index

    with Timer("Splitting yearly data into monthly segments"):
        monthly_data_dict = {}
        for year, data in yearly_data.items():
            logging.info(f"Processing year {year}, data type: {type(data).__name__}")
            log_message("Processing year", year=year, data_type=type(data).__name__)

            # Create an empty template slice (same type, no rows) for months with no data
            empty_template = data.iloc[0:0]

            # ===== SPLITTING =====
            monthly_slices = []
            for month in range(1, 13):
                month_data = data.loc[data.index.month == month]  # Filter data for each month
                if month_data.empty:
                    monthly_slices.append(empty_template.copy())  # Append empty slice if no data
                    logging.info(f"Empty month slice added for year {year}, month {month}")
                    log_message("Empty month slice added", year=year, month=month)
                else:
                    monthly_slices.append(month_data)  # Append actual data for month
                    logging.info(f"Month slice added for year {year}, month {month}, length {len(month_data)}")
                    log_message("Month slice added", year=year, month=month, length=len(month_data))

            monthly_data_dict[year] = monthly_slices  # Store monthly slices for the year

        logging.info(f"Completed splitting all years, total years processed: {len(monthly_data_dict)}")
        log_message("Completed splitting all years", years_processed=len(monthly_data_dict))
        return monthly_data_dict
###############################################################################

###############################################################################
def split_to_yearly(
    series: pd.Series, 
    unique_years: List[Union[int, str]]
) -> Dict[Union[int, str], pd.Series]:
    """
    Split a pandas Series with a datetime index into a dictionary keyed by year.

    Parameters
    ----------
    series : pd.Series
        Time-indexed pandas Series with a datetime index.
    unique_years : list of int or str
        List of years to split the series into. Years can be int or string representations.

    Returns
    -------
    dict of year (int or str) to pd.Series
        Dictionary keyed by year containing the Series filtered for that year.

    Raises
    ------
    ValueError
        If the series does not have a DatetimeIndex or unique_years contains invalid types.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2020-01-01', periods=365)
    >>> s = pd.Series(range(365), index=dates)
    >>> split_yearly = split_to_yearly(s, [2020])
    >>> list(split_yearly.keys())
    [2020]
    >>> split_yearly[2020].index.year.unique()
    Int64Index([2020], dtype='int64')
    """
    # ===== VALIDATION =====
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("❌ series must have a DatetimeIndex ❌")  # Ensure datetime index
    if not all(isinstance(year, (int, str)) for year in unique_years):
        raise ValueError("❌ unique_years must contain only ints or strs ❌")  # Validate years list

    with Timer("split_to_yearly computation"):
        logging.info(f"Received series with index type: {type(series.index)}")
        log_message("Input validation passed", unique_years=unique_years)

        yearly_data = {}

        # ===== SPLITTING =====
        for year in unique_years:
            try:
                year_int = int(year)  # Convert year to int for comparison
            except ValueError:
                raise ValueError(f"❌ Year {year} cannot be converted to int ❌")

            filtered = series.loc[series.index.year == year_int].copy()

            yearly_data[year] = filtered  # Store filtered series keyed by original year type

            logging.info(f"Year {year} filtered with {len(filtered)} records")
            log_message("Yearly data filtered", year=year, count=len(filtered))

        logging.info(f"Split completed for years: {list(yearly_data.keys())}")
        log_message("Completed splitting to yearly", years=list(yearly_data.keys()))

        return yearly_data
###############################################################################

###############################################################################
def get_common_years(
    data_dict: Dict[str, Dict[Union[int, str], Any]], 
    mod_key: str, 
    sat_key: str
) -> List[Union[int, str]]:
    """
    Get sorted years present in both model and satellite datasets.

    Parameters
    ----------
    data_dict : dict
        Dictionary where keys are dataset names and values are dictionaries keyed by year.
    mod_key : str
        Key for the model dataset in data_dict.
    sat_key : str
        Key for the satellite dataset in data_dict.

    Returns
    -------
    List[int or str]
        Sorted list of years present in both model and satellite datasets.

    Raises
    ------
    ValueError
        If mod_key or sat_key are not present in data_dict or their values are not dictionaries.

    Examples
    --------
    >>> data = {
    ...     'model': {2020: 'data1', 2021: 'data2'},
    ...     'satellite': {2021: 'dataA', 2022: 'dataB'}
    ... }
    >>> get_common_years(data, 'model', 'satellite')
    [2021]
    """
    # ===== VALIDATE KEYS =====
    if mod_key not in data_dict:
        raise ValueError(f"❌ mod_key '{mod_key}' not found in data_dict ❌")  # Ensure model key exists
    if sat_key not in data_dict:
        raise ValueError(f"❌ sat_key '{sat_key}' not found in data_dict ❌")  # Ensure satellite key exists
    if not isinstance(data_dict[mod_key], dict):
        raise ValueError(f"❌ data_dict[{mod_key}] must be a dictionary ❌")  # Model data must be dict
    if not isinstance(data_dict[sat_key], dict):
        raise ValueError(f"❌ data_dict[{sat_key}] must be a dictionary ❌")  # Satellite data must be dict

    with Timer("get_common_years computation"):
        logging.info(f"Validated presence and type of keys: '{mod_key}', '{sat_key}'")
        log_message("Validated keys", mod_key=mod_key, sat_key=sat_key)

        # ===== COMPUTE COMMON YEARS =====
        mod_years = set(data_dict[mod_key].keys())  # Set of model years
        sat_years = set(data_dict[sat_key].keys())  # Set of satellite years

        common_years = sorted(mod_years.intersection(sat_years))

        logging.info(f"Common years computed: {common_years}")
        log_message("Computed common years", common_years=common_years)

        return common_years
###############################################################################

###############################################################################
def get_season_mask(
    dates: Union[pd.DatetimeIndex, pd.Series], 
    season_name: str
) -> np.ndarray:
    """
    Generate a boolean mask for the given season on a datetime index or series.

    Parameters
    ----------
    dates : pd.DatetimeIndex or pd.Series
        Datetime-like index or pandas Series with a datetime index.
    season_name : str
        Season name. Must be one of {'DJF', 'MAM', 'JJA', 'SON'}.

    Returns
    -------
    np.ndarray
        Boolean mask array indicating whether each date falls in the specified season.

    Raises
    ------
    ValueError
        If `season_name` is not one of the expected values.
    TypeError
        If `dates` is not a pandas DatetimeIndex or a Series with DatetimeIndex.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2023-01-01', periods=12, freq='M')
    >>> get_season_mask(dates, 'DJF')
    array([ True,  True, False, False, False, False, False, False, False, False, False,  True])
    """
    # ===== VALIDATE SEASON NAME =====
    valid_seasons = {'DJF', 'MAM', 'JJA', 'SON'}
    if season_name not in valid_seasons:
        raise ValueError(f"❌ Invalid season name: {season_name}. Expected one of {valid_seasons} ❌")

    # ===== VALIDATE DATES TYPE AND EXTRACT DATETIME INDEX =====
    if isinstance(dates, pd.Series):
        if not isinstance(dates.index, pd.DatetimeIndex):
            raise TypeError("❌ If input is pd.Series, its index must be a pd.DatetimeIndex ❌")
        dt_index = dates.index  # Use index if Series
    elif isinstance(dates, pd.DatetimeIndex):
        dt_index = dates  # Directly use DatetimeIndex
    else:
        raise TypeError("❌ dates must be a pandas DatetimeIndex or a Series with DatetimeIndex ❌")

    with Timer(f"get_season_mask computation for season '{season_name}'"):
        logging.info(f"Validated season_name: {season_name}")
        log_message("Validated season_name", season_name=season_name)

        logging.info(f"Using datetime index of length {len(dt_index)}")
        log_message("Datetime index info", length=len(dt_index))

        # ===== MAP SEASON TO MONTHS =====
        season_months = {
            'DJF': {12, 1, 2},
            'MAM': {3, 4, 5},
            'JJA': {6, 7, 8},
            'SON': {9, 10, 11},
        }

        months = dt_index.month  # Extract month numbers from dates
        mask = np.isin(months, list(season_months[season_name]))  # Boolean mask for season months

        logging.info(f"Computed season mask for '{season_name}'")
        log_message("Computed season mask", season_name=season_name, mask_shape=mask.shape)

        return mask
###############################################################################

###############################################################################
def resample_and_compute(
    model_sst_chunked: Union[xr.DataArray, xr.Dataset],
    sat_sst_chunked: Union[xr.DataArray, xr.Dataset]
) -> Tuple[Union[xr.DataArray, xr.Dataset], Union[xr.DataArray, xr.Dataset]]:
    """
    Resample the input chunked SST datasets to monthly means and compute them concurrently.

    Parameters
    ----------
    model_sst_chunked : xarray.DataArray or Dataset
        The model SST dataset chunked for dask processing.
    sat_sst_chunked : xarray.DataArray or Dataset
        The satellite SST dataset chunked for dask processing.

    Returns
    -------
    model_sst_monthly : xarray.DataArray or Dataset
        The computed monthly mean resampled model SST.
    sat_sst_monthly : xarray.DataArray or Dataset
        The computed monthly mean resampled satellite SST.
    """
    with Timer("Resample and compute monthly SST datasets concurrently"):
        # ===== RESAMPLE TO MONTHLY MEAN (LAZY) =====
        model_sst_monthly_lazy = model_sst_chunked.resample(time='1MS').mean()  # Resample model to start-of-month monthly means (lazy)
        sat_sst_monthly_lazy = sat_sst_chunked.resample(time='1MS').mean()      # Resample satellite similarly (lazy)

        logging.info("Resampled model and satellite SST to monthly means (lazy)")
        log_message("Resampled SST to monthly means", model_resample="lazy", sat_resample="lazy")

        # ===== COMPUTE RESAMPLED DATA IN PARALLEL =====
        with ProgressBar(), ThreadPoolExecutor(max_workers=2) as executor:
            # Schedule parallel compute tasks for model and satellite with thread scheduler
            future_model = executor.submit(model_sst_monthly_lazy.compute, scheduler='threads')
            future_sat = executor.submit(sat_sst_monthly_lazy.compute, scheduler='threads')

            model_sst_monthly = future_model.result()  # Get computed model results
            sat_sst_monthly = future_sat.result()      # Get computed satellite results

        logging.info("Computed monthly SST datasets concurrently")
        log_message("Computed SST datasets", model_computed=True, satellite_computed=True)

        return model_sst_monthly, sat_sst_monthly
    
###############################################################################

###############################################################################

def is_invalid_time_index(
    time_index: Union[pd.Index, np.ndarray]
) -> bool:
    """
    Check whether a given time index is invalid based on dtype and value range.

    This function validates if the input `time_index` is a valid datetime index.
    It considers the time index invalid if:
    - The dtype is not a datetime64 type.
    - All timestamps fall within a very narrow range starting from the Unix epoch (1970-01-01)
      and the differences between consecutive timestamps are extremely small (less than 1 millisecond).
    Such a time index might indicate corrupted or placeholder data.

    Parameters
    ----------
    time_index : array-like
        An array or pandas Index representing time values, expected to be datetime-like.

    Returns
    -------
    bool
        True if the time index is considered invalid, False otherwise.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Valid datetime index
    >>> idx = pd.date_range("2023-01-01", periods=3)
    >>> is_invalid_time_index(idx)
    False
    >>> # Invalid: non-datetime dtype
    >>> is_invalid_time_index(np.array([1, 2, 3]))
    True
    """
    # Check if the dtype of time_index is a subtype of np.datetime64
    if not np.issubdtype(time_index.dtype, np.datetime64):
        return True  # Not datetime, so invalid

    # Convert to pandas DatetimeIndex for easier time operations
    dt_index = pd.DatetimeIndex(time_index)
    start = pd.Timestamp("1970-01-01")  # Unix epoch start
    max_allowed_delta = pd.Timedelta('1 day')  # Max allowed range for invalid detection

    # Check if all times are within one day from the Unix epoch start
    if (dt_index.min() >= start) and (dt_index.max() <= start + max_allowed_delta):
        # Calculate differences between consecutive timestamps
        diffs = dt_index.to_series().diff().dropna()
        # Only flag as invalid if diffs exist and all are less than 1 millisecond
        if len(diffs) > 0 and all(diffs < pd.Timedelta(microseconds=1000)):
            return True

    # Otherwise, the time index is valid
    return False

###############################################################################

def prompt_for_datetime_index(length: int) -> pd.DatetimeIndex:
    """
    Prompt the user to manually enter a valid datetime index for a time series.

    When an invalid or missing time index is detected, this function interacts
    with the user to obtain a valid start date and frequency. It then generates
    a pandas DatetimeIndex of the specified length with the given frequency.

    Parameters
    ----------
    length : int
        The desired length of the datetime index to generate.

    Returns
    -------
    pd.DatetimeIndex
        A pandas DatetimeIndex object starting from the user-provided date,
        with the specified frequency and length.

    Example
    -------
    >>> idx = prompt_for_datetime_index(10)
    Enter the start date for the time series (e.g. 2000-01-01): 2020-01-01
    Enter the frequency (e.g. 'D' for daily, 'H' for hourly): D
    Generated datetime index from 2020-01-01 00:00:00 with frequency 'D'.
    """

    # Notify user about the invalid or missing time index
    print("⚠️ Detected invalid or missing time index.")
    
    border = "#" * 60

    while True:
        # Prompt for start date input as string
        start_date_str = input("Enter the start date for the time series (e.g. 2000-01-01): ").strip()

        # Prompt for frequency string, e.g., 'D', 'H', 'M'
        freq = input("Enter the frequency (e.g. 'D' for daily, 'H' for hourly): ").strip()

        try:
            # Convert input string to pandas Timestamp
            start_date = pd.to_datetime(start_date_str)

            # Generate a datetime index with the given start, length, and frequency
            time_index = pd.date_range(start=start_date, periods=length, freq=freq)

            # Inform user about the generated datetime index
            print(f"Generated datetime index from {start_date} with frequency '{freq}'.")
            print("\n" + border + "\n")

            # Return the generated datetime index
            return time_index

        except Exception as e:
            # Catch errors (e.g. invalid date format or freq) and prompt again
            print(f"Invalid input: {e}. Please try again.")
            print("\n" + border + "\n")
            
###############################################################################

###############################################################################

def ensure_datetime_index(series: pd.Series, label: str) -> pd.Series:
    """
    Ensure that a pandas Series has a DatetimeIndex. If not, prompt the user to create one.

    This function checks whether the index of the provided pandas Series is a DatetimeIndex.
    If the index is not datetime-based, it asks the user to input a start date and frequency,
    then generates and assigns a new DatetimeIndex to the Series accordingly.

    Parameters
    ----------
    series : pd.Series
        The pandas Series whose index is to be checked and possibly converted.
    label : str
        A descriptive name for the series, used in prompts and messages.

    Returns
    -------
    pd.Series
        The original Series if it already had a DatetimeIndex, or the Series with a newly
        created DatetimeIndex based on user input.

    Example
    -------
    >>> s = pd.Series([1, 2, 3])
    >>> s = ensure_datetime_index(s, "Sample Series")
    Please enter the start date for Sample Series (YYYY-MM-DD): 2020-01-01
    Please enter the data frequency (e.g. 'D') for Sample Series: D
    DatetimeIndex created for Sample Series from 2020-01-01 with frequency 'D'.
    """
    border = "#" * 60

    # Check if the series index is already a DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        print(f"{label} has no DatetimeIndex.")

        # Prompt user for the start date string
        start_date_str = input(f"Please enter the start date for {label} (YYYY-MM-DD): ").strip()

        # Prompt user for frequency string (e.g., 'D' for daily)
        freq_str = input(f"Please enter the data frequency (e.g. 'D') for {label}: ").strip()

        # Convert input start date string to pandas Timestamp
        start_date = pd.to_datetime(start_date_str)

        # Generate a new DatetimeIndex based on user inputs, matching series length
        series.index = pd.date_range(start=start_date, periods=len(series), freq=freq_str)

        print(f"DatetimeIndex created for {label} from {start_date_str} with frequency '{freq_str}'.")
        print("\n" + border + "\n")

    # Return the series, original or modified
    return series


###############################################################################
##                                                                           ##
##                                 TIMER                                     ##
##                                                                           ##
###############################################################################


# ----- SETUP OF THE LOGGER AND TIMER -----

# Clear existing handlers on root logger
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# --- File handler: capture everything (including third-party libs) ---
file_handler = logging.FileHandler("app.log", mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)  # or DEBUG if you want
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# --- Stream handler: only show YOUR logs in console ---
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)  # Filter out INFO/DEBUG logs from chat
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)
root_logger.addHandler(EliotHandler())

root_logger.setLevel(logging.INFO)

# Optional: raise level for noisy libs to WARNING so they skip stream
logging.getLogger("xarray").setLevel(logging.INFO)  # still captured in file, but not printed
logging.getLogger("netCDF4").setLevel(logging.INFO)
logging.getLogger("h5netcdf").setLevel(logging.INFO)

# --- Timer context manager that logs to both systems ---
@contextmanager
def Timer(description):
    start = datetime.now()
    log_message("Timer started", description=description, start=str(start))
    logging.info(f"[Timer Start] {description} at {start}")
    
    with start_action(action_type="timed block", description=description, start_time=str(start)) as action:
        try:
            yield
        finally:
            end = datetime.now()
            elapsed = end - start
            action.add_success_fields(end_time=str(end), elapsed=str(elapsed))
            log_message("Timer ended", description=description, end=str(end), elapsed=str(elapsed))
            logging.info(f"[Timer End] {description} at {end} (Elapsed: {elapsed})")