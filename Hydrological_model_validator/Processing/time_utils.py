from typing import Dict, List, Union, Any
import pandas as pd
import numpy as np
import itertools
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor

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

    # Leap year if divisible by 4 but not 100, or divisible by 400
    return int((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0))
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

    # ===== CALCULATION =====
    # Generate all years covered by the file ranges inclusive
    years = itertools.chain.from_iterable(range(start, end + 1) for start, end in zip(chlfstart, chlfend))
    
    # Sum days per year adding 1 day if leap year, else 0
    total_days = sum(DinY + leapyear(year) for year in years)

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

    monthly_data_dict = {}
    for year, data in yearly_data.items():
        if not isinstance(year, int):
            raise ValueError(f"❌ Year keys must be int, got {type(year)}. ❌")  # Ensure year keys are ints
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError(f"❌ Values must be pandas Series or DataFrame, got {type(data)}. ❌")  # Validate types
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError(f"❌ Data index for year {year} must be datetime-like. ❌")  # Confirm datetime index

        # Create an empty template slice (same type, no rows) for months with no data
        empty_template = data.iloc[0:0]

        # ===== SPLITTING =====
        monthly_slices = []
        for month in range(1, 13):
            month_data = data.loc[data.index.month == month]  # Filter data for each month
            if month_data.empty:
                monthly_slices.append(empty_template.copy())  # Append empty slice if no data
            else:
                monthly_slices.append(month_data)  # Append actual data for month

        monthly_data_dict[year] = monthly_slices  # Store monthly slices for the year

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

    yearly_data = {}

    # ===== SPLITTING =====
    for year in unique_years:
        try:
            year_int = int(year)  # Convert year to int for comparison
        except ValueError:
            raise ValueError(f"❌ Year {year} cannot be converted to int ❌")

        # Filter series by year and copy slice to avoid reference issues
        filtered = series.loc[series.index.year == year_int].copy()

        yearly_data[year] = filtered  # Store filtered series keyed by original year type

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

    # ===== COMPUTE COMMON YEARS =====
    mod_years = set(data_dict[mod_key].keys())  # Set of model years
    sat_years = set(data_dict[sat_key].keys())  # Set of satellite years

    # Intersection gives years present in both datasets
    common_years = sorted(mod_years.intersection(sat_years))

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

    # ===== MAP SEASON TO MONTHS =====
    season_months = {
        'DJF': {12, 1, 2},
        'MAM': {3, 4, 5},
        'JJA': {6, 7, 8},
        'SON': {9, 10, 11},
    }

    months = dt_index.month  # Extract month numbers from dates
    mask = np.isin(months, list(season_months[season_name]))  # Boolean mask for season months

    return mask
###############################################################################

###############################################################################
def resample_and_compute(model_sst_chunked, sat_sst_chunked):
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
    # ===== RESAMPLE TO MONTHLY MEAN (LAZY) =====
    model_sst_monthly_lazy = model_sst_chunked.resample(time='1MS').mean()  # Resample model to start-of-month monthly means (lazy)
    sat_sst_monthly_lazy = sat_sst_chunked.resample(time='1MS').mean()      # Resample satellite similarly (lazy)

    # ===== COMPUTE RESAMPLED DATA IN PARALLEL =====
    with ProgressBar(), ThreadPoolExecutor(max_workers=2) as executor:
        # Schedule parallel compute tasks for model and satellite with thread scheduler
        future_model = executor.submit(model_sst_monthly_lazy.compute, scheduler='threads')
        future_sat = executor.submit(sat_sst_monthly_lazy.compute, scheduler='threads')

        model_sst_monthly = future_model.result()  # Get computed model results
        sat_sst_monthly = future_sat.result()      # Get computed satellite results

    return model_sst_monthly, sat_sst_monthly