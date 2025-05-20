from typing import Dict, List, Union
import pandas as pd
import numpy as np

###############################################################################
def leapyear(year: int) -> int:
    """
    Check if a given year is a leap year.

    Args:
        year (int): Year as positive integer.

    Returns:
        int: 1 if leap year, 0 otherwise.
    """
    assert isinstance(year, int) and year > 0, "Year must be a positive integer"
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 1
    return 0
###############################################################################

###############################################################################
def true_time_series_length(
    nf: int, chlfstart: List[int], chlfend: List[int], DinY: int
) -> int:
    """
    Calculate the true time series length in days over multiple files.

    Args:
        nf (int): Number of files.
        chlfstart (List[int]): List of start years per file.
        chlfend (List[int]): List of end years per file.
        DinY (int): Number of days in a year (typically 365).

    Returns:
        int: Total number of days including leap years.
    """
    assert isinstance(nf, int) and nf > 0, "nf must be a positive integer"
    assert isinstance(chlfstart, list) and isinstance(chlfend, list), "chlfstart and chlfend must be lists"
    assert len(chlfstart) == nf and len(chlfend) == nf, "chlfstart and chlfend must have length equal to nf"
    assert all(isinstance(x, int) for x in chlfstart + chlfend), "chlfstart and chlfend must contain integers"
    assert all(end >= start for start, end in zip(chlfstart, chlfend)), "Each chlfend must be >= corresponding chlfstart"
    assert isinstance(DinY, int) and DinY == 365, "DinY must be 365"

    Truedays = 0
    for n in range(nf):
        for y in range(chlfstart[n], chlfend[n] + 1):
            Truedays += DinY + leapyear(y)

    return Truedays
###############################################################################

###############################################################################
def split_to_monthly(
    yearly_data: Dict[int, Union[pd.Series, pd.DataFrame]]
) -> Dict[int, List[Union[pd.Series, pd.DataFrame]]]:
    """
    Args: 
        yearly_data (dict): Dictionary keyed by year, values are pandas Series or DataFrames with datetime index.

    Returns:
    dict: Dictionary keyed by year, values are lists of monthly data (Series/DataFrames).
    """
    monthly_data_dict = {}
    for year, data in yearly_data.items():
        monthly_data = [data[data.index.month == month] for month in range(1, 13)]
        monthly_data_dict[year] = monthly_data

    return monthly_data_dict
###############################################################################

###############################################################################
def split_to_yearly(
    series: pd.Series, 
    unique_years: List[Union[int, str]]
) -> Dict[Union[int, str], pd.Series]:
    """
    Splits a pandas Series into a dictionary by year based on the datetime index.

    Args:
        series (pd.Series): Time-indexed pandas Series with a datetime index.
        unique_years (List[int|str]): List of years to split the series into.

    Returns:
        Dict[year, pd.Series]: Dictionary keyed by year containing Series filtered by that year.
    """
    assert isinstance(series.index, pd.DatetimeIndex), "series must have a DatetimeIndex"
    assert all(isinstance(year, (int, str)) for year in unique_years), "unique_years must contain only ints or strs"

    yearly_data = {}
    for year in unique_years:
        year_data = series[series.index.year == int(year)]
        yearly_data[year] = year_data
    return yearly_data
###############################################################################

###############################################################################
def get_common_years(
    data_dict: Dict[str, Dict[Union[int, str], object]], 
    mod_key: str, 
    sat_key: str
) -> List[Union[int, str]]:
    """Get sorted years present in both model and satellite datasets."""
    mod_years = set(data_dict.get(mod_key, {}).keys())
    sat_years = set(data_dict.get(sat_key, {}).keys())
    common_years = sorted(mod_years.intersection(sat_years))
    return common_years
###############################################################################

###############################################################################
def get_season_mask(dates: Union[pd.DatetimeIndex, pd.Series], season_name: str) -> np.ndarray:
    """
    Generate a boolean mask for the given season on a datetime index or series.

    Args:
        dates (pd.DatetimeIndex or pd.Series): Datetime-like index or series with datetime index.
        season_name (str): Season name. One of {'DJF', 'MAM', 'JJA', 'SON'}.

    Returns:
        np.ndarray: Boolean mask array indicating whether each date falls in the season.

    Raises:
        ValueError: If season_name is not one of the expected values.
    """
    valid_seasons = {'DJF', 'MAM', 'JJA', 'SON'}
    if season_name not in valid_seasons:
        raise ValueError(f"Invalid season name: {season_name}. Expected one of {valid_seasons}")

    months = dates.month
    if season_name == 'DJF':
        return (months == 12) | (months == 1) | (months == 2)
    elif season_name == 'MAM':
        return (months >= 3) & (months <= 5)
    elif season_name == 'JJA':
        return (months >= 6) & (months <= 8)
    elif season_name == 'SON':
        return (months >= 9) & (months <= 11)
###############################################################################
    