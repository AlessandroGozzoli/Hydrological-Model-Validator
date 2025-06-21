import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch

from Hydrological_model_validator.Processing.time_utils import (
    leapyear,
    true_time_series_length,
    split_to_monthly,
    split_to_yearly,
    get_common_years,
    get_season_mask,
    resample_and_compute,
    is_invalid_time_index,
    prompt_for_datetime_index,
    ensure_datetime_index
)

###############################################################################
# leapyear tests
###############################################################################


# Tests that a typical leap year (divisible by 4 but not 100) returns 1.
def test_leapyear_common_leap():
    assert leapyear(2020) == 1

# Tests that a century year not divisible by 400 is not a leap year (returns 0).
def test_leapyear_century_non_leap():
    assert leapyear(1900) == 0

# Tests that a century year divisible by 400 is a leap year (returns 1).
def test_leapyear_century_leap():
    assert leapyear(2000) == 1

# Tests that a non-leap year returns 0.
def test_leapyear_non_leap():
    assert leapyear(2023) == 0

# Tests that an invalid negative year raises a ValueError.
def test_leapyear_invalid_year():
    with pytest.raises(ValueError):
        leapyear(-4)


###############################################################################
# true_time_series_length tests
###############################################################################


# Checks correct length calculation for a single leap year (366 days).
def test_true_time_series_length_single_year():
    # 2020 is a leap year with 366 days
    # Expect the function to correctly return 366 days for one year period
    assert true_time_series_length([2020], [2020], 365) == 366

# Checks correct length for multiple years including leap and non-leap years.
def test_true_time_series_length_multiple_years():
    # Range covers 1999 to 2002 (4 years), including 2000 leap year
    # 3 normal years * 365 + 1 leap year * 366 = 1461 total days
    assert true_time_series_length([1999, 2001], [2000, 2002], 365) == 1461

# Ensures ValueError is raised when start and end year lists have different lengths.
def test_true_time_series_length_mismatched_lists():
    # Mismatched lengths for start_years and end_years should cause error
    # Function expects paired lists of equal length to compute durations
    with pytest.raises(ValueError):
        true_time_series_length([2000], [2001, 2002], 365)

# Ensures ValueError is raised when the daily-in-year (diny) parameter is invalid.
def test_true_time_series_length_invalid_diny():
    # Only 365 is valid for diny here; 366 would be invalid parameter
    # Function should raise error for invalid daily count per year
    with pytest.raises(ValueError):
        true_time_series_length([2000], [2001], 366)


###############################################################################
# split_to_monthly tests
###############################################################################


# Tests basic splitting of a yearly Series into 12 monthly Series.
def test_split_to_monthly_basic():
    # Create daily datetime index for entire year 2020
    dates = pd.date_range('2020-01-01', '2020-12-31')
    # Create a Series with incremental values indexed by the dates
    data = pd.Series(np.arange(len(dates)), index=dates)
    yearly = {2020: data}
    
    # Split the yearly data into months
    monthly = split_to_monthly(yearly)
    
    # Expect 12 monthly Series, one for each month of the year
    assert len(monthly[2020]) == 12
    
    # Verify that each month is a pandas Series
    assert all(isinstance(m, pd.Series) for m in monthly[2020])

# Tests that months without data are empty Series and January has correct length.
def test_split_to_monthly_empty_month():
    # Create data only for January 2020
    dates = pd.date_range('2020-01-01', '2020-01-31')
    data = pd.Series(np.arange(len(dates)), index=dates)
    yearly = {2020: data}
    
    # Split yearly data to monthly
    monthly = split_to_monthly(yearly)
    
    # January (month 0) should have 31 days of data
    assert monthly[2020][0].shape[0] == 31
    
    # Other months (1-11) should be empty Series since no data
    for m in range(1, 12):
        assert monthly[2020][m].empty

# Tests that a non-integer year key raises a ValueError.
def test_split_to_monthly_invalid_year_key():
    # Passing a string key instead of int should cause error,
    # since year keys are expected to be integers for processing
    with pytest.raises(ValueError):
        split_to_monthly({'2020': pd.Series([1], index=pd.date_range('2020-01-01', periods=1))})

# Tests that input data not a pandas Series raises a ValueError.
def test_split_to_monthly_invalid_data_type():
    # Passing list instead of pandas Series should raise error
    # Function expects pandas Series indexed by datetime
    with pytest.raises(ValueError):
        split_to_monthly({2020: [1, 2, 3]})


###############################################################################
#  split_to_yearly tests
###############################################################################


# Tests basic splitting of a Series with datetime index into yearly Series by given years.
def test_split_to_yearly_basic():
    # Create a daily datetime index covering 2020 and 2021
    dates = pd.date_range('2020-01-01', '2021-12-31')
    # Create a Series with values indexed by the datetime
    s = pd.Series(range(len(dates)), index=dates)
    
    # Split the Series into yearly chunks for years 2020 and 2021
    result = split_to_yearly(s, [2020, 2021])
    
    # Check that the output keys correspond exactly to the requested years
    assert set(result.keys()) == {2020, 2021}
    
    # Verify each value in the result is a pandas Series
    assert all(isinstance(v, pd.Series) for v in result.values())

# Tests that input Series with non-datetime index raises a ValueError.
def test_split_to_yearly_non_datetime_index():
    # Create a Series with integer index (non-datetime)
    s = pd.Series([1, 2, 3], index=[1, 2, 3])
    
    # Expect a ValueError since the function requires datetime index
    with pytest.raises(ValueError):
        split_to_yearly(s, [2020])

# Tests that invalid or None years in unique_years raise a ValueError.
def test_split_to_yearly_invalid_unique_years():
    # Create Series with valid datetime index for 2020
    dates = pd.date_range('2020-01-01', '2020-12-31')
    s = pd.Series(range(len(dates)), index=dates)
    
    # Passing None as one of the years is invalid and should raise error
    with pytest.raises(ValueError):
        split_to_yearly(s, [2020, None])

# Tests that non-integer year values in unique_years raise a ValueError.
def test_split_to_yearly_year_conversion_fail():
    # Create Series with valid datetime index for 2020
    dates = pd.date_range('2020-01-01', '2020-12-31')
    s = pd.Series(range(len(dates)), index=dates)
    
    # Passing a non-integer string year 'abc' should cause ValueError on conversion
    with pytest.raises(ValueError):
        split_to_yearly(s, ['abc'])


###############################################################################
# get_common_years tests
###############################################################################


# Tests basic functionality to find common years between model and satellite keys.
def test_get_common_years_basic():
    data = {'model': {2020: 1, 2021: 2}, 'satellite': {2021: 'a', 2022: 'b'}}
    
    # Only year 2021 is common to both model and satellite keys, so expect [2021]
    assert get_common_years(data, 'model', 'satellite') == [2021]

# Tests that empty list is returned when no common years exist.
def test_get_common_years_no_common():
    data = {'model': {2020: 1}, 'satellite': {2021: 'a'}}
    
    # No years overlap between model and satellite, so expect empty list
    assert get_common_years(data, 'model', 'satellite') == []

# Tests that a ValueError is raised when the model key is missing in data.
def test_get_common_years_missing_mod_key():
    data = {'satellite': {2021: 'a'}}
    
    # Since 'model' key is missing, function should raise ValueError to indicate bad input
    with pytest.raises(ValueError):
        get_common_years(data, 'model', 'satellite')

# Tests that a ValueError is raised when the value for model key is not a dict.
def test_get_common_years_non_dict_value():
    data = {'model': [2020, 2021], 'satellite': {2021: 'a'}}
    
    # 'model' key exists but points to a list, not a dict, so expect ValueError
    with pytest.raises(ValueError):
        get_common_years(data, 'model', 'satellite')


###############################################################################
# get_season_mask tests
###############################################################################


# Tests that the mask correctly identifies months in the DJF season.
def test_get_season_mask_basic():
    dates = pd.date_range('2023-01-01', periods=12, freq='ME')
    
    # Generate a boolean mask for the 'DJF' season (December, January, February)
    mask = get_season_mask(dates, 'DJF')
    
    assert isinstance(mask, np.ndarray)
    
    # DJF covers Dec, Jan, Feb. The date range starts in January, so Dec not included here,
    # thus mask should identify either 2 (Jan & Feb) or possibly 3 months depending on edge case.
    assert sum(mask) in [2, 3]

# Tests that an invalid season string raises a ValueError.
def test_get_season_mask_invalid_season():
    dates = pd.date_range('2023-01-01', periods=3)
    
    # Passing an invalid season name should raise a ValueError to indicate incorrect input
    with pytest.raises(ValueError):
        get_season_mask(dates, 'ABC')

# Tests that providing non-datetime-like input raises a TypeError.
def test_get_season_mask_invalid_dates_type():
    
    # Providing a list of strings instead of a datetime-like object should raise TypeError
    with pytest.raises(TypeError):
        get_season_mask(['2023-01-01'], 'DJF')

# Tests that function works correctly when a pandas Series is provided instead of a DatetimeIndex.
def test_get_season_mask_series_input():
    dates = pd.date_range('2023-01-01', periods=12, freq='ME')
    s = pd.Series(range(len(dates)), index=dates)
    
    # Should still work when input is a pandas Series indexed by dates, returning boolean mask array
    mask = get_season_mask(s, 'JJA')
    assert isinstance(mask, np.ndarray)


###############################################################################
# resample_and_compute tests
###############################################################################


# Creates a dummy xarray DataArray with daily random data and chunks it.
def create_dummy_xarray(start, periods):
    # Create a daily datetime index starting from 'start' with 'periods' days
    times = pd.date_range(start, periods=periods, freq='D')
    # Create xarray DataArray with random data and the generated times as coordinate
    data = xr.DataArray(np.random.rand(periods), coords=[times], dims=["time"])
    # Chunk the DataArray into blocks of 5 for dask parallelism
    return data.chunk(5)

# Tests that resample_and_compute correctly resamples to monthly frequency and returns expected dims and size.
def test_resample_and_compute_basic():
    model = create_dummy_xarray('2020-01-01', 60)
    sat = create_dummy_xarray('2020-01-01', 60)
    
    # Call the function to resample daily data into monthly aggregates
    model_monthly, sat_monthly = resample_and_compute(model, sat)
    
    # Check that the time dimension is preserved in output DataArrays
    assert 'time' in model_monthly.dims
    assert 'time' in sat_monthly.dims
    
    # Since 60 days span roughly two months, the monthly time dimension size should be <= 2
    assert model_monthly.time.size <= 2
    assert sat_monthly.time.size <= 2

# Tests that resample_and_compute works correctly on chunked xarray DataArrays.
def test_resample_and_compute_chunked():
    model = create_dummy_xarray('2020-01-01', 30).chunk(10)
    sat = create_dummy_xarray('2020-01-01', 30).chunk(10)
    
    # Resample chunked data to monthly frequency
    model_monthly, sat_monthly = resample_and_compute(model, sat)
    
    # 30 days fit into 1 month, so output time dimension size should be <= 1
    assert model_monthly.time.size <= 1

# Tests that the return types from resample_and_compute are xarray DataArrays.
def test_resample_and_compute_return_types():
    model = create_dummy_xarray('2020-01-01', 15)
    sat = create_dummy_xarray('2020-01-01', 15)
    
    model_monthly, sat_monthly = resample_and_compute(model, sat)
    
    # Verify outputs are xarray DataArrays, ensuring expected data structure
    assert isinstance(model_monthly, xr.DataArray)
    assert isinstance(sat_monthly, xr.DataArray)

# Tests that the returned xarray DataArrays are fully computed (not lazy dask arrays).
def test_resample_and_compute_dask_computed():
    model = create_dummy_xarray('2020-01-01', 30)
    sat = create_dummy_xarray('2020-01-01', 30)
    
    model_monthly, sat_monthly = resample_and_compute(model, sat)
    
    # If the data were still lazy dask arrays, they would have a 'compute' method
    # Assert that output arrays are fully computed (no lazy dask compute method present)
    assert not hasattr(model_monthly.data, 'compute')
    assert not hasattr(sat_monthly.data, 'compute')


###############################################################################
# is_invalid_time_index tests
###############################################################################


# Test that a valid datetime index spanning multiple days is NOT considered invalid.
def test_valid_datetime_index():
    # Create a datetime index with 5 daily timestamps
    idx = pd.date_range("2023-01-01", periods=5, freq='D')
    # Should return False because it is a valid, properly spaced datetime index
    assert not is_invalid_time_index(idx), "Valid datetime index flagged as invalid"

# Test that a non-datetime numpy array is considered invalid.
def test_non_datetime_dtype():
    # Create a simple integer numpy array (not datetime)
    arr = np.array([1, 2, 3, 4])
    # Should return True since dtype is not datetime64
    assert is_invalid_time_index(arr), "Non-datetime array not flagged invalid"

# Test that times all within one day but with large time differences are NOT invalid.
def test_all_within_one_day_but_large_diffs():
    # Timestamps all on 1970-01-01 but spaced by several seconds and minutes
    times = pd.to_datetime([
        "1970-01-01 00:00:00",
        "1970-01-01 00:00:10",
        "1970-01-01 00:01:00",
    ])
    # Differences are large (> 1 ms), so time index should be valid
    assert not is_invalid_time_index(times), "Large diffs within 1 day flagged invalid"

# Test that times all within one day with very small differences are considered invalid.
def test_all_within_one_day_and_small_diffs():
    # Timestamps are all within 1 ms differences, likely dummy data
    times = pd.to_datetime([
        "1970-01-01 00:00:00.000000",
        "1970-01-01 00:00:00.000500",
        "1970-01-01 00:00:00.000900",
    ])
    # This should return True, flagged as invalid (dummy/corrupted data)
    assert is_invalid_time_index(times), "Small diffs within 1 day not flagged invalid"

# Test that an empty time index is NOT considered invalid.
def test_empty_time_index():
    # Create an empty datetime index
    idx = pd.DatetimeIndex([])
    # Should return False; empty indexes aren't invalid per the function logic
    assert not is_invalid_time_index(idx), "Empty index incorrectly flagged invalid"

# Test that a single timestamp (no differences) is NOT considered invalid.
def test_single_timestamp():
    # Single timestamp at Unix epoch
    idx = pd.DatetimeIndex(["1970-01-01"])
    # Since no differences exist, it should be considered valid
    assert not is_invalid_time_index(idx), "Single timestamp incorrectly flagged invalid"
    
    
###############################################################################
# prompt_for_datetime_index tests
###############################################################################


# Test that valid user inputs generate a correct DatetimeIndex of requested length and freq
@patch("builtins.input", side_effect=["2020-01-01", "D"])
def test_valid_inputs_generate_correct_index(mock_input):
    """
    Aim: Confirm function returns correct DatetimeIndex with valid date and frequency inputs.
    """
    length = 4
    result = prompt_for_datetime_index(length)
    
    # Assert returned object is DatetimeIndex of requested length
    assert isinstance(result, pd.DatetimeIndex)
    assert len(result) == length
    
    # Assert first date and frequency match user input
    assert result[0] == pd.Timestamp("2020-01-01")
    assert result.freqstr == "D"

# Test that invalid start date input triggers error and reprompt until valid inputs entered
@patch("builtins.input", side_effect=[
    "not-a-date", "D",  # invalid date then daily freq
    "2021-02-01", "D"   # valid date and freq on second try
])
@patch("builtins.print")
def test_invalid_date_input_prompts_again(mock_print, mock_input):
    """
    Aim: Verify function handles invalid date input by reprompting until valid date provided.
    """
    length = 3
    result = prompt_for_datetime_index(length)

    # Check print called at least once for invalid input message
    assert any("Invalid input" in call.args[0] for call in mock_print.call_args_list)

    # Confirm valid result after retry
    assert isinstance(result, pd.DatetimeIndex)
    assert len(result) == length
    assert result[0] == pd.Timestamp("2021-02-01")

# Test that invalid frequency input triggers error and reprompt until valid frequency entered
@patch("builtins.input", side_effect=[
    "2022-03-01", "invalid-freq",  # invalid freq
    "2022-03-01", "H"              # valid freq on second try
])
@patch("builtins.print")
def test_invalid_frequency_prompts_again(mock_print, mock_input):
    """
    Aim: Ensure function handles invalid frequency input by reprompting until valid frequency provided.
    """
    length = 2
    result = prompt_for_datetime_index(length)
    
    # Check error message printed for invalid freq
    assert any("Invalid input" in call.args[0] for call in mock_print.call_args_list)
    
    # Confirm valid DatetimeIndex returned after correction
    assert isinstance(result, pd.DatetimeIndex)
    assert len(result) == length
    assert result[0] == pd.Timestamp("2022-03-01")
    assert result.freqstr == "h"

# Test multiple invalid inputs before finally entering valid date and frequency
@patch("builtins.input", side_effect=[
    "bad-date", "bad-freq",        # both invalid first try
    "2020-12-31", "wrong-freq",    # invalid freq second try
    "2020-12-31", "D"              # valid on third try
])
@patch("builtins.print")
def test_multiple_invalid_inputs_then_valid(mock_print, mock_input):
    """
    Aim: Confirm function can handle multiple rounds of invalid input before success.
    """
    length = 1
    result = prompt_for_datetime_index(length)
    
    # At least two error messages expected
    error_msgs = [call.args[0] for call in mock_print.call_args_list if "Invalid input" in call.args[0]]
    assert len(error_msgs) >= 2
    
    # Final valid result returned
    assert isinstance(result, pd.DatetimeIndex)
    assert len(result) == length
    assert result[0] == pd.Timestamp("2020-12-31")
    assert result.freqstr == "D"
    

###############################################################################
# ensure_datetime_index tests
###############################################################################


# Test that if Series already has a DatetimeIndex, it is returned unchanged without prompts
def test_series_with_datetime_index_returns_unchanged():
    """
    Aim: Confirm function returns the original Series unchanged if index is already DatetimeIndex.
    """
    dates = pd.date_range("2023-01-01", periods=3)
    s = pd.Series([10, 20, 30], index=dates)
    result = ensure_datetime_index(s, "Test Series")
    
    # The index should be unchanged and identical
    assert isinstance(result.index, pd.DatetimeIndex)
    assert all(result.index == dates)

# Test that non-datetime index triggers user prompts and creates new DatetimeIndex with given inputs
@patch("builtins.input", side_effect=["2020-05-01", "D"])
@patch("builtins.print")
def test_non_datetime_index_prompts_and_creates_index(mock_print, mock_input):
    """
    Aim: Check that non-datetime index triggers input prompts and creates a proper DatetimeIndex.
    """
    s = pd.Series([1, 2, 3], index=[0, 1, 2])  # non-datetime index
    
    result = ensure_datetime_index(s, "Sample Series")
    
    # Confirm index is now DatetimeIndex of correct length
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result.index) == len(s)
    assert result.index[0] == pd.Timestamp("2020-05-01")
    assert result.index.freqstr == "D"
    
    # Confirm appropriate print statements were made (including creation message)
    printed_messages = [call.args[0] for call in mock_print.call_args_list]
    assert any("has no DatetimeIndex" in msg for msg in printed_messages)
    assert any("DatetimeIndex created for Sample Series" in msg for msg in printed_messages)

# Test invalid date input should raise error (since ensure_datetime_index does not catch exceptions)
@patch("builtins.input", side_effect=["invalid-date", "D"])
def test_invalid_date_input_raises(monkeypatch):
    """
    Aim: Ensure invalid date input raises an exception since no internal error handling exists.
    """
    s = pd.Series([1, 2], index=[0, 1])
    with pytest.raises(Exception):
        ensure_datetime_index(s, "Bad Date Series")

# Test invalid frequency input raises error (no internal error handling)
@patch("builtins.input", side_effect=["2020-01-01", "invalid-freq"])
def test_invalid_frequency_input_raises(monkeypatch):
    """
    Aim: Ensure invalid frequency input raises an exception (no try/except in function).
    """
    s = pd.Series([1, 2], index=[0, 1])
    with pytest.raises(Exception):
        ensure_datetime_index(s, "Bad Freq Series")