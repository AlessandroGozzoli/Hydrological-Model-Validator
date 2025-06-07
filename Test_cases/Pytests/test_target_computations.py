import numpy as np
import pandas as pd
import pytest
from Hydrological_model_validator.Processing.Target_computations import (
    compute_single_target_stat,
    compute_single_month_target_stat,
    compute_normalised_target_stats,
    compute_normalised_target_stats_by_month,
    compute_target_extent_monthly,
    compute_target_extent_yearly
)


###############################################################################
# Tests for compute_single_target_stat
###############################################################################


# Test compute_single_target_stat with matching model and satellite arrays; expect tuple result with 4 elements including target label
def test_single_target_stat_normal_case():
    mod = np.array([1, 2, 3])
    sat = np.array([1, 2, 3])
    
    # Call function with matching model and satellite data to compute stats
    result = compute_single_target_stat("2020", mod, sat)
    
    # Expect result to be a tuple containing 3 statistics plus the target label
    assert isinstance(result, tuple) and len(result) == 4
    
    # The last element should be the target label passed in
    assert result[3] == "2020"

# Test compute_single_target_stat when satellite data has zero standard deviation; expect None result due to invalid stat calculation
def test_single_target_stat_zero_std():
    # Satellite data with no variability leads to undefined or invalid stats
    sat = np.array([5, 5, 5])
    mod = np.array([1, 2, 3])
    
    # Function should detect zero std deviation and return None to indicate failure or invalid result
    result = compute_single_target_stat("2020", mod, sat)
    assert result is None

# Test compute_single_target_stat returns floats for statistics and preserves target label
def test_single_target_stat_check_values():
    mod = np.array([2, 4, 6])
    sat = np.array([1, 3, 5])
    
    # Compute stats with valid input arrays
    res = compute_single_target_stat("2020", mod, sat)
    
    # The first 3 returned values should be floats representing computed statistics
    assert all(isinstance(x, float) for x in res[:3])
    
    # The last element should be the target label string
    assert res[3] == "2020"

# Test compute_single_target_stat accepts various target label types and returns tuple
def test_single_target_stat_input_types():
    mod = np.array([0])
    sat = np.array([1])
    
    # Test that the function accepts any target label type (string here) and returns a tuple
    result = compute_single_target_stat("year", mod, sat)
    
    # Ensure output is a tuple as expected
    assert isinstance(result, tuple)
    
# Test for input validation
def test_compute_single_target_stat_input_validation():
    valid_year = "2020"
    valid_mod = np.array([1.0, 2.0, 3.0])
    valid_sat = np.array([1.0, 2.0, 3.0])

    # Year is not a string
    with pytest.raises(ValueError, match="year.*string"):
        compute_single_target_stat(2020, valid_mod, valid_sat)

    # mod is not a numpy array
    with pytest.raises(ValueError, match="mod.*numpy arrays"):
        compute_single_target_stat(valid_year, [1, 2, 3], valid_sat)

    # sat is not a numpy array
    with pytest.raises(ValueError, match="mod.*numpy arrays"):
        compute_single_target_stat(valid_year, valid_mod, [1, 2, 3])

    # Shape mismatch
    with pytest.raises(ValueError, match="same shape"):
        compute_single_target_stat(valid_year, np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    # Non-numeric mod
    with pytest.raises(ValueError, match="contain numeric data"):
        compute_single_target_stat(valid_year, np.array(['a', 'b', 'c']), valid_sat)

    # Non-numeric sat
    with pytest.raises(ValueError, match="contain numeric data"):
        compute_single_target_stat(valid_year, valid_mod, np.array(['x', 'y', 'z']))


###############################################################################
# Tests for compute_single_month_target_stat
###############################################################################


# Test compute_single_month_target_stat with normal input; expects tuple result with target year as string
def test_single_month_target_stat_normal_case():
    mod = np.array([1, 2, 3])
    sat = np.array([1, 2, 3])
    
    # Compute stats for a given year and month with valid matching arrays
    result = compute_single_month_target_stat(2020, 5, mod, sat)
    
    # Result should be a tuple containing computed statistics plus the target year label as string
    assert isinstance(result, tuple)
    
    # The fourth element should be the year converted to string
    assert result[3] == "2020"

# Test compute_single_month_target_stat returns None when satellite data has zero standard deviation (no variation)
def test_single_month_target_stat_zero_std():
    # Satellite data with zero variance leads to invalid statistics
    sat = np.array([0, 0, 0])
    mod = np.array([1, 2, 3])
    
    # Passing zero std data should cause function to return None indicating invalid stats
    result = compute_single_month_target_stat(2020, 0, mod, sat)
    assert result is None

# Test compute_single_month_target_stat handles invalid month inputs gracefully and returns a non-None result
def test_single_month_target_stat_invalid_month():
    mod = np.array([1])
    sat = np.array([1])
    
    # Even with an invalid month (e.g., 11 if only 0-10 expected), function should not crash and return a result
    result = compute_single_month_target_stat(2020, 11, mod, sat)
    
    # Result is expected to be something (not None) indicating graceful handling
    assert result is not None

# Test compute_single_month_target_stat returns a 4-element tuple with first element as float statistic
def test_single_month_target_stat_output_values():
    mod = np.array([3, 4])
    sat = np.array([2, 3])
    
    # Compute stats for a valid year and month, expecting a tuple with stats + label
    res = compute_single_month_target_stat(2019, 3, mod, sat)
    
    # Verify the returned tuple length is exactly 4 (three stats + label)
    assert len(res) == 4
    
    # The first element of the tuple should be a float, representing one computed statistic
    assert isinstance(res[0], float)

# Test for input validation 
def test_compute_single_month_target_stat_input_validation():
    valid_mod = np.array([1.0, 2.0, 3.0])
    valid_sat = np.array([1.0, 2.0, 3.0])

    # year is not int
    with pytest.raises(ValueError, match="year.*integer"):
        compute_single_month_target_stat("2020", 5, valid_mod, valid_sat)

    # month is not int
    with pytest.raises(ValueError, match="month.*between 0 and 11"):
        compute_single_month_target_stat(2020, "5", valid_mod, valid_sat)

    # month out of range
    with pytest.raises(ValueError, match="month.*between 0 and 11"):
        compute_single_month_target_stat(2020, 12, valid_mod, valid_sat)

    # mod is not ndarray
    with pytest.raises(ValueError, match="mod.*numpy arrays"):
        compute_single_month_target_stat(2020, 5, [1, 2, 3], valid_sat)

    # sat is not ndarray
    with pytest.raises(ValueError, match="mod.*numpy arrays"):
        compute_single_month_target_stat(2020, 5, valid_mod, [1, 2, 3])

    # shape mismatch
    with pytest.raises(ValueError, match="same shape"):
        compute_single_month_target_stat(2020, 5, np.array([1, 2]), np.array([1, 2, 3]))

    # non-numeric mod (same shape)
    non_numeric_mod = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="contain numeric data"):
        compute_single_month_target_stat(2020, 5, non_numeric_mod, valid_sat)

    # non-numeric sat (same shape)
    non_numeric_sat = np.array(["x", "y", "z"])
    with pytest.raises(ValueError, match="contain numeric data"):
        compute_single_month_target_stat(2020, 5, valid_mod, non_numeric_sat)
        
        
###############################################################################
# Tests for compute_normalised_target_stats
###############################################################################


# Test compute_normalised_target_stats returns correct numpy arrays and labels for valid data input with monkeypatched dependencies
def test_normalised_target_stats_valid(monkeypatch):
    # Mock the function that extracts keys ('model' and 'satellite') from input dictionary
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys',
        lambda d: ('model', 'satellite')
    )

    # Mock the function that returns aligned time series data per year, here returning valid arrays for year 2020
    def mock_get_common_series_by_year(data_dict):
        return [("2020", np.array([1, 2, 3]), np.array([1, 2, 3]))]

    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.data_alignment.get_common_series_by_year',
        mock_get_common_series_by_year
    )

    # Prepare dummy data dictionary with model and satellite time series for year 2020
    dates = pd.date_range('2020-01-01', periods=3)
    data_dict = {
        'model': {2020: pd.Series([1, 2, 3], index=dates)},
        'satellite': {2020: pd.Series([1, 2, 3], index=dates)}
    }

    # Call function under test with monkeypatched dependencies
    bias, crmsd, rmsd, labels = compute_normalised_target_stats(data_dict)

    # Confirm outputs are numpy arrays for the statistics
    assert isinstance(bias, np.ndarray)
    assert isinstance(crmsd, np.ndarray)
    assert isinstance(rmsd, np.ndarray)

    # Confirm labels is a list of strings
    assert isinstance(labels, list)

    # Check that the year label is correctly passed as a string
    assert labels == ["2020"]

    # Confirm that the bias values are floats (each element in the array)
    assert all(isinstance(x, float) for x in bias)

# Test compute_normalised_target_stats raises ValueError when get_common_series_by_year returns no data
def test_normalised_target_stats_no_data(monkeypatch):
    # Patch get_common_series_by_year to return empty list simulating no data available
    monkeypatch.setattr('Hydrological_model_validator.Processing.data_alignment.get_common_series_by_year', lambda _: [])
    
    # Expect the function to raise ValueError when no aligned data is found
    with pytest.raises(ValueError):
        compute_normalised_target_stats({})

# Test compute_normalised_target_stats raises ValueError when all statistics return None due to zero variance in data
def test_normalised_target_stats_all_none(monkeypatch):
    # Patch get_common_series_by_year to return data with zero variance in both arrays (invalid for stats)
    def mock_data(_):
        return [("2020", np.array([1, 1, 1]), np.array([1, 1, 1]))]
    monkeypatch.setattr('Hydrological_model_validator.Processing.data_alignment.get_common_series_by_year', mock_data)

    # Expect the function to raise ValueError because no valid statistics can be computed
    with pytest.raises(ValueError):
        compute_normalised_target_stats({})

# Test compute_normalised_target_stats returns arrays with correct shapes and labels for multi-year input data
def test_normalised_target_stats_output_shapes(monkeypatch):
    # Patch extraction of keys to return standard keys
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys',
        lambda d: ('model', 'satellite')
    )

    # Patch data retrieval to simulate two years with valid numpy arrays
    def mock_data(_):
        return [("2020", np.array([1, 2]), np.array([1, 2])),
                ("2021", np.array([2, 3]), np.array([2, 3]))]
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.data_alignment.get_common_series_by_year',
        mock_data
    )

    # Prepare dummy multi-year data dict (not actually used inside due to monkeypatch)
    data_dict = {
        'model': {
            2020: pd.Series([1.0, 2.0, 3.0]),
            2021: pd.Series([2.0, 3.0, 4.0])
        },
        'satellite': {
            2020: pd.Series([1.0, 2.0, 3.0]),
            2021: pd.Series([2.0, 3.0, 4.0])
        }
    }

    # Call function under test with the multi-year patched data
    bias, crmsd, rmsd, labels = compute_normalised_target_stats(data_dict)

    # Check that output statistics are numpy arrays
    assert isinstance(bias, np.ndarray)
    assert isinstance(crmsd, np.ndarray)
    assert isinstance(rmsd, np.ndarray)

    # Labels should be a list of strings for each year
    assert isinstance(labels, list)
    assert labels == ["2020", "2021"]

    # Output arrays should have one value per year, i.e., shape equals number of years
    assert bias.shape == (2,)
    assert crmsd.shape == (2,)
    assert rmsd.shape == (2,)

# Test for input validation 
def test_compute_normalised_target_stats_input_validation():
    # Not a dictionary
    with pytest.raises(ValueError, match="must be a dictionary"):
        compute_normalised_target_stats(["not", "a", "dict"])

    # Empty dictionary (no valid model/satellite keys)
    with pytest.raises(ValueError, match="No suitable model key"):
        compute_normalised_target_stats({})

    # Now test valid structure but no usable data (e.g., all std == 0)
    data_dict = {
        "model": {
            2000: pd.Series([1.0, 1.0, 1.0]),
            2001: pd.Series([1.0, 1.0, 1.0])
        },
        "satellite": {
            2000: pd.Series([2.0, 2.0, 2.0]),
            2001: pd.Series([3.0, 3.0, 3.0])
        }
    }

    with pytest.raises(ValueError, match="No valid data available to compute statistics"):
        compute_normalised_target_stats(data_dict)
        

###############################################################################
# Tests for compute_normalised_target_stats_by_month
###############################################################################


# Test compute_normalised_target_stats_by_month returns arrays and labels correctly for valid input and monkeypatched dependencies
def test_normalised_target_stats_by_month_valid(monkeypatch):
    # Patch to return expected model and satellite keys
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys',
        lambda d: ('model', 'satellite')
    )

    # Mock function to return aligned monthly model-satellite series for months 0 and 1
    def mock_data(_):
        return [
            (2020, 0, np.array([1, 2]), np.array([1, 2])),
            (2020, 1, np.array([3, 4]), np.array([3, 4]))
        ]
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.data_alignment.get_common_series_by_year_month',
        mock_data
    )

    # Empty pandas Series for months without data
    empty_month = pd.Series(dtype=float)

    # Input dictionary simulating data for model and satellite, with only Jan and Feb populated
    data_dict = {
        'model': {
            2020: [
                pd.Series([1.0, 2.0]),  # January (month 0)
                pd.Series([3.0, 4.0]),  # February (month 1)
            ] + [empty_month] * 10  # Remaining months are empty
        },
        'satellite': {
            2020: [
                pd.Series([1.0, 2.0]),  # January
                pd.Series([3.0, 4.0]),  # February
            ] + [empty_month] * 10
        }
    }

    # Run the function for January (month 0)
    bias, crmsd, rmsd, labels = compute_normalised_target_stats_by_month(data_dict, 0)

    # Validate the output types: expected are numpy arrays and list of labels
    assert isinstance(bias, np.ndarray)
    assert isinstance(crmsd, np.ndarray)
    assert isinstance(rmsd, np.ndarray)
    assert isinstance(labels, list)

# Test compute_normalised_target_stats_by_month raises ValueError when no data returned from get_common_series_by_year_month
def test_normalised_target_stats_by_month_no_data(monkeypatch):
    # Patch to simulate no data found for any month
    monkeypatch.setattr('Hydrological_model_validator.Processing.data_alignment.get_common_series_by_year_month', lambda _: [])

    # Expect ValueError due to lack of usable data
    with pytest.raises(ValueError):
        compute_normalised_target_stats_by_month({}, 0)

# Test compute_normalised_target_stats_by_month raises ValueError for invalid month index
def test_normalised_target_stats_by_month_invalid_month():
    # Month index -1 is invalid (should be 0–11), so function is expected to raise ValueError
    with pytest.raises(ValueError):
        compute_normalised_target_stats_by_month({}, -1)

# Test compute_normalised_target_stats_by_month raises ValueError when all statistics are None (e.g., zero variance)
def test_normalised_target_stats_by_month_all_none(monkeypatch):
    # Patch to return arrays with no variation, which results in statistical measures being None
    def mock_data(_):
        return [(2020, 0, np.array([1, 1]), np.array([1, 1]))]
    monkeypatch.setattr('Hydrological_model_validator.Processing.data_alignment.get_common_series_by_year_month', mock_data)

    # Expect function to raise ValueError due to inability to compute valid statistics
    with pytest.raises(ValueError):
        compute_normalised_target_stats_by_month({}, 0)

# Test for input validation 
def test_compute_normalised_target_stats_by_month_input_validation():
    # Not a dictionary
    with pytest.raises(Exception):  # This might raise a TypeError internally in your `get_common_series_by_year_month`
        compute_normalised_target_stats_by_month("not a dict", 0)

def test_compute_normalised_target_stats_by_month_input_validation_pt2():
    # Valid structure, but requested month is missing
    data_dict = {
        "model": {
            2000: pd.Series([1.0, 2.0], index=pd.date_range("2000-01-01", periods=2, freq='D')),
        },
        "satellite": {
            2000: pd.Series([1.1, 2.1], index=pd.date_range("2000-01-01", periods=2, freq='D')),
        }
    }

    # Should work for month 0 (January)
    data_dict = {
        "model": {
            2000: [np.array([1.0, 2.0])] + [np.array([])] * 11
        },
        "satellite": {
            2000: [np.array([1.1, 2.1])] + [np.array([])] * 11
        }
    }
    
    # This should not raise an error
    compute_normalised_target_stats_by_month(data_dict, 0)

    # Request month index not present (e.g., month 5 with empty arrays)
    with pytest.raises(ValueError, match=r"❌ 'month_index' 5 not found in data\. Available months: \[0\] ❌"):
        compute_normalised_target_stats_by_month(data_dict, 5)

def test_compute_normalised_target_stats_by_month_input_validation_pt():
    # Valid structure, but all data has zero std dev (mod is constant)
    constant_data_dict = {
        "model": {
            2000: [np.array([1.0, 1.0, 1.0])] + [np.array([])] * 11
        },
        "satellite": {
            2000: [np.array([2.0, 2.0, 2.0])] + [np.array([])] * 11
        }
    }

    with pytest.raises(ValueError, match="No valid data available to compute statistics"):
        compute_normalised_target_stats_by_month(constant_data_dict, 0)

    # Mismatched or non-numeric data
    invalid_data_dict = {
        "model": {
            2000: [np.array(["a", "b", "c"])] + [np.array([])] * 11
        },
        "satellite": {
            2000: [np.array([1.0, 2.0, 3.0])] + [np.array([])] * 11
        }
    }

    with pytest.raises(ValueError, match="contain numeric data"):
        compute_normalised_target_stats_by_month(invalid_data_dict, 0)

###############################################################################
# Tests for compute_target_extent_monthly
###############################################################################


# Test compute_target_extent_monthly returns float when valid RMSD data exists for at least one month
def test_target_extent_monthly_valid(monkeypatch):
    # Mock to simulate RMSD values only for January (month 0)
    def mock_compute(data, month):
        if month == 0:
            bias = np.array([0.1])
            crmsd = np.array([0.2])
            rmsd = np.array([0.5, 1.2])  # Valid RMSD data
            labels = ['label1', 'label2']
            return (bias, crmsd, rmsd, labels)
        else:
            return (np.array([]), np.array([]), np.array([]), [])
    # Patch internal monthly stats function
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats_by_month',
        mock_compute
    )
    # Patch rounding utility
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.stats_math_utils.round_up_to_nearest',
        lambda x: np.ceil(x)
    )
    # Create dummy nested data structure with 12 months of arrays
    year = 2000
    monthly_arrays = [np.array([1.0, 2.0])] + [np.array([]) for _ in range(11)]  # Data only in January
    dummy_data = {
        'model': {
            year: monthly_arrays
        },
        'satellite': {
            year: monthly_arrays
        }
    }
    # Call function with valid input
    result = compute_target_extent_monthly(dummy_data)
    # Assert result is a float (based on how target extent is calculated)
    assert isinstance(result, float)

# Test compute_target_extent_monthly raises ValueError when no valid monthly RMSD data is available
def test_target_extent_monthly_no_valid_data(monkeypatch):
    # Mock all months to return empty data
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats_by_month',
        lambda d, m: (np.array([]), np.array([]), np.array([]), [])
    )
    # Expect function to raise ValueError due to lack of usable RMSD data
    with pytest.raises(ValueError):
        compute_target_extent_monthly({})

# Test that compute_target_extent_monthly correctly applies rounding function on RMSD value
def test_target_extent_monthly_check_rounding(monkeypatch):
    # Mock the monthly stats computation to return a fixed RMSD for one month
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats_by_month',
        lambda d, m: (np.array([]), np.array([]), np.array([0.9]), [])
    )
    # Mock the rounding function to always return 1.0
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.stats_math_utils.round_up_to_nearest',
        lambda x: 1.0
    )
    # Provide valid dummy input with minimal structure
    dummy_data = {
        'model': {
            2000: [np.array([1.0])] + [np.array([]) for _ in range(11)]  # January has data
        },
        'satellite': {
            2000: [np.array([1.0])] + [np.array([]) for _ in range(11)]
        }
    }
    # Call the function and assert the expected result
    result = compute_target_extent_monthly(dummy_data)
    assert isinstance(result, float)
    assert result == 1.0

# Test compute_target_extent_monthly aggregates RMSD values across multiple valid months
def test_target_extent_monthly_combined_data(monkeypatch):
    # Simulate multiple months (0 and 1) having valid RMSD data
    def mock_compute(data, month):
        if month in [0, 1]:
            return (
                np.array([]),
                np.array([]),
                np.array([0.5, 0.7]),  # Multiple RMSD entries
                []
            )
        return (np.array([]), np.array([]), np.array([]), [])  # Other months have no data
    # Patch target stat computation
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats_by_month',
        mock_compute
    )
    # Patch rounding function to apply ceiling
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.stats_math_utils.round_up_to_nearest',
        lambda x: np.ceil(x)
    )
    # Provide valid dummy data for one year with 12 months
    dummy_data = {
        'model': {
            2000: [np.array([1.0])] * 12
        },
        'satellite': {
            2000: [np.array([1.0])] * 12
        }
    }
    # Run the computation
    result = compute_target_extent_monthly(dummy_data)
    # Check result type and correctness
    assert isinstance(result, float)
    assert result == 1.0  # Ceil of max([0.5, 0.7]) is 1.0


###############################################################################
# Tests for compute_target_extent_yearly
###############################################################################


# Test that a valid RMSD list returns the correct maximum (after rounding)
def test_target_extent_yearly_valid(monkeypatch):
    # Mock RMSD values with valid data; max is 2.3
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats',
        lambda d: ([], [], np.array([0.5, 2.3]), [])
    )

    # Rounding function returns ceil value
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.stats_math_utils.round_up_to_nearest',
        lambda x: np.ceil(x)
    )

    # Should round up max RMSD (2.3) to 3.0
    result = compute_target_extent_yearly({})
    assert result >= 2.3  # Rounded value should be >= actual max

# Test that compute_target_extent_yearly raises ValueError when no RMSD data is available
def test_target_extent_yearly_no_valid_data(monkeypatch):
    # Simulate no RMSD data
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats',
        lambda d: ([], [], np.array([]), [])
    )

    import pytest
    # Expect ValueError due to absence of RMSD values
    with pytest.raises(ValueError):
        compute_target_extent_yearly({})

# Test that when all RMSD values are below 1, the result rounds up to at least 1.0
def test_target_extent_yearly_below_threshold(monkeypatch):
    # Only one RMSD value, below 1.0
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats',
        lambda d: ([], [], np.array([0.5]), [])
    )

    # Rounding logic will ceil 0.5 to 1.0
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.stats_math_utils.round_up_to_nearest',
        lambda x: np.ceil(x)
    )

    # Expect minimum extent of 1.0
    result = compute_target_extent_yearly({})
    assert result == 1.0

# Test that an RMSD value exactly at the threshold returns that value
def test_target_extent_yearly_exact_threshold(monkeypatch):
    # RMSD value is exactly 1.0
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats',
        lambda d: ([], [], np.array([1.0]), [])
    )

    # Rounding should keep it as 1.0
    monkeypatch.setattr(
        'Hydrological_model_validator.Processing.stats_math_utils.round_up_to_nearest',
        lambda x: np.ceil(x)
    )

    result = compute_target_extent_yearly({})
    assert result == 1.0
