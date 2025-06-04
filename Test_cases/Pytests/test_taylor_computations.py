import numpy as np
import pandas as pd
import pytest
import skill_metrics as sm
from Hydrological_model_validator.Processing.Taylor_computations import (
    compute_taylor_stat_tuple,
    compute_std_reference,
    compute_norm_taylor_stats,
    build_all_points,
    compute_yearly_taylor_stats,
)

###############################################################################
# --- Tests for compute_taylor_stat_tuple ---
###############################################################################


# Tests when model and satellite data are identical.
def test_taylor_stat_basic():
    mod = np.array([1, 2, 3])
    sat = np.array([1, 2, 3])
    label = "test"
    result = compute_taylor_stat_tuple(mod, sat, label)
    
    # The output should be a tuple containing the label and stats
    assert isinstance(result, tuple)
    
    # The first element must be the label passed in, ensuring identification
    assert result[0] == label
    
    # The subsequent elements represent statistics and should be floats
    assert all(isinstance(x, float) for x in result[1:])

# Tests with reversed model and satellite data.
def test_taylor_stat_different_arrays():
    mod = np.array([1, 2, 3])
    sat = np.array([3, 2, 1])
    label = "diff"
    
    _, sdev, crmsd, ccoef = compute_taylor_stat_tuple(mod, sat, label)
    
    # Standard deviation should be positive since data is not constant
    assert sdev > 0
    
    # Correlation coefficient must lie between -1 and 1 (inclusive)
    assert -1 <= ccoef <= 1
    
    # Centered RMS difference should be a float value representing error magnitude
    assert isinstance(crmsd, float)

# Tests that empty input arrays raise an error.
def test_taylor_stat_empty_arrays():
    mod = np.array([])
    sat = np.array([])
    label = "empty"
    
    # Function should raise ValueError when inputs are empty to avoid invalid stats
    with pytest.raises(ValueError):
        compute_taylor_stat_tuple(mod, sat, label)

# Tests handling of single-value arrays.
def test_taylor_stat_single_value():
    mod = np.array([1])
    sat = np.array([1])
    label = "single"
    
    result = compute_taylor_stat_tuple(mod, sat, label)
    
    # Label must be preserved regardless of input size
    assert result[0] == label
    
    # Stats outputs remain floats even if based on single values (might be NaN or zero)
    assert all(isinstance(x, float) for x in result[1:])

# Tests handling of arrays with NaN values.
def test_taylor_stat_nan_values():
    mod = np.array([1, np.nan, 3])
    sat = np.array([1, 2, np.nan])
    label = "nan"
    
    result = compute_taylor_stat_tuple(mod, sat, label)
    
    # Label must still be preserved even when NaNs are present
    assert result[0] == label
    
    # The statistics returned should be floats indicating NaNs handled internally
    assert all(isinstance(x, float) for x in result[1:])


###############################################################################
# --- Tests for compute_std_reference ---
###############################################################################


# Tests std computation with valid multi-year, multi-month data.
def test_std_reference_basic():
    # Prepare sample data for two years, each with two months of data as numpy arrays
    data = {
        2000: [np.array([1, 2]), np.array([3, 4])],
        2001: [np.array([2, 3]), np.array([4, 5])]
    }
    # List of years to include in std calculation
    years = [2000, 2001]

    # Compute std deviation for the first month (index 0) across years
    std = compute_std_reference(data, years, 0)
    
    # Standard deviation should be positive with varied input data
    assert std > 0

# Tests error raised for invalid month index (>11).
def test_std_reference_invalid_month_index():
    # Data with only one month, index 0
    data = {2000: [np.array([1, 2])]}
    years = [2000]
    
    # Month index 12 is invalid, expect a ValueError to be raised
    with pytest.raises(ValueError):
        compute_std_reference(data, years, 12)

# Tests error raised when no data exists for specified month.
def test_std_reference_no_data_for_month():
    # Only one month of data (month 0), but querying for month 1 which doesn't exist
    data = {2000: [np.array([1, 2])]}
    years = [2000]
    
    # Expect a ValueError since requested month data is missing
    with pytest.raises(ValueError):
        compute_std_reference(data, years, 1)

# Tests error raised when data arrays for month are empty.
def test_std_reference_empty_data_arrays():
    # Month 0 exists but contains an empty array (no data points)
    data = {2000: [np.array([])]}
    years = [2000]
    
    # Should raise ValueError because std dev cannot be computed on empty data
    with pytest.raises(ValueError):
        compute_std_reference(data, years, 0)

# Tests std computation on single year with constant values (std = 0).
def test_std_reference_single_year_single_month():
    # Data for single year and month with constant values, so std deviation should be zero
    data = {2020: [np.array([10.0, 10.0, 10.0])]}
    years = [2020]

    # Calculate std dev for month 0; expect 0.0 because data has no variation
    std = compute_std_reference(data, years, 0)
    assert std == 0.0

    
###############################################################################
# --- Tests for compute_norm_taylor_stats ---
###############################################################################


# Tests compute_norm_taylor_stats with typical valid input arrays.
def test_norm_taylor_stats_valid_data():
    # Typical model and satellite arrays with some variation
    mod = np.array([1, 2, 3])
    sat = np.array([1, 2, 4])
    # Standard deviation reference to normalize against
    std_ref = 1.0

    # Run the normalized Taylor statistics computation
    result = compute_norm_taylor_stats(mod, sat, std_ref)

    # Assert that the returned dictionary contains all expected keys
    assert "sdev" in result and "crmsd" in result and "ccoef" in result

# Tests that zero std_ref raises ValueError as expected.
def test_norm_taylor_stats_std_ref_zero():
    # Model and satellite data (arbitrary valid arrays)
    mod = np.array([1, 2, 3])
    sat = np.array([1, 2, 3])

    # Zero std_ref is invalid because normalization by zero is undefined
    with pytest.raises(ValueError):
        compute_norm_taylor_stats(mod, sat, 0)

# Tests that negative std_ref raises ValueError as expected.
def test_norm_taylor_stats_std_ref_negative():
    # Model and satellite data (arbitrary valid arrays)
    mod = np.array([1, 2, 3])
    sat = np.array([1, 2, 3])

    # Negative std_ref is invalid since standard deviation cannot be negative
    with pytest.raises(ValueError):
        compute_norm_taylor_stats(mod, sat, -1)

# Tests handling of arrays containing only NaNs (should return None).
def test_norm_taylor_stats_no_valid_data():
    # Both input arrays contain only NaNs (no valid data points)
    mod = np.array([np.nan, np.nan])
    sat = np.array([np.nan, np.nan])
    std_ref = 1.0

    # Expect function to return None as no statistics can be computed
    result = compute_norm_taylor_stats(mod, sat, std_ref)
    assert result is None

# Tests function behavior with partially valid data (some NaNs present).
def test_norm_taylor_stats_partial_valid_data():
    # Model array has NaNs, but satellite array is valid for some elements
    mod = np.array([1, np.nan, 3])
    sat = np.array([1, 2, 3])
    std_ref = 1.0

    # The function should successfully compute stats ignoring NaNs
    result = compute_norm_taylor_stats(mod, sat, std_ref)
    assert result is not None


###############################################################################
# --- Tests for build_all_points ---
###############################################################################


# Basic functionality test for build_all_points with typical input data.
def test_build_all_points_basic():
    # Prepare sample data with 12 months of arrays for one year, repeated values
    data = {
        "model": {
            2020: [np.array([1, 2]), np.array([3, 4])] * 6  # 12 arrays total (6x2)
        },
        "satellite": {
            2020: [np.array([1, 2]), np.array([3, 4])] * 6
        }
    }
    # Call function to build combined dataframe and list of years
    df, years = build_all_points(data)

    # Check output is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)
    # Confirm 'sdev' column exists, indicating Taylor stats computed
    assert "sdev" in df.columns
    # Confirm 2020 is correctly included in the list of years
    assert 2020 in years

# Tests build_all_points with data that has full 12 months instead of expected 5.
def test_build_all_points_missing_month():
    # Use data with 12 months of identical arrays, possibly more than expected
    data = {
        "model": {2020: [np.array([1, 2])] * 12},        # 12 months instead of 5
        "satellite": {2020: [np.array([1, 2])] * 12}     # 12 months instead of 5
    }
    # The function should still build a non-empty dataframe
    df, years = build_all_points(data)
    assert not df.empty

# Tests behavior when all monthly data arrays are empty, expecting an error.
def test_build_all_points_empty_data():
    # Prepare data with empty arrays for all months
    data = {
        "model": {2020: [np.array([])] * 12},
        "satellite": {2020: [np.array([])] * 12}
    }
    # Expect ValueError because no valid data exists to compute stats
    with pytest.raises(ValueError):
        df, years = build_all_points(data)

# Verifies that build_all_points correctly handles multiple years of data.
def test_build_all_points_multiple_years():
    # Provide data for two years, each with 12 months of simple arrays
    data = {
        "model": {
            2020: [np.array([1])] * 12,
            2021: [np.array([2])] * 12,
        },
        "satellite": {
            2020: [np.array([1])] * 12,
            2021: [np.array([2])] * 12,
        }
    }
    # Generate dataframe and list of years
    df, years = build_all_points(data)

    # Assert both years are present in the output
    assert set(years) == {2020, 2021}

# Tests handling of NaN returned by compute_std_reference, expecting an empty dataframe.
def test_build_all_points_nan_std_ref(monkeypatch):
    # Mock compute_std_reference to always return NaN, simulating invalid std
    def nan_std_ref(*args, **kwargs):
        return float('nan')

    monkeypatch.setattr('Hydrological_model_validator.Processing.Taylor_computations.compute_std_reference', nan_std_ref)

    data = {
        "model": {2020: [np.array([1])] * 12},
        "satellite": {2020: [np.array([1])] * 12}
    }

    # When std_ref is NaN, build_all_points should return an empty DataFrame
    df, years = build_all_points(data)
    assert df.empty


###############################################################################
# --- Tests for compute_yearly_taylor_stats ---
###############################################################################


# Tests yearly Taylor stats with valid data for a single year, expecting proper outputs.
def test_yearly_taylor_stats_basic():
    # Create monthly datetime index for all months in 2020 (start of each month)
    months = pd.date_range("2020-01-01", periods=12, freq="MS")

    # Create model data with 12 values alternating 1 and 2, indexed by month
    model_data_2020 = pd.Series(np.array([1, 2] * 6), index=months)
    # Create satellite data with the same values as model data (perfect match)
    satellite_data_2020 = pd.Series(np.array([1, 2] * 6), index=months)

    data = {
        "model": {2020: model_data_2020},
        "satellite": {2020: satellite_data_2020}
    }

    # Compute yearly Taylor statistics and reference std dev
    stats, std_ref = compute_yearly_taylor_stats(data)

    # Assert that returned stats are a list (presumably tuples per year/month)
    assert isinstance(stats, list)
    # Assert std_ref is a float representing the std dev reference value
    assert isinstance(std_ref, float)
    # Ensure std_ref is positive, indicating meaningful variation in data
    assert std_ref > 0

# Ensures function raises ValueError when input data arrays are empty.
def test_yearly_taylor_stats_empty_data():
    data = {
        "model": {
            2020: [np.array([])] * 12,  # 12 months with empty arrays, no data
        },
        "satellite": {
            2020: [np.array([])] * 12,  # Same for satellite data
        }
    }
    # Function should raise ValueError due to lack of data to process
    with pytest.raises(ValueError):
        compute_yearly_taylor_stats(data)

# Simulates NaN standard deviation reference causing ValueError during computation.
def test_yearly_taylor_stats_nan_std_ref(monkeypatch):
    # Mock numpy.concatenate to always return array with NaN,
    # simulating failure to produce valid concatenated data
    def nan_concat(*args, **kwargs):
        return np.array([np.nan])

    monkeypatch.setattr('numpy.concatenate', nan_concat)

    data = {
        "model": {2020: [np.array([1])] * 12},
        "satellite": {2020: [np.array([1])] * 12}
    }

    # Expect ValueError because std_ref calculation results in NaN
    with pytest.raises(ValueError):
        compute_yearly_taylor_stats(data)

# Tests that missing year data raises a ValueError.
def test_yearly_taylor_stats_missing_years():
    data = {
        "model": {},       # No years provided in model data
        "satellite": {},   # No years provided in satellite data
    }
    # Should raise error since there's nothing to compute stats on
    with pytest.raises(ValueError):
        compute_yearly_taylor_stats(data)

# Verifies function handles multiple years of data, returning stats for each year.
def test_yearly_taylor_stats_multiple_years():
    # Create monthly datetime indices for two consecutive years
    months_2020 = pd.date_range("2020-01-01", periods=12, freq="MS")
    months_2021 = pd.date_range("2021-01-01", periods=12, freq="MS")

    # Model data: year 2020 with ones, year 2021 with twos
    model_2020 = pd.Series(np.ones(12), index=months_2020)
    model_2021 = pd.Series(np.ones(12) * 2, index=months_2021)

    # Satellite data mirrors model data exactly per year
    satellite_2020 = pd.Series(np.ones(12), index=months_2020)
    satellite_2021 = pd.Series(np.ones(12) * 2, index=months_2021)

    data = {
        "model": {
            2020: model_2020,
            2021: model_2021,
        },
        "satellite": {
            2020: satellite_2020,
            2021: satellite_2021,
        }
    }

    # Compute yearly Taylor stats for both years
    stats, std_ref = compute_yearly_taylor_stats(data)

    # Assert stats list contains results for both years (2 entries)
    assert len(stats) == 2
