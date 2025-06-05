import numpy as np
import pytest
import xarray as xr
import pandas as pd

from Hydrological_model_validator.Processing.Efficiency_metrics import (r_squared,
                                                                        monthly_r_squared,
                                                                        weighted_r_squared,
                                                                        monthly_weighted_r_squared,
                                                                        nse,
                                                                        monthly_nse,
                                                                        relative_nse,
                                                                        monthly_relative_nse,
                                                                        index_of_agreement,
                                                                        monthly_index_of_agreement,
                                                                        ln_nse,
                                                                        monthly_ln_nse,
                                                                        nse_j,
                                                                        monthly_nse_j,
                                                                        index_of_agreement_j,
                                                                        monthly_index_of_agreement_j,
                                                                        relative_index_of_agreement,
                                                                        monthly_relative_index_of_agreement,
                                                                        compute_spatial_efficiency,
                                                                        compute_error_timeseries,
                                                                        compute_stats_single_time)

################################################################################
# Tests for r_squared
################################################################################


# Test perfect positive correlation returns R² close to 1
def test_r_squared_perfect_correlation():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    assert r_squared(obs, pred) == pytest.approx(1.0)

# Test perfect negative correlation returns R² close to 1 (explained by linear fit)
def test_r_squared_perfect_negative_correlation():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([4, 3, 2, 1])
    assert r_squared(obs, pred) == pytest.approx(1.0)

# Test R² calculation correctly ignores NaN values and computes with valid pairs only
def test_r_squared_with_nan_values():
    obs = np.array([1, 2, np.nan, 4])
    pred = np.array([1, 2, 3, np.nan])
    result = r_squared(obs, pred)
    # Only valid pairs (indices 0 and 1) used in calculation
    assert result == pytest.approx(1.0)

# Test that R² returns NaN when there is insufficient valid data for calculation
def test_r_squared_insufficient_data():
    obs = np.array([np.nan, 1])
    pred = np.array([1, np.nan])
    result = r_squared(obs, pred)
    # No valid pairs, so result should be NaN
    assert np.isnan(result)

# Test function accepts list inputs and computes R² correctly
def test_r_squared_non_numpy_input():
    obs = [1, 2, 3, 4]
    pred = [1, 2, 3, 4]
    assert r_squared(obs, pred) == pytest.approx(1.0)


################################################################################
# Tests for monthly_r_squared
################################################################################


# Test monthly R² returns perfect correlation when model and satellite data match exactly
def test_monthly_r_squared_basic():
    data_dict = {
        "modData": {
            year: [np.arange(4).reshape(2, 2) + i for i in range(12)]
            for year in [2000, 2001]
        },
        "satData": {
            year: [np.arange(4).reshape(2, 2) + i for i in range(12)]
            for year in [2000, 2001]
        },
    }
    r2 = monthly_r_squared(data_dict)
    assert len(r2) == 12
    # Expect all months to have R² close to 1 due to identical data
    assert all(np.isclose(val, 1.0) for val in r2)

# Test monthly R² handles NaNs gracefully and returns valid scores
def test_monthly_r_squared_partial_nans():
    data_dict = {
        "modData": {
            2000: [np.array([[1, 2], [np.nan, 4]]) for _ in range(12)],
            2001: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
        },
        "satData": {
            2000: [np.array([[1, 2], [3, np.nan]]) for _ in range(12)],
            2001: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
        },
    }
    r2 = monthly_r_squared(data_dict)
    assert len(r2) == 12
    # R² should be computed ignoring NaNs, no NaN results expected here
    assert all(not np.isnan(val) for val in r2)

# Test monthly R² raises KeyError if required keys are missing in input dict
def test_monthly_r_squared_missing_keys():
    data_dict = {
        "model": {2000: [np.ones((2, 2)) for _ in range(12)]},
        # 'satData' key is missing here
    }
    with pytest.raises(KeyError):
        monthly_r_squared(data_dict)

# Test monthly R² returns NaN for all months when input data is all NaNs
def test_monthly_r_squared_all_nans():
    data_dict = {
        "modData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},
        "satData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},
    }
    r2 = monthly_r_squared(data_dict)
    # All data are NaN, so R² should be NaN for all months
    assert all(np.isnan(val) for val in r2)

# Test monthly R² computes values when model and satellite have different constant values over multiple years
def test_monthly_r_squared_variable_years():
    data_dict = {
        "modData": {
            2000: [np.ones((2, 2)) for _ in range(12)],
            2001: [np.ones((2, 2)) * 2 for _ in range(12)],
        },
        "satData": {
            2000: [np.ones((2, 2)) * 1.5 for _ in range(12)],
            2001: [np.ones((2, 2)) * 1.5 for _ in range(12)],
        },
    }
    r2 = monthly_r_squared(data_dict)
    assert len(r2) == 12
    # R² can be NaN or between 0 and 1 when values differ
    assert all((np.isnan(val) or (0 <= val <= 1)) for val in r2)

    
################################################################################
# Tests for weighted_r_squared
################################################################################


# Test weighted R² returns 1.0 for perfect prediction with slope exactly 1
def test_weighted_r_squared_perfect_fit_slope_1():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    result = weighted_r_squared(obs, pred)
    # slope = 1, so weight = 1 and weighted R² equals standard R² (1)
    assert result == pytest.approx(1.0)

# Test weighted R² is less than 1 when slope of prediction is less than 1
def test_weighted_r_squared_slope_less_than_1():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([0.5, 1, 1.5, 2])  # slope approx 0.5 < 1
    r2 = weighted_r_squared(obs, pred)
    # weight should be close to 0.5, so weighted R² < 1
    assert 0 < r2 < 1

# Test weighted R² is less than 1 when slope of prediction is greater than 1
def test_weighted_r_squared_slope_greater_than_1():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([2, 4, 6, 8])  # slope approx 2 > 1
    r2 = weighted_r_squared(obs, pred)
    # weight ~ 1/2 = 0.5, so weighted R² < 1
    assert 0 < r2 < 1

# Test weighted R² handles NaN values without resulting in NaN output
def test_weighted_r_squared_with_nans():
    obs = np.array([1, 2, np.nan, 4])
    pred = np.array([0.5, 1, 1.5, np.nan])
    result = weighted_r_squared(obs, pred)
    # Should handle NaNs by ignoring them internally
    assert not np.isnan(result)

# Test weighted R² returns NaN when there is insufficient valid data
def test_weighted_r_squared_insufficient_data():
    obs = np.array([np.nan, 1])
    pred = np.array([1, np.nan])
    result = weighted_r_squared(obs, pred)
    assert np.isnan(result)


################################################################################
# Tests for monthly_weighted_r_squared
################################################################################


# Test monthly weighted R² with near-perfect data but small noise and offset
def test_monthly_weighted_r_squared_basic():
    np.random.seed(0)

    mod_data = {
        year: [
            np.full((2, 2), i, dtype=float) + np.random.normal(0, 1e-3, (2, 2))
            for i in range(12)
        ]
        for year in [2000, 2001]
    }

    sat_data = {
        year: [
            np.full((2, 2), i + 0.5, dtype=float) + np.random.normal(0, 1e-3, (2, 2))
            for i in range(12)
        ]
        for year in [2000, 2001]
    }

    data_dict = {"modData": mod_data, "satData": sat_data}

    wr2 = monthly_weighted_r_squared(data_dict)

    assert len(wr2) == 12
    # Offset reduces correlation, so expect values in [0, 0.5]
    assert all(0.0 <= val <= 0.5 for val in wr2)

# Test monthly weighted R² handles partial NaNs correctly and does not produce NaNs in output
def test_monthly_weighted_r_squared_partial_nans():
    data_dict = {
        "modData": {
            2000: [np.array([[1, 2], [np.nan, 4]]) for _ in range(12)],
            2001: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
        },
        "satData": {
            2000: [np.array([[1, 2], [3, np.nan]]) for _ in range(12)],
            2001: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
        },
    }
    wr2 = monthly_weighted_r_squared(data_dict)
    assert len(wr2) == 12
    # NaNs in input should not propagate to output
    assert all(not np.isnan(val) for val in wr2)

# Test monthly weighted R² raises KeyError if required satellite data key is missing
def test_monthly_weighted_r_squared_missing_keys():
    data_dict = {
        "model": {2000: [np.ones((2, 2)) for _ in range(12)]},
        # no satellite key present
    }
    with pytest.raises(KeyError):
        monthly_weighted_r_squared(data_dict)

# Test monthly weighted R² returns NaNs for all months if all data is NaN
def test_monthly_weighted_r_squared_all_nans():
    data_dict = {
        "modData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},
        "satData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},
    }
    wr2 = monthly_weighted_r_squared(data_dict)
    assert all(np.isnan(val) for val in wr2)

# Test monthly weighted R² works correctly when data spans variable years with slight noise
def test_monthly_weighted_r_squared_variable_years():
    data_dict = {
        "modData": {
            2000: [np.ones((2, 2)) + np.random.normal(0, 1e-6, (2, 2)) for _ in range(12)],
            2001: [np.ones((2, 2)) * 2 + np.random.normal(0, 1e-6, (2, 2)) for _ in range(12)],
        },
        "satData": {
            2000: [np.ones((2, 2)) * 1.5 + np.random.normal(0, 1e-6, (2, 2)) for _ in range(12)],
            2001: [np.ones((2, 2)) * 1.5 + np.random.normal(0, 1e-6, (2, 2)) for _ in range(12)],
        },
    }
    wr2 = monthly_weighted_r_squared(data_dict)
    assert len(wr2) == 12
    # Weighted R² should be between 0 and 1 even with noise
    assert all(0 <= val <= 1 for val in wr2)

    
################################################################################
# Tests for nse
################################################################################


# Test NSE returns 1.0 for a perfect fit (predictions exactly match observations)
def test_nse_perfect_fit():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    result = nse(obs, pred)
    assert result == pytest.approx(1.0)

# Test NSE returns a negative value for a poor fit (predictions opposite of observations)
def test_nse_poor_fit():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([4, 3, 2, 1])
    result = nse(obs, pred)
    # Negative NSE indicates poor prediction performance
    assert result < 0

# Test NSE handles NaN values in input arrays and returns a valid number (not NaN)
def test_nse_with_nans():
    obs = np.array([1, 2, np.nan, 4])
    pred = np.array([1, np.nan, 3, 4])
    result = nse(obs, pred)
    # Should ignore NaNs and return a numeric NSE
    assert not np.isnan(result)

# Test NSE returns NaN if there is insufficient valid data to compute the score
def test_nse_insufficient_data():
    obs = np.array([np.nan, 1])
    pred = np.array([1, np.nan])
    result = nse(obs, pred)
    assert np.isnan(result)

# Test NSE returns NaN if observed data has zero variance (undefined NSE)
def test_nse_zero_variance_obs():
    obs = np.array([2, 2, 2, 2])
    pred = np.array([1, 2, 3, 4])
    result = nse(obs, pred)
    # NSE is undefined when observations have zero variance
    assert np.isnan(result)


################################################################################
# Tests for monthly_nse
################################################################################


# Test monthly NSE with basic data having small variations, expecting mostly non-positive NSE values
def test_monthly_nse_basic():
    data_dict = {
        "modData": {
            year: [
                np.ones((2, 2)) * (i + 1e-6) + np.array([[0, 0.1], [0.2, 0.3]])  # add variation
                for i in range(12)
            ]
            for year in [2000, 2001]
        },
        "satData": {
            year: [
                np.ones((2, 2)) * (i + 1e-6) + np.array([[0, 0.05], [0.1, 0.15]])  # add variation
                for i in range(12)
            ]
            for year in [2000, 2001]
        },
    }
    results = monthly_nse(data_dict)
    assert len(results) == 12
    # Expect mostly non-positive NSE due to offset and variation
    assert all(val <= 0.0 for val in results)

# Test monthly NSE handles partial NaNs in the data without returning NaN results
def test_monthly_nse_partial_nans():
    data_dict = {
        "modData": {
            2000: [np.array([[1, 2], [np.nan, 4]]) for _ in range(12)],
            2001: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
        },
        "satData": {
            2000: [np.array([[1, 2], [3, np.nan]]) for _ in range(12)],
            2001: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
        },
    }
    results = monthly_nse(data_dict)
    assert len(results) == 12
    # Ensure results are valid numbers, not NaN, despite some NaNs in input
    assert all(not np.isnan(val) for val in results)

# Test monthly NSE raises a KeyError if the satellite data key is missing
def test_monthly_nse_missing_keys():
    data_dict = {
        "model": {2000: [np.ones((2, 2)) for _ in range(12)]},
        # missing satellite key
    }
    with pytest.raises(KeyError):
        monthly_nse(data_dict)

# Test monthly NSE returns all NaN results when all input data are NaNs
def test_monthly_nse_all_nans():
    data_dict = {
        "modData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},
        "satData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},
    }
    results = monthly_nse(data_dict)
    assert all(np.isnan(val) for val in results)

# Test monthly NSE with variable years and values, ensuring results are within valid NSE bounds or NaN
def test_monthly_nse_variable_years():
    data_dict = {
        "modData": {
            2000: [np.ones((2, 2)) for _ in range(12)],
            2001: [np.ones((2, 2)) * 2 for _ in range(12)],
        },
        "satData": {
            2000: [np.ones((2, 2)) * 1.5 for _ in range(12)],
            2001: [np.ones((2, 2)) * 1.5 for _ in range(12)],
        },
    }
    results = monthly_nse(data_dict)
    assert len(results) == 12
    # NSE valid range is (-∞, 1], allow NaN if calculation is invalid
    assert all(np.isnan(val) or (-np.inf < val <= 1) for val in results)


################################################################################
# Tests for index_of_agreement
################################################################################


# Test IOA returns 1.0 for a perfect fit between observed and predicted values
def test_ioa_perfect_fit():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    # Expect IOA of 1 for perfect match
    result = index_of_agreement(obs, pred)
    assert result == pytest.approx(1.0)

# Test IOA returns a value less than 0.5 for a poor fit (inverse predictions)
def test_ioa_poor_fit():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([4, 3, 2, 1])
    # Poor fit should produce low IOA value
    result = index_of_agreement(obs, pred)
    assert result < 0.5

# Test IOA handles NaN values in observed and predicted arrays without returning NaN
def test_ioa_with_nans():
    obs = np.array([1, 2, np.nan, 4])
    pred = np.array([1, np.nan, 3, 4])
    # NaNs are ignored during calculation, so output should be valid
    result = index_of_agreement(obs, pred)
    assert not np.isnan(result)

# Test IOA returns NaN when there is insufficient valid data for calculation
def test_ioa_insufficient_data():
    obs = np.array([np.nan, 1])
    pred = np.array([1, np.nan])
    # Not enough valid pairs to calculate IOA, result is NaN
    result = index_of_agreement(obs, pred)
    assert np.isnan(result)

# Test IOA returns NaN when denominator is zero (no variation in observed data)
def test_ioa_zero_denominator():
    obs = np.array([2, 2, 2, 2])
    pred = np.array([2, 2, 2, 2])
    # No variability in obs causes zero denominator, returns NaN
    result = index_of_agreement(obs, pred)
    assert np.isnan(result)


################################################################################
# Tests for monthly_index_of_agreement
################################################################################


# Test monthly IOA calculation on synthetic data with slight variation, expecting values between 0 and 1
def test_monthly_ioa_basic():
    data_dict = {
        "modData": {
            year: [
                np.ones((2, 2)) * (i + 1e-6) + np.array([[0, 0.1], [0.2, 0.3]])  # add slight variation to model data
                for i in range(12)
            ]
            for year in [2000, 2001]
        },
        "satData": {
            year: [
                np.ones((2, 2)) * (i + 1e-6) + np.array([[0, 0.05], [0.1, 0.15]])  # add slight variation to satellite data
                for i in range(12)
            ]
            for year in [2000, 2001]
        },
    }
    # Compute monthly IOA values
    results = monthly_index_of_agreement(data_dict)
    assert len(results) == 12
    # IOA values should be valid and between 0 and 1
    assert all(0.0 < val <= 1.0 for val in results)

# Test monthly IOA handles partial NaN values correctly, expecting no NaN in output
def test_monthly_ioa_partial_nans():
    data_dict = {
        "modData": {
            2000: [np.array([[1, 2], [np.nan, 4]]) for _ in range(12)],  # some NaNs in model data
            2001: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
        },
        "satData": {
            2000: [np.array([[1, 2], [3, np.nan]]) for _ in range(12)],  # some NaNs in satellite data
            2001: [np.array([[1, 2], [3, 4]]) for _ in range(12)],
        },
    }
    # IOA calculation should handle NaNs gracefully and produce valid numbers
    results = monthly_index_of_agreement(data_dict)
    assert len(results) == 12
    assert all(not np.isnan(val) for val in results)

# Test function raises KeyError when satellite data key is missing
def test_monthly_ioa_missing_keys():
    data_dict = {
        "model": {2000: [np.ones((2, 2)) for _ in range(12)]},
        # satellite data key intentionally missing
    }
    # Should raise error due to missing satellite data
    with pytest.raises(KeyError):
        monthly_index_of_agreement(data_dict)

# Test monthly IOA returns NaN values when all inputs are NaN
def test_monthly_ioa_all_nans():
    data_dict = {
        "modData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},  # all NaNs in model data
        "satData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},  # all NaNs in satellite data
    }
    # IOA results should be NaN due to no valid data
    results = monthly_index_of_agreement(data_dict)
    assert all(np.isnan(val) for val in results)

# Test monthly IOA with varying years and values, expecting outputs within valid range
def test_monthly_ioa_variable_years():
    data_dict = {
        "modData": {
            2000: [np.ones((2, 2)) for _ in range(12)],
            2001: [np.ones((2, 2)) * 2 for _ in range(12)],
        },
        "satData": {
            2000: [np.ones((2, 2)) * 1.5 for _ in range(12)],
            2001: [np.ones((2, 2)) * 1.5 for _ in range(12)],
        },
    }
    # IOA values should be within expected range (-inf to 1) considering possible data scenarios
    results = monthly_index_of_agreement(data_dict)
    assert len(results) == 12
    assert all(-np.inf < val <= 1 for val in results)

    
################################################################################
# Tests for ln_nse
################################################################################


# Test perfect fit returns ln NSE close to 1
def test_ln_nse_perfect_fit():
    obs = np.array([1, 10, 100])
    pred = np.array([1, 10, 100])
    # Expect ln NSE near 1 for exact match
    result = ln_nse(obs, pred)
    assert result == pytest.approx(1.0)

# Test poor fit yields ln NSE less than 0.5
def test_ln_nse_poor_fit():
    obs = np.array([1, 10, 100])
    pred = np.array([100, 10, 1])
    # Expect low ln NSE value for poor fit
    result = ln_nse(obs, pred)
    assert result < 0.5

# Test handling of NaNs and zero/non-positive values gracefully
def test_ln_nse_with_nans_and_nonpositive():
    obs = np.array([1, 10, 0, np.nan, 5])
    pred = np.array([1, 10, 1, 2, np.nan])
    # Should handle NaNs and zero or negative values without error or NaN result
    result = ln_nse(obs, pred)
    assert not np.isnan(result)

# Test insufficient valid data returns NaN
def test_ln_nse_insufficient_data():
    obs = np.array([np.nan, 1])
    pred = np.array([1, np.nan])
    # Insufficient valid pairs should produce NaN
    result = ln_nse(obs, pred)
    assert np.isnan(result)

# Test zero denominator case returns NaN (e.g., zero variance in log-transformed obs)
def test_ln_nse_zero_denominator():
    obs = np.array([10, 10, 10])
    pred = np.array([5, 5, 5])
    # Zero variance in log(obs) leads to undefined ln NSE (NaN)
    result = ln_nse(obs, pred)
    assert np.isnan(result)


################################################################################
# Tests for monthly_ln_nse
################################################################################


# Test basic case where model and satellite data perfectly match,
# expecting ln NSE values close to 1 for each month.
def test_monthly_ln_nse_basic():
    data_dict = {
        "modelData": {year: [np.arange(1, 5).reshape(2, 2) * (i + 1) for i in range(12)] for year in [2000, 2001]},
        "satData": {year: [np.arange(1, 5).reshape(2, 2) * (i + 1) for i in range(12)] for year in [2000, 2001]},
    }
    results = monthly_ln_nse(data_dict)
    print(results)  # debug output
    # Ensure we get one result per month (12 months)
    assert len(results) == 12
    # ln NSE close to 1 means near perfect model prediction matching satellite data
    assert all(0.9 <= val <= 1.0 for val in results)


# Test handling of some zero, NaN and invalid values in model/satellite data,
# ensuring the function skips invalid points and returns valid ln NSE scores.
def test_monthly_ln_nse_partial_invalid_values():
    data_dict = {
        "modelData": {
            2000: [np.array([[1, 10], [0, np.nan]]) for _ in range(12)],
            2001: [np.array([[1, 10], [5, 7]]) for _ in range(12)],
        },
        "satData": {
            2000: [np.array([[1, 10], [5, 0]]) for _ in range(12)],
            2001: [np.array([[1, 10], [5, 7]]) for _ in range(12)],
        },
    }
    results = monthly_ln_nse(data_dict)
    # Check we get a score per month
    assert len(results) == 12
    # Make sure invalid points (zero, NaN) don't cause NaN outputs
    assert all(not np.isnan(val) for val in results)


# Test that function raises KeyError when expected keys ('modelData' or 'satData') are missing.
def test_monthly_ln_nse_missing_keys():
    data_dict = {
        "model": {2000: [np.ones((2, 2)) * 1 for _ in range(12)]},
        # satellite key missing
    }
    # Expect KeyError because the function requires both keys for computation
    with pytest.raises(KeyError):
        monthly_ln_nse(data_dict)


# Test case where all data values are invalid (zero or NaN),
# expecting the result to be NaN for all months.
def test_monthly_ln_nse_all_invalid():
    data_dict = {
        "modelData": {2000: [np.full((2, 2), 0) for _ in range(12)]},
        "satData": {2000: [np.full((2, 2), np.nan) for _ in range(12)]},
    }
    results = monthly_ln_nse(data_dict)
    # Since no valid data exists, expect all results to be NaN
    assert all(np.isnan(val) for val in results)


# Test data with multiple years having different magnitudes,
# expecting valid ln NSE values for each month, possibly less than or equal to 1.
def test_monthly_ln_nse_variable_years():
    data_dict = {
        "modelData": {
            2000: [np.ones((2, 2)) * 5 for _ in range(12)],
            2001: [np.ones((2, 2)) * 10 for _ in range(12)],
        },
        "satData": {
            2000: [np.array([[7, 8], [9, 8]]) for _ in range(12)],
            2001: [np.array([[7, 8], [9, 8]]) for _ in range(12)],
        },
    }
    results = monthly_ln_nse(data_dict)
    # Confirm we get results for all months
    assert len(results) == 12
    # ln NSE values are bounded by -inf and 1 (max possible)
    assert all(-np.inf < val <= 1 for val in results)

    
################################################################################
# Tests for nse_j
################################################################################


# Test perfect fit where observed and predicted arrays match exactly,
# expecting NSE-j score to be approximately 1.
def test_nse_j_perfect_fit():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    # Perfect fit means numerator and denominator match closely => NSE-j near 1
    assert nse_j(obs, pred) == pytest.approx(1.0)

# Test poor fit where prediction reverses the observations,
# expecting NSE-j score to be less than or equal to zero and less than 1.
def test_nse_j_poor_fit():
    obs = np.array([1, 2, 3])
    pred = np.array([3, 2, 1])
    val = nse_j(obs, pred)
    # Poor fit usually results in NSE-j <= 0, also less than perfect score of 1
    assert val < 1 and val <= 0

# Test behavior when changing the power j parameter,
# expecting stricter penalty (lower or equal NSE-j) with higher j.
def test_nse_j_with_j_greater_than_1():
    obs = np.array([1, 2, 3])
    pred = np.array([1, 2, 2])
    val_j1 = nse_j(obs, pred, j=1)
    val_j2 = nse_j(obs, pred, j=2)
    # Increasing power j typically penalizes errors more, lowering NSE-j value
    assert val_j2 <= val_j1

# Test insufficient data case where valid paired points are missing,
# expecting the function to return NaN.
def test_nse_j_insufficient_data():
    obs = np.array([np.nan, 1])
    pred = np.array([1, np.nan])
    # Not enough valid pairs means calculation can't proceed => NaN result
    assert np.isnan(nse_j(obs, pred))

# Test case with zero variance in observed data,
# expecting the function to return NaN due to zero denominator.
def test_nse_j_zero_denominator():
    obs = np.array([5, 5, 5])
    pred = np.array([1, 2, 3])
    # Zero variance in obs causes division by zero in NSE-j formula => NaN
    assert np.isnan(nse_j(obs, pred))


################################################################################
# Tests for monthly_nse_j
################################################################################


# Test NSE-j calculation on monthly data with high similarity and minimal noise,
# expecting NSE-j values mostly between -1.0 and 0.0 (less than or equal to zero).
def test_monthly_nse_j_high_similarity():
    rng = np.random.default_rng(seed=42)
    data = {
        "model": {
            year: [np.ones((10, 10)) * (i + 1) + rng.normal(0, 0.00001, (10, 10)) for i in range(12)]
            for year in [2000, 2001]
        },
        "sat": {
            year: [np.ones((10, 10)) * (i + 1) + rng.normal(0, 0.00001, (10, 10)) for i in range(12)]
            for year in [2000, 2001]
        },
    }
    results = monthly_nse_j(data)
    # Check that all valid results fall within expected NSE-j range for high similarity
    assert all(-1.0 < v <= 0.0 for v in results if not np.isnan(v))

# Test NSE-j calculation with j=2 on constant-valued data,
# expecting NaN results due to zero variance in observations.
def test_monthly_nse_j_with_j_2():
    data = {
        "model": {2000: [np.ones((2, 2)) * 5 for _ in range(12)]},
        "sat": {2000: [np.ones((2, 2)) * 6 for _ in range(12)]},
    }
    results = monthly_nse_j(data, j=2)
    # Ensure output is a list of correct length
    assert isinstance(results, list)
    assert len(results) == 12
    # Zero variance in obs leads to NaN for all months
    assert all(np.isnan(v) for v in results)

# Test behavior when input dictionary is missing required keys,
# expecting a KeyError to be raised.
def test_monthly_nse_j_missing_keys():
    data = {
        "modelData": {2000: [np.ones((2, 2)) for _ in range(12)]},
        # satellite key missing
    }
    # Missing keys should cause function to raise KeyError
    with pytest.raises(KeyError):
        monthly_nse_j(data)

# Test handling of NaN values in model and satellite data,
# expecting no NaN in the results (function should handle NaNs gracefully).
def test_monthly_nse_j_with_nans():
    data = {
        "model": {2000: [np.array([[1, np.nan], [3, 4]]) for _ in range(12)]},
        "sat": {2000: [np.array([[1, 2], [np.nan, 4]]) for _ in range(12)]},
    }
    results = monthly_nse_j(data)
    # Check result length matches months
    assert len(results) == 12
    # NaNs should be handled internally, so results are valid numbers, not NaN
    assert all(not np.isnan(v) for v in results)

# Test behavior when input arrays for each month are empty,
# expecting all results to be NaN.
def test_monthly_nse_j_empty_month():
    data = {
        "model": {2000: [np.array([[]]) for _ in range(12)]},
        "sat": {2000: [np.array([[]]) for _ in range(12)]},
    }
    results = monthly_nse_j(data)
    # Empty arrays yield no valid data points, so all results should be NaN
    assert all(np.isnan(v) for v in results)

    
##############################################################################
# Tests for index_of_agreement_j
##############################################################################


# Test index_of_agreement_j returns 1.0 for perfect fit of observed and predicted arrays.
def test_index_of_agreement_j_perfect_fit():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    # Perfect match should yield the maximum index value of 1.0
    assert index_of_agreement_j(obs, pred) == pytest.approx(1.0)

# Test index_of_agreement_j returns a value between 0 and 1 for a poor fit,
# ensuring the score respects its expected bounds.
def test_index_of_agreement_j_poor_fit():
    obs = np.array([1, 2, 3])
    pred = np.array([3, 2, 1])
    val = index_of_agreement_j(obs, pred)
    # Index of agreement is bounded [0, 1]; here value should be less than 1 but not negative
    assert val < 1 and val >= 0

# Test index_of_agreement_j with j > 1, verifying outputs remain bounded between 0 and 1,
# and that the results differ for different j values.
def test_index_of_agreement_j_with_j_greater_than_1():
    obs = np.array([1, 2, 3])
    pred = np.array([1, 2, 2])
    val_j1 = index_of_agreement_j(obs, pred, j=1)
    val_j2 = index_of_agreement_j(obs, pred, j=2)

    # For different j, output remains between 0 and 1
    assert 0 <= val_j1 <= 1
    assert 0 <= val_j2 <= 1
    # Results should differ because j controls sensitivity to errors
    assert val_j1 != val_j2

# Test index_of_agreement_j returns NaN when data is insufficient (due to NaNs).
def test_index_of_agreement_j_insufficient_data():
    obs = np.array([np.nan, 1])
    pred = np.array([1, np.nan])
    # With no valid pairs, function returns NaN
    assert np.isnan(index_of_agreement_j(obs, pred))

# Test index_of_agreement_j returns NaN when denominator is zero
# (i.e., no variance in observed values).
def test_index_of_agreement_j_zero_denominator():
    obs = np.array([5, 5, 5])
    pred = np.array([5, 5, 5])  # same as obs → zero denominator

    result = index_of_agreement_j(obs, pred)
    # Zero variance in obs leads to division by zero, returning NaN
    assert np.isnan(result)


##############################################################################
# Tests for monthly_index_of_agreement_j
##############################################################################


# Test monthly_index_of_agreement_j returns values close to 1 for nearly identical model and satellite data.
def test_monthly_index_of_agreement_j_basic():
    data = {
        "model": {
            year: [np.array([[i+1, i+2], [i+3, i+4]]) + 0.01 for i in range(12)]  # slight offset added for near-perfect match
            for year in [2000, 2001]
        },
        "sat": {
            year: [np.array([[i+1, i+2], [i+3, i+4]]) for i in range(12)]  # original values, perfect match apart from small offset
            for year in [2000, 2001]
        },
    }
    results = monthly_index_of_agreement_j(data)
    assert len(results) == 12  # Expect 12 monthly results
    # Values should be very close to 1 for near-perfect agreement
    assert all(0.99 < v <= 1 for v in results if not np.isnan(v))

# Test monthly_index_of_agreement_j with j=3, verifying outputs stay within expected bounds [0,1].
def test_monthly_index_of_agreement_j_with_j_3():
    data = {
        "model": {2000: [np.ones((2, 2)) * 5 for _ in range(12)]},  # constant model data
        "sat": {2000: [np.ones((2, 2)) * 6 for _ in range(12)]},    # constant satellite data different from model
    }
    results = monthly_index_of_agreement_j(data, j=3)
    assert len(results) == 12  # Should return a result for each month
    # Index of agreement is always between 0 and 1, inclusive
    assert all(0 <= v <= 1 for v in results)

# Test that monthly_index_of_agreement_j raises KeyError if required keys are missing in input data.
def test_monthly_index_of_agreement_j_missing_keys():
    data = {
        "mod_data": {2000: [np.ones((2, 2)) for _ in range(12)]},  # wrong key, missing 'model' and 'sat'
    }
    # Missing required keys should raise KeyError
    with pytest.raises(KeyError):
        monthly_index_of_agreement_j(data)

# Test monthly_index_of_agreement_j handles NaNs properly, returning no NaN results for valid input.
def test_monthly_index_of_agreement_j_with_nans():
    data = {
        "model": {2000: [np.array([[1, np.nan], [3, 4]]) for _ in range(12)]},  # contains NaN values
        "sat": {2000: [np.array([[1, 2], [np.nan, 4]]) for _ in range(12)]},    # contains NaNs differently positioned
    }
    results = monthly_index_of_agreement_j(data)
    assert len(results) == 12  # 12 months of results expected
    # The function should gracefully handle NaNs, producing valid scores (no NaNs in results)
    assert all(not np.isnan(v) for v in results)

# Test monthly_index_of_agreement_j returns NaN for months with empty arrays (no data).
def test_monthly_index_of_agreement_j_empty_month():
    data = {
        "model": {2000: [np.array([[]]) for _ in range(12)]},  # empty arrays for each month
        "sat": {2000: [np.array([[]]) for _ in range(12)]},
    }
    results = monthly_index_of_agreement_j(data)
    # No data means calculation is not possible, so all results should be NaN
    assert all(np.isnan(v) for v in results)

    
##############################################################################
# Tests for relative_nse
##############################################################################


# Test relative_nse returns 1.0 for a perfect match between observed and predicted values.
def test_relative_nse_perfect_fit():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    # Perfect match should yield relative NSE of 1.0
    assert relative_nse(obs, pred) == pytest.approx(1.0)

# Test relative_nse returns a value less than 1 for a poor fit between observed and predicted values.
def test_relative_nse_poor_fit():
    obs = np.array([1, 2, 3])
    pred = np.array([3, 2, 1])
    val = relative_nse(obs, pred)
    # Poor fit should produce relative NSE less than 1 (and possibly <= 1)
    assert val <= 1
    assert val < 1

# Test relative_nse correctly handles zeros in observed data by excluding them from calculation.
def test_relative_nse_with_zeros_in_obs():
    obs = np.array([0, 2, 3, 0])
    pred = np.array([1, 2, 2, 0])
    # Points where obs == 0 are excluded, so the calculation only uses valid nonzero obs points
    val = relative_nse(obs, pred)
    # Result should still be a float value (valid calculation)
    assert isinstance(val, float)

# Test relative_nse returns NaN when all observed values are zero, making the metric undefined.
def test_relative_nse_all_obs_zero():
    obs = np.zeros(5)
    pred = np.ones(5)
    # All zero observed values lead to zero denominator, so relative NSE is undefined (NaN)
    assert np.isnan(relative_nse(obs, pred))

# Test relative_nse returns NaN when there is insufficient valid data (e.g., all NaNs or zeros).
def test_relative_nse_insufficient_valid_data():
    obs = np.array([np.nan, 0])
    pred = np.array([1, np.nan])
    # With all invalid points excluded, no data remains → result should be NaN
    assert np.isnan(relative_nse(obs, pred))

# Test relative_nse returns NaN when denominator in calculation is zero (e.g., zero variance in obs).
def test_relative_nse_zero_denominator():
    obs = np.array([2, 2, 2])
    pred = np.array([1, 2, 3])
    # Zero variance in obs causes denominator to be zero → NaN result expected
    assert np.isnan(relative_nse(obs, pred))


##############################################################################
# Tests for monthly_relative_nse
##############################################################################


# Test monthly_relative_nse returns near-perfect scores when model and satellite data match exactly.
def test_monthly_relative_nse_basic():
    data = {
        "model": {
            year: [np.array([[i + 1, i + 2], [i + 3, i + 4]]) for i in range(12)]
            for year in [2000, 2001]
        },
        "sat": {
            year: [np.array([[i + 1, i + 2], [i + 3, i + 4]]) for i in range(12)]
            for year in [2000, 2001]
        },
    }
    results = monthly_relative_nse(data)
    print(results)
    assert len(results) == 12
    assert all(0.99 < v <= 1 for v in results if not np.isnan(v))

# Test monthly_relative_nse excludes zero values in satellite (observed) data during calculations.
def test_monthly_relative_nse_with_zeros_in_sat():
    data = {
        "model": {2000: [np.ones((2, 2)) * 5 for _ in range(12)]},
        "sat": {2000: [np.array([[0, 5], [5, 5]]) for _ in range(12)]},
    }
    results = monthly_relative_nse(data)
    assert len(results) == 12
    # Check that calculation excludes zeros in satellite (obs) data
    assert all(isinstance(v, float) for v in results)

# Test monthly_relative_nse raises KeyError if expected keys ("model" or "sat") are missing.
def test_monthly_relative_nse_missing_keys():
    data = {
        "mod_data": {2000: [np.ones((2, 2)) for _ in range(12)]},
        # missing satellite key
    }
    with pytest.raises(KeyError):
        monthly_relative_nse(data)

# Test monthly_relative_nse handles NaNs in model and satellite data without returning NaN results.
def test_monthly_relative_nse_with_nans():
    data = {
        "model": {2000: [np.array([[1, np.nan], [3, 4]]) for _ in range(12)]},
        "sat": {2000: [np.array([[1, 2], [np.nan, 4]]) for _ in range(12)]},
    }
    results = monthly_relative_nse(data)
    assert len(results) == 12
    assert all(not np.isnan(v) for v in results)

# Test monthly_relative_nse returns NaN for months with empty arrays (no data).
def test_monthly_relative_nse_empty_month():
    data = {
        "model": {2000: [np.array([[]]) for _ in range(12)]},
        "sat": {2000: [np.array([[]]) for _ in range(12)]},
    }
    results = monthly_relative_nse(data)
    assert all(np.isnan(v) for v in results)


##############################################################################
# relative_index_of_agreement
##############################################################################


# Test relative_index_of_agreement returns 1.0 for perfect match between observed and predicted data.
def test_relative_index_of_agreement_perfect_fit():
    obs = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    # Perfect match means all differences are zero, so relative index of agreement should be 1
    assert relative_index_of_agreement(obs, pred) == pytest.approx(1.0)

# Test relative_index_of_agreement returns a value less than 1 for poor fit.
def test_relative_index_of_agreement_poor_fit():
    obs = np.array([1, 2, 3])
    pred = np.array([3, 2, 1])
    val = relative_index_of_agreement(obs, pred)
    # Poor fit expected to produce relative IOA < 1, but value cannot exceed 1
    assert val <= 1
    assert val < 1

# Test relative_index_of_agreement handles zeros in observed data by excluding them properly.
def test_relative_index_of_agreement_with_zeros_in_obs():
    obs = np.array([0, 2, 3, 0])
    pred = np.array([1, 2, 2, 0])
    # Zeros in observed data are excluded, so calculation uses only valid obs points
    val = relative_index_of_agreement(obs, pred)
    # Result should be a float, indicating valid calculation despite zeros
    assert isinstance(val, float)

# Test relative_index_of_agreement returns NaN if all observed values are zero.
def test_relative_index_of_agreement_all_obs_zero():
    obs = np.zeros(5)
    pred = np.ones(5)
    # With all obs zero, denominator in IOA calculation is zero → metric undefined → NaN
    assert np.isnan(relative_index_of_agreement(obs, pred))

# Test relative_index_of_agreement returns NaN when there is insufficient valid data (NaNs and zeros).
def test_relative_index_of_agreement_insufficient_valid_data():
    obs = np.array([np.nan, 0])
    pred = np.array([1, np.nan])
    # No valid paired data points left after excluding NaNs and zeros → NaN expected
    assert np.isnan(relative_index_of_agreement(obs, pred))

# Test relative_index_of_agreement returns NaN when denominator is zero (zero variance in obs).
def test_relative_index_of_agreement_zero_denominator():
    obs = np.array([2, 2, 2])
    pred = np.array([1, 2, 3])
    # Zero variance in observed data makes denominator zero → NaN result expected
    assert np.isnan(relative_index_of_agreement(obs, pred))

 
##############################################################################
# monthly relative_index_of_agreement
##############################################################################    
 

# Test monthly_relative_index_of_agreement returns values close to 1 for similar model and satellite data with small noise.
def test_monthly_relative_index_of_agreement_basic():
    np.random.seed(42)
    data = {
        "model": {
            year: [np.ones((2, 2)) * (i + 1) + np.random.normal(0, 0.001, (2, 2)) for i in range(12)]
            for year in [2000, 2001]
        },
        "sat": {
            year: [np.ones((2, 2)) * (i + 1) + np.random.normal(0, 0.001, (2, 2)) for i in range(12)]
            for year in [2000, 2001]
        },
    }
    results = monthly_relative_index_of_agreement(data)
    print(results)
    # Expecting 12 results (one per month)
    assert len(results) == 12
    # Values should be positive and <= 1, close to 1 since data are similar with only small noise
    assert all(0.0 < v <= 1 for v in results)

# Test monthly_relative_index_of_agreement properly handles zeros in satellite data (observation).
def test_monthly_relative_index_of_agreement_with_zeros_in_sat():
    data = {
        "model": {2000: [np.ones((2, 2)) * 5 for _ in range(12)]},
        "sat": {2000: [np.array([[0, 5], [5, 5]]) for _ in range(12)]},
    }
    results = monthly_relative_index_of_agreement(data)
    # We get one result per month
    assert len(results) == 12
    # All results should be floats, showing function handles zeros gracefully
    assert all(isinstance(v, float) for v in results)

# Test monthly_relative_index_of_agreement raises KeyError when expected keys are missing.
def test_monthly_relative_index_of_agreement_missing_keys():
    data = {
        "mod_data": {2000: [np.ones((2, 2)) for _ in range(12)]},
        # missing 'model' and 'sat' keys → should raise KeyError
    }
    with pytest.raises(KeyError):
        monthly_relative_index_of_agreement(data)

# Test monthly_relative_index_of_agreement handles NaNs properly and returns valid results.
def test_monthly_relative_index_of_agreement_with_nans():
    data = {
        "model": {2000: [np.array([[1, np.nan], [3, 4]]) for _ in range(12)]},
        "sat": {2000: [np.array([[1, 2], [np.nan, 4]]) for _ in range(12)]},
    }
    results = monthly_relative_index_of_agreement(data)
    # Should produce 12 results (one per month)
    assert len(results) == 12
    # No NaN values in results, meaning function handles NaNs internally and produces valid metrics
    assert all(not np.isnan(v) for v in results)

# Test monthly_relative_index_of_agreement returns NaN for months with empty data arrays.
def test_monthly_relative_index_of_agreement_empty_month():
    data = {
        "model": {2000: [np.array([[]]) for _ in range(12)]},
        "sat": {2000: [np.array([[]]) for _ in range(12)]},
    }
    results = monthly_relative_index_of_agreement(data)
    # Empty arrays means no valid data → expect NaN results for all months
    assert all(np.isnan(v) for v in results)


##############################################################################
# vompute spatial efficiency
##############################################################################


# Helper to create a DataArray with given time values for testing.
def create_test_da(time_values):
    # Create dummy data with shape (time, 2, 2) and coordinates on time dimension
    data = np.arange(len(time_values) * 4).reshape(len(time_values), 2, 2)
    coords = {"time": time_values}
    dims = ("time", "x", "y")
    return xr.DataArray(data, coords=coords, dims=dims)

# Test compute_spatial_efficiency with monthly time grouping returns expected dimension and length.
def test_compute_spatial_efficiency_monthly():
    # Create 12 consecutive daily dates starting Jan 1, 2020
    time = np.array("2020-01-01", dtype="datetime64[D]") + np.arange(12)
    da_model = create_test_da(time)
    da_sat = create_test_da(time)
    
    # Compute spatial efficiency with monthly grouping
    mb_all, sde_all, cc_all, rm_all, ro_all, urmse_all = compute_spatial_efficiency(da_model, da_sat, time_group="month")
    
    # Expect dimension named 'month' in output, and length 12 (one per day in January)
    assert mb_all.dims[0] == "month"
    assert len(mb_all) == 12
    assert mb_all.shape[0] == 12

# Test compute_spatial_efficiency with yearly time grouping returns expected dimension and length.
def test_compute_spatial_efficiency_yearly():
    # Create ~2 years of daily data starting Jan 1, 2019
    time = np.array("2019-01-01", dtype="datetime64[D]") + np.arange(24 * 30)  # Approx 2 years
    da_model = create_test_da(time)
    da_sat = create_test_da(time)
    
    # Compute spatial efficiency with yearly grouping
    mb_all, sde_all, cc_all, rm_all, ro_all, urmse_all = compute_spatial_efficiency(da_model, da_sat, time_group="year")
    
    # Expect dimension named 'year' in output and length 2 (two years of data)
    assert mb_all.dims[0] == "year"
    assert len(mb_all) == 2
    assert mb_all.shape[0] == 2

# Test compute_spatial_efficiency raises ValueError if an invalid time_group argument is passed.
def test_compute_spatial_efficiency_invalid_time_group():
    time = np.array("2020-01-01", dtype="datetime64[D]") + np.arange(12)
    da_model = create_test_da(time)
    da_sat = create_test_da(time)
    
    # Invalid grouping string should raise ValueError
    with pytest.raises(ValueError):
        compute_spatial_efficiency(da_model, da_sat, time_group="invalid")
        
###############################################################################
# compute_error_timeseries
###############################################################################


# Helper function to create dummy DataArrays
def create_dummy_xr(time_len=3, lat_len=2, lon_len=2, fill_value=1.0):
    times = pd.date_range("2000-01-01", periods=time_len)
    data = np.full((time_len, lat_len, lon_len), fill_value)
    return xr.DataArray(data, coords=[times, np.arange(lat_len), np.arange(lon_len)], dims=["time", "lat", "lon"])


# Test that compute_error_timeseries returns a DataFrame with correct shape and expected columns.
def test_basic_functionality():
    model = create_dummy_xr(fill_value=2.0)
    sat = create_dummy_xr(fill_value=1.0)
    mask = create_dummy_xr(fill_value=True, time_len=1)  # mask same shape but with bool True
    mask = mask.astype(bool)

    df = compute_error_timeseries(model, sat, mask)
    assert isinstance(df, pd.DataFrame)
    # Ensure the output has a row for each time step in the model data
    assert df.shape[0] == model.sizes['time']
    # Check that the output contains all expected statistical metrics
    assert all(col in df.columns for col in ["mean_bias", "unbiased_rmse", "std_error", "correlation"])


# Test that mean_bias is zero when model and satellite data are all zeros.
def test_all_zeros():
    model = create_dummy_xr(fill_value=0.0)
    sat = create_dummy_xr(fill_value=0.0)
    mask = create_dummy_xr(fill_value=True).astype(bool)
    df = compute_error_timeseries(model, sat, mask)
    # mean_bias should be zero because there is no difference between model and satellite data
    assert all(abs(df['mean_bias']) < 1e-10)


# Test that all statistics are NaN when the mask excludes all data points.
def test_no_masked_points():
    model = create_dummy_xr(fill_value=1.0)
    sat = create_dummy_xr(fill_value=1.0)
    mask = create_dummy_xr(fill_value=False).astype(bool)  # mask everything out
    df = compute_error_timeseries(model, sat, mask)
    # All output values should be NaN since no valid data points exist for calculation
    assert df.isna().all().all()


# Test that mean_bias and std_error are zero when model and satellite data perfectly match.
def test_perfect_match():
    model = create_dummy_xr(fill_value=3.5)
    sat = create_dummy_xr(fill_value=3.5)
    mask = create_dummy_xr(fill_value=True).astype(bool)
    df = compute_error_timeseries(model, sat, mask)
    # Perfect agreement implies no systematic bias
    assert all(abs(df['mean_bias']) < 1e-10)
    # Perfect agreement implies no spread/error difference
    assert all(abs(df['std_error']) < 1e-10)


# Test that compute_error_timeseries handles partial masking without producing all NaNs.
def test_partial_mask():
    model = create_dummy_xr(fill_value=2.0)
    sat = create_dummy_xr(fill_value=1.0)
    mask = create_dummy_xr(fill_value=True).astype(bool)
    # Intentionally exclude a few points to test if function can handle partial data
    mask.values[0, 0, 0] = False
    mask.values[0, 1, 1] = False
    df = compute_error_timeseries(model, sat, mask)
    # Should still produce valid statistics since some data remain unmasked
    assert not df.isna().all().all()


# Test that function correctly processes different lengths of the time dimension.
@pytest.mark.parametrize("time_len", [1, 5, 10])
def test_different_time_lengths(time_len):
    model = create_dummy_xr(time_len=time_len, fill_value=2.0)
    sat = create_dummy_xr(time_len=time_len, fill_value=1.0)
    mask = create_dummy_xr(time_len=time_len, fill_value=True).astype(bool)
    df = compute_error_timeseries(model, sat, mask)
    # Ensure function scales with input time dimension length, producing output with correct shape
    assert df.shape[0] == time_len


###############################################################################
# compute_stats_single_time
###############################################################################


# Test stats calculation on normal input arrays without NaNs.
def test_normal_case():
    m = np.array([1, 2, 3, 4])
    o = np.array([1.5, 2, 2.5, 5])
    stats = compute_stats_single_time(m, o)
    assert isinstance(stats, dict)
    # Verify all expected statistics keys are present
    assert all(key in stats for key in ["mean_bias", "unbiased_rmse", "std_error", "correlation"])
    # mean_bias should match the average difference between model and observation values
    np.testing.assert_almost_equal(stats["mean_bias"], np.mean(m - o))


# Test stats calculation correctly ignores NaNs in input arrays.
def test_partial_nans():
    m = np.array([1, np.nan, 3, 4])
    o = np.array([1, 2, np.nan, 4])
    stats = compute_stats_single_time(m, o)
    # Only non-NaN matching indices (0 and 3) are used for stats calculation
    expected_m = np.array([1, 4])
    expected_o = np.array([1, 4])
    # mean_bias should be computed only from valid pairs
    np.testing.assert_almost_equal(stats["mean_bias"], np.mean(expected_m - expected_o))
    # correlation should be perfect because valid points are identical
    assert np.isclose(stats["correlation"], 1.0)


# Test that all returned stats are NaN when all input values are NaN.
def test_all_nans():
    m = np.array([np.nan, np.nan])
    o = np.array([np.nan, np.nan])
    stats = compute_stats_single_time(m, o)
    # When no valid data points exist, all statistics should be NaN
    assert np.isnan(stats["mean_bias"])
    assert np.isnan(stats["unbiased_rmse"])
    assert np.isnan(stats["std_error"])
    assert np.isnan(stats["correlation"])


# Test that stats reflect zero error and correlation NaN on perfectly matching inputs.
def test_perfect_match_single():
    m = np.array([2, 2, 2])
    o = np.array([2, 2, 2])
    stats = compute_stats_single_time(m, o)
    # No difference means zero bias and zero error metrics
    assert np.isclose(stats["mean_bias"], 0)
    assert np.isclose(stats["unbiased_rmse"], 0)
    assert np.isclose(stats["std_error"], 0)
    # Correlation can be NaN due to zero variance, which is acceptable here
    assert np.isnan(stats["correlation"]) or np.isclose(stats["correlation"], 1)

