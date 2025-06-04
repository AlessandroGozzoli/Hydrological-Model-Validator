import pytest
import numpy as np
import pandas as pd
import xarray as xr

from Hydrological_model_validator.Processing.stats_math_utils import (
    fit_huber,
    fit_lowess,
    round_up_to_nearest,
    compute_coverage_stats,
    detrend_dim,
    mean_bias,
    standard_deviation_error,
    cross_correlation,
    corr_no_nan,
    std_dev,
    unbiased_rmse
)

###############################################################################
# Tests for fit_huber
###############################################################################


# Test fit_huber returns arrays of length 100 with sorted x-values for valid input
def test_fit_huber_basic():
    mod = np.array([1, 2, 3, 4])
    sat = np.array([1.2, 1.9, 3.1, 3.9])
    
    # Run fit_huber on sample model and satellite data
    x, y = fit_huber(mod, sat)
    
    # Verify output arrays have expected length (default 100 points for smooth fit line)
    assert len(x) == 100 and len(y) == 100
    
    # Ensure x-values are strictly increasing to represent a valid fitted curve domain
    assert np.all(np.diff(x) > 0)

# Test fit_huber raises ValueError when input arrays are empty
def test_fit_huber_empty_input():
    # Empty input arrays should cause the function to raise an error
    with pytest.raises(ValueError):
        fit_huber(np.array([]), np.array([]))

# Test fit_huber raises ValueError when input arrays have mismatched lengths
def test_fit_huber_mismatched_length():
    # Model and satellite arrays of different lengths are invalid inputs
    with pytest.raises(ValueError):
        fit_huber(np.array([1,2]), np.array([1]))

# Test fit_huber raises ValueError when inputs are not numpy arrays
def test_fit_huber_non_numpy_input():
    # Passing Python lists instead of numpy arrays should raise ValueError
    with pytest.raises(ValueError):
        fit_huber([1, 2, 3], [1, 2, 3])


###############################################################################
# Tests for fit_lowess
###############################################################################


# Test that fit_lowess returns a smoothed array with the correct shape,
# and that the x-values are sorted non-decreasingly.
def test_fit_lowess_basic():
    mod = np.array([1, 2, 3, 4])
    sat = np.array([1.2, 1.8, 2.5, 3.9])
    
    # Apply LOWESS smoothing with a moderate fraction for smoothing
    smooth = fit_lowess(mod, sat, frac=0.5)
    
    # The result should be a 2D array with the same number of points as input
    assert smooth.shape == (4, 2)
    
    # Check that the first column (x-values) are sorted non-decreasingly
    # to ensure a valid regression curve
    assert np.all(np.diff(smooth[:,0]) >= 0)

# Test that providing frac=0 raises a ValueError (invalid smoothing fraction).
def test_fit_lowess_frac_out_of_bounds():
    mod = np.array([1, 2, 3])
    sat = np.array([1, 2, 3])
    
    # frac=0 is invalid for LOWESS smoothing fraction parameter
    with pytest.raises(ValueError):
        fit_lowess(mod, sat, frac=0)

# Test that ValueError is raised when the input arrays have different lengths.
def test_fit_lowess_mismatched_length():
    mod = np.array([1, 2])
    sat = np.array([1])
    
    # Mismatched input lengths should raise an error because
    # LOWESS requires paired data points
    with pytest.raises(ValueError):
        fit_lowess(mod, sat)

# Test that ValueError is raised if inputs are not numpy arrays.
def test_fit_lowess_non_numpy_input():
    # Passing Python lists instead of numpy arrays should cause an error
    with pytest.raises(ValueError):
        fit_lowess([1,2,3], [1,2,3])

        
###############################################################################
# Tests for round_up_to_nearest
###############################################################################


# Test rounding 5.3 up to nearest multiple of 2, expecting 6.0
def test_round_up_basic():
    # 5.3 is not a multiple of 2, so expect rounding up to 6.0
    assert round_up_to_nearest(5.3, 2) == 6.0

# Test rounding 7 up to nearest multiple of 0.5, expecting 7.0 (already a multiple)
def test_round_up_base_0_5():
    # 7 is already a multiple of 0.5, so no change expected
    assert round_up_to_nearest(7, 0.5) == 7.0

# Test that passing a negative base raises a ValueError
def test_round_up_base_negative():
    # Negative base does not make sense for rounding, expect error
    with pytest.raises(ValueError):
        round_up_to_nearest(5, -1)

# Test rounding integer 10 up to nearest multiple of 3, expecting 12.0
def test_round_up_integer_input():
    # 10 is not a multiple of 3, next multiple is 12
    assert round_up_to_nearest(10, 3) == 12.0


###############################################################################
# Tests for compute_coverage_stats
###############################################################################


# Test compute_coverage_stats with basic input: checks output shape and valid percentage range
def test_compute_coverage_basic():
    data = np.array([
        [[1, np.nan], [3, 4]],    # 2 time steps, 2x2 grid, some NaNs included
        [[5, 6], [np.nan, 8]],
    ])
    mask = np.array([[True, False], [True, True]])  # mask selects some valid grid points
    data_pct, cloud_pct = compute_coverage_stats(data, mask)
    # Expect output arrays matching number of time steps
    assert data_pct.shape[0] == 2
    # Percentages should be within valid range [0, 100]
    assert np.all(data_pct <= 100) and np.all(data_pct >= 0)

# Test that a mask with shape not matching data raises ValueError
def test_compute_coverage_mask_shape_mismatch():
    data = np.zeros((1, 2, 2))
    mask = np.zeros((3, 3))  # shape mismatch with data grid dimensions
    # Function should validate mask shape and raise error
    with pytest.raises(ValueError):
        compute_coverage_stats(data, mask)

# Test behavior when mask contains no True points: expect NaN percentages in output
def test_compute_coverage_empty_mask():
    data = np.ones((2, 2, 2))
    mask = np.zeros((2, 2), dtype=bool)  # mask excludes all grid points
    data_pct, cloud_pct = compute_coverage_stats(data, mask)
    # With no valid points, data and cloud percentages should be NaN
    assert np.all(np.isnan(data_pct))
    assert np.all(np.isnan(cloud_pct))

# Test that passing data not 3D raises a ValueError
def test_compute_coverage_data_not_3d():
    data = np.zeros((2, 2))  # only 2D data, not time x grid
    mask = np.ones((2, 2), dtype=bool)
    # Function expects 3D data, should raise error for invalid shape
    with pytest.raises(ValueError):
        compute_coverage_stats(data, mask)


###############################################################################
# Tests for detrend_dim
###############################################################################


# Test detrend_dim on a simple 1D DataArray with time dimension, expect output same shape and type
def test_detrend_dim_basic():
    time = pd.date_range('2000-01-01', periods=10)
    data = xr.DataArray(np.arange(10) + np.random.randn(10)*0.1, dims='time', coords={'time': time})
    result = detrend_dim(data, 'time')
    # Output should be same type and shape as input
    assert isinstance(result, xr.DataArray)
    assert result.shape == data.shape

# Test detrend_dim with a mask, ensure values masked False become NaN in output
def test_detrend_dim_with_mask():
    time = pd.date_range('2000-01-01', periods=10)
    data = xr.DataArray(np.arange(10), dims='time', coords={'time': time})
    mask = xr.DataArray([True]*5 + [False]*5, dims='time', coords={'time': time})
    result = detrend_dim(data, 'time', mask=mask)
    # Values where mask is False should be NaN after detrending
    assert np.isnan(result.values[5:]).all()

# Test detrend_dim when there are insufficient valid points, expect entire result to be NaN
def test_detrend_dim_insufficient_points():
    time = pd.date_range('2000-01-01', periods=6)
    data = xr.DataArray([1, np.nan, np.nan, np.nan, np.nan, np.nan], dims='time', coords={'time': time})
    result = detrend_dim(data, 'time', min_valid_points=2)
    # If not enough valid data points to detrend, entire result is NaN
    assert np.isnan(result).all()

# Test detrend_dim handles NaNs correctly and keeps them in output at the same positions
def test_detrend_dim_nan_handling():
    time = pd.date_range('2000-01-01', periods=10)
    data = xr.DataArray(np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]), dims='time', coords={'time': time})
    result = detrend_dim(data, 'time')
    # NaNs in input remain NaN in output at the same positions
    assert np.isnan(result.values[2])


###############################################################################
# Tests for mean_bias
###############################################################################


# Test mean_bias with simple 1D arrays, expecting the correct numeric bias
def test_mean_bias_basic():
    m = xr.DataArray([1, 2, 3], dims='time')
    o = xr.DataArray([1, 1, 1], dims='time')
    bias = mean_bias(m, o)
    assert float(bias.values) == pytest.approx(1.0)

# Test mean_bias with inputs of different lengths to verify it handles alignment without errors
def test_mean_bias_different_length():
    m = xr.DataArray([1, 2], dims='time')
    o = xr.DataArray([1, 1, 1], dims='time')
    bias = mean_bias(m, o)  # should not raise an error
    # Result can be scalar or DataArray depending on alignment
    assert isinstance(bias, (float, int, np.floating, xr.DataArray))

# Test mean_bias with 2D data arrays and specify dimension over which to compute bias
def test_mean_bias_custom_dim():
    m = xr.DataArray([[1, 2], [3, 4]], dims=('time', 'x'))
    o = xr.DataArray([[0, 1], [2, 3]], dims=('time', 'x'))
    bias = mean_bias(m, o, time_dim='time')
    # Since bias is mean over 'time', expect output shape matching remaining dims
    assert bias.shape == (2,)

# Test mean_bias when model and observation are exactly equal, expecting zero bias
def test_mean_bias_all_equal():
    m = xr.DataArray([5, 5, 5], dims='time')
    o = xr.DataArray([5, 5, 5], dims='time')
    bias = mean_bias(m, o)
    assert float(bias.values) == 0.0


###############################################################################
# Tests for standard_deviation_error
###############################################################################


# Test that standard_deviation_error returns a non-negative value for basic input
def test_standard_deviation_error_basic():
    m = xr.DataArray([1, 2, 3], dims='time')
    o = xr.DataArray([1, 1, 1], dims='time')
    val = standard_deviation_error(m, o)
    assert val >= 0

# Test that standard_deviation_error returns zero when model and observation are equal
def test_standard_deviation_error_equal_data():
    m = xr.DataArray([1, 2, 3], dims='time')
    o = xr.DataArray([1, 2, 3], dims='time')
    val = standard_deviation_error(m, o)
    assert val == 0

# Test that standard_deviation_error raises an exception when inputs have different lengths
def test_standard_deviation_error_different_length():
    m = xr.DataArray([1, 2], dims='time')
    o = xr.DataArray([1, 2, 3], dims='time')
    with pytest.raises(Exception):
        standard_deviation_error(m, o)

# Test that standard_deviation_error handles negative values correctly (result should be non-negative)
def test_standard_deviation_error_negative_values():
    m = xr.DataArray([-1, -2, -3], dims='time')
    o = xr.DataArray([-1, -1, -1], dims='time')
    val = standard_deviation_error(m, o)
    assert val >= 0


###############################################################################
# Tests for cross_correlation
###############################################################################


# Test that cross_correlation returns 1.0 for perfectly positively correlated inputs
def test_cross_correlation_basic():
    # Create two identical series, correlation should be perfect positive (1.0)
    m = xr.DataArray([1, 2, 3, 4], dims='time')
    o = xr.DataArray([1, 2, 3, 4], dims='time')
    
    corr = cross_correlation(m, o)
    
    # Assert correlation is 1.0, meaning perfect positive linear relationship
    assert pytest.approx(corr) == 1.0

# Test that cross_correlation returns -1.0 for perfectly negatively correlated inputs
def test_cross_correlation_inverse():
    # Create one series and its exact negative, correlation should be -1.0 (perfect negative)
    m = xr.DataArray([1, 2, 3], dims='time')
    o = xr.DataArray([-1, -2, -3], dims='time')
    
    corr = cross_correlation(m, o)
    
    # Assert correlation is -1.0, indicating perfect inverse linear relationship
    assert pytest.approx(corr) == -1.0

# Test that cross_correlation raises an exception if input arrays have different lengths
def test_cross_correlation_different_length():
    # Create series of different lengths, correlation calculation is invalid
    m = xr.DataArray([1, 2], dims='time')
    o = xr.DataArray([1, 2, 3], dims='time')
    
    # Expect an exception because lengths mismatch prevents valid correlation computation
    with pytest.raises(Exception):
        cross_correlation(m, o)

# Test that cross_correlation handles constant series (zero variance) safely, returning NaN or zero
def test_cross_correlation_constant_series():
    # Create constant series with zero variance, correlation is undefined mathematically
    m = xr.DataArray([1, 1, 1], dims='time')
    o = xr.DataArray([2, 2, 2], dims='time')
    
    corr = cross_correlation(m, o)
    
    # Assert that the function handles zero variance safely by returning NaN or zero instead of error
    assert np.isnan(corr) or corr == 0


###############################################################################
# Tests for corr_no_nan
###############################################################################


# Test correlation calculation when one series contains NaNs, ensuring result is within valid correlation range
def test_corr_no_nan_basic():
    # Create two series where the first has one NaN value, which should be ignored in correlation calculation
    s1 = pd.Series([1, 2, 3, np.nan])
    s2 = pd.Series([1, 2, 4, 5])
    
    corr = corr_no_nan(s1, s2)
    
    # Correlation must be between -1 and 1 for any valid numeric data
    assert -1 <= corr <= 1

# Test correlation result when both series contain only NaNs, expecting NaN as output
def test_corr_no_nan_all_nan():
    # Both series contain only NaNs, so no valid data points to correlate
    s1 = pd.Series([np.nan, np.nan])
    s2 = pd.Series([np.nan, np.nan])
    
    corr = corr_no_nan(s1, s2)
    
    # Result should be NaN because correlation is undefined without any valid paired data
    assert np.isnan(corr)

# Test correlation when one series has some NaNs and the other has valid data, result should be within valid range
def test_corr_no_nan_partial_nan():
    # Series s2 has one NaN, so correlation is calculated only for indices where both are valid
    s1 = pd.Series([1, 2, 3, 4])
    s2 = pd.Series([1, np.nan, 3, 4])
    
    corr = corr_no_nan(s1, s2)
    
    # Correlation value must be valid between -1 and 1 despite missing data points
    assert -1 <= corr <= 1

# Test correlation for two series with no NaNs, verifying result is within valid correlation range
def test_corr_no_nan_no_nan():
    # Both series contain only valid numeric data, normal correlation calculation
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([4, 5, 6])
    
    corr = corr_no_nan(s1, s2)
    
    # Confirm the computed correlation is within valid bounds
    assert -1 <= corr <= 1


###############################################################################
# Tests for std_dev
###############################################################################


# Test standard deviation calculation on a simple 1D DataArray with varying values; expect positive std dev
def test_std_dev_basic():
    # Create DataArray with increasing values, so variation is expected
    da = xr.DataArray([1, 2, 3, 4], dims='time')
    
    std = std_dev(da)
    
    # Standard deviation should be positive since data values vary
    assert std > 0

# Test standard deviation for a constant DataArray; expect std dev to be zero
def test_std_dev_constant():
    # Constant DataArray with no variation
    da = xr.DataArray([3, 3, 3], dims='time')
    
    std = std_dev(da)
    
    # Standard deviation should be zero when all values are identical
    assert std == 0

# Test standard deviation behavior on an empty DataArray; expect result to be NaN
def test_std_dev_empty():
    # Empty DataArray contains no data points
    da = xr.DataArray([], dims='time')
    
    std = std_dev(da)
    
    # Standard deviation is undefined for empty data, so result should be NaN
    assert np.isnan(std)

# Test standard deviation calculation on a multi-dimensional DataArray; expect output to be a DataArray
def test_std_dev_multidim():
    # Multi-dimensional DataArray with shape (time=2, x=2)
    da = xr.DataArray([[1, 2], [3, 4]], dims=('time','x'))
    
    std = std_dev(da)
    
    # Output should remain an xarray DataArray, reflecting std dev over specified dims
    assert isinstance(std, xr.DataArray)


###############################################################################
# Tests for unbiased_rmse
###############################################################################


# Test unbiased RMSE calculation on basic arrays with differing values; expect non-negative result
def test_unbiased_rmse_basic():
    # Model and observation arrays with different values to simulate typical error
    m = xr.DataArray([1, 2, 3], dims='time')
    o = xr.DataArray([1, 1, 1], dims='time')
    
    rmse = unbiased_rmse(m, o)
    
    # RMSE measures error magnitude, so it should always be zero or positive
    assert rmse >= 0

# Test unbiased RMSE when model and observation are identical; expect zero error
def test_unbiased_rmse_equal():
    # Model and observation are exactly the same, so error should be zero
    m = xr.DataArray([1, 2, 3], dims='time')
    o = xr.DataArray([1, 2, 3], dims='time')
    
    rmse = unbiased_rmse(m, o)
    
    # No difference means unbiased RMSE should be exactly zero
    assert rmse == 0

# Test unbiased RMSE with NaN values present in the model data; expect a valid non-negative result
def test_unbiased_rmse_with_nan():
    # Model contains a NaN value to check if function correctly ignores or handles missing data
    m = xr.DataArray([1, 2, np.nan], dims='time')
    o = xr.DataArray([1, 2, 3], dims='time')
    
    rmse = unbiased_rmse(m, o)
    
    # Despite NaN in input, RMSE should still compute a valid non-negative number
    assert rmse >= 0

# Test unbiased RMSE raises an exception when input arrays have mismatched lengths
def test_unbiased_rmse_different_length():
    # Model and observation arrays with different lengths should raise an error
    m = xr.DataArray([1, 2], dims='time')
    o = xr.DataArray([1, 2, 3], dims='time')
    
    # Expect function to raise due to length mismatch, which is invalid input
    with pytest.raises(Exception):
        unbiased_rmse(m, o)
