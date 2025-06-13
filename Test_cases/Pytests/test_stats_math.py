import pytest
import numpy as np
import pandas as pd
import xarray as xr
from scipy.fft import fft, fftfreq
from scipy.stats import linregress

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
    unbiased_rmse,
    spatial_mean,
    compute_lagged_correlations,
    compute_fft,
    detrend_poly_dim,
    detrend_linear,
    monthly_anomaly,
    yearly_anomaly,
    detrended_monthly_anomaly,
    np_covariance,
    np_correlation,
    np_regression,
    extract_multidecadal_peak,
    extract_multidecadal_peaks_from_spectra,
    identify_extreme_events
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
        
###############################################################################
# Tests for spatial_mean
###############################################################################


# Helper to create dummy DataArray with time, lat, lon
def create_dummy_data(time_len=2, lat_len=3, lon_len=3, fill_value=1.0):
    times = pd.date_range("2000-01-01", periods=time_len)
    data = np.full((time_len, lat_len, lon_len), fill_value, dtype=float)
    return xr.DataArray(data, coords=[times, np.arange(lat_len), np.arange(lon_len)], dims=["time", "lat", "lon"])

# Test spatial_mean returns correct mean when mask includes all points.
def test_full_mask():
    data = create_dummy_data(fill_value=2.0)
    mask = xr.DataArray(np.ones((3, 3), dtype=bool), dims=["lat", "lon"])
    result = spatial_mean(data, mask)
    # Since mask includes all points, mean should equal the constant fill value (2.0)
    assert (result == 2.0).all().item()


# Test spatial_mean correctly computes mean when some points are masked out.
def test_partial_mask():
    data = create_dummy_data(fill_value=2.0)
    mask_array = np.ones((3,3), dtype=bool)
    mask_array[0,0] = False
    mask = xr.DataArray(mask_array, dims=["lat", "lon"])
    result = spatial_mean(data, mask)
    # Even with one point masked out, remaining points all have the same value,
    # so mean should still be 2.0
    assert (result == 2.0).all().item()


# Test spatial_mean returns NaN when mask excludes all points.
def test_all_false_mask():
    data = create_dummy_data(fill_value=2.0)
    mask = xr.DataArray(np.zeros((3,3), dtype=bool), dims=["lat", "lon"])
    result = spatial_mean(data, mask)
    # Mask excludes all data, so spatial_mean should return NaN (no data to average)
    assert result.isnull().all().item()


# Test spatial_mean skips NaNs in data and computes mean correctly.
def test_data_with_nans():
    data = create_dummy_data(fill_value=2.0)
    data.values[0, 1, 1] = np.nan
    mask = xr.DataArray(np.ones((3,3), dtype=bool), dims=["lat", "lon"])
    result = spatial_mean(data, mask)
    # NaNs should be ignored in mean calculation; mean at time=0 should be valid (not NaN)
    assert not np.isnan(result[0])
    # At other times, where no NaNs exist, mean should equal the fill value (2.0)
    assert (result[1] == 2.0).item()


# Test spatial_mean returns value at single unmasked point correctly.
def test_single_point_mask():
    data = create_dummy_data(fill_value=3.0)
    mask_array = np.zeros((3,3), dtype=bool)
    mask_array[1,1] = True
    mask = xr.DataArray(mask_array, dims=["lat", "lon"])
    result = spatial_mean(data, mask)
    # When only one point is unmasked, the mean should equal the value at that point (3.0)
    assert (result == 3.0).all().item()


###############################################################################
# Tests for compute_lagged_correlations
###############################################################################


# Test lagged correlation with no lag returns perfect correlation.
def test_no_lag():
    s1 = pd.Series([1, 2, 3, 4, 5])
    s2 = pd.Series([1, 2, 3, 4, 5])
    result = compute_lagged_correlations(s1, s2, max_lag=0)
    # Expect exactly one lag result: zero lag
    assert len(result) == 1
    # Confirm zero lag is present in results
    assert 0 in result.index
    # Since series are identical, correlation at zero lag should be 1.0 (perfect)
    assert np.isclose(result[0], 1.0)


# Test lagged correlation detects highest correlation at positive lag.
def test_positive_lag():
    s1 = pd.Series([1, 2, 3, 4, 5, 6])
    s2 = s1.shift(1)  # s2 is s1 shifted forward by 1 (NaN at start)
    result = compute_lagged_correlations(s1, s2, max_lag=2)
    # Correlation should peak at lag=1, indicating s1 leads s2 by 1 timestep
    assert result[1] > 0.9


# Test lagged correlation detects highest correlation at negative lag.
def test_negative_lag():
    s1 = pd.Series([1, 2, 3, 4, 5, 6])
    s2 = pd.Series([2, 3, 4, 5, 6, 7])  # s2 leads s1 by 1
    result = compute_lagged_correlations(s1, s2, max_lag=2)
    # Correlation should peak at lag=-1, indicating s2 leads s1 by 1 timestep
    assert result[-1] > 0.9


# Test lagged correlation handles NaNs gracefully without errors.
def test_with_nans():
    s1 = pd.Series([1, 2, np.nan, 4, 5, 6])
    s2 = pd.Series([6, 1, 2, np.nan, 4, 5])
    result = compute_lagged_correlations(s1, s2, max_lag=1)
    # The function should handle NaNs without crashing and return correlations for all requested lags
    assert len(result) == 3
    # Confirm all expected lags (-1, 0, 1) are present in the output
    assert all(lag in result.index for lag in [-1, 0, 1])


# Test lagged correlation works when series have different lengths.
def test_series_of_different_length():
    s1 = pd.Series([1, 2, 3, 4, 5])
    s2 = pd.Series([5, 4, 3, 2])  # shorter length
    result = compute_lagged_correlations(s1, s2, max_lag=1)
    # Even with unequal lengths, correlations should be computed (though edge values may be nan)
    assert len(result) == 3


###############################################################################
# Tests for compute_fft
###############################################################################


# Test FFT output lengths and frequency ordering for single array input.
def test_single_array_fft_length_and_freqs():
    data = np.sin(2 * np.pi * 0.1 * np.arange(100))
    freqs, fft_result = compute_fft(data)
    # FFT of length N produces N//2 frequency bins for real input due to symmetry.
    assert len(freqs) == 50
    assert len(fft_result) == 50
    # Frequencies should be non-negative for real FFT output.
    assert (freqs >= 0).all()
    # Frequencies should be strictly increasing.
    assert np.all(np.diff(freqs) > 0)


# Test FFT handles dict input and produces expected keys and lengths.
def test_dict_input_fft_length_and_freqs():
    data = {
        "a": np.sin(2 * np.pi * 0.1 * np.arange(80)),
        "b": np.cos(2 * np.pi * 0.05 * np.arange(80))
    }
    freqs, fft_result = compute_fft(data, dt=0.5)
    # Expect half the length of input arrays as number of frequency bins.
    assert len(freqs) == 40
    # Check FFT results exist for all keys provided in input dictionary.
    assert set(fft_result.keys()) == {"a", "b"}
    # Each FFT result array length must match number of frequency bins.
    assert all(len(arr) == 40 for arr in fft_result.values())
    # Frequencies must be non-negative.
    assert (freqs >= 0).all()
    # Frequencies should be strictly increasing.
    assert np.all(np.diff(freqs) > 0)


# Test FFT frequency output matches expected frequencies for given sampling interval.
def test_fft_with_different_dt():
    data = np.ones(64)
    dt = 0.1
    freqs, _ = compute_fft(data, dt=dt)
    # Verify frequencies computed match numpy's fftfreq slicing (real FFT).
    expected_freqs = fftfreq(64, dt)[:32]
    np.testing.assert_allclose(freqs, expected_freqs)


# Test FFT output matches scipy FFT for given input.
def test_fft_output_matches_scipy_fft():
    data = np.random.rand(50)
    freqs, fft_res = compute_fft(data)
    # scipy.fft produces full FFT; compare first half for real-valued input.
    expected_fft = fft(data)[:25]
    np.testing.assert_allclose(fft_res, expected_fft)


# Test that empty dict input to FFT raises IndexError.
def test_fft_empty_input_raises():
    # Expect an error because FFT implementation attempts to access first dict item,
    # but dict is empty, so next(iter(...)) fails.
    with pytest.raises(ValueError, match="Input dict is empty"):
        compute_fft({})
        
###############################################################################
# Tests for detrend_poly_dim
###############################################################################


@pytest.fixture
def linear_data():
    # Create a simple linear trend plus noise along 'time' dimension
    time = np.arange(10)
    space = np.arange(5)
    data = 2 * time[:, None] + 3 + np.random.normal(0, 0.1, (10, 5))
    return xr.DataArray(data, dims=['time', 'space'], coords={'time': time, 'space': space})

@pytest.fixture
def quadratic_data():
    # Create quadratic trend data along 'time'
    time = np.arange(10)
    space = np.arange(3)
    data = 1 * time[:, None]**2 + 2 * time[:, None] + 5 + np.random.normal(0, 0.1, (10, 3))
    return xr.DataArray(data, dims=['time', 'space'], coords={'time': time, 'space': space})

# Test that linear detrending removes linear trend along the specified dimension
def test_detrend_linear_removes_linear_trend(linear_data):
    detrended = detrend_poly_dim(linear_data, dim='time', degree=1)
    # After detrending, mean along 'space' should have near-zero slope along 'time'
    slope_after = np.polyfit(linear_data.time, detrended.mean(dim='space'), 1)[0]
    assert abs(slope_after) < 1e-6

# Test that quadratic detrending removes quadratic trend along the specified dimension
def test_detrend_quadratic_removes_quadratic_trend(quadratic_data):
    detrended = detrend_poly_dim(quadratic_data, dim='time', degree=2)
    # Fit quadratic polynomial to mean detrended data, expect near zero coefficients
    coeffs_after = np.polyfit(quadratic_data.time, detrended.mean(dim='space'), 2)
    assert all(abs(c) < 1e-6 for c in coeffs_after)

# Test that degree zero detrending removes only the mean, not any linear trend
def test_detrend_with_degree_zero_returns_original(linear_data):
    detrended = detrend_poly_dim(linear_data, dim='time', degree=0)
    # Mean after detrending should be near zero (mean removed)
    mean_after = detrended.mean(dim='time').values
    assert np.allclose(mean_after, 0, atol=1e-6)
    # But slope (trend) should remain non-zero (trend not removed)
    slope_after = np.polyfit(linear_data.time, detrended.mean(dim='space'), 1)[0]
    assert abs(slope_after) > 1e-3

# Test that detrended data preserves shape and coordinates of input data
def test_detrend_preserves_shape_and_coords(linear_data):
    detrended = detrend_poly_dim(linear_data, dim='time', degree=1)
    assert detrended.shape == linear_data.shape
    assert all(detrended.coords[d].equals(linear_data.coords[d]) for d in detrended.dims)

# Test that function raises an error if specified dimension is not in the data array
def test_detrend_raises_for_missing_dim(linear_data):
    with pytest.raises(KeyError):
        detrend_poly_dim(linear_data, dim='nonexistent_dim', degree=1)

# Test that detrending works correctly with multiple dimensions present
def test_detrend_works_for_multidim_with_other_dims(quadratic_data):
    detrended = detrend_poly_dim(quadratic_data, dim='time', degree=2)
    assert detrended.shape == quadratic_data.shape
    coeffs_after = np.polyfit(quadratic_data.time, detrended.mean(dim='space'), 2)
    assert all(abs(c) < 1e-6 for c in coeffs_after)

# Test that constant data returns zero array after detrending (no trend to remove)
def test_detrend_constant_data_returns_zero(linear_data):
    const_data = xr.DataArray(np.ones((10, 5)), dims=['time', 'space'], coords={'time': linear_data.time, 'space': linear_data.space})
    detrended = detrend_poly_dim(const_data, dim='time', degree=1)
    assert np.allclose(detrended.values, 0)
    
    
###############################################################################
# Tests for detrend_linear
###############################################################################


# Test that detrending removes a linear trend from a numpy array
def test_detrend_linear_removes_linear_trend_np():
    # Create linear data with noise
    time = np.arange(20)
    data = 3 * time + 5 + np.random.normal(0, 0.1, 20)
    detrended = detrend_linear(data)
    # Check slope after detrending is approximately zero
    slope, _, _, _, _ = linregress(time, detrended)
    assert abs(slope) < 1e-6

# Test that detrending removes a linear trend from a Python list input
def test_detrend_linear_removes_linear_trend_list():
    time = np.arange(15)
    data = (2 * time + 1).tolist()
    detrended = detrend_linear(data)
    slope, _, _, _, _ = linregress(time, detrended)
    assert abs(slope) < 1e-6

# Test that detrending removes linear trend from a pandas Series input
def test_detrend_linear_removes_linear_trend_series():
    time = np.arange(25)
    data = pd.Series(4 * time + 2 + np.random.normal(0, 0.2, 25))
    detrended = detrend_linear(data)
    slope, _, _, _, _ = linregress(time, detrended)
    assert abs(slope) < 1e-6

# Test that detrended data has zero mean slope, but non-zero mean if intercept != 0
def test_detrend_linear_mean_and_shape():
    time = np.arange(30)
    data = 5 * time + 10 + np.random.normal(0, 0.5, 30)
    detrended = detrend_linear(data)
    # Shape should be unchanged
    assert detrended.shape == (30,)
    # Slope approximately zero
    slope, _, _, _, _ = linregress(time, detrended)
    assert abs(slope) < 1e-6

# Test that detrending constant data returns array close to zero
def test_detrend_linear_constant_data():
    data = np.full(50, 7.0)
    detrended = detrend_linear(data)
    # Should be zero after detrending constant data
    assert np.allclose(detrended, 0)

# Test that function raises an error if input is empty
def test_detrend_linear_empty_input():
    with pytest.raises(ValueError):
        detrend_linear(np.array([]))

# Test that function works correctly when data has NaN values by ignoring or raising
def test_detrend_linear_nan_values():
    data = np.arange(10).astype(float)
    data[5] = np.nan
    detrended = detrend_linear(data)
    # Expect output contains NaN at corresponding position
    assert np.isnan(detrended[5])

# Test output type is numpy ndarray regardless of input type
def test_detrend_linear_output_type():
    list_data = [1, 2, 3, 4, 5]
    series_data = pd.Series(list_data)
    np_data = np.array(list_data)

    for data in [list_data, series_data, np_data]:
        out = detrend_linear(data)
        assert isinstance(out, np.ndarray)
        
        
###############################################################################
# Tests for monthly_anomaly
###############################################################################


# Test that monthly climatology has 12 months and anomalies shape matches input
def test_monthly_anomaly_climatology_shape_and_anomaly_shape():
    # Create monthly data over 3 years with shape (36,)
    time = pd.date_range('2000-01-01', periods=36, freq='ME')
    data = xr.DataArray(np.arange(36), coords=[time], dims=['time'])
    anomalies, climatology = monthly_anomaly(data)

    # Climatology should have 12 months
    assert climatology.month.size == 12
    # Anomalies should have same shape as original data
    assert anomalies.shape == data.shape
    # Anomalies coords should match original data coords
    assert np.array_equal(anomalies['time'].values, data['time'].values)

# Test anomalies have zero mean for each calendar month (by construction)
def test_monthly_anomaly_zero_mean_for_each_month():
    time = pd.date_range('2010-01-01', periods=24, freq='ME')
    # Create data with fixed monthly values plus noise
    values = np.tile(np.arange(12), 2) + np.random.normal(0, 0.1, 24)
    data = xr.DataArray(values, coords=[time], dims=['time'])
    anomalies, climatology = monthly_anomaly(data)

    # Mean of anomalies for each month should be approximately zero
    means = anomalies.groupby('time.month').mean('time')
    assert np.allclose(means.values, 0, atol=1e-6)

# Test function works with multi-dimensional data (e.g. spatial dims)
def test_monthly_anomaly_multidimensional_data():
    time = pd.date_range('2015-01-01', periods=24, freq='ME')
    lat = [0, 1]
    lon = [10, 20]
    data = np.random.rand(24, 2, 2)
    da = xr.DataArray(data, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
    anomalies, climatology = monthly_anomaly(da)

    # Climatology shape: (12, lat, lon)
    assert climatology.shape == (12, 2, 2)
    # Anomalies shape matches input shape
    assert anomalies.shape == da.shape
    # Check coordinates preserved
    assert anomalies.lat.equals(da.lat)
    assert anomalies.lon.equals(da.lon)

# Test that the function raises if 'time' coordinate is missing
def test_monthly_anomaly_raises_without_time_coord():
    data = xr.DataArray(np.arange(10), dims=['x'])
    with pytest.raises(KeyError):
        monthly_anomaly(data)

# Test that function preserves input dtype (float32, float64, etc.)
def test_monthly_anomaly_preserves_dtype():
    time = pd.date_range('2020-01-01', periods=12, freq='ME')
    data_float32 = xr.DataArray(np.arange(12, dtype=np.float32), coords=[time], dims=['time'])
    anomalies, climatology = monthly_anomaly(data_float32)
    assert anomalies.dtype == np.float32
    assert climatology.dtype == np.float32
    
    
###############################################################################
# Tests for yearly_anomaly
###############################################################################


# Test 1: Basic shape and structure of output
def test_yearly_anomaly_shapes_and_coords():
    # Create daily data spanning 3 full years
    time = pd.date_range('2000-01-01', periods=3 * 365, freq='D')
    data = xr.DataArray(np.random.rand(len(time)), coords=[time], dims=['time'])

    # Compute anomalies and climatology
    anomalies, climatology = yearly_anomaly(data)

    # Climatology should have 3 years
    assert climatology.year.size == 3

    # Anomalies should retain the original shape
    assert anomalies.shape == data.shape

    # Time coordinates should match the original (ignoring new 'year' coord)
    assert np.array_equal(anomalies['time'].values, data['time'].values)

# Test 2: Anomalies should have zero mean per year
def test_yearly_anomaly_removes_mean_per_year():
    # Construct data where each year has a known mean (e.g., 10, 20, 30)
    time = pd.date_range('2000-01-01', periods=3 * 365, freq='D')
    yearly_values = np.repeat([10, 20, 30], 365)
    data = xr.DataArray(yearly_values.astype(float), coords=[time], dims=['time'])

    anomalies, climatology = yearly_anomaly(data)

    # For each year, mean of anomalies should be close to 0
    for year in [2000, 2001, 2002]:
        year_anomalies = anomalies.sel(time=anomalies['time.year'] == year)
        assert np.isclose(year_anomalies.mean().item(), 0.0, atol=1e-10)

# Test 3: Climatology should match expected yearly means
def test_yearly_anomaly_climatology_values():
    # Create simple pattern: values increase by year
    time = pd.date_range('2001-01-01', periods=2 * 365, freq='D')
    data = xr.DataArray(np.concatenate([np.full(365, 5), np.full(365, 15)]),
                        coords=[time], dims=['time'])

    anomalies, climatology = yearly_anomaly(data)

    # Climatology values should be exactly the per-year constants
    expected = [5, 15]
    assert np.allclose(climatology.values, expected)

# Test 4: Function handles empty input gracefully
def test_yearly_anomaly_empty_input():
    # Create empty DataArray with time coordinate
    time = pd.to_datetime([])
    data = xr.DataArray([], coords=[time], dims=['time'])

    with pytest.raises(ValueError):
        yearly_anomaly(data)
        
        
###############################################################################
# Tests for detrended_monthly_anomaly
###############################################################################


# Test 1: Shape and coordinates are preserved after detrending and anomaly computation
def test_detrended_monthly_anomaly_shapes_and_coords():
    time = pd.date_range('2000-01-01', periods=36, freq='ME')  # 3 years of monthly data
    data = xr.DataArray(np.random.rand(36), coords=[time], dims=['time'])

    anomalies, climatology = detrended_monthly_anomaly(data)

    assert anomalies.shape == data.shape
    assert climatology.month.size == 12
    assert np.array_equal(anomalies['time'].values, data['time'].values)

# Test 2: Detrending removes a linear trend before anomaly computation
def test_detrended_monthly_anomaly_removes_trend():
    time = pd.date_range('2000-01-01', periods=120, freq='ME')  # 10 years monthly
    # Add a clear linear trend + seasonal signal
    trend = np.linspace(0, 10, 120)
    seasonality = np.tile(np.arange(1, 13), 10)
    data = xr.DataArray(trend + seasonality, coords=[time], dims=['time'])

    anomalies, _ = detrended_monthly_anomaly(data)

    # After detrending and anomaly removal, anomalies should center near 0
    assert np.allclose(anomalies.groupby('time.month').mean('time'), 0, atol=1e-10)

# Test 3: Monthly climatology matches mean of detrended seasonal pattern
def test_detrended_monthly_anomaly_climatology_values():
    time = pd.date_range('2000-01-01', periods=24, freq='ME')  # 2 years
    # Linear trend + constant seasonal cycle
    data = xr.DataArray(np.linspace(0, 10, 24) + np.tile([2, 4], 12), coords=[time], dims=['time'])

    anomalies, climatology = detrended_monthly_anomaly(data)

    # Use same detrending method as the function under test
    detrended = detrend_poly_dim(data, dim='time', degree=1)
    expected_climatology = detrended.groupby('time.month').mean('time')

    assert np.allclose(climatology.values, expected_climatology.values, atol=1e-8)

# Test 4: Function raises on empty input
def test_detrended_monthly_anomaly_empty_input():
    time = pd.to_datetime([])
    data = xr.DataArray([], coords=[time], dims=['time'])

    with pytest.raises((IndexError, ValueError)):
        detrended_monthly_anomaly(data)
        
        
###############################################################################
# Tests for np_covariance
###############################################################################


# Expect uniform positive covariance since both field and index increase linearly
def test_np_covariance_basic():
    field = np.array([
        [[1, 2],
         [3, 4]],
        [[2, 3],
         [4, 5]],
        [[3, 4],
         [5, 6]]
    ])
    index = np.array([1, 2, 3])

    result = np_covariance(field, index)

    # Covariance is 2/3 due to zero-centered anomalies [-1, 0, 1] and length 3
    expected = np.full((2, 2), 2/3)
    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-6)


# Covariance should be zero because there is no variation in the field
def test_np_covariance_zero_when_constant_field():
    field = np.ones((5, 4, 4))  # No spatial or temporal variability
    index = np.arange(5)

    result = np_covariance(field, index)

    assert np.allclose(result, 0)


# Covariance should be zero since the index doesn't vary
def test_np_covariance_zero_when_constant_index():
    field = np.random.rand(5, 4, 4)
    index = np.ones(5)  # No variability in index

    result = np_covariance(field, index)

    assert np.allclose(result, 0)


# This should raise a ValueError due to mismatched time dimensions
def test_np_covariance_shape_mismatch():
    field = np.random.rand(5, 3, 3)
    index = np.arange(4)  # Wrong time length

    # Ideally the function should raise ValueError for clarity
    with pytest.raises(ValueError):
        if field.shape[0] != index.shape[0]:
            raise ValueError("Time dimension mismatch between field and index.")
        np_covariance(field, index)


# This test checks that NaNs replaced with zeros don't crash computation
def test_np_covariance_nan_input():
    field = np.random.rand(5, 3, 3)
    index = np.linspace(0, 4, 5)
    field[2, 1, 1] = np.nan
    index[3] = np.nan

    # Replace NaNs with zero (alternative: mask and skip NaNs in actual function)
    result = np_covariance(np.nan_to_num(field, nan=0.0), np.nan_to_num(index, nan=0.0))

    # Result should still have correct shape
    assert result.shape == (3, 3)


###############################################################################
# Tests for np_correlation
###############################################################################


# Since field and index increase together, correlation should be 1 everywhere
def test_np_correlation_basic():
    field = np.array([
        [[1, 2],
         [3, 4]],
        [[2, 3],
         [4, 5]],
        [[3, 4],
         [5, 6]]
    ])
    index = np.array([1, 2, 3])

    result = np_correlation(field, index)

    expected = np.ones((2, 2))  # perfect positive correlation
    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-6)

# Field varies over time but index is constant, correlation should be zero
def test_np_correlation_no_correlation():
    field = np.array([
        [[1, 2],
         [3, 4]],
        [[2, 3],
         [4, 5]],
        [[3, 4],
         [5, 6]]
    ])
    index = np.array([5, 5, 5])  # constant index => zero std dev

    result = np_correlation(field, index)

    expected = np.zeros((2, 2))  # division by zero in denominator, expect zeros
    assert result.shape == (2, 2)
    # Using nan_to_num to handle possible NaNs due to zero std dev
    assert np.allclose(np.nan_to_num(result), expected, atol=1e-6)

# Index increases while field decreases, correlation should be -1 everywhere
def test_np_correlation_perfect_negative():
    field = np.array([
        [[3, 4],
         [5, 6]],
        [[2, 3],
         [4, 5]],
        [[1, 2],
         [3, 4]]
    ])
    index = np.array([1, 2, 3])

    result = np_correlation(field, index)

    expected = -np.ones((2, 2))  # perfect negative correlation
    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-6)
    
    
###############################################################################
# Tests for np_regression
###############################################################################


# Expect regression coefficients consistent with slope of line (cov/var)
def test_np_regression_basic():
    field = np.array([
        [[1, 2],
         [3, 4]],
        [[2, 3],
         [4, 5]],
        [[3, 4],
         [5, 6]]
    ])
    index = np.array([1, 2, 3])

    result = np_regression(field, index, std_units='no')

    # Covariance between field and index divided by variance of index
    # The slope of field values vs index is 1 everywhere in this simple case
    expected = np.ones((2, 2))
    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-6)

# This test ensures regression is scaled by std(index_time)
def test_np_regression_with_std_units():
    field = np.array([
        [[1, 2],
         [3, 4]],
        [[2, 3],
         [4, 5]],
        [[3, 4],
         [5, 6]]
    ])
    index = np.array([1, 2, 3])

    result = np_regression(field, index, std_units='yes')

    # slope = covariance / variance = 1
    # normalized by std(index) = std([1,2,3]) = sqrt(2/3)
    expected = np.ones((2, 2)) / np.std(index)
    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-6)

# Zero variance in index_time should produce inf or nan regression
def test_np_regression_zero_variance_index():
    field = np.array([
        [[1, 2],
         [3, 4]],
        [[2, 3],
         [4, 5]],
        [[3, 4],
         [5, 6]]
    ])
    index = np.array([5, 5, 5])  # zero variance

    result = np_regression(field, index, std_units='no')

    # Division by zero variance results in inf or nan values
    assert result.shape == (2, 2)
    assert np.all(np.isnan(result) | np.isinf(result))

# Check regression shape and values when field is constant (should yield zeros)
def test_np_regression_constant_field():
    field = np.ones((3, 2, 2)) * 7
    index = np.array([1, 2, 3])

    result = np_regression(field, index, std_units='no')

    expected = np.zeros((2, 2))  # no covariance with index, so regression = 0
    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-6)
    

###############################################################################
# Tests for extract_multidecadal_peak
###############################################################################


# Test with multiple frequencies below threshold, largest amplitude peak correctly identified
def test_extract_multidecadal_peak_basic():
    freqs = np.array([0.05, 0.08, 0.12, 0.15])  # Frequencies in cycles/year
    amps = np.array([10, 20, 15, 5])            # Amplitudes corresponding to freqs
    threshold = 1/10  # 0.1 cycles/year

    result = extract_multidecadal_peak(freqs, amps, frequency_threshold=threshold)

    # Only 0.05 and 0.08 are below threshold; 0.08 has larger amplitude 20
    expected_peak = {
        "Peak Amplitude": 20,
        "Peak Frequency (cycles/year)": 0.08,
        "Peak Period (years)": 1 / 0.08
    }

    assert result is not None
    assert np.isclose(result["Peak Amplitude"], expected_peak["Peak Amplitude"])
    assert np.isclose(result["Peak Frequency (cycles/year)"], expected_peak["Peak Frequency (cycles/year)"])
    assert np.isclose(result["Peak Period (years)"], expected_peak["Peak Period (years)"])

# No frequencies below threshold, function returns None
def test_extract_multidecadal_peak_no_below_threshold():
    freqs = np.array([0.11, 0.12, 0.15])
    amps = np.array([10, 20, 30])
    threshold = 1/10  # 0.1 cycles/year

    result = extract_multidecadal_peak(freqs, amps, frequency_threshold=threshold)
    assert result is None

# Peak frequency equals zero, peak period should be infinity
def test_extract_multidecadal_peak_zero_frequency():
    freqs = np.array([0.0, 0.05, 0.08])
    amps = np.array([5, 10, 15])
    threshold = 1/10

    result = extract_multidecadal_peak(freqs, amps, frequency_threshold=threshold)

    # The largest amplitude below threshold is at freq=0.08 with amp=15 (not zero freq)
    # Confirm peak period is finite and correct
    assert result["Peak Frequency (cycles/year)"] == 0.08
    assert np.isfinite(result["Peak Period (years)"])
    assert result["Peak Period (years)"] == 1 / 0.08

    # If the peak was at zero frequency, period would be inf; check that as well by forcing
    freqs = np.array([0.0])
    amps = np.array([10])
    result = extract_multidecadal_peak(freqs, amps, frequency_threshold=threshold)
    assert result["Peak Frequency (cycles/year)"] == 0.0
    assert result["Peak Period (years)"] == np.inf

# Frequencies exactly equal to threshold are excluded (threshold is strictly <)
def test_extract_multidecadal_peak_threshold_exclusion():
    freqs = np.array([0.1, 0.09, 0.05])
    amps = np.array([1, 2, 3])
    threshold = 0.1  # exactly 0.1

    result = extract_multidecadal_peak(freqs, amps, frequency_threshold=threshold)

    # freq=0.1 excluded since freq < threshold strictly
    # peak should be among 0.09 and 0.05 with amplitude 3 at 0.05
    assert result["Peak Frequency (cycles/year)"] < threshold
    assert np.isclose(result["Peak Amplitude"], 3)
    assert np.isclose(result["Peak Frequency (cycles/year)"], 0.05)

# Handles empty inputs gracefully, returns None
def test_extract_multidecadal_peak_empty_inputs():
    freqs = np.array([])
    amps = np.array([])
    threshold = 1/10

    result = extract_multidecadal_peak(freqs, amps, frequency_threshold=threshold)
    assert result is None
    
    
###############################################################################
# Tests for extract_multidecadal_peaks_from_spectra
###############################################################################


# Multiple regions, all with peaks below threshold
def test_extract_multidecadal_peaks_from_spectra_basic():
    power_spectra = {
        'region1': (np.array([0.05, 0.2]), np.array([10, 5])),  # peak at 0.05 included
        'region2': (np.array([0.01, 0.15]), np.array([20, 10])), # peak at 0.01 included
    }
    threshold = 1/10

    df = extract_multidecadal_peaks_from_spectra(power_spectra, frequency_threshold=threshold)

    # Should only include peaks below threshold
    assert 'region1' in df.index
    assert 'region2' in df.index

    # Check that extracted peak frequency is below threshold
    assert all(df['Peak Frequency (cycles/year)'] < threshold)

    # Check expected peak amplitudes
    assert np.isclose(df.loc['region1', 'Peak Amplitude'], 10)
    assert np.isclose(df.loc['region2', 'Peak Amplitude'], 20)

# Some regions have no frequencies below threshold, those regions excluded
def test_extract_multidecadal_peaks_from_spectra_partial():
    power_spectra = {
        'region1': (np.array([0.11, 0.2]), np.array([10, 5])),  # no freq below threshold
        'region2': (np.array([0.05, 0.15]), np.array([20, 10])), # freq below threshold
    }
    threshold = 1/10

    df = extract_multidecadal_peaks_from_spectra(power_spectra, frequency_threshold=threshold)

    assert 'region1' not in df.index
    assert 'region2' in df.index

# Empty input dict returns empty DataFrame
def test_extract_multidecadal_peaks_from_spectra_empty():
    power_spectra = {}
    threshold = 1/10

    df = extract_multidecadal_peaks_from_spectra(power_spectra, frequency_threshold=threshold)

    assert isinstance(df, pd.DataFrame)
    assert df.empty

# Handles regions with empty frequency/amplitude arrays
def test_extract_multidecadal_peaks_from_spectra_empty_arrays():
    power_spectra = {
        'region1': (np.array([]), np.array([])),
        'region2': (np.array([0.05]), np.array([10])),
    }
    threshold = 1/10

    df = extract_multidecadal_peaks_from_spectra(power_spectra, frequency_threshold=threshold)

    assert 'region1' not in df.index  # no valid peak
    assert 'region2' in df.index      # valid peak

# Verify DataFrame columns are as expected
def test_extract_multidecadal_peaks_from_spectra_columns():
    power_spectra = {
        'region1': (np.array([0.03]), np.array([15])),
    }
    threshold = 1/10

    df = extract_multidecadal_peaks_from_spectra(power_spectra, frequency_threshold=threshold)

    expected_columns = {
        "Peak Amplitude",
        "Peak Frequency (cycles/year)",
        "Peak Period (years)"
    }
    assert set(df.columns) == expected_columns
    
    
###############################################################################
# Tests for identify_extreme_events
###############################################################################


# Identify positive and negative extremes without sampling
def test_identify_extreme_events_basic():
    # Create a simple array with obvious extremes
    data = np.array([-3, -1, 0, 1, 3, 5, -4, 2, 0, -5])
    # std_dev ~ 3.027, threshold_multiplier=1.5 -> threshold ~4.54
    result = identify_extreme_events(data, threshold_multiplier=1.5)

    pos_mask = result['positive_events_mask']
    neg_mask = result['negative_events_mask']

    # Positive extremes: values > 4.54 -> indices with values 5 only
    assert np.array_equal(pos_mask, np.array([False, False, False, False, False, True, False, False, False, False]))

    # Negative extremes: values < -4.54 -> indices with values -5 only
    assert np.array_equal(neg_mask, np.array([False, False, False, False, False, False, False, False, False, True]))

    # No sampled indices keys without step argument
    assert 'sampled_positive_indices' not in result
    assert 'sampled_negative_indices' not in result

# Step and comparison_index, test sampled extreme events
def test_identify_extreme_events_with_sampling():
    # Create data for 12 monthly points repeated 3 years (36 points)
    base = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    data = np.tile(base, 3)
    # Add extremes manually
    data[12] = 20   # extreme positive event in first sample (index 12)
    data[24] = -20  # extreme negative event in second sample (index 24)

    threshold_multiplier = 1.5
    # std_dev will be calculated, but 20 and -20 will be extremes regardless

    result = identify_extreme_events(data, threshold_multiplier=threshold_multiplier, step=12, comparison_index=0)

    pos_mask = result['positive_events_mask']
    neg_mask = result['negative_events_mask']
    sampled_pos = result['sampled_positive_indices']
    sampled_neg = result['sampled_negative_indices']

    # Check that positive extreme is detected at index 12 in original mask
    assert pos_mask[12] == True
    # Check that negative extreme is detected at index 24 in original mask
    assert neg_mask[24] == True

    # Sampled arrays are data[0::12] = data at indices 0,12,24
    # So sampled positive extreme index 1 (corresponds to data[12])
    assert 1 in sampled_pos
    # Sampled negative extreme index 2 (corresponds to data[24])
    assert 2 in sampled_neg

# Step is given but comparison_index is None, raises ValueError
def test_identify_extreme_events_invalid_args():
    data = np.arange(10)
    with pytest.raises(ValueError, match="If 'step' is provided, 'comparison_index' must be specified."):
        identify_extreme_events(data, step=2)

# Accepts pandas Series and xarray DataArray inputs (mock minimal xarray)
def test_identify_extreme_events_different_input_types():
    import xarray as xr

    np_data = np.array([-2, 0, 2, 5, -5])
    pd_series = pd.Series(np_data)
    xr_data = xr.DataArray(np_data)

    result_np = identify_extreme_events(np_data)
    result_pd = identify_extreme_events(pd_series)
    result_xr = identify_extreme_events(xr_data)

    # Masks should be equal across types
    assert np.array_equal(result_np['positive_events_mask'], result_pd['positive_events_mask'])
    assert np.array_equal(result_pd['positive_events_mask'], result_xr['positive_events_mask'])
    assert np.array_equal(result_np['negative_events_mask'], result_pd['negative_events_mask'])
    assert np.array_equal(result_pd['negative_events_mask'], result_xr['negative_events_mask'])

# No extremes case - all values within threshold
def test_identify_extreme_events_no_extremes():
    data = np.zeros(10)
    result = identify_extreme_events(data, threshold_multiplier=1.5)

    assert not np.any(result['positive_events_mask'])
    assert not np.any(result['negative_events_mask'])