import numpy as np
import pandas as pd
import pytest

from Hydrological_model_validator.Processing.data_alignment import (
    get_valid_mask,
    get_valid_mask_pandas,
    align_pandas_series,
    align_numpy_arrays,
    get_common_series_by_year,
    get_common_series_by_year_month,
    extract_mod_sat_keys,
    gather_monthly_data_across_years,
    apply_3d_mask,
)

###############################################################################
###############################################################################
###############################################################################

@pytest.fixture
def mock_series_data():
    index = pd.date_range("2000-01-01", periods=5)
    mod = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=index)
    sat = pd.Series([1.1, 2.1, np.nan, 4.1, np.nan], index=index)
    return mod, sat

@pytest.fixture
def mock_year_dict():
    return {
        "model": {
            2000: pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2000-01-01", periods=3)),
            2001: pd.Series([4.0, np.nan, 6.0], index=pd.date_range("2001-01-01", periods=3)),
        },
        "satellite": {
            2000: pd.Series([1.1, 2.2, 3.3], index=pd.date_range("2000-01-01", periods=3)),
            2001: pd.Series([4.1, 5.1, np.nan], index=pd.date_range("2001-01-01", periods=3)),
        }
    }

@pytest.fixture
def mock_month_dict():
    return {
        "mod": {
            2000: [np.array([1.0, np.nan, 3.0])] * 12,
            2001: [np.array([np.nan, 5.0, 6.0])] * 12,
        },
        "sat": {
            2000: [np.array([1.0, 2.0, 3.0])] * 12,
            2001: [np.array([7.0, 5.0, np.nan])] * 12,
        }
    }

###############################################################################
# Tests for get_valid_mask
###############################################################################


# Test get_valid_mask with some NaNs, verifying mask correctly flags valid (non-NaN) pairs.
def test_get_valid_mask_basic():
    mod = np.array([1.0, 2.0, np.nan])
    sat = np.array([1.1, np.nan, 3.0])
    
    # Expected mask is True only where both mod and sat are non-NaN
    expected = np.array([True, False, False])
    
    # Run function to check if valid mask correctly identifies valid paired data
    result = get_valid_mask(mod, sat)
    np.testing.assert_array_equal(result, expected)

# Test get_valid_mask when all values are valid (no NaNs), expecting all True mask.
def test_get_valid_mask_all_valid():
    mod = np.array([1.0, 2.0, 3.0])
    sat = np.array([4.0, 5.0, 6.0])
    
    # Since no NaNs present, mask should mark all elements as valid
    expected = np.array([True, True, True])
    
    result = get_valid_mask(mod, sat)
    np.testing.assert_array_equal(result, expected)

# Test get_valid_mask when all values are NaN, expecting all False mask.
def test_get_valid_mask_all_nan():
    mod = np.array([np.nan, np.nan])
    sat = np.array([np.nan, np.nan])
    
    # All NaNs means no valid paired data points, mask should be all False
    expected = np.array([False, False])
    
    result = get_valid_mask(mod, sat)
    np.testing.assert_array_equal(result, expected)

# Test get_valid_mask raises TypeError or ValueError for invalid input types or mismatched shapes.
def test_get_valid_mask_type_shape_errors():
    # List input instead of np.array should raise TypeError to enforce correct types
    with pytest.raises(TypeError):
        get_valid_mask([1, 2, 3], np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        get_valid_mask(np.array([1, 2, 3]), [1, 2, 3])
    
    # Arrays with mismatched shapes should raise ValueError to prevent silent errors
    with pytest.raises(ValueError):
        get_valid_mask(np.array([1, 2]), np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        get_valid_mask(np.array([1, 2, 3]), np.array([1, 2]))


###############################################################################
# Tests for get_valid_mask_pandas
###############################################################################


# Test get_valid_mask_pandas with mixed valid and NaN values, verifying correct boolean mask with matching index.
def test_get_valid_mask_pandas(mock_series_data):
    mod, sat = mock_series_data
    
    # Expected mask True only where both mod and sat have valid (non-NaN) values
    # Also ensures the output mask preserves the original pandas index
    expected = pd.Series([True, False, False, True, False], index=mod.index)
    
    result = get_valid_mask_pandas(mod, sat)
    pd.testing.assert_series_equal(result, expected)

# Test get_valid_mask_pandas when all values are valid (no NaNs), expecting all True mask.
def test_get_valid_mask_pandas_all_valid():
    idx = pd.date_range("2020-01-01", periods=3)
    mod = pd.Series([1, 2, 3], index=idx)
    sat = pd.Series([4, 5, 6], index=idx)
    
    # With no NaNs present, mask should mark all entries as valid (True)
    expected = pd.Series([True, True, True], index=idx)
    
    result = get_valid_mask_pandas(mod, sat)
    pd.testing.assert_series_equal(result, expected)

# Test get_valid_mask_pandas when all values are NaN, expecting all False mask.
def test_get_valid_mask_pandas_all_nan():
    idx = pd.date_range("2020-01-01", periods=3)
    mod = pd.Series([np.nan, np.nan, np.nan], index=idx)
    sat = pd.Series([np.nan, np.nan, np.nan], index=idx)
    
    # When all values are NaN, no valid data exists, mask should be all False
    expected = pd.Series([False, False, False], index=idx)
    
    result = get_valid_mask_pandas(mod, sat)
    pd.testing.assert_series_equal(result, expected)

# Test get_valid_mask_pandas raises TypeError for invalid input types.
def test_get_valid_mask_pandas_type_errors():
    # Input types must be pandas Series, otherwise raise TypeError to ensure proper usage
    with pytest.raises(TypeError):
        get_valid_mask_pandas(np.array([1, 2]), pd.Series([1, 2]))
    with pytest.raises(TypeError):
        get_valid_mask_pandas(pd.Series([1, 2]), np.array([1, 2]))
    with pytest.raises(TypeError):
        get_valid_mask_pandas("invalid", pd.Series([1, 2]))
    with pytest.raises(TypeError):
        get_valid_mask_pandas(pd.Series([1, 2]), None)


###############################################################################
# Tests for align_pandas_series
###############################################################################


# Test align_pandas_series with overlapping indices, ensuring values are correctly aligned.
def test_align_pandas_series(mock_series_data):
    mod, sat = mock_series_data
    
    # Align the two series to their common timestamps so they can be directly compared
    mod_vals, sat_vals = align_pandas_series(mod, sat)
    
    # Confirm values correspond correctly after alignment to the intersection of indices
    assert np.allclose(mod_vals, [1.0, 4.0])
    assert np.allclose(sat_vals, [1.1, 4.1])

# Test align_pandas_series with no overlapping indices, expecting empty aligned arrays.
def test_align_pandas_series_no_overlap():
    idx1 = pd.date_range("2020-01-01", periods=3)
    idx2 = pd.date_range("2021-01-01", periods=3)
    mod = pd.Series([1, 2, 3], index=idx1)
    sat = pd.Series([4, 5, 6], index=idx2)
    
    # No common index means no data points to align, so output arrays should be empty
    mod_vals, sat_vals = align_pandas_series(mod, sat)
    assert mod_vals.size == 0
    assert sat_vals.size == 0

# Test align_pandas_series when all values are NaN, expecting empty arrays after alignment.
def test_align_pandas_series_all_nan():
    idx = pd.date_range("2020-01-01", periods=3)
    mod = pd.Series([np.nan, np.nan, np.nan], index=idx)
    sat = pd.Series([np.nan, np.nan, np.nan], index=idx)
    
    # Although indices overlap, all values are invalid, so filtered output arrays should be empty
    mod_vals, sat_vals = align_pandas_series(mod, sat)
    assert mod_vals.size == 0
    assert sat_vals.size == 0

# Test align_pandas_series raises TypeError when input types are not pandas Series.
def test_align_pandas_series_type_error():
    # Inputs must be pandas Series for consistent index alignment; otherwise raise error
    with pytest.raises(TypeError):
        align_pandas_series(np.array([1,2]), pd.Series([1,2]))
    with pytest.raises(TypeError):
        align_pandas_series(pd.Series([1,2]), np.array([1,2]))


###############################################################################
# Tests for align_numpy_arrays
###############################################################################


# Test align_numpy_arrays filters out pairs where either array has NaN, returning only valid aligned values.
def test_align_numpy_arrays():
    mod = np.array([1.0, np.nan, 3.0])
    sat = np.array([1.1, 2.1, np.nan])
    
    # Align arrays by removing any pair where either value is NaN to ensure valid comparisons
    mod_aligned, sat_aligned = align_numpy_arrays(mod, sat)
    
    # Only the first elements (non-NaN pairs) should remain after filtering
    np.testing.assert_array_equal(mod_aligned, np.array([1.0]))
    np.testing.assert_array_equal(sat_aligned, np.array([1.1]))

# Test align_numpy_arrays when all values are valid, ensuring full arrays are returned unchanged.
def test_align_numpy_arrays_all_valid():
    mod = np.array([1, 2, 3])
    sat = np.array([4, 5, 6])
    
    # No NaNs means all values are valid, so output arrays should match inputs exactly
    mod_aligned, sat_aligned = align_numpy_arrays(mod, sat)
    np.testing.assert_array_equal(mod_aligned, mod)
    np.testing.assert_array_equal(sat_aligned, sat)

# Test align_numpy_arrays when all values are NaN, expecting empty arrays after alignment.
def test_align_numpy_arrays_all_nan():
    mod = np.array([np.nan, np.nan])
    sat = np.array([np.nan, np.nan])
    
    # With all values NaN, filtering removes all elements leaving empty aligned arrays
    mod_aligned, sat_aligned = align_numpy_arrays(mod, sat)
    assert mod_aligned.size == 0
    assert sat_aligned.size == 0

# Test align_numpy_arrays raises ValueError when input arrays have mismatched shapes.
def test_align_numpy_arrays_shape_mismatch():
    mod = np.array([1, 2])
    sat = np.array([1, 2, 3])
    
    # Arrays must be the same shape to align elementwise; otherwise, raise an error
    with pytest.raises(ValueError):
        align_numpy_arrays(mod, sat)


###############################################################################
# Tests for get_common_series_by_year
###############################################################################


# Test get_common_series_by_year returns correct aligned non-NaN model and satellite series per year.
def test_get_common_series_by_year(mock_year_dict):
    # Get aligned model and satellite series per year with only overlapping valid (non-NaN) data
    output = get_common_series_by_year(mock_year_dict)
    
    # Expect output for each year to include aligned arrays of equal shape
    assert len(output) == 2
    for year, mod, sat in output:
        # Ensure model and satellite data have same length after alignment
        assert mod.shape == sat.shape
        
        # Verify that aligned data contains no NaNs to guarantee valid comparisons
        assert not np.isnan(mod).any()
        assert not np.isnan(sat).any()

# Test get_common_series_by_year raises ValueError when input dictionary is empty.
def test_get_common_series_by_year_empty_dict():
    # Function should reject empty input, as there’s no data to align or compare
    with pytest.raises(ValueError):
        get_common_series_by_year({})

# Test get_common_series_by_year returns empty list when all data for a year is NaN.
def test_get_common_series_by_year_nan_only():
    data = {
        "model": {
            2000: pd.Series([np.nan, np.nan], index=pd.date_range("2000-01-01", periods=2))
        },
        "satellite": {
            2000: pd.Series([np.nan, np.nan], index=pd.date_range("2000-01-01", periods=2))
        }
    }
    # When all data points are NaN for a year, no valid aligned data should be returned
    output = get_common_series_by_year(data)
    
    # Expect empty output list as no valid comparisons can be made
    assert len(output) == 0  

# Test get_common_series_by_year raises TypeError when input data structure types are incorrect.
def test_get_common_series_by_year_type_error():
    # Input must be dicts of pandas Series per year, reject invalid types like empty lists
    with pytest.raises(TypeError):
        get_common_series_by_year({"model": [], "satellite": []})


###############################################################################
# Tests for get_common_series_by_year_month
###############################################################################


# Test get_common_series_by_year_month returns aligned, non-NaN model and satellite arrays for each year-month.
def test_get_common_series_by_year_month(mock_month_dict):
    # Retrieve aligned data per year and month with only valid (non-NaN) values
    output = get_common_series_by_year_month(mock_month_dict)
    
    # Expect 24 year-month tuples (e.g., 2 years × 12 months)
    assert len(output) == 24
    
    for year, month, mod_vals, sat_vals in output:
        # Confirm year and month are integers (valid date components)
        assert isinstance(year, int)
        assert isinstance(month, int)
        
        # Model and satellite arrays should be the same shape after alignment
        assert mod_vals.shape == sat_vals.shape
        
        # Both arrays should contain no NaNs to ensure valid comparisons
        assert not np.isnan(mod_vals).any()
        assert not np.isnan(sat_vals).any()

# Test get_common_series_by_year_month raises ValueError if input dictionary is empty.
def test_get_common_series_by_year_month_empty_dict():
    # Function should raise error if no input data is provided
    with pytest.raises(ValueError):
        get_common_series_by_year_month({})

# Test get_common_series_by_year_month raises ValueError if input data structure is invalid.
def test_get_common_series_by_year_month_invalid_structure():
    # Satellite data for a month is not a numpy array, should raise ValueError
    d = {"mod": {2000: [np.array([1])]}, "sat": {2000: [1]}}  
    with pytest.raises(ValueError):
        get_common_series_by_year_month(d)

# Test get_common_series_by_year_month raises TypeError if input types are incorrect.
def test_get_common_series_by_year_month_type_error():
    # Input dict values should be dicts/lists of arrays, not None or other types
    with pytest.raises(TypeError):
        get_common_series_by_year_month({"mod": None, "sat": None})


###############################################################################
# Tests for extract_mod_sat_keys
###############################################################################


# Test extract_mod_sat_keys returns expected default keys when keys are 'model' and 'satellite'.
def test_extract_mod_sat_keys():
    d = {"model": {}, "satellite": {}}
    mod_key, sat_key = extract_mod_sat_keys(d)
    assert mod_key == "model"
    assert sat_key == "satellite"

# Test extract_mod_sat_keys correctly identifies alternate keys 'mod' and 'sat'.
def test_extract_mod_sat_keys_alternate_keys():
    d = {"mod": {}, "sat": {}}
    mod_key, sat_key = extract_mod_sat_keys(d)
    assert mod_key == "mod"
    assert sat_key == "sat"

# Test extract_mod_sat_keys raises ValueError if expected keys are missing.
def test_extract_mod_sat_keys_failure():
    with pytest.raises(ValueError):
        extract_mod_sat_keys({"foo": {}, "bar": {}})

# Test extract_mod_sat_keys raises TypeError if input is not a dictionary.
def test_extract_mod_sat_keys_type_error():
    with pytest.raises(TypeError):
        extract_mod_sat_keys(None)
    with pytest.raises(TypeError):
        extract_mod_sat_keys("not a dict")


###############################################################################
# Tests for gather_monthly_data_across_years
###############################################################################


# Test gather_monthly_data_across_years returns a 1D numpy array without NaNs for valid data.
def test_gather_monthly_data_across_years(mock_month_dict):
    # Call function for "mod" key and month index 0
    result = gather_monthly_data_across_years(mock_month_dict, "mod", 0)
    
    # Check result is a numpy array (expected output type)
    assert isinstance(result, np.ndarray)
    
    # Check result is 1D array (flattened monthly data across years)
    assert result.ndim == 1
    
    # Ensure no NaN values remain (function should filter out invalid/missing data)
    assert not np.isnan(result).any()

# Test gather_monthly_data_across_years raises ValueError if input dictionary is empty.
def test_gather_monthly_data_across_years_empty_dict():
    # Passing empty dictionary should raise error as there's no data to process
    with pytest.raises(ValueError):
        gather_monthly_data_across_years({}, "mod", 0)

# Test gather_monthly_data_across_years raises ValueError if the specified key is missing.
def test_gather_monthly_data_across_years_missing_key():
    # If the key is not present in the dictionary, the function cannot proceed
    with pytest.raises(ValueError):
        gather_monthly_data_across_years({"mod": {}}, "foo", 0)

# Test gather_monthly_data_across_years raises IndexError when month index is out of bounds.
def test_gather_monthly_data_across_years_invalid_month_index():
    d = {"mod": {2000: [np.array([1, 2])] * 12}}  # 12 months of data for year 2000
    
    # Month index 12 is invalid since valid indices are 0 to 11, should raise IndexError
    with pytest.raises(IndexError):
        gather_monthly_data_across_years(d, "mod", 12)

# Test gather_monthly_data_across_years returns empty array if all values for the month are NaN.
def test_gather_monthly_data_across_years_nan_values():
    d = {"mod": {2000: [np.array([np.nan, np.nan])] * 12}}  # All NaNs for every month
    
    result = gather_monthly_data_across_years(d, "mod", 0)
    
    # Since all data is NaN, result should be an empty array (filtered out all invalid data)
    assert result.size == 0


###############################################################################
# Tests for apply_3d_mask
###############################################################################

# Test apply_3d_mask masks values correctly using a mixed binary mask
def test_apply_3d_mask_valid():
    data = np.ones((2, 3, 3))
    mask = np.array([[[1, 0, 1],
                      [1, 1, 0],
                      [0, 1, 1]],
                     [[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]]])
    result = apply_3d_mask(data, mask)
    # mask zeros should produce NaNs
    assert np.isnan(result[0, 0, 1])
    assert np.isnan(result[0, 1, 2])
    assert np.isnan(result[0, 2, 0])
    assert np.isnan(result[1, 0, 0])
    # mask ones leave values unchanged
    assert result[0, 0, 0] == 1
    assert result[1, 0, 1] == 1

# Test apply_3d_mask with mask all ones leaves data unchanged
def test_apply_3d_mask_all_ones():
    data = np.full((1, 2, 2), 5)
    mask = np.ones((1, 2, 2))
    result = apply_3d_mask(data, mask)
    np.testing.assert_array_equal(result, data)

# Test apply_3d_mask with mask all zeros results in all NaNs
def test_apply_3d_mask_all_zeros():
    data = np.full((1, 2, 2), 5)
    mask = np.zeros((1, 2, 2))
    result = apply_3d_mask(data, mask)
    assert np.isnan(result).all()

# Test apply_3d_mask raises ValueError if shapes are incompatible
def test_apply_3d_mask_shape_error():
    data = np.ones((2, 3, 3))
    mask = np.ones((3, 3))
    with pytest.raises(ValueError):
        apply_3d_mask(data, mask)

# Test apply_3d_mask raises TypeError if mask is not an array
def test_apply_3d_mask_type_error():
    data = np.ones((1, 2, 2))
    mask = "not an array"
    with pytest.raises(TypeError):
        apply_3d_mask(data, mask)


###############################################################################