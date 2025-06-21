import pytest
import numpy as np
import xarray as xr
import pandas as pd

from Hydrological_model_validator.Processing.utils import (  # replace 'your_module' with your actual module name
    find_key,
    extract_options,
    infer_years_from_path,
    build_bfm_filename,
    temp_threshold,
    hal_threshold,
    find_key_variable,
    _to_dataarray,
    convert_dataarrays_in_df
)

###############################################################################
# Tests for find_key
###############################################################################


# Tests that find_key returns the first matching key ignoring case from the possible keys list.
def test_find_key_basic():
    d = {'Temperature': 23, 'Salinity': 35}
    assert find_key(d, ['temp', 'sal']) == 'Temperature'

# Tests case-insensitive matching of find_key.
def test_find_key_case_insensitive():
    d = {'TEMPERATURE': 20, 'salinity': 35}
    assert find_key(d, ['temp']) == 'TEMPERATURE'

# Tests that find_key returns None when no keys match.
def test_find_key_no_match():
    d = {'Pressure': 1013}
    assert find_key(d, ['temp']) is None

# Tests that find_key raises ValueError on invalid input types.
def test_find_key_invalid_inputs():
    with pytest.raises(ValueError):
        find_key("not a dict", ['temp'])  # dictionary not dict

    with pytest.raises(ValueError):
        find_key({'a': 1}, 'not an iterable of strings')  # possible_keys invalid (string)


###############################################################################
# Tests for extract_options
###############################################################################


# Tests that extract_options correctly overrides defaults using user args with the specified prefix.
def test_extract_options_basic():
    defaults = {'color': 'blue', 'linewidth': 2}
    user_args = {'plot_color': 'red', 'linewidth': 3}
    
    # extract_options should detect keys starting with 'plot_' prefix and override defaults accordingly
    result = extract_options(user_args, defaults, prefix='plot_')
    
    # The 'color' should be overridden by 'plot_color' in user_args
    assert result['color'] == 'red'
    # The 'linewidth' should be overridden by the direct user_args key 'linewidth'
    assert result['linewidth'] == 3

# Tests extract_options behavior when no prefix is provided (direct key matching).
def test_extract_options_no_prefix():
    defaults = {'color': 'blue', 'linewidth': 2}
    user_args = {'color': 'green', 'linewidth': 4}
    
    # Without prefix, user_args keys directly override defaults
    result = extract_options(user_args, defaults)
    
    # Result should exactly match user_args overriding defaults
    assert result == {'color': 'green', 'linewidth': 4}

# Tests that keys with prefix override both prefixed and non-prefixed keys in defaults.
def test_extract_options_prefix_overrides():
    defaults = {'color': 'blue', 'linewidth': 2}
    user_args = {'color': 'green', 'plot_linewidth': 5}
    
    # Even though 'color' is in user_args, 'plot_linewidth' with prefix should override 'linewidth' default
    result = extract_options(user_args, defaults, prefix='plot_')
    
    # 'color' remains from user_args direct key
    # 'linewidth' is overridden by the prefixed user_args key 'plot_linewidth'
    assert result == {'color': 'green', 'linewidth': 5}

# Tests that extract_options raises ValueError on invalid argument types.
def test_extract_options_invalid_inputs():
    # Passing non-dict for user_args should raise ValueError
    with pytest.raises(ValueError):
        extract_options("not a dict", {}, "")
    # Passing non-dict for defaults should raise ValueError
    with pytest.raises(ValueError):
        extract_options({}, "not a dict", "")
    # Passing non-string for prefix should raise ValueError
    with pytest.raises(ValueError):
        extract_options({}, {}, 123)


###############################################################################
# Tests for infer_years_from_path
###############################################################################


# Tests infer_years_from_path correctly extracts years from filenames in a directory.
def test_infer_years_from_path_files(tmp_path):
    # Create files with years embedded in their filenames
    filenames = ["data_2000.nc", "data_2001.nc", "data_2002.nc"]
    for f in filenames:
        (tmp_path / f).write_text("dummy")

    # Call the function to infer years from file names using regex pattern
    ybeg, yend, years = infer_years_from_path(
        tmp_path,
        target_type='file',
        pattern=r'_(\d{4})\.nc$',
        debug=False
    )

    # The earliest year should be 2000
    assert ybeg == 2000
    # The latest year should be 2002
    assert yend == 2002
    # The full list of extracted years should be in ascending order
    assert years == [2000, 2001, 2002]

# Tests infer_years_from_path correctly extracts years from folder names in a directory.
def test_infer_years_from_path_folders(tmp_path):
    # Create folders with years embedded in their folder names
    for year in [1999, 2000, 2001]:
        (tmp_path / f"output_{year}").mkdir()

    # Call the function to infer years from folder names using regex pattern
    ybeg, yend, years = infer_years_from_path(
        tmp_path,
        target_type='folder',
        pattern=r'_(\d{4})$',
        debug=False
    )

    # The earliest year should be 1999
    assert ybeg == 1999
    # The latest year should be 2001
    assert yend == 2001
    # The full list of extracted years should be in ascending order
    assert years == [1999, 2000, 2001]

# Tests that infer_years_from_path raises ValueError if the directory does not exist.
def test_infer_years_from_path_invalid_dir():
    # Pass a non-existent directory path
    with pytest.raises(ValueError):
        infer_years_from_path("/nonexistent/directory", target_type='file')

# Tests that infer_years_from_path raises ValueError if no files/folders match the pattern.
def test_infer_years_from_path_no_matches(tmp_path):
    # Create a file that does NOT match the year pattern
    (tmp_path / "file.txt").write_text("dummy")

    # Expect ValueError since no files/folders match the regex pattern
    with pytest.raises(ValueError):
        infer_years_from_path(tmp_path, target_type='file', pattern=r'_(\d{4})\.nc$')


###############################################################################
# Tests for build_bfm_filename
###############################################################################


# Test basic filename construction with all fragments as strings
def test_build_bfm_filename_basic():
    frags = {'ffrag1': 'A', 'ffrag2': 'B', 'ffrag3': 'C'}
    fn = build_bfm_filename(2020, frags)
    assert fn == "ADR2020A2020B2020C.nc"

# Test that the filename correctly includes the year and starts with proper prefix
def test_build_bfm_filename_year_format():
    frags = {'ffrag1': '_frag1_', 'ffrag2': '_frag2_', 'ffrag3': '_frag3_'}
    fn = build_bfm_filename(1999, frags)
    assert "1999" in fn
    assert fn.startswith("ADR1999")

# Test behavior when some required fragment keys are missing (should raise KeyError)
def test_build_bfm_filename_missing_keys():
    frags = {'ffrag1': 'X', 'ffrag3': 'Z'}
    with pytest.raises(KeyError):
        build_bfm_filename(2020, frags)

# Test filename construction when fragment values are non-string (integers)
def test_build_bfm_filename_non_string_fragments():
    frags = {'ffrag1': 1, 'ffrag2': 2, 'ffrag3': 3}
    fn = build_bfm_filename(2021, frags)
    assert fn == "ADR20211" + "20212" + "20213.nc"


###############################################################################
# Tests for temp_threshold
###############################################################################


# Test temp_threshold with a mix of valid and invalid temperature values for shallow and deep masks
def test_temp_threshold_valid_and_invalid():
    data = np.array([[10, 4], [36, 20]])
    mask_shallow = np.array([[True, True], [False, False]])
    mask_deep = np.array([[False, False], [True, True]])
    result = temp_threshold(data, mask_shallow, mask_deep)
    expected = np.array([[False, True], [True, False]])
    assert np.array_equal(result, expected)

# Test temp_threshold where all data points satisfy the temperature thresholds (all valid)
def test_temp_threshold_all_valid():
    data = np.array([[6, 10], [9, 15]])
    mask_shallow = np.array([[True, True], [False, False]])
    mask_deep = np.array([[False, False], [True, True]])
    result = temp_threshold(data, mask_shallow, mask_deep)
    assert not result.any()

# Test temp_threshold where all data points fail the temperature thresholds (all invalid)
def test_temp_threshold_all_invalid():
    data = np.array([[4, 36], [7, 26]])
    mask_shallow = np.array([[True, True], [False, False]])
    mask_deep = np.array([[False, False], [True, True]])
    result = temp_threshold(data, mask_shallow, mask_deep)
    assert result.all()

# Test temp_threshold behavior when no points are masked as shallow or deep (should return all False)
def test_temp_threshold_no_masked_points():
    data = np.array([[4, 36], [7, 26]])
    mask_shallow = np.array([[False, False], [False, False]])
    mask_deep = np.array([[False, False], [False, False]])
    result = temp_threshold(data, mask_shallow, mask_deep)
    assert not result.any()


###############################################################################
# Tests for hal_threshold
###############################################################################


# Test hal_threshold with a mix of valid and invalid halogen values for shallow and deep masks
def test_hal_threshold_valid_and_invalid():
    data = np.array([[30, 24], [37, 39]])
    mask_shallow = np.array([[True, True], [False, False]])
    mask_deep = np.array([[False, False], [True, True]])
    result = hal_threshold(data, mask_shallow, mask_deep)
    expected = np.array([[False, True], [False, False]])
    assert np.array_equal(result, expected)

# Test hal_threshold where all data points satisfy the halogen thresholds (all valid)
def test_hal_threshold_all_valid():
    data = np.array([[26, 35], [37, 38]])
    mask_shallow = np.array([[True, True], [False, False]])
    mask_deep = np.array([[False, False], [True, True]])
    result = hal_threshold(data, mask_shallow, mask_deep)
    assert not result.any()

# Test hal_threshold where all data points fail the halogen thresholds (all invalid)
def test_hal_threshold_all_invalid():
    data = np.array([[24, 41], [34, 35]])
    mask_shallow = np.array([[True, True], [False, False]])
    mask_deep = np.array([[False, False], [True, True]])
    result = hal_threshold(data, mask_shallow, mask_deep)
    assert result.all()

# Test hal_threshold behavior when no points are masked as shallow or deep (should return all False)
def test_hal_threshold_no_masked_points():
    data = np.array([[24, 41], [35, 36]])
    mask_shallow = np.array([[False, False], [False, False]])
    mask_deep = np.array([[False, False], [False, False]])
    result = hal_threshold(data, mask_shallow, mask_deep)
    assert not result.any()


###############################################################################
# Tests for find_key_variable
###############################################################################


# Test that find_key_variable returns the first matching candidate variable found in vars_
def test_find_key_variable_variable_found():
    vars_ = ['temp', 'salinity', 'oxygen']
    candidates = ['salinity', 'temperature']
    assert find_key_variable(vars_, candidates) == 'salinity'

# Test that find_key_variable raises KeyError when no candidates are found in vars_
def test_find_key_variable_not_found():
    vars_ = ['temp', 'oxygen']
    candidates = ['salinity', 'temperature']
    with pytest.raises(KeyError):
        find_key_variable(vars_, candidates)

# Test that find_key_variable raises KeyError if vars_ list is empty
def test_find_key_variable_empty_vars():
    vars_ = []
    candidates = ['salinity']
    with pytest.raises(KeyError):
        find_key_variable(vars_, candidates)

# Test that find_key_variable returns the first candidate found in the order of the candidates list
def test_find_key_variable_multiple_candidates():
    vars_ = ['salinity', 'temperature', 'oxygen']
    candidates = ['temperature', 'salinity']
    # should return 'temperature' as it appears first in candidates
    assert find_key_variable(vars_, candidates) == 'temperature'


###############################################################################
# Tests for _to_dataarray
###############################################################################


# Helper to create dummy DataArray with dims time, lat, lon
def create_reference_da(has_time=True):
    if has_time:
        times = pd.date_range("2000-01-01", periods=2)
        data = np.random.rand(2, 3, 3)
        return xr.DataArray(data, coords=[times, np.arange(3), np.arange(3)], dims=["time", "lat", "lon"])
    else:
        data = np.random.rand(3, 3)
        return xr.DataArray(data, coords=[np.arange(3), np.arange(3)], dims=["lat", "lon"])


# Test that input already an xarray.DataArray returns itself unchanged.
def test_val_is_dataarray():
    val = xr.DataArray(np.ones((3,3)), dims=["lat", "lon"])
    ref = create_reference_da()
    result = _to_dataarray(val, ref)
    assert isinstance(result, xr.DataArray)
    # Returning the same object saves unnecessary copying and preserves metadata.
    assert result is val


# Test scalar input converts to DataArray matching reference without time dimension.
def test_scalar_val_with_time_dim():
    val = 5
    ref = create_reference_da(has_time=True)
    result = _to_dataarray(val, ref)
    assert isinstance(result, xr.DataArray)
    # When scalar is passed, produce a DataArray matching spatial dims of reference,
    # but drop time because scalar cannot meaningfully broadcast over time dimension.
    assert 'time' not in result.dims
    assert result.shape == (3, 3)
    # All values should equal the scalar, indicating correct filling/broadcasting.
    assert (result.values == val).all()


# Test scalar input converts to DataArray matching reference without time dimension when no time in reference.
def test_scalar_val_without_time_dim():
    val = 7
    ref = create_reference_da(has_time=False)
    result = _to_dataarray(val, ref)
    assert isinstance(result, xr.DataArray)
    # Should produce DataArray with the same shape and dims as the reference.
    assert result.shape == (3, 3)
    # Values filled with the scalar value.
    assert (result.values == val).all()


# Test scalar input fills DataArray matching reference with extra non-time dimensions.
def test_reference_with_extra_dims():
    # Reference has additional dims beyond lat, lon, e.g. extra_dim.
    data = np.random.rand(3, 3, 2)
    ref = xr.DataArray(data, dims=["lat", "lon", "extra_dim"])
    val = 10
    result = _to_dataarray(val, ref)
    # Check that all dims in reference are preserved to maintain compatibility.
    assert set(result.dims) == set(ref.dims)
    assert result.shape == ref.shape
    # All values filled with scalar val to support consistent broadcasting.
    assert (result.values == val).all()


# Test non-DataArray array-like input converts to DataArray by broadcasting to reference shape.
def test_val_is_array_like_but_not_dataarray():
    val = np.array([1, 2, 3])
    ref = create_reference_da(has_time=False)
    with pytest.raises(ValueError, match="Expected scalar or DataArray"):
        _to_dataarray(val, ref)
        

###############################################################################
# Tests for find_key
###############################################################################


# Verify DataFrame without any DataArray cells remains unchanged after conversion.
def test_convert_no_dataarray_cells_remain_unchanged():
    df_plain = pd.DataFrame({"A": [1, 2], "B": ["foo", "bar"]})
    result = convert_dataarrays_in_df(df_plain)
    # No conversion should occur; result must be identical to input
    pd.testing.assert_frame_equal(result, df_plain)


# Verify that DataArray cells are converted properly to dicts with correct keys and values.
def test_convert_single_dataarray_cell_to_dict():
    da = xr.DataArray(np.array([[1, 2], [3, 4]]), dims=("x", "y"), coords={"x": [0, 1], "y": [10, 20]})
    df = pd.DataFrame({"A": [da, 2], "B": ["foo", "bar"]})
    result = convert_dataarrays_in_df(df)

    converted = result.loc[0, "A"]
    # Check that the cell is converted to a dict with required keys
    assert isinstance(converted, dict), "DataArray cell should convert to dict"
    assert set(converted.keys()) == {"dims", "coords", "data"}

    # Confirm dims match
    assert converted["dims"] == ("x", "y")
    # Confirm coords converted properly to lists
    assert converted["coords"] == {"x": [0, 1], "y": [10, 20]}
    # Confirm data values converted properly
    assert converted["data"] == [[1, 2], [3, 4]]

    # Confirm other non-DataArray cells remain unchanged
    assert result.loc[1, "A"] == 2
    assert result.loc[0, "B"] == "foo"


# Verify that DataFrames with all DataArray cells convert every cell properly.
def test_convert_all_dataarray_cells():
    da = xr.DataArray(np.array([[1, 2], [3, 4]]), dims=("x", "y"), coords={"x": [0, 1], "y": [10, 20]})
    df = pd.DataFrame({"A": [da, da], "B": [da, da]})
    result = convert_dataarrays_in_df(df)

    for idx in df.index:
        for col in df.columns:
            cell = result.loc[idx, col]
            # Each cell must be a dict with dims, coords, data keys
            assert isinstance(cell, dict)
            assert "dims" in cell and "coords" in cell and "data" in cell


# Verify that an empty DataFrame returns an empty DataFrame without error.
def test_convert_empty_dataframe():
    df = pd.DataFrame()
    result = convert_dataarrays_in_df(df)
    # Result must be empty and equal to input
    pd.testing.assert_frame_equal(result, df)


# Verify mixed types including None, numeric, strings remain unchanged; only DataArrays convert.
def test_convert_mixed_types_only_dataarray_converted():
    da = xr.DataArray(np.array([[1]]), dims=("x", "y"), coords={"x": [0], "y": [0]})
    df = pd.DataFrame({"A": [da, None], "B": [5.5, "text"]})
    result = convert_dataarrays_in_df(df)

    # DataArray cell converted
    assert isinstance(result.loc[0, "A"], dict)
    # None remains None
    assert result.loc[1, "A"] is None
    # Other types unchanged
    assert result.loc[0, "B"] == 5.5
    assert result.loc[1, "B"] == "text"