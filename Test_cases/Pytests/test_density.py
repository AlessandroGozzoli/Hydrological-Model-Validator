import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime

from Hydrological_model_validator.Processing.Density import (
    compute_density_bottom,
    compute_Bmost,
    compute_Bleast,
    filter_dense_water_masses,
    calc_density,
    compute_dense_water_volume,
)


@pytest.fixture
# Fixture providing simple 2D temperature and salinity arrays for testing
def simple_temp_sal_2d():
    temp = np.array([[10.0, 15.0], [20.0, 25.0]])
    sal = np.array([[35.0, 36.0], [34.5, 33.0]])
    return temp, sal

@pytest.fixture
# Fixture providing simple 3D temperature, salinity, and depth arrays for testing
def simple_temp_sal_3d():
    temp = np.array([
        [[10, 15], [20, 25]],
        [[11, 14], [19, 24]],
    ], dtype=float)
    sal = np.array([
        [[35, 36], [34.5, 33]],
        [[34.8, 35.5], [34.0, 33.2]],
    ], dtype=float)
    depths = np.array([0, 2])
    return temp, sal, depths

@pytest.fixture
# Fixture providing a simple 3D mask array for testing
def simple_mask3d():
    return np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ])

# Dummy 3D mask array used for patching in tests
dummy_mask3d = np.zeros((3, 4, 5), dtype=bool)
# Dummy filename fragments dictionary used for patching
dummy_fragments = {'ffrag1': 'ADR', 'ffrag2': '2000', 'ffrag3': '2000'}
# Density calculation method string used in tests
density_method = 'EOS'
# Dummy shape for temperature and salinity arrays used in patching
dummy_shape = (12, 3, 4, 5)
# Dummy temperature array filled with 10.0 for patching
dummy_temp = np.full(dummy_shape, 10.0)
# Dummy salinity array filled with 35.0 for patching
dummy_sal = np.full(dummy_shape, 35.0)
# Dummy density array composed of two concatenated arrays for patching
dummy_density = np.concatenate([
    np.full((6, 3, 4, 5), 1030.0),
    np.full((6, 3, 4, 5), 1025.0)
])

@pytest.fixture
# Fixture patching all external dependencies used in compute_dense_water_volume tests
def patch_all_dependencies():
    # Patch external functions and methods to control their outputs for isolated unit testing
    with patch('Hydrological_model_validator.Processing.Density.infer_years_from_path') as mock_infer_years, \
         patch('Hydrological_model_validator.Processing.utils.build_bfm_filename') as mock_build_filename, \
         patch('pathlib.Path.exists', return_value=True) as mock_exists, \
         patch('Hydrological_model_validator.Processing.file_io.read_nc_variable_from_gz_in_memory') as mock_read_nc, \
         patch('Hydrological_model_validator.Processing.utils.temp_threshold') as mock_temp_thresh, \
         patch('Hydrological_model_validator.Processing.utils.hal_threshold') as mock_hal_thresh, \
         patch('Hydrological_model_validator.Processing.Density.calc_density') as mock_calc_density:

        # Mock return values simulate expected outputs so tests can focus on target function logic
        mock_infer_years.return_value = (2000, 2000, [2000])
        mock_build_filename.return_value = "ADR200020002000.nc"
        # Return dummy temperature or salinity arrays based on requested variable name to mimic data reading
        mock_read_nc.side_effect = lambda file, var: dummy_temp if var == 'votemper' else dummy_sal
        # Return empty masks to simulate no threshold crossing during tests
        mock_temp_thresh.return_value = np.zeros(dummy_temp.shape, dtype=bool)
        mock_hal_thresh.return_value = np.zeros(dummy_temp.shape, dtype=bool)
        # Return dummy density array to provide predictable density data for testing downstream calculations
        mock_calc_density.return_value = dummy_density

        yield (
            mock_infer_years,
            mock_build_filename,
            mock_exists,
            mock_read_nc,
            mock_temp_thresh,
            mock_hal_thresh,
            mock_calc_density,
        )

################################################################################
# ---------- Tests for compute_density_bottom ----------
################################################################################


# Test the compute_density_bottom function using the EOS method, verifying output type, keys, shape, and expected density values.
def test_compute_density_bottom_eos_method(simple_temp_sal_2d, simple_mask3d):
    temp, sal = simple_temp_sal_2d
    
    # Compute bottom-most indices to focus density calculations on seafloor layers where bottom water properties matter
    Bmost = compute_Bmost(simple_mask3d)
    
    # Organize data by year for compatibility with expected input structure of compute_density_bottom
    temperature_data = {2020: [temp]}
    salinity_data = {2020: [sal]}
    
    # Call density calculation with EOS method to test a specific equation of state formulation
    result = compute_density_bottom(temperature_data, salinity_data, Bmost, method="EOS", dz=2.0)
    
    assert isinstance(result, dict)
    assert 2020 in result
    assert len(result[2020]) == 1
    
    density = result[2020][0]
    assert density.shape == temp.shape
    
    # Use simplified EOS formula to independently verify density output correctness for given temperature and salinity
    expected = 1025 * (1 - 0.0002 * (temp - 10) + 0.0008 * (sal - 35))
    np.testing.assert_allclose(density, expected, rtol=1e-5)

# Test compute_density_bottom using the EOS80 method, checking the result type and that density values are physically reasonable (>1000).
def test_compute_density_bottom_eos80_method(simple_temp_sal_2d, simple_mask3d):
    temp, sal = simple_temp_sal_2d
    
    # Focus on bottom-most valid data points for realistic bottom water density estimation
    Bmost = compute_Bmost(simple_mask3d)
    
    temperature_data = {2021: [temp]}
    salinity_data = {2021: [sal]}
    
    # Validate EOS80 method output to ensure the older but still common density formulation returns plausible values
    result = compute_density_bottom(temperature_data, salinity_data, Bmost, method="EOS80")
    
    assert isinstance(result, dict)
    density = result[2021][0]
    
    # Physical seawater density should always be above 1000 kg/m³; use this to sanity check results
    assert np.all(density > 1000)

# Test compute_density_bottom using the TEOS10 method, ensuring output shape correctness and density values above 1000.
def test_compute_density_bottom_teos10_method(simple_temp_sal_2d, simple_mask3d):
    temp, sal = simple_temp_sal_2d
    
    # Bottom layer indices needed because density depends on local conditions at ocean bottom
    Bmost = compute_Bmost(simple_mask3d)
    
    temperature_data = {2022: [temp]}
    salinity_data = {2022: [sal]}
    
    # TEOS10 is the modern standard for seawater density; test its correct integration and output shape
    result = compute_density_bottom(temperature_data, salinity_data, Bmost, method="TEOS10", dz=2.0)
    density = result[2022][0]
    
    assert density.shape == temp.shape
    
    # Check that computed densities fall within realistic ocean values to confirm calculation integrity
    assert np.all(density > 1000)

# Test compute_density_bottom raises a ValueError when called with an invalid method argument.
def test_compute_density_bottom_invalid_method(simple_temp_sal_2d, simple_mask3d):
    temp, sal = simple_temp_sal_2d
    
    # Bottom-most indices necessary for full function argument requirements
    Bmost = compute_Bmost(simple_mask3d)
    
    # Validate error handling to ensure robustness against unsupported method inputs
    with pytest.raises(ValueError):
        compute_density_bottom({2020: [temp]}, {2020: [sal]}, Bmost, method="INVALID")

def get_input_test_var():
    # Setup minimal valid inputs to modify
    year = 2000
    valid_temp = {year: [np.zeros((5, 5)) for _ in range(12)]}
    valid_sal = {year: [np.zeros((5, 5)) for _ in range(12)]}
    Bmost = np.ones((5, 5), dtype=int)
    return year, valid_temp, valid_sal, Bmost

# Tests for input validation
def test_compute_density_bottom_input_validation():
    year, valid_temp, valid_sal, Bmost = get_input_test_var()
    
    # 1. temperature_data not dict
    with pytest.raises(TypeError, match="temperature_data must be a dictionary"):
        compute_density_bottom("not a dict", valid_sal, Bmost, "EOS")

    # 2. salinity_data not dict
    with pytest.raises(TypeError, match="salinity_data must be a dictionary"):
        compute_density_bottom(valid_temp, "not a dict", Bmost, "EOS")

    # 3. Bmost not ndarray
    with pytest.raises(TypeError, match="Bmost must be a numpy.ndarray"):
        compute_density_bottom(valid_temp, valid_sal, "not an array", "EOS")

def test_compute_density_bottom_input_validation_pt2():
    year, valid_temp, valid_sal, Bmost = get_input_test_var()
    
    # 4. Bmost not 2D
    with pytest.raises(ValueError, match="Bmost must be a 2D array"):
        compute_density_bottom(valid_temp, valid_sal, np.ones((5, 5, 5)), "EOS")

    # 5. Bmost values less than 1
    Bmost_invalid = np.zeros((5, 5))
    with pytest.raises(ValueError, match="All values in Bmost must be >= 1"):
        compute_density_bottom(valid_temp, valid_sal, Bmost_invalid, "EOS")

    # 6. dz not positive
    with pytest.raises(ValueError, match="dz must be a positive float"):
        compute_density_bottom(valid_temp, valid_sal, Bmost, "EOS", dz=-1.0)

    # 7. temperature_data and salinity_data keys mismatch
    sal_wrong_keys = {1999: [np.zeros((5, 5)) for _ in range(12)]}
    with pytest.raises(ValueError, match="must have the same years as keys"):
        compute_density_bottom(valid_temp, sal_wrong_keys, Bmost, "EOS")

def test_compute_density_bottom_input_validation_pt3():
    year, valid_temp, valid_sal, Bmost = get_input_test_var()
    # 8. temperature or salinity data not list for a year
    temp_not_list = {year: "not a list"}
    with pytest.raises(TypeError, match="must be lists of arrays"):
        compute_density_bottom(temp_not_list, valid_sal, Bmost, "EOS")

    # 9. temperature and salinity lists length mismatch
    sal_wrong_length = {year: [np.zeros((5, 5)) for _ in range(11)]}
    with pytest.raises(ValueError, match="must have the same number of monthly arrays"):
        compute_density_bottom(valid_temp, sal_wrong_length, Bmost, "EOS")

    # 10. temperature or salinity monthly data not ndarray
    temp_not_array = {year: [np.zeros((5, 5)) for _ in range(11)] + ["not an array"]}
    sal_valid = {year: [np.zeros((5, 5)) for _ in range(12)]}
    with pytest.raises(TypeError, match="temperature and salinity must be numpy arrays"):
        compute_density_bottom(temp_not_array, sal_valid, Bmost, "EOS")

def test_compute_density_bottom_input_validation_pt4():
    year, valid_temp, valid_sal, Bmost = get_input_test_var()
    # 11. temperature and salinity monthly arrays shape mismatch
    temp_shape = {year: [np.zeros((5, 5)) for _ in range(11)] + [np.zeros((5, 4))]}
    sal_shape = {year: [np.zeros((5, 5)) for _ in range(12)]}
    with pytest.raises(ValueError, match="temperature and salinity arrays must have the same shape"):
        compute_density_bottom(temp_shape, sal_shape, Bmost, "EOS")

    # 12. temperature shape does not match Bmost shape
    temp_wrong_spatial = {year: [np.zeros((4, 4)) for _ in range(12)]}
    sal_wrong_spatial = {year: [np.zeros((4, 4)) for _ in range(12)]}  # must match temp shape to get past earlier check
    with pytest.raises(ValueError, match="temperature/salinity shape .* does not match Bmost shape"):
        compute_density_bottom(temp_wrong_spatial, sal_wrong_spatial, Bmost, "EOS")

    # 13. Unsupported method
    with pytest.raises(ValueError, match="Unsupported method"):
        compute_density_bottom(valid_temp, valid_sal, Bmost, "INVALID_METHOD")


################################################################################
# ---------- Tests for compute_Bmost ----------
################################################################################


# Test compute_Bmost with a simple 3D mask, verifying output equals sum over the vertical axis.
def test_compute_Bmost_simple_mask(simple_mask3d):
    # compute_Bmost should aggregate mask values vertically to get counts or sums per horizontal location
    result = compute_Bmost(simple_mask3d)
    
    # Expected result is the vertical sum of the mask, representing total valid layers per horizontal cell
    expected = np.sum(simple_mask3d, axis=0)
    np.testing.assert_array_equal(result, expected)

# Test compute_Bmost with an all-zero mask, expecting a zero 2D output.
def test_compute_Bmost_all_zeros():
    mask = np.zeros((3, 4, 5))
    
    # With no valid points in mask, output should reflect zero counts across horizontal grid
    result = compute_Bmost(mask)
    expected = np.zeros((4, 5))
    np.testing.assert_array_equal(result, expected)

# Test compute_Bmost with a single-layer mask, expecting the output to match that single layer.
def test_compute_Bmost_single_layer():
    mask = np.ones((1, 2, 2))
    
    # When only one vertical layer exists, output should directly reflect that single layer's values
    result = compute_Bmost(mask)
    expected = np.ones((2, 2))
    np.testing.assert_array_equal(result, expected)

# Test compute_Bmost with non-binary mask values, verifying output sums correctly over the vertical axis.
def test_compute_Bmost_nonbinary_values():
    mask = np.array([
        [[2, 3], [4, 5]],
        [[1, 1], [0, 1]]
    ])
    
    # Ensure function sums over vertical axis correctly even when mask contains values other than 0/1
    result = compute_Bmost(mask)
    expected = np.sum(mask, axis=0)
    np.testing.assert_array_equal(result, expected)

# Tests for validation input
def test_compute_Bmost_input_validation():

    # 1. mask3d not a numpy array (e.g., list)
    with pytest.raises(TypeError, match="mask3d must be a numpy.ndarray"):
        compute_Bmost([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])  # list, not ndarray

    # 2. mask3d is numpy array but not 3D (e.g., 2D)
    mask_2d = np.ones((10, 10))
    with pytest.raises(ValueError, match="mask3d must be a 3D array"):
        compute_Bmost(mask_2d)

    # 3. mask3d is numpy array but 1D
    mask_1d = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="mask3d must be a 3D array"):
        compute_Bmost(mask_1d)

def test_compute_Bmost_input_validation_pt2():
    # Valid input for comparison
    valid_mask3d = np.ones((5, 10, 15), dtype=int)
    
    # 4. mask3d is numpy array but 4D
    mask_4d = np.ones((2, 3, 4, 5))
    with pytest.raises(ValueError, match="mask3d must be a 3D array"):
        compute_Bmost(mask_4d)

    # 5. Valid input returns correct shape and sum
    result = compute_Bmost(valid_mask3d)
    assert isinstance(result, np.ndarray)
    assert result.shape == valid_mask3d.shape[1:]  # (rows, cols)
    assert np.all(result == 5)  # sum of all ones along depth axis is depth size

    # 6. Test with a binary mask with mixed 0 and 1
    mask3d_mixed = np.array([[[1, 0], [0, 1]],
                             [[0, 1], [1, 0]],
                             [[1, 1], [0, 0]]])
    expected = np.array([[2, 2],
                         [1, 1]])
    np.testing.assert_array_equal(compute_Bmost(mask3d_mixed), expected)


################################################################################
# ---------- Tests for compute_Bleast ----------
################################################################################


# Test compute_Bleast with a simple 3D mask, verifying output equals the top (first) layer of the mask.
def test_compute_Bleast_simple_mask(simple_mask3d):
    # compute_Bleast should return the top layer because it targets the shallowest valid data points
    result = compute_Bleast(simple_mask3d)
    
    # Expected output is the first vertical layer of the mask, representing the shallowest layer
    expected = simple_mask3d[0, :, :]
    np.testing.assert_array_equal(result, expected)

# Test compute_Bleast with an all-zero mask, expecting output to match the first layer (all zeros).
def test_compute_Bleast_all_zeros():
    mask = np.zeros((3, 4, 5))
    
    # When all layers are zero, the shallowest layer output should reflect this zero state
    result = compute_Bleast(mask)
    expected = mask[0, :, :]
    np.testing.assert_array_equal(result, expected)

# Test compute_Bleast with a single-layer mask, expecting output to equal that single layer.
def test_compute_Bleast_single_layer():
    mask = np.ones((1, 2, 2))
    
    # With only one vertical layer, the shallowest layer is the only layer available
    result = compute_Bleast(mask)
    expected = mask[0, :, :]
    np.testing.assert_array_equal(result, expected)

# Test compute_Bleast with non-binary mask values, verifying output equals the first layer regardless of values.
def test_compute_Bleast_nonbinary_values():
    mask = np.array([
        [[7, 8], [9, 10]],
        [[1, 1], [0, 1]]
    ])
    
    # The function should always return the top layer exactly, irrespective of mask value types or magnitude
    result = compute_Bleast(mask)
    expected = mask[0, :, :]
    np.testing.assert_array_equal(result, expected)

# Tests for validation input
def test_compute_Bleast_input_validation():

    # 1. mask3d not a numpy array (e.g., list)
    with pytest.raises(TypeError, match="mask3d must be a numpy.ndarray"):
        compute_Bleast([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])  # list instead of ndarray

    # 2. mask3d not 3D (2D array)
    mask_2d = np.ones((4, 5))
    with pytest.raises(ValueError, match="mask3d must be a 3D array"):
        compute_Bleast(mask_2d)

    # 3. mask3d empty in depth (depth=0)
    mask_empty_depth = np.ones((0, 4, 5))
    with pytest.raises(ValueError, match="mask3d must have at least one depth layer"):
        compute_Bleast(mask_empty_depth)

def test_compute_Bleast_input_validation_pt2():
    # Valid mask for comparison
    valid_mask3d = np.ones((3, 4, 5), dtype=int)
    
    # 4. Valid input returns first layer as 2D array with correct shape and values
    result = compute_Bleast(valid_mask3d)
    assert isinstance(result, np.ndarray)
    assert result.shape == (valid_mask3d.shape[1], valid_mask3d.shape[2])
    assert np.all(result == 1)

    # 5. Test with a known 3D array for exact output
    mask3d = np.array([[[1, 0],
                        [0, 1]],
                       [[0, 1],
                        [1, 0]]])
    expected = np.array([[1, 0],
                         [0, 1]])
    np.testing.assert_array_equal(compute_Bleast(mask3d), expected)
    

################################################################################
# ---------- Tests for filter_dense_water_masses ----------
################################################################################


# Test filtering dense water masses by threshold, checking correct NaN masking for values below threshold.
def test_filter_dense_water_masses_basic():
    density_data = {
        2000: [np.array([[1029.5, 1028], [1030, 1025]])],
        2001: [np.array([[1028, 1029.3], [1029.1, 1027]])]
    }
    
    # Filter out densities below threshold to isolate dense water masses (important for oceanographic analyses)
    filtered = filter_dense_water_masses(density_data, threshold=1029.2)
    
    arr0 = filtered[2000][0]
    # Confirm values below threshold are masked as NaN to exclude them from further dense water processing
    assert np.isnan(arr0[0, 1])
    # Confirm values above threshold remain unchanged (not NaN)
    assert not np.isnan(arr0[0, 0])
    assert not np.isnan(arr0[1, 0])
    
    arr1 = filtered[2001][0]
    assert np.isnan(arr1[0, 0])
    assert not np.isnan(arr1[0, 1])
    assert np.isnan(arr1[1, 0])  

# Test filtering when no values exceed the threshold, expecting all values to be NaN.
def test_filter_dense_water_masses_no_values_above_threshold():
    density_data = {2000: [np.array([[1028, 1027], [1025, 1026]])]}
    
    # When no densities surpass threshold, all values should be masked as NaN to indicate absence of dense water
    filtered = filter_dense_water_masses(density_data, threshold=1029.2)
    
    arr = filtered[2000][0]
    assert np.all(np.isnan(arr))

# Test filtering when all values exceed the threshold, expecting no NaNs in output.
def test_filter_dense_water_masses_all_values_above_threshold():
    density_data = {2000: [np.array([[1030, 1031], [1032, 1033]])]}
    
    # When all densities are above threshold, no masking occurs, so no NaNs expected in output
    filtered = filter_dense_water_masses(density_data, threshold=1029.2)
    
    arr = filtered[2000][0]
    assert np.all(~np.isnan(arr))

# Test filtering with empty input dictionary, expecting an empty dictionary as output.
def test_filter_dense_water_masses_empty_dict():
    # Handle edge case where input is empty; output should also be empty to maintain function consistency
    filtered = filter_dense_water_masses({})
    assert filtered == {}

# Tests for input validations
def test_filter_dense_water_masses():

    # 1. Input type validation: density_data must be dict
    with pytest.raises(TypeError, match="density_data must be a dictionary"):
        filter_dense_water_masses("not a dict")

    # 2. Year keys must be int
    bad_keys = {"2000": [np.ones((2, 2)) for _ in range(12)]}
    with pytest.raises(TypeError, match="Year keys must be integers"):
        filter_dense_water_masses(bad_keys)

    # 3. Monthly arrays must be list
    bad_monthly = {2000: "not a list"}
    with pytest.raises(TypeError, match="density data must be a list"):
        filter_dense_water_masses(bad_monthly)

def test_filter_dense_water_masses_pt2():
    # Valid input density data
    density_data = {
        2000: [np.array([[1029.3, 1028.9], [1029.5, 1027.0]]) for _ in range(12)],
        2001: [np.array([[1029.1, 1029.0], [1028.0, 1030.0]]) for _ in range(12)]
    }
    
    # 4. Each monthly array must be numpy.ndarray
    bad_array = {2000: [np.ones((2, 2)), "not an array"] + [np.ones((2, 2))]*10}
    with pytest.raises(TypeError, match="density data must be a numpy.ndarray"):
        filter_dense_water_masses(bad_array)

    # 5. Each array must be 2D
    bad_ndim = {2000: [np.ones((2, 2)), np.ones((2, 2, 2))] + [np.ones((2, 2))]*10}
    with pytest.raises(ValueError, match="density array must be 2D"):
        filter_dense_water_masses(bad_ndim)

    # 6. threshold must be numeric
    with pytest.raises(TypeError, match="threshold must be a numeric value"):
        filter_dense_water_masses(density_data, threshold="not a number")

    # 7. Correct filtering behavior: values >= threshold kept, others nan
    threshold = 1029.2
    filtered = filter_dense_water_masses(density_data, threshold=threshold)
    for year, monthly_arrays in filtered.items():
        for arr in monthly_arrays:
            assert arr.shape == (2, 2)
            assert np.all(np.isnan(arr[arr < threshold]))
            assert np.all(arr[arr >= threshold] >= threshold)

    # 8. Test example data output matches expected
    expected_2000 = np.array([[1029.3, np.nan], [1029.5, np.nan]])
    expected_2001 = np.array([[np.nan, np.nan], [np.nan, 1030.0]])
    np.testing.assert_array_equal(filtered[2000][0], expected_2000)
    np.testing.assert_array_equal(filtered[2001][0], expected_2001)
    

################################################################################
# ---------- Tests for calc_density ----------
################################################################################


# Test calc_density with EOS method, verifying calculated density matches expected formula on valid data.
def test_calc_density_eos(simple_temp_sal_3d):
    temp, sal, depths = simple_temp_sal_3d
    
    # Create mask to limit calculations to valid (non-NaN) temperature and salinity points
    valid_mask = ~np.isnan(temp) & ~np.isnan(sal)
    
    # Calculate density using EOS method to check correct implementation of simplified equation of state
    density = calc_density(temp, sal, depths, valid_mask, density_method="EOS")
    
    # Expected formula based on EOS simplification for seawater density at given temp and salinity
    expected = 1025 * (1 - 0.0002 * (temp[valid_mask] - 10) + 0.0008 * (sal[valid_mask] - 35))
    
    # Verify computed densities match expected theoretical values within tolerance
    np.testing.assert_allclose(density[valid_mask], expected, rtol=1e-5)

# Test calc_density with EOS80 method, ensuring all valid density values are above 1000.
def test_calc_density_eos80(simple_temp_sal_3d):
    temp, sal, depths = simple_temp_sal_3d
    valid_mask = ~np.isnan(temp) & ~np.isnan(sal)
    
    # EOS80 is a more detailed legacy standard; ensure outputs are physically plausible (>1000)
    density = calc_density(temp, sal, depths, valid_mask, density_method="EOS80")
    assert np.all(density[valid_mask] > 1000)

# Test calc_density with TEOS10 method, checking all valid density values exceed 1000.
def test_calc_density_teos10(simple_temp_sal_3d):
    temp, sal, depths = simple_temp_sal_3d
    valid_mask = ~np.isnan(temp) & ~np.isnan(sal)
    
    # TEOS10 is the modern standard for seawater density; confirm output densities are physically reasonable
    density = calc_density(temp, sal, depths, valid_mask, density_method="TEOS10")
    assert np.all(density[valid_mask] > 1000)

# Test calc_density raises ValueError when given an invalid density method.
def test_calc_density_invalid_method(simple_temp_sal_3d):
    temp, sal, depths = simple_temp_sal_3d
    valid_mask = ~np.isnan(temp) & ~np.isnan(sal)
    
    # Invalid method input should raise an error to prevent silent failures or incorrect computations
    with pytest.raises(ValueError):
        calc_density(temp, sal, depths, valid_mask, density_method="BADMETHOD")

# Test for input validations
def test_calc_density_input_validation():
    # Setup dummy valid arrays for normal use
    temp = np.ones((3, 2, 2))
    sal = np.ones((3, 2, 2))
    depths = np.array([0, 10, 20])

    # 1. temp_3d must be np.ndarray
    with pytest.raises(TypeError, match="temp_3d must be a numpy.ndarray"):
        calc_density("not an array", sal, depths, None, "EOS")

    # 2. sal_3d must be np.ndarray
    with pytest.raises(TypeError, match="sal_3d must be a numpy.ndarray"):
        calc_density(temp, "not an array", depths, None, "EOS")

    # 3. depths must be np.ndarray
    with pytest.raises(TypeError, match="depths must be a numpy.ndarray"):
        calc_density(temp, sal, "not an array", None, "EOS")

    # 4. temp_3d and sal_3d must have same shape
    sal_diff_shape = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="temp_3d and sal_3d must have the same shape"):
        calc_density(temp, sal_diff_shape, depths, None, "EOS")

    # 5. depths length must match temp_3d first dimension
    bad_depths = np.array([0, 10])
    with pytest.raises(ValueError, match="depths length must match the first dimension"):
        calc_density(temp, sal, bad_depths, None, "EOS")

    # 6. density_method must be one of allowed strings
    with pytest.raises(ValueError, match="Unsupported density method"):
        calc_density(temp, sal, depths, None, "invalid_method")


################################################################################
# ---------- Tests for compute_dense_water_volume ----------
################################################################################


# Test compute_dense_water_volume with empty mask and missing files, expecting an empty list result.
def test_compute_dense_water_volume_empty(tmp_path, mocker):
    year_dir = tmp_path / "output2000"
    year_dir.mkdir()
    
    # Provide an empty 3D mask indicating no water to process (edge case)
    mask3d = np.zeros((1, 1, 1), dtype=bool)
    
    # Dummy filename fragments to simulate input parameters
    filename_fragments = {'ffrag1': '', 'ffrag2': '', 'ffrag3': ''}
    
    # Patch the file reading function to simulate missing data files (FileNotFoundError)
    # This ensures the function handles missing inputs gracefully without crashing
    mocker.patch('Hydrological_model_validator.Processing.file_io.read_nc_variable_from_gz_in_memory', side_effect=FileNotFoundError)
    
    # Run compute_dense_water_volume with these conditions
    result = compute_dense_water_volume(
        IDIR=tmp_path,
        mask3d=mask3d,
        filename_fragments=filename_fragments,
        density_method='EOS',
        dens_threshold=1029.2,
    )
    
    # The function should return an empty list when no valid data is available
    assert isinstance(result, list)
    assert len(result) == 0

# Test that missing gz file causes compute_dense_water_volume to skip processing and return an empty list.
def test_missing_gz_file_skips_year(patch_all_dependencies):
    # Patch dependencies so the file presence check returns False, simulating missing files
    patch_all_dependencies[2].return_value = False
    
    # Run compute_dense_water_volume with dummy inputs under missing file condition
    volume_series = compute_dense_water_volume(
        IDIR="dummy_dir",
        mask3d=dummy_mask3d,
        filename_fragments=dummy_fragments,
        density_method=density_method
    )
    
    # The function should skip processing and return an empty list as no data could be read
    assert volume_series == []

# Tests for input validation
def get_input_test_var_dense():
    # Create minimal valid dicts for temperature and salinity
    # For one year '2000' with 12 monthly arrays, each array shape (Y, X)
    year = 2000
    shape = (5, 5)
    monthly_temps = [np.full(shape, 10.0) for _ in range(12)]
    monthly_sals = [np.full(shape, 35.0) for _ in range(12)]

    temperature_data = {year: monthly_temps}
    salinity_data = {year: monthly_sals}
    
    valid_dir = "/valid/path"  # or Path("/valid/path")
    valid_mask = np.zeros((10, 5, 5), dtype=bool)  # 3D boolean mask
    valid_fragments = {'ffrag1': 'a', 'ffrag2': 'b', 'ffrag3': 'c'}
    valid_method = "EOS80"

    # Bmost: 2D array of bottom layer indices, here just ones
    Bmost = np.ones(shape, dtype=int)

    return year, temperature_data, salinity_data, Bmost, valid_dir, valid_fragments, valid_mask, valid_method

def test_compute_dense_water_volume_input_validation():
    year, temperature_data, salinity_data, Bmost, valid_dir, valid_fragments, valid_mask, valid_method = get_input_test_var_dense()

    # IDIR must be str or Path
    with pytest.raises(TypeError, match="IDIR must be a string or Path object"):
        compute_dense_water_volume(123, valid_mask, valid_fragments, valid_method)

    # mask3d must be ndarray
    with pytest.raises(TypeError, match="mask3d must be a numpy ndarray"):
        compute_dense_water_volume(valid_dir, "not an ndarray", valid_fragments, valid_method)

    # mask3d must be boolean dtype
    bad_mask = np.zeros((10, 5, 5), dtype=int)
    with pytest.raises(ValueError, match="mask3d must be a boolean numpy array"):
        compute_dense_water_volume(valid_dir, bad_mask, valid_fragments, valid_method)

def test_compute_dense_water_volume_input_validation_pt2():
    year, temperature_data, salinity_data, Bmost, valid_dir, valid_fragments, valid_mask, valid_method = get_input_test_var_dense()
    
    # mask3d must be 3D
    bad_mask_2d = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="mask3d must be a 3D numpy array"):
        compute_dense_water_volume(valid_dir, bad_mask_2d, valid_fragments, valid_method)

    # filename_fragments must be dict
    with pytest.raises(TypeError, match="filename_fragments must be a dict"):
        compute_dense_water_volume(valid_dir, valid_mask, "not a dict", valid_method)

    # filename_fragments must have required keys
    missing_keys = {'ffrag1': 'a', 'ffrag3': 'c'}
    with pytest.raises(ValueError, match="filename_fragments is missing required keys"):
        compute_dense_water_volume(valid_dir, valid_mask, missing_keys, valid_method)

    # density_method must be one of allowed
    with pytest.raises(ValueError, match="density_method must be one of"):
        compute_dense_water_volume(valid_dir, valid_mask, valid_fragments, "INVALID")
        
def test_compute_dense_water_volume_input_validation_pt3():
    year, temperature_data, salinity_data, Bmost, valid_dir, valid_fragments, valid_mask, valid_method = get_input_test_var_dense()
    
    # dz, dx, dy must be positive numbers (except dens_threshold can be any number)
    for param, bad_val in [('dz', -1), ('dx', 0), ('dy', -0.1)]:
        kwargs = dict(IDIR=valid_dir, mask3d=valid_mask, filename_fragments=valid_fragments, density_method=valid_method)
        kwargs[param] = bad_val
        with pytest.raises(ValueError, match=f"{param} must be positive"):
            compute_dense_water_volume(**kwargs)

    # dz, dx, dy must be numeric types
    for param, bad_val in [('dz', "nope"), ('dx', None), ('dy', [1,2])]:
        kwargs = dict(IDIR=valid_dir, mask3d=valid_mask, filename_fragments=valid_fragments, density_method=valid_method)
        kwargs[param] = bad_val
        with pytest.raises(TypeError, match=f"{param} must be a number"):
            compute_dense_water_volume(**kwargs)

    # dens_threshold must be a number
    with pytest.raises(TypeError, match="dens_threshold must be a number"):
        compute_dense_water_volume(valid_dir, valid_mask, valid_fragments, valid_method, dens_threshold="high")

# Mock values
def make_synthetic_temp_sal(shape):
    # Simple constant arrays with shape (time, depth, Y, X)
    temp = np.full(shape, 10.0)  # degrees C
    sal = np.full(shape, 35.0)   # PSU
    return temp, sal

# Test for the main loop
def test_compute_dense_water_volume_core_logic():

    # Setup parameters
    IDIR = Path("/fake/dir")
    mask3d = np.zeros((5, 4, 4), dtype=bool)  # No masked cells
    filename_fragments = {'ffrag1': 'a', 'ffrag2': 'b', 'ffrag3': 'c'}
    density_method = "EOS80"
    dz, dx, dy = 2.0, 800.0, 800.0
    dens_threshold = 1029.2

    # Synthetic data shape (12 months, 5 depths, 4, 4)
    shape = (12, 5, 4, 4)
    synthetic_temp, synthetic_sal = make_synthetic_temp_sal(shape)

    # Patch everything used inside compute_dense_water_volume
    with patch('Hydrological_model_validator.Processing.Density.read_nc_variable_from_gz_in_memory') as mock_read_nc, \
         patch('Hydrological_model_validator.Processing.utils.temp_threshold') as mock_temp_threshold, \
         patch('Hydrological_model_validator.Processing.utils.hal_threshold') as mock_hal_threshold, \
         patch('Hydrological_model_validator.Processing.Density.calc_density') as mock_calc_density, \
         patch('Hydrological_model_validator.Processing.utils.infer_years_from_path') as mock_infer_years, \
         patch('Hydrological_model_validator.Processing.utils.build_bfm_filename') as mock_build_filename, \
         patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.exists', autospec=True) as mock_exists, \
         patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.is_dir', autospec=True) as mock_is_dir, \
         patch('pathlib.Path.iterdir') as mock_iterdir:

        # Mock file existence check logic
        def exists_side_effect(self, *args, **kwargs):
            fake_output_dir = IDIR / "output2000"
            return str(self).startswith(str(fake_output_dir)) or self == IDIR
        mock_exists.side_effect = exists_side_effect

        def is_dir_side_effect(self, *args, **kwargs):
            return self == Path("/fake/dir")
        mock_is_dir.side_effect = is_dir_side_effect

        # Mock one output directory
        mock_output_dir = MagicMock()
        mock_output_dir.name = "output2000"
        mock_output_dir.is_dir.return_value = True
        mock_iterdir.return_value = [mock_output_dir]

        # Mock filename construction
        mock_build_filename.return_value = "nonexistent_file.nc"

        # Mock years
        mock_infer_years.return_value = (2000, 2000, [2000])

        # Mock read_nc returning synthetic temp/sal
        mock_read_nc.side_effect = [synthetic_temp, synthetic_sal]

        # Mock threshold masks (no invalid data)
        mock_temp_threshold.return_value = np.zeros(shape, dtype=bool)
        mock_hal_threshold.return_value = np.zeros(shape, dtype=bool)

        # Mock density: top-left quarter is dense
        density = np.full(shape, dens_threshold - 0.5)
        density[:, :, :2, :2] = dens_threshold + 1.0
        mock_calc_density.return_value = density

        # Run the function
        results = compute_dense_water_volume(
            IDIR, mask3d, filename_fragments, density_method,
            dz=dz, dx=dx, dy=dy, dens_threshold=dens_threshold
        )

        # Assert result length
        assert len(results) == 12

        # Check correct dates and volumes
        cell_volume = dx * dy * dz
        expected_volume = 20 * cell_volume  # 5 depths * 4 cells (2x2)

        for i, record in enumerate(results):
            assert record['date'] == datetime(2000, i + 1, 1)
            assert abs(record['volume_m3'] - expected_volume) < 1e-6