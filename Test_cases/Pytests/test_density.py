import numpy as np
import pytest
from unittest.mock import patch

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

