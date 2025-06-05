import pytest
import numpy as np
from unittest import mock
from pathlib import Path
from unittest.mock import MagicMock

from Hydrological_model_validator.Processing.BFM_data_reader import (
    extract_bottom_layer,
    extract_and_filter_benthic_data,
    process_year,
    read_benthic_parameter,
    read_bfm_chemical,
)

###############################################################################
###############################################################################
###############################################################################

# Create consistent dummy inputs to decouple tests from real data or I/O
@pytest.fixture
def dummy_mask3d():
    return np.ones((5, 4, 3), dtype=int)  # 3D mask (depth, lat, lon)

@pytest.fixture
def dummy_Bmost():
    return np.array([[1, 2, 3],
                     [4, 5, 1],
                     [2, 1, 3],
                     [5, 4, 2]], dtype=int)  # Bottom layer index (2D)

@pytest.fixture
def dummy_4d_data():
    return np.arange(2*5*4*3).reshape(2,5,4,3).astype(float)  # Time × Depth × Lat × Lon

@pytest.fixture
def dummy_filename_fragments():
    return {'ffrag1': 'a', 'ffrag2': 'b', 'ffrag3': 'c'}  # Fake filename metadata

@pytest.fixture
def dummy_years():
    return [2000, 2001]

class DummyPath(Path):
    # Dummy class to mock Path objects in directory listing
    def __new__(cls, name):
        return Path.__new__(cls, name)

    def is_dir(self):
        # We want these dummy paths to be considered directories
        return True

    def is_file(self):
        # Not files in this test
        return False

###############################################################################
###############################################################################
###############################################################################

# Validate logic that extracts benthic layers using bottom-most index map
def test_extract_bottom_layer_basic(dummy_4d_data, dummy_Bmost):
    bottom_layers = extract_bottom_layer(dummy_4d_data, dummy_Bmost)

    # We expect one output per time step
    assert isinstance(bottom_layers, list)
    assert len(bottom_layers) == dummy_4d_data.shape[0]

    # Each extracted layer must match horizontal dimensions
    for arr in bottom_layers:
        assert arr.shape == dummy_Bmost.shape

    # Check values are pulled from correct depths based on Bmost (1-based indexing)
    assert np.allclose(bottom_layers[0][0,0], dummy_4d_data[0,0,0,0])
    assert np.allclose(bottom_layers[0][1,1], dummy_4d_data[0,4,1,1])
    
###############################################################################

###############################################################################

# Ensure temperature threshold logic is invoked for 'votemper'
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.temp_threshold')
def test_extract_and_filter_benthic_data_with_temp_threshold(mock_temp_thresh, dummy_4d_data, dummy_Bmost):
    # Simulate that all values fail the threshold
    mock_temp_thresh.return_value = np.zeros((dummy_4d_data.shape[0], dummy_Bmost.shape[0], dummy_Bmost.shape[1]), dtype=bool)

    # Apply filtering
    result = extract_and_filter_benthic_data(dummy_4d_data, dummy_Bmost, variable_key='votemper')

    # Confirm temp threshold was applied and output dimensions are preserved
    mock_temp_thresh.assert_called_once()
    assert result.shape[0] == dummy_4d_data.shape[0]
    
###############################################################################

###############################################################################

# Ensure halocline threshold logic is invoked for 'vosaline'
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.hal_threshold')
def test_extract_and_filter_benthic_data_with_hal_threshold(mock_hal_thresh, dummy_4d_data, dummy_Bmost):
    # Simulate that all values fail the threshold
    mock_hal_thresh.return_value = np.zeros((dummy_4d_data.shape[0], dummy_Bmost.shape[0], dummy_Bmost.shape[1]), dtype=bool)

    # Apply filtering
    result = extract_and_filter_benthic_data(dummy_4d_data, dummy_Bmost, variable_key='vosaline')

    mock_hal_thresh.assert_called_once()
    assert result.shape[0] == dummy_4d_data.shape[0]
    
###############################################################################

###############################################################################

# Test successful file handling, extraction, filtering, and output for a single year
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.is_dir', autospec=True)
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.exists', autospec=True)
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.build_bfm_filename')
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.gzip.open')
@mock.patch('xarray.open_dataset')
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.extract_and_filter_benthic_data')
def test_process_year_success(
    mock_extract_filter, mock_xr_open, mock_gzip_open,
    mock_build_filename, mock_path_exists, mock_path_is_dir,
):
    year = 2000
    IDIR = Path("/fake/dir")
    variable_key = "votemper"
    filename = "file.nc"
    mock_build_filename.return_value = filename

    def exists_side_effect(self, *args, **kwargs):
        expected_file = IDIR / f"output{year}" / (filename + ".gz")
        if self == IDIR:
            return True
        if self == expected_file:
            return True
        return False
    mock_path_exists.side_effect = exists_side_effect

    # Mock is_dir to return True only for IDIR
    def is_dir_side_effect(self, *args, **kwargs):
        return self == IDIR
    mock_path_is_dir.side_effect = is_dir_side_effect

    # The rest stays the same ...
    mock_gzip_open.return_value.__enter__.return_value.read.return_value = b"dummybytes"

    dummy_ds = mock.MagicMock()
    dummy_ds.__enter__.return_value = dummy_ds
    dummy_ds.__contains__.side_effect = lambda key: key == variable_key
    dummy_ds.__getitem__.return_value.values = np.ones((12, 5, 4, 3))

    mock_xr_open.return_value = dummy_ds

    mock_extract_filter.return_value = np.ones((12, 4, 3))

    mask3d = np.ones((5, 4, 3))
    dummy_Bmost = np.array([[1, 2, 3],
                            [4, 5, 1],
                            [2, 1, 3],
                            [5, 4, 2]])

    result_year, result_data = process_year(
        year, IDIR, mask3d, dummy_Bmost,
        {'ffrag1': 'a', 'ffrag2': 'b', 'ffrag3': 'c'},
        variable_key
    )

    assert result_year == year
    np.testing.assert_array_equal(result_data, mock_extract_filter.return_value)

    
###############################################################################

###############################################################################

# Raise error if constructed filename cannot be resolved to a valid file
@mock.patch('Hydrological_model_validator.Processing.utils.build_bfm_filename')
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.exists', autospec=True)
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.is_dir', autospec=True)
def test_process_year_file_not_found(mock_is_dir, mock_exists, mock_build_filename, dummy_mask3d, dummy_Bmost):
    mock_build_filename.return_value = "nonexistent_file.nc"

    def exists_side_effect(self, *args, **kwargs):
        if self == Path("/fake/dir"):
            return True
        return False
    mock_exists.side_effect = exists_side_effect

    def is_dir_side_effect(self, *args, **kwargs):
        return self == Path("/fake/dir")
    mock_is_dir.side_effect = is_dir_side_effect

    with pytest.raises(FileNotFoundError):
        process_year(
            2000,
            Path("/fake/dir"),
            dummy_mask3d,    # injected by pytest fixture
            dummy_Bmost,     # injected by pytest fixture
            {'ffrag1': 'a', 'ffrag2': 'b', 'ffrag3': 'c'},
            "votemper"
        )
        
###############################################################################

###############################################################################

# Validate full yearly looped reading with realistic mocks
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.infer_years_from_path')
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.process_year')
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.exists', return_value=True, autospec=True)
def test_read_benthic_parameter_basic(mock_path_exists, mock_process_year, mock_infer_years,
                                      dummy_mask3d, dummy_Bmost, dummy_filename_fragments):
    IDIR = Path("/fake/dir")
    mock_infer_years.return_value = (2000, 2001, [2000, 2001])
    mock_process_year.side_effect = lambda y, *_: (y, [np.ones((4,3))]*12)  # 12 months of 2D

    # Run full parameter reader
    results = read_benthic_parameter(IDIR, dummy_mask3d, dummy_Bmost, dummy_filename_fragments, variable_key="votemper")

    # Output should be a dict: year → list of 12 monthly 2D arrays
    assert isinstance(results, dict)
    assert set(results.keys()) == {2000, 2001}
    for vals in results.values():
        assert isinstance(vals, list) and len(vals) == 12
        for arr in vals:
            assert arr.shape == dummy_Bmost.shape
            
###############################################################################

###############################################################################

# Ensure invalid directory, shapes, or keys raise appropriate exceptions
def test_read_benthic_parameter_invalid_inputs(dummy_mask3d, dummy_Bmost, dummy_filename_fragments):
    with pytest.raises(FileNotFoundError):
        read_benthic_parameter(Path("/no/such/dir"), dummy_mask3d, dummy_Bmost, dummy_filename_fragments, "votemper")

    with mock.patch('pathlib.Path.exists', return_value=True):
        # Invalid shape of mask
        with pytest.raises(ValueError):
            read_benthic_parameter("/fake", np.ones((2,3)), dummy_Bmost, dummy_filename_fragments, "votemper")
        # Invalid shape of Bmost
        with pytest.raises(ValueError):
            read_benthic_parameter("/fake", dummy_mask3d, np.ones((2,3,4)), dummy_filename_fragments, "votemper")
        # Missing filename fragments
        with pytest.raises(ValueError):
            read_benthic_parameter("/fake", dummy_mask3d, dummy_Bmost, None, "votemper")
        # Invalid filename fragment field
        with pytest.raises(ValueError):
            read_benthic_parameter("/fake", dummy_mask3d, dummy_Bmost,
                                   {'ffrag1': None, 'ffrag2': 'b', 'ffrag3': 'c'}, "votemper")
            
###############################################################################

###############################################################################

# Simulate reading and processing benthic chemical data across years with all intermediate steps mocked
@mock.patch("pathlib.Path.iterdir")
@mock.patch("pathlib.Path.exists")
@mock.patch("os.remove")
@mock.patch("Hydrological_model_validator.Processing.BFM_data_reader.extract_bottom_layer")
@mock.patch("Hydrological_model_validator.Processing.data_alignment.apply_3d_mask")
@mock.patch("Hydrological_model_validator.Processing.BFM_data_reader.read_nc_variable_from_unzipped_file")
@mock.patch("Hydrological_model_validator.Processing.file_io.unzip_gz_to_file")
@mock.patch("Hydrological_model_validator.Processing.utils.build_bfm_filename")
@mock.patch("Hydrological_model_validator.Processing.utils.infer_years_from_path")
def test_read_bfm_chemical_basic(
    mock_infer_years, mock_build_filename, mock_unzip, mock_read_nc,
    mock_apply_mask, mock_extract_bottom_layer, mock_os_remove,
    mock_exists, mock_iterdir,
    dummy_mask3d, dummy_Bmost
):
    mock_exists.return_value = True

    # Simulate folder structure of yearly outputs
    folder_2000 = MagicMock(spec=Path)
    folder_2000.name = "output2000"
    folder_2000.is_dir.return_value = True

    folder_2001 = MagicMock(spec=Path)
    folder_2001.name = "output2001"
    folder_2001.is_dir.return_value = True

    mock_iterdir.return_value = iter([folder_2000, folder_2001])
    mock_infer_years.return_value = (2000, 2001, [2000, 2001])

    # Simulate full chain of operations from filename to post-processed data
    mock_build_filename.side_effect = lambda *args, **kwargs: f"ADR{args[0]}X{args[0]}Y{args[0]}Z.nc"
    mock_unzip.side_effect = lambda filename: filename  # Simulate unzipping
    mock_read_nc.return_value = np.ones((12, 5, 4, 3))  # Raw 4D data
    mock_apply_mask.return_value = np.ones((12, 5, 4, 3))  # Masked data
    mock_extract_bottom_layer.return_value = [np.ones((4, 3)) for _ in range(12)]  # Benthic slices

    # Run chemical reader
    results = read_bfm_chemical(Path("/fake/dir"), dummy_mask3d, dummy_Bmost,
                                {"ffrag1": "X", "ffrag2": "Y", "ffrag3": "Z"},
                                variable_key="Chl")

    # Validate expected shape and format of result
    assert isinstance(results, dict)
    assert set(results.keys()) == {2000, 2001}
    for year_data in results.values():
        assert isinstance(year_data, list) and len(year_data) == 12
        for monthly_array in year_data:
            assert isinstance(monthly_array, np.ndarray)
            assert monthly_array.shape == (4, 3)
            
###############################################################################

###############################################################################

def test_extract_bottom_layer_clipping(dummy_4d_data):
    # Construct dummy Bmost with an invalid index (6),
    # expecting it to be clipped to max depth index (5) internally
    dummy_Bmost = np.array([[1, 2, 3],
                            [4, 5, 1],
                            [2, 1, 3],
                            [5, 4, 2]], dtype=int)

    result = extract_bottom_layer(dummy_4d_data, dummy_Bmost)

    # Check result is a list with one 2D array per time slice
    assert isinstance(result, list)
    assert len(result) == dummy_4d_data.shape[0]

    for layer_2d in result:
        # Each 2D layer must have the same shape as dummy_Bmost
        assert layer_2d.shape == dummy_Bmost.shape

        # There should be valid (non-NaN) data somewhere
        assert not np.isnan(layer_2d).all()
        

###############################################################################

###############################################################################

def test_extract_and_filter_benthic_data_unknown_variable(dummy_4d_data, dummy_Bmost):
    # Call function with unknown variable_key, expect no error and output shape correct
    result = extract_and_filter_benthic_data(dummy_4d_data, dummy_Bmost, variable_key='unknown_var')
    assert isinstance(result, np.ndarray)
    assert result.shape == (dummy_4d_data.shape[0], dummy_4d_data.shape[2], dummy_4d_data.shape[3])

###############################################################################

# Test that process_year raises error if variable_key missing in dataset
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.is_dir', return_value=True)
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.exists', return_value=True)
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.build_bfm_filename')
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.gzip.open')
@mock.patch('xarray.open_dataset')
def test_process_year_missing_variable_key(
    mock_xr_open,
    mock_gzip_open,
    mock_build_filename,
    mock_path_exists,
    mock_path_is_dir,
    dummy_mask3d,
    dummy_Bmost,
    dummy_filename_fragments
):
    year = 2000
    variable_key = "missing_var"
    IDIR = Path("/fake/dir")
    filename = "file.nc"
    mock_build_filename.return_value = filename

    # Mock dataset that does NOT contain variable_key
    dummy_ds = mock.MagicMock()
    dummy_ds.__enter__.return_value = dummy_ds
    dummy_ds.__contains__.return_value = False
    mock_xr_open.return_value = dummy_ds

    mock_gzip_open.return_value.__enter__.return_value.read.return_value = b"dummybytes"

    with pytest.raises(KeyError):
        process_year(year, IDIR, dummy_mask3d, dummy_Bmost, dummy_filename_fragments, variable_key)


###############################################################################

@mock.patch("Hydrological_model_validator.Processing.BFM_data_reader.infer_years_from_path")
@mock.patch("Hydrological_model_validator.Processing.utils.build_bfm_filename")
@mock.patch("Hydrological_model_validator.Processing.file_io.unzip_gz_to_file")
@mock.patch("Hydrological_model_validator.Processing.BFM_data_reader.read_nc_variable_from_unzipped_file")
@mock.patch("Hydrological_model_validator.Processing.data_alignment.apply_3d_mask")
@mock.patch("Hydrological_model_validator.Processing.BFM_data_reader.extract_bottom_layer")
@mock.patch("os.remove")
@mock.patch("pathlib.Path.exists", return_value=True)
@mock.patch("pathlib.Path.iterdir")
def test_read_bfm_chemical_empty_dir(
    mock_iterdir, mock_exists, mock_os_remove,
    mock_extract_bottom_layer, mock_apply_mask, mock_read_nc,
    mock_unzip, mock_build_filename, mock_infer_years,
    dummy_mask3d, dummy_Bmost
):
    # Simulate an empty directory: no folders returned by iterdir()
    mock_iterdir.return_value = iter([])

    # Mock infer_years_from_path to return no years found, avoiding ValueError
    mock_infer_years.return_value = (None, None, [])

    # Even if these are called, just mock their return values to defaults
    mock_build_filename.side_effect = lambda *args, **kwargs: "fake.nc"
    mock_unzip.side_effect = lambda filename: filename
    mock_read_nc.return_value = np.ones((12, 5, 4, 3))
    mock_apply_mask.return_value = np.ones((12, 5, 4, 3))
    mock_extract_bottom_layer.return_value = [np.ones((4, 3)) for _ in range(12)]

    # Call the function with the empty directory path
    results = read_bfm_chemical(
        Path("/empty/dir"),
        dummy_mask3d,
        dummy_Bmost,
        {"ffrag1": "X", "ffrag2": "Y", "ffrag3": "Z"},
        variable_key="Chl"
    )

    # Since no years, expect an empty dict
    assert isinstance(results, dict)
    assert results == {}
    
###############################################################################

# Test read_bfm_chemical raises error if filename fragments missing keys
@mock.patch('Hydrological_model_validator.Processing.BFM_data_reader.Path.exists', return_value=True)
def test_read_bfm_chemical_missing_fragments(dummy_mask3d, dummy_Bmost):
    with pytest.raises(ValueError):
        read_bfm_chemical(Path("/fake/dir"), dummy_mask3d, dummy_Bmost,
                          {"ffrag1": "X", "ffrag3": "Z"},  # Missing ffrag2
                          variable_key="Chl")

###############################################################################

# Test extract_bottom_layer returns correct dtype (float) even if input is int
def test_extract_bottom_layer_dtype(dummy_Bmost):
    # Create integer dummy data with float conversion
    data = np.arange(2*5*4*3).reshape(2,5,4,3).astype(int)
    layers = extract_bottom_layer(data, dummy_Bmost)
    for layer in layers:
        assert isinstance(layer, np.ndarray)
        assert np.issubdtype(layer.dtype, np.integer)  # Input was int, output same dtype expected

###############################################################################
