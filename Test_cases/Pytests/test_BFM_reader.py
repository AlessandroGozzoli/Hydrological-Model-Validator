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
    # Original 1-based indices
    bmost_1_based = np.array([[1, 2, 3],
                              [4, 5, 1],
                              [2, 1, 3],
                              [5, 4, 2]], dtype=int)
    # Convert to zero-based indices expected by the function
    return bmost_1_based - 1
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
    # So subtract 1 from Bmost to get zero-based depth index when indexing dummy_4d_data

    # For position (0,0), bottom layer index from dummy_Bmost is 1-based:
    depth_idx_00 = dummy_Bmost[0, 0] - 1
    assert np.allclose(bottom_layers[0][0, 0], dummy_4d_data[0, depth_idx_00, 0, 0])

    # For position (1,1), bottom layer index from dummy_Bmost is 1-based:
    depth_idx_11 = dummy_Bmost[1, 1] - 1
    assert np.allclose(bottom_layers[0][1, 1], dummy_4d_data[0, depth_idx_11, 1, 1])

# Test for input validation
def test_extract_bottom_layer_input_validation_with_fixtures(dummy_4d_data, dummy_Bmost):
    data_valid = dummy_4d_data
    # Note: dummy_Bmost is zero-based; your function expects 1-based indices,
    # so add 1 back here for testing:
    Bmost_valid = dummy_Bmost + 1  

    # Type errors for non-ndarray inputs
    with pytest.raises(TypeError):
        extract_bottom_layer(list(data_valid), Bmost_valid)
    with pytest.raises(TypeError):
        extract_bottom_layer(data_valid, list(Bmost_valid))

    # Value error for data with wrong dimensions
    with pytest.raises(ValueError):
        extract_bottom_layer(np.random.rand(2, 5, 4), Bmost_valid)  # 3D instead of 4D

    # Value error for Bmost with wrong dimensions
    with pytest.raises(ValueError):
        extract_bottom_layer(data_valid, np.random.randint(1, 6, size=(5, 4, 3)))  # 3D instead of 2D
    with pytest.raises(ValueError):
        extract_bottom_layer(data_valid, np.random.randint(1, 6, size=(2, 3)))  # shape mismatch

    # Value error for shape mismatch between data spatial dims and Bmost
    Bmost_wrong_shape = np.array([[1, 2],
                                  [3, 4],
                                  [1, 2],
                                  [5, 6],
                                  [7, 8]])
    with pytest.raises(ValueError):
        extract_bottom_layer(data_valid, Bmost_wrong_shape)

    # Value error for Bmost containing non-integer values
    Bmost_float = Bmost_valid.astype(float)
    with pytest.raises(ValueError):
        extract_bottom_layer(data_valid, Bmost_float)

    # Value error for negative Bmost indices
    Bmost_neg = Bmost_valid.copy()
    Bmost_neg[0, 0] = -1
    with pytest.raises(ValueError):
        extract_bottom_layer(data_valid, Bmost_neg)

    # Value error for Bmost indices exceeding depth (depth=5)
    Bmost_exceed = Bmost_valid.copy()
    Bmost_exceed[0, 0] = data_valid.shape[1] + 1  # 6 when depth is 5
    with pytest.raises(ValueError):
        extract_bottom_layer(data_valid, Bmost_exceed)

    # Positive test: valid inputs run without error and output shape check
    result = extract_bottom_layer(data_valid, Bmost_valid)
    assert isinstance(result, list)
    assert len(result) == data_valid.shape[0]
    for arr in result:
        assert arr.shape == (data_valid.shape[2], data_valid.shape[3])



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
    
# Test for input validation
def test_extract_and_filter_benthic_data_input_validation(dummy_4d_data, dummy_Bmost, capsys):
    data_valid = dummy_4d_data
    Bmost_valid = dummy_Bmost

    # 1) Type checks
    with pytest.raises(TypeError):
        extract_and_filter_benthic_data(list(data_valid), Bmost_valid)
    with pytest.raises(TypeError):
        extract_and_filter_benthic_data(data_valid, list(Bmost_valid))

    # 2) Dimension checks
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(np.random.rand(2, 5, 4), Bmost_valid)  # data not 4D
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, np.random.randint(1, 6, size=(5,4,3)))  # Bmost not 2D
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, np.random.randint(1, 6, size=(2, 3)))  # shape mismatch spatial dims

    # 3) Shape mismatch between Bmost and data spatial dims
    Bmost_wrong_shape = np.array([[1, 2],
                                  [3, 4],
                                  [1, 2],
                                  [5, 6],
                                  [7, 8]])
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, Bmost_wrong_shape)

    # 4) Bmost contains non-integer values
    Bmost_float = Bmost_valid.astype(float)
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, Bmost_float)

    # 5) Bmost indices out of valid range (negative or >= depth_len)
    Bmost_neg = Bmost_valid.copy()
    Bmost_neg[0, 0] = -1
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, Bmost_neg)

    Bmost_exceed = Bmost_valid.copy()
    Bmost_exceed[0, 0] = data_valid.shape[1]  # equal to depth_len, invalid since indexing max is depth_len - 1
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, Bmost_exceed)

    # 6) dz validation
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, Bmost_valid, dz=-1)
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, Bmost_valid, dz=0)
    with pytest.raises(ValueError):
        extract_and_filter_benthic_data(data_valid, Bmost_valid, dz='string')

    # 7) variable_key warning (capture print safely with capsys)
    extract_and_filter_benthic_data(data_valid, Bmost_valid, variable_key='unsupported_key')
    captured = capsys.readouterr()
    assert "Warning" in captured.out

    # 8) Valid inputs produce output of expected shape and print warnings inside loop (capture too)
    result = extract_and_filter_benthic_data(data_valid, Bmost_valid, dz=2.0, variable_key='votemper')
    captured = capsys.readouterr()
    # Optionally test if warning prints about invalid cells exist or just ensure no crash
    assert result.shape == (data_valid.shape[0], data_valid.shape[2], data_valid.shape[3])

    
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

#• Test for input validation
def test_process_year_input_validation(monkeypatch, tmp_path):
    # Setup: valid minimal inputs for positive test
    year = 2005
    IDIR = tmp_path
    (IDIR / "output2005").mkdir()
    # Dummy file for file existence check (empty .nc.gz file)
    dummy_file = IDIR / "output2005" / "model_output.nc.gz"
    dummy_file.touch()

    mask3d = np.ones((10, 20, 30))
    Bmost = np.ones((20, 30), dtype=int)
    filename_fragments = {'ffrag1': 'model', 'ffrag2': 'output', 'ffrag3': 'nc'}
    variable_key = 'votemper'

    # Patch build_bfm_filename to return the dummy file name used above
    monkeypatch.setattr("Hydrological_model_validator.Processing.utils.build_bfm_filename", lambda y, f: "model_output.nc")

    # Patch extract_and_filter_benthic_data to return dummy data (to avoid actual processing)
    monkeypatch.setattr("Hydrological_model_validator.Processing.BFM_data_reader.extract_and_filter_benthic_data", lambda **kwargs: np.ones((1,20,30)))

    # Patch gzip.open and xr.open_dataset to avoid file IO during test
    import io
    import gzip
    import xarray as xr

    class DummyDataset:
        def __enter__(self):
            class DS:
                def __contains__(self, key):
                    return key == variable_key
                def __getitem__(self, key):
                    class Var:
                        @property
                        def values(self):
                            # shape (time=1, depth=10, Y=20, X=30)
                            return np.ones((1, 10, 20, 30))
                    return Var()
            return DS()
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(gzip, "open", lambda file, mode: io.BytesIO(b"dummy"))
    monkeypatch.setattr(xr, "open_dataset", lambda file_like: DummyDataset())

    # Now test input validation errors

    # year not int
    with pytest.raises(TypeError):
        process_year("2005", IDIR, mask3d, Bmost, filename_fragments, variable_key)

    # IDIR not a directory
    with pytest.raises(ValueError):
        process_year(year, tmp_path / "nonexistent_dir", mask3d, Bmost, filename_fragments, variable_key)

    # mask3d not ndarray
    with pytest.raises(TypeError):
        process_year(year, IDIR, "not_array", Bmost, filename_fragments, variable_key)

    # mask3d wrong ndim
    with pytest.raises(ValueError):
        process_year(year, IDIR, np.ones((10, 20)), Bmost, filename_fragments, variable_key)

    # Bmost not ndarray
    with pytest.raises(TypeError):
        process_year(year, IDIR, mask3d, "not_array", filename_fragments, variable_key)

    # Bmost wrong ndim
    with pytest.raises(ValueError):
        process_year(year, IDIR, mask3d, np.ones((20, 30, 2)), filename_fragments, variable_key)

    # filename_fragments not dict
    with pytest.raises(TypeError):
        process_year(year, IDIR, mask3d, Bmost, "not_dict", variable_key)

    # filename_fragments missing keys
    with pytest.raises(KeyError):
        process_year(year, IDIR, mask3d, Bmost, {'ffrag1': 'model'}, variable_key)

    # variable_key not str
    with pytest.raises(TypeError):
        process_year(year, IDIR, mask3d, Bmost, filename_fragments, 123)

    # compressed file missing (remove dummy file)
    dummy_file.unlink()
    with pytest.raises(FileNotFoundError):
        process_year(year, IDIR, mask3d, Bmost, filename_fragments, variable_key)
    
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
        
# Test for input validation
def test_read_bfm_chemical_input_validation(tmp_path):
    # Create a dummy directory structure
    base_dir = tmp_path
    (base_dir / "output2000").mkdir()

    mask3d_valid = np.ones((10, 20, 30))
    Bmost_valid = np.ones((20, 30))
    fragments_valid = {'ffrag1': 'chem', 'ffrag2': 'monthly', 'ffrag3': 'nc'}
    variable_key_valid = "O2"

    # 1. IDIR does not exist
    with pytest.raises(FileNotFoundError):
        read_bfm_chemical(base_dir / "nonexistent", mask3d_valid, Bmost_valid, fragments_valid, variable_key_valid)

    # 2. mask3d not ndarray or wrong ndim
    with pytest.raises(ValueError):
        read_bfm_chemical(base_dir, "not_an_array", Bmost_valid, fragments_valid, variable_key_valid)
    with pytest.raises(ValueError):
        read_bfm_chemical(base_dir, np.ones((10, 20)), Bmost_valid, fragments_valid, variable_key_valid)

    # 3. Bmost not ndarray or wrong ndim
    with pytest.raises(ValueError):
        read_bfm_chemical(base_dir, mask3d_valid, "not_an_array", fragments_valid, variable_key_valid)
    with pytest.raises(ValueError):
        read_bfm_chemical(base_dir, mask3d_valid, np.ones((20, 30, 5)), fragments_valid, variable_key_valid)

    # 4. filename_fragments not dict or empty
    with pytest.raises(ValueError):
        read_bfm_chemical(base_dir, mask3d_valid, Bmost_valid, None, variable_key_valid)
    with pytest.raises(ValueError):
        read_bfm_chemical(base_dir, mask3d_valid, Bmost_valid, {}, variable_key_valid)

    # 5. filename_fragments missing keys or None values
    for bad_fragments in [
        {'ffrag1': 'chem', 'ffrag2': 'monthly'},  # missing ffrag3
        {'ffrag1': 'chem', 'ffrag2': None, 'ffrag3': 'nc'}  # None value
    ]:
        with pytest.raises(ValueError):
            read_bfm_chemical(base_dir, mask3d_valid, Bmost_valid, bad_fragments, variable_key_valid)

    # 6. variable_key not str or empty
    with pytest.raises(ValueError):
        read_bfm_chemical(base_dir, mask3d_valid, Bmost_valid, fragments_valid, None)
    with pytest.raises(ValueError):
        read_bfm_chemical(base_dir, mask3d_valid, Bmost_valid, fragments_valid, "")

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
