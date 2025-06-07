import os
import gzip
import shutil
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from netCDF4 import Dataset

# Import the function to test
from Hydrological_model_validator.Processing.MOD_data_reader import read_model_data 


# ==== Mock helpers ====
def fake_infer_years_from_path(path, target_type, pattern):
    # Return a fixed year range for testing
    return 2000, 2001, [2000, 2001]

def fake_leapyear(year):
    # Return 365 for 2000 and 366 for 2001 as example
    return 366 if year == 2000 else 365

# Mock Dataset context manager to simulate NetCDF file with expected variables
class MockDataset:
    def __init__(self, variables, shape_first_dim):
        self.variables = variables
        self.shape_first_dim = shape_first_dim
    def __enter__(self):
        # Create dummy variable arrays with specified shape
        self.variables = {k: np.zeros((self.shape_first_dim,) + v[1:]) for k, v in self.variables.items()}
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass


# ==== TESTS ====

# Test invalid Dmod type raises TypeError
def test_invalid_Dmod_type_raises():
    with pytest.raises(TypeError):
        read_model_data(123, (np.array([0]), np.array([0])), 'chl')

# Test non-existent directory raises FileNotFoundError
def test_nonexistent_Dmod_raises(tmp_path):
    non_dir = tmp_path / "nonexistent"
    with pytest.raises((FileNotFoundError, ValueError)):
        read_model_data(str(non_dir), (np.array([0]), np.array([0])), 'chl')

# Test Dmod that is a file (not a directory) raises NotADirectoryError
def test_Dmod_not_directory(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")
    with pytest.raises(NotADirectoryError):
        read_model_data(str(file_path), (np.array([0]), np.array([0])), 'chl')

# Test invalid Mfsm type raises TypeError
def test_invalid_Mfsm_type(tmp_path):
    dmod = tmp_path
    with pytest.raises(TypeError):
        read_model_data(str(dmod), "not_a_tuple", 'chl')

# Test Mfsm tuple arrays have different shapes raises ValueError
def test_Mfsm_shape_mismatch(tmp_path):
    dmod = tmp_path
    Mfsm = (np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        read_model_data(str(dmod), Mfsm, 'chl')

# Test invalid variable_name raises ValueError
def test_invalid_variable_name(tmp_path):
    dmod = tmp_path
    Mfsm = (np.zeros((1,)), np.zeros((1,)))
    with pytest.raises(ValueError):
        read_model_data(str(dmod), Mfsm, 'invalid_var')

# Test no matching files raises FileNotFoundError
@patch('Hydrological_model_validator.Processing.utils.infer_years_from_path', side_effect=fake_infer_years_from_path)
@patch('Hydrological_model_validator.Processing.time_utils.leapyear', side_effect=fake_leapyear)
def test_no_matching_files_raises(mock_leap, mock_infer, tmp_path):
    dmod = tmp_path
    # Create directories for years
    (dmod / 'output2000').mkdir()
    (dmod / 'output2001').mkdir()
    Mfsm = (np.zeros((1,)), np.zeros((1,)))
    with pytest.raises(FileNotFoundError):
        read_model_data(str(dmod), Mfsm, 'chl')

# Test normal flow for chl variable with masking applied
@patch('Hydrological_model_validator.Processing.utils.infer_years_from_path', side_effect=fake_infer_years_from_path)
@patch('Hydrological_model_validator.Processing.time_utils.leapyear', side_effect=fake_leapyear)
@patch('Hydrological_model_validator.Processing.MOD_data_reader.Dataset')
def test_normal_chl_flow(mock_dataset, mock_leap, mock_infer, tmp_path):
    dmod = tmp_path

    # Setup directories and files
    for year in [2000, 2001]:
        folder = dmod / f"output{year}"
        folder.mkdir()
        # Create dummy .nc files (non-gz)
        file_path = folder / f"dummy_Chl.nc"
        file_path.write_text("dummy")

    # Setup Mfsm mask indices
    Mfsm = (np.array([0]), np.array([0]))

    # Setup Dataset mock to provide variable with shape matching amileap
    def dataset_side_effect(path, mode):
        path = Path(path)
        folder_name = path.parent.name  # e.g. 'output2000'
        y = int(folder_name.replace("output", ""))
        shape = (366, 10, 10) if y == 2000 else (365, 10, 10)

        mock_var = MagicMock()
        mock_var.__getitem__.side_effect = lambda s: np.ones(shape)[s]
        mock_var.shape = shape

        mock_ds = MagicMock()
        mock_ds.variables = {'Chlasat_od': mock_var}
    
        # Add context manager support to mock_ds:
        mock_ds.__enter__.return_value = mock_ds
        mock_ds.__exit__.return_value = None

        return mock_ds

    mock_dataset.side_effect = dataset_side_effect

    # Run function, should not raise and output shape matches total days
    result = read_model_data(str(dmod), Mfsm, 'chl')
    assert isinstance(result, np.ndarray)
    expected_days = 366 + 365
    assert result.shape[0] == expected_days
    # Check that mask was applied (first day, masked indices are nan)
    assert np.isnan(result[0, Mfsm[0][0], Mfsm[1][0]])

# Test masking works on all days for sst variable
@patch('Hydrological_model_validator.Processing.utils.infer_years_from_path', side_effect=fake_infer_years_from_path)
@patch('Hydrological_model_validator.Processing.time_utils.leapyear', side_effect=fake_leapyear)
@patch('Hydrological_model_validator.Processing.MOD_data_reader.Dataset')
def test_masking_all_days_sst(mock_dataset, mock_leap, mock_infer, tmp_path):
    dmod = tmp_path

    for year in [2000, 2001]:
        folder = dmod / f"output{year}"
        folder.mkdir()
        file_path = folder / f"dummy_1d_test_grid_T.nc"
        file_path.write_text("dummy")

    Mfsm = (np.array([0, 1]), np.array([0, 1]))

    def dataset_side_effect(path, mode):
        path = Path(path)
        folder_name = path.parent.name  # e.g. 'output2000'
        y = int(folder_name.replace("output", ""))
        shape = (366, 5, 5) if y == 2000 else (365, 5, 5)
    
        mock_ds = MagicMock()
        mock_ds.variables = {'sst': np.ones(shape)}
        # Add context manager support
        mock_ds.__enter__.return_value = mock_ds
        mock_ds.__exit__.return_value = None
        return mock_ds

    mock_dataset.side_effect = dataset_side_effect

    result = read_model_data(str(dmod), Mfsm, 'sst')
    assert result.shape[0] == 366 + 365
    # All masked indices in all days should be NaN
    for i in range(result.shape[0]):
        for r, c in zip(Mfsm[0], Mfsm[1]):
            assert np.isnan(result[i, r, c])

# Test decompression block is triggered when only .gz file exists
@patch('Hydrological_model_validator.Processing.utils.infer_years_from_path', side_effect=fake_infer_years_from_path)
@patch('Hydrological_model_validator.Processing.time_utils.leapyear', side_effect=fake_leapyear)
@patch('Hydrological_model_validator.Processing.MOD_data_reader.Dataset')
@patch('gzip.open')
@patch('shutil.copyfileobj')
@patch('os.remove')
def test_decompression_block(mock_remove, mock_copy, mock_gzip, mock_dataset, mock_leap, mock_infer, tmp_path):
    dmod = tmp_path

    for year in [2000]:
        folder = dmod / f"output{year}"
        folder.mkdir()
        # Create only .gz file, no uncompressed file
        gz_file = folder / f"dummy_Chl.nc.gz"
        gz_file.write_text("compressed")

    Mfsm = (np.array([0]), np.array([0]))

    def dataset_side_effect(path, mode):
        path = Path(path)
        folder_name = path.parent.name  # e.g. 'output2000'
        y = int(folder_name.replace("output", ""))
        shape = (366, 10, 10) if y == 2000 else (365, 10, 10)

        mock_var = MagicMock()
        mock_var.__getitem__.side_effect = lambda s: np.ones(shape)[s]
        mock_var.shape = shape

        mock_ds = MagicMock()
        mock_ds.variables = {'Chlasat_od': mock_var}
    
        # Add context manager support to mock_ds:
        mock_ds.__enter__.return_value = mock_ds
        mock_ds.__exit__.return_value = None

        return mock_ds

    mock_dataset.side_effect = dataset_side_effect

    # Patch Path.exists to simulate missing uncompressed but existing .gz file
    original_exists = Path.exists

    def exists_side_effect(self):
        # Do not break the directory structure checks
        if self.is_dir() or self == dmod or self.parent == dmod:
            return True
        # Your simulated logic
        if self.suffix == '':
            return False
        if self.suffix == '.gz':
            return True
        return original_exists(self)

    with patch.object(Path, "exists", new=exists_side_effect):
        result = read_model_data(str(dmod), Mfsm, 'chl')
        assert result.shape[0] == 366
        mock_gzip.assert_called_once()
        mock_copy.assert_called_once()
        mock_remove.assert_called_once()

# Test KeyError raised when expected variable key is missing in NetCDF file
@patch('Hydrological_model_validator.Processing.utils.infer_years_from_path', side_effect=fake_infer_years_from_path)
@patch('Hydrological_model_validator.Processing.time_utils.leapyear', side_effect=fake_leapyear)
@patch('Hydrological_model_validator.Processing.MOD_data_reader.Dataset')
def test_keyerror_missing_variable(mock_dataset, mock_leap, mock_infer, tmp_path):
    dmod = tmp_path
    (dmod / "output2000").mkdir()
    file_path = dmod / "output2000" / "dummy_Chl.nc"
    file_path.write_text("dummy")
    Mfsm = (np.array([0]), np.array([0]))

    # Mock Dataset to have no variables
    mock_ds = MagicMock()
    mock_ds.variables = {}
    mock_dataset.return_value = mock_ds

    with pytest.raises(KeyError):
        read_model_data(str(dmod), Mfsm, 'chl')

# Test ValueError raised when number of days in data doesn't match expected
@patch('Hydrological_model_validator.Processing.utils.infer_years_from_path', side_effect=fake_infer_years_from_path)
@patch('Hydrological_model_validator.Processing.time_utils.leapyear', side_effect=fake_leapyear)
@patch('Hydrological_model_validator.Processing.MOD_data_reader.Dataset')
def test_days_mismatch_raises(mock_dataset, mock_leap, mock_infer, tmp_path):
    dmod = tmp_path
    (dmod / "output2000").mkdir()
    file_path = dmod / "output2000" / "dummy_Chl.nc"
    file_path.write_text("dummy")
    Mfsm = (np.array([0]), np.array([0]))

    # Side effect function to simulate the dataset open & variable access with wrong shape
    def dataset_side_effect(path, mode):
        path = Path(path)
        folder_name = path.parent.name  # e.g. 'output2000'
        y = int(folder_name.replace("output", ""))
        # Provide the "wrong" shape: only 100 days instead of expected 366 for 2000
        shape = (100, 10, 10)

        mock_var = MagicMock()
        mock_var.__getitem__.side_effect = lambda s: np.ones(shape)[s]
        mock_var.shape = shape

        mock_ds = MagicMock()
        mock_ds.variables = {'Chlasat_od': mock_var}
        mock_ds.__enter__.return_value = mock_ds
        mock_ds.__exit__.return_value = None
        return mock_ds

    mock_dataset.side_effect = dataset_side_effect

    with pytest.raises(ValueError):
        read_model_data(str(dmod), Mfsm, 'chl')
