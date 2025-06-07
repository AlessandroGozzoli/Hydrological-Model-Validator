import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import gzip
import shutil
import os

from Hydrological_model_validator.Processing.SAT_data_reader import sat_data_loader

# Helper fake function for find_key_variable to mock the key finder in variables dict
def fake_find_key_variable(variables, keys):
    for key in keys:
        if key in variables:
            return key
    return None

# Patch find_key_variable inside the module to use our fake
@pytest.fixture(autouse=True)
def patch_find_key_variable(monkeypatch):
    monkeypatch.setattr("Hydrological_model_validator.Processing.utils.find_key_variable", fake_find_key_variable)

# Base temporary directory setup fixture
@pytest.fixture
def tmp_dir(tmp_path):
    # Create dummy .gz files with data_level in name
    f1 = tmp_path / "data_l3_2000.nc.gz"
    f2 = tmp_path / "data_l3_2001.nc.gz"
    f1.write_text("compressed content")
    f2.write_text("compressed content")
    return tmp_path

def mock_path_exists(self: Path):
    """
    Side effect function for Path.exists used in SAT tests.
    Pretends .gz files and directories exist, others don't.
    """
    if self.is_dir():
        return True
    if self.suffix == '.gz':
        return True
    if self.suffix == '':
        return False
    return Path.__orig_exists__(self)
Path.__orig_exists__ = Path.exists

# ====== INPUT VALIDATION TESTS ======

def test_invalid_D_sat_type():
    with pytest.raises(TypeError):
        sat_data_loader('l3', 12345, 'chl')

def test_D_sat_not_exist(tmp_path):
    non_exist_path = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError):
        sat_data_loader('l3', non_exist_path, 'chl')

def test_D_sat_not_dir(tmp_path):
    file_path = tmp_path / "afile.txt"
    file_path.write_text("hello")
    with pytest.raises(NotADirectoryError):
        sat_data_loader('l3', file_path, 'chl')

def test_invalid_data_level_type(tmp_dir):
    with pytest.raises(TypeError):
        sat_data_loader(123, tmp_dir, 'chl')

def test_invalid_data_level_value(tmp_dir):
    with pytest.raises(ValueError):
        sat_data_loader('L5', tmp_dir, 'chl')

def test_invalid_varname_type(tmp_dir):
    with pytest.raises(TypeError):
        sat_data_loader('l3', tmp_dir, 123)

def test_invalid_varname_value(tmp_dir):
    with pytest.raises(ValueError):
        sat_data_loader('l3', tmp_dir, 'temp')

# ====== FILE DISCOVERY TEST ======

def test_no_files_found(tmp_path):
    # Directory with no .gz files
    with pytest.raises(FileNotFoundError):
        sat_data_loader('l3', tmp_path, 'chl')

# ====== MOCK NETCDF DATA AND FILE PROCESSING ======

@patch("Hydrological_model_validator.Processing.SAT_data_reader.gzip.open")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.shutil.copyfileobj")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.os.remove")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.ds")
def test_successful_load_and_decompression(mock_ds, mock_remove, mock_copy, mock_gzip, tmp_dir):
    # Setup mock for gzip.open and shutil.copyfileobj to do nothing
    mock_gzip.return_value.__enter__.return_value = MagicMock()
    mock_copy.return_value = None
    mock_remove.return_value = None

    # We simulate that uncompressed files do NOT exist initially
    original_exists = Path.exists
    def exists_side_effect(self):
        # Return True for the test directory itself
        if self == tmp_dir:
            return True
        # .gz files exist
        if self.suffix == '.gz':
            return True
        # uncompressed files don't exist
        if self.suffix == '':
            # But if it's a directory other than tmp_dir, defer to original
            return original_exists(self)
        return original_exists(self)

    with patch.object(Path, "exists", new=exists_side_effect):

        # Prepare fake NetCDF variables data
        def ds_side_effect(path, mode):
            mock_nc = MagicMock()

            # Variables keys for longitude, latitude, time, and data variable
            mock_nc.variables = {
                'lon': np.linspace(0, 9, 10),
                'lat': np.linspace(0, 4, 5),
                'time': np.arange(3),
                'chl': np.ones((3, 5, 10)),
            }

            # Context manager support
            mock_nc.__enter__.return_value = mock_nc
            mock_nc.__exit__.return_value = None
            return mock_nc

        mock_ds.side_effect = ds_side_effect

        # Run loader
        T, data, lon, lat = sat_data_loader('l3', tmp_dir, 'chl')

        # Basic asserts on shapes and types
        assert isinstance(T, np.ndarray)
        assert isinstance(data, np.ndarray)
        assert isinstance(lon, np.ndarray)
        assert isinstance(lat, np.ndarray)
        assert lon.shape == lat.T.shape
        assert data.shape[1:] == lon.shape
        assert T.size == data.shape[0]

# ====== TEST KEY ERROR FOR MISSING VARIABLES ======

@patch("Hydrological_model_validator.Processing.SAT_data_reader.gzip.open")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.shutil.copyfileobj")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.os.remove")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.ds")
def test_missing_lon_lat_raises_keyerror(mock_ds, mock_remove, mock_copy, mock_gzip, tmp_dir):
    # Set up mocks
    mock_gzip.return_value.__enter__.return_value = MagicMock()
    mock_copy.return_value = None
    mock_remove.return_value = None

    def ds_side_effect(path, mode):
        mock_nc = MagicMock()
        mock_nc.variables = {
            'latitude': np.array([1, 2]),
            'time': np.arange(2),
            'chl': np.ones((2, 2, 2)),
        }
        mock_nc.__enter__.return_value = mock_nc
        mock_nc.__exit__.return_value = None
        return mock_nc

    mock_ds.side_effect = ds_side_effect

    with patch.object(Path, "exists", new=mock_path_exists):
        with pytest.raises(KeyError):
            sat_data_loader('l3', tmp_dir, 'chl')

@patch("Hydrological_model_validator.Processing.SAT_data_reader.gzip.open")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.shutil.copyfileobj")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.os.remove")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.ds")
def test_missing_time_variable_raises_keyerror(mock_ds, mock_remove, mock_copy, mock_gzip, tmp_dir):
    # Mock gzip.open context manager to avoid real decompression
    mock_gzip.return_value.__enter__.return_value = MagicMock()
    mock_copy.return_value = None
    mock_remove.return_value = None

    def ds_side_effect(path, mode):
        mock_nc = MagicMock()
        mock_nc.variables = {
            'lon': np.array([1, 2]),
            'lat': np.array([1, 2]),
            # time variable intentionally missing here
            'chl': np.ones((2, 2, 2)),
        }
        mock_nc.__enter__.return_value = mock_nc
        mock_nc.__exit__.return_value = None
        return mock_nc

    mock_ds.side_effect = ds_side_effect

    with patch.object(Path, "exists", new=mock_path_exists):
        with pytest.raises(KeyError):
            sat_data_loader('l3', tmp_dir, 'chl')

@patch("Hydrological_model_validator.Processing.SAT_data_reader.gzip.open")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.shutil.copyfileobj")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.os.remove")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.ds")
def test_missing_data_var_raises_keyerror(mock_ds, mock_remove, mock_copy, mock_gzip, tmp_dir):
    # Mock gzip.open to avoid real decompression
    mock_gzip.return_value.__enter__.return_value = MagicMock()
    mock_copy.return_value = None
    mock_remove.return_value = None

    def ds_side_effect(path, mode):
        mock_nc = MagicMock()
        mock_nc.variables = {
            'lon': np.array([1, 2]),
            'lat': np.array([1, 2]),
            'time': np.arange(2),
            # missing 'chl' data variable here
        }
        mock_nc.__enter__.return_value = mock_nc
        mock_nc.__exit__.return_value = None
        return mock_nc

    mock_ds.side_effect = ds_side_effect

    with patch.object(Path, "exists", new=mock_path_exists):
        with pytest.raises(KeyError):
            sat_data_loader('l3', tmp_dir, 'chl')

# ====== TEST VALUE ERROR FOR LON/LAT SHAPE AND DATA SHAPE ======

@patch("Hydrological_model_validator.Processing.SAT_data_reader.gzip.open")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.shutil.copyfileobj")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.os.remove")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.ds")
def test_lon_lat_not_1d_raises_valueerror(mock_ds, mock_remove, mock_copy, mock_gzip, tmp_dir):
    # Mock gzip.open to avoid real decompression
    mock_gzip.return_value.__enter__.return_value = MagicMock()
    mock_copy.return_value = None
    mock_remove.return_value = None

    def ds_side_effect(path, mode):
        mock_nc = MagicMock()
        mock_nc.variables = {
            'lon': np.array([[1, 2], [3, 4]]),  # 2D array triggers ValueError
            'lat': np.array([[1, 2], [3, 4]]),
            'time': np.arange(2),
            'chl': np.ones((2, 2, 2)),
        }
        mock_nc.__enter__.return_value = mock_nc
        mock_nc.__exit__.return_value = None
        return mock_nc

    mock_ds.side_effect = ds_side_effect

    with patch.object(Path, "exists", new=mock_path_exists):
        with pytest.raises(ValueError):
            sat_data_loader('l3', tmp_dir, 'chl')

@patch("Hydrological_model_validator.Processing.SAT_data_reader.gzip.open")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.shutil.copyfileobj")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.os.remove")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.ds")
def test_data_not_3d_raises_valueerror(mock_ds, mock_remove, mock_copy, mock_gzip, tmp_dir):
    # Mock gzip.open to avoid actual gzip file handling
    mock_gzip.return_value.__enter__.return_value = MagicMock()
    mock_copy.return_value = None
    mock_remove.return_value = None

    def ds_side_effect(path, mode):
        mock_nc = MagicMock()
        mock_nc.variables = {
            'lon': np.array([1, 2]),
            'lat': np.array([1, 2]),
            'time': np.arange(2),
            'chl': np.ones((2, 2)),  # 2D data should cause ValueError
        }
        mock_nc.__enter__.return_value = mock_nc
        mock_nc.__exit__.return_value = None
        return mock_nc

    mock_ds.side_effect = ds_side_effect

    with patch.object(Path, "exists", new=mock_path_exists):
        with pytest.raises(ValueError):
            sat_data_loader('l3', tmp_dir, 'chl')


# ====== TEST SST KELVIN TO CELSIUS CONVERSION ======

@patch("Hydrological_model_validator.Processing.SAT_data_reader.gzip.open")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.shutil.copyfileobj")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.os.remove")
@patch("Hydrological_model_validator.Processing.SAT_data_reader.ds")
def test_sst_kelvin_to_celsius_conversion(mock_ds, mock_remove, mock_copy, mock_gzip, tmp_dir):
    mock_gzip.return_value.__enter__.return_value = MagicMock()
    mock_copy.return_value = None
    mock_remove.return_value = None

    def ds_side_effect(path, mode):
        mock_nc = MagicMock()
        mock_nc.variables = {
            'lon': np.linspace(0, 9, 10),
            'lat': np.linspace(0, 4, 5),
            'time': np.arange(3),
            'sst': np.full((3, 5, 10), 300.0),  # 300K
        }
        mock_nc.__enter__.return_value = mock_nc
        mock_nc.__exit__.return_value = None
        return mock_nc

    mock_ds.side_effect = ds_side_effect

    T, data, lon, lat = sat_data_loader('l3', tmp_dir, 'sst')
    # Confirm that Kelvin was converted to Celsius
    assert np.allclose(data, 300 - 273.15)

