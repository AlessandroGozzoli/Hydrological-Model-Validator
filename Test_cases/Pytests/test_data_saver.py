import numpy as np
import xarray as xr
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, mock_open
import tempfile
import json
import pandas as pd

from Hydrological_model_validator.Processing.Data_saver import (
    save_satellite_data, 
    save_model_data,
    save_to_netcdf,
    convert_to_serializable,
    save_variable_to_json
)

################################################################################
# Tests for save_satellite_data
################################################################################

# Test that save_satellite_data correctly saves .mat files when user selects option '1'
def test_save_satellite_data_mat_format():
    # Generate sample longitude, latitude, and 3D satellite data arrays with random values
    Sat_lon = np.random.rand(10, 10)
    Sat_lat = np.random.rand(10, 10)
    SatData_complete = np.random.rand(5, 10, 10)
    
    # Use a temporary directory to isolate file outputs for test
    with TemporaryDirectory() as tmpdir:
        # Mock user input to select saving format '1' (MAT file)
        with patch("builtins.input", return_value='1'):
            save_satellite_data(tmpdir, Sat_lon, Sat_lat, SatData_complete)
        
        # Check that the expected .mat file is created successfully
        # This verifies that the function correctly writes in MATLAB format
        assert (Path(tmpdir) / "SatData_clean.mat").exists()

# Test that save_satellite_data correctly saves NetCDF files when user selects option '2'
def test_save_satellite_data_netcdf_format():
    # Create sample input data for satellite longitude, latitude, and data arrays
    Sat_lon = np.random.rand(10, 10)
    Sat_lat = np.random.rand(10, 10)
    SatData_complete = np.random.rand(5, 10, 10)
    
    with TemporaryDirectory() as tmpdir:
        # Mock input to simulate user selecting NetCDF saving option '2'
        with patch("builtins.input", return_value='2'):
            save_satellite_data(tmpdir, Sat_lon, Sat_lat, SatData_complete)
        
        # Confirm that separate NetCDF files are created for lon, lat, and the data cube
        # This ensures the function splits and saves data appropriately in NetCDF format
        assert (Path(tmpdir) / "Sat_lon.nc").exists()
        assert (Path(tmpdir) / "Sat_lat.nc").exists()
        assert (Path(tmpdir) / "SatData_complete.nc").exists()

# Test that passing an invalid directory path raises a ValueError
def test_save_satellite_data_invalid_path():
    # Intentionally pass a non-existent directory to confirm proper error handling
    # This test verifies the function validates the output directory before saving
    with pytest.raises(ValueError, match="not a valid directory"):
        save_satellite_data("not/a/real/path", np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((3, 2, 2)))

# Test that providing a 2D array instead of 3D for SatData_complete raises a ValueError
def test_save_satellite_data_invalid_dims():
    # Generate longitude and latitude as 2D arrays
    Sat_lon = np.random.rand(10, 20)
    Sat_lat = np.random.rand(10, 20)
    
    # Provide invalid SatData_complete as 2D array, where 3D is expected (time, lat, lon)
    SatData_complete = np.random.rand(10, 20)  # Incorrect dimensions
    
    with TemporaryDirectory() as tmpdir:
        # Expect function to raise ValueError because 3D data cube is required for SatData_complete
        with pytest.raises(ValueError, match="should be 3D"):
            with patch("builtins.input", return_value='1'):
                save_satellite_data(tmpdir, Sat_lon, Sat_lat, SatData_complete)

# Test for input validation
def test_save_satellite_data_input_validation(tmp_path):
    # Prepare valid arrays with correct dims
    valid_lon = np.zeros((4, 5))
    valid_lat = np.zeros((4, 5))
    valid_data = np.zeros((3, 4, 5))

    # output_path not string/Path
    with pytest.raises(TypeError):
        save_satellite_data(123, valid_lon, valid_lat, valid_data)

    # output_path not a directory
    fake_file = tmp_path / "not_a_dir"
    fake_file.touch()
    with pytest.raises(ValueError):
        save_satellite_data(fake_file, valid_lon, valid_lat, valid_data)

    # Sat_lon not ndarray/xr.DataArray
    with pytest.raises(TypeError):
        save_satellite_data(tmp_path, "not_array", valid_lat, valid_data)

    # Sat_lat not ndarray/xr.DataArray
    with pytest.raises(TypeError):
        save_satellite_data(tmp_path, valid_lon, "not_array", valid_data)

    # SatData_complete not ndarray/xr.DataArray
    with pytest.raises(TypeError):
        save_satellite_data(tmp_path, valid_lon, valid_lat, "not_array")

    # Sat_lon wrong dims
    with pytest.raises(ValueError):
        save_satellite_data(tmp_path, np.zeros((4,)), valid_lat, valid_data)

    # Sat_lat wrong dims
    with pytest.raises(ValueError):
        save_satellite_data(tmp_path, valid_lon, np.zeros((4,)), valid_data)

    # SatData_complete wrong dims
    with pytest.raises(ValueError):
        save_satellite_data(tmp_path, valid_lon, valid_lat, np.zeros((4,5)))

# Json save test
@patch("builtins.input", return_value='4')  # simulate user chooses JSON save
@patch("builtins.open", new_callable=mock_open)
def test_save_satellite_data_json_save(mock_file, mock_input, tmp_path):
    # Prepare valid arrays
    Sat_lon = np.array([[1.0, 2.0], [3.0, 4.0]])
    Sat_lat = np.array([[5.0, 6.0], [7.0, 8.0]])
    SatData_complete = np.zeros((1, 2, 2))

    # Run function, should attempt to write JSON file
    save_satellite_data(tmp_path, Sat_lon, Sat_lat, SatData_complete)

    # Check file was opened for writing the json
    json_file_path = tmp_path / "SatData_clean.json"
    mock_file.assert_called_with(json_file_path, 'w')

    # Extract the actual written JSON data string
    handle = mock_file()
    written_data = ''.join(call.args[0] for call in handle.write.call_args_list)

    # Parse back JSON to dict to verify keys and shapes
    data = json.loads(written_data)
    assert 'Sat_lon' in data
    assert 'Sat_lat' in data
    assert 'SatData_complete' in data
    assert data['Sat_lon'] == Sat_lon.tolist()
    assert data['Sat_lat'] == Sat_lat.tolist()
    assert data['SatData_complete'] == SatData_complete.tolist()
    
    
################################################################################
# Tests for save_model_data
################################################################################

# Test that save_model_data correctly saves a .mat file when user selects option '1'
def test_save_model_data_mat_format():
    # Generate sample 3D model data array with random values
    ModData_complete = np.random.rand(4, 10, 10)
    
    with TemporaryDirectory() as tmpdir:
        # Mock input to select MATLAB saving option
        with patch("builtins.input", return_value='1'):
            save_model_data(tmpdir, ModData_complete)
        
        # Check that the .mat file is created to confirm correct saving behavior
        assert (Path(tmpdir) / "ModData_complete.mat").exists()

# Test that save_model_data correctly saves a NetCDF file when user selects option '2'
def test_save_model_data_netcdf_format():
    # Generate sample model data cube
    ModData_complete = np.random.rand(4, 10, 10)
    
    with TemporaryDirectory() as tmpdir:
        # Mock input for NetCDF saving option
        with patch("builtins.input", return_value='2'):
            save_model_data(tmpdir, ModData_complete)
        
        # Assert that NetCDF file is generated, verifying function behavior for this format
        assert (Path(tmpdir) / "ModData_complete.nc").exists()

# Test that providing a 2D array instead of 3D for ModData_complete raises a ValueError
def test_save_model_data_invalid_ndim():
    # Create invalid 2D model data (missing time dimension)
    ModData_complete = np.random.rand(10, 10)  # Only 2D
    
    with TemporaryDirectory() as tmpdir:
        # Expect error due to incorrect number of dimensions
        with pytest.raises(ValueError, match="should be 3D"):
            with patch("builtins.input", return_value='1'):
                save_model_data(tmpdir, ModData_complete)

# Test that passing a non-array (e.g., string) raises a TypeError
def test_save_model_data_invalid_type():
    with TemporaryDirectory() as tmpdir:
        # Passing a string instead of NumPy or xarray array should raise TypeError
        # This confirms type-checking is enforced
        with pytest.raises(TypeError, match="must be a NumPy array or xarray DataArray"):
            with patch("builtins.input", return_value='1'):
                save_model_data(tmpdir, "not_an_array")

################################################################################
# Tests for save_to_netcdf
################################################################################

# Test that save_to_netcdf successfully creates NetCDF files from dictionary of NumPy arrays
def test_save_to_netcdf_creates_files():
    # Prepare sample data dictionary with variable names and 2D arrays
    data_dict = {
        "var1": np.ones((2, 2)),
        "var2": np.zeros((2, 2))
    }
    
    with TemporaryDirectory() as tmpdir:
        # Call function to save each array as individual NetCDF files
        save_to_netcdf(data_dict, tmpdir)
        
        # Assert that files are created for all variables in dictionary
        assert (Path(tmpdir) / "var1.nc").exists()
        assert (Path(tmpdir) / "var2.nc").exists()

# Test that passing an invalid directory path raises a ValueError
def test_save_to_netcdf_invalid_path():
    # Test invalid directory handling to confirm function validates output path
    with pytest.raises(ValueError, match="not a valid directory"):
        save_to_netcdf({"test": np.ones((2, 2))}, "nonexistent/path")

# Test that save_to_netcdf accepts xarray DataArray in the data dictionary
def test_save_to_netcdf_accepts_xarray_dataarray():
    # Prepare xarray DataArray instead of NumPy array to check compatibility
    data = xr.DataArray(np.ones((2, 2)), name="var1")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Saving DataArray should succeed and produce NetCDF file
        save_to_netcdf({"var1": data}, output_dir)
        
        # Verify NetCDF file creation confirms DataArray support
        assert (output_dir / "var1.nc").exists()

# Test that save_to_netcdf overwrites existing NetCDF files with new data
def test_save_to_netcdf_overwrites_existing_files():
    # Initial data: zeros in a 3x3 array
    data = xr.DataArray(np.zeros((3, 3)), name="testvar")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # First save creates initial NetCDF file
        save_to_netcdf({"testvar": data}, output_dir)

        # Overwrite with new data: ones in same shape and name
        new_data = xr.DataArray(np.ones((3, 3)), name="testvar")
        save_to_netcdf({"testvar": new_data}, output_dir)

        # Open saved NetCDF to confirm data was overwritten (values are ones, not zeros)
        filepath = output_dir / "testvar.nc"
        with xr.open_dataset(filepath) as ds:
            assert np.allclose(ds["testvar"].values, 1.0)

################################################################################
# Tests for convert_to_serializable
################################################################################


# Dummy classes
class DummyWithToDict:
    def to_dict(self):
        return {"foo": "bar"}

class DummyDict:
    def to_dict(self):
        return {"key": "value"}

class DummyStr:
    def __str__(self):
        return "dummy"

class Unknown:
    def __str__(self):
        return "fallback"


def test_json_native_types():
    assert convert_to_serializable("text") == "text"
    assert convert_to_serializable(42.5) == 42.5
    assert convert_to_serializable(True) is True
    assert convert_to_serializable(None) is None


def test_iterables():
    assert convert_to_serializable([DummyStr(), DummyStr()]) == ["dummy", "dummy"]
    assert convert_to_serializable((DummyStr(),)) == ["dummy"]
    assert set(convert_to_serializable({DummyStr()})) == {"dummy"}


def test_dict():
    d = {"a": DummyStr(), 2: DummyStr()}
    result = convert_to_serializable(d)
    assert result == {"a": "dummy", "2": "dummy"}


def test_numpy_array():
    arr = np.array([[1, 2], [3, 4]])
    assert convert_to_serializable(arr) == [[1, 2], [3, 4]]


def test_pandas_objects():
    df = pd.DataFrame({'x': [1, 2]})
    assert convert_to_serializable(df) == [{'x': 1}, {'x': 2}]
    assert isinstance(json.dumps(convert_to_serializable(df)), str)

    s = pd.Series([1, 2], index=['a', 'b'])
    assert convert_to_serializable(s) == {'a': 1, 'b': 2}
    assert isinstance(json.dumps(convert_to_serializable(s)), str)


def test_xarray_dataarray():
    da = xr.DataArray(
        np.array([[10, 20], [30, 40]]),
        dims=("lat", "lon"),
        coords={"lat": [1, 2], "lon": [3, 4]}
    )
    result = convert_to_serializable(da)
    assert result["dims"] == ("lat", "lon")
    assert result["coords"] == {"lat": [1, 2], "lon": [3, 4]}
    assert result["data"] == [[10, 20], [30, 40]]
    assert isinstance(json.dumps(result), str)


def test_xarray_dataset():
    ds = xr.Dataset({"foo": (("x",), [1, 2, 3])})
    result = convert_to_serializable(ds)
    assert isinstance(result, dict)
    assert "data_vars" in result
    assert isinstance(json.dumps(result), str)


def test_object_with_to_dict():
    obj = DummyDict()
    assert convert_to_serializable(obj) == {"key": "value"}

    obj2 = DummyWithToDict()
    assert convert_to_serializable(obj2) == {"foo": "bar"}


def test_fallback_str():
    assert convert_to_serializable(Unknown()) == "fallback"
    assert isinstance(convert_to_serializable(object()), str)
    assert "function" in convert_to_serializable(lambda x: x)
    
    
################################################################################
# Tests for save_variable_to_json
################################################################################


# Test saving a simple dictionary to JSON and verifying file contents
def test_save_simple_dict(tmp_path):
    data = {"a": 1, "b": [2, 3]}
    out_file = tmp_path / "test1.json"
    save_variable_to_json(data, out_file)  # Save dictionary to JSON file
    
    # Open the saved JSON file and load its contents
    with open(out_file) as f:
        loaded = json.load(f)
    
    # Assert the loaded data matches the original dictionary exactly
    assert loaded == data

# Test saving a NumPy array to JSON and verifying file contents
def test_save_numpy_array(tmp_path):
    arr = np.array([[1, 2], [3, 4]])
    out_file = tmp_path / "test2.json"
    save_variable_to_json(arr, out_file)  # Save numpy array to JSON file
    
    # Load JSON file back as Python list of lists
    with open(out_file) as f:
        loaded = json.load(f)
    
    # Check that numpy array was correctly converted to nested lists in JSON
    assert loaded == [[1, 2], [3, 4]]

# Test that saving to a non-.json file raises a ValueError
def test_invalid_extension(tmp_path):
    out_file = tmp_path / "not_json.txt"
    try:
        # Attempt to save with an invalid file extension (should fail)
        save_variable_to_json({"x": 1}, out_file)
    except ValueError as e:
        # Verify error message contains correct info about file extension requirement
        assert "must have a .json extension" in str(e)
    else:
        # Fail the test if no exception was raised
        assert False, "Expected ValueError for wrong extension"

# Test that serialization failure raises a TypeError (monkeypatch convert_to_serializable)
def test_serialization_failure(tmp_path, monkeypatch):
    # Define a replacement function that always raises TypeError to simulate failure
    def fail_serialization(obj):
        raise TypeError("Cannot serialize")
    
    # Monkeypatch the convert_to_serializable function to simulate serialization failure
    monkeypatch.setattr("Hydrological_model_validator.Processing.Data_saver.convert_to_serializable", fail_serialization)

    out_file = tmp_path / "fail.json"
    try:
        # Try saving a valid dict, expecting a serialization failure due to monkeypatch
        save_variable_to_json({"x": 1}, out_file)
    except TypeError as e:
        # Confirm that TypeError was raised and contains expected message
        assert "Cannot serialize" in str(e)
    else:
        # Fail the test if no exception was raised
        assert False, "Expected TypeError from serialization failure"
