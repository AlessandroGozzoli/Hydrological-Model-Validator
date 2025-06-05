import numpy as np
import xarray as xr
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
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


# JSON-native types
assert convert_to_serializable("text") == "text"
assert convert_to_serializable(42.5) == 42.5

# Iterable types
assert convert_to_serializable([1, 2, 3]) == [1, 2, 3]
assert convert_to_serializable((4, 5)) == [4, 5]  # tuple converted to list

# Dictionary
assert convert_to_serializable({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}
assert convert_to_serializable({1: "x", 2: "y"}) == {"1": "x", "2": "y"}  # keys to str

# NumPy arrays
assert convert_to_serializable(np.array([1, 2, 3])) == [1, 2, 3]
assert convert_to_serializable(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]

# pandas DataFrame
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
assert convert_to_serializable(df) == [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
assert isinstance(json.dumps(convert_to_serializable(df)), str)

# pandas Series
s = pd.Series([10, 20], index=["x", "y"])
assert convert_to_serializable(s) == {"x": 10, "y": 20}
assert isinstance(json.dumps(convert_to_serializable(s)), str)

# xarray DataArray
da = xr.DataArray(np.array([1, 2]), dims="x", coords={"x": [10, 20]})
assert convert_to_serializable(da) == {"dims": ('x',), "coords": {"x": [10, 20]}, "data": [1, 2]}
assert isinstance(json.dumps(convert_to_serializable(da)), str)

# xarray Dataset
ds = xr.Dataset({"temp": (("x",), [1, 2])}, coords={"x": [10, 20]})
assert isinstance(convert_to_serializable(ds), dict)
assert isinstance(json.dumps(convert_to_serializable(ds)), str)

# Object with to_dict()
class Dummy:
    def to_dict(self): 
        return {"key": "value"}

dummy = Dummy()
assert convert_to_serializable(dummy) == {"key": "value"}
assert isinstance(json.dumps(convert_to_serializable(dummy)), str)

# Unsupported type fallback
assert "function" in convert_to_serializable(lambda x: x)
assert isinstance(convert_to_serializable(object()), str)


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
