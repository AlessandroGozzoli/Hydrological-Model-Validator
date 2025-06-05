import pytest
import numpy as np
import xarray as xr
import gzip

from Hydrological_model_validator.Processing.file_io import (
    mask_reader,
    load_dataset,
    unzip_gz_to_file,
    read_nc_variable_from_unzipped_file,
    read_nc_variable_from_gz_in_memory,
    call_interpolator,
)

###############################################################################
# --- mask_reader tests ---
###############################################################################


# Test that a correctly formatted mesh_mask.nc file is read successfully and returns valid outputs
def test_mask_reader_success(tmp_path):
    from netCDF4 import Dataset

    file_path = tmp_path / "mesh_mask.nc"
    with Dataset(file_path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("depth", 2)
        ds.createDimension("y", 4)
        ds.createDimension("x", 5)

        # Create a mask variable filled with ones (valid ocean points)
        tmask = ds.createVariable("tmask", "i4", ("time", "depth", "y", "x"))
        tmask[0, :, :, :] = np.ones((2, 4, 5), dtype=int)

        # Set a single point to zero to simulate a land or invalid point in the mask
        tmask[0, 0, 1, 1] = 0

        # Create latitude and longitude variables
        lat = ds.createVariable("nav_lat", "f4", ("y", "x"))
        lon = ds.createVariable("nav_lon", "f4", ("y", "x"))

        # Fill lat/lon with sequential values to verify coordinate shapes and alignment
        lat[:, :] = np.arange(20).reshape((4, 5))
        lon[:, :] = np.arange(20).reshape((4, 5))

    # Read the mask and coordinates from the file using the reader function
    Mmask, Mfsm, Mfsm_3d, Mlat, Mlon = mask_reader(tmp_path)

    # Verify mask has expected spatial shape (y, x)
    assert Mmask.shape == (4, 5)

    # Check fractional masks are returned as tuples (mask, inverse mask or weight)
    assert isinstance(Mfsm, tuple)
    assert isinstance(Mfsm_3d, tuple)

    # Ensure lat/lon coordinate arrays match the spatial dimensions of the mask
    assert Mlat.shape == Mmask.shape
    assert Mlon.shape == Mmask.shape

# Test that a FileNotFoundError is raised when the expected mesh_mask.nc file is missing
def test_mask_reader_missing_file(tmp_path):
    # Expect failure due to missing file
    with pytest.raises(FileNotFoundError):
        mask_reader(tmp_path)

# Test that a TypeError is raised when an invalid (non-path-like) argument is passed
def test_mask_reader_wrong_type():
    # Expect failure due to invalid input type
    with pytest.raises(TypeError):
        mask_reader(123)


###############################################################################
# --- load_dataset tests ---
###############################################################################


# Test that load_dataset successfully loads a NetCDF file for the given year
def test_load_dataset_success(tmp_path):
    year = 2020
    # Create a dummy NetCDF file with minimal content
    ds_file = tmp_path / f"Msst_{year}.nc"
    data = xr.Dataset({"var": ("x", np.arange(5))})
    data.to_netcdf(ds_file)

    # Attempt to load the dataset using the function
    y, ds = load_dataset(year, tmp_path)

    # Check that the year is returned correctly and a Dataset is loaded
    assert y == year
    assert isinstance(ds, xr.Dataset)
    ds.close()

# Test that load_dataset returns None if the file for the given year does not exist
def test_load_dataset_file_not_found(tmp_path):
    year = 1999
    # File for this year does not exist; function should return None for the dataset
    y, ds = load_dataset(year, tmp_path)
    assert y == year
    assert ds is None

# Test that load_dataset raises a ValueError if the provided path is not a directory
def test_load_dataset_bad_dir(tmp_path):
    fake_dir = tmp_path / "notadir"
    # Simulate an invalid path: create a file where a directory is expected
    fake_dir.write_text("not a dir")

    # Function should raise ValueError when a non-directory path is given
    with pytest.raises(ValueError):
        load_dataset(2020, fake_dir)


###############################################################################
# --- unzip_gz_to_file tests ---
###############################################################################


# Test that a .gz file is successfully unzipped and contents match the original
def test_unzip_gz_to_file_success(tmp_path):
    content = b"test content"
    gz_path = tmp_path / "file.txt.gz"
    out_path = tmp_path / "file.txt"

    # Write compressed content to the .gz file
    with gzip.open(gz_path, "wb") as f:
        f.write(content)

    # Attempt to unzip the file to the specified output path
    unzip_gz_to_file(gz_path, out_path)

    # Confirm the output file was created
    assert out_path.exists()

    # Confirm the uncompressed content matches the original
    with open(out_path, "rb") as f:
        assert f.read() == content

# Test that attempting to unzip a non-existent .gz file raises FileNotFoundError
def test_unzip_gz_to_file_missing_file(tmp_path):
    gz_path = tmp_path / "missing.gz"
    out_path = tmp_path / "file.txt"

    # Function should raise FileNotFoundError for a missing input file
    with pytest.raises(FileNotFoundError):
        unzip_gz_to_file(gz_path, out_path)

# Test that unzip_gz_to_file creates parent directories if they don't exist
def test_unzip_gz_to_file_creates_parent_dir(tmp_path):
    content = b"content"
    gz_path = tmp_path / "file.gz"
    out_dir = tmp_path / "nested/dir"
    out_path = out_dir / "file.txt"

    # Write compressed content to the .gz file
    with gzip.open(gz_path, "wb") as f:
        f.write(content)

    # Function should create missing directories before writing the output
    unzip_gz_to_file(gz_path, out_path)

    # Check that the file was successfully written in the nested directory
    assert out_path.exists()


###############################################################################
# --- read_nc_variable_from_unzipped_file tests ---
###############################################################################


# Test that a variable can be successfully read from an existing NetCDF file
def test_read_nc_variable_from_unzipped_file_success(tmp_path):
    from netCDF4 import Dataset

    nc_path = tmp_path / "test.nc"

    # Create a NetCDF file with one dimension and one variable to test reading
    with Dataset(nc_path, "w") as ds:
        ds.createDimension("x", 3)
        var = ds.createVariable("var1", "f4", ("x",))
        var[:] = np.array([1.0, 2.0, 3.0])

    # Read the variable from the created file and verify its contents match what was written
    data = read_nc_variable_from_unzipped_file(nc_path, "var1")
    np.testing.assert_array_equal(data, np.array([1.0, 2.0, 3.0]))

# Test that reading from a missing NetCDF file raises FileNotFoundError
def test_read_nc_variable_from_unzipped_file_missing_file(tmp_path):
    # Attempting to read a variable from a nonexistent file should raise an error
    with pytest.raises(FileNotFoundError):
        read_nc_variable_from_unzipped_file(tmp_path / "nofile.nc", "var1")

# Test that trying to read a non-existent variable raises KeyError
def test_read_nc_variable_from_unzipped_file_missing_var(tmp_path):
    from netCDF4 import Dataset

    nc_path = tmp_path / "test.nc"

    # Create an empty NetCDF file with a dimension but no variables
    with Dataset(nc_path, "w") as ds:
        ds.createDimension("x", 3)

    # Trying to read a variable not present in the file should raise KeyError
    with pytest.raises(KeyError):
        read_nc_variable_from_unzipped_file(nc_path, "missing_var")


###############################################################################
# --- read_nc_variable_from_gz_in_memory tests ---
###############################################################################


# Test reading a variable from an in-memory unzipped NetCDF file inside a gzip archive
def test_read_nc_variable_from_gz_in_memory_success(tmp_path):
    # Create a simple xarray Dataset and convert it to NetCDF bytes
    ds = xr.Dataset({"var1": ("x", [10, 20, 30])})
    nc_bytes = ds.to_netcdf()
    
    gz_path = tmp_path / "file.nc.gz"
    
    # Write the NetCDF bytes into a gzip compressed file
    with gzip.open(gz_path, "wb") as f:
        f.write(nc_bytes)
    
    # Read the variable from the gzip compressed NetCDF file in memory and check its values
    data = read_nc_variable_from_gz_in_memory(gz_path, "var1")
    np.testing.assert_array_equal(data, np.array([10, 20, 30]))

# Test that reading from a missing gzip file raises FileNotFoundError
def test_read_nc_variable_from_gz_in_memory_missing_file(tmp_path):
    # Expect a FileNotFoundError when the gzip file does not exist
    with pytest.raises(FileNotFoundError):
        read_nc_variable_from_gz_in_memory(tmp_path / "nofile.nc.gz", "var1")

# Test that trying to read a non-existent variable from gzipped NetCDF raises KeyError
def test_read_nc_variable_from_gz_in_memory_missing_var(tmp_path):
    # Create a NetCDF gzip file with a single variable
    ds = xr.Dataset({"var1": ("x", [1,2,3])})
    nc_bytes = ds.to_netcdf()
    gz_path = tmp_path / "file.nc.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(nc_bytes)
    
    # Attempt to read a variable that does not exist, expecting a KeyError
    with pytest.raises(KeyError):
        read_nc_variable_from_gz_in_memory(gz_path, "missing_var")


###############################################################################
# --- call_interpolator tests ---
###############################################################################

import sys
import types

def ensure_matlab_engine_module():
    # Create dummy matlab module
    if 'matlab' not in sys.modules:
        matlab_mod = types.ModuleType('matlab')
        sys.modules['matlab'] = matlab_mod
    else:
        matlab_mod = sys.modules['matlab']

    # Create dummy matlab.engine submodule
    if not hasattr(matlab_mod, 'engine'):
        engine_mod = types.ModuleType('matlab.engine')
        matlab_mod.engine = engine_mod
    else:
        engine_mod = matlab_mod.engine

    # Add dummy start_matlab to avoid AttributeError during monkeypatch
    if not hasattr(engine_mod, 'start_matlab'):
        engine_mod.start_matlab = lambda: None


# Test that call_interpolator starts the MATLAB engine and quits without errors under normal conditions
def test_call_interpolator_starts_and_quits(monkeypatch):
    ensure_matlab_engine_module()

    class DummyEngine:
        def addpath(self, *args, **kwargs): pass
        def Interpolator_v2(self, *args, **kwargs): pass
        def quit(self): pass

    def fake_start_matlab():
        return DummyEngine()

    monkeypatch.setattr("matlab.engine.start_matlab", fake_start_matlab)

    # Call your function under test
    call_interpolator("var", 1, "input", "output", "maskfile")

# Test that call_interpolator raises RuntimeError if MATLAB engine fails to start
def test_call_interpolator_start_fail(monkeypatch):
    ensure_matlab_engine_module()

    def fake_start_matlab():
        raise RuntimeError("Cannot start")

    monkeypatch.setattr("matlab.engine.start_matlab", fake_start_matlab)

    with pytest.raises(RuntimeError):
        call_interpolator("var", 1, "input", "output", "maskfile")

# Test that call_interpolator raises RuntimeError if Interpolator_v2 function inside MATLAB engine fails
def test_call_interpolator_function_fail(monkeypatch):
    ensure_matlab_engine_module()

    class DummyEngine:
        def addpath(self, *args, **kwargs): pass
        def Interpolator_v2(self, *args, **kwargs): raise RuntimeError("Error")
        def quit(self): pass

    def fake_start_matlab():
        return DummyEngine()

    monkeypatch.setattr("matlab.engine.start_matlab", fake_start_matlab)

    with pytest.raises(RuntimeError):
        call_interpolator("var", 1, "input", "output", "maskfile")
