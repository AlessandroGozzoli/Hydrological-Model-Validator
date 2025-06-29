import pytest
import numpy as np
import xarray as xr
import gzip
from unittest.mock import patch

# ====== EARLY MOCK TO FAKE matlab.engine BEFORE ANY IMPORTS ======
def ensure_matlab_engine_module():
    import sys
    import types

    # Create 'matlab' package
    if 'matlab' not in sys.modules:
        matlab_mod = types.ModuleType('matlab')
        matlab_mod.__path__ = []  # mark as package
        sys.modules['matlab'] = matlab_mod
    else:
        matlab_mod = sys.modules['matlab']
        if not hasattr(matlab_mod, '__path__'):
            matlab_mod.__path__ = []

    # Create 'matlab.engine' submodule
    if 'matlab.engine' not in sys.modules:
        engine_mod = types.ModuleType('matlab.engine')
        sys.modules['matlab.engine'] = engine_mod
    else:
        engine_mod = sys.modules['matlab.engine']

    # Link matlab.engine as an attribute of matlab
    setattr(matlab_mod, 'engine', engine_mod)

    # Add dummy function
    if not hasattr(engine_mod, 'start_matlab'):
        engine_mod.start_matlab = lambda: None

# Inject mock BEFORE importing the target module
ensure_matlab_engine_module()

from Hydrological_model_validator.Processing.file_io import (
    mask_reader,
    load_dataset,
    unzip_gz_to_file,
    read_nc_variable_from_unzipped_file,
    read_nc_variable_from_gz_in_memory,
    call_interpolator,
    select_3d_variable,
    find_file_with_keywords
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

# Test that a KeyError is raised if required variables are missing in the NetCDF file
def test_mask_reader_missing_required_vars(tmp_path):
    class DummyDataset:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        @property
        def variables(self):
            return {}  # no variables at all

    (tmp_path / "mesh_mask.nc").touch()  # Create empty mask file

    with patch("netCDF4.Dataset", return_value=DummyDataset()):
        with pytest.raises(KeyError):
            mask_reader(tmp_path)

# Test that ValueError is raised if 'tmask' variable has invalid dimensions (not 3D or 4D)
def test_mask_reader_tmask_invalid_dims(tmp_path):
    class DummyDataset:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        @property
        def variables(self):
            return {
                "tmask": np.zeros((2, 2)),  # Invalid shape, 2D only
                "nav_lat": np.zeros((1, 1)),
                "nav_lon": np.zeros((1, 1)),
            }

    (tmp_path / "mesh_mask.nc").touch()  # Create empty mask file

    with patch("netCDF4.Dataset", return_value=DummyDataset()):
        with pytest.raises(ValueError):
            mask_reader(tmp_path)

# Test that ValueError is raised if lat/lon shapes don't match surface mask shape
def test_mask_reader_lat_lon_shape_mismatch(tmp_path):
    class DummyDataset:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

        @property
        def variables(self):
            tmask_data = np.ones((1, 3, 4, 5))  # shape (time=1, depth=3, y=4, x=5)
            nav_lat = np.zeros((3, 4))       # mismatch y dimension
            nav_lon = np.zeros((3, 5))       # mismatch x dimension

            return {
                "tmask": tmask_data,
                "nav_lat": nav_lat,
                "nav_lon": nav_lon,
            }

    (tmp_path / "mesh_mask.nc").touch()  # Create empty mask file

    with patch("netCDF4.Dataset", return_value=DummyDataset()):
        with pytest.raises(ValueError):
            mask_reader(tmp_path)

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

# Test for input validation
def test_load_dataset_input_validation(tmp_path):
    # Valid directory path for other tests
    valid_dir = tmp_path

    # Invalid year type
    with pytest.raises(TypeError):
        load_dataset(3.14, valid_dir)  # float is invalid

    # Invalid IDIR type
    with pytest.raises(TypeError):
        load_dataset(2023, 12345)  # int is invalid

    # IDIR does not exist or not a directory
    non_exist_dir = tmp_path / "non_exist"
    with pytest.raises(ValueError):
        load_dataset(2023, non_exist_dir)  # path does not exist

    # Create a file instead of directory to simulate not a directory
    file_path = tmp_path / "not_a_dir"
    file_path.touch()
    with pytest.raises(ValueError):
        load_dataset(2023, file_path)  # path is file, not directory

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

def test_call_interpolator_starts_and_quits(monkeypatch):
    class DummyEngine:
        def addpath(self, *args, **kwargs): pass
        def Interpolator_v2(self, *args, **kwargs): pass
        def quit(self): pass

    def fake_start_matlab():
        return DummyEngine()

    monkeypatch.setattr("matlab.engine.start_matlab", fake_start_matlab)

    call_interpolator("var", 'l3', "input", "output", "maskfile")


def test_call_interpolator_start_fail(monkeypatch):
    def fake_start_matlab():
        raise RuntimeError("Cannot start")

    monkeypatch.setattr("matlab.engine.start_matlab", fake_start_matlab)

    with pytest.raises(RuntimeError, match="Failed to start MATLAB engine"):
        call_interpolator("var", 'l3', "input", "output", "maskfile")


def test_call_interpolator_function_fail(monkeypatch):
    class DummyEngine:
        def addpath(self, *args, **kwargs): pass
        def Interpolator_v2(self, *args, **kwargs): raise RuntimeError("Error in MATLAB")
        def quit(self): pass

    def fake_start_matlab():
        return DummyEngine()

    monkeypatch.setattr("matlab.engine.start_matlab", fake_start_matlab)

    with pytest.raises(RuntimeError, match="MATLAB interpolation failed"):
        call_interpolator("var", 'l3', "input", "output", "maskfile")
        
def test_call_interpolator_input_validation():
    valid_str = "varname"
    valid_lev = 'l3'
    valid_path_str = "/valid/path"

    # varname not string
    with pytest.raises(TypeError):
        call_interpolator(123, "0", valid_path_str, valid_path_str, valid_path_str)

    # data_level not string
    with pytest.raises(TypeError):
        call_interpolator(valid_str, 999, valid_path_str, valid_path_str, valid_path_str)

    # input_dir invalid type
    with pytest.raises(TypeError):
        call_interpolator(valid_str, valid_lev, 123, valid_path_str, valid_path_str)

    # output_dir invalid type
    with pytest.raises(TypeError):
        call_interpolator(valid_str, valid_lev, valid_path_str, 123, valid_path_str)

    # mask_file invalid type
    with pytest.raises(TypeError):
        call_interpolator(valid_str, valid_lev, valid_path_str, valid_path_str, 123)
        

###############################################################################
# --- find_file_with_keywords tests ---
###############################################################################


# Dummy file-like class to simulate files with a .name attribute
class DummyFile:
    def __init__(self, name):
        self.name = name

# Test that the function returns the only file matching keywords without prompting the user
def test_single_match_returns_file():
    files = [DummyFile("data_obs.nc"), DummyFile("data_sim.nc"), DummyFile("readme.txt")]
    keywords = ["obs", "observed"]
    # Only "data_obs.nc" contains 'obs' → function should return it directly
    result = find_file_with_keywords(files, keywords, "observed data")
    assert result.name == "data_obs.nc"

# Test that when multiple files match keywords,
# the function prompts the user and returns the file matching user's exact input
@patch("builtins.input", return_value="data_obs.nc")
@patch("builtins.print")
def test_multiple_matches_user_selects_valid(mock_print, mock_input):
    files = [
        DummyFile("data_obs.nc"),
        DummyFile("data_observed_2023.nc"),
        DummyFile("data_sim.nc"),
    ]
    keywords = ["obs", "observed"]
    # Two files match: "data_obs.nc", "data_observed_2023.nc"
    # User inputs "data_obs.nc", so it should be returned
    result = find_file_with_keywords(files, keywords, "observed data")
    
    # Confirm the multiple matches prompt was printed
    mock_print.assert_called()
    assert result.name == "data_obs.nc"

# Test that when multiple files match but user inputs a filename not in matches,
# the function raises FileNotFoundError
@patch("builtins.input", return_value="not_existing_file.nc")
@patch("builtins.print")
def test_multiple_matches_user_selects_invalid_raises(mock_print, mock_input):
    files = [
        DummyFile("data_obs.nc"),
        DummyFile("data_observed_2023.nc"),
        DummyFile("data_sim.nc"),
    ]
    keywords = ["obs", "observed"]
    # User input filename not in matched files → should raise error
    with pytest.raises(FileNotFoundError):
        find_file_with_keywords(files, keywords, "observed data")

# Test that if no files match keywords,
# user is prompted and entering a valid filename from full file list returns that file
@patch("builtins.input", return_value="data_sim.nc")
@patch("builtins.print")
def test_no_matches_user_selects_valid(mock_print, mock_input):
    files = [DummyFile("data_obs.nc"), DummyFile("data_sim.nc"), DummyFile("readme.txt")]
    keywords = ["nonexistentkeyword"]  # no matches expected

    # User inputs "data_sim.nc" which exists → should be returned
    result = find_file_with_keywords(files, keywords, "some data")

    # input() should be called with the correct prompt string (verify prompt includes "No files matching keywords")
    mock_input.assert_called()
    prompt_arg = mock_input.call_args[0][0]
    assert "No files matching keywords" in prompt_arg

    assert result.name == "data_sim.nc"

# Test that if no files match keywords,
# user inputs a filename not in the full list and function raises FileNotFoundError
@patch("builtins.input", return_value="not_existing_file.nc")
@patch("builtins.print")
def test_no_matches_user_selects_invalid_raises(mock_print, mock_input):
    files = [DummyFile("data_obs.nc"), DummyFile("data_sim.nc"), DummyFile("readme.txt")]
    keywords = ["nonexistentkeyword"]  # no matches expected
    
    # User inputs invalid filename → should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        find_file_with_keywords(files, keywords, "some data")
        

###############################################################################
# --- find_file_with_keywords tests ---
###############################################################################


# Helper to create an xarray Dataset with variables of specified names and shapes
# Each variable contains zeros and dims are named dim0, dim1, ...
def create_dataset_with_vars(vars_dims: dict[str, tuple[int, ...]]) -> xr.Dataset:
    data_vars = {}
    for varname, shape in vars_dims.items():
        # Create unique dim names per variable to avoid conflicts
        dims = tuple(f"{varname}_dim{i}" for i in range(len(shape)))
        data = np.zeros(shape)
        data_vars[varname] = (dims, data)
    return xr.Dataset(data_vars)

# Test that ValueError is raised when no 3D variables are present
def test_no_3d_variables_raises():
    # Create dataset with a 2D and a 4D variable (but distinct dims for each)
    ds = create_dataset_with_vars({"var2d": (5, 5), "var4d": (2, 3, 4, 5)})
    with pytest.raises(ValueError):
        select_3d_variable(ds, "test dataset")

# Test that when exactly one 3D variable is present, it is returned directly
def test_single_3d_variable_returns_it():
    ds = create_dataset_with_vars({"var3d": (2, 3, 4), "var2d": (5, 5)})
    result = select_3d_variable(ds, "Test dataset")
    assert result.name == "var3d"
    assert result.ndim == 3

# Test that when multiple 3D variables are present, user input selects the correct variable
@patch("builtins.input", return_value="1")
@patch("builtins.print")
def test_multiple_3d_variables_user_selects_valid(mock_print, mock_input):
    ds = create_dataset_with_vars({"varA": (2, 3, 4), "varB": (5, 6, 7), "var2d": (5, 5)})
    result = select_3d_variable(ds, "Test dataset")

    mock_print.assert_any_call("⚠️ Multiple 3D variables found in Test dataset: ['varA', 'varB']")
    mock_print.assert_any_call("1: varA (shape: (2, 3, 4))")
    mock_print.assert_any_call("2: varB (shape: (5, 6, 7))")

    assert result.name == "varA"
    assert result.ndim == 3

# Test that invalid numeric input triggers prompt again, then valid selection is accepted
@patch("builtins.input", side_effect=["10", "2"])
@patch("builtins.print")
def test_multiple_3d_variables_invalid_then_valid_selection(mock_print, mock_input):
    ds = create_dataset_with_vars({"varA": (2, 3, 4), "varB": (5, 6, 7)})
    result = select_3d_variable(ds, "Test dataset")

    mock_print.assert_any_call("Invalid selection.")
    assert result.name == "varB"

# Test that non-integer input triggers prompt again, then valid numeric selection is accepted
@patch("builtins.input", side_effect=["notanumber", "1"])
@patch("builtins.print")
def test_multiple_3d_variables_nonint_then_valid_selection(mock_print, mock_input):
    ds = create_dataset_with_vars({"varA": (2, 3, 4), "varB": (5, 6, 7)})
    result = select_3d_variable(ds, "Test dataset")

    mock_print.assert_any_call("Please enter a valid number.")
    assert result.name == "varA"
