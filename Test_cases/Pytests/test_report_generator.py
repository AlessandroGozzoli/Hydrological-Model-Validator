import pytest
from unittest.mock import patch
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from PIL import Image
import re
from datetime import datetime

from Hydrological_model_validator.Report_generator import generate_full_report

################################################################################
# ---------- Auxilliary and dummy data ----------
################################################################################

# Dummy time series index
ts_index = pd.date_range("2000-01-01", periods=365, freq="D")

# Dummy 3D spatial data (time, lat, lon)
times = ts_index
y_coords = [0, 1]
x_coords = [0, 1]
spatial_data = xr.Dataset(
    {
        "var": (("time", "lat", "lon"), np.random.rand(len(times), len(y_coords), len(x_coords)))
    },
    coords={
        "time": times,
        "lat": y_coords,
        "lon": x_coords
    }
)

# Dummy time series data
obs_ts_df = pd.DataFrame({"value": np.random.rand(365)}, index=ts_index)
sim_ts_df = pd.DataFrame({"value": np.random.rand(365)}, index=ts_index)

# Dummy mask variables
# Mask selects 3 out of 4 points (True) - to avoid zero selected points issue
tmask_values = np.array([[[1, 1],
                          [1, 0]]], dtype=np.int8)  # shape (1, 2, 2)

lat_values = np.array([[45.0, 45.1],
                       [45.2, 45.3]])
lon_values = np.array([[7.0, 7.1],
                       [7.2, 7.3]])

# Series of mocking values to be used with input prompts
def input_mock(prompt):
    if "start date" in prompt.lower():
        return "2000-01-01"
    elif "variable" in prompt.lower():
        return "var"  # must match dummy spatial_data variable name
    elif "unit" in prompt.lower():
        return "unit"
    elif "method" in prompt.lower():
        return "1"
    elif "grid shape" in prompt.lower():
        return "(2, 2)"  # Example shape (lat, lon)
    elif "correction factor" in prompt.lower():
        return "0.0"  # epsilon
    elif "starting longitude" in prompt.lower():
        return "10.0"
    elif "x-step" in prompt.lower():
        return "0.5"
    elif "starting latitude" in prompt.lower():
        return "45.0"
    elif "y-step" in prompt.lower():
        return "0.5"
    elif "save output" in prompt.lower() or "default report" in prompt.lower():
        return "y"
    elif "data frequency" in prompt.lower() or "frequency" in prompt.lower():
        return "D"
    elif "exact filename to use for observed spatial data" in prompt.lower():
        return "obs_spatial.nc"
    elif "exact filename to use for simulated spatial data" in prompt.lower():
        return "simulated_field.nc"
    elif "exact filename to use for observed timeseries" in prompt.lower():
        return "observed_timeseries.nc"
    elif "exact filename to use for simulated timeseries" in prompt.lower():
        return "model_timeseries.nc"
    elif "exact filename to use for mask" in prompt.lower():
        return "ocean_mask.nc"
    else:
        raise ValueError(f"Unhandled prompt: {prompt}")
        
################################################################################
# ---------- Tests for generate_full_report ----------
################################################################################

# Check that generate_full_report can load data from file paths, create a PDF report, and save it correctly
def test_generate_full_report_with_file_paths(tmp_path):
    # Write example spatial data to NetCDF files for observed and simulated datasets
    obs_spatial_path = tmp_path / "obs_spatial.nc"
    sim_spatial_path = tmp_path / "sim_spatial.nc"
    spatial_data.to_netcdf(obs_spatial_path)  # Save observed spatial data
    spatial_data.to_netcdf(sim_spatial_path)  # Save simulated spatial data

    # Create a NetCDF mask file with the expected dimensions and variables (3D mask with lat/lon)
    mask_path = tmp_path / "mask.nc"
    with Dataset(mask_path, "w", format="NETCDF4") as ds_mask:
        ds_mask.createDimension("z", 1)  # depth dimension (single layer)
        ds_mask.createDimension("y", 2)  # latitude dimension
        ds_mask.createDimension("x", 2)  # longitude dimension

        tmask = ds_mask.createVariable("tmask", "i1", ("z", "y", "x"))  # mask variable
        nav_lat = ds_mask.createVariable("nav_lat", "f4", ("y", "x"))  # latitudes
        nav_lon = ds_mask.createVariable("nav_lon", "f4", ("y", "x"))  # longitudes

        tmask[:, :, :] = tmask_values  # Fill mask data (predefined values)
        nav_lat[:, :] = lat_values     # Fill latitude data
        nav_lon[:, :] = lon_values     # Fill longitude data

    # Convert time series DataFrames to xarray and save as NetCDF for obs and sim
    obs_ts_path = tmp_path / "obs_ts.nc"
    sim_ts_path = tmp_path / "sim_ts.nc"
    obs_ts_df.rename_axis("time").to_xarray().to_netcdf(obs_ts_path)  # Observed TS
    sim_ts_df.rename_axis("time").to_xarray().to_netcdf(sim_ts_path)  # Simulated TS

    # Dictionary with file paths for observed and simulated spatial & time series data plus mask
    data_folder = {
        "obs_spatial": str(obs_spatial_path),
        "sim_spatial": str(sim_spatial_path),
        "obs_ts": str(obs_ts_path),
        "sim_ts": str(sim_ts_path),
        "mask": str(mask_path),
    }

    # Dummy image function to mock PIL.Image.open (avoid actual file reads)
    def dummy_image(*args, **kwargs):
        return Image.new("RGB", (100, 100), (0, 0, 0))

    # Dummy draw function to mock PDF canvas drawing method (no-op)
    def dummy_draw_image(self, *args, **kwargs):
        pass  # Avoid errors in PDF image drawing during test

    # Patch image open, PDF draw, and input confirmation to avoid user interaction
    with patch("PIL.Image.open", side_effect=dummy_image), \
         patch("reportlab.pdfgen.canvas.Canvas.drawImage", new=dummy_draw_image), \
         patch("builtins.input", side_effect=input_mock):  # Force 'yes' or default input
        # Call report generation with paths as inputs, disable verbosity for test clarity
        generate_full_report(
            data_folder,
            output_dir=tmp_path,  # Output to test folder to capture generated files
            generate_pdf=True,    # Enable PDF report generation
            verbose=False,
            variable="var"        # Specify variable name to appear in report filename
        )

    # Construct expected output run folder based on today's date, format run_YYYY-MM-DD
    today_str = datetime.now().strftime("run_%Y-%m-%d")
    output_run_folder = tmp_path / today_str

    # Verify that PDF report files were generated in the expected folder
    pdf_files = list(output_run_folder.glob("Report_*.pdf"))
    assert pdf_files, f"No PDF report generated in {output_run_folder}."

    # Check that at least one PDF filename contains the variable name and correct timestamp pattern
    assert any(
        re.match(r"Report_var_run_\d{4}-\d{2}-\d{2}\.pdf", p.name) for p in pdf_files
    ), "PDF report filename does not contain expected variable and timestamp pattern."
   
    
# Verify that generate_full_report raises an error if a required input key ("obs_ts") is missing
def test_generate_full_report_missing_key():
    # Prepare inputs dictionary missing the "obs_ts" key, which is required for report generation
    inputs = {
        "obs_spatial": xr.Dataset(),  # Dummy empty xarray dataset for observed spatial data
        "sim_spatial": xr.Dataset(),  # Dummy empty xarray dataset for simulated spatial data
        # "obs_ts" key intentionally omitted to test error handling
        "sim_ts": pd.DataFrame(),     # Dummy empty DataFrame for simulated time series
        "mask": {"mask": np.ones((2, 2))},  # Simple mask as numpy array inside dict
    }

    # Expect generate_full_report to raise ValueError due to missing required key
    with pytest.raises(ValueError):
        generate_full_report(inputs)

        
# Test that generate_full_report raises ValueError when observed and simulated time series have non-overlapping time indices
def test_generate_full_report_mismatched_timeseries_index(tmp_path):
    # Save spatial NetCDF files for observed and simulated data (same data for simplicity)
    spatial_data.to_netcdf(tmp_path / "obs_spatial.nc")
    spatial_data.to_netcdf(tmp_path / "sim_spatial.nc")

    # Create and save mask NetCDF with required 3D structure and lat/lon coordinates
    mask_path = tmp_path / "mask.nc"
    with Dataset(mask_path, "w", format="NETCDF4") as ds_mask:
        ds_mask.createDimension("z", 1)  # Depth dimension (single layer)
        ds_mask.createDimension("y", 2)  # Latitude dimension
        ds_mask.createDimension("x", 2)  # Longitude dimension

        tmask = ds_mask.createVariable("tmask", "i1", ("z", "y", "x"))  # Mask variable
        nav_lat = ds_mask.createVariable("nav_lat", "f4", ("y", "x"))  # Latitude coordinates
        nav_lon = ds_mask.createVariable("nav_lon", "f4", ("y", "x"))  # Longitude coordinates

        # Fill variables with predefined values
        tmask[:, :, :] = tmask_values
        nav_lat[:, :] = lat_values
        nav_lon[:, :] = lon_values

    # Create observed and simulated time series DataFrames with mismatched time indices
    obs_ts = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.date_range("2000-01-01", periods=2))
    sim_ts = pd.DataFrame({"value": [1.5, 2.5]}, index=pd.date_range("2001-01-01", periods=2))  # Different year

    # Save time series as NetCDF files
    obs_ts_path = tmp_path / "obs_ts.nc"
    sim_ts_path = tmp_path / "sim_ts.nc"
    obs_ts.rename_axis("time").to_xarray().to_netcdf(obs_ts_path)
    sim_ts.rename_axis("time").to_xarray().to_netcdf(sim_ts_path)

    # Build input dictionary with paths to spatial, time series, and mask files
    inputs = {
        "obs_spatial": str(tmp_path / "obs_spatial.nc"),
        "sim_spatial": str(tmp_path / "sim_spatial.nc"),
        "obs_ts": str(obs_ts_path),
        "sim_ts": str(sim_ts_path),
        "mask": str(mask_path),
    }

    # Expect generate_full_report to raise ValueError due to no overlapping time indices in time series
    with pytest.raises(ValueError):
        generate_full_report(inputs)


# Ensure generate_full_report raises ValueError when called with an empty input dictionary
def test_generate_full_report_empty_input():
    # Pass an empty dictionary which lacks all required inputs
    with pytest.raises(ValueError):
        generate_full_report({})


# Check that generate_full_report raises an error when input data types are incorrect or unexpected
def test_generate_full_report_incorrect_data_types():
    # Inputs contain invalid types instead of expected xarray.Dataset, pd.DataFrame, or mask structure
    inputs = {
        "obs_spatial": "not a dataset",      # Should be xarray.Dataset, given string instead
        "sim_spatial": 42,                    # Should be xarray.Dataset, given int instead
        "obs_ts": "string not dataframe",    # Should be pd.DataFrame or path, given string
        "sim_ts": None,                      # Should be pd.DataFrame or path, given None
        "mask": "mask",                      # Should be dict or dataset/array, given string
    }

    # Expect ValueError or TypeError when processing invalid input types
    with pytest.raises((ValueError, TypeError)):
        generate_full_report(inputs)
        
        
# Ensure generate_full_report raises ValueError when spatial data has unexpected dimensions that prevent transposing to the required format
def test_generate_full_report_transpose_failure(tmp_path):
    # Create a spatial dataset with incompatible dimensions ("lat", "lon", "depth")
    # instead of the expected ("time", "lat", "lon"), which will trigger a transpose failure
    spatial_data_bad_dims = xr.Dataset(
        {
            "var": (("lat", "lon", "depth"), np.random.rand(2, 2, 3))
        },
        coords={
            "lat": [0, 1],
            "lon": [0, 1],
            "depth": [0, 1, 2]
        }
    )
    obs_spatial_path = tmp_path / "obs_spatial_bad.nc"
    spatial_data_bad_dims.to_netcdf(obs_spatial_path)

    # Create a valid mask NetCDF file with expected structure and dimensions
    mask_path = tmp_path / "mask.nc"
    with Dataset(mask_path, "w", format="NETCDF4") as ds_mask:
        ds_mask.createDimension("z", 1)
        ds_mask.createDimension("y", 2)
        ds_mask.createDimension("x", 2)

        tmask = ds_mask.createVariable("tmask", "i1", ("z", "y", "x"))
        nav_lat = ds_mask.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon = ds_mask.createVariable("nav_lon", "f4", ("y", "x"))

        # Fill mask and coordinate values
        tmask[:, :, :] = np.array([[[1, 1], [1, 0]]], dtype=np.int8)
        nav_lat[:, :] = np.array([[45.0, 45.1], [45.2, 45.3]])
        nav_lon[:, :] = np.array([[7.0, 7.1], [7.2, 7.3]])

    # Create dummy observed and simulated time series with matching time indices
    ts_index = pd.date_range("2000-01-01", periods=10, freq="D")
    obs_ts = pd.DataFrame({"value": np.random.rand(10)}, index=ts_index)
    sim_ts = pd.DataFrame({"value": np.random.rand(10)}, index=ts_index)

    # Save time series to NetCDF
    obs_ts_path = tmp_path / "obs_ts.nc"
    sim_ts_path = tmp_path / "sim_ts.nc"
    obs_ts.rename_axis("time").to_xarray().to_netcdf(obs_ts_path)
    sim_ts.rename_axis("time").to_xarray().to_netcdf(sim_ts_path)

    # Prepare input dictionary using the spatial file with bad dimensions
    data_folder = {
        "obs_spatial": str(obs_spatial_path),
        "sim_spatial": str(obs_spatial_path),  # Reuse the same faulty spatial file for both inputs
        "obs_ts": str(obs_ts_path),
        "sim_ts": str(sim_ts_path),
        "mask": str(mask_path),
    }

    # Expect ValueError because the spatial data dimensions cannot be transposed to the expected layout
    with pytest.raises(ValueError, match=r"Cannot transpose"):
        generate_full_report(
            data_folder,
            output_dir=tmp_path,
            generate_pdf=False,
            verbose=True,
            variable="var"
        )

    
# Test that generate_full_report uses default output directory when output_dir=None and input confirmation is patched to 'yes'
def test_generate_full_report_uses_default_output_dir_on_yes(tmp_path, monkeypatch):
    # Save dummy spatial data to NetCDF files for observed and simulated inputs
    obs_spatial_path = tmp_path / "obs_spatial.nc"
    sim_spatial_path = tmp_path / "sim_spatial.nc"
    spatial_data.to_netcdf(obs_spatial_path)
    spatial_data.to_netcdf(sim_spatial_path)

    # Create a valid NetCDF mask with expected variables and dimensions
    mask_path = tmp_path / "mask.nc"
    with Dataset(mask_path, "w", format="NETCDF4") as ds_mask:
        ds_mask.createDimension("z", 1)
        ds_mask.createDimension("y", 2)
        ds_mask.createDimension("x", 2)

        tmask = ds_mask.createVariable("tmask", "i1", ("z", "y", "x"))
        nav_lat = ds_mask.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon = ds_mask.createVariable("nav_lon", "f4", ("y", "x"))

        tmask[:, :, :] = tmask_values
        nav_lat[:, :] = lat_values
        nav_lon[:, :] = lon_values

    # Convert and save observed and simulated time series to NetCDF
    obs_ts_path = tmp_path / "obs_ts.nc"
    sim_ts_path = tmp_path / "sim_ts.nc"
    obs_ts_df.rename_axis("time").to_xarray().to_netcdf(obs_ts_path)
    sim_ts_df.rename_axis("time").to_xarray().to_netcdf(sim_ts_path)

    # Prepare inputs dictionary with file paths
    data_folder = {
        "obs_spatial": str(obs_spatial_path),
        "sim_spatial": str(sim_spatial_path),
        "obs_ts": str(obs_ts_path),
        "sim_ts": str(sim_ts_path),
        "mask": str(mask_path),
    }

    # Patch input to simulate user confirmation as 'yes' (or defaulting to yes)
    monkeypatch.setattr("builtins.input", input_mock)

    # Run the report generation, letting it use the default output directory
    generate_full_report(
        data_folder,
        output_dir=None,       # No output_dir passed; triggers default usage
        generate_pdf=False,    # Disable PDF to speed up test
        verbose=False,
        variable="var",
    )

    # Assert the default output directory 'REPORT' was created
    default_outdir = tmp_path / "REPORT"
    assert default_outdir.exists()


# Test that generate_full_report prompts for and uses a custom output directory when user declines the default
def test_generate_full_report_asks_for_custom_output_dir_on_no(tmp_path, monkeypatch):
    # Define file paths
    obs_spatial_path = tmp_path / "obs_spatial.nc"
    sim_spatial_path = tmp_path / "sim_spatial.nc"
    mask_path = tmp_path / "mask.nc"
    obs_ts_path = tmp_path / "obs_ts.nc"
    sim_ts_path = tmp_path / "sim_ts.nc"
    custom_output_path = tmp_path / "my_custom_report_folder"

    # Save dummy spatial data to NetCDF files
    spatial_data.to_netcdf(obs_spatial_path)
    spatial_data.to_netcdf(sim_spatial_path)

    # Create dummy mask NetCDF with expected structure
    with Dataset(mask_path, "w", format="NETCDF4") as ds_mask:
        ds_mask.createDimension("z", 1)
        ds_mask.createDimension("y", 2)
        ds_mask.createDimension("x", 2)

        tmask = ds_mask.createVariable("tmask", "i1", ("z", "y", "x"))
        nav_lat = ds_mask.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon = ds_mask.createVariable("nav_lon", "f4", ("y", "x"))

        tmask[:, :, :] = tmask_values
        nav_lat[:, :] = lat_values
        nav_lon[:, :] = lon_values

    # Save dummy time series to NetCDF
    obs_ts_df.rename_axis("time").to_xarray().to_netcdf(obs_ts_path)
    sim_ts_df.rename_axis("time").to_xarray().to_netcdf(sim_ts_path)

    # Prepare input dictionary for report generation
    data_folder = {
        "obs_spatial": str(obs_spatial_path),
        "sim_spatial": str(sim_spatial_path),
        "obs_ts": str(obs_ts_path),
        "sim_ts": str(sim_ts_path),
        "mask": str(mask_path),
    }

    # Simulate user input responses to prompts
    def input_mock_no_folder(prompt):
        if "start date" in prompt.lower():
            return "2000-01-01"
        elif "variable" in prompt.lower():
            return "var"
        elif "unit" in prompt.lower():
            return "unit"
        elif "method" in prompt.lower():
            return "1"
        elif "grid shape" in prompt.lower():
            return "(2, 2)"
        elif "correction factor" in prompt.lower():
            return "0.0"
        elif "starting longitude" in prompt.lower():
            return "10.0"
        elif "x-step" in prompt.lower():
            return "0.5"
        elif "starting latitude" in prompt.lower():
            return "45.0"
        elif "y-step" in prompt.lower():
            return "0.5"
        elif "save output" in prompt.lower() or "default report" in prompt.lower():
            return "n"  # Choose not to use default output folder
        elif "base output directory path" in prompt.lower():
            return str(custom_output_path)  # Provide a custom directory
        elif "data frequency" in prompt.lower():
            return "D"
        else:
            raise ValueError(f"Unhandled prompt: {prompt}")

    # Patch built-in input to simulate interaction
    monkeypatch.setattr("builtins.input", input_mock_no_folder)

    # Run the report generator
    generate_full_report(
        data_folder,
        output_dir=None,       # Trigger prompt for custom output directory
        generate_pdf=False,    # Skip PDF to speed up test
        verbose=False,
        variable="var",
    )

    # Validate that the custom directory was created
    assert custom_output_path.exists(), "Custom output directory was not created"
    
    
def test_generate_full_report_with_string_folder_path(tmp_path):
    # Create fake spatial and time series files with keyword-friendly names ----
    obs_spatial = tmp_path / "obs_spatial.nc"
    sim_spatial = tmp_path / "simulated_field.nc"
    obs_ts = tmp_path / "observed_timeseries.nc"
    sim_ts = tmp_path / "model_timeseries.nc"
    mask_file = tmp_path / "ocean_mask.nc"

    # Match time range to input_mock() start date ("2000-01-01")
    time_index = pd.date_range("2000-01-01", periods=365)

    dummy_array = xr.DataArray(
        np.random.rand(365, 2, 2),
        dims=["time", "lat", "lon"],
        coords={"time": time_index, "lat": [0, 1], "lon": [0, 1]}
    )
    dummy_ts = pd.DataFrame({"value": np.random.rand(365)}, index=time_index)

    dummy_array.to_netcdf(obs_spatial)
    dummy_array.to_netcdf(sim_spatial)
    dummy_ts.rename_axis("time").to_xarray().to_netcdf(obs_ts)
    dummy_ts.rename_axis("time").to_xarray().to_netcdf(sim_ts)

    # Create a dummy mask file with required variables
    from netCDF4 import Dataset
    with Dataset(mask_file, "w", format="NETCDF4") as ds:
        ds.createDimension("z", 1)
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        tmask = ds.createVariable("tmask", "i1", ("z", "y", "x"))
        nav_lat = ds.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon = ds.createVariable("nav_lon", "f4", ("y", "x"))
        tmask[:, :, :] = [[[1, 0], [1, 1]]]
        nav_lat[:, :] = [[10, 11], [12, 13]]
        nav_lon[:, :] = [[20, 21], [22, 23]]

    # Patch input to simulate interactive responses
    with patch("builtins.input", side_effect=input_mock):
        generate_full_report(
            str(tmp_path),          # Path to folder with files
            output_dir=None,    # Where to save report
            generate_pdf=False,     # Skip PDF to speed up test
            verbose=False,
            variable="testvar"
        )

    # Sice nothing at the end should change
    # assert that a report folder was created
    default_outdir = tmp_path / "REPORT"
    assert default_outdir.exists()
    

def test_spatial_data_no_time_coordinate_triggers_prompt(tmp_path):
    # Create a NetCDF file with a 'time' dimension but NO 'time' coordinate
    no_time_coord_path = tmp_path / "obs_spatial.nc"
    with Dataset(no_time_coord_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 365)
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 2)
        var = ds.createVariable("temperature", "f4", ("time", "lat", "lon"))
        var[:] = np.random.rand(365, 2, 2)

    # Create other required files to avoid crashing during full report flow
    sim_path = tmp_path / "sim_spatial.nc"
    xr.DataArray(np.random.rand(365, 2, 2), dims=["time", "lat", "lon"],
                 coords={"time": pd.date_range("2001-01-01", periods=365),
                         "lat": [0, 1], "lon": [0, 1]}).to_dataset(name="sim").to_netcdf(sim_path)

    ts = pd.DataFrame({"val": np.random.rand(365)}, index=pd.date_range("2001-01-01", periods=365))
    ts.rename_axis("time").to_xarray().to_netcdf(tmp_path / "obs_ts.nc")
    ts.rename_axis("time").to_xarray().to_netcdf(tmp_path / "sim_ts.nc")

    mask_path = tmp_path / "mask.nc"
    with Dataset(mask_path, "w", format="NETCDF4") as ds:
        ds.createDimension("z", 1)
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        tmask = ds.createVariable("tmask", "i1", ("z", "y", "x"))
        nav_lat = ds.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon = ds.createVariable("nav_lon", "f4", ("y", "x"))
        tmask[:, :, :] = [[[1, 1], [1, 0]]]
        nav_lat[:, :] = [[10, 11], [12, 13]]
        nav_lon[:, :] = [[20, 21], [22, 23]]

    # Mock `prompt_for_datetime_index()` to return a fake DatetimeIndex
    fake_index = pd.date_range("2001-01-01", periods=365, freq="D")

    with patch("Hydrological_model_validator.Processing.time_utils.prompt_for_datetime_index", return_value=fake_index) as prompt_mock:
        # Patch input() with input_mock instead of a fixed return value
        with patch("builtins.input", side_effect=input_mock), \
             patch("PIL.Image.open"), \
             patch("reportlab.pdfgen.canvas.Canvas.drawImage"):

            generate_full_report(
                {
                    "obs_spatial": str(no_time_coord_path),
                    "sim_spatial": str(sim_path),
                    "obs_ts": str(tmp_path / "obs_ts.nc"),
                    "sim_ts": str(tmp_path / "sim_ts.nc"),
                    "mask": str(mask_path),
                },
                output_dir=tmp_path,
                generate_pdf=False,
                verbose=False,
                variable="var"
            )

    prompt_mock.assert_called_once_with(365)