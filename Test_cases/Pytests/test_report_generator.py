import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from PIL import Image
import re
from datetime import datetime

from Hydrological_model_validator.Report_generator import generate_full_report

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
    else:
        raise ValueError(f"Unhandled prompt: {prompt}")
        

# === Test ===

def test_generate_full_report_with_file_paths(tmp_path):
    # --- Save spatial NetCDFs ---
    obs_spatial_path = tmp_path / "obs_spatial.nc"
    sim_spatial_path = tmp_path / "sim_spatial.nc"
    spatial_data.to_netcdf(obs_spatial_path)
    spatial_data.to_netcdf(sim_spatial_path)

    # --- Save mask.nc with required 3D structure ---
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

    # --- Save time series NetCDFs ---
    obs_ts_path = tmp_path / "obs_ts.nc"
    sim_ts_path = tmp_path / "sim_ts.nc"
    obs_ts_df.rename_axis("time").to_xarray().to_netcdf(obs_ts_path)
    sim_ts_df.rename_axis("time").to_xarray().to_netcdf(sim_ts_path)

    # --- Prepare input dictionary ---
    data_folder = {
        "obs_spatial": str(obs_spatial_path),
        "sim_spatial": str(sim_spatial_path),
        "obs_ts": str(obs_ts_path),
        "sim_ts": str(sim_ts_path),
        "mask": str(mask_path),
    }

    def dummy_image(*args, **kwargs):
        return Image.new("RGB", (100, 100), (0, 0, 0))

    def dummy_draw_image(self, *args, **kwargs):
        # no-op: prevent errors when drawing images to PDF
        pass

    # Patch PIL.Image.open, Canvas.drawImage, and builtins.input for smooth test run
    with patch("PIL.Image.open", side_effect=dummy_image), \
         patch("reportlab.pdfgen.canvas.Canvas.drawImage", new=dummy_draw_image), \
         patch("builtins.input", side_effect=input_mock):  # Always confirm default folder

        # Pass tmp_path explicitly as output_dir arg to avoid prompts and force output location
        generate_full_report(
            data_folder,
            output_dir=tmp_path,
            generate_pdf=True,
            verbose=False,
            variable="var"
        )

    # Now check for PDF files inside the timestamped run folder under tmp_path
    # The output folder is tmp_path / run_YYYY-MM-DD (date today)
    today_str = datetime.now().strftime("run_%Y-%m-%d")
    output_run_folder = tmp_path / today_str

    pdf_files = list(output_run_folder.glob("Report_*.pdf"))
    assert pdf_files, f"No PDF report generated in {output_run_folder}."

    # Optionally check filename contains variable name and timestamp
    assert any(
        re.match(r"Report_var_run_\d{4}-\d{2}-\d{2}\.pdf", p.name) for p in pdf_files
    ), "PDF report filename does not contain expected variable and timestamp pattern."
    
    
def test_generate_full_report_missing_key():
    inputs = {
        "obs_spatial": xr.Dataset(),
        "sim_spatial": xr.Dataset(),
        # Missing "obs_ts"
        "sim_ts": pd.DataFrame(),
        "mask": {"mask": np.ones((2, 2))},
    }

    with pytest.raises(ValueError):
        generate_full_report(inputs)
        
def test_generate_full_report_mismatched_timeseries_index(tmp_path):
    # Create and save spatial data files
    spatial_data.to_netcdf(tmp_path / "obs_spatial.nc")
    spatial_data.to_netcdf(tmp_path / "sim_spatial.nc")

    # Create and save mask NetCDF as you do in your example
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

    # Create mismatched time series dataframes and save them
    obs_ts = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.date_range("2000-01-01", periods=2))
    sim_ts = pd.DataFrame({"value": [1.5, 2.5]}, index=pd.date_range("2001-01-01", periods=2))

    obs_ts_path = tmp_path / "obs_ts.nc"
    sim_ts_path = tmp_path / "sim_ts.nc"

    obs_ts.rename_axis("time").to_xarray().to_netcdf(obs_ts_path)
    sim_ts.rename_axis("time").to_xarray().to_netcdf(sim_ts_path)

    # Build the dict of file paths
    inputs = {
        "obs_spatial": str(tmp_path / "obs_spatial.nc"),
        "sim_spatial": str(tmp_path / "sim_spatial.nc"),
        "obs_ts": str(obs_ts_path),
        "sim_ts": str(sim_ts_path),
        "mask": str(mask_path),
    }

    with pytest.raises(ValueError):
        generate_full_report(inputs)


def test_generate_full_report_empty_input():
    with pytest.raises(ValueError):
        generate_full_report({})


def test_generate_full_report_incorrect_data_types():
    inputs = {
        "obs_spatial": "not a dataset",
        "sim_spatial": 42,
        "obs_ts": "string not dataframe",
        "sim_ts": None,
        "mask": "mask",
    }

    with pytest.raises((ValueError, TypeError)):
        generate_full_report(inputs)
