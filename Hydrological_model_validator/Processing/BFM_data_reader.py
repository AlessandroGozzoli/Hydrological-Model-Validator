import gzip
import numpy as np
import xarray as xr
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any
import io
import os

from .utils import (infer_years_from_path, build_bfm_filename, 
                    temp_threshold, hal_threshold)

from .file_io import unzip_gz_to_file, read_nc_variable_from_unzipped_file

from .data_alignment import apply_3d_mask

###############################################################################
def extract_bottom_layer(
    data: np.ndarray,
    Bmost: np.ndarray
) -> List[np.ndarray]:
    """
    Extract bottom layer data using Bmost indices.
    Assumes data shape: (time, depth, y, x)
    Returns list of 2D arrays (time slices) with shape (y, x)
    """
    time_len, depth_len, ny, nx = data.shape
    bottom_data = np.full((time_len, ny, nx), np.nan, dtype=data.dtype)

    for t in range(time_len):
        for j in range(ny):
            for i in range(nx):
                k = int(Bmost[j, i]) - 1  # zero-based index
                if 0 <= k < depth_len:
                    bottom_data[t, j, i] = data[t, k, j, i]

    return [bottom_data[t] for t in range(time_len)]
###############################################################################

###############################################################################
def extract_and_filter_benthic_data(
    data_4d: np.ndarray,
    Bmost: np.ndarray,
    dz: float = 2.0,
    variable_key: str = None
) -> np.ndarray:
    """
    Extract bottom layer data from 4D array and apply depth-based threshold filtering.

    Parameters
    ----------
    data_4d : np.ndarray
        4D array with shape (time, depth, Y, X).
    Bmost : np.ndarray
        2D array with 1-based indices of bottom layer per (Y, X).
    dz : float, optional
        Thickness of each vertical layer in meters (default is 2.0).
    variable_key : str, optional
        Variable name to apply thresholds ('votemper', 'vosaline'). 
        If None or unknown, no filtering applied.

    Returns
    -------
    np.ndarray
        3D array (time, Y, X) of extracted and filtered benthic data.
    """
    time_len, depth_len, Y, X = data_4d.shape

    # Convert Bmost from 1-based to 0-based indexing; invalid indices become -1
    Bmost_zero = np.where((Bmost > 0) & (Bmost <= depth_len), Bmost - 1, -1)

    # Prepare output array filled with NaNs
    benthic_data = np.full((time_len, Y, X), np.nan, dtype=data_4d.dtype)

    y_idx, x_idx = np.indices((Y, X))
    valid_mask = Bmost_zero >= 0

    # Extract bottom layer values where valid
    if np.any(valid_mask):
        benthic_data[:, valid_mask] = data_4d[
            :,
            Bmost_zero[valid_mask],
            y_idx[valid_mask],
            x_idx[valid_mask]
        ]

    # Compute depths (m) at bottom layer cells
    depths = Bmost * dz  # shape (Y, X)

    # Masks for depth ranges
    mask_shallow = (depths > 0) & (depths <= 50)
    mask_deep = (depths >= 51) & (depths <= 200)

    # Apply variable-specific filtering if applicable
    for t in range(time_len):
        slice_data = benthic_data[t]

        if variable_key == 'votemper':
            invalid_mask = temp_threshold(slice_data, mask_shallow, mask_deep)

        elif variable_key == 'vosaline':
            invalid_mask = hal_threshold(slice_data, mask_shallow, mask_deep)

        else:
            # No filtering for unknown or chemical variables
            invalid_mask = np.zeros_like(slice_data, dtype=bool)

        n_invalid = np.count_nonzero(invalid_mask)
        if n_invalid > 0:
            print(f"Month {t+1}: {n_invalid} cells outside valid {variable_key or 'variable'} range set to NaN")

        slice_data[invalid_mask] = np.nan
        benthic_data[t] = slice_data

    return benthic_data
###############################################################################

###############################################################################
def process_year(year: int,
                 IDIR: Union[str, Path],
                 mask3d: np.ndarray,
                 Bmost: np.ndarray,
                 filename_fragments: Dict[str, str],
                 variable_key: str) -> Tuple[int, List[np.ndarray], Any]:
    """
    Process benthic parameter data for a single year.

    Parameters
    ----------
    year : int
        Year to process.
    IDIR : str or Path
        Base directory containing MODEL data.
    mask3d : np.ndarray
        3D mask array (depth, Y, X), where 0 indicates invalid points.
    Bmost : np.ndarray
        2D array (Y, X), 1-based index of bottom layer.
    filename_fragments : dict
        Dictionary with keys 'ffrag1', 'ffrag2', 'ffrag3'.
    variable_key : str
        Key for the variable to extract from dataset (e.g., 'votemper', 'vosaline').

    Returns
    -------
    Tuple[int, List[np.ndarray], Any]
        Year, list of monthly 2D arrays of bottom parameter, and Bmost array.
    """
    IDIR = Path(IDIR)
    year_str = str(year)
    filename = build_bfm_filename(year, filename_fragments)
    file_nc = IDIR / f"output{year_str}" / filename
    file_gz = Path(str(file_nc) + ".gz")

    if not file_gz.exists():
        raise FileNotFoundError(f"Compressed file not found: {filename}")

    print(f"Handling file {filename}...")
    with gzip.open(file_gz, 'rb') as f_in:
        decompressed_bytes = f_in.read()

    with xr.open_dataset(io.BytesIO(decompressed_bytes)) as ds:
        if variable_key not in ds:
            raise KeyError(f"Variable '{variable_key}' not found in dataset.")
        data = ds[variable_key].values  # shape: (time, depth, Y, X)

    if data.shape[2:] != mask3d.shape[1:]:
        raise ValueError(f"Shape mismatch between data spatial dims {data.shape[2:]} and mask3d {mask3d.shape[1:]}")

    # Mask invalid points based on mask3d
    mask_4d = np.broadcast_to(mask3d == 0, data.shape)
    data = np.where(mask_4d, np.nan, data)

    # Use the helper function to extract bottom data and apply filtering
    benthic_data = extract_and_filter_benthic_data(
        data_4d=data,
        Bmost=Bmost,
        dz=2.0,
        variable_key=variable_key
    )

    monthly_values = [benthic_data[t] for t in range(benthic_data.shape[0])]
    print(f"Year {year} processed.")

    return year, monthly_values
###############################################################################

###############################################################################
def read_benthic_parameter(IDIR: Union[str, Path],
                           mask3d: np.ndarray,
                           Bmost: np.ndarray,
                           filename_fragments: Dict[str, str],
                           variable_key: str) -> Dict[int, List[np.ndarray]]:
    """
    Reads benthic parameter (e.g., temperature, salinity) from monthly averaged .nc.gz files over available years.

    Parameters
    ----------
    IDIR : str or Path
        Base directory containing MODEL data.
    mask3d : np.ndarray
        3D mask array (depth, Y, X), 0 indicates invalid points.
    Bmost : np.ndarray
        2D array (Y, X), 1-based index of bottom layer.
    filename_fragments : dict
        Dictionary with keys 'ffrag1', 'ffrag2', 'ffrag3'.
    variable_key : str
        Key of the variable to extract (e.g., 'votemper', 'vosaline').

    Returns
    -------
    Dict[int, List[np.ndarray]]
        Dictionary keyed by year, each value is a list of 12 monthly 2D arrays of the selected benthic variable.

    Raises
    ------
    ValueError
        If any filename fragment is missing or inputs have wrong shapes/dtypes.
    FileNotFoundError
        If expected files are not found in IDIR.
    """
    IDIR = Path(IDIR)
    if not IDIR.exists():
        raise FileNotFoundError(f"Directory {IDIR} does not exist.")

    if not isinstance(mask3d, np.ndarray) or mask3d.ndim != 3:
        raise ValueError("mask3d must be a 3D numpy array.")
    if not isinstance(Bmost, np.ndarray) or Bmost.ndim != 2:
        raise ValueError("Bmost must be a 2D numpy array.")
    if not filename_fragments:
        raise ValueError("filename_fragments must be provided and not None.")
    for key in ('ffrag1', 'ffrag2', 'ffrag3'):
        if key not in filename_fragments or filename_fragments[key] is None:
            raise ValueError(f"Missing filename fragment: '{key}'")

    print("Scanning directory to determine available years...")
    Ybeg, Yend, ysec = infer_years_from_path(IDIR, target_type="folder", pattern=r'output\s*(\d{4})')
    print(f"Found years from {Ybeg} to {Yend}: {ysec}")
    print("-" * 45)

    parameter_data: Dict[int, List[np.ndarray]] = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_year, y, IDIR, mask3d, Bmost, filename_fragments, variable_key): y for y in ysec
        }
        for future in concurrent.futures.as_completed(futures):
            year = futures[future]
            try:
                yr, values = future.result()
                parameter_data[yr] = values
            except Exception as exc:
                print(f"Error processing year {year}: {exc}")

    return dict(sorted(parameter_data.items()))
###############################################################################

###############################################################################
def read_bfm_chemical(
    IDIR: Union[str, Path],
    mask3d: np.ndarray,
    Bmost: np.ndarray,
    filename_fragments: Dict[str, str],
    variable_key: str
) -> Dict[int, List[np.ndarray]]:
    """
    Reads BFM NetCDF model output for a specific chemical variable,
    applies mask, extracts bottom layer values, and returns data by year.

    Parameters
    ----------
    IDIR : str or Path
        Base directory of the BFM model outputs.
    mask3d : np.ndarray
        3D binary mask array for applying NaNs.
    Bmost : np.ndarray
        2D array indicating bottom-most valid level for each grid cell.
    filename_fragments : dict
        Filename fragments to construct full NetCDF filenames.
    variable_key : str
        Variable name to extract.

    Returns
    -------
    dict[int, list[np.ndarray]]
        Dictionary keyed by year with list of 2D arrays (one per time step).
    """
    data = {}

    print("Scanning directory to determine available years...")
    Ybeg, Yend, ysec = infer_years_from_path(IDIR, target_type="folder", pattern=r'output\s*(\d{4})')
    print(f"Found years from {Ybeg} to {Yend}: {ysec}")
    print("-" * 45)

    for year in ysec:
        ystr = str(year)
        print(f"Retrieving year: {ystr}")

        filename = build_bfm_filename(year, filename_fragments)
        file_nc = Path(IDIR) / f"output{year}" / filename
        file_gz = Path(str(file_nc) + ".gz")

        print(f"Currently handling {filename}")

        # Unzip file to disk
        unzip_gz_to_file(file_gz, file_nc)
        print("File successfully unzipped")

        # Read variable
        P_orig = read_nc_variable_from_unzipped_file(file_nc, variable_key)

        # Clean up unzipped file
        os.remove(file_nc)

        # Apply mask
        P = apply_3d_mask(P_orig, mask3d)
        print(f"Mask applied to field '{variable_key}'")

        # Extract bottom layer
        bottom_layers = extract_bottom_layer(P, Bmost)
        data[year] = bottom_layers

        print(f"Bottom layer data extracted for year {ystr}")
        print("-" * 45)

    return data
###############################################################################