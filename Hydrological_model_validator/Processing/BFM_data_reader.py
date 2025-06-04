import gzip
import numpy as np
import xarray as xr
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
import io
import os

from .utils import (infer_years_from_path, build_bfm_filename, 
                    temp_threshold, hal_threshold)

from .file_io import unzip_gz_to_file, read_nc_variable_from_unzipped_file

from .data_alignment import apply_3d_mask

###############################################################################
def extract_bottom_layer(data: np.ndarray, Bmost: np.ndarray) -> List[np.ndarray]:
    """
    Extract bottom layer data using Bmost indices.
    Assumes data shape: (time, depth, y, x)
    Returns list of 2D arrays (time slices) with shape (y, x)
    """
    time_len, depth_len, ny, nx = data.shape
    bottom_data = np.full((time_len, ny, nx), np.nan, dtype=data.dtype)
    
    # Convert Bmost to zero-based indices and flatten spatial dims
    B_idx = Bmost.astype(int) - 1  # shape (ny, nx)
    
    # Clip indices to valid depth range (optional)
    B_idx = np.clip(B_idx, 0, depth_len - 1)
    
    # Flatten spatial indices for advanced indexing
    flat_spatial = np.arange(ny * nx)
    
    for t in range(time_len):
        # Extract the 2D bottom layer at time t
        # data[t] shape: (depth, ny, nx) -> reshape to (depth, ny*nx)
        data_t_flat = data[t].reshape(depth_len, ny * nx)
        
        # Use B_idx flattened as depth indices per spatial point
        bottom_data[t] = data_t_flat[B_idx.ravel(), flat_spatial].reshape(ny, nx)
    
    return [bottom_data[t] for t in range(time_len)]
###############################################################################

###############################################################################
def extract_and_filter_benthic_data(data_4d: np.ndarray,
                                    Bmost: np.ndarray,
                                    dz: float = 2.0,
                                    variable_key: Optional[str] = None) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        3D array (time, Y, X) of extracted and filtered benthic data.
    """
    time_len, depth_len, Y, X = data_4d.shape

    # Convert Bmost to zero-based indices and create valid mask
    valid_mask = (Bmost > 0) & (Bmost <= depth_len)
    Bmost_zero = np.where(valid_mask, Bmost - 1, 0)

    # Get indices for fancy indexing
    y_idx, x_idx = np.indices((Y, X))
    flat_indices = (np.arange(time_len)[:, None, None], Bmost_zero[None, :, :], y_idx[None, :, :], x_idx[None, :, :])

    benthic_data = np.full((time_len, Y, X), np.nan, dtype=data_4d.dtype)
    benthic_data[:, valid_mask] = data_4d[flat_indices][:, valid_mask]

    depths = Bmost * dz
    mask_shallow = (depths > 0) & (depths <= 50)  # shape (Y, X)
    mask_deep = (depths >= 51) & (depths <= 200)

    if variable_key in {'votemper', 'vosaline'}:
        # Broadcast masks to shape (time_len, Y, X)
        mask_shallow_3d = np.broadcast_to(mask_shallow, (time_len, Y, X))
        mask_deep_3d = np.broadcast_to(mask_deep, (time_len, Y, X))

        if variable_key == 'votemper':
            invalid_mask = temp_threshold(benthic_data, mask_shallow_3d, mask_deep_3d)
        else:
            invalid_mask = hal_threshold(benthic_data, mask_shallow_3d, mask_deep_3d)

        # Count invalids per time and print
        invalid_counts = np.sum(invalid_mask, axis=(1, 2))
        for t, count in enumerate(invalid_counts):
            if count > 0:
                print(f"Month {t+1}: {count} cells outside valid {variable_key} range set to NaN")

        benthic_data[invalid_mask] = np.nan

    return benthic_data
###############################################################################

###############################################################################
def process_year(year: int,
                 IDIR: Union[str, Path],
                 mask3d: np.ndarray,
                 Bmost: np.ndarray,
                 filename_fragments: Dict[str, str],
                 variable_key: str) -> Tuple[int, np.ndarray]:
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
    Tuple[int, np.ndarray]
        Year and 3D array (time, Y, X) of bottom parameter values.
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

    # Apply mask3d without making a large intermediate mask_4d array
    data = np.where(mask3d[None, :, :, :] == 0, np.nan, data)

    benthic_data = extract_and_filter_benthic_data(
        data_4d=data,
        Bmost=Bmost,
        dz=2.0,
        variable_key=variable_key
    )

    print(f"Year {year} processed.")

    return year, benthic_data
###############################################################################

###############################################################################
def read_benthic_parameter(IDIR: Union[str, Path],
                           mask3d: np.ndarray,
                           Bmost: np.ndarray,
                           filename_fragments: Dict[str, str],
                           variable_key: str) -> Tuple[int, List[np.ndarray]]:
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
    Tuple[int, List[np.ndarray]]
        The processed year and a list of 12 monthly 2D arrays of the bottom parameter.

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
    IDIR = Path(IDIR)

    print("Scanning directory to determine available years...")
    Ybeg, Yend, ysec = infer_years_from_path(IDIR, target_type="folder", pattern=r'output\s*(\d{4})')
    print(f"Found years from {Ybeg} to {Yend}: {ysec}")
    print("-" * 45)

    results = {}

    def worker(year: int) -> None:
        year_str = str(year)
        print(f"Retrieving year: {year_str}")

        filename = build_bfm_filename(year, filename_fragments)
        file_nc = IDIR / f"output{year_str}" / filename
        file_gz = Path(str(file_nc) + ".gz")

        print(f"Currently handling {filename}")

        if not file_nc.exists():
            unzip_gz_to_file(file_gz, file_nc)
            print("File successfully unzipped")
        else:
            print("Unzipped file already exists, skipping unzip")

        P_orig = read_nc_variable_from_unzipped_file(file_nc, variable_key)

        try:
            os.remove(file_nc)
        except OSError as e:
            print(f"Warning: Could not delete unzipped file {file_nc}: {e}")

        P = apply_3d_mask(P_orig, mask3d)
        print(f"Mask applied to field '{variable_key}'")

        bottom_layers = extract_bottom_layer(P, Bmost)
        print(f"Bottom layer data extracted for year {year_str}")
        print("-" * 45)

        results[year] = bottom_layers

    with ThreadPoolExecutor() as executor:
        list(executor.map(worker, ysec))

    return dict(sorted(results.items()))
###############################################################################