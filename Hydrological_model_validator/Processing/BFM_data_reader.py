###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# Standard library imports
import gzip
import io
import os
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union, Tuple, Optional

# Data handling libraries
import numpy as np
import xarray as xr

# Logging and tracing
import logging
from eliot import start_action, log_message

# Module utilities and modules
from Hydrological_model_validator.Processing.time_utils import Timer
from Hydrological_model_validator.Processing.utils import (infer_years_from_path, 
                                                           build_bfm_filename, 
                                                           temp_threshold, 
                                                           hal_threshold)
from Hydrological_model_validator.Processing.file_io import (unzip_gz_to_file, 
                                                             read_nc_variable_from_unzipped_file)
from Hydrological_model_validator.Processing.data_alignment import apply_3d_mask

###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def extract_bottom_layer(data: np.ndarray, Bmost: np.ndarray) -> list[np.ndarray]:
    """
    Extract the bottom layer data from a 4D array using provided bottom layer indices.

    This function extracts the bottom-most layer values for each spatial location (y, x)
    from a 4D data array with dimensions (time, depth, y, x). The bottom layer is specified
    by the 1-based indices provided in `Bmost`. The function returns a list of 2D arrays,
    each corresponding to a time slice, containing the bottom layer data.

    Parameters
    ----------
    data : np.ndarray
        4D numpy array with shape (time, depth, y, x), representing data collected over time,
        vertical layers, and spatial grid.
    Bmost : np.ndarray
        2D array of shape (y, x) containing 1-based indices indicating the bottom layer depth
        for each spatial location. Indices are converted internally to zero-based.

    Returns
    -------
    list of np.ndarray
        A list of 2D numpy arrays, each of shape (y, x), where each array corresponds to a
        time slice containing the extracted bottom layer data.

    Raises
    ------
    TypeError
        If input arrays are not numpy ndarrays.
    ValueError
        If input arrays do not have the expected dimensions or shapes, or if indices are invalid.

    Example
    -------
    >>> import numpy as np
    >>> data = np.random.rand(3, 4, 2, 2)  # time=3, depth=4, y=2, x=2
    >>> Bmost = np.array([[4, 3],
    ...                   [2, 1]])  # bottom indices for each (y,x)
    >>> bottom_layers = extract_bottom_layer(data, Bmost)
    >>> print(len(bottom_layers))
    3
    >>> print(bottom_layers[0].shape)
    (2, 2)
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(data, np.ndarray):
        raise TypeError(f"❌ data must be a numpy.ndarray, got {type(data)} ❌")
    if not isinstance(Bmost, np.ndarray):
        raise TypeError(f"❌ Bmost must be a numpy.ndarray, got {type(Bmost)} ❌")

    if data.ndim != 4:
        raise ValueError(f"❌ data must be a 4D array with shape (time, depth, y, x), got shape {data.shape} ❌")
    if Bmost.ndim != 2:
        raise ValueError(f"❌ Bmost must be a 2D array with shape (y, x), got shape {Bmost.shape} ❌")

    time_len, depth_len, ny, nx = data.shape

    if Bmost.shape != (ny, nx):
        raise ValueError(f"❌ Bmost shape {Bmost.shape} must match spatial dimensions (y, x) of data {(ny, nx)} ❌")

    if not np.issubdtype(Bmost.dtype, np.integer):
        raise ValueError("❌ Bmost must contain integer values (1-based indices) ❌")
    if np.any(Bmost < 0):
        raise ValueError("❌ Bmost indices must be >= 0 ❌")
    if np.any(Bmost > depth_len):
        raise ValueError(f"❌ Bmost indices must be <= depth dimension of data ({depth_len}), found values exceeding it ❌")

    with Timer("extract_bottom_layer function"):
        with start_action(action_type="extract_bottom_layer function"):
            log_message("Input validation passed",
                        data_shape=data.shape,
                        Bmost_shape=Bmost.shape)
            logging.info(f"Input validation passed for data shape {data.shape} and Bmost shape {Bmost.shape}.")

            # ===== PREPARE OUTPUT ARRAY =====
            bottom_data = np.full((time_len, ny, nx), np.nan, dtype=data.dtype)
            log_message("Initialized output array with NaNs",
                        output_shape=bottom_data.shape)
            logging.info(f"Initialized bottom_data array with shape {bottom_data.shape} filled with NaNs.")

            # ===== CONVERT Bmost TO ZERO-BASED INDICES =====
            B_idx = Bmost.astype(int) - 1
            log_message("Converted Bmost indices from 1-based to 0-based",
                        B_idx_min=B_idx.min(),
                        B_idx_max=B_idx.max())
            logging.info(f"Converted Bmost indices to zero-based indexing. Min: {B_idx.min()}, Max: {B_idx.max()}.")

            # ===== FLATTEN SPATIAL GRID FOR VECTORIZED INDEXING =====
            flat_spatial = np.arange(ny * nx)
            log_message("Created flattened spatial indices for vectorized extraction",
                        flat_spatial_length=len(flat_spatial))
            logging.info(f"Created flat spatial index array with length {len(flat_spatial)}.")

            # ===== EXTRACT BOTTOM LAYER DATA FOR EACH TIME SLICE =====
            for t in range(time_len):
                data_t_flat = data[t].reshape(depth_len, ny * nx)
                bottom_data[t] = data_t_flat[B_idx.ravel(), flat_spatial].reshape(ny, nx)
                log_message(f"Extracted bottom layer for time step {t}",
                            time_step=t,
                            extracted_shape=bottom_data[t].shape)
                logging.info(f"Extracted bottom layer data for time step {t} with shape {bottom_data[t].shape}.")

            log_message("Completed extraction of bottom layer data",
                        time_len=time_len,
                        spatial_dims=(ny, nx))
            logging.info(f"Completed bottom layer data extraction for all {time_len} time steps with spatial dims {(ny, nx)}.")

            # ===== RETURN AS LIST OF 2D ARRAYS =====
            return [bottom_data[t] for t in range(time_len)]
        
###############################################################################

###############################################################################

def extract_and_filter_benthic_data(data_4d: np.ndarray,
                                    Bmost: np.ndarray,
                                    dz: float = 2.0,
                                    variable_key: Optional[str] = None) -> np.ndarray:
    """
    Extract bottom layer data from a 4D array and apply depth-based threshold filtering.

    This function extracts data from the bottom-most layer indicated by `Bmost` indices
    from a 4D array `data_4d` with dimensions (time, depth, Y, X). It then applies filtering
    based on depth-dependent thresholds for specific variables ('votemper' for temperature,
    'vosaline' for salinity). Invalid data points outside the thresholds are set to NaN.

    Parameters
    ----------
    data_4d : np.ndarray
        4D numpy array with shape (time, depth, Y, X), containing the variable data over
        time, vertical layers, and spatial grid.
    Bmost : np.ndarray
        2D array with shape (Y, X) of 1-based indices indicating the bottom layer depth at
        each spatial position.
    dz : float, optional
        Thickness of each vertical layer in meters. Default is 2.0.
    variable_key : str, optional
        Variable name to apply threshold filtering. Supported values are 'votemper' (temperature)
        and 'vosaline' (salinity). If None or unsupported, no filtering is applied.

    Returns
    -------
    np.ndarray
        3D numpy array with shape (time, Y, X) containing the extracted bottom layer data,
        filtered by the specified depth-dependent thresholds.

    Example
    -------
    >>> import numpy as np
    >>> data_4d = np.random.rand(12, 10, 5, 5)  # 12 time steps, 10 depth layers, 5x5 spatial grid
    >>> Bmost = np.array([[10, 9, 8, 10, 7],
    ...                   [10, 10, 9, 8, 7],
    ...                   [9, 10, 10, 9, 8],
    ...                   [8, 9, 10, 10, 9],
    ...                   [7, 8, 9, 10, 10]])  # bottom layer indices (1-based)
    >>> benthic_filtered = extract_and_filter_benthic_data(data_4d, Bmost, dz=2.0, variable_key='votemper')
    >>> print(benthic_filtered.shape)
    (12, 5, 5)
    """
    # ===== Input validation =====
    if not isinstance(data_4d, np.ndarray):
        raise TypeError(f"❌ data_4d must be a numpy.ndarray, got {type(data_4d)} ❌")
    if not isinstance(Bmost, np.ndarray):
        raise TypeError(f"❌ Bmost must be a numpy.ndarray, got {type(Bmost)} ❌")
    if data_4d.ndim != 4:
        raise ValueError(f"❌ data_4d must be 4D with shape (time, depth, Y, X), got {data_4d.shape} ❌")
    if Bmost.ndim != 2:
        raise ValueError(f"❌ Bmost must be 2D with shape (Y, X), got {Bmost.shape} ❌")
    time_len, depth_len, Y, X = data_4d.shape
    if Bmost.shape != (Y, X):
        raise ValueError(f"❌ Bmost shape {Bmost.shape} must match spatial dims (Y, X) of data_4d {(Y, X)} ❌")
    if not np.issubdtype(Bmost.dtype, np.integer):
        raise ValueError("❌ Bmost must contain integer values (1-based indices) ❌")
    if np.any(Bmost < 0) or np.any(Bmost >= depth_len):
        raise ValueError(f"❌ Bmost indices must be in the range 0 to depth_len={depth_len} ❌")
    if not isinstance(dz, (float, int)) or dz <= 0:
        raise ValueError("❌ dz must be a positive number ❌")
    if variable_key is not None and variable_key not in {'votemper', 'vosaline'}:
        # Just warn, no filtering will be applied
        print(f"⚠️ Warning: Unsupported variable_key '{variable_key}', no filtering will be applied ⚠️")

    with Timer("extract_and_filter_benthic_data function"):
        with start_action(action_type="extract_and_filter_benthic_data function"):

            log_message("Input validation passed",
                        data_shape=data_4d.shape,
                        Bmost_shape=Bmost.shape,
                        dz=dz,
                        variable_key=variable_key)
            logging.info(f"Input validation passed for data_4d shape {data_4d.shape} and Bmost shape {Bmost.shape}.")

            # ===== EXTRACT THE DATA =====
            # Extract dimensions for clarity and to guide indexing logic
            time_len, depth_len, Y, X = data_4d.shape

            # ===== CREATE THE MASK =====
            # Create a mask to identify spatial points where bottom indices are valid
            valid_mask = (Bmost > 0) & (Bmost <= depth_len)
            log_message("Created valid_mask for bottom layer indices",
                        valid_points=np.sum(valid_mask))
            logging.info(f"Valid bottom layer points count: {np.sum(valid_mask)}.")

            # ===== GET BOTTOM DATA WITH MASK =====
            # Convert 1-based bottom layer indices to zero-based for Python indexing
            # For invalid indices, temporarily assign 0 to avoid indexing errors
            Bmost_zero = np.where(valid_mask, Bmost - 1, 0)

            # Generate coordinate grids to index time, y, and x dimensions simultaneously
            y_idx, x_idx = np.indices((Y, X))

            # Build a tuple of indices for advanced indexing into the 4D array
            # This fetches data from the bottom layer depth (Bmost_zero) for all time steps and spatial points
            flat_indices = (np.arange(time_len)[:, None, None], Bmost_zero[None, :, :], y_idx[None, :, :], x_idx[None, :, :])

            # Initialize output array with NaNs to hold bottom layer data
            # Using NaNs helps clearly identify missing or invalid data after extraction and filtering
            benthic_data = np.full((time_len, Y, X), np.nan, dtype=data_4d.dtype)

            # ===== EXTRACT THE BOTTOM DATA =====
            # Extract bottom layer data only at valid locations; invalid locations remain NaN
            benthic_data[:, valid_mask] = data_4d[flat_indices][:, valid_mask]
            log_message("Extracted bottom layer data",
                        extracted_points=np.sum(~np.isnan(benthic_data)))
            logging.info(f"Extracted bottom layer data for {np.sum(~np.isnan(benthic_data))} valid points.")

            # Calculate actual depth (meters) of each bottom cell to apply depth-dependent filters
            depths = Bmost * dz
            log_message("Calculated depths from Bmost and dz",
                        depth_min=np.min(depths),
                        depth_max=np.max(depths))
            logging.info(f"Depths calculated, range: {np.min(depths)} to {np.max(depths)} meters.")

            # ===== FILTERING OF INVALID DATA =====
            # Define masks for shallow and deep regions to apply variable-specific thresholds
            # These ranges are based on known environmental zones with distinct physical/chemical properties
            mask_shallow = (depths > 0) & (depths <= 50)
            mask_deep = (depths >= 51) & (depths <= 200)
            log_message("Defined shallow and deep masks",
                        shallow_count=np.sum(mask_shallow),
                        deep_count=np.sum(mask_deep))
            logging.info(f"Shallow cells: {np.sum(mask_shallow)}, Deep cells: {np.sum(mask_deep)}.")

            # Only apply filtering if a recognized variable key is provided
            if variable_key in {'votemper', 'vosaline'}:
                # Broadcast shallow and deep masks to 3D shape to match time dimension for masking invalid data
                mask_shallow_3d = np.broadcast_to(mask_shallow, (time_len, Y, X))
                mask_deep_3d = np.broadcast_to(mask_deep, (time_len, Y, X))

                # Apply variable-specific thresholding functions that return masks of invalid data points
                # This step removes physically unrealistic or erroneous measurements
                if variable_key == 'votemper':
                    invalid_mask = temp_threshold(benthic_data, mask_shallow_3d, mask_deep_3d)
                else:
                    invalid_mask = hal_threshold(benthic_data, mask_shallow_3d, mask_deep_3d)

                # Log counts of invalid cells per time step to monitor filtering impact
                invalid_counts = np.sum(invalid_mask, axis=(1, 2))
                for t, count in enumerate(invalid_counts):
                    if count > 0:
                        print(f"⚠️ Month {t+1}: {count} cells outside valid {variable_key} range set to NaN ⚠️")

                # Set invalid data points to NaN to exclude them from further analysis
                benthic_data[invalid_mask] = np.nan

                log_message("Applied filtering to benthic data",
                            total_invalid=np.sum(invalid_mask))
                logging.info(f"Filtering applied: total invalid data points set to NaN = {np.sum(invalid_mask)}.")

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
    Process benthic parameter data for a single year by reading, masking, extracting,
    and filtering bottom layer data from model output files.

    This function locates the compressed model data file for the specified year,
    decompresses it, loads the specified variable's data, applies a 3D mask to
    filter out invalid points, and extracts the bottom layer benthic parameter
    values using depth indices (`Bmost`). It then applies variable-specific filtering
    to ensure data quality.

    Parameters
    ----------
    year : int
        The year of the dataset to process.
    IDIR : str or Path
        Base directory path where the model output files are stored.
    mask3d : np.ndarray
        A 3D mask array of shape (depth, Y, X) where zeros indicate invalid data points
        to be masked out (set as NaN).
    Bmost : np.ndarray
        A 2D array of shape (Y, X) containing 1-based indices that indicate the bottom
        vertical layer for each spatial point.
    filename_fragments : dict
        Dictionary containing filename fragments with keys 'ffrag1', 'ffrag2', and 'ffrag3',
        used to construct the filename of the model output.
    variable_key : str
        The name of the variable to extract from the dataset (e.g., 'votemper', 'vosaline').

    Returns
    -------
    Tuple[int, np.ndarray]
        A tuple containing:
        - The processed year (int).
        - A 3D numpy array of shape (time, Y, X) with the extracted and filtered bottom
          layer parameter data.

    Raises
    ------
    FileNotFoundError
        If the compressed model output file for the given year does not exist.
    KeyError
        If the specified `variable_key` is not found in the dataset.
    ValueError
        If the spatial dimensions of the data do not match those of the provided mask.

    Example
    -------
    >>> year = 2005
    >>> base_dir = "/data/model_output"
    >>> mask = np.ones((10, 20, 30))  # example mask with all valid points
    >>> Bmost_indices = np.full((20, 30), 10)  # bottom layer is the 10th depth for all
    >>> fragments = {'ffrag1': 'model', 'ffrag2': 'output', 'ffrag3': 'nc'}
    >>> variable = 'votemper'
    >>> yr, benthic_arr = process_year(year, base_dir, mask, Bmost_indices, fragments, variable)
    >>> print(yr)
    2005
    >>> print(benthic_arr.shape)
    (time_steps, 20, 30)
    """
    # ===== INPUT VALIDATION =====
    if not isinstance(year, int):
        raise TypeError(f"❌ year must be int, got {type(year)} ❌")
    if not isinstance(mask3d, np.ndarray):
        raise TypeError(f"❌ mask3d must be np.ndarray, got {type(mask3d)} ❌")
    if mask3d.ndim != 3:
        raise ValueError(f"❌ mask3d must be 3D (depth, Y, X), got shape {mask3d.shape} ❌")
    if not isinstance(Bmost, np.ndarray):
        raise TypeError(f"❌ Bmost must be np.ndarray, got {type(Bmost)} ❌")
    if Bmost.ndim != 2:
        raise ValueError(f"❌ Bmost must be 2D (Y, X), got shape {Bmost.shape} ❌")
    if not isinstance(filename_fragments, dict):
        raise TypeError(f"❌ filename_fragments must be dict, got {type(filename_fragments)} ❌")
    required_keys = {'ffrag1', 'ffrag2', 'ffrag3'}
    missing_keys = required_keys - filename_fragments.keys()
    if missing_keys:
        raise KeyError(f"❌ filename_fragments missing keys: {missing_keys} ❌")
    if not isinstance(variable_key, str):
        raise TypeError(f"❌ variable_key must be str, got {type(variable_key)} ❌")

    log_message("Input validation passed",
                year=year,
                mask3d_shape=mask3d.shape,
                Bmost_shape=Bmost.shape,
                filename_fragments_keys=list(filename_fragments.keys()),
                variable_key=variable_key)
    logging.info(f"Input validation passed for year {year}.")

    # ===== PATH CHECK =====
    IDIR = Path(IDIR)
    if not IDIR.exists() or not IDIR.is_dir():
        raise ValueError(f"❌ IDIR path does not exist or is not a directory: {IDIR} ❌")

    year_str = str(year)
    log_message("Input directory verified",
                IDIR=str(IDIR))

    with Timer("process_year function"):
        with start_action(action_type="process_year function", year=year):
            # ===== BUILD FILENAME AND PATH =====
            filename = build_bfm_filename(year, filename_fragments)
            file_nc = IDIR / f"output{year_str}" / filename
            file_gz = Path(str(file_nc) + ".gz")

            if not file_gz.exists():
                raise FileNotFoundError(f"❌ Compressed file not found: {file_gz} ❌")

            log_message("Located compressed file",
                        filename=str(file_gz))
            logging.info(f"Located compressed file: {file_gz}")

            print(f"Handling file {filename}...")

            # ===== DECOMPRESS AND READ DATA =====
            with gzip.open(file_gz, 'rb') as f_in:
                decompressed_bytes = f_in.read()

            with xr.open_dataset(io.BytesIO(decompressed_bytes)) as ds:
                if variable_key not in ds:
                    raise KeyError(f"❌ Variable '{variable_key}' not found in dataset. ❌")
                data = ds[variable_key].values

            log_message("Loaded data from dataset",
                        variable_key=variable_key,
                        data_shape=data.shape)
            logging.info(f"Loaded variable '{variable_key}' data with shape {data.shape}.")

            # ===== VERIFY SHAPE MATCH =====
            if data.shape[2:] != mask3d.shape[1:]:
                raise ValueError(f"❌ Shape mismatch between data spatial dims {data.shape[2:]} and mask3d {mask3d.shape[1:]} ❌")

            log_message("Verified spatial dimensions match",
                        data_spatial=data.shape[2:],
                        mask3d_spatial=mask3d.shape[1:])
            logging.info("Spatial dimensions match between data and mask3d.")

            # ===== MASKING =====
            data = np.where(mask3d[None, :, :, :] == 0, np.nan, data)
            log_message("Applied 3D mask to data",
                        mask_shape=mask3d.shape)

            # ===== EXTRACT AND FILTER BOTTOM DATA =====
            benthic_data = extract_and_filter_benthic_data(
                data_4d=data,
                Bmost=Bmost,
                dz=2.0,
                variable_key=variable_key
            )

            log_message("Extracted and filtered bottom layer benthic data",
                        benthic_shape=benthic_data.shape)

            print(f"\033[92m✅ Year {year} processed.\033[0m")

            return year, benthic_data
        
###############################################################################

###############################################################################

def read_benthic_parameter(IDIR: Union[str, Path],
                           mask3d: np.ndarray,
                           Bmost: np.ndarray,
                           filename_fragments: Dict[str, str],
                           variable_key: str) -> Dict[int, List[np.ndarray]]:
    """
    Reads benthic parameter data (e.g., temperature, salinity) from monthly averaged
    compressed NetCDF files over all available years in a specified directory.

    The function scans the given directory for yearly folders matching a pattern,
    then concurrently processes each year’s data by applying spatial and depth masks,
    extracting bottom layer values, and filtering the data based on the variable's
    quality thresholds. The results are collected as a dictionary mapping each year
    to a list of 12 monthly 2D arrays representing the benthic parameter.

    Parameters
    ----------
    IDIR : str or Path
        Base directory containing the MODEL output data folders.
    mask3d : np.ndarray
        3D mask array with shape (depth, Y, X), where 0 indicates invalid points that
        should be masked out (set to NaN).
    Bmost : np.ndarray
        2D array with shape (Y, X) containing 1-based indices of the bottom vertical
        layer at each spatial location.
    filename_fragments : dict
        Dictionary with keys 'ffrag1', 'ffrag2', 'ffrag3' containing parts of the filename
        used to locate the NetCDF files.
    variable_key : str
        Key name of the variable to extract from the dataset, such as 'votemper' or 'vosaline'.

    Returns
    -------
    Dict[int, List[np.ndarray]]
        Dictionary mapping each processed year (int) to a list of 12 numpy 2D arrays
        (Y, X) representing monthly bottom parameter values.

    Raises
    ------
    ValueError
        If any required filename fragment is missing, or if inputs have incorrect types
        or dimensions.
    FileNotFoundError
        If the specified data directory does not exist.

    Example
    -------
    >>> base_dir = "/data/model_output"
    >>> mask = np.ones((10, 50, 60))  # mask with all valid points
    >>> Bmost_indices = np.full((50, 60), 10)  # bottom layer index = 10 everywhere
    >>> fragments = {'ffrag1': 'model', 'ffrag2': 'output', 'ffrag3': 'nc'}
    >>> variable = 'votemper'
    >>> data_by_year = read_benthic_parameter(base_dir, mask, Bmost_indices, fragments, variable)
    >>> for year, monthly_data in data_by_year.items():
    ...     print(f"Year {year} has {len(monthly_data)} months of data, shape {monthly_data[0].shape}")
    """
    # ===== INPUT VALIDATIONS =====
    # Convert input path to Path object for consistent filesystem operations
    IDIR = Path(IDIR)

    # Check existence early to avoid wasted computation on missing data
    if not IDIR.exists():
        raise FileNotFoundError(f"❌ Directory {IDIR} does not exist. ❌")

    # Validate input types and dimensions to prevent downstream errors
    if not isinstance(mask3d, np.ndarray) or mask3d.ndim != 3:
        raise ValueError("❌ mask3d must be a 3D numpy array. ❌")
    if not isinstance(Bmost, np.ndarray) or Bmost.ndim != 2:
        raise ValueError("❌ Bmost must be a 2D numpy array. ❌")
    if not filename_fragments:
        raise ValueError("❌ filename_fragments must be provided and not None. ❌")
    for key in ('ffrag1', 'ffrag2', 'ffrag3'):
        # Ensure all required filename parts are available to correctly locate files
        if key not in filename_fragments or filename_fragments[key] is None:
            raise ValueError(f"❌ Missing filename fragment: '{key}' ❌")

    with Timer("read_benthic_parameter function"):
        with start_action(action_type="read_benthic_parameter function"):
            log_message("Input validation passed",
                        IDIR=str(IDIR),
                        mask3d_shape=mask3d.shape,
                        Bmost_shape=Bmost.shape,
                        filename_fragments_keys=list(filename_fragments.keys()),
                        variable_key=variable_key)
            logging.info("Input validation passed for read_benthic_parameter")

            # ===== SCAN THE FOLDER FOR YEAR RANGE =====
            # Identify the range of years available by scanning directory structure
            print("Scanning directory to determine available years...")
            Ybeg, Yend, ysec = infer_years_from_path(IDIR, target_type="folder", pattern=r'output\s*(\d{4})')
            print(f"Found years from {Ybeg} to {Yend}: {ysec}")
            print("-" * 45)

            parameter_data: Dict[int, List[np.ndarray]] = {}

            # ===== GET DATA ALL TOGETHER =====
            # Use a thread pool to process multiple years concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_year, y, IDIR, mask3d, Bmost, filename_fragments, variable_key): y for y in ysec
                }
                for future in concurrent.futures.as_completed(futures):
                    year = futures[future]
                    try:
                        # Collect processed benthic data for each year
                        yr, values = future.result()
                        parameter_data[yr] = values
                        log_message("Year processed successfully",
                                    year=yr,
                                    data_shape=values.shape if isinstance(values, np.ndarray) else "list")
                        logging.info(f"Year {yr} processed successfully in read_benthic_parameter.")
                    except Exception as exc:
                        # Log errors but continue processing other years
                        log_message("Error processing year",
                                    year=year,
                                    error=str(exc))
                        logging.error(f"Error processing year {year}: {exc}")
                        print(f"❌ Error processing year {year}: {exc} ❌")

            # Sort results by year to provide ordered output
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
    Reads BFM NetCDF model output for a specified chemical variable over multiple years,
    applies a spatial mask, extracts bottom layer values, and returns the data organized by year.

    This function scans the given base directory for yearly output folders matching a pattern,
    constructs filenames from provided fragments, and processes each year’s compressed
    NetCDF files. For each year, it unzips the file (if needed), reads the specified variable,
    applies a 3D mask to invalidate certain points, extracts the bottom-most valid layer
    based on Bmost indices, and collects the monthly or time-step data as 2D arrays.

    Parameters
    ----------
    IDIR : str or Path
        Base directory path where BFM model output folders are located.
    mask3d : np.ndarray
        3D numpy array (depth, Y, X) used as a mask; elements with zero are masked (set to NaN).
    Bmost : np.ndarray
        2D numpy array (Y, X) containing 1-based indices indicating the bottom-most valid layer per grid cell.
    filename_fragments : dict
        Dictionary with keys such as 'ffrag1', 'ffrag2', 'ffrag3' used to construct filenames for the NetCDF files.
    variable_key : str
        Name of the chemical variable to extract from the NetCDF files (e.g., 'O2', 'NO3').

    Returns
    -------
    dict[int, list[np.ndarray]]
        Dictionary where each key is a year (int) and each value is a list of 2D numpy arrays
        representing the extracted bottom layer chemical parameter for each time step (e.g., months).

    Notes
    -----
    - The function will unzip compressed .gz files if uncompressed NetCDF files are not already present.
    - Unzipped files are deleted after reading to conserve disk space.
    - Processing is done concurrently across years using a thread pool for speed.

    Example
    -------
    >>> base_dir = "/path/to/bfm/output"
    >>> mask = np.ones((10, 50, 60))
    >>> Bmost_indices = np.full((50, 60), 10)
    >>> fragments = {'ffrag1': 'chem', 'ffrag2': 'monthly', 'ffrag3': 'nc'}
    >>> chemical_var = 'O2'
    >>> data_by_year = read_bfm_chemical(base_dir, mask, Bmost_indices, fragments, chemical_var)
    >>> for year, monthly_layers in data_by_year.items():
    ...     print(f"Year {year} has {len(monthly_layers)} time slices, shape {monthly_layers[0].shape}")
    """
    # ===== INPUT VALIDATIONS =====
    IDIR = Path(IDIR)

    if not IDIR.exists():
        raise FileNotFoundError(f"❌ Directory {IDIR} does not exist. ❌")

    if not isinstance(mask3d, np.ndarray) or mask3d.ndim != 3:
        raise ValueError("❌ mask3d must be a 3D numpy array. ❌")

    if not isinstance(Bmost, np.ndarray) or Bmost.ndim != 2:
        raise ValueError("❌ Bmost must be a 2D numpy array. ❌")

    if not filename_fragments or not isinstance(filename_fragments, dict):
        raise ValueError("❌ filename_fragments must be a non-empty dictionary. ❌")

    for key in ('ffrag1', 'ffrag2', 'ffrag3'):
        if key not in filename_fragments or filename_fragments[key] is None:
            raise ValueError(f"❌ Missing filename fragment: '{key}' ❌")

    if not isinstance(variable_key, str) or not variable_key:
        raise ValueError("❌ variable_key must be a non-empty string. ❌")

    with Timer("read_bfm_chemical function"):
        with start_action(action_type="read_bfm_chemical function"):
            log_message("Input validation passed",
                        IDIR=str(IDIR),
                        mask3d_shape=mask3d.shape,
                        Bmost_shape=Bmost.shape,
                        filename_fragments_keys=list(filename_fragments.keys()),
                        variable_key=variable_key)
            logging.info("Input validation passed for read_bfm_chemical")

            # ===== SCAN FOLDER TO OBTAIN YEAR INFO =====
            # Identify available years by scanning output folders named like "outputYYYY"
            # Using a regex pattern to capture four-digit year numbers dynamically
            print("Scanning directory to determine available years...")
            Ybeg, Yend, ysec = infer_years_from_path(IDIR, target_type="folder", pattern=r'output\s*(\d{4})')
            print(f"Found years from {Ybeg} to {Yend}: {ysec}")
            print("-" * 45)

            results = {}

            # ===== WORKERS =====
            # Worker function to process one year at a time
            def worker(year: int) -> None:
                year_str = str(year)
                print(f"Retrieving year: {year_str}")

                with start_action(action_type="process_year", year=year):
                    # Construct full filename based on the year and filename fragments
                    filename = build_bfm_filename(year, filename_fragments)
                    file_nc = IDIR / f"output{year_str}" / filename
                    file_gz = Path(str(file_nc) + ".gz")
                    print(f"Currently handling {filename}")

                    # If the NetCDF file is not yet uncompressed, unzip it from the .gz archive
                    if not file_nc.exists():
                        unzip_gz_to_file(file_gz, file_nc)
                        print("\033[92m✅ File successfully unzipped\033[0m")

                    # Read the target variable data from the unzipped NetCDF file
                    P_orig = read_nc_variable_from_unzipped_file(file_nc, variable_key)

                    # Remove the uncompressed file to save disk space after reading
                    try:
                        os.remove(file_nc)
                    except OSError:
                        # If file removal fails (e.g., permission issues), ignore and continue
                        pass

                    # Apply the 3D mask to invalidate points by setting masked locations to NaN
                    P = apply_3d_mask(P_orig, mask3d)

                    # Extract the bottom-most valid layer using Bmost indices (1-based)
                    bottom_layers = extract_bottom_layer(P, Bmost)

                    # Store the extracted bottom layer data indexed by year
                    results[year] = bottom_layers

                    log_message("Year processed successfully",
                                year=year,
                                data_shape=bottom_layers.shape if isinstance(bottom_layers, np.ndarray) else "list")
                    logging.info(f"Year {year} processed successfully in read_bfm_chemical.")

                    print(f"\033[92m✅ Correct layer data extracted for year {year_str}\033[0m")
                    print("-" * 45)

            # ===== READS ALL FILES =====
            # Use a thread pool to process each year concurrently for faster I/O and CPU utilization
            with ThreadPoolExecutor() as executor:
                list(executor.map(worker, ysec))

            # Return the results sorted by year for consistent ordering
            return dict(sorted(results.items()))