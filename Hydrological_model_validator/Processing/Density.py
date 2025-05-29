import gsw  # TEOS-10/EOS-80 library
import numpy as np
from typing import Dict, List, Union
from pathlib import Path
from datetime import datetime

from .utils import (infer_years_from_path, temp_threshold, 
                    hal_threshold, build_bfm_filename)

from .file_io import read_nc_variable_from_gz_in_memory

###############################################################################
def compute_density_bottom(temperature_data: dict,
                    salinity_data: dict,
                    Bmost: np.ndarray,
                    method: str,
                    dz: float = 2.0) -> dict:
    """
    Compute seawater density (kg/m³) at benthic depth using the specified method.

    Parameters
    ----------
    temperature_data : dict
        Dictionary {year: [12 arrays]} with bottom temperature per month.
    salinity_data : dict
        Dictionary {year: [12 arrays]} with bottom salinity per month.
    Bmost : ndarray
        2D array (Y, X) with index (1-based) of deepest valid vertical level.
    method : str
        Method for computing density. Choose from: 'EOS', 'EOS80', 'TEOS10'.
    dz : float, optional
        Vertical layer resolution in meters (default is 2.0 m).

    Returns
    -------
    density_data : dict
        Dictionary {year: [12 arrays]} with computed density fields.
    """
    if method not in {"EOS", "EOS80", "TEOS10"}:
        raise ValueError(f"Unsupported method '{method}'. Choose from: 'EOS', 'EOS80', 'TEOS10'.")

    depth = Bmost * dz  # shape (Y, X)
    density_data = {}

    for year, temp_list in temperature_data.items():
        density_data[year] = []

        for month_idx, temp_2d in enumerate(temp_list):
            sal_2d = salinity_data[year][month_idx]

            # Add fake time dimension for compatibility (T=1)
            temp_3d = temp_2d[None, ...]  # shape (1, Y, X)
            sal_3d = sal_2d[None, ...]

            if method == "EOS":
                # Linear EOS approximation
                alpha = 0.0002  # thermal expansion
                beta = 0.0008   # haline contraction
                rho0 = 1025     # reference density
                density = rho0 * (1 - alpha * (temp_3d - 10) + beta * (sal_3d - 35))

            elif method == "EOS80":
                # EOS-80 potential density at surface
                density = gsw.density.sigma0(sal_3d, temp_3d) + 1000

            elif method == "TEOS10":
                # TEOS-10 absolute density using in-situ pressure
                density = gsw.density.rho(sal_3d, temp_3d, depth)

            # Take the 2D slice (remove time dimension safely)
            if density.ndim == 3 and density.shape[0] == 1:
                density_2d = density[0]
            else:
                raise ValueError(f"Unexpected density shape: {density.shape}")

            density_data[year].append(density_2d)

    return density_data
###############################################################################

###############################################################################
def compute_density_3d(temp_3d: np.ndarray, sal_3d: np.ndarray, depths: np.ndarray, method: str = "EOS") -> np.ndarray:
    density_3d = np.full(temp_3d.shape, np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(temp_3d) & ~np.isnan(sal_3d)

    if method == "EOS":
        alpha = 0.0002
        beta = 0.0008
        rho0 = 1025
        density_3d[valid_mask] = rho0 * (1 - alpha * (temp_3d[valid_mask] - 10) + beta * (sal_3d[valid_mask] - 35))

    elif method == "EOS80":
        density_3d[valid_mask] = gsw.density.sigma0(sal_3d[valid_mask], temp_3d[valid_mask]) + 1000

    elif method == "TEOS10":
        # convert depths (m) to pressure (dbar)
        pressure = depths / 10.0  # 1 dbar ~ 1 m depth
        # Broadcast pressure to 3D shape
        pressure_3d = np.broadcast_to(pressure[:, None, None], temp_3d.shape)
        density_3d[valid_mask] = gsw.density.rho(sal_3d[valid_mask], temp_3d[valid_mask], pressure_3d[valid_mask])

    else:
        raise ValueError(f"Unsupported density method: {method}")

    return density_3d
###############################################################################

###############################################################################
def compute_Bmost(mask3d: np.ndarray) -> np.ndarray:
    """
    Compute the 2D array Bmost by summing the 3D mask array along its first axis.

    Parameters
    ----------
    mask3d : np.ndarray
        3D array of shape (depth, rows, cols).

    Returns
    -------
    np.ndarray
        2D array of shape (rows, cols), where each element is the sum of
        mask3d over the first dimension.

    Notes
    -----
    The function sums over axis=0, equivalent to looping over
    mask3d[:, j, i] for each j, i.

    Examples
    --------
    >>> mask3d = np.array([[[1, 0], [0, 1]],
                          [[0, 1], [1, 0]]])
    >>> compute_Bmost(mask3d)
    array([[1, 1],
           [1, 1]])
    """
    return np.sum(mask3d, axis=0).squeeze()
###############################################################################

###############################################################################
def compute_Bleast(mask3d: np.ndarray) -> np.ndarray:
    """
    Extract the first layer of a 3D mask array along the first dimension.

    Parameters
    ----------
    mask3d : np.ndarray
        3D array of shape (depth, rows, cols).

    Returns
    -------
    np.ndarray
        2D array of shape (rows, cols) corresponding to mask3d[0, :, :].

    Examples
    --------
    >>> mask3d = np.array([[[1, 0], [0, 1]],
                          [[0, 1], [1, 0]]])
    >>> compute_Bleast(mask3d)
    array([[1, 0],
           [0, 1]])
    """
    return np.squeeze(mask3d[0, :, :])
###############################################################################

###############################################################################
def filter_dense_water_masses(
    density_data: Dict[int, List[np.ndarray]],
    threshold: float = 1029.2
) -> Dict[int, List[np.ndarray]]:
    """
    Filter density data to retain only dense water masses (values >= threshold).

    Parameters
    ----------
    density_data : dict
        Dictionary {year: [12 2D arrays]} containing seawater density data.
    threshold : float, optional
        Density threshold for dense water masses (default is 29.2 kg/m³).

    Returns
    -------
    filtered_data : dict
        Dictionary with same structure as input, but non-dense values are set to np.nan.
    """
    filtered_data = {}
    
    filtered_data = {
        year: [
            np.where(density_2d >= threshold, density_2d, np.nan)
            for density_2d in monthly_arrays
        ]
        for year, monthly_arrays in density_data.items()
    }
    
    return filtered_data
###############################################################################

###############################################################################
def calc_density(
    temp_3d: np.ndarray,
    sal_3d: np.ndarray,
    depths: np.ndarray,
    density_method: str,
) -> np.ndarray:
    """
    Calculate seawater density based on temperature, salinity, and depth.

    Parameters
    ----------
    temp_3d : np.ndarray
        3D array of temperature values.
    sal_3d : np.ndarray
        3D array of salinity values.
    depths : np.ndarray
        1D array of depth levels (meters).
    density_method : str
        Density calculation method: "EOS", "EOS80", or "TEOS10".

    Returns
    -------
    np.ndarray
        3D array of calculated density values.
    """
    valid_mask = ~np.isnan(temp_3d) & ~np.isnan(sal_3d)
    density = np.full(temp_3d.shape, np.nan, dtype=np.float64)

    if density_method == "EOS":
        alpha, beta, rho0 = 0.0002, 0.0008, 1025
        density[valid_mask] = rho0 * (1 - alpha * (temp_3d[valid_mask] - 10) + beta * (sal_3d[valid_mask] - 35))
    elif density_method == "EOS80":
        density[valid_mask] = gsw.density.sigma0(sal_3d[valid_mask], temp_3d[valid_mask]) + 1000
    elif density_method == "TEOS10":
        pressure = depths[:, None, None] / 10.0  # decibar
        pressure_3d = np.broadcast_to(pressure, temp_3d.shape)
        density[valid_mask] = gsw.density.rho(sal_3d[valid_mask], temp_3d[valid_mask], pressure_3d[valid_mask])
    else:
        raise ValueError(f"Unsupported density method: {density_method}")

    return density
###############################################################################

###############################################################################
def compute_dense_water_volume(
    IDIR: Union[str, Path],
    mask3d: np.ndarray,
    filename_fragments: dict,
    density_method: str,
    dz: float = 2.0,
    dx: float = 800.0,
    dy: float = 800.0,
) -> List[Dict]:
    """
    Compute the volume of dense water (density >= 1029.2 kg/m³) over time.

    Parameters
    ----------
    IDIR : Union[str, Path]
        Directory path containing yearly subfolders with compressed NetCDF files.
    mask3d : np.ndarray
        3D boolean mask array of shape (depth, Y, X). True means masked/excluded.
    filename_fragments : dict
        Dictionary with keys 'ffrag1', 'ffrag2', 'ffrag3' to build filenames.
    density_method : str
        One of "EOS", "EOS80", or "TEOS10" for density calculation methods.
    dz : float, optional
        Vertical thickness of each layer in meters (default 2.0).
    dx : float, optional
        Horizontal grid spacing in meters along x (default 800.0).
    dy : float, optional
        Horizontal grid spacing in meters along y (default 800.0).

    Returns
    -------
    List[Dict]
        List of dictionaries with 'date' and 'volume_m3' keys for each month.
    """
    IDIR = Path(IDIR)

    print("Scanning directory to determine available years...")
    Ybeg, Yend, ysec = infer_years_from_path(IDIR, target_type="folder", pattern=r'output\s*(\d{4})')
    print(f"Found years from {Ybeg} to {Yend}: {ysec}")
    print("-" * 45)

    cell_area = dx * dy  # m²
    cell_volume = cell_area * dz  # m³ per vertical grid cell

    volume_time_series = []

    for year in ysec:
        filename = build_bfm_filename(year, filename_fragments)
        file_nc = IDIR / f"output{year}" / filename
        file_gz = Path(str(file_nc) + ".gz")

        print(f"Working on year {year}")
        print(f"Handling file {filename}")

        if not file_gz.exists():
            print(f"File missing: {file_gz}, skipping year {year}")
            continue

        temp = read_nc_variable_from_gz_in_memory(file_gz, 'votemper')
        sal = read_nc_variable_from_gz_in_memory(file_gz, 'vosaline')

        if temp.shape != sal.shape:
            raise ValueError("Temperature and salinity data shape mismatch")

        time_len, depth_len, Y, X = temp.shape

        # Broadcast mask to 4D
        mask_4d = np.broadcast_to(mask3d == 0, temp.shape)  # True means masked

        # Depth vector and masks
        depths = np.arange(depth_len) * dz
        mask_shallow = (depths > 0) & (depths <= 50)
        mask_deep = (depths > 50) & (depths <= 200)

        mask_shallow_3d = np.broadcast_to(mask_shallow[:, None, None], (depth_len, Y, X))
        mask_deep_3d = np.broadcast_to(mask_deep[:, None, None], (depth_len, Y, X))

        # Prepare 4D shallow and deep masks for all time steps by broadcasting
        mask_shallow_4d = np.broadcast_to(mask_shallow_3d, temp.shape)
        mask_deep_4d = np.broadcast_to(mask_deep_3d, temp.shape)

        # Apply masking: masked cells -> NaN
        temp = np.where(mask_4d, np.nan, temp)
        sal = np.where(mask_4d, np.nan, sal)

        # Identify invalid data over entire 4D arrays
        invalid_temp = temp_threshold(temp, mask_shallow_4d, mask_deep_4d)
        invalid_sal = hal_threshold(sal, mask_shallow_4d, mask_deep_4d)
        invalid_mask = invalid_temp | invalid_sal

        temp = np.where(invalid_mask, np.nan, temp)
        sal = np.where(invalid_mask, np.nan, sal)

        valid_mask = ~np.isnan(temp) & ~np.isnan(sal)

        # Calculate density over entire 4D arrays using calc_density
        density_4d = calc_density(temp, sal, depths, valid_mask, density_method)

        # Boolean mask of dense water cells (4D)
        dense_cells = density_4d >= 1029.2

        # Sum dense cells per time slice (axis 1,2,3)
        dense_counts = np.sum(dense_cells, axis=(1, 2, 3))

        # Calculate dense volumes per time slice
        dense_volumes = dense_counts * cell_volume

        # Collect date and volume info for each month in this year
        for month_idx in range(time_len):
            date = datetime(year, month_idx + 1, 1)
            volume = dense_volumes[month_idx]
            print(f"Dense water volume for {date.strftime('%Y-%m')}: {volume:.2f} m³")
            print("-" * 45)
            volume_time_series.append({'date': date, 'volume_m3': volume})

    return volume_time_series