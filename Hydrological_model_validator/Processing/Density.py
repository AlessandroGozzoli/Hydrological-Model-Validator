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
        Dictionary keyed by year (int or str), each containing a list of 12 arrays
        representing bottom temperature for each month.
    salinity_data : dict
        Dictionary keyed by year, each containing a list of 12 arrays representing
        bottom salinity for each month.
    Bmost : np.ndarray
        2D array (Y, X) containing the 1-based index of the deepest valid vertical level.
    method : str
        Method to compute density. Supported options:
        - 'EOS': Linear equation of state approximation.
        - 'EOS80': EOS-80 potential density at surface.
        - 'TEOS10': TEOS-10 absolute density with pressure.
    dz : float, optional
        Vertical layer thickness in meters (default is 2.0 m).

    Returns
    -------
    density_data : dict
        Dictionary keyed by year, each containing a list of 12 arrays with computed
        density fields corresponding to the monthly bottom data.

    Raises
    ------
    ValueError
        If an unsupported method is passed or the density output shape is unexpected.

    Example
    -------
    >>> # Assume temperature_data and salinity_data are loaded dictionaries as described
    >>> # Bmost is a 2D numpy array indicating bottom layer indices
    >>> method = 'EOS80'
    >>> density = compute_density_bottom(temperature_data, salinity_data, Bmost, method)
    >>> # density is a dict {year: [12 arrays]}, each array is density at benthic depth for that month
    """
    # Verify the method parameter is supported
    if method not in {"EOS", "EOS80", "TEOS10"}:
        raise ValueError(f"Unsupported method '{method}'. Choose from: 'EOS', 'EOS80', 'TEOS10'.")

    # Convert bottom layer indices to depth by multiplying by vertical layer thickness dz
    depth = Bmost * dz
    # Initialize output dictionary to hold density results for each year and month
    density_data = {}

    # Loop over each year and its list of monthly temperature arrays
    for year, temp_list in temperature_data.items():
        density_data[year] = []

        # Loop over each month (0-based index) and its corresponding temperature 2D array
        for month_idx, temp_2d in enumerate(temp_list):
            # Fetch matching monthly salinity 2D array for the same year and month
            sal_2d = salinity_data[year][month_idx]

            # Add a new axis to create a 3D array with a single vertical level (needed for gsw functions)
            temp_3d = temp_2d[None, ...]
            sal_3d = sal_2d[None, ...]

            # Calculate density depending on the specified method:
            if method == "EOS":
                # Linear equation of state approximation parameters:
                alpha = 0.0002  # thermal expansion coefficient (1/°C)
                beta = 0.0008   # haline contraction coefficient (1/psu)
                rho0 = 1025     # reference density (kg/m³)
                # Compute density using linear approximation around T=10°C and S=35 psu
                density = rho0 * (1 - alpha * (temp_3d - 10) + beta * (sal_3d - 35))

            elif method == "EOS80":
                # Compute potential density referenced to the surface using EOS-80 (from Gibbs SeaWater toolbox)
                density = gsw.density.sigma0(sal_3d, temp_3d) + 1000  # sigma0 returns density anomaly (kg/m³ - 1000)

            elif method == "TEOS10":
                # Compute absolute density at pressure corresponding to bottom depth using TEOS-10
                # Depth is converted to pressure internally by gsw.density.rho if needed
                density = gsw.density.rho(sal_3d, temp_3d, depth)

            # Ensure the density output has the expected shape: 3D with singleton vertical dimension
            if density.ndim == 3 and density.shape[0] == 1:
                # Extract the 2D slice for the single bottom layer
                density_2d = density[0]
            else:
                # Raise error if output dimensions are unexpected to avoid silent bugs
                raise ValueError(f"Unexpected density shape: {density.shape}")

            # Append the 2D density array for this month to the output list for the year
            density_data[year].append(density_2d)

    # Return dictionary containing benthic density arrays by year and month
    return density_data
###############################################################################

###############################################################################
def compute_Bmost(mask3d: np.ndarray) -> np.ndarray:
    """
    Compute a 2D array by summing the 3D mask array along the depth axis.

    Parameters
    ----------
    mask3d : np.ndarray
        3D binary mask array with shape (depth, rows, cols), where
        valid data points are typically marked as 1, invalid as 0.

    Returns
    -------
    np.ndarray
        2D array (rows, cols) where each element is the count of valid
        depth levels (sum of mask) at that spatial location.

    Notes
    -----
    Summation is performed over the depth dimension (axis=0), equivalent
    to counting valid levels for each (row, col).

    Examples
    --------
    >>> mask3d = np.array([[[1, 0], [0, 1]],
                          [[0, 1], [1, 0]]])
    >>> compute_Bmost(mask3d)
    array([[1, 1],
           [1, 1]])
    """
    # Sum mask values along the depth axis (axis=0).
    # Since mask is binary (1 for valid, 0 for invalid), 
    # the sum at each (row, col) counts how many depth layers are valid.
    return np.sum(mask3d, axis=0).squeeze()
###############################################################################

###############################################################################
def compute_Bleast(mask3d: np.ndarray) -> np.ndarray:
    """
    Extract the first (top) layer from a 3D mask array along the depth axis.

    Parameters
    ----------
    mask3d : np.ndarray
        3D array of shape (depth, rows, cols).

    Returns
    -------
    np.ndarray
        2D array of shape (rows, cols) corresponding to the first layer mask3d[0, :, :].

    Examples
    --------
    >>> mask3d = np.array([[[1, 0], [0, 1]],
                          [[0, 1], [1, 0]]])
    >>> compute_Bleast(mask3d)
    array([[1, 0],
           [0, 1]])
    """
    # Extract the first depth layer (index 0) from the 3D mask array.
    # The squeeze removes any single-dimensional entries from the shape, 
    # but here it mainly ensures a 2D array output (rows x cols).
    return np.squeeze(mask3d[0, :, :])
###############################################################################

###############################################################################
def filter_dense_water_masses(
    density_data: Dict[int, List[np.ndarray]],
    threshold: float = 1029.2
) -> Dict[int, List[np.ndarray]]:
    """
    Filter density data to retain only dense water masses with density values 
    greater than or equal to the specified threshold.

    Parameters
    ----------
    density_data : dict
        Dictionary with keys as years (int) and values as lists of 12 2D numpy arrays,
        each representing monthly seawater density fields.
    threshold : float, optional
        Density threshold in kg/m³ for defining dense water masses.
        Values below this threshold will be masked out (default is 1029.2).

    Returns
    -------
    filtered_data : dict
        Dictionary with the same structure as input, where density values below the
        threshold are replaced with np.nan, retaining only dense water masses.

    Examples
    --------
    >>> import numpy as np
    >>> density_data = {
    ...     2000: [np.array([[1029.3, 1028.9], [1029.5, 1027.0]]) for _ in range(12)],
    ...     2001: [np.array([[1029.1, 1029.0], [1028.0, 1030.0]]) for _ in range(12)]
    ... }
    >>> filtered = filter_dense_water_masses(density_data, threshold=1029.2)
    >>> print(filtered[2000][0])
    [[1029.3    nan]
     [1029.5    nan]]
    >>> print(filtered[2001][0])
    [[   nan    nan]
     [   nan 1030.0]]
    """
    filtered_data = {
        year: [
            # For each 2D monthly density array:
            # Keep values >= threshold (dense water masses),
            # mask out (set to np.nan) all values below threshold (less dense water)
            np.where(density_2d >= threshold, density_2d, np.nan)
            for density_2d in monthly_arrays
        ]
        # Loop through each year and its corresponding list of monthly arrays
        for year, monthly_arrays in density_data.items()
    }
    
    return filtered_data
###############################################################################

###############################################################################
def calc_density(
    temp_3d: np.ndarray,
    sal_3d: np.ndarray,
    depths: np.ndarray,
    valid_mask,
    density_method: str,
) -> np.ndarray:
    """
    Calculate seawater density based on temperature, salinity, and depth using
    the specified density method.

    Parameters
    ----------
    temp_3d : np.ndarray
        3D array of temperature values (depth x lat x lon).
    sal_3d : np.ndarray
        3D array of salinity values (depth x lat x lon).
    depths : np.ndarray
        1D array of depth levels in meters (length = depth dimension of temp_3d).
    valid_mask : np.ndarray or None
        Optional mask array; if None, valid values are where temp and sal are not NaN.
        (In this function, valid_mask is redefined internally, so input is ignored.)
    density_method : str
        Method for density calculation. One of: "EOS", "EOS80", "TEOS10".

    Returns
    -------
    np.ndarray
        3D array of seawater density values (depth x lat x lon).

    Raises
    ------
    ValueError
        If density_method is not one of the supported options.

    Examples
    --------
    >>> import numpy as np
    >>> import gsw
    >>> depth_levels = np.array([0, 10, 20])
    >>> temp = np.array([
    ...     [[10, 11], [12, 13]],
    ...     [[9, 10], [11, 12]],
    ...     [[8, 9], [10, 11]]
    ... ])
    >>> sal = np.array([
    ...     [[35, 35], [35, 35]],
    ...     [[34.5, 34.5], [34.5, 34.5]],
    ...     [[34, 34], [34, 34]]
    ... ])
    >>> density = calc_density(temp, sal, depth_levels, None, "EOS")
    >>> print(density.shape)
    (3, 2, 2)
    >>> print(np.round(density, 2))
    [[[1025.    1024.8 ]
      [1025.2   1025.4 ]]

     [[1024.6   1024.4 ]
      [1024.8   1025.  ]]

     [[1024.2   1024.  ]
      [1024.4   1024.6 ]]]
    """
    # Define valid data points where neither temperature nor salinity is NaN
    valid_mask = ~np.isnan(temp_3d) & ~np.isnan(sal_3d)

    # Initialize density array with NaNs to preserve invalid points
    density = np.full(temp_3d.shape, np.nan, dtype=np.float64)

    if density_method == "EOS":
        # Linear equation of state parameters
        alpha, beta, rho0 = 0.0002, 0.0008, 1025
        # Calculate density only at valid points using linear EOS approximation
        density[valid_mask] = rho0 * (1 - alpha * (temp_3d[valid_mask] - 10) + beta * (sal_3d[valid_mask] - 35))

    elif density_method == "EOS80":
        # Use EOS-80 potential density at surface from Gibbs SeaWater (gsw) toolbox
        # Add 1000 to convert sigma0 to absolute density in kg/m^3
        density[valid_mask] = gsw.density.sigma0(sal_3d[valid_mask], temp_3d[valid_mask]) + 1000

    elif density_method == "TEOS10":
        # Convert depths (m) to pressure (dbar) assuming 1 dbar ≈ 10 m depth
        pressure = depths[:, None, None] / 10.0
        # Broadcast pressure to match 3D data shape for pointwise computation
        pressure_3d = np.broadcast_to(pressure, temp_3d.shape)
        # Compute absolute density with pressure using TEOS-10 standard from gsw
        density[valid_mask] = gsw.density.rho(sal_3d[valid_mask], temp_3d[valid_mask], pressure_3d[valid_mask])

    else:
        # Raise error if method is unsupported
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
    dens_threshold: float = 1029.2,
) -> List[Dict]:
    """
    Compute the volume of dense water masses (density >= dens_threshold kg/m³) 
    over time from oceanographic temperature and salinity data.

    Parameters
    ----------
    IDIR : Union[str, Path]
        Directory path containing yearly subfolders with compressed NetCDF files.
        Each subfolder named like 'outputYYYY' contains the data for year YYYY.
    mask3d : np.ndarray
        3D boolean mask array of shape (depth, Y, X), where True means the cell is masked/excluded.
    filename_fragments : dict
        Dictionary with keys 'ffrag1', 'ffrag2', 'ffrag3' used to construct filenames
        for the data files.
    density_method : str
        Method to calculate seawater density. Must be one of "EOS", "EOS80", or "TEOS10".
    dz : float, optional
        Vertical thickness of each grid layer in meters (default 2.0 m).
    dx : float, optional
        Horizontal grid spacing in meters along the x-axis (default 800.0 m).
    dy : float, optional
        Horizontal grid spacing in meters along the y-axis (default 800.0 m).
    dens_threshold : float, optional
        Density threshold in kg/m³ to define dense water (default 1029.2 kg/m³).

    Returns
    -------
    List[Dict]
        List of dictionaries, each containing:
        - 'date': datetime object for the first day of each month,
        - 'volume_m3': volume of dense water (in cubic meters) for that month.

    Notes
    -----
    - The function expects yearly subdirectories named as 'outputYYYY' inside IDIR.
    - Temperature and salinity are read from compressed NetCDF files.
    - Cells masked by mask3d are excluded from volume calculation.
    - Density is computed per cell per time using the specified density method.
    - The dense water volume is calculated by counting cells exceeding the density
      threshold and multiplying by the cell volume (dx * dy * dz).

    Examples
    --------
    >>> from pathlib import Path
    >>> import numpy as np
    >>> mask = np.zeros((50, 100, 100), dtype=bool)  # no masked cells
    >>> filename_fragments = {'ffrag1': 'data', 'ffrag2': 'temp', 'ffrag3': 'sal'}
    >>> IDIR = Path('/path/to/ocean_data')
    >>> dense_volumes = compute_dense_water_volume(
    ...     IDIR=IDIR,
    ...     mask3d=mask,
    ...     filename_fragments=filename_fragments,
    ...     density_method="EOS80",
    ...     dz=2.0,
    ...     dx=800.0,
    ...     dy=800.0,
    ...     dens_threshold=1029.2
    ... )
    >>> print(dense_volumes[0])
    {'date': datetime.datetime(2000, 1, 1, 0, 0), 'volume_m3': 1234567.89}
    """
    IDIR = Path(IDIR)

    # Identify years available based on folder naming convention
    print("Scanning directory to determine available years...")
    Ybeg, Yend, ysec = infer_years_from_path(IDIR, target_type="folder", pattern=r'output\s*(\d{4})')
    print(f"Found years from {Ybeg} to {Yend}: {ysec}")
    print("-" * 45)

    # Compute grid cell dimensions and volume
    cell_area = dx * dy
    cell_volume = cell_area * dz

    volume_time_series = []  # Stores output: list of dicts with volume per month

    for year in ysec:
        # Construct expected filename path for current year
        filename = build_bfm_filename(year, filename_fragments)
        file_nc = IDIR / f"output{year}" / filename
        file_gz = Path(str(file_nc) + ".gz")

        print(f"Working on year {year}")

        # Skip year if data file is missing
        if not file_gz.exists():
            print(f"File missing: {file_gz}, skipping year {year}")
            continue

        # Read temperature and salinity arrays from compressed NetCDF file
        temp = read_nc_variable_from_gz_in_memory(file_gz, 'votemper')
        sal = read_nc_variable_from_gz_in_memory(file_gz, 'vosaline')

        # Validate array shapes
        if temp.shape != sal.shape:
            raise ValueError("Temperature and salinity data shape mismatch")

        time_len, depth_len, Y, X = temp.shape

        # Apply static 3D spatial mask (broadcast to 4D to match time dimension)
        mask_4d = np.broadcast_to(mask3d == 0, temp.shape)

        # Create 1D array of depths and define logical depth ranges
        depths = np.arange(depth_len) * dz
        mask_shallow = (depths > 0) & (depths <= 50)
        mask_deep = (depths > 50) & (depths <= 200)

        # Convert shallow/deep masks to full 4D masks for later filtering
        mask_shallow_4d = np.broadcast_to(mask_shallow[:, None, None], temp.shape)
        mask_deep_4d = np.broadcast_to(mask_deep[:, None, None], temp.shape)

        # Apply mask to remove excluded spatial regions
        temp = np.where(mask_4d, np.nan, temp)
        sal = np.where(mask_4d, np.nan, sal)

        # Identify and mask invalid temperature and salinity values based on depth
        invalid_temp = temp_threshold(temp, mask_shallow_4d, mask_deep_4d)
        invalid_sal = hal_threshold(sal, mask_shallow_4d, mask_deep_4d)
        invalid_mask = invalid_temp | invalid_sal

        temp = np.where(invalid_mask, np.nan, temp)
        sal = np.where(invalid_mask, np.nan, sal)

        # Define mask of valid values for density computation
        valid_mask = ~np.isnan(temp) & ~np.isnan(sal)

        # Calculate density using the specified method
        density_4d = calc_density(temp, sal, depths, valid_mask, density_method)

        # Identify cells with density ≥ threshold (i.e., dense water)
        dense_cells = density_4d >= dens_threshold

        # Count number of dense cells for each time slice (month)
        dense_counts = np.sum(dense_cells, axis=(1, 2, 3))

        # Convert count of dense cells to volume in cubic meters
        dense_volumes = dense_counts * cell_volume

        # Build output record for each month in the year
        for month_idx in range(time_len):
            date = datetime(year, month_idx + 1, 1)
            volume = dense_volumes[month_idx]
            print(f"Dense water volume for {date.strftime('%Y-%m')}: {volume:.2f} m³")
            print("-" * 45)
            volume_time_series.append({'date': date, 'volume_m3': volume})

    return volume_time_series