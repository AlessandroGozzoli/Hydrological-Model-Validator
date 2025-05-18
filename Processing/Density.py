import gsw  # TEOS-10/EOS-80 library

def compute_density(temp_2d, sal_2d, Bmost, method):
    """
    Compute density (in kg/m³) at benthic depth for each grid cell and time step.

    Parameters:
    - temp_2d: ndarray (T, Y, X) - Temperature at bottom layer
    - sal_2d: ndarray (T, Y, X) - Salinity at bottom layer
    - Bmost: ndarray (Y, X) - Index (1-based) of deepest valid level
    - dz: float - vertical layer resolution in meters (default is 2m)
    - method: str - Method to calculate density ('ESO', 'EOS80', or 'TEOS10')

    Returns:
    - density_2d: ndarray (T, Y, X) - Seawater density in kg/m³
    """
    
    # Convert Bmost to 0-based index and then to depth (in meters)
    depth = (Bmost) * 2.0  # shape (Y, X)

    if method == "EOS":
        # Simplified density formula (ESO)
        alpha = 0.0002  # Thermal expansion coefficient
        beta = 0.0008   # Haline contraction coefficient
        rho0 = 1025     # Reference seawater density in kg/m³

        # Simplified density calculation using the ESO equation
        density = rho0 * (1 - alpha * (temp_2d - 10) + beta * (sal_2d - 35))

    elif method == 'EOS80':
        # For EOS80, use gsw's density function (specific to EOS-80)
        density = (gsw.density.sigma0(sal_2d, temp_2d))+1000  # Potential density at surface (sigma0)
        
    elif method == 'TEOS10':
        # For TEOS10, use gsw's TEOS-10 equation of state
        density = gsw.density.rho(sal_2d, temp_2d, depth)  # Use rho for density in TEOS10

    else:
        raise ValueError(f"Method {method} is not supported.")
    
    return density