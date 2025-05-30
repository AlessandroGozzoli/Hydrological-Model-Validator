import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from typing import Any, Dict, List
import pandas as pd
import itertools
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.cm as cm
import cmocean
import matplotlib.dates as mdates
import seaborn as sns
import os

# Cartopy (for map projections and geospatial features)
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .formatting import (swifs_colormap,
                         format_unit,
                         style_axes_spines,
                         get_benthic_plot_parameters,
                         cmocean_to_plotly,
                         invert_colorscale)

from .default_bfm_plot_options import (DEFAULT_BENTHIC_PLOT_OPTIONS,
                                       DEFAULT_BENTHIC_PHYSICAL_PLOT_OPTIONS,
                                       DEFAULT_BENTHIC_CHEMICAL_PLOT_OPTIONS)

from ..Processing.utils import extract_options

###############################################################################
def Benthic_depth(Bmost: np.ndarray,
                  geo_coords: dict,
                  output_path: Path,
                  **user_opts: Any) -> None:
    """
    Plot the benthic layer depth from a 2D bottom index array using Cartopy,
    with geolocalized coordinates for spatial referencing.

    Parameters
    ----------
    Bmost (np.ndarray)               : 2D array with bottom layer indices.
    geo_coords (dict)               : Dictionary with geolocalized coordinates and extents,
                                     including keys:
                                     - 'latp', 'lonp'       : 2D arrays of latitudes and longitudes.
                                     - 'Epsilon'            : Float, spatial offset for plotting.
                                     - 'MinLambda', 'MaxLambda' : Longitude bounds for extent.
                                     - 'MinPhi', 'MaxPhi'   : Latitude bounds for extent.
    output_path (str or Path)       : Required. Directory path where the figure PNG will be saved.

    kwargs (keyword arguments)
    --------------------------
    - figsize (tuple of float)          : Size of the figure in inches (default: (10, 10)).
    - projection (str)                  : Cartopy projection name as string (default: "PlateCarree").
    - contour_levels (int)              : Number of contour levels (default: 26).
    - cmap (str or Colormap)            : Colormap to use for the contour plot (default: "jet").
    - contour_extend (str)              : Extend option for contour ('both', 'neither', etc., default: "both").
    - coastline_linewidth (float)       : Line width of coastlines (default: 1.5).
    - borders_linestyle (str)           : Line style for country borders (default: ":").
    - gridline_color (str)              : Color of grid lines (default: "gray").
    - gridline_linestyle (str)          : Line style of grid lines (default: "--").
    - grid_draw_labels (bool)           : Whether to draw grid labels (default: True).
    - grid_dms (bool)                   : Display grid labels in degrees, minutes, seconds (default: True).
    - grid_x_inline (bool)              : Whether x-axis grid labels are inline (default: False).
    - grid_y_inline (bool)              : Whether y-axis grid labels are inline (default: False).
    - colorbar_width (float)            : Width of the colorbar (default: 0.65).
    - colorbar_height (float)           : Height of the colorbar (default: 0.025).
    - colorbar_left (float or None)    : Left position of the colorbar axes (default: None, computed automatically).
    - colorbar_bottom (float)           : Bottom position of the colorbar axes (default: 0.175).
    - colorbar_label (str)              : Label for the colorbar (default: "[m]").
    - colorbar_labelsize (int)          : Font size for colorbar label (default: 12).
    - colorbar_tick_length (int)        : Length of colorbar ticks (default: 18).
    - colorbar_tick_labelsize (int)     : Font size for colorbar tick labels (default: 10).
    - colorbar_ticks (list or None)     : List of ticks on colorbar (default: None, computed automatically).
    - spine_linewidth (float)           : Line width for plot spines (default: 2).
    - spine_edgecolor (str)             : Edge color for plot spines (default: "black").
    - title (str)                      : Title of the plot (default: "Benthic Layer Depth").
    - title_fontsize (int)              : Font size of the title (default: 16).
    - title_fontweight (str)            : Font weight of the title (default: "bold").
    - dpi (int)                        : DPI resolution for saved figure (default: 150).
    - filename (str)                   : Filename for saved plot (default: "NA - Benthic Depth.png").

    Returns
    -------
    None
        Saves the generated plot to the specified output path and displays it briefly.

    Example
    -------
    >>> from pathlib import Path
    >>> import numpy as np
    >>> geo_coords = {
    ...     "latp": np.linspace(30, 40, 50).reshape(5,10),
    ...     "lonp": np.linspace(-120, -110, 50).reshape(5,10),
    ...     "Epsilon": 0.1,
    ...     "MinLambda": -120,
    ...     "MaxLambda": -110,
    ...     "MinPhi": 30,
    ...     "MaxPhi": 40,
    ... }
    >>> Bmost = np.random.randint(1, 50, size=(5, 10))
    >>> output_dir = Path("./figures")
    >>> Benthic_depth(Bmost, geo_coords, output_dir,
    ...              figsize=(12, 8),
    ...              cmap="viridis",
    ...              title="Custom Benthic Depth",
    ...              dpi=200)
    """
    if not isinstance(Bmost, np.ndarray) or Bmost.ndim != 2:
        raise ValueError("Bmost must be a 2D NumPy array")

    # Extract options using your default extract_options function
    options = extract_options(user_opts, DEFAULT_BENTHIC_PLOT_OPTIONS)

    # Compute dynamic defaults
    if options['colorbar_left'] is None:
        options['colorbar_left'] = (1 - options['colorbar_width']) / 2

    Bmost_depth = np.where(Bmost == 0, np.nan, Bmost * 2)
    cmap = getattr(cmocean.cm, 'deep')  # get colormap directly by attribute

    latp = geo_coords['latp']
    lonp = geo_coords['lonp']
    epsilon = geo_coords.get('Epsilon', 0.06)
    min_lon, max_lon = geo_coords['MinLambda'], geo_coords['MaxLambda']
    min_lat, max_lat = geo_coords['MinPhi'], geo_coords['MaxPhi']

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat + epsilon, max_lat], crs=ccrs.PlateCarree())

    contour_levels = np.linspace(np.nanmin(Bmost_depth), np.nanmax(Bmost_depth), 26)
    contour = ax.contourf(
        lonp + (0.4 * epsilon), latp + (0.1 * epsilon), Bmost_depth,
        levels=contour_levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree()
    )

    ax.coastlines(linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.set_title("Benthic Layer Depth", fontsize=options['title_fontsize'], fontweight='bold')

    ax.gridlines(
        draw_labels=True, dms=True, x_inline=False, y_inline=False,
        color='gray', linestyle='--'
    )

    cbar_ax = plt.gcf().add_axes([
        options['colorbar_left'],
        options['colorbar_bottom'],
        options['colorbar_width'],
        options['colorbar_height'],
    ])

    norm = mcolors.BoundaryNorm(
        np.linspace(np.nanmin(Bmost_depth), np.nanmax(Bmost_depth), 11), contour.cmap.N
    )
    cbar = plt.colorbar(contour, cax=cbar_ax, orientation='horizontal', norm=norm, extend='both')
    cbar.set_label('[m]', fontsize=options['colorbar_labelsize'])

    if options['colorbar_ticks'] is None:
        ticks = np.linspace(np.nanmin(Bmost_depth), np.nanmax(Bmost_depth), 6).astype(int)
    else:
        ticks = options['colorbar_ticks']
    cbar.set_ticks(ticks)

    cbar.ax.tick_params(direction='in', length=options['colorbar_tick_length'], labelsize=options['colorbar_ticklabelsize'])

    style_axes_spines(ax, linewidth=options['spine_linewidth'], edgecolor=options['spine_edgecolor'])

    filename = "NA - Benthic Depth.png"
    save_path = Path(output_path, filename)
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(2)
    plt.close()
###############################################################################

###############################################################################
def plot_benthic_3d_mesh(Bmost, geo_coords, layer_thickness=2, plot_type='surface', save_path=None):
    """
    Plot 3D surface mesh of benthic depth with interactive rotation.

    Parameters:
    -----------
    Bmost : np.ndarray (2D)
        Bottom layer indices array.
    geo_coords : dict
        Dictionary with 'latp' and 'lonp' 2D arrays for coordinates.
    layer_thickness : float
        Thickness of each layer (default=2).
    plot_type : str
        'surface' or 'mesh3d' (default='surface').

    Returns:
    --------
    None
        Shows an interactive 3D plot.
    """
    latp = geo_coords['latp']
    lonp = geo_coords['lonp']

    # Convert Bmost indices to depth (negative so z increases downward)
    depth = -Bmost * layer_thickness
    
    plotly_cscale = cmocean_to_plotly('deep')
    plotly_cscale_inverted = invert_colorscale(plotly_cscale)

    if plot_type == 'surface':
        depth = depth.astype(float)

        # Mask the southern edge (minimum latitude) by setting depths to NaN
        min_lat = np.min(latp)
        tol = 1e-6
        depth_masked = depth.copy()
        depth_masked[np.abs(latp - min_lat) < tol] = np.nan

        fig = go.Figure(data=[go.Surface(
            x=lonp,
            y=latp,
            z=depth_masked,
            colorscale=plotly_cscale_inverted,
            colorbar=dict(title='Depth (m)'),
            showscale=True,
        )])

        fig.update_layout(
            title='3D Basin Depth Surface',
            scene=dict(
                xaxis_title='Longitude (°)',
                yaxis_title='Latitude (°)',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange=True),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

    elif plot_type == 'mesh3d':
        x = lonp.flatten()
        y = latp.flatten()
        z = depth.flatten()

        # Identify the southern boundary (minimum latitude)
        min_lat = np.min(latp)
        tol = 1e-6

        # Normalize depth values for color mapping
        norm = mcolors.Normalize(vmin=np.min(z), vmax=np.max(z))
        colormap = cm.get_cmap('viridis')

        vertex_colors = []
        for yi, zi in zip(y, z):
            if abs(yi - min_lat) < tol:
                vertex_colors.append('rgba(0,0,0,0)')  # Transparent
            else:
                rgba = colormap(norm(zi))
                rgba_255 = tuple(int(255 * c) for c in rgba[:3]) + (255,)
                vertex_colors.append(f'rgba{rgba_255}')

        fig = go.Figure(data=[go.Mesh3d(
            x=x,
            y=y,
            z=z,
            vertexcolor=vertex_colors,
            opacity=1.0,
            flatshading=True,
            showscale=False,
        )])

        fig.update_layout(
            title='3D Basin Depth Mesh',
            scene=dict(
                xaxis_title='Longitude (°)',
                yaxis_title='Latitude (°)',
                zaxis_title='Depth (m)',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

    else:
        raise ValueError("plot_type must be 'surface' or 'mesh3d'")

    fig.show()

    # Save to file if path provided
    # Set filename
    filename = f"3D Basin Depth {plot_type}.html"

    # Optionally allow user-defined folder
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = Path(save_path, filename)

        pio.write_html(fig, filename, auto_open=True)
###############################################################################

###############################################################################
def Benthic_physical_plot(var_dataframe: dict, geo_coord: dict, **kwargs) -> None:
    """
    Plot benthic variable maps (e.g., temperature, salinity) across all years and months.

    Parameters
    ----------
    var_dataframe (dict)             : Dict of years (int) each containing a list of 12 monthly 2D arrays (Y, X).
    geo_coord (dict)                 : Dictionary with geolocalized coordinates, keys:
                                     - 'lonp' (2D array of longitudes)
                                     - 'latp' (2D array of latitudes)
    kwargs (keyword arguments)
    --------------------------
    - bfm2plot (str)                 : Variable code to plot (default: 'votemper').
    - unit (str)                    : Unit of measurement (default: '°C').
    - description (str)             : Full variable description for title (default: 'Bottom Temperature').
    - output_path (str or Path)     : Directory path where figures are saved (default: 'output').
    - figsize (tuple of float)      : Figure size in inches (default: (10, 10)).
    - dpi (int)                    : Resolution of the saved figure (default: 150).
    - coastline_linewidth (float)   : Width of coastlines (default: 2).
    - border_linestyle (str)        : Linestyle for country borders (default: ':').
    - gridline_color (str)          : Gridline color (default: 'gray').
    - gridline_linestyle (str)      : Gridline linestyle (default: '--').
    - title_fontsize (int)          : Font size of the plot title (default: 16).
    - title_fontweight (str)        : Font weight of the plot title (default: 'bold').
    - colorbar_position (list)      : Colorbar axes position [left, bottom, width, height] (default: [0.175, 0.175, 0.65, 0.025]).
    - colorbar_labelsize (int)      : Font size for colorbar label (default: 14).
    - colorbar_tick_length (int)    : Length of colorbar ticks (default: 18).
    - colorbar_tick_labelsize (int) : Font size of colorbar tick labels (default: 10).

    Returns
    -------
    None
        Saves plots for all months and years under the specified output directory.
    """

    # You should have this function defined elsewhere to extract defaults from kwargs
    opts = extract_options(kwargs, DEFAULT_BENTHIC_PHYSICAL_PLOT_OPTIONS)

    bfm2plot = opts["bfm2plot"]
    unit = opts["unit"]
    description = opts["description"]
    output_path = Path(opts["output_path"])
    figsize = opts["figsize"]
    dpi = opts["dpi"]
    coastline_linewidth = opts["coastline_linewidth"]
    border_linestyle = opts["border_linestyle"]
    gridline_color = opts["gridline_color"]
    gridline_linestyle = opts["gridline_linestyle"]
    title_fontsize = opts["title_fontsize"]
    title_fontweight = opts["title_fontweight"]
    colorbar_position = opts["colorbar_position"]
    colorbar_labelsize = opts["colorbar_labelsize"]
    colorbar_tick_length = opts["colorbar_tick_length"]
    colorbar_tick_labelsize = opts["colorbar_tick_labelsize"]

    lonp, latp = geo_coord["lonp"], geo_coord["latp"]
    MinLambda, MaxLambda = lonp.min(), lonp.max()
    MinPhi, MaxPhi = latp.min(), latp.max()
    epsilon = 0.06

    # Select plotting parameters based on variable type
    vmin, vmax, levels, num_ticks, cmap, _, _ = get_benthic_plot_parameters(
        bfm2plot, var_dataframe, opts
    )

    # Validate num_ticks to avoid errors
    if num_ticks is None or num_ticks < 2:
        num_ticks = 5  # default fallback

    extend = "both"
    timestamp = datetime.now().strftime("run_%Y-%m-%d")

    for year, month_idx in itertools.product(var_dataframe.keys(), range(12)):
        data2D = var_dataframe[year][month_idx]
        if data2D is None or np.all(np.isnan(data2D)):
            print(f"Skipping {year}-{month_idx+1:02d}: no data")
            continue

        month_name = datetime(1900, month_idx + 1, 1).strftime("%B")
        print(f"Plotting {description} for {month_name} {year}")

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent([MinLambda, MaxLambda, MinPhi, MaxPhi], crs=ccrs.PlateCarree())

        ax.contourf(
            lonp + 0.4 * epsilon,
            latp + 0.2 * epsilon,
            data2D,
            levels=levels,
            cmap=cmap if isinstance(cmap, str) else cmap,
            vmin=vmin,
            vmax=vmax,
            extend=extend,
            transform=ccrs.PlateCarree(),
        )

        if bfm2plot == "dense_water":
            mask = ~np.isnan(data2D)
            ax.contour(
                lonp + 0.025,
                latp + 0.015,
                mask.astype(float),
                levels=[0.5],
                colors="black",
                linewidths=1.5,
                transform=ccrs.PlateCarree(),
            )

        ax.coastlines(linewidth=coastline_linewidth)
        ax.add_feature(cfeature.BORDERS, linestyle=border_linestyle)
        gl = ax.gridlines(draw_labels=True, dms=True, color=gridline_color, linestyle=gridline_linestyle)
        gl.top_labels = False
        gl.right_labels = False

        ax.set_title(f"{description} | {year} - {month_name}", fontsize=title_fontsize, fontweight=title_fontweight)

        # Prepare colormap object correctly
        if isinstance(cmap, str):
            cmap_obj = plt.get_cmap(cmap)
        else:
            cmap_obj = cmap

        norm = BoundaryNorm(levels, ncolors=cmap_obj.N)
        mappable = ScalarMappable(norm=norm, cmap=cmap_obj)

        cbar_ax = fig.add_axes(colorbar_position)
        cbar = plt.colorbar(mappable, cax=cbar_ax, orientation="horizontal", extend=extend)

        # Format unit string for label (you should have format_unit defined elsewhere)
        field_units = format_unit(unit)
        cbar.set_label(rf"$\left[{field_units[1:-1]}\right]$", fontsize=colorbar_labelsize)

        ticks = np.linspace(vmin, vmax, num_ticks)
        cbar.set_ticks(ticks)
        cbar.ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])
        cbar.ax.tick_params(direction="in", length=colorbar_tick_length, labelsize=colorbar_tick_labelsize)

        style_axes_spines(ax, linewidth=2, edgecolor="black")  # you should have this function defined elsewhere

        save_dir = output_path / timestamp / str(year)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"Benthic_{bfm2plot}_{year}_{month_name}.png"
        plt.savefig(save_dir / filename, bbox_inches="tight", dpi=dpi)

        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)

    print('-' * 45)
###############################################################################    

###############################################################################
def Benthic_chemical_plot(var_dataframe, geo_coord, location=None, **kwargs):
    """
    Plot benthic variable maps (e.g., temperature, chlorophyll) across all years and months.

    Parameters
    ----------
    var_dataframe : dict
        Dict of years (int), each containing a list of 12 monthly 2D arrays (Y, X).
    geo_coord : dict
        Dictionary with geolocalized coordinates, keys:
          - 'lonp' (2D array of longitudes)
          - 'latp' (2D array of latitudes)
    kwargs : keyword arguments
        See docstring for detailed options and defaults.

    Returns
    -------
    None
        Saves plots for all months and years under the specified output directory.
    """

    # Extract plotting options with defaults (you need to implement extract_options and define your defaults)
    opts = extract_options(kwargs, DEFAULT_BENTHIC_CHEMICAL_PLOT_OPTIONS, prefix="")

    bfm2plot = opts["bfm2plot"]
    unit = opts["unit"]
    description = opts["description"]
    output_path = Path(opts["output_path"])
    epsilon = opts["epsilon"]
    figsize = opts["figsize"]
    dpi = opts["dpi"]
    coastline_linewidth = opts["coastline_linewidth"]
    border_linestyle = opts["border_linestyle"]
    gridline_color = opts["gridline_color"]
    gridline_linestyle = opts["gridline_linestyle"]
    title_fontsize = opts["title_fontsize"]
    title_fontweight = opts["title_fontweight"]
    colorbar_position = opts["colorbar_position"]
    colorbar_labelsize = opts["colorbar_labelsize"]
    colorbar_tick_length = opts["colorbar_tick_length"]
    colorbar_tick_labelsize = opts["colorbar_tick_labelsize"]

    timestamp = datetime.now().strftime("run_%Y-%m-%d")

    lonp, latp = geo_coord['lonp'], geo_coord['latp']
    extent = [lonp.min(), lonp.max(), latp.min(), latp.max()]

    # Get plotting parameters, including whether to use a custom colormap
    (vmin, vmax, levels, num_ticks, 
     cmap_name, use_custom_cmap, hypoxia_threshold, 
     hyperoxia_threshold) = get_benthic_plot_parameters(
        bfm2plot, var_dataframe, opts
    )

    for year, monthly_data in var_dataframe.items():
        for month_idx, data2D in enumerate(monthly_data):
            if data2D is None or np.all(np.isnan(data2D)):
                continue

            month_name = datetime(1900, month_idx + 1, 1).strftime('%B')
            print(f"Plotting {month_name}, year {year}...")

            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            if use_custom_cmap:
                # Assume swifs_colormap returns (_, cmap_obj, norm, ticks, tick_labels)
                _, cmap_obj, norm, ticks, tick_labels = swifs_colormap(data2D, bfm2plot)
                cs = ax.contourf(
                    lonp + 0.4 * epsilon, latp + 0.2 * epsilon, data2D,
                    levels=ticks, cmap=cmap_obj, norm=norm, extend='both', transform=ccrs.PlateCarree()
                )
            else:
                cmap_obj = plt.get_cmap(cmap_name)
                cs = ax.contourf(
                    lonp + 0.4 * epsilon, latp + 0.2 * epsilon, data2D,
                    levels=levels, cmap=cmap_obj, vmin=vmin, vmax=vmax, extend='both', transform=ccrs.PlateCarree()
                )

            ax.coastlines(linewidth=coastline_linewidth)
            ax.add_feature(cfeature.BORDERS, linestyle=border_linestyle)
            
            title_str = f"{description} | {year} - {month_name}"
            if location:
                title_str = f"{location} " + title_str

            ax.set_title(title_str, fontsize=title_fontsize, fontweight=title_fontweight)

            gl = ax.gridlines(draw_labels=True, dms=True, color=gridline_color, linestyle=gridline_linestyle)
            gl.top_labels = False
            gl.right_labels = False

            # Colorbar
            cbar_ax = fig.add_axes(colorbar_position)
            if use_custom_cmap:
                cbar = plt.colorbar(cs, cax=cbar_ax, orientation='horizontal', extend='both')
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)
            else:
                norm = BoundaryNorm(levels, ncolors=cmap_obj.N)
                mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
                cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='horizontal', extend='both')
                cbar.set_ticks(np.linspace(vmin, vmax, num_ticks or 15))

            # Add hypoxia threshold marker if applicable
                if hypoxia_threshold is not None and vmin < hypoxia_threshold < vmax:
                    # Calculate normalized position on colorbar axis
                    # Note: cbar_ax.get_xlim() gives the x-axis limits of the colorbar axes
                    x_min, x_max = cbar_ax.get_xlim()
                    norm_val = (hypoxia_threshold - vmin) / (vmax - vmin)
                    x_pos = x_min + norm_val * (x_max - x_min)

                    # Draw vertical dashed red line at hypoxia threshold
                    cbar_ax.axvline(x_pos, color='red', linestyle='--', linewidth=2)
                    # Add text label slightly above the line
                    cbar_ax.text(x_pos, 1.25, 'Hypoxia', color='red', ha='center', va='bottom', fontsize=10, rotation=0)
                    cbar_ax.text(x_pos, -1.25, '62.5', color='red', ha='center', va='bottom', fontsize=10, rotation=0)
                    
                if hyperoxia_threshold is not None and vmin < hyperoxia_threshold < vmax:
                    # Calculate normalized position on colorbar axis
                    # Note: cbar_ax.get_xlim() gives the x-axis limits of the colorbar axes
                    x_min, x_max = cbar_ax.get_xlim()
                    norm_val = (hyperoxia_threshold - vmin) / (vmax - vmin)
                    x_pos = x_min + norm_val * (x_max - x_min)

                    # Draw vertical dashed red line at hypoxia threshold
                    cbar_ax.axvline(x_pos, color='#B8860B', linestyle='--', linewidth=2)
                    # Add text label slightly above the line
                    cbar_ax.text(x_pos, 1.25, 'Hyperoxia', color='#B8860B', ha='center', va='bottom', fontsize=10, rotation=0)
                    cbar_ax.text(x_pos, -1.25, '312.5', color='#B8860B', ha='center', va='bottom', fontsize=10, rotation=0)
                    
            cbar.ax.tick_params(direction='in', length=colorbar_tick_length, labelsize=colorbar_tick_labelsize)

            # Clean unit string for LaTeX
            units_clean = format_unit(unit)[1:-1]
            cbar.set_label(rf'$\left[{units_clean}\right]$', fontsize=colorbar_labelsize)

            style_axes_spines(ax, linewidth=2, edgecolor="black")

            # Create output directory and save figure
            save_dir = output_path / timestamp / str(year)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"Benthic_{bfm2plot}_{year}_{month_name}.png", bbox_inches='tight', dpi=dpi)

            plt.show(block=False)
            plt.pause(2)
            plt.close(fig)

    print('-' * 45)
###############################################################################    

###############################################################################    
def dense_water_timeseries(
    data_lists: Dict[str, List[dict]],
    title: str = "Dense Water Volume Time Series",
    xlabel: str = "Time",
    ylabel: str = "Dense Water Volume (km³)",
    figsize: tuple = (14, 6),
    legend_loc: str = "best",
    date_format: str = "%Y-%m"
):
    sns.set(style="whitegrid", context='notebook')
    sns.set_style("ticks")
    
    plt.figure(figsize=figsize)

    combined_df = pd.DataFrame()
    for label, data_list in data_lists.items():
        dates = [entry['date'] for entry in data_list]
        volumes_km3 = [entry['volume_m3'] / 1e9 for entry in data_list]  # km³
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'volume_km3': volumes_km3,
            'series': label
        })
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    ax1 = plt.gca()
    sns.lineplot(data=combined_df, x='date', y='volume_km3', hue='series', marker='o', ax=ax1)

    # Fill area between December and June for each year and each series
    for label in combined_df['series'].unique():
        df_series = combined_df[combined_df['series'] == label].copy()
        df_series['year'] = df_series['date'].dt.year
        df_series['month'] = df_series['date'].dt.month

        # For each year, find Dec (prev year) to June (current year)
        years = df_series['year'].unique()
        for year in years:
            # Define start and end dates for fill region
            start_date = pd.Timestamp(year=year-1, month=12, day=1)
            end_date = pd.Timestamp(year=year, month=6, day=30)

            # Select data in this range
            mask_fill = (df_series['date'] >= start_date) & (df_series['date'] <= end_date)
            df_fill = df_series.loc[mask_fill]

            if len(df_fill) < 2:
                # Not enough points to fill
                continue

            # Fill between with alpha for transparency, red color
            ax1.fill_between(df_fill['date'], df_fill['volume_km3'], color='purple', alpha=0.15)

    ax1.set_xlabel('')
    ax1.set_ylabel(ylabel)
    ax1.grid(True)
    
    ax1.set_title(title, fontsize=18, fontweight='bold')

    # Second y-axis for Sverdrup
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', colors='black')
    ax2.spines['right'].set_color('black')
    ax2.grid(True, axis='y', linestyle='--', color='gray', alpha=0.7)

    seconds_per_month = 30 * 24 * 3600
    def km3_to_sv(x):
        return (x * 1e9) / seconds_per_month / 1e6

    y1_lim = ax1.get_ylim()
    y2_lim = (km3_to_sv(y1_lim[0]), km3_to_sv(y1_lim[1]))
    ax2.set_ylim(y2_lim)
    ax2.set_ylabel("Dense Water Volume (Sverdrup)")

    ax1.legend(loc=legend_loc)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
###############################################################################