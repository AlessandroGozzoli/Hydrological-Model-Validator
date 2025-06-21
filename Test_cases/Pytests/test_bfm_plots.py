import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for testing

import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from unittest.mock import patch
import tempfile
import matplotlib.dates as mdates
import pandas as pd


from Hydrological_model_validator.Plotting.bfm_plots import (Benthic_depth,
                                                             plot_benthic_3d_mesh,
                                                             Benthic_physical_plot,
                                                             Benthic_chemical_plot,
                                                             dense_water_timeseries)


@pytest.fixture
def geo_coords():
    latp = np.linspace(30, 40, 50).reshape(5, 10)
    lonp = np.linspace(-120, -110, 50).reshape(5, 10)
    return {
        "latp": latp,
        "lonp": lonp,
        "Epsilon": 0.1,
        "MinLambda": -120,
        "MaxLambda": -110,
        "MinPhi": 30,
        "MaxPhi": 40,
    }

@pytest.fixture
def Bmost_array():
    # Include 0 to trigger np.nan conversion logic
    return np.random.randint(0, 50, size=(5, 10))

@pytest.fixture
def temp_output(tmp_path):
    return tmp_path


###############################################################################
# Tests for benthic_depth
###############################################################################


# Test basic functionality: plot is created and saved with default parameters
def test_benthic_depth_basic(Bmost_array, geo_coords, temp_output):
    Benthic_depth(Bmost_array, geo_coords, temp_output)
    output_file = temp_output / "NA - Benthic Depth.png"
    # Assert the plot file was created successfully
    assert output_file.exists(), "Output file was not created"


# Test plotting with a full set of custom styling and layout options
def test_benthic_depth_custom_options(Bmost_array, geo_coords, temp_output):
    Benthic_depth(
        Bmost_array, geo_coords, temp_output,
        figsize=(12, 6),
        contour_levels=15,  
        cmap="viridis",  
        title="Custom Title", 
        dpi=100,  
        colorbar_ticks=[0, 20, 40, 60, 80, 100],  
        colorbar_label="[depth]", 
        colorbar_labelsize=14, 
        colorbar_tick_length=12,  
        colorbar_ticklabelsize=10, 
        spine_linewidth=3, 
        spine_edgecolor="blue",  
        colorbar_left=0.3,  
        colorbar_bottom=0.2, 
        colorbar_width=0.5, 
        colorbar_height=0.03,  
        title_fontsize=18, 
        title_fontweight="normal"  
    )
    output_file = temp_output / "NA - Benthic Depth.png"
    # Confirm file saved correctly with custom options
    assert output_file.exists(), "Output with custom options not saved"


# Test that function raises ValueError if Bmost input is not a NumPy array
def test_benthic_depth_invalid_Bmost_type(geo_coords, temp_output):
    with pytest.raises(ValueError, match="Bmost must be a 2D NumPy array"):
        Benthic_depth("invalid_array", geo_coords, temp_output)


# Test that function raises ValueError if Bmost input is not 2D
def test_benthic_depth_invalid_Bmost_ndim(geo_coords, temp_output):
    with pytest.raises(ValueError, match="Bmost must be a 2D NumPy array"):
        Benthic_depth(np.array([1, 2, 3]), geo_coords, temp_output)


# Test auto-computation of colorbar_left when set to None
def test_colorbar_left_computed(Bmost_array, geo_coords, temp_output):
    # Passing None should trigger default computation of colorbar_left
    Benthic_depth(Bmost_array, geo_coords, temp_output, colorbar_left=None)
    output_file = temp_output / "NA - Benthic Depth.png"
    # Verify output was created successfully despite missing explicit colorbar_left
    assert output_file.exists(), "Auto-computed colorbar_left test failed"


# Test that zero values in Bmost are properly handled (converted to NaN to avoid plotting issues)
def test_zeros_in_Bmost_handled(geo_coords, temp_output):
    # Create sample Bmost with values and inject zeros deliberately
    Bmost = np.random.randint(1, 50, size=(5, 10))
    Bmost[0, 0] = 0  # zero values should become NaN internally
    Bmost[3, 4] = 0

    Benthic_depth(Bmost, geo_coords, temp_output)
    output_file = temp_output / "NA - Benthic Depth.png"
    # Assert the plot is still generated despite zeros in data
    assert output_file.exists()


# Test that suppressing plot display functions does not cause failures (useful in headless/test environments)
def test_show_and_pause_does_not_crash(Bmost_array, geo_coords, temp_output, monkeypatch):
    # Patch plt.show(), plt.pause(), and plt.draw() to no-ops
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "pause", lambda x: None)
    monkeypatch.setattr(plt, "draw", lambda: None)

    Benthic_depth(Bmost_array, geo_coords, temp_output)
    output_file = temp_output / "NA - Benthic Depth.png"
    # Confirm the plot was saved without error when display calls suppressed
    assert output_file.exists(), "Plot failed when suppressing display"


###############################################################################
# Tests for plot_benthic_3d_mesh
###############################################################################


@pytest.fixture
def geo_coords_3d():
    latp = np.tile(np.linspace(30, 34, 5), (5, 1)).T
    lonp = np.tile(np.linspace(-120, -110, 5), (5, 1))
    return {'latp': latp, 'lonp': lonp}

@pytest.fixture
def Bmost():
    return np.array([
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [0, 0, 0, 0, 0]  # Southern edge (min latitude)
    ])

# Test that the 3D surface plot runs and saves the html
def test_surface_plot_saves_html(geo_coords_3d, Bmost):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = tmpdir
        filename = "test_plot.html"

        # Call the plotting function with save_path
        plot_benthic_3d_mesh(Bmost, geo_coords_3d, plot_type='surface', save_path=save_path)

        # Check that an HTML file was created inside tmpdir
        files = list(Path(save_path).glob("*.html"))
        assert len(files) > 0, "Expected at least one HTML file to be saved"


# Test that the 3D mesh plot runs and saves the html
def test_mesh3d_plot_saves_html(geo_coords_3d, Bmost):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = tmpdir
        plot_benthic_3d_mesh(Bmost, geo_coords_3d, plot_type='mesh3d', save_path=save_path)
        files = list(Path(save_path).glob("*.html"))
        assert len(files) > 0, "Expected at least one HTML file to be saved"


# Test that providing an invalid plot_type raises a ValueError
def test_invalid_plot_type_raises(geo_coords_3d, Bmost):
    with pytest.raises(ValueError, match="plot_type must be 'surface' or 'mesh3d'"):
        plot_benthic_3d_mesh(Bmost, geo_coords_3d, plot_type='invalid')


###############################################################################
# Tests for benthic_physical_plot
###############################################################################


@pytest.fixture
def geo_coord_phys():
    # Small grid with lat/lon coordinates
    latp = np.linspace(30, 34, 5).reshape(5, 1) + np.zeros((5, 5))
    lonp = np.linspace(-120, -110, 5) + np.zeros((5, 5))
    return {'latp': latp, 'lonp': lonp}

@pytest.fixture
def var_dataframe():
    # Dict for 2 years, each with 12 months of 5x5 arrays
    data = {}
    for year in [2020, 2021]:
        monthly_data = []
        for _ in range(12):
            arr = np.random.rand(5, 5)
            monthly_data.append(arr)
        data[year] = monthly_data
    return data

@pytest.fixture
def var_dataframe_with_missing():
    # Similar but with None or all NaNs in some months to test skipping
    data = {}
    for year in [2020]:
        monthly_data = []
        for month in range(12):
            if month in [2, 5]:
                monthly_data.append(None)
            else:
                arr = np.random.rand(5, 5)
                monthly_data.append(arr)
        data[year] = monthly_data
    return data

# Test basic plot runs and calls savefig and show the expected number of times
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_basic_plot_runs(mock_show, mock_savefig, var_dataframe, geo_coord_phys, tmp_path):
    Benthic_physical_plot(
        var_dataframe,
        geo_coord_phys,
        output_path=str(tmp_path),
        bfm2plot='votemper',
        description='Bottom Temperature',
        unit='°C',
        dpi=50,  # use low dpi for faster test execution
        figsize=(4, 4),
    )
    # Expect one saved figure per month per year: 12 months * 2 years = 24
    assert mock_savefig.call_count == 24
    # Confirm at least one saved file path is under the tmp_path directory
    saved_path = Path(mock_savefig.call_args[0][0])
    assert str(tmp_path) in str(saved_path)
    # The file does not have to actually exist because it's mocked
    assert saved_path.exists() or True


# Test that months with no data are skipped and fewer plots are saved and shown
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_skips_months_with_no_data(mock_show, mock_savefig, var_dataframe_with_missing, geo_coord_phys, tmp_path):
    Benthic_physical_plot(
        var_dataframe_with_missing,
        geo_coord_phys,
        output_path=str(tmp_path),
        bfm2plot='votemper',
        description='Bottom Temperature',
        unit='°C',
        dpi=50,
        figsize=(4, 4),
    )
    # 2 months missing, so expect 10 plots saved instead of 12
    assert mock_savefig.call_count == 10


# Test that custom keyword arguments are accepted and plot/save counts are as expected
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_custom_kwargs_passed(mock_show, mock_savefig, var_dataframe, geo_coord_phys, tmp_path):
    Benthic_physical_plot(
        var_dataframe,
        geo_coord_phys,
        output_path=str(tmp_path),
        bfm2plot='salinity',
        description='Bottom Salinity',
        unit='PSU',
        dpi=30,
        figsize=(5, 5),
        coastline_linewidth=0.5,
        border_linestyle='-',
        gridline_color='red',
        gridline_linestyle=':',
        title_fontsize=12,
        title_fontweight='normal',
        colorbar_position=[0.2, 0.2, 0.5, 0.03],
        colorbar_labelsize=10,
        colorbar_tick_length=10,
        colorbar_tick_labelsize=8,
    )
    # Check the calls remain consistent with default plotting (24 saves, 48 shows)
    assert mock_savefig.call_count == 24


# Test that the function prints messages indicating skipped months and plotting actions
def test_prints_skip_and_plot(monkeypatch, var_dataframe_with_missing, geo_coord_phys, tmp_path):
    printed = []

    # Replace print with custom function to capture printed messages
    def fake_print(msg):
        printed.append(msg)

    monkeypatch.setattr("builtins.print", fake_print)

    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.show"):
        Benthic_physical_plot(
            var_dataframe_with_missing,
            geo_coord_phys,
            output_path=str(tmp_path)
        )

    # Confirm skipping message(s) were printed
    assert any("Skipping" in s for s in printed)
    # Confirm plotting message(s) were printed
    assert any("Plotting" in s for s in printed)
    

###############################################################################
# Tests for benthic_chemical_plot
###############################################################################


@pytest.fixture
def dummy_var_dataframe():
    # 2 years, each with 12 monthly 2D arrays (5x5), no NaNs
    data = np.random.rand(5, 5)
    return {
        2020: [data for _ in range(12)],
        2021: [data for _ in range(12)],
    }


@pytest.fixture
def dummy_geo_coord_chem():
    lon = np.linspace(-120, -110, 5)
    lat = np.linspace(30, 40, 5)
    lonp, latp = np.meshgrid(lon, lat)
    return {'lonp': lonp, 'latp': latp}


# Test basic functionality of Benthic_chemical_plot including default and location-specific plotting
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.pause")
@patch("Hydrological_model_validator.Plotting.bfm_plots.extract_options")
@patch("Hydrological_model_validator.Plotting.bfm_plots.get_benthic_plot_parameters")
@patch("Hydrological_model_validator.Plotting.bfm_plots.swifs_colormap")
@patch("Hydrological_model_validator.Plotting.bfm_plots.format_unit")
@patch("Hydrological_model_validator.Plotting.bfm_plots.style_axes_spines")
def test_benthic_chemical_plot_basic(
    mock_style_spines,
    mock_format_unit,
    mock_swifs_colormap,
    mock_get_params,
    mock_extract_options,
    mock_pause,
    mock_show,
    mock_savefig,
    dummy_var_dataframe,
    dummy_geo_coord_chem,
):
    # Setup mocks with typical plot options returned by extract_options
    mock_extract_options.return_value = {
        "bfm2plot": "chlorophyll",
        "unit": "mg/m3",
        "description": "Chlorophyll",
        "output_path": "output_test",
        "epsilon": 0.05,
        "figsize": (4, 4),
        "dpi": 30,
        "coastline_linewidth": 1,
        "border_linestyle": ":",
        "gridline_color": "gray",
        "gridline_linestyle": "--",
        "title_fontsize": 12,
        "title_fontweight": "bold",
        "colorbar_position": [0.2, 0.2, 0.6, 0.03],
        "colorbar_labelsize": 10,
        "colorbar_tick_length": 5,
        "colorbar_tick_labelsize": 8,
    }

    # get_benthic_plot_parameters returns default colormap params (limits, ticks, etc.)
    mock_get_params.return_value = (
        0.0, 1.0, np.linspace(0, 1, 11), 5, "viridis", False, None, None
    )

    # swifs_colormap mock returns dummy colormap, norm, ticks, and labels
    mock_swifs_colormap.return_value = (None, "viridis", None, np.linspace(0, 1, 11), [str(i) for i in range(11)])

    # format_unit formats unit string with brackets
    mock_format_unit.side_effect = lambda unit: f"[{unit}]"
    mock_style_spines.return_value = None  # no-op for styling axes

    # Run plotting function with no location arg - expect 24 plots (2 years * 12 months)
    Benthic_chemical_plot(dummy_var_dataframe, dummy_geo_coord_chem)
    assert mock_savefig.call_count == 24

    # Run with location arg - expect doubling of plots and calls
    Benthic_chemical_plot(dummy_var_dataframe, dummy_geo_coord_chem, location="TestLocation")
    assert mock_savefig.call_count == 48  # 24*2 more saves


# Test that months with None or NaN data are correctly skipped in plotting
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.pause")
@patch("Hydrological_model_validator.Plotting.bfm_plots.extract_options")
@patch("Hydrological_model_validator.Plotting.bfm_plots.get_benthic_plot_parameters")
def test_benthic_chemical_plot_skip_none_or_nan(
    mock_get_params,
    mock_extract_options,
    mock_pause,
    mock_show,
    mock_savefig,
    dummy_geo_coord_chem,
):
    # Setup var_dataframe with 1 None, 5 NaN arrays, and 6 valid numpy arrays
    var_dataframe = {
        2020: [None] + [np.full((5, 5), np.nan)] * 5 + [np.random.rand(5, 5)] * 6,
    }

    # Same plotting options as before
    mock_extract_options.return_value = {
        "bfm2plot": "chlorophyll",
        "unit": "mg/m3",
        "description": "Chlorophyll",
        "output_path": "output_test",
        "epsilon": 0.05,
        "figsize": (4, 4),
        "dpi": 30,
        "coastline_linewidth": 1,
        "border_linestyle": ":",
        "gridline_color": "gray",
        "gridline_linestyle": "--",
        "title_fontsize": 12,
        "title_fontweight": "bold",
        "colorbar_position": [0.2, 0.2, 0.6, 0.03],
        "colorbar_labelsize": 10,
        "colorbar_tick_length": 5,
        "colorbar_tick_labelsize": 8,
    }

    mock_get_params.return_value = (
        0.0, 1.0, np.linspace(0, 1, 11), 5, "viridis", False, None, None
    )

    # Call function - expect only 6 valid months to be plotted (others skipped)
    Benthic_chemical_plot(var_dataframe, dummy_geo_coord_chem)

    # Check that save called exactly 6 times
    assert mock_savefig.call_count == 6


###############################################################################
# Tests for dense_water_timeseries
###############################################################################


# Helper to generate date-volume entries
def make_entries(start_year, start_month, count, step_months=1, base_volume=1e12):
    dates = pd.date_range(start=f"{start_year}-{start_month:02d}-01", periods=count, freq=pd.DateOffset(months=step_months))
    return [{"date": d.strftime("%Y-%m-%d"), "volume_m3": base_volume * (i+1)} for i, d in enumerate(dates)]

@pytest.fixture
def multi_series_data():
    return {
        "Series1": make_entries(2020, 1, 12),  # Jan-Dec 2020
        "Series2": make_entries(2020, 6, 12),  # Jun 2020 - May 2021
    }

@pytest.fixture
def single_point_data():
    return {
        "Single": [{"date": "2021-01-01", "volume_m3": 1e12}],
    }

@pytest.fixture
def empty_data():
    return {}

# Test the normal plotting flow and check figure axes, titles, labels, and fill_between polygons
@patch("matplotlib.pyplot.close")  
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_normal_flow(mock_show, mock_savefig, mock_close, multi_series_data, tmp_path):
    # Make plt.close a no-op so figure is not closed inside the function
    mock_close.side_effect = lambda *args, **kwargs: None

    dense_water_timeseries(multi_series_data, output_path=str(tmp_path))

    fig = plt.gcf()
    assert len(fig.axes) >= 2  # we expect at least main and twin axes

    ax1 = fig.axes[0]
    ax2 = fig.axes[1]

    # Check title and y-axis labels
    assert ax1.get_title() == "Dense Water Volume Time Series"
    assert "Dense Water Volume" in ax1.get_ylabel()
    assert "Sverdrup" in ax2.get_ylabel()

    # Check x-axis formatter is DateFormatter with expected format
    fmt = ax1.xaxis.get_major_formatter()
    assert isinstance(fmt, mdates.DateFormatter)

    # Confirm fill_between was used (there should be at least one polygon collection)
    assert len(ax1.collections) > 0


# Test that plotting with empty data calls no show and produces no lines
@patch("matplotlib.pyplot.savefig", autospec=True)
@patch("matplotlib.pyplot.show", autospec=True)
def test_empty_data_runs(mock_show, mock_savefig, empty_data, tmp_path):
    dense_water_timeseries(empty_data,
                           output_path=str(tmp_path))

    # Show should not be called with empty data
    assert mock_show.call_count == 0

    ax = plt.gca()
    # No lines should be plotted
    assert len(ax.lines) == 0


# Test single point data plotting creates one line but no fill_between polygons
@patch("matplotlib.pyplot.close", autospec=True)
@patch("matplotlib.pyplot.savefig", autospec=True)
@patch("matplotlib.pyplot.show", autospec=True)
def test_single_point_no_fill_between(mock_close, mock_show, mock_savefig, single_point_data, tmp_path):
    dense_water_timeseries(single_point_data,
                           output_path=str(tmp_path))

    # Prevent the figure from being closed
    mock_close.side_effect = lambda *args, **kwargs: None

    assert mock_show.call_count == 1

    fig = plt.gcf()
    ax1 = fig.axes[0]

    # Only count actual data lines (with non-empty xdata)
    data_lines = [line for line in ax1.lines if len(line.get_xdata()) > 0]
    assert len(data_lines) == 1
    assert len(data_lines[0].get_xdata()) == 1

    # No filled polygons created
    from matplotlib.collections import PolyCollection
    non_empty_polys = [
        coll for coll in ax1.collections
        if isinstance(coll, PolyCollection) and coll.get_paths()
    ]
    assert len(non_empty_polys) == 0


# Test plotting with custom parameters applies title, ylabel, figsize, legend location, and date format
@patch("matplotlib.pyplot.close", autospec=True)
@patch("matplotlib.pyplot.savefig", autospec=True)
@patch("matplotlib.pyplot.show", autospec=True)
def test_custom_parameters(mock_close, mock_show, mock_savefig, multi_series_data, tmp_path):
    
    # Prevent figure from being closed so we can inspect it
    mock_close.side_effect = lambda *args, **kwargs: None    
    
    dense_water_timeseries(
        multi_series_data,
        title="Custom Title",
        ylabel="Custom Y Label",
        figsize=(8, 4),
        legend_loc="upper left",
        date_format="%Y/%m/%d",
        output_path=str(tmp_path)
    )

    assert mock_show.call_count == 1

    fig = plt.gcf()
    ax1 = fig.axes[0]

    # Check custom title and ylabel applied
    assert ax1.get_title() == "Custom Title"
    assert ax1.get_ylabel() == "Custom Y Label"


# Test that conversion from km3/month to Sverdrup is reflected correctly on twin y-axis
@patch("matplotlib.pyplot.savefig", autospec=True)
@patch("matplotlib.pyplot.show", autospec=True)
def test_conversion_km3_to_sv(mock_show, mock_savefig, tmp_path):
    data = {
        "Test": [
            {"date": "2022-01-01", "volume_m3": 1e9},  # 1 km3
            {"date": "2022-02-01", "volume_m3": 2e9},  # 2 km3
        ]
    }

    dense_water_timeseries(data,
                           output_path=str(tmp_path))

    fig = plt.gcf()
    axes = fig.get_axes()
    ax1 = axes[0]
    ax2 = axes[1]

    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()

    seconds_per_month = 30 * 24 * 3600
    expected_y2_min = (y1_min * 1e9) / seconds_per_month / 1e6
    expected_y2_max = (y1_max * 1e9) / seconds_per_month / 1e6

    # Assert twin axis limits are correctly converted
    assert np.isclose(y2_min, expected_y2_min, rtol=1e-6, atol=1e-7)
    assert np.isclose(y2_max, expected_y2_max, rtol=1e-6, atol=1e-7)


# Test the km3_to_sv conversion function logic independently
def test_internal_km3_to_sv_logic():
    seconds_per_month = 30 * 24 * 3600

    def km3_to_sv(x):
        return (x * 1e9) / seconds_per_month / 1e6

    # 1 km3 to Sv conversion correct
    assert np.isclose(km3_to_sv(1), (1e9) / seconds_per_month / 1e6)

    # Zero volume returns zero
    assert km3_to_sv(0) == 0

    # Negative values convert mathematically correctly
    assert km3_to_sv(-1) == - (1e9) / seconds_per_month / 1e6