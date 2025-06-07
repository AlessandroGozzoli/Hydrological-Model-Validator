from itertools import cycle
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap
import cmocean
from cmocean import cm
from matplotlib.cm import get_cmap

from Hydrological_model_validator.Plotting.formatting import (
    plot_line,
    compute_geolocalized_coords,
    swifs_colormap,
    get_benthic_plot_parameters,
    cmocean_to_plotly,
    invert_colorscale,
    custom_oxy,
    format_unit,
    get_variable_label_unit,
    fill_annular_region,
    get_min_max_for_identity_line,
    style_axes_spines
)

###############################################################################
# format_unit tests
###############################################################################


# Test formatting of unit string with numerator and denominator (fraction)
def test_format_unit_fraction():
    # Check that a unit with "/" is correctly converted to LaTeX fraction format with subscript if present
    assert format_unit("O2/m3") == '$\\frac{O_{2}}{m^{3}}$'

# Test formatting of unit string with no denominator (simple subscript only)
def test_format_unit_no_denominator():
    # Check that units without denominator are formatted with subscripts only (no fraction)
    assert format_unit("PO4") == '$PO_{4}$'

# Test formatting of unit string containing LaTeX special characters
def test_format_unit_with_Latex_chars():
    # Check that common unit with slash (e.g. mg/L) is formatted correctly as LaTeX fraction
    assert format_unit("mg/L") == '$\\frac{mg}{L}$'

# Test that invalid (empty) input raises a ValueError
def test_format_unit_invalid_input():
    # Ensure the function properly rejects empty strings to avoid meaningless formatting
    with pytest.raises(ValueError):
        format_unit("")


###############################################################################
# get_variable_label tests
###############################################################################        


# Test known variable SST returns correct label and unit
def test_known_variable_SST():
    # Check that "SST" maps to the correct descriptive label and unit string
    label, unit = get_variable_label_unit("SST")
    assert label == "Sea Surface Temperature"
    assert unit == "[$°C$]"

# Test known variable CHL_L3 returns correct label and unit with fraction formatting
def test_known_variable_CHL_L3():
    # Check that "CHL_L3" maps correctly including LaTeX formatting for units with fractions
    label, unit = get_variable_label_unit("CHL_L3")
    assert label == "Chlorophyll (Level 3)"
    assert unit == "[$\\frac{mg}{m^{3}}$]"

# Test unknown variable returns variable name as label and empty unit
def test_unknown_variable():
    # Ensure unknown variables fallback to variable name as label and empty unit string
    label, unit = get_variable_label_unit("XYZ")
    assert label == "XYZ"
    assert unit == ""

# Test invalid input (empty or whitespace) raises ValueError
def test_invalid_input_raises():
    # Confirm that input with only whitespace or empty string raises an error to prevent invalid lookups
    with pytest.raises(ValueError):
        get_variable_label_unit("   ")

 
###############################################################################
# test fill_anular_region
###############################################################################


# Test fill_annular_region works correctly with valid input
def test_fill_annular_valid_input():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # Ensure function completes without error given valid polar Axes and proper radii
    fill_annular_region(ax, 1.0, 2.0, "blue", 0.5)

# Test that passing a non-Axes object raises ValueError
def test_invalid_ax_type():
    # The function expects a matplotlib Axes object; other types are invalid
    with pytest.raises(ValueError):
        fill_annular_region("not_an_axes", 1.0, 2.0, "blue")

# Test that providing inner radius larger than outer radius raises ValueError
def test_invalid_radii_order():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # Inner radius must be less than outer radius to define a valid annular region
    with pytest.raises(ValueError):
        fill_annular_region(ax, 2.0, 1.0, "red")

# Test that alpha outside [0,1] range raises ValueError
def test_invalid_alpha_range():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # Transparency (alpha) must be between 0 and 1 inclusive
    with pytest.raises(ValueError):
        fill_annular_region(ax, 0.5, 1.5, "red", alpha=1.5)
        
# Tests for input validation
@pytest.mark.parametrize("ax", [None, "not_an_axes", 123])
def test_invalid_ax(ax):
    with pytest.raises(ValueError, match=".*Input 'ax' must be a matplotlib.axes.Axes.*"):
        fill_annular_region(ax, r_in=1, r_out=2, color="blue")

@pytest.mark.parametrize("r_in", [-1, "zero", None])
def test_invalid_r_in(r_in):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    with pytest.raises(ValueError, match=".*Input 'r_in' must be a non-negative number.*"):
        fill_annular_region(ax, r_in=r_in, r_out=2, color="blue")

@pytest.mark.parametrize("r_out, r_in", [(0.5, 1.0), ("2", 1.0), (None, 0.0)])
def test_invalid_r_out(r_out, r_in):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    with pytest.raises(ValueError, match=".*Input 'r_out' must be a number greater than or equal to 'r_in'.*"):
        fill_annular_region(ax, r_in=r_in, r_out=r_out, color="blue")

@pytest.mark.parametrize("alpha", [-0.1, 1.5, "opaque", None])
def test_invalid_alpha(alpha):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    with pytest.raises(ValueError, match=".*Input 'alpha' must be a number between 0 and 1.*"):
        fill_annular_region(ax, r_in=1.0, r_out=2.0, color="blue", alpha=alpha)

@pytest.mark.parametrize("color", [None, 123, ['red']])
def test_invalid_color(color):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    with pytest.raises(ValueError, match=".*Input 'color' must be a string.*"):
        fill_annular_region(ax, r_in=1.0, r_out=2.0, color=color)


###############################################################################
# get_min_max tests
###############################################################################        
        

# Test basic case with positive finite values
def test_basic_min_max():
    # The min should be the smallest value among all inputs, max the largest; both finite and positive
    assert get_min_max_for_identity_line([1, 2], [3, 4]) == (1.0, 4.0)

# Test handling of NaNs and infinite values (should ignore them in min/max calculation)
def test_with_nans_and_infs():
    # NaNs and infinities are ignored; min/max computed only from finite values
    assert get_min_max_for_identity_line([np.nan, 2], [1, np.inf]) == (1.0, 2.0)

# Test with negative and positive finite values combined
def test_negative_and_positive_values():
    # Negative and positive values together test correct min/max spanning negative to positive range
    assert get_min_max_for_identity_line([-10, 0], [5]) == (-10.0, 5.0)

# Test that function raises ValueError if all input values are non-finite
def test_all_nonfinite_values():
    # All values non-finite means no valid min/max can be computed; function should raise error
    with pytest.raises(ValueError):
        get_min_max_for_identity_line([np.nan], [np.inf])


###############################################################################
# test for style_axes_spine
###############################################################################        


# Test that style_axes_spines correctly applies linewidth and edgecolor to all spines
def test_style_axes_spines_valid():
    fig, ax = plt.subplots()
    style_axes_spines(ax, linewidth=3, edgecolor='red')
    # Check all spines have the intended linewidth to ensure uniform spine styling
    for spine in ax.spines.values():
        assert spine.get_linewidth() == 3
        # Confirm edgecolor is correctly converted to RGBA tuple for 'red'
        assert spine.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)  # RGBA for 'red'

# Test that invalid linewidth values (e.g. 0) raise ValueError
def test_invalid_linewidth():
    fig, ax = plt.subplots()
    # Linewidth must be positive; zero or negative should raise error to avoid invisible spines
    with pytest.raises(ValueError):
        style_axes_spines(ax, linewidth=0)

# Test that passing an invalid edgecolor type raises ValueError
def test_invalid_edgecolor_type():
    fig, ax = plt.subplots()
    # Edgecolor must be a color specifier string or tuple, not a number, to avoid type errors
    with pytest.raises(ValueError):
        style_axes_spines(ax, edgecolor=123)
        
# Test for input validation
@pytest.mark.parametrize("ax", [None, "not_an_axes", 42])
def test_invalid_ax_spine(ax):
    with pytest.raises(ValueError, match=".*Input 'ax' must be a matplotlib.axes.Axes instance.*"):
        style_axes_spines(ax=ax)


@pytest.mark.parametrize("linewidth", [0, -1, "thick", None])
def test_invalid_linewidth_spine(linewidth):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=".*Input 'linewidth' must be a positive number.*"):
        style_axes_spines(ax=ax, linewidth=linewidth)


@pytest.mark.parametrize("edgecolor", [None, 123, ['black']])
def test_invalid_edgecolor(edgecolor):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=".*Input 'edgecolor' must be a string.*"):
        style_axes_spines(ax=ax, edgecolor=edgecolor)


###############################################################################
# test plot_line
###############################################################################


# Test plot_line with a pandas Series input, expecting one line added to the axes
def test_plot_line_with_series():
    fig, ax = plt.subplots()
    series = pd.Series([1, 2, 3])
    # Using a cycle of colors to allow repeated use of the same palette without exhaustion
    plot_line("mod", series, ax, {"mod": "Model"}, cycle(["blue"]), 2)
    # Confirm exactly one line was added to the axes after plotting the series data
    assert len(ax.lines) == 1

# Test plot_line with a list input, expecting one line added to the axes
def test_plot_line_with_list_input():
    fig, ax = plt.subplots()
    # Similar to series test, but with a list to verify plot_line handles different data types
    plot_line("obs", [1, 2, 3], ax, {"obs": "Observation"}, cycle(["green"]), 1.5)
    # Ensure a single line was drawn for the provided list input
    assert len(ax.lines) == 1

# Test that plot_line raises ValueError when given an invalid color palette (non-iterable)
def test_invalid_color_palette():
    fig, ax = plt.subplots()
    # Color palette must be iterable (e.g., list or cycle); passing string triggers error
    with pytest.raises(ValueError):
        plot_line("k", [1, 2], ax, {"k": "Label"}, "not_iterable", 1)

# Tests for input validation
@pytest.mark.parametrize("key", [123, None, ['model']])
def test_invalid_key(key):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=".*Input 'key' must be a string.*"):
        plot_line(
            key=key,
            daily_mean=[1, 2, 3],
            ax=ax,
            label_lookup={'model': 'Model Output'},
            color_palette=cycle(['blue']),
            line_width=2
        )


@pytest.mark.parametrize("ax", [None, "not an ax", 42])
def test_invalid_ax_line(ax):
    with pytest.raises(ValueError, match=".*Input 'ax' must be a matplotlib.axes.Axes instance.*"):
        plot_line(
            key='model',
            daily_mean=[1, 2, 3],
            ax=ax,
            label_lookup={'model': 'Model Output'},
            color_palette=cycle(['blue']),
            line_width=2
        )


@pytest.mark.parametrize("label_lookup", [None, "not a dict", [("model", "label")]])
def test_invalid_label_lookup(label_lookup):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=".*Input 'label_lookup' must be a dictionary.*"):
        plot_line(
            key='model',
            daily_mean=[1, 2, 3],
            ax=ax,
            label_lookup=label_lookup,
            color_palette=cycle(['blue']),
            line_width=2
        )


@pytest.mark.parametrize("color_palette", [None, ["red", "blue"], "not an iterator"])
def test_invalid_color_palette_line(color_palette):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=".*Input 'color_palette' must be an iterator.*"):
        plot_line(
            key='model',
            daily_mean=[1, 2, 3],
            ax=ax,
            label_lookup={'model': 'Model Output'},
            color_palette=color_palette,
            line_width=2
        )


@pytest.mark.parametrize("line_width", [0, -1, "wide", None])
def test_invalid_line_width(line_width):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=".*Input 'line_width' must be a positive number.*"):
        plot_line(
            key='model',
            daily_mean=[1, 2, 3],
            ax=ax,
            label_lookup={'model': 'Model Output'},
            color_palette=cycle(['blue']),
            line_width=line_width
        )
        
        
###############################################################################
# test geolocalized_coords
###############################################################################


# Test that compute_geolocalized_coords returns correctly shaped arrays and expected coordinate values
def test_compute_geo_coords_valid_output():
    # Compute coordinates for a grid of shape (2,3) with given spacing and boundaries
    result = compute_geolocalized_coords((2, 3), 0.1, -10, 0.5, 50, 0.25)
    # Verify latitude array shape matches input grid dimensions
    assert result["latp"].shape == (2, 3)
    # Verify longitude array shape matches input grid dimensions
    assert result["lonp"].shape == (2, 3)
    # Confirm MinLambda is correctly adjusted by subtracting epsilon (0.1) from input MinLambda (-10)
    assert result["MinLambda"] == pytest.approx(-10.1)
    # Confirm MaxPhi is correctly adjusted by adding epsilon (0.1) to input MaxPhi (0.25 + 50)
    assert result["MaxPhi"] == pytest.approx(50.25 + 0.1)

# Test that passing an invalid type for grid_shape raises a ValueError
def test_invalid_grid_shape_type():
    # Passing a string instead of a tuple for grid_shape should raise an error for input validation
    with pytest.raises(ValueError):
        compute_geolocalized_coords("2x3", 0.1, 0, 1, 0, 1)

# Test that a negative epsilon value raises a ValueError
def test_invalid_epsilon():
    # Epsilon must be positive since it represents a small buffer distance; negative values are invalid
    with pytest.raises(ValueError):
        compute_geolocalized_coords((2, 3), -0.1, 0, 1, 0, 1)

# Test that zero or negative step sizes raise a ValueError
def test_invalid_step_size():
    # Step sizes control coordinate increments and must be positive and non-zero for meaningful grid generation
    with pytest.raises(ValueError):
        compute_geolocalized_coords((2, 3), 0.1, 0, 0, 0, -1)


###############################################################################
# tests swifs_colormap
###############################################################################  

      
# Test swifs_colormap with valid chlorophyll-a data returns correct types and matching shapes
def test_swifs_colormap_chla():
    data = np.array([[0.03, 0.1], [1.5, 10.0]])
    # Call swifs_colormap to get color-mapped data, colormap, and normalization for "Chla" variable
    result = swifs_colormap(data, "Chla")
    # Ensure the returned colormap object is of type ListedColormap (used for discrete colors)
    assert isinstance(result[1], ListedColormap)
    # Ensure the returned normalization object is of type BoundaryNorm (defines discrete color boundaries)
    assert isinstance(result[2], BoundaryNorm)
    # Check the output data shape matches input, preserving spatial structure
    assert result[0].shape == data.shape
    # Verify NaN values in input remain NaN in the output, preserving missing data locations
    assert np.all(np.isnan(data) == np.isnan(result[0]))

# Test swifs_colormap raises ValueError for unknown variable name
def test_swifs_colormap_invalid_variable():
    # Calling with an unknown variable name should raise ValueError with message containing "Unknown"
    with pytest.raises(ValueError, match="Unknown"):
        swifs_colormap(np.array([[1, 2]]), "InvalidVar")

# Test swifs_colormap raises ValueError when input data contains only NaNs
def test_swifs_colormap_all_nans():
    # Passing data that is all NaNs is invalid; should raise an error indicating no valid data
    with pytest.raises(ValueError, match="contains only NaNs"):
        swifs_colormap(np.full((2, 2), np.nan), "Chla")

# Test swifs_colormap raises ValueError for unrecognized variable when using default bins
def test_swifs_colormap_default_bins():
    data = np.array([[1, 2], [3, 4]])
    # Using a variable not recognized by the function with default bins should raise a ValueError with clear message
    with pytest.raises(ValueError, match="Unknown variable 'Unrecognized'"):
        swifs_colormap(data, "Unrecognized")
        
# Branch selection tests
# Utility: small valid data array
valid_data = np.array([[0.1, 1.0], [5.0, 10.0]])


# Parameterized test for each valid variable and expected cmap
@pytest.mark.parametrize("variable, expected_cmap_func", [
    ("Chla", lambda: get_cmap("viridis")),
    ("N1p", lambda: get_cmap("YlGnBu")),
    ("N3n", lambda: get_cmap("YlGnBu")),
    ("N4n", lambda: get_cmap("YlGnBu")),
    ("P1c", lambda: cmocean.cm.algae),
    ("P2c", lambda: cmocean.cm.algae),
    ("P3c", lambda: cmocean.cm.algae),
    ("P4c", lambda: cmocean.cm.algae),
    ("Z3c", lambda: cmocean.cm.turbid),
    ("Z4c", lambda: cmocean.cm.turbid),
    ("Z5c", lambda: cmocean.cm.turbid),
    ("Z6c", lambda: cmocean.cm.turbid),
    ("R6c", lambda: cmocean.cm.matter),
])
def test_swifs_colormap_variable_specific_branches(variable, expected_cmap_func):
    data_clipped, cmap, norm, ticks, labels = swifs_colormap(valid_data, variable)

    # Type checks
    assert isinstance(data_clipped, np.ndarray)
    assert isinstance(cmap, ListedColormap)
    assert isinstance(norm, BoundaryNorm)
    assert isinstance(ticks, np.ndarray)
    assert all(isinstance(label, str) for label in labels)

    # Check colormap origin (by name or color similarity)
    expected_colors = expected_cmap_func()(np.linspace(0, 1, len(ticks) - 1))
    np.testing.assert_allclose(cmap.colors, expected_colors, atol=1e-6)

def test_swifs_colormap_data_all_nan():
    nan_data = np.full((2, 2), np.nan)
    with pytest.raises(ValueError, match="❌ data_in contains only NaNs.*❌"):
        swifs_colormap(nan_data, "Chla")


###############################################################################
# test get_benthic_paramerers
###############################################################################    
    

# Test get_benthic_plot_parameters returns expected default limits and thresholds for oxygen variable "O2o"
def test_get_benthic_plot_parameters_o2o():
    dummy_data = {2020: [np.array([[100.0]])]}
    opts = {}
    vmin, vmax, levels, num_ticks, cmap, use_custom, hypo, hyper = get_benthic_plot_parameters("O2o", dummy_data, opts)
    
    # Ensure minimum value is set to zero, typical lower bound for oxygen
    assert vmin == 0.0
    
    # Verify maximum value corresponds to expected upper bound for oxygen
    assert vmax == 350.0
    
    # Check hypoxia threshold matches expected default value
    assert hypo == 62.5
    
    # Check hyperoxia threshold matches expected default value
    assert hyper == 312.5
    
    # Confirm the function does not flag usage of custom settings in this default case
    assert use_custom is False

# Test get_benthic_plot_parameters correctly applies custom vmin/vmax options for temperature variable
def test_get_benthic_plot_parameters_temperature():
    dummy_data = {2020: [np.array([[15.0]])]}
    
    # Provide custom min and max values to test overriding default parameters
    opts = {"vmin_votemper": 10, "vmax_votemper": 30}
    
    vmin, vmax, levels, num_ticks, cmap, use_custom, _, _ = get_benthic_plot_parameters("votemper", dummy_data, opts)
    
    # Verify that the custom minimum value is applied correctly
    assert vmin == 10
    
    # Verify that the custom maximum value is applied correctly
    assert vmax == 30
    
    # Check that contour levels are generated based on the custom limits
    assert len(levels) > 0
    
    # Confirm the function recognizes these as non-custom settings (if that's the expected logic)
    assert use_custom is False


def test_get_benthic_plot_parameters_chla_variable():
    dummy_data = {2020: [np.array([[0.1]])]}
    opts = {}
    vmin, vmax, levels, num_ticks, cmap, use_custom, _, _ = get_benthic_plot_parameters("Chla", dummy_data, opts)
    assert use_custom is True
    assert all(x is None for x in [vmin, vmax, levels, num_ticks, cmap])

def test_get_benthic_plot_parameters_missing_data():
    dummy_data = {}
    opts = {}
    with pytest.raises(ValueError, match="No data arrays found"):
        get_benthic_plot_parameters("unknown_var", dummy_data, opts)
        
# Test salinity case with all required options provided
def test_salinity_plotting():
    opts = {
        "vmin_vosaline": 30.0,
        "vmax_vosaline": 38.0,
        "levels_vosaline": 5,
        "num_ticks_vosaline": 3
    }
    result = get_benthic_plot_parameters("vosaline", {}, opts)
    vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypo, hyper = result

    assert vmin == 30.0
    assert vmax == 38.0
    assert len(levels) == 5
    assert num_ticks == 3
    assert cmap == cm.haline
    assert not use_custom_cmap
    assert hypo is None and hyper is None

# Test density case with correct options
def test_density_plotting():
    opts = {
        "vmin_density": 1015.0,
        "vmax_density": 1028.0,
        "levels_density": 7,
        "num_ticks_density": 4
    }
    result = get_benthic_plot_parameters("dense_water", {}, opts)
    vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypo, hyper = result

    assert vmin == 1015.0
    assert vmax == 1028.0
    assert len(levels) == 7
    assert num_ticks == 4
    assert cmap == cm.dense
    assert not use_custom_cmap
    assert hypo is None and hyper is None

# Test fallback case where an unknown variable is passed but data is valid
def test_default_fallback_plotting():
    dummy_data = np.array([[1.0, 2.0], [3.0, np.nan]])
    var_dataframe = {
        2000: [dummy_data],
        2001: [np.full((2, 2), 4.0)]
    }
    result = get_benthic_plot_parameters("unknown_var", var_dataframe, {})
    vmin, vmax, levels, num_ticks, cmap, use_custom_cmap, hypo, hyper = result

    assert vmin == 1.0
    assert vmax == 4.0
    assert len(levels) == 20
    assert num_ticks == 5
    assert cmap == "jet"
    assert not use_custom_cmap
    assert hypo is None and hyper is None

# Test salinity case with missing vmin/vmax in options (should raise ValueError)
def test_salinity_missing_opts_raises():
    with pytest.raises(ValueError, match="vmin_vosaline and vmax_vosaline must be specified"):
        get_benthic_plot_parameters("vosaline", {}, {})

# Test density case with missing vmin/vmax in options (should raise ValueError)
def test_density_missing_opts_raises():
    with pytest.raises(ValueError, match="vmin_density and vmax_density must be specified"):
        get_benthic_plot_parameters("density", {}, {})

# Test fallback case when var_dataframe contains no valid arrays (should raise ValueError)
def test_fallback_empty_var_dataframe_raises():
    var_dataframe = {2000: [None], 2001: []}
    with pytest.raises(ValueError, match="No data arrays found"):
        get_benthic_plot_parameters("some_unknown", var_dataframe, {})

# Test fallback case when data arrays contain only NaNs (should raise ValueError)
def test_fallback_only_nans_raises():
    var_dataframe = {2000: [np.full((2, 2), np.nan)]}
    with pytest.raises(ValueError, match="No valid data found"):
        get_benthic_plot_parameters("some_unknown", var_dataframe, {})

###############################################################################

###############################################################################        

def test_cmocean_to_plotly_output_length():
    scale = cmocean_to_plotly("thermal", n=10)
    assert len(scale) == 10
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in scale)

def test_cmocean_to_plotly_rgb_format():
    scale = cmocean_to_plotly("haline", n=5)
    for _, color in scale:
        assert color.startswith("rgb(") and color.endswith(")")
        parts = color[4:-1].split(',')
        assert len(parts) == 3
        assert all(0 <= int(p) <= 255 for p in parts)

def test_cmocean_to_plotly_normalized_positions():
    scale = cmocean_to_plotly("dense", n=5)
    positions = [x[0] for x in scale]
    assert positions == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

def test_cmocean_to_plotly_invalid_name():
    with pytest.raises(AttributeError):
        cmocean_to_plotly("notacolormap")
 
###############################################################################

###############################################################################
        
def test_invert_colorscale_simple():
    original = [[0.0, "rgb(255,0,0)"], [1.0, "rgb(0,0,255)"]]
    inverted = invert_colorscale(original)
    assert inverted == [[0.0, "rgb(0,0,255)"], [1.0, "rgb(255,0,0)"]]

def test_invert_colorscale_midpoints():
    original = [[0.0, "rgb(1,1,1)"], [0.5, "rgb(2,2,2)"], [1.0, "rgb(3,3,3)"]]
    inverted = invert_colorscale(original)
    assert inverted == [[0.0, "rgb(3,3,3)"], [0.5, "rgb(2,2,2)"], [1.0, "rgb(1,1,1)"]]

def test_invert_colorscale_preserves_colors():
    original = [[0.0, "rgb(10,10,10)"], [0.5, "rgb(20,20,20)"]]
    inverted = invert_colorscale(original)
    assert [color for _, color in inverted] == ["rgb(20,20,20)", "rgb(10,10,10)"]

def test_invert_colorscale_empty():
    assert invert_colorscale([]) == []
    
###############################################################################

###############################################################################

def test_custom_oxy_returns_colormap():
    cmap = custom_oxy()  # No [0] since it returns only one object
    assert isinstance(cmap, mcolors.LinearSegmentedColormap)
    assert cmap.N == 256

def test_custom_oxy_colormap_at_thresholds():
    cmap = custom_oxy(vmin=0, vmax=100, low=25, high=75)
    # Check colors at thresholds: low (25%), high (75%)
    rgb_low = cmap(0.25)[:3]
    rgb_high = cmap(0.75)[:3]
    assert all(0 <= c <= 1 for c in rgb_low)
    assert all(0 <= c <= 1 for c in rgb_high)

def test_custom_oxy_invalid_thresholds():
    with pytest.raises(ValueError, match="Thresholds normalized must be in ascending order"):
        custom_oxy(vmin=0, vmax=100, low=80, high=60)

def test_custom_oxy_custom_name():
    cmap = custom_oxy(name="my_cmap")  # no [0]
    assert isinstance(cmap, mcolors.LinearSegmentedColormap)
    assert cmap.name == "my_cmap"  # check the name is set
