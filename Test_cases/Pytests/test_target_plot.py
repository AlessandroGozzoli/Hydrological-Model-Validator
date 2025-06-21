import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for testing

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Hydrological_model_validator.Plotting.Target_plots import comprehensive_target_diagram, target_diagram_by_month

dummy_data_dict = {
    "model": {
        2000: pd.Series([1, 2, 3]),
        2001: pd.Series([4, 5, 6])
    },
    "sat": {
        2000: pd.Series([1, 2, 3]),
        2001: pd.Series([4, 5, 6])
    }
}


mock_bias = np.array([0.1, 0.2])
mock_crmsd = np.array([0.3, 0.4])
mock_rmsd = np.array([0.5, 0.6])
mock_labels = ["2000", "2001"]

mock_base_opts = {"markerLabelColor": "black", "alpha": 0.5, "markersize": 10, "circlelinespec": "--"}
mock_overlay_opts = {"markerLabelColor": "blue", "alpha": 0.3, "markersize": 8, "circlelinespec": ":" , "circles": [0.5, 1.0]}
mock_data_marker_opts = {"markerLabelColor": "red", "markersize": 12, "alpha": 0.8, "circles": None, "overlay": None}
mock_plt_opts = {"figsize": (6, 6), "dpi": 80, "title_pad": 15, "title_fontweight": "bold"}

###############################################################################
# Test for yearly plot
###############################################################################

# Test that missing output_path argument raises a ValueError
def test_missing_output_path_raises():
    with pytest.raises(ValueError, match="output_path must be specified"):
        comprehensive_target_diagram(dummy_data_dict)

# Test that missing variable_name and variable/unit arguments raise a ValueError
def test_missing_variable_and_unit_raises():
    with pytest.raises(ValueError, match="You must provide either 'variable' and 'unit' or 'variable_name'"):
        comprehensive_target_diagram(dummy_data_dict, output_path="some/path")

# Test that zone_bounds argument with invalid type raises a ValueError
def test_invalid_zone_bounds_type_raises():
    with pytest.raises(ValueError, match="'zone_bounds' must be a tuple of two numbers"):
        comprehensive_target_diagram(
            dummy_data_dict,
            output_path="some/path",
            variable_name="chl",
            zone_bounds="not_a_tuple"  # invalid type, should be tuple
        )

# Test that zone_bounds argument with incorrect tuple length raises a ValueError
def test_invalid_zone_bounds_length_raises():
    with pytest.raises(ValueError, match="'zone_bounds' must be a tuple of two numbers"):
        comprehensive_target_diagram(
            dummy_data_dict,
            output_path="some/path",
            variable_name="chl",
            zone_bounds=(0.5,)  # tuple length must be 2
        )

# Test that zone_bounds argument with non-numeric content raises a ValueError
def test_invalid_zone_bounds_contents_raises():
    with pytest.raises(ValueError, match="'zone_bounds' must be a tuple of two numbers"):
        comprehensive_target_diagram(
            dummy_data_dict,
            output_path="some/path",
            variable_name="chl",
            zone_bounds=(0.5, "string")  # second element is not a number
        )

# Test successful plot creation, file output, and expected plot function calls
@patch("Hydrological_model_validator.Plotting.Target_plots.fill_annular_region")
@patch("Hydrological_model_validator.Plotting.Target_plots.sm.target_diagram")
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Processing.Target_computations.compute_target_extent_yearly", return_value=1.0)
@patch("Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats", return_value=(mock_bias, mock_crmsd, mock_rmsd, mock_labels))
@patch("Hydrological_model_validator.Processing.utils.extract_options", side_effect=[mock_base_opts, mock_overlay_opts, mock_data_marker_opts, mock_plt_opts])
@patch("Hydrological_model_validator.Plotting.Target_plots.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
def test_successful_plot_creates_file_and_calls(
    mock_get_var,
    mock_extract_opts,
    mock_stats,
    mock_extent,
    mock_show,
    mock_target_diagram,
    mock_fill_region,
    tmp_path,
):
    outdir = tmp_path / "plots"
    comprehensive_target_diagram(
        dummy_data_dict,
        output_path=outdir,
        variable_name="chl",
        filename="test_target.png",
        title="My Target Plot",
        zone_bounds=(0.5, 0.7),  # explicit zone bounds for test consistency
    )
    expected_file = outdir / "test_target.png"
    # Check that the plot image file was created successfully
    assert expected_file.exists()
    # Confirm that the annular region was filled three times (expected regions)
    assert mock_fill_region.call_count == 3
    # Confirm at least three calls to the target diagram plotting function
    assert mock_target_diagram.call_count >= 3

# Test marker shapes cycle usage in target diagrams
@patch("Hydrological_model_validator.Plotting.Target_plots.sm.target_diagram")
@patch("Hydrological_model_validator.Plotting.formatting.fill_annular_region")
@patch("Hydrological_model_validator.Processing.Target_computations.compute_target_extent_yearly", return_value=1.0)
@patch("Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats", return_value=(mock_bias, mock_crmsd, mock_rmsd, mock_labels))
@patch("Hydrological_model_validator.Processing.utils.extract_options", side_effect=[mock_base_opts, mock_overlay_opts, mock_data_marker_opts, mock_plt_opts])
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
def test_marker_shapes_cycle_usage(
    mock_get_var,
    mock_extract_opts,
    mock_stats,
    mock_extent,
    mock_fill_region,
    mock_target_diagram,
    tmp_path,
):
    # Use a single marker shape and verify it is applied in calls
    comprehensive_target_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        marker_shapes=['o']  # single shape to cycle
    )
    calls = mock_target_diagram.call_args_list
    data_calls = calls[2:]  # skip base and overlay calls to focus on data calls
    for call in data_calls:
        kwargs = call[1]
        # Each call should specify a marker symbol
        assert "markersymbol" in kwargs

# Test title customization propagates correctly to the plot
@patch("matplotlib.pyplot.title")
@patch("Hydrological_model_validator.Plotting.Target_plots.sm.target_diagram")
@patch("Hydrological_model_validator.Plotting.formatting.fill_annular_region")
@patch("Hydrological_model_validator.Processing.Target_computations.compute_target_extent_yearly", return_value=1.0)
@patch("Hydrological_model_validator.Processing.Target_computations.compute_normalised_target_stats", return_value=(mock_bias, mock_crmsd, mock_rmsd, mock_labels))
@patch("Hydrological_model_validator.Processing.utils.extract_options", side_effect=[mock_base_opts, mock_overlay_opts, mock_data_marker_opts, mock_plt_opts])
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
def test_title_customization(
    mock_get_var,
    mock_extract_opts,
    mock_stats,
    mock_extent,
    mock_fill_region,
    mock_target_diagram,
    mock_title,
    tmp_path,
):
    custom_title = "Custom Target Title"
    comprehensive_target_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        title=custom_title
    )
    # Verify that the target diagram was called (plot generated)
    mock_target_diagram.assert_called()

    # Check the plot title matches the custom title provided
    args, kwargs = mock_title.call_args
    assert args[0] == custom_title

    # Check title properties are set correctly
    assert kwargs.get("pad") == 40
    assert kwargs.get("fontweight") == "bold"
    

###############################################################################
# Tests for monthly plot
###############################################################################


@pytest.mark.parametrize("kwargs", [{}, {"variable": None, "unit": None}])
# Test that missing output_path raises ValueError regardless of variable/unit presence
def test_target_diagram_by_month_raises_without_output_path(kwargs):
    dummy_data_dict = {"mod": {}, "sat": {}}
    with pytest.raises(ValueError, match="output_path must be specified"):
        target_diagram_by_month(dummy_data_dict, **kwargs)

# Test that missing variable_name and variable/unit raises ValueError
def test_target_diagram_by_month_raises_without_variable_info():
    dummy_data_dict = {"mod": {}, "sat": {}}
    with pytest.raises(ValueError, match="You must provide either 'variable' and 'unit' or 'variable_name'"):
        target_diagram_by_month(dummy_data_dict, output_path=Path("./tmp"))

# Test that providing variable_name triggers get_variable_label_unit call
@patch("Hydrological_model_validator.Plotting.Target_plots.get_variable_label_unit", return_value=("Var", "unit"))
@patch("Hydrological_model_validator.Plotting.Target_plots.extract_options", side_effect=lambda kwargs, defaults=None, prefix="": {})
@patch("Hydrological_model_validator.Plotting.Target_plots.compute_target_extent_monthly", return_value=3.0)
@patch("Hydrological_model_validator.Plotting.Target_plots.sm.target_diagram")
@patch("Hydrological_model_validator.Plotting.Target_plots.fill_annular_region")
@patch("Hydrological_model_validator.Plotting.Target_plots.compute_normalised_target_stats_by_month", side_effect=ValueError)
@patch("Hydrological_model_validator.Plotting.Target_plots.extract_mod_sat_keys", return_value=("mod", "sat"))
@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.title")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.draw")
@patch("matplotlib.pyplot.pause")
@patch("matplotlib.pyplot.close")
def test_variable_name_triggers_get_variable_label_unit(
    mock_close, mock_pause, mock_draw, mock_show, mock_savefig, mock_title,
    mock_subplots, mock_extract_mod_sat, mock_stats, mock_fill, mock_target_diagram,
    mock_extent, mock_extract_opts, mock_get_var
):
    mock_subplots.return_value = (MagicMock(), MagicMock())
    data_dict = {"mod": {"2000": []}, "sat": {}}
    output_path = Path("./tmp")
    target_diagram_by_month(data_dict, output_path=output_path, variable_name="varname")
    # Verify that get_variable_label_unit was called once with variable_name
    mock_get_var.assert_called_once_with("varname")

# Test that fill_annular_region is called exactly three times with expected colors
@patch("Hydrological_model_validator.Plotting.Target_plots.fill_annular_region")
@patch("Hydrological_model_validator.Plotting.Target_plots.compute_target_extent_monthly", return_value=2.5)
@patch("Hydrological_model_validator.Plotting.Target_plots.extract_options", side_effect=lambda kwargs, defaults=None, prefix="": {})
@patch("Hydrological_model_validator.Plotting.Target_plots.get_variable_label_unit", return_value=("Variable", "unit"))
@patch("Hydrological_model_validator.Plotting.Target_plots.extract_mod_sat_keys", return_value=("mod", "sat"))
@patch("Hydrological_model_validator.Plotting.Target_plots.sm.target_diagram")
@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.title")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.draw")
@patch("matplotlib.pyplot.pause")
@patch("matplotlib.pyplot.close")
def test_fill_annular_region_called_3_times(
    mock_close, mock_pause, mock_draw, mock_show, mock_savefig, mock_title,
    mock_subplots, mock_target_diagram, mock_extract_mod_sat, mock_get_var,
    mock_extract_opts, mock_extent, mock_fill
):
    mock_subplots.return_value = (MagicMock(), MagicMock())
    data_dict = {"mod": {"2000": []}, "sat": {}}
    target_diagram_by_month(
        data_dict,
        output_path=Path("./tmp"),
        variable_name="varname",
        zone_bounds=(0.4, 0.8)  # specify zone bounds to test annular region coloring
    )
    # Assert fill_annular_region called three times for three regions
    assert mock_fill.call_count == 3
    # Extract colors used in each call to verify they match expected sequence
    colors = [call.kwargs.get("color") for call in mock_fill.call_args_list]
    assert colors == ['lightgreen', 'khaki', 'lightcoral']

# Test that ValueError in stats computation for one month skips plotting that month
@patch("Hydrological_model_validator.Plotting.Target_plots.compute_normalised_target_stats_by_month")
@patch("Hydrological_model_validator.Plotting.Target_plots.sm.target_diagram")
@patch("Hydrological_model_validator.Plotting.Target_plots.extract_mod_sat_keys", return_value=("mod", "sat"))
@patch("Hydrological_model_validator.Plotting.Target_plots.get_variable_label_unit", return_value=("Var", "unit"))
@patch("Hydrological_model_validator.Plotting.Target_plots.compute_target_extent_monthly", return_value=3.0)
@patch("Hydrological_model_validator.Plotting.Target_plots.extract_options", side_effect=lambda kwargs, defaults=None, prefix="": {})
@patch("matplotlib.pyplot.title")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.draw")
@patch("matplotlib.pyplot.pause")
@patch("matplotlib.pyplot.close")
def test_stats_value_error_skips_month(
    mock_close, mock_pause, mock_draw, mock_show, mock_savefig, mock_title,
    mock_extract_opts, mock_extent, mock_get_var,
    mock_extract_mod_sat, mock_target_diagram, mock_stats
):
    # Use real matplotlib figure and axes instead of mocking subplots
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    with patch("matplotlib.pyplot.subplots", return_value=(fig, ax)):
        def side_effect(data_dict, month_index):
            # Raise ValueError for month index 1 to simulate failure
            if month_index == 1:
                raise ValueError
            return ([0.1], [0.2], [0.3], ["2000"])
        mock_stats.side_effect = side_effect
        data_dict = {"mod": {"2000": []}, "sat": {}}
        target_diagram_by_month(data_dict, output_path=Path("./tmp"), variable_name="var")
    # Ensure compute_normalised_target_stats_by_month was called multiple times
    assert mock_stats.call_count >= 2
    # Verify target_diagram called with markersymbol argument in any call
    calls = mock_target_diagram.call_args_list
    assert any("markersymbol" in str(call) for call in calls)

# Test that savefig is called with a custom filename argument
@patch("matplotlib.pyplot.savefig")
@patch("Hydrological_model_validator.Plotting.Target_plots.get_variable_label_unit", return_value=("Var", "unit"))
@patch("Hydrological_model_validator.Plotting.Target_plots.extract_options", side_effect=lambda kwargs, defaults=None, prefix="": {})
@patch("Hydrological_model_validator.Plotting.Target_plots.compute_target_extent_monthly", return_value=2.0)
@patch("Hydrological_model_validator.Plotting.Target_plots.sm.target_diagram")
@patch("Hydrological_model_validator.Plotting.Target_plots.extract_mod_sat_keys", return_value=("mod", "sat"))
@patch("matplotlib.pyplot.subplots")
def test_savefig_called_with_custom_filename(
    mock_subplots, mock_extract_mod_sat, mock_target_diagram, mock_extent,
    mock_extract_opts, mock_get_var, mock_savefig
):
    # Use real matplotlib figure and axes
    real_fig, real_ax = plt.figure(), plt.axes(projection='polar')

    # Patch subplots to return real figure and axes to avoid mock complications
    mock_subplots.return_value = (real_fig, real_ax)

    data_dict = {"mod": {"2000": []}, "sat": {}}
    out_dir = Path("./tmp")
    filename = "custom_name.png"

    # Call the function with a custom filename argument
    target_diagram_by_month(
        data_dict,
        output_path=out_dir,
        variable_name="var",
        filename=filename
    )

    expected_filepath = out_dir / filename

    # Assert savefig was called at least once
    mock_savefig.assert_called()

    # Extract the actual filepath argument passed to savefig
    args, kwargs = mock_savefig.call_args

    # Check that savefig was called with the expected custom filename path
    assert args[0] == expected_filepath