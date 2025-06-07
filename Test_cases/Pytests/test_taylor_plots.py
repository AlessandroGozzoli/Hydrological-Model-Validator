import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for testing (no display)

from Hydrological_model_validator.Plotting.Taylor_diagrams import comprehensive_taylor_diagram, monthly_taylor_diagram

# Dummy Taylor stats: (year, std, crmsd, corrcoef)
mock_stats = [('2000', 2.0, 1.0, 0.8), ('2001', 1.5, 0.7, 0.85)]
mock_std_ref = 1.0

import pandas as pd

dummy_data_dict = {
    'model': {
        '2000': pd.Series([1, 2, 3]),
        '2001': pd.Series([2, 3, 4])
    },
    'sat': {
        '2000': pd.Series([1, 2, 3]),
        '2001': pd.Series([2, 3, 5])
    }
}

# Mock DataFrame returned by build_all_points
mock_df = pd.DataFrame({
    "sdev": [1.0, 0.9, 1.1],
    "crmsd": [0.0, 0.1, 0.2],
    "ccoef": [1.0, 0.95, 0.9],
    "year": ["Ref", "2000", "2000"],
    "month": ["Ref", "Jan", "Feb"],
})


# Define a minimal dictionary for month colors
mock_month_colors = {
    "Jan": "blue",
    "Feb": "green",
    "Mar": "red",
    "Apr": "purple",
    "May": "orange",
    "Jun": "cyan",
    "Jul": "magenta",
    "Aug": "yellow",
    "Sep": "brown",
    "Oct": "pink",
    "Nov": "gray",
    "Dec": "olive",
}


###############################################################################
# Test for yearly plot
###############################################################################

# Raise error if output_path is missing
def test_missing_output_path():
    with pytest.raises(ValueError, match="output_path must be specified"):
        comprehensive_taylor_diagram(dummy_data_dict)
        
# Raise error if both variable_name and (variable, unit) are missing
def test_missing_variable_and_unit():
    with pytest.raises(ValueError, match="If 'variable_name' is not provided"):
        comprehensive_taylor_diagram(dummy_data_dict, output_path='tmp')

# Run successfully using variable_name and full plotting options
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_successful_plot_with_variable_name(mock_sm, mock_stats_func, mock_label_func, mock_show, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        marker_shapes=['o', 's'],
        figsize=(8, 6),
        dpi=100,
        title="Test Taylor",
        title_fontsize=14,
        title_fontweight='bold',
        title_pad=10,
        tickrms=[0.5, 1.0],
        Ref_markersymbol='*',
        Ref_markercolor='r',
        Ref_markersize=12,
        markersymbol='o',
        markercolor='b',
        markersize=8
    )
    assert mock_sm.call_count >= 3
    
# Run with direct variable/unit instead of variable_name
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_variable_unit_direct(mock_sm, mock_stats_func, mock_show, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable="Chlorophyll",
        unit="mg/m³"
    )
    assert mock_sm.call_count >= 3
    
# Trigger fallback marker shape logic when not enough provided
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_marker_shape_autofill(mock_sm, mock_stats_func, mock_label_func, mock_show, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        marker_shapes=['o']  # Only one provided, two needed
    )
    assert mock_sm.call_count >= 3
    
# Verify the output directory and file are created
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_output_directory_created(mock_sm, mock_stats_func, mock_show, tmp_path):
    out_dir = tmp_path / "subdir"
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=out_dir,
        variable="Chlorophyll",
        unit="mg/m³"
    )
    assert out_dir.exists()
    assert (out_dir / "Taylor_diagram_summary.png").exists()

# Ensure RMSD label is rendered when titleRMS is off
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_rmsd_label_displayed(mock_sm, mock_stats_func, mock_label_func, mock_show, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        tickrms=[0.5],
        titleRMS='off'
    )
    assert mock_sm.call_count >= 3

# Confirm that reference marker options are processed correctly
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_reference_marker_options(mock_sm, mock_stats_func, mock_label_func, mock_show, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        Ref_markersymbol='^',
        Ref_markercolor='green',
        Ref_markersize=10
    )
    assert mock_sm.call_count >= 3
    
###############################################################################
# Tests for monthly plots
###############################################################################

# Ensures plotting and saving works with full input setup
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.default_month_colors", mock_month_colors)
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.build_all_points", return_value=(mock_df, ["2000"]))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_successful_monthly_plot_with_variable_name(mock_plot, mock_label_func, mock_build_func, mock_show, tmp_path):
    monthly_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        title="Test Monthly Taylor",
        dpi=80,
        title_fontsize=14,
        title_fontweight="bold"
    )
    save_path = tmp_path / "Unified_Taylor_Diagram.png"
    assert save_path.exists()

# Ensures fallback to manual variable/unit works
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.default_month_colors", mock_month_colors)
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.build_all_points", return_value=(mock_df, ["2000"]))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_successful_monthly_plot_with_variable_and_unit(mock_plot, mock_build_func, mock_show, tmp_path):
    monthly_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable="Chlorophyll",
        unit="mg/m³",
        title="Alt Monthly Taylor"
    )
    assert (tmp_path / "Unified_Taylor_Diagram.png").exists()

# Validates missing mandatory `output_path` argument
def test_missing_output_path_raises_error():
    with pytest.raises(ValueError, match="output_path must be specified"):
        monthly_taylor_diagram(dummy_data_dict)

# Checks enforcement of variable metadata
def test_missing_variable_and_unit_raises_error(tmp_path):
    with pytest.raises(ValueError, match="both 'variable' and 'unit' must be specified"):
        monthly_taylor_diagram(dummy_data_dict, output_path=tmp_path)

# Ensures failure if 'Ref' row is absent from build_all_points result
@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.build_all_points", return_value=(
    pd.DataFrame({  # No "Ref" row
        "sdev": [1.0], "crmsd": [0.0], "ccoef": [1.0], "year": ["2000"], "month": ["Jan"]
    }),
    ["2000"]
))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_missing_ref_raises_index_error(mock_plot, mock_label_func, mock_build_func, mock_show, tmp_path):
    with pytest.raises(IndexError):
        monthly_taylor_diagram(dummy_data_dict, output_path=tmp_path, variable_name="chl")

@patch("matplotlib.pyplot.show")
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.default_month_colors", mock_month_colors)
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.build_all_points", return_value=(mock_df, ["2000"]))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_plot_function_calls(mock_plot, mock_label_func, mock_build_func, mock_show, tmp_path):
    monthly_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl"
    )
    # Expected calls:
    # 1 for base diagram
    # 1 for reference marker overlay
    # 2 for monthly points Jan, Feb
    assert mock_plot.call_count == 4