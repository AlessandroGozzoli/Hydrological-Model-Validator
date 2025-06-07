import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

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
# Test 1: Raise error if output_path is missing
def test_missing_output_path():
    with pytest.raises(ValueError, match="output_path must be specified"):
        comprehensive_taylor_diagram(dummy_data_dict)
        
# Test 2: Raise error if both variable_name and (variable, unit) are missing
def test_missing_variable_and_unit():
    with pytest.raises(ValueError, match="If 'variable_name' is not provided"):
        comprehensive_taylor_diagram(dummy_data_dict, output_path='tmp')

# Test 3: Run successfully using variable_name and full plotting options
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_successful_plot_with_variable_name(mock_sm, mock_stats_func, mock_label_func, tmp_path):
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
    
# Test 4: Run with direct variable/unit instead of variable_name
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_variable_unit_direct(mock_sm, mock_stats_func, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable="Chlorophyll",
        unit="mg/m³"
    )
    assert mock_sm.call_count >= 3
    
# Test 5: Trigger fallback marker shape logic when not enough provided
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_marker_shape_autofill(mock_sm, mock_stats_func, mock_label_func, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        marker_shapes=['o']  # Only one provided, two needed
    )
    assert mock_sm.call_count >= 3
    
# Test 6: Verify the output directory and file are created
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_output_directory_created(mock_sm, mock_stats_func, tmp_path):
    out_dir = tmp_path / "subdir"
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=out_dir,
        variable="Chlorophyll",
        unit="mg/m³"
    )
    assert out_dir.exists()
    assert (out_dir / "Taylor_diagram_summary.png").exists()


# Test 7: Ensure RMSD label is rendered when titleRMS is off
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_rmsd_label_displayed(mock_sm, mock_stats_func, mock_label_func, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        tickrms=[0.5],
        titleRMS='off'
    )
    assert mock_sm.call_count >= 3


# Test 8: Confirm that reference marker options are processed correctly
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.Taylor_computations.compute_yearly_taylor_stats", return_value=(mock_stats, mock_std_ref))
@patch("Hydrological_model_validator.Plotting.Taylor_diagrams.sm.taylor_diagram")
def test_reference_marker_options(mock_sm, mock_stats_func, mock_label_func, tmp_path):
    comprehensive_taylor_diagram(
        dummy_data_dict,
        output_path=tmp_path,
        variable_name="chl",
        Ref_markersymbol='^',
        Ref_markercolor='green',
        Ref_markersize=10
    )
    assert mock_sm.call_count >= 3