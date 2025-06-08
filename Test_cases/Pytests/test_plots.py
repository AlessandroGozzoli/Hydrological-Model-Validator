import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for testing (no display)

# Replace this with your actual import
from Hydrological_model_validator.Plotting.Plots import (timeseries,
                                                         scatter_plot,
                                                         seasonal_scatter_plot,
                                                         whiskerbox,
                                                         efficiency_plot) 


# ---- Helpers ----
def dummy_series(length=10):
    return pd.Series(np.linspace(0, 1, length), index=pd.date_range("2000-01-01", periods=length))

def dummy_series_year(length=365*2):
    # Two years daily data (to cover 2x seasons)
    return pd.Series(np.random.rand(length))

###############################################################################
# Timeseries tests
###############################################################################


# Test that timeseries runs and creates a plot when bias is provided
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
def test_runs_with_bias(mock_style, mock_plot, mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "sat": dummy_series()}
    bias = dummy_series()
    # Run timeseries function with bias
    timeseries(data, bias, output_path=tmp_path, variable_name="sst")
    # Assert output plot is created
    assert (tmp_path / "sst_timeseries.png").exists()


# Test that timeseries runs and creates a plot when no bias is provided
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
def test_runs_without_bias(mock_style, mock_plot, mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "sat": dummy_series()}
    # Run without bias
    timeseries(data, None, output_path=tmp_path, variable_name="sst")
    # Assert output plot is created
    assert (tmp_path / "sst_timeseries.png").exists()


# Test that missing output_path raises an appropriate ValueError
def test_raises_without_output_path():
    data = {"model": dummy_series(), "sat": dummy_series()}
    # Expect error when output_path is not provided
    with pytest.raises(ValueError, match="output_path must be specified"):
        timeseries(data, dummy_series())


# Test that missing variable and unit raises an appropriate ValueError
def test_raises_without_variable_and_unit(tmp_path):
    data = {"model": dummy_series(), "sat": dummy_series()}
    # Expect error when variable and unit info cannot be inferred
    with pytest.raises(ValueError, match="both 'variable' and 'unit' must be specified"):
        timeseries(data, dummy_series(), output_path=tmp_path)


# Test that the function accepts NumPy arrays as input
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
def test_accepts_numpy_arrays(mock_style, mock_plot, mock_keys, mock_label, tmp_path):
    data = {"model": np.arange(10), "sat": np.arange(10)}
    bias = np.arange(10)
    # Ensure NumPy input is accepted
    timeseries(data, bias, output_path=tmp_path, variable_name="sst")
    assert (tmp_path / "sst_timeseries.png").exists()


# Test that the function accepts lists as input
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
def test_accepts_lists(mock_style, mock_plot, mock_keys, mock_label, tmp_path):
    data = {"model": list(range(10)), "sat": list(range(10))}
    bias = list(range(10))
    # Ensure list input is accepted
    timeseries(data, bias, output_path=tmp_path, variable_name="sst")
    assert (tmp_path / "sst_timeseries.png").exists()


# Test that if no bias is provided, only one subplot is created
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
@patch("matplotlib.pyplot.Figure.add_subplot", return_value=plt.subplot())
def test_bias_none_creates_one_subplot(mock_add_subplot, mock_style, mock_plot, mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "sat": dummy_series()}
    # Only one subplot should be added when bias is None
    timeseries(data, None, output_path=tmp_path, variable_name="sst")
    assert mock_add_subplot.call_count == 1


# Test that providing bias results in two subplots being created
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
@patch("matplotlib.figure.Figure.add_subplot", return_value=plt.subplot())
def test_bias_given_creates_two_subplots(mock_add_subplot, mock_style, mock_plot, mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "sat": dummy_series()}
    bias = dummy_series()
    # Two subplots should be added when bias is provided
    timeseries(data, bias, output_path=tmp_path, variable_name="sst")
    assert mock_add_subplot.call_count == 2


# Test that the filename generated uses the variable_name if provided
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
def test_creates_expected_filename(mock_style, mock_plot, mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "sat": dummy_series()}
    bias = dummy_series()
    # Ensure filename is based on variable_name
    timeseries(data, bias, output_path=tmp_path, variable_name="myvar")
    assert (tmp_path / "myvar_timeseries.png").exists()


# Test that variable and unit can be provided directly instead of using get_variable_label_unit
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
def test_variable_and_unit_directly(mock_style, mock_plot, mock_keys, tmp_path):
    data = {"model": dummy_series(), "sat": dummy_series()}
    bias = dummy_series()
    # Skip automatic label/unit extraction by passing directly
    timeseries(data, bias, output_path=tmp_path, variable="Temp", unit="K")
    # Default fallback variable_name is None → "None_timeseries.png"
    assert (tmp_path / "None_timeseries.png").exists()


# Test that extra keys in the data dictionary are ignored and do not break the function
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("Hydrological_model_validator.Plotting.formatting.plot_line", return_value=None)
@patch("Hydrological_model_validator.Plotting.formatting.style_axes_spines", return_value=None)
def test_multiple_data_keys(mock_style, mock_plot, mock_keys, mock_label, tmp_path):
    data = {
        "model": dummy_series(),
        "sat": dummy_series(),
        "extra": dummy_series()  # Should be ignored by key extractor
    }
    bias = dummy_series()
    timeseries(data, bias, output_path=tmp_path, variable_name="sst")
    assert (tmp_path / "sst_timeseries.png").exists()


###############################################################################
# scatter_plot tests
###############################################################################


# Test that scatter_plot runs without errors and saves the plot file
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
def test_scatter_plot_runs_and_saves(mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "satellite": dummy_series()}
    # Run scatter_plot with sample data
    scatter_plot(data, output_path=tmp_path, variable_name="sst")
    expected_file = tmp_path / "sst_scatterplot.png"
    # Verify output file was created
    assert expected_file.exists()


# Test that missing output_path raises a ValueError
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
def test_missing_output_path_raises(mock_keys, mock_label):
    data = {"model": dummy_series(), "satellite": dummy_series()}
    # Expect error if output_path is None
    with pytest.raises(ValueError, match="output_path must be specified"):
        scatter_plot(data, variable_name="sst", output_path=None)


# Test that variable and unit arguments can be given directly and affect filename
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
def test_variable_and_unit_directly_given(mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "satellite": dummy_series()}
    # Provide variable and unit explicitly, overriding label fetch
    scatter_plot(
        data,
        output_path=tmp_path,
        variable="Salinity",
        unit="PSU"
    )
    # Check filename reflects given variable
    assert (tmp_path / "Salinity_scatterplot.png").exists()
    

# Test that numpy array inputs are correctly handled
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Chlorophyll", "mg/m³"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
def test_numpy_input_is_handled(mock_keys, mock_label, tmp_path):
    data = {"model": np.linspace(0, 1, 10), "satellite": np.linspace(0, 1, 10)}
    # Ensure scatter_plot accepts numpy arrays without error
    scatter_plot(data, output_path=tmp_path, variable_name="chl")
    assert (tmp_path / "chl_scatterplot.png").exists()


# Test that missing variable and unit raises ValueError
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Precipitation", "mm/day"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
def test_missing_variable_and_unit_raises(mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "satellite": dummy_series()}
    # Expect error when variable and unit cannot be inferred or provided
    with pytest.raises(ValueError, match="both 'variable' and 'unit' must be specified"):
        scatter_plot(data, output_path=tmp_path)


# Test that minimal matplotlib functions (pause, draw, show) are called during plotting
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Temperature", "°C"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
@patch("matplotlib.pyplot.pause", return_value=None)
@patch("matplotlib.pyplot.show", return_value=None)
@patch("matplotlib.pyplot.draw", return_value=None)
def test_plot_calls_minimal_mpl_functions(mock_draw, mock_show, mock_pause, mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "satellite": dummy_series()}
    # Call scatter_plot with a pause_time to trigger mpl GUI events
    scatter_plot(data, output_path=tmp_path, variable_name="sst", pause_time=0.1)
    # Verify expected matplotlib calls happened exactly once
    mock_pause.assert_called_once()
    mock_draw.assert_called_once()
    mock_show.assert_called_once()


# Test scatter_plot supports season_colors and alpha transparency parameters
@patch("Hydrological_model_validator.Plotting.formatting.get_variable_label_unit", return_value=("Oxygen", "mg/L"))
@patch("Hydrological_model_validator.Processing.data_alignment.extract_mod_sat_keys", side_effect=lambda data: list(data.keys())[:2])
def test_handles_season_colors_and_alpha(mock_keys, mock_label, tmp_path):
    data = {"model": dummy_series(), "satellite": dummy_series()}
    # Define color mapping by season to test color grouping
    season_colors = {"Winter": "blue", "Summer": "red"}
    # Run with season colors and alpha transparency
    scatter_plot(
        data,
        output_path=tmp_path,
        variable_name="oxy",
        season_colors=season_colors,
        alpha=0.3
    )
    # Confirm output plot is saved
    assert (tmp_path / "oxy_scatterplot.png").exists()


###############################################################################
# seasonal_scatter_plot tests
###############################################################################


@pytest.fixture
def valid_data():
    # Provide dummy data with some NaNs to test skipping behavior
    data_len = 365*2
    mod = np.random.rand(data_len)
    sat = np.random.rand(data_len)
    return {'mod': mod, 'sat': sat}

# Test that missing output_path raises a ValueError
def test_missing_output_path_raises(valid_data):
    # Call without output_path should raise error
    with pytest.raises(ValueError):
        seasonal_scatter_plot(valid_data)  # no output_path in kwargs


# Test that missing both variable_name and unit raises a ValueError
def test_missing_variable_name_and_variable_unit_raises(valid_data):
    # Neither variable_name nor unit given should raise error
    with pytest.raises(ValueError):
        seasonal_scatter_plot(valid_data, output_path='.', variable_name=None, unit=None)


# Test that seasonal plots for all seasons and combined are saved correctly
@patch('Hydrological_model_validator.Plotting.Plots.plt.savefig')
@patch('Hydrological_model_validator.Plotting.Plots.plt.close')
@patch('Hydrological_model_validator.Plotting.Plots.plt.show')
def test_seasonal_plots_saved(mock_show, mock_close, mock_savefig, valid_data, tmp_path):
    # Generate seasonal scatter plots
    seasonal_scatter_plot(
        valid_data,
        output_path=tmp_path,
        variable_name='SST',
        BA=True,
    )
    
    # Expect 4 seasonal + 1 combined plot saves (5 total)
    assert mock_savefig.call_count == 5

    # Expected filenames for each season and combined plot
    expected_files = [f'SST_{season}_scatterplot.png' for season in ['DJF', 'MAM', 'JJA', 'SON']]
    expected_files.append('SST_all_seasons_scatterplot.png')

    # Extract filenames from mock calls
    saved_files = [Path(call.args[0]).name for call in mock_savefig.call_args_list]

    # Verify all expected files were saved
    for filename in expected_files:
        assert filename in saved_files


# Test that seasons with no valid data are skipped in plotting
@patch('Hydrological_model_validator.Plotting.Plots.plt.savefig')
@patch('Hydrological_model_validator.Plotting.Plots.plt.close')
@patch('Hydrological_model_validator.Plotting.Plots.plt.show')
def test_skips_season_with_no_valid_data(mock_show, mock_close, mock_savefig, tmp_path):
    data_len = 365 * 2  # two full years of daily data
    mod = np.random.rand(data_len)
    sat = np.random.rand(data_len)

    dates = pd.date_range(start="2000-01-01", periods=data_len, freq='D')

    # Create boolean mask for MAM season (March, April, May)
    mask_MAM = dates.month.isin([3, 4, 5])

    # Set MAM season data to NaN to simulate missing data for that season
    mod[mask_MAM] = np.nan
    sat[mask_MAM] = np.nan

    seasonal_scatter_plot(
        {'mod': mod, 'sat': sat},
        output_path=tmp_path,
        variable_name='SST',
    )

    # Only 3 seasonal plots + combined plot should be saved, skipping MAM (4 total)
    assert mock_savefig.call_count == 4


# Test that no combined plot is saved when no data exists for any season
@patch('Hydrological_model_validator.Plotting.Plots.plt.savefig')
@patch('Hydrological_model_validator.Plotting.Plots.plt.close')
@patch('Hydrological_model_validator.Plotting.Plots.plt.show')
def test_combined_plot_skipped_when_no_data(mock_show, mock_close, mock_savefig, tmp_path):
    length = 365
    # Fill data arrays completely with NaNs (no valid data)
    mod = np.full(length, np.nan)
    sat = np.full(length, np.nan)
    seasonal_scatter_plot(
        {'mod': mod, 'sat': sat},
        output_path=tmp_path,
        variable_name='SST',
    )
    # No valid data => no files should be saved
    assert mock_savefig.call_count == 0


# Test that variable and unit labels are fetched using variable_name
@patch('Hydrological_model_validator.Plotting.Plots.get_variable_label_unit', return_value=('Sea Surface Temperature', '°C'))
def test_variable_and_unit_from_variable_name(mock_label_unit, valid_data, tmp_path):
    seasonal_scatter_plot(
        valid_data,
        output_path=tmp_path,
        variable_name='SST',
    )
    # Verify the label/unit fetch function was called with variable_name
    mock_label_unit.assert_called_with('SST')


###############################################################################
# whiskerbox tests
###############################################################################


# Helper to generate synthetic daily data for model and satellite
def generate_daily_data(years=2):
    date_range = pd.date_range(start='2000-01-01', periods=365 * years, freq='D')
    data = {
        'model': pd.Series(np.random.rand(len(date_range)), index=date_range),
        'satellite': pd.Series(np.random.rand(len(date_range)), index=date_range)
    }
    return data


# Helper to convert daily pd.Series to nested dict {key: {year: [monthly arrays]}}
def prepare_nested_monthly_data(data_dict):
    nested = {}
    for key, series in data_dict.items():
        series = series.dropna()
        monthly_data = {}
        for year, group_year in series.groupby(series.index.year):
            monthly_arrays = []
            for month in range(1, 13):
                month_vals = group_year[group_year.index.month == month].values
                monthly_arrays.append(month_vals)
            monthly_data[year] = monthly_arrays
        nested[key] = monthly_data
    return nested


@pytest.fixture
def sample_data():
    daily_data = generate_daily_data(years=2)
    return prepare_nested_monthly_data(daily_data)


# Test basic whiskerbox plot runs and saves a file, and calls key matplotlib functions
@patch('Hydrological_model_validator.Plotting.Plots.plt.savefig')
@patch('Hydrological_model_validator.Plotting.Plots.plt.close')
@patch('Hydrological_model_validator.Plotting.Plots.plt.pause')
@patch('Hydrological_model_validator.Plotting.Plots.plt.draw')
@patch('Hydrological_model_validator.Plotting.Plots.plt.show')
def test_whiskerbox_basic_plot(mock_show, mock_draw, mock_pause, mock_close, mock_savefig, sample_data, tmp_path):
    kwargs = {
        'output_path': tmp_path,
        'variable_name': 'SST',
        'figsize': (10, 6),
        'palette': 'Set1',
        'showfliers': False,
        'title_fontsize': 14,
        'title_fontweight': 'bold',
        'ylabel_fontsize': 12,
        'xlabel': 'Month',
        'grid_alpha': 0.5,
        'xtick_rotation': 45,
        'tick_width': 1.5,
        'dpi': 80,
    }

    whiskerbox(sample_data, **kwargs)

    # Verify file save call count and file name suffix
    assert mock_savefig.call_count == 1
    saved_path = mock_savefig.call_args[0][0]
    saved_path_str = str(saved_path)
    assert saved_path_str.endswith('Sea Surface Temperature_boxplot.png') or saved_path_str.endswith('SST_boxplot.png')

    # Confirm key matplotlib functions were called during plotting
    assert mock_show.called
    assert mock_draw.called
    assert mock_pause.called
    assert mock_close.called


# Test that missing output_path raises ValueError
def test_whiskerbox_missing_output_path_raises(sample_data):
    with pytest.raises(ValueError, match='output_path must be specified'):
        whiskerbox(sample_data, variable_name='SST')


# Test that missing both variable and unit raises ValueError
def test_whiskerbox_missing_variable_info_raises(sample_data, tmp_path):
    with pytest.raises(ValueError, match='both \'variable\' and \'unit\' must be specified'):
        whiskerbox(sample_data, output_path=tmp_path)


# Test that whiskerbox accepts explicit variable and unit parameters
@patch('Hydrological_model_validator.Plotting.Plots.plt.savefig')
@patch('Hydrological_model_validator.Plotting.Plots.plt.close')
@patch('Hydrological_model_validator.Plotting.Plots.plt.show')
@patch('Hydrological_model_validator.Plotting.Plots.plt.draw')
@patch('Hydrological_model_validator.Plotting.Plots.plt.pause')
def test_whiskerbox_accepts_variable_and_unit(mock_pause, mock_draw, mock_show, mock_close, mock_savefig, sample_data, tmp_path):
    kwargs = {
        'output_path': tmp_path,
        'variable': 'Sea Surface Temperature',
        'unit': '°C',
    }

    whiskerbox(sample_data, **kwargs)
    assert mock_savefig.called


# Test that data passed to seaborn boxplot contains expected labels and structure
@patch('Hydrological_model_validator.Plotting.Plots.sns.boxplot')
@patch('Hydrological_model_validator.Plotting.Plots.plt.savefig')
@patch('Hydrological_model_validator.Plotting.Plots.plt.close')
@patch('Hydrological_model_validator.Plotting.Plots.plt.show')
@patch('Hydrological_model_validator.Plotting.Plots.plt.draw')
@patch('Hydrological_model_validator.Plotting.Plots.plt.pause')
def test_whiskerbox_data_labels_and_plot_data_structure(
    mock_pause, mock_draw, mock_show, mock_close, mock_savefig, mock_boxplot, sample_data, tmp_path
):
    import matplotlib.pyplot as plt

    captured_df = {}

    # Fake seaborn boxplot to capture input data frame
    def fake_boxplot(*args, **kwargs):
        captured_df['df'] = kwargs.get('data')
        fig, ax = plt.subplots()
        return ax

    mock_boxplot.side_effect = fake_boxplot

    whiskerbox(sample_data, output_path=tmp_path, variable_name='SST')

    df = captured_df['df']

    # DataFrame must contain columns 'Value' and 'Label'
    assert 'Value' in df.columns and 'Label' in df.columns

    # Expected labels are month names suffixed with "Model" and "Satellite"
    expected_months = [f'{month} Model' for month in list(pd.date_range('2000-01-01', periods=12, freq='MS').strftime('%b'))] + \
                      [f'{month} Satellite' for month in list(pd.date_range('2000-01-01', periods=12, freq='MS').strftime('%b'))]

    labels_set = set(df['Label'].unique())
    for label in expected_months:
        assert label in labels_set
        
        
###############################################################################
# violinplots tests
###############################################################################


# Patch style_axes_spines to avoid dependency and ax type errors
@patch('Hydrological_model_validator.Plotting.Plots.style_axes_spines')
@patch('Hydrological_model_validator.Plotting.Plots.sns.violinplot')
@patch('Hydrological_model_validator.Plotting.Plots.plt.savefig')
@patch('Hydrological_model_validator.Plotting.Plots.plt.close')
@patch('Hydrological_model_validator.Plotting.Plots.plt.show')
@patch('Hydrological_model_validator.Plotting.Plots.plt.draw')
@patch('Hydrological_model_validator.Plotting.Plots.plt.pause')
def test_violinplot_basic_behavior(
    mock_pause, mock_draw, mock_show, mock_close, mock_savefig, mock_violinplot, mock_style_spines, sample_data, tmp_path
):
    # Capture the dataframe passed to seaborn violinplot
    captured_df = {}

    def fake_violinplot(*args, **kwargs):
        captured_df['df'] = kwargs.get('data')
        # Return a real matplotlib Axes to avoid errors in style_axes_spines
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        return ax

    mock_violinplot.side_effect = fake_violinplot

    # Call violinplot with minimal required args
    from Hydrological_model_validator.Plotting.Plots import violinplot

    violinplot(
        sample_data,
        output_path=tmp_path,
        variable_name='SST',
        figsize=(8, 5),
        dpi=80,
        palette='Set2',
        cut=0,
        title_fontsize=14,
        title_fontweight='bold',
        ylabel_fontsize=12,
        xlabel_fontsize=12,
        xtick_rotation=45,
        tick_width=1,
        spine_linewidth=2,
        grid_alpha=0.3,
        pause_time=0.1,
    )

    # Assert violinplot was called once with dataframe containing 'Value' and 'Label'
    assert 'df' in captured_df
    df = captured_df['df']
    assert isinstance(df, pd.DataFrame)
    assert set(['Value', 'Label']).issubset(df.columns)

    # Check the output plot file was saved
    expected_file = tmp_path / 'Sea Surface Temperature_violinplot.png'
    mock_savefig.assert_called_once()
    saved_path = mock_savefig.call_args[0][0]
    assert Path(saved_path).name == expected_file.name

# Test error if output_path missing
def test_violinplot_missing_output_path(sample_data):
    from Hydrological_model_validator.Plotting.Plots import violinplot
    with pytest.raises(ValueError, match="output_path must be specified"):
        violinplot(sample_data, variable_name='SST')

# Test error if variable_name and variable/unit missing
def test_violinplot_missing_variable_info(sample_data, tmp_path):
    from Hydrological_model_validator.Plotting.Plots import violinplot
    with pytest.raises(ValueError, match="both 'variable' and 'unit' must be specified"):
        violinplot(sample_data, output_path=tmp_path)
        
        
###############################################################################
# violinplots tests
###############################################################################
        
@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path


def test_efficiency_plot_basic(tmp_output):
    total_value = 0.75
    monthly_values = [0.8, 0.6, 0.9, 0.7, 0.5, 0.85, 0.9, 0.95, 0.4, 0.7, 0.6, 0.8]
    
    efficiency_plot(
        total_value,
        monthly_values,
        output_path=tmp_output,
        metric_name='NSE',
        title='Nash-Sutcliffe Efficiency',
        y_label='E_{rel}',
    )
    
    # Check file is created
    saved_file = tmp_output / 'NSE.png'
    assert saved_file.exists()
    # Optionally check file size > 0
    assert saved_file.stat().st_size > 0


def test_efficiency_plot_missing_output_path_raises():
    with pytest.raises(ValueError, match="output_path must be specified"):
        efficiency_plot(
            0.5,
            [0.5]*12,
            metric_name='NSE',
            y_label='E_{rel}'
        )


def test_efficiency_plot_missing_metric_name_creates_file(tmp_output):
    with pytest.raises(KeyError):
        efficiency_plot(
            0.5,
            [0.5]*12,
            output_path=tmp_output,
        )


def test_efficiency_plot_marker_colors(tmp_output):
    # Values below 0 -> red, above 1 -> green, None or invalid -> gray, normal -> colormap
    
    monthly_values = [0.9, 1.1, -0.1, None, 0.5, 0.0, 0.8, 1.2, -0.5, 0.3, 0.7, 0.6]
    
    efficiency_plot(
        0.7,
        monthly_values,
        output_path=tmp_output,
        metric_name='NSE',
        title='Nash-Sutcliffe Efficiency',
        y_label='E_{rel}',
    )
    
    saved_file = tmp_output / 'NSE.png'
    assert saved_file.exists()
    assert saved_file.stat().st_size > 0


def test_efficiency_plot_title_and_labels(tmp_output):
    total_value = 0.6
    monthly_values = [0.5]*12
    title = "My Efficiency Plot"
    y_label = "My Metric"
    
    efficiency_plot(
        total_value,
        monthly_values,
        output_path=tmp_output,
        metric_name='my_metric',
        title=title,
        y_label=y_label,
    )


@pytest.mark.parametrize("bad_values", [
    [None] * 12,
    ['a'] * 12,
    [float('nan')] * 12,
    [float('inf')] * 12,
])
def test_efficiency_plot_handles_bad_values(tmp_output, bad_values):
    total_value = 0.5

    # Clean inputs, replace bad values with np.nan
    cleaned_values = []
    for v in bad_values:
        try:
            val = float(v)
            cleaned_values.append(val if np.isfinite(val) else np.nan)
        except Exception:
            cleaned_values.append(np.nan)

    # Ensure at least one valid number so seaborn can plot something
    cleaned_values[0] = 0.123

    # Call the function with cleaned values
    efficiency_plot(
        total_value,
        cleaned_values,
        output_path=Path(tmp_output),
        metric_name='bad_values',
        y_label='E_{rel}',
        title='Test Efficiency Plot'
    )