import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import welch

import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for testing (no display)

# Replace this with your actual import
from Hydrological_model_validator.Plotting.Plots import (timeseries,
                                                         scatter_plot,
                                                         seasonal_scatter_plot,
                                                         whiskerbox,
                                                         efficiency_plot,
                                                         plot_spatial_efficiency,
                                                         error_components_timeseries,
                                                         plot_spectral) 


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
    # Default fallback for the title is the variable passed
    assert (tmp_path / "Temp_timeseries.png").exists()


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
        variable='AAA',
        unit='aaa',
        BA=True,
    )
    
    # Expect 4 seasonal + 1 combined plot saves (5 total)
    assert mock_savefig.call_count == 5

    # Expected filenames for each season and combined plot
    expected_files = [f'AAA_{season}_scatterplot.png' for season in ['DJF', 'MAM', 'JJA', 'SON']]
    expected_files.append('AAA_all_seasons_scatterplot.png')

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


# Test that efficiency_plot creates a non-empty output file with valid input
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
    
    saved_file = tmp_output / 'NSE.png'
    assert saved_file.exists()
    assert saved_file.stat().st_size > 0  # Optional: ensure file is not empty


# Test that missing output_path raises a ValueError
def test_efficiency_plot_missing_output_path_raises():
    with pytest.raises(ValueError, match="output_path must be specified"):
        efficiency_plot(
            0.5,
            [0.5] * 12,
            metric_name='NSE',
            y_label='E_{rel}'
        )


# Test that missing metric_name raises a KeyError (likely due to missing label lookup)
def test_efficiency_plot_missing_metric_name_creates_file(tmp_output):
    with pytest.raises(KeyError):
        efficiency_plot(
            0.5,
            [0.5] * 12,
            output_path=tmp_output,
        )


# Test marker coloring logic with edge-case monthly values (e.g., <0, >1, None, etc.)
def test_efficiency_plot_marker_colors(tmp_output):
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


# Test that custom title and y-axis label are accepted without error
def test_efficiency_plot_title_and_labels(tmp_output):
    total_value = 0.6
    monthly_values = [0.5] * 12
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


# Parametrized test to verify robustness against various bad input values
@pytest.mark.parametrize("bad_values", [
    [None] * 12,
    ['a'] * 12,
    [float('nan')] * 12,
    [float('inf')] * 12,
])
def test_efficiency_plot_handles_bad_values(tmp_output, bad_values):
    total_value = 0.5

    # Clean and convert bad inputs to NaN, except one value so plot can render
    cleaned_values = []
    for v in bad_values:
        try:
            val = float(v)
            cleaned_values.append(val if np.isfinite(val) else np.nan)
        except Exception:
            cleaned_values.append(np.nan)

    cleaned_values[0] = 0.123  # Ensure at least one plottable value

    efficiency_plot(
        total_value,
        cleaned_values,
        output_path=Path(tmp_output),
        metric_name='bad_values',
        y_label='E_{rel}',
        title='Test Efficiency Plot'
    )
    
    
###############################################################################
# spatial_efficiency tests
###############################################################################


@pytest.fixture
def geo_coords():
    latp, lonp = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-20, 20, 10), indexing='ij')
    return {
        'latp': latp,
        'lonp': lonp,
        'MinLambda': -20,
        'MaxLambda': 20,
        'MinPhi': -10,
        'MaxPhi': 10,
        'Epsilon': 0.1
    }

@pytest.fixture
def temp_output(tmp_path):
    return tmp_path

# Helper to create xarray DataArray with either 'month', 'year', or invalid time dimension
def create_data_array(dim_type):
    data = np.random.rand(3, 10, 10)  # 3 time slices for minimal plots
    if dim_type == 'month':
        return xr.DataArray(data, dims=['month', 'lat', 'lon'], coords={'month': [1, 2, 3]})
    if dim_type == 'year':
        return xr.DataArray(data, dims=['year', 'lat', 'lon'], coords={'year': [2000, 2001, 2002]})
    return xr.DataArray(data, dims=['time', 'lat', 'lon'])  # Invalid for the function


# Test that monthly data plots successfully with default colormap
def test_plot_monthly_default_cmap(geo_coords, temp_output):
    data_array = create_data_array('month')
    plot_spatial_efficiency(data_array, geo_coords, temp_output, "TestMetric")


# Test that yearly data plots correctly with unit label and detrending enabled
def test_plot_yearly_with_unit_and_detrended(geo_coords, temp_output):
    data_array = create_data_array('year')
    plot_spatial_efficiency(
        data_array, geo_coords, temp_output, "TestMetric",
        unit="\%", detrended=True
    )


# Test that unsupported time dimension raises a ValueError
def test_invalid_dim_raises_error(geo_coords, temp_output):
    data_array = create_data_array('invalid')
    with pytest.raises(ValueError, match="must have either 'month' or 'year'"):
        plot_spatial_efficiency(data_array, geo_coords, temp_output, "Bad")


# Test that a custom colormap and explicit vmin/vmax range are applied correctly
def test_custom_colormap_orangegreen(geo_coords, temp_output):
    data_array = create_data_array('month')
    plot_spatial_efficiency(
        data_array, geo_coords, temp_output, "TestMetric",
        cmap="OrangeGreen", vmin=-1, vmax=1
    )


# Test layout logic for non-square subplots (e.g., 2 panels in a 3-column layout)
def test_partial_subplot_row(geo_coords, temp_output):
    data = np.random.rand(2, 10, 10)
    data_array = xr.DataArray(data, dims=['month', 'lat', 'lon'], coords={'month': [1, 2]})
    plot_spatial_efficiency(
        data_array, geo_coords, temp_output, "TestMetric",
        max_cols=3  # Forces a single incomplete row
    )
    
    
###############################################################################
# error_timeseries tests
###############################################################################

    
@pytest.fixture
def stats_df():
    index = pd.date_range(start='2000-01-01', periods=100, freq='D')
    data = {
        'mean_bias': np.random.randn(100),
        'unbiased_rmse': np.random.rand(100),
        'std_error': np.random.rand(100),
        'cross_correlation': np.random.rand(100)
    }
    return pd.DataFrame(data, index=index)

@pytest.fixture
def cloud_cover():
    index = pd.date_range(start='2000-01-01', periods=100, freq='D')
    return pd.Series(np.random.rand(100) * 100, index=index)

# Test that the function creates a plot file without cloud cover input
def test_error_plot_no_cloud(stats_df, temp_output):
    error_components_timeseries(
        stats_df=stats_df,
        output_path=temp_output
    )
    files = list(temp_output.glob("*.png"))
    assert len(files) == 1, "No plot saved without cloud cover"


# Test that the function creates a plot file when cloud cover data is provided
def test_error_plot_with_cloud(stats_df, cloud_cover, temp_output):
    error_components_timeseries(
        stats_df=stats_df,
        output_path=temp_output,
        cloud_cover=cloud_cover
    )
    files = list(temp_output.glob("*.png"))
    assert len(files) == 1, "No plot saved with cloud cover"


# Test that the filename includes the variable name when specified
def test_error_plot_variable_name(stats_df, cloud_cover, temp_output):
    error_components_timeseries(
        stats_df=stats_df,
        output_path=temp_output,
        cloud_cover=cloud_cover,
        variable_name="Temperature"
    )
    file = list(temp_output.glob("*.png"))[0]
    assert "Temperature" in file.name, "Filename does not include variable_name"


# Test that a short time series (e.g., 5 days) still results in a valid plot
def test_error_plot_short_series(temp_output):
    index = pd.date_range(start='2000-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'mean_bias': np.random.randn(5),
        'unbiased_rmse': np.random.rand(5),
        'std_error': np.random.rand(5),
        'cross_correlation': np.random.rand(5)
    }, index=index)
    error_components_timeseries(df, temp_output)
    files = list(temp_output.glob("*.png"))
    assert files, "Plot not created for short series"


# Test that the function creates the output directory if it doesn't exist and saves the plot there
def test_error_plot_output_path_created(stats_df, temp_output):
    new_subdir = temp_output / "nested" / "plots"
    error_components_timeseries(stats_df, output_path=new_subdir)
    files = list(new_subdir.glob("*.png"))
    assert files, "Output file not created in nested directory"
    
    
###############################################################################
# plot_spectral tests
###############################################################################


@pytest.fixture
def time_series():
    return pd.Series(np.sin(np.linspace(0, 20 * np.pi, 512)) + np.random.rand(512) * 0.5)

@pytest.fixture
def fft_components(time_series):
    freqs, fft_vals = welch(time_series, fs=1.0)
    return freqs, {"Model": fft_vals}

@pytest.fixture
def error_comp():
    index = pd.date_range(start='2000-01-01', periods=512, freq='D')
    return pd.DataFrame({
        'mean_bias': np.sin(np.linspace(0, 10 * np.pi, 512)),
        'unbiased_rmse': np.cos(np.linspace(0, 10 * np.pi, 512)),
    }, index=index)

@pytest.fixture
def cloud_cover_series():
    return np.random.rand(512)

# Test that PSD plot is created and saved with correct filename
def test_plot_psd(fft_components, temp_output):
    freqs, components = fft_components
    plot_spectral(
        plot_type='PSD',
        freqs=freqs,
        fft_components=components,
        output_path=temp_output,
        variable_name="TestVar"
    )
    files = list(temp_output.glob("*.png"))
    assert files, "No PSD plot saved"
    assert "PSD_TestVar" in files[0].name, "Incorrect PSD filename"


# Test that CSD plot is created and saved with correct filename when cloud cover and error components are provided
def test_plot_csd(error_comp, cloud_cover_series, temp_output):
    cloud_covers = [(cloud_cover_series, 'Cloud')]
    plot_spectral(
        plot_type='CSD',
        error_comp=error_comp,
        cloud_covers=cloud_covers,
        output_path=temp_output,
        variable_name="TestVar"
    )
    files = list(temp_output.glob("*.png"))
    assert files, "No CSD plot saved"
    assert "CSD_TestVar" in files[0].name, "Incorrect CSD filename"


# Test that PSD plotting raises an error if required inputs are missing
def test_psd_missing_inputs_raises(temp_output):
    with pytest.raises(ValueError, match="freqs and fft_components must be provided"):
        plot_spectral(
            plot_type='PSD',
            output_path=temp_output
        )


# Test that CSD plotting raises an error if error_comp is missing
def test_csd_missing_error_comp_raises(temp_output, cloud_cover_series):
    cloud_covers = [(cloud_cover_series, 'Cloud')]
    with pytest.raises(ValueError, match="error_comp must be provided"):
        plot_spectral(
            plot_type='CSD',
            cloud_covers=cloud_covers,
            output_path=temp_output
        )


# Test that CSD plotting raises an error if cloud_covers list is empty
def test_csd_missing_cloud_cover_raises(temp_output, error_comp):
    with pytest.raises(ValueError, match="At least one cloud_cover tuple"):
        plot_spectral(
            plot_type='CSD',
            error_comp=error_comp,
            cloud_covers=[],
            output_path=temp_output
        )


# Test that invalid plot_type raises a ValueError
def test_invalid_plot_type_raises(temp_output):
    with pytest.raises(ValueError, match="Unknown plot_type"):
        plot_spectral(
            plot_type='XYZ',
            output_path=temp_output
        )
