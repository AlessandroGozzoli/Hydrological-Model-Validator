import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import calendar
import matplotlib.colors as mcolors
from pathlib import Path
import numpy as np 
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import spearmanr
from sklearn.linear_model import TheilSenRegressor

def plot_daily_means(output_path, daily_means_dict, variable_name, BIAS_Bavg, BA=False):
    """
    Plots the daily mean values for each dataset in the dictionary along with BIAS.

    Parameters:
    - daily_means_dict: Dictionary where the keys are dataset names, and the values are 1D arrays 
      containing the daily mean values.
    - variable_name: Name of the variable being plotted (e.g., 'SST').
    - BASSTsat: 1D array containing the daily mean values of satellite SST data.
    - BASSTmod: 1D array containing the daily mean values of model SST data.
    - BA: Boolean indicating if it's Basin Average. If True, title will include "(Basin Average)".
    """

    # Create the plot title for the first subplot
    title = f'Daily Mean Values for {variable_name} Datasets'
    if BA:
        title += ' (Basin Average)'

    # Create the figure and GridSpec (2 rows, 1 column)
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1, height_ratios=[8, 4])  # Top row 10 units high, bottom row 6 units high

    # First subplot (top row) for daily means
    ax1 = fig.add_subplot(gs[0])
    for key, daily_mean in daily_means_dict.items():
        ax1.plot(daily_mean, label=key)  # Plot daily mean for each dataset
    ax1.set_title(title)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Daily Mean')
    ax1.legend()
    ax1.grid(True)

    # Second subplot (bottom row) for BIAS
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(BIAS_Bavg, 'k-', label='BIAS')  # Plot BIAS with black line
    ax2.set_title(f'BIAS ({variable_name})')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('BIAS')
    ax2.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    output_path = Path(output_path)
    filename = f'{variable_name}_timeseries.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()  # <-- Force rendering
    plt.pause(3)
    plt.close()
    
# Function to plot the overall metric and the monthly metrics
def plot_metric(metric_name, overall_value, monthly_values, y_label, output_path):
    months = [calendar.month_name[i + 1] for i in range(12)]

    # Normalize and create color map
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Assign colors to each value
    marker_colors = []
    for val in monthly_values:
        if val < 0:
            marker_colors.append('red')
        elif val > 1:
            marker_colors.append('green')
        else:
            marker_colors.append(cmap(norm(val)))

    # Start plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add horizontal line at y=0 for relevant metrics
    metrics_with_zero_ref = [
        "Nash-Sutcliffe Efficiency",
        "Nash-Sutcliffe Efficiency (Logarithmic)",
        "Modified NSE (E₁, j=1)",
        "Relative NSE ($E_{rel}$)"
    ]
    if metric_name in metrics_with_zero_ref:
        ax.axhline(y=0, linestyle='-.', lw=3, color='r')

    # Plot overall value line
    ax.axhline(y=overall_value, linestyle='--', lw=2, label="Overall", color='k')

    # Connect monthly points with line (neutral color)
    ax.plot(months, monthly_values, color='b', lw=1.75, label="Monthly Trend")

    # Plot monthly values with colored markers
    for i, (month, val, color) in enumerate(zip(months, monthly_values, marker_colors)):
        ax.plot(month, val, marker='o', markersize=10, color=color, alpha=1, markeredgecolor='black', markeredgewidth=1.2)
    
    # Styling
    ax.set_title(f'{metric_name}', fontsize=14)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc='best')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', color='gray')
    plt.tight_layout()
    output_path = Path(output_path)
    filename = f'{metric_name}.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()  # <-- Force rendering
    plt.pause(3)
    plt.close()
    
def scatter_plot(output_path, daily_means_dict, variable_name, BA=False):
    """
    Creates a scatter plot comparing the model SST (BASSTmod) and satellite SST (BASSTsat) for each dataset
    in the daily_means_dict. It automatically extracts the necessary data from the dictionary.

    Parameters:
    - daily_means_dict: Dictionary where the keys are dataset names, and the values are 1D arrays 
      containing the daily mean values for the datasets.
    - variable_name: Name of the variable being plotted (e.g., 'SST').
    - BA: Boolean indicating if it's Basin Average. If True, title will include "(Basin Average)".
    """
    # Extract model and satellite datasets from the daily_means_dict
    BAmod = None
    BAsat = None

    # Loop through the dictionary to find the model and satellite data
    for key, daily_mean in daily_means_dict.items():
        if 'mod' in key.lower():  # Assuming 'mod' indicates model data
            BAmod = daily_mean
        elif 'sat' in key.lower():  # Assuming 'sat' indicates satellite data
            BAsat = daily_mean

    # Check if both model and satellite data were found
    if BAmod is None or BAsat is None:
        raise ValueError("Model ('mod') or Satellite ('sat') data not found in the dictionary.")

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(BAmod, BAsat, color='blue', alpha=0.5, label=f'{variable_name} Comparison')

    # Add title, labels, and legend
    title = f'Scatter Plot of {variable_name} (Model vs. Satellite)'
    if BA:
        title += ' (Basin Average)'

    plt.title(title)
    plt.xlabel(f'{variable_name} (Model)')
    plt.ylabel(f'{variable_name} (Satellite)')
    plt.legend()

    # Add a diagonal line (y=x) for reference
    min_val = min(BAsat.min(), BAmod.min())
    max_val = max(BAsat.max(), BAmod.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (Ideal)')

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    output_path = Path(output_path)
    filename = f'{variable_name}_scatterplot.png'
    save_path = output_path / filename
    plt.savefig(save_path)
    plt.show(block=False)
    plt.draw()  # <-- Force rendering
    plt.pause(3)
    plt.close()

def scatter_plot_by_season(output_path, daily_means_dict, variable_name, BA=False):
    """
    Creates seasonal scatter plots (DJF, MAM, JJA, SON) comparing model vs satellite values,
    assuming the data starts from 01/01/2000 and is daily and continuous.
    Additionally, generates a comprehensive plot with all seasons combined.

    Parameters:
    - output_path: Path to save the plots.
    - daily_means_dict: Dictionary with 'mod' and 'sat' keys and 1D daily mean value arrays.
    - variable_name: Name of the variable (e.g., 'SST').
    - BA: Boolean for Basin Average (adds to title).
    """
    # Assign dates starting from Jan 1, 2000
    sample_array = next(iter(daily_means_dict.values()))
    dates = pd.date_range(start="2000-01-01", periods=len(sample_array), freq='D')

    # Define seasons, months, and colors
    seasons = {
        'DJF': {'months': [12, 1, 2], 'color': 'gray'},
        'MAM': {'months': [3, 4, 5], 'color': 'green'},
        'JJA': {'months': [6, 7, 8], 'color': 'red'},
        'SON': {'months': [9, 10, 11], 'color': 'gold'}
    }

    # Extract model and satellite arrays
    BAmod = None
    BAsat = None
    for key, daily_mean in daily_means_dict.items():
        if 'mod' in key.lower():
            BAmod = np.array(daily_mean)
        elif 'sat' in key.lower():
            BAsat = np.array(daily_mean)

    if BAmod is None or BAsat is None:
        raise ValueError("Model ('mod') or Satellite ('sat') data not found in the dictionary.")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize list to store all season data for the comprehensive plot
    all_mod_points = []
    all_sat_points = []
    all_colors = []

    # Loop over each season
    for season_name, season_info in seasons.items():
        months = season_info['months']
        color = season_info['color']

        # Create mask for the season
        if season_name == 'DJF':
            is_season = ((dates.month == 12) & (dates.day >= 1)) | \
                        ((dates.month == 1) | (dates.month == 2))
        else:
            is_season = dates.month.isin(months)

        # Apply mask
        mod_season = BAmod[is_season]
        sat_season = BAsat[is_season]
        mod_season = np.array(mod_season)
        sat_season = np.array(sat_season)

        # Exclude NaNs
        valid_mask = ~np.isnan(mod_season) & ~np.isnan(sat_season)
        mod_season = mod_season[valid_mask]
        sat_season = sat_season[valid_mask]

        if len(mod_season) == 0 or len(sat_season) == 0:
            print(f"Skipping {season_name}: no data found.")
            continue

        # Scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(mod_season, sat_season, alpha=0.7, label=f'{variable_name} {season_name}', color=color)

        # Dynamic x range
        x_min, x_max = mod_season.min(), mod_season.max()
        x_vals = np.linspace(x_min, x_max, 100)

        # Linear regression (Huber)
        mod_season_reshaped = mod_season.reshape(-1, 1)
        huber = HuberRegressor().fit(mod_season_reshaped, sat_season)
        slope = huber.coef_[0]
        intercept = huber.intercept_
        linear_fit = slope * x_vals + intercept
        plt.plot(x_vals, linear_fit, color='black', linestyle='-', linewidth=2, label='Linear Fit (Huber)')

        # LOWESS smoother
        sorted_idx = np.argsort(mod_season)
        smoothed = lowess(sat_season[sorted_idx], mod_season[sorted_idx], frac=0.3)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color='magenta', linestyle='-.', linewidth=2, label='Smoothed Fit (LOWESS)')

        # Diagonal y = x line
        min_val = min(sat_season.min(), mod_season.min())
        max_val = max(sat_season.max(), mod_season.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Ideal Fit')

        # Title and labels
        title = f'{variable_name} Scatter Plot (Model vs Satellite) - {season_name}'
        if BA:
            title += ' (Basin Average)'
        plt.title(title)
        plt.xlabel(f'{variable_name} (Model - {season_name})')
        plt.ylabel(f'{variable_name} (Satellite - {season_name})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        filename = f"{variable_name}_{season_name}_scatterplot.png"
        save_path = output_path / filename
        plt.savefig(save_path)
        plt.show(block=False)
        plt.draw()
        plt.pause(2)
        plt.close()

        # Store points
        all_mod_points.extend(mod_season)
        all_sat_points.extend(sat_season)
        all_colors.extend([color] * len(mod_season))

    # Comprehensive plot
    all_mod_points = np.array(all_mod_points)
    all_sat_points = np.array(all_sat_points)

    if len(all_mod_points) > 0:
        plt.figure(figsize=(10, 8))
        plt.scatter(all_mod_points, all_sat_points, c=all_colors, alpha=0.7)
        plt.title(f'{variable_name} Scatter Plot (Model vs Satellite) - All Seasons')
        plt.xlabel(f'{variable_name} (Model - All Seasons)')
        plt.ylabel(f'{variable_name} (Satellite - All Seasons)')
        plt.grid(True)
        plt.tight_layout()

        # Diagonal y = x
        min_val_all = min(all_sat_points.min(), all_mod_points.min())
        max_val_all = max(all_sat_points.max(), all_mod_points.max())
        plt.plot([min_val_all, max_val_all], [min_val_all, max_val_all], 'b--', linewidth=2, label='Ideal Fit')

        # Linear fit for all
        x_min_all, x_max_all = all_mod_points.min(), all_mod_points.max()
        x_vals_all = np.linspace(x_min_all, x_max_all, 100)
        all_mod_points_reshaped = all_mod_points.reshape(-1, 1)
        huber_all = HuberRegressor().fit(all_mod_points_reshaped, all_sat_points)
        slope_all = huber_all.coef_[0]
        intercept_all = huber_all.intercept_
        linear_fit_all = slope_all * x_vals_all + intercept_all
        plt.plot(x_vals_all, linear_fit_all, color='black', linestyle='-', linewidth=2, label='Linear Fit (Huber)')

        # LOWESS smoother for all
        sorted_idx_all = np.argsort(all_mod_points)
        smoothed_all = lowess(all_sat_points[sorted_idx_all], all_mod_points[sorted_idx_all], frac=0.3)
        plt.plot(smoothed_all[:, 0], smoothed_all[:, 1], color='magenta', linestyle='-.', linewidth=2, label='Smoothed Fit (LOWESS)')

        # Custom legend
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='DJF'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='MAM'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='JJA'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10, label='SON'),
            Line2D([0], [0], color='b', linestyle='--', linewidth=2, label='Ideal Fit'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Linear Fit (Huber)'),
            Line2D([0], [0], color='magenta', linestyle='-.', linewidth=2, label='Smoothed Fit (LOWESS)')
        ]
        plt.legend(handles=handles, loc='upper left')

        # Save plot
        comprehensive_filename = f"{variable_name}_all_seasons_scatterplot.png"
        comprehensive_save_path = output_path / comprehensive_filename
        plt.savefig(comprehensive_save_path)
        plt.show(block=False)
        plt.draw()
        plt.pause(2)
        plt.close()