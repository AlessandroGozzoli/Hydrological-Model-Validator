import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import calendar
import matplotlib.colors as mcolors
from pathlib import Path

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