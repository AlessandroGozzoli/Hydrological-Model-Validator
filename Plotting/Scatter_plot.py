import matplotlib.pyplot as plt

def scatter_plot_BASST(daily_means_dict, variable_name, BA=False):
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
    plt.figure(figsize=(10, 8), dpi=300)
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
    plt.show()