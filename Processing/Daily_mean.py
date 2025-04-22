def compute_daily_mean_from_dataset(data_dict):
    """
    Computes the daily mean for the single variable in each xarray.Dataset in the dictionary.
    Ignores NaN values during the mean calculation. It assumes each dataset has one variable.

    Parameters:
    - data_dict: Dictionary where values are xarray Datasets containing the variable.
    
    Returns:
    - daily_means_dict: Dictionary with the same keys as input, but values are 1D arrays
      containing the daily mean for each variable.
    """
    daily_means_dict = {}

    for key, dataset in data_dict.items():
        # Get the name of the first variable in the dataset (assuming only one variable exists)
        variable_name = list(dataset.data_vars.keys())[0]
        print(f"'{variable_name}' found in {key} dataset.")

        # Access the variable data
        variable_data = dataset[variable_name]  # Access the variable

        # Compute daily mean (ignoring NaN values)
        daily_mean = variable_data.mean(dim=["lat", "lon"], skipna=True)

        # Store the result
        daily_means_dict[key] = daily_mean.values

    return daily_means_dict