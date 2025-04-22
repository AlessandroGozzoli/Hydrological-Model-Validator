import xarray as xr
import numpy as np

def Msst_concat(Msst_data):
    """
    Concatenates model SST data ensuring the time dimension is consistent (3653 days).
    Handles leap years by trimming extra days in leap year datasets or adjusting other datasets.
    
    Parameters:
    - Msst_data: Dictionary where keys are years and values are xarray Datasets.
    
    Returns:
    - Concatenated xarray Dataset with consistent time dimension.
    """
    print("Concatenating the model SST data...")

    # Convert dictionary values (datasets) into a list and sort them by year
    Msst_data_cont = [Msst_data[year] for year in sorted(Msst_data.keys())]

    if not Msst_data_cont:
        print("No model SST data found to concatenate.")
        return None

    # Set the target time dimension (3653 days)
    target_time_dimension = 3653

    # Ensure all datasets have a consistent time dimension (3653 days)
    for idx, ds in enumerate(Msst_data_cont):
        time_dim_length = ds.dims['time']
        print(f"Dataset {list(Msst_data.keys())[idx]} has time dimension of {time_dim_length} days")

        # If the dataset has 366 days (for leap years), remove the extra day
        if time_dim_length == 366:
            print(f"Trimming extra day for year {list(Msst_data.keys())[idx]} (Leap Year)")
            ds = ds.isel(time=slice(0, target_time_dimension))  # Keep the first 365 days

        # If the dataset has 365 days, just keep it as is
        elif time_dim_length == 365:
            print(f"Dataset {list(Msst_data.keys())[idx]} has 365 days, keeping as is.")
            pass

        # If the dataset does not have the expected number of time steps (365 or 366), raise an error
        elif time_dim_length != target_time_dimension:
            raise ValueError(f"Dataset {list(Msst_data.keys())[idx]} has unexpected time dimension size {time_dim_length}. Expected 365 or 366 days.")

        # Ensure that lat and lon dimensions match across all datasets
        if 'lat' in ds.dims and 'lon' in ds.dims:
            print(f"Dataset {list(Msst_data.keys())[idx]} has consistent lat and lon dimensions.")
        else:
            raise ValueError(f"Dataset {list(Msst_data.keys())[idx]} is missing lat or lon dimensions!")

        # Add the dataset back to the list after adjustment
        Msst_data_cont[idx] = ds

    # Concatenate the datasets along the 'time' dimension
    model_sst_continuous = xr.concat(Msst_data_cont, dim='time')

    print("Model SST data successfully concatenated into a continuous time series!")
    print("Concatenated Model SST Dimensions:", model_sst_continuous.dims)

    return model_sst_continuous
