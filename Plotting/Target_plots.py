import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm
import calendar

from Costants import Ybeg, Tspan, DinY, ysec

def target_diagram_by_month(taylor_dict, month_index):
    """
    Generate and plot a Target diagram for the model and reference data in the provided taylor_dict,
    using only the data for the specified month (0 = January, 1 = February, ..., 11 = December) from all years.

    Parameters:
    taylor_dict (dict): Dictionary containing monthly model and satellite data for each year.
    month_index (int): The month to use for the target diagram (0 = January, 1 = February, ..., 11 = December).
    """
    # Validate the month input
    if month_index < 0 or month_index > 11:
        raise ValueError("Month must be between 0 and 11.")

    # Get the month name from the month index
    month_name = calendar.month_name[month_index + 1]  # +1 because months are 1-indexed in `calendar`

    # Close any previously open graphics windows
    plt.close('all')

    # Prepare the lists to hold statistics for the target diagram
    bias = []
    crmsd = []
    rmsd = []
    label = []
    
    # Dynamically extract the model and satellite keys from the taylor_dict
    model_key = [key for key in taylor_dict.keys() if 'mod' in key.lower()]
    sat_key = [key for key in taylor_dict.keys() if 'sat' in key.lower()]

    # Loop through the years corresponding to Ybeg to Yend (e.g., 2000 to 2009)
    for year in range(Tspan):  # Looping through years Ybeg to Yend
        # Get the model data for the current year and the ith month
        model_data = taylor_dict[model_key[0]][year + Ybeg][month_index]

        # Get the corresponding satellite data for the ith month
        ref_data = taylor_dict[sat_key[0]][year + Ybeg][month_index]

        valid_indices = ~np.isnan(model_data) & ~np.isnan(ref_data)
        
        filtered_model_data = model_data[valid_indices]
        filtered_ref_data = ref_data[valid_indices]

        # Compute the target statistics for each year
        target_stats = sm.target_statistics(filtered_model_data, filtered_ref_data, 'data')

        # Extract the statistics and append them to the lists
        bias.append(target_stats['bias'])  # Model bias
        crmsd.append(target_stats['crmsd'])  # Model CRMSD
        rmsd.append(target_stats['rmsd'])  # Model RMSD
        label.append(str(Ybeg + year))  # Append the year label
        
    print(f"The Target plot for {month_name} has been plotted!")

    # Convert lists to numpy arrays for easier manipulation
    bias = np.array(bias)
    crmsd = np.array(crmsd)
    rmsd = np.array(rmsd)

    # Produce the target diagram
    sm.target_diagram(bias, crmsd, rmsd, markerLabel=label, markerLabelColor='r')

    # Add title with the month name
    plt.title(f"Target Plot for {month_name}", pad = 50)

    # Optionally, save or show the plot
    plt.savefig(f'C:/Tesi Magistrale/Codici/Python/Plot output/Target/CHL/L4/target_plot_{month_name}.png')
    plt.show()
    plt.close()

def comprehensive_target_diagram(taylor_dict):
    """
    Generate and plot a Target diagram for the model and reference data in the provided taylor_dict.

    Parameters:
    taylor_dict (dict): Dictionary with model and reference data for each year.
    """
    # Close any previously open graphics windows
    plt.close('all')

    # Prepare the lists to hold statistics for the target diagram
    bias = []
    crmsd = []
    rmsd = []
    label = []

    # Dynamically extract the model and satellite keys from the taylor_dict
    model_key = [key for key in taylor_dict.keys() if 'mod' in key.lower()]
    sat_key = [key for key in taylor_dict.keys() if 'sat' in key.lower()]

    # Check if we found exactly one model and one satellite key
    if len(model_key) != 1 or len(sat_key) != 1:
        raise ValueError("The input dictionary must contain one key with 'mod' (model) and one key with 'sat' (satellite) data.")

    # Loop through the years corresponding to Ybeg to Yend (e.g., 2000 to 2009)
    for year in range(Tspan):  # Looping through years Ybeg to Yend
        # Get the model data for the current year from the dynamically extracted model_key
        model_data = taylor_dict[model_key[0]][year]

        # Get the corresponding satellite data from the dynamically extracted sat_key
        ref_data = taylor_dict[sat_key[0]][year]  # Access satellite data for the correct year

        # Dynamically compute the leap years based on the starting and ending year
        leap_years = [year for year in ysec if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))]

        # If the year is not a leap year, remove the extra day (366th day)
        if Ybeg + year not in leap_years:
            model_data = model_data[:DinY]  # Remove the last day of model data
            ref_data = ref_data[:DinY]  # Remove the last day of reference (satellite) data

        valid_indices = ~np.isnan(model_data) & ~np.isnan(ref_data)
        
        filtered_model_data = model_data[valid_indices]
        filtered_ref_data = ref_data[valid_indices]

        # Compute the target statistics for each year
        target_stats = sm.target_statistics(filtered_model_data, filtered_ref_data, 'data')

        # Extract the statistics and append them to the lists
        bias.append(target_stats['bias'])  # Model bias
        crmsd.append(target_stats['crmsd'])  # Model CRMSD
        rmsd.append(target_stats['rmsd'])  # Model RMSD
        label.append(str(Ybeg + year))  # Append the year label

    # Convert lists to numpy arrays for easier manipulation
    bias = np.array(bias)
    crmsd = np.array(crmsd)
    rmsd = np.array(rmsd)

    # Produce the target diagram
    sm.target_diagram(bias, crmsd, rmsd, markerLabel=label, markerLabelColor='r')

    plt.title("Comprehensive Target Plot (Yearly performance)", pad = 40)

    # Optionally, save or show the plot
    plt.savefig('C:/Tesi Magistrale/Codici/Python/Plot output/Target/CHL/L4/target_plot.png')
    plt.show()
    plt.close()
