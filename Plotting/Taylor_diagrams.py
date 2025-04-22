import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import skill_metrics as sm
import calendar

from Costants import (
    Ybeg,
    Tspan,
    DinY,
    ysec
)

def comprehensive_taylor_diagram(taylor_dict, taylor_options, std_ref):
    """
    Generate and plot a Taylor diagram for the model and reference data in the provided taylor_dict.

    Parameters:
    taylor_dict (dict): Dictionary with model and reference data for each year.
    Vedi target 11 per come definire markers
    """
    # Set the figure properties (optional)
    rcParams["figure.figsize"] = [10.0, 8.4]
    rcParams['lines.linewidth'] = 1  # line width for plots
    rcParams.update({'font.size': 12})  # font size of axes text

    # Close any previously open graphics windows
    plt.close('all')

    # Prepare the lists to hold statistics
    # Compute standard deviation of reference data (satellite data) for the first year as an example
    sat_key = [key for key in taylor_dict.keys() if 'sat' in key.lower()]
    if len(sat_key) != 1:
        raise ValueError("The input dictionary must contain exactly one key with 'sat' (satellite) data.")

    sdev = [std_ref]  # Reference standard deviation (from satellite data)
    crmsd = [0.0]  # Reference RMSD (set manually)
    ccoef = [1.0]  # Reference correlation coefficient (set manually)
    label = ["Ref"]  # Label for the reference data

    # Dynamically compute the leap years based on the starting and ending year
    leap_years = [year for year in ysec if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))]

    # Dynamically extract the model and satellite keys from the taylor_dict
    model_key = [key for key in taylor_dict.keys() if 'mod' in key.lower()]

    # Check if we found exactly one model key
    if len(model_key) != 1:
        raise ValueError("The input dictionary must contain exactly one key with 'mod' (model) data.")

    # Loop through the years corresponding to Ybeg to Yend (e.g., 2000 to 2009)
    for year in range(Tspan):  # Looping through years Ybeg to Yend
        # Get the model data for the current year from the dynamically extracted model_key
        model_data = taylor_dict[model_key[0]][year]

        # Get the corresponding satellite data from the dynamically extracted sat_key
        ref_data = taylor_dict[sat_key[0]][year]  # Access satellite data for the correct year

        # If the year is not a leap year, remove the extra day (366th day)
        if Ybeg + year not in leap_years:
            model_data = model_data[:DinY]  # Remove the last day of model data
            ref_data = ref_data[:DinY]  # Remove the last day of reference (satellite) data
            
        valid_indices = ~np.isnan(model_data) & ~np.isnan(ref_data)
        
        filtered_model_data = model_data[valid_indices]
        filtered_ref_data = ref_data[valid_indices]

        # Compute the Taylor statistics for each year
        taylor_stats = sm.taylor_statistics(filtered_model_data, filtered_ref_data, 'data')

        # Extract the statistics and append them to the lists
        sdev = np.append(sdev, taylor_stats['sdev'][1])  # Model standard deviation
        crmsd = np.append(crmsd, taylor_stats['crmsd'][1])  # Model RMSD
        ccoef = np.append(ccoef, taylor_stats['ccoef'][1])  # Model correlation coefficient

        label.append(Ybeg + year)  # Append the year label

    # Convert lists to numpy arrays for easier manipulation
    sdev = np.array(sdev)
    crmsd = np.array(crmsd)
    ccoef = np.array(ccoef)

    # Produce the Taylor diagram
    sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label, 
                      taylor_options_file=taylor_options)
    
    plt.title("Comprehensive Taylor Diagram (Yearly performance)", pad = 40)

    # Optionally, save or show the plot
    plt.savefig('C:/Tesi Magistrale/Codici/Python/Plot output/Taylor/CHL/L4/Comprehensive_taylor_diagram.png')
    plt.show()
    plt.close()

def monthly_taylor_diagram(taylor_dict, month_index, taylor_options_monthly):
    """
    Extracts satellite data for a specific month across all years
    and computes its standard deviation.

    Parameters:
    taylor_dict (dict): Dictionary with model and reference monthly data.
    month_index (int): Index of the month (0 = January, 11 = December).

    Returns:
    float: Standard deviation of the satellite data for the selected month.
    """

    # Dynamically extract the model and satellite keys from the taylor_dict
    model_key = [key for key in taylor_dict.keys() if 'mod' in key.lower()]
    sat_key = [key for key in taylor_dict.keys() if 'sat' in key.lower()]
    
    reference_monthly = taylor_dict[sat_key[0]]
    
    # Concatenate the data across all years for the given month
    monthly_sat_data = np.concatenate([
        reference_monthly[year][month_index] for year in reference_monthly
    ])

    # Compute standard deviation
    std_ref = np.nanstd(monthly_sat_data)
    
    sdev = [std_ref]
    crmsd = [0.0]  # Reference RMSD (set manually)
    ccoef = [1.0]  # Reference correlation coefficient (set manually)
    label = ["Ref"]  # Label for the reference data
    
    # Get the month name from the month index
    month_name = calendar.month_name[month_index + 1]  # +1 because months are 1-indexed in `calendar`
    
    for year in range(Tspan):  # Looping through years Ybeg to Yend
        # Get the model data for the current year and the ith month
        model_data = taylor_dict[model_key[0]][year + Ybeg][month_index]

        # Get the corresponding satellite data for the ith month
        ref_data = taylor_dict[sat_key[0]][year + Ybeg][month_index]
        
        valid_indices = ~np.isnan(model_data) & ~np.isnan(ref_data)
        
        filtered_model_data = model_data[valid_indices]
        filtered_ref_data = ref_data[valid_indices]

        # Compute the Taylor statistics for each year
        taylor_stats = sm.taylor_statistics(filtered_model_data, filtered_ref_data, 'data')
        
        # Extract the statistics and append them to the lists
        sdev.append(taylor_stats['sdev'][1])  # Model standard deviation
        crmsd.append(taylor_stats['crmsd'][1])  # Model RMSD
        ccoef.append(taylor_stats['ccoef'][1])  # Model correlation coefficient

        label.append(Ybeg + year)  # Append the year label

    # You can process or store the model_data and ref_data here
    print(f"The Taylor diagram for {month_name} has been plotted!")
    
    # Convert lists to numpy arrays for easier manipulation
    sdev = np.array(sdev)
    crmsd = np.array(crmsd)
    ccoef = np.array(ccoef)

    # Produce the Taylor diagram
    sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label, 
                      taylor_options_file=taylor_options_monthly)
    
    # Add title with the month name
    plt.title(f"Taylor Diagram for {month_name}", pad = 40)

    # Optionally, save or show the plot
    plt.savefig(f'C:/Tesi Magistrale/Codici/Python/Plot output/Taylor/CHL/L4/Taylor_diagram_month{month_index}.png')
    plt.show()
    plt.close()