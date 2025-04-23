import numpy as np

from Costants import days_in_months_non_leap, days_in_months_leap, ysec

# Function to check if a year is a leap year
def leapyear(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 1
    return 0

# Main function to calculate the true time series length in days
def true_time_series_length(nf, chlfstart, chlfend, DinY):
    Truedays = 0  # Initialize the total number of days
    fdays = [0] * nf  # List to store the number of days for each file
    nspan = [0] * nf  # List to store the span of years for each file
    
    for n in range(nf):
        # Define the time span (in years) for each file
        nspan[n] = chlfend[n] - chlfstart[n] + 1
        fdays[n] = 0
        
        # Define the "true" number of days in each file
        for y in range(chlfstart[n], chlfend[n] + 1):
            # If year "y" is a leap year, one day is added
            fdays[n] += DinY + leapyear(y)
        
        Truedays += fdays[n]
    
    return Truedays

# Function to convert yearly data to monthly data
def convert_to_monthly_data(yearly_data):

    # Initialize an empty dictionary to hold the monthly data
    monthly_data_dict = {}

    # Loop over each year's data
    for i, year_data in enumerate(yearly_data):
        year = ysec[i]
        
        # Determine the number of days in the year
        if leapyear(year):
            days_in_months = days_in_months_leap
            expected_days = 366
        else:
            days_in_months = days_in_months_non_leap
            expected_days = 365
        
        # Pad shorter years with NaNs if necessary
        if len(year_data) < expected_days:
            year_data = np.pad(year_data, (0, expected_days - len(year_data)), constant_values=np.nan)
        
        # Initialize a list to store the months for this year
        year_months = []
        start_idx = 0  # Start at the beginning of the year
        
        # For each month, slice the data
        for month_days in days_in_months:
            end_idx = start_idx + month_days
            month_data = year_data[start_idx:end_idx]
            year_months.append(month_data)
            start_idx = end_idx  # Move to the next month
        
        # Save the monthly data for the current year in the dictionary
        monthly_data_dict[year] = year_months

    # Return the dictionary containing monthly data for each year
    return monthly_data_dict