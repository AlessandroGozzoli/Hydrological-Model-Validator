from sklearn.linear_model import HuberRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np

def fill_annular_region(ax, r_in, r_out, color, alpha=0.3):
    theta = np.linspace(0, 2 * np.pi, 500)
    x_outer = r_out * np.cos(theta)
    y_outer = r_out * np.sin(theta)
    x_inner = r_in * np.cos(theta[::-1])
    y_inner = r_in * np.sin(theta[::-1])
    x = np.concatenate([x_outer, x_inner])
    y = np.concatenate([y_outer, y_inner])
    ax.fill(x, y, color=color, alpha=alpha, zorder=0)
    
def get_min_max_for_identity_line(x, y):
    min_val = min(np.nanmin(x), np.nanmin(y))
    max_val = max(np.nanmax(x), np.nanmax(y))
    return min_val, max_val


def get_variable_label_unit(variable_name):
    if variable_name == 'SST':
        variable='Sea Surface Temperature'
        unit='[$°C$]'
        return variable, unit
    elif variable_name == 'CHL_L3':
        variable = 'Chlorophyll (Level 3)'
        unit = '[$mg/m^3$]'
        return variable, unit
    elif variable_name == 'CHL_L4':
        variable = 'Chlorophyll (Level 4)'
        unit = '[$mg/m^3$]'
        return variable, unit 
    return variable_name, ''

def fit_huber(mod_data, sat_data):
    """
    Fits a robust linear regression (Huber) between mod_data and sat_data.
    Returns x values and predicted y values for plotting the regression line.
    """
    model = HuberRegressor().fit(mod_data.reshape(-1, 1), sat_data)
    x_vals = np.linspace(mod_data.min(), mod_data.max(), 100)
    y_vals = model.predict(x_vals.reshape(-1, 1))
    return x_vals, y_vals

def fit_lowess(mod_data, sat_data, frac=0.3):
    """
    Computes LOWESS smoothing of sat_data vs mod_data.
    Returns the smoothed points as an array of shape (N, 2) for plotting.
    """
    sorted_idx = np.argsort(mod_data)
    smoothed = lowess(sat_data[sorted_idx], mod_data[sorted_idx], frac=frac)
    return smoothed

def get_season_mask(dates, season_name):
    if season_name == 'DJF':
        return (dates.month == 12) | (dates.month == 1) | (dates.month == 2)
    elif season_name == 'MAM':
        return (dates.month >= 3) & (dates.month <= 5)
    elif season_name == 'JJA':
        return (dates.month >= 6) & (dates.month <= 8)
    elif season_name == 'SON':
        return (dates.month >= 9) & (dates.month <= 11)
    else:
        raise ValueError(f"Invalid season name: {season_name}")
        
def gather_monthly_data_across_years(data_dict, key, month_idx):
    """
    Gathers, flattens, concatenates, and removes NaNs from all years' data
    for a specific key and month index.

    Parameters:
        data_dict (dict): Dictionary with {year: [12-month arrays]}
        key (str): Dictionary key for model or satellite
        month_idx (int): Month index (0 = January, 11 = December)

    Returns:
        np.ndarray: Cleaned, 1D array of all valid data for that month
    """
    values = [np.asarray(data_dict[key][year][month_idx]).flatten()
              for year in data_dict[key]]
    all_data = np.concatenate(values)
    return all_data[~np.isnan(all_data)]
