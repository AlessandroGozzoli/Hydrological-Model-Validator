import numpy as np

# 1. Coefficient of Determination (r²)
def r_squared(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if np.sum(mask) < 2:
        return np.nan
    return np.corrcoef(obs[mask], pred[mask])[0, 1] ** 2

# 1.5 Monthly r²
def monthly_r_squared(dictionary):
    from .Efficiency_metrics import r_squared
    
    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]
    
    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]

    r2_monthly = []

    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs pairwise
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data)
            mod_vals.extend(mod_data[mask])
            sat_vals.extend(sat_data[mask])

        r2 = r_squared(np.array(sat_vals), np.array(mod_vals))
        r2_monthly.append(r2)

    return r2_monthly

# 2. Weighted Coefficient of Determination (wr²)
def weighted_r_squared(obs, pred):
    r2 = r_squared(obs, pred)
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if np.sum(mask) < 2:
        return np.nan
    slope, _ = np.polyfit(obs[mask], pred[mask], 1)
    if slope <= 1:
        return abs(slope) * r2
    else:
        return abs(1 / slope) * r2
    
# 2.5 Monthly wr²
def monthly_weighted_r_squared(dictionary):
    from .Efficiency_metrics import weighted_r_squared
    
    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]
    
    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]
    
    wr2_monthly = []
    
    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs pairwise
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data)
            mod_vals.extend(mod_data[mask])
            sat_vals.extend(sat_data[mask])

        wr2 = weighted_r_squared(np.array(sat_vals), np.array(mod_vals))
        wr2_monthly.append(wr2)

    return wr2_monthly

# 3. Nash–Sutcliffe Efficiency (NSE)
def nse(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    return 1 - np.sum((obs[mask] - pred[mask]) ** 2) / np.sum((obs[mask] - np.mean(obs[mask])) ** 2)

# 3.5 Monthly-NSE
def monthly_nse(dictionary):
    from .Efficiency_metrics import nse

    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]

    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]

    nse_monthly = []

    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs pairwise
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data)
            if np.any(mask):
                mod_vals.extend(mod_data[mask])
                sat_vals.extend(sat_data[mask])

        nse_val = nse(np.array(sat_vals), np.array(mod_vals))
        nse_monthly.append(nse_val)

    return nse_monthly

# 4. Index of Agreement (d)
def index_of_agreement(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    return 1 - np.sum((obs[mask] - pred[mask]) ** 2) / np.sum((np.abs(pred[mask] - np.mean(obs[mask])) + np.abs(obs[mask] - np.mean(obs[mask]))) ** 2)

# 4.5 Monthly Index of Agreement
def monthly_index_of_agreement(dictionary):
    from .Efficiency_metrics import index_of_agreement

    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]

    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]

    d_monthly = []

    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs pairwise
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data)
            if np.any(mask):
                mod_vals.extend(mod_data[mask])
                sat_vals.extend(sat_data[mask])

        d = index_of_agreement(np.array(sat_vals), np.array(mod_vals))
        d_monthly.append(d)

    return d_monthly

# 5. NSE with Logarithmic Values (ln NSE)
# !!! Make sure that the datasets only have positive numbers !!!
def ln_nse(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred) & (obs > 0) & (pred > 0)
    return 1 - np.sum((np.log(obs[mask]) - np.log(pred[mask])) ** 2) / np.sum((np.log(obs[mask]) - np.mean(np.log(obs[mask]))) ** 2)

# 5.5 monthly log-NSE
def monthly_ln_nse(dictionary):
    from .Efficiency_metrics import ln_nse

    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]

    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]

    ln_nse_monthly = []

    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs and non-positive values (for log)
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data) & (mod_data > 0) & (sat_data > 0)
            if np.any(mask):
                mod_vals.extend(mod_data[mask])
                sat_vals.extend(sat_data[mask])

        ln_nse_val = ln_nse(np.array(sat_vals), np.array(mod_vals))
        ln_nse_monthly.append(ln_nse_val)

    return ln_nse_monthly


# 6. Modified NSE (E₁) - j=1
def nse_j(obs, pred, j=1):
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    return 1 - np.sum(np.abs(obs[mask] - pred[mask]) ** j) / np.sum(np.abs(obs[mask] - np.mean(obs[mask])) ** j)

# 6.5 Monthly modified NSE
def monthly_nse_j(dictionary, j=1):
    from .Efficiency_metrics import nse_j

    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]

    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]

    nse_j_monthly = []

    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs pairwise
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data)
            if np.any(mask):
                mod_vals.extend(mod_data[mask])
                sat_vals.extend(sat_data[mask])

        e1 = nse_j(np.array(sat_vals), np.array(mod_vals), j=j)
        nse_j_monthly.append(e1)

    return nse_j_monthly

# 7. Modified Index of Agreement (d₁) - j=1
def index_of_agreement_j(obs, pred, j=1):
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    return 1 - np.sum(np.abs(obs[mask] - pred[mask]) ** j) / np.sum((np.abs(pred[mask] - np.mean(obs[mask])) + np.abs(obs[mask] - np.mean(obs[mask]))) ** j)

# 7.5 Monthly modified index of aggreement
def monthly_index_of_agreement_j(dictionary, j=1):
    from .Efficiency_metrics import index_of_agreement_j

    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]

    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]

    d1_monthly = []

    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs pairwise
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data)
            if np.any(mask):
                mod_vals.extend(mod_data[mask])
                sat_vals.extend(sat_data[mask])

        d1 = index_of_agreement_j(np.array(sat_vals), np.array(mod_vals), j=j)
        d1_monthly.append(d1)

    return d1_monthly

# 8. Relative NSE (E_rel)
def relative_nse(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred) & (obs != 0)
    return 1 - np.sum(((obs[mask] - pred[mask]) / obs[mask]) ** 2) / np.sum(((obs[mask] - np.mean(obs[mask])) / np.mean(obs[mask])) ** 2)

# 8.5 Monthly relative NSE
def monthly_relative_nse(dictionary):
    from .Efficiency_metrics import relative_nse

    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]

    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]

    e_rel_monthly = []

    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs and ensure obs != 0
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data) & (sat_data != 0)
            if np.any(mask):
                mod_vals.extend(mod_data[mask])
                sat_vals.extend(sat_data[mask])

        e_rel = relative_nse(np.array(sat_vals), np.array(mod_vals))
        e_rel_monthly.append(e_rel)

    return e_rel_monthly

# 9. Relative Index of Agreement (d_rel)
def relative_index_of_agreement(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    mask = ~np.isnan(obs) & ~np.isnan(pred) & (obs != 0)
    numerator = np.sum(((obs[mask] - pred[mask]) / obs[mask]) ** 2)
    denominator = np.sum(((np.abs(pred[mask] - np.mean(obs[mask])) + np.abs(obs[mask] - np.mean(obs[mask]))) / np.mean(obs[mask])) ** 2)
    return 1 - numerator / denominator

# Monthly realtive index of aggreement
def monthly_relative_index_of_agreement(dictionary):
    from .Efficiency_metrics import relative_index_of_agreement

    model_key = [key for key in dictionary.keys() if 'mod' in key.lower()]
    sat_key = [key for key in dictionary.keys() if 'sat' in key.lower()]

    if not model_key or not sat_key:
        raise KeyError("Model or satellite key not found in the dictionary.")

    mod_monthly = dictionary[model_key[0]]
    sat_monthly = dictionary[sat_key[0]]

    d_rel_monthly = []

    for month in range(12):
        mod_vals = []
        sat_vals = []

        for year in mod_monthly:
            mod_data = mod_monthly[year][month]
            sat_data = sat_monthly[year][month]

            # Remove NaNs and ensure obs != 0
            mask = ~np.isnan(mod_data) & ~np.isnan(sat_data) & (sat_data != 0)
            if np.any(mask):
                mod_vals.extend(mod_data[mask])
                sat_vals.extend(sat_data[mask])

        d_rel = relative_index_of_agreement(np.array(sat_vals), np.array(mod_vals))
        d_rel_monthly.append(d_rel)

    return d_rel_monthly
