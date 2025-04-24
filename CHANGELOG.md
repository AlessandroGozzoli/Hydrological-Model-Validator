# **Version:** 2.7  
**Date:** 24/04/2025  

## Summary

Updated the SST analysing script, minor path bugs fixed

## Sea Surface Temperature data analysing script version 2.0

- SST analysing script has been updated to version 2.0, allowing for a better user dialogue through the prints and for a better handling of the plots.
- Added the dynamical path creation used in the previous scripts to create ad-hoc folders to save the plots.
- The plots are now displayed for only 3 seconds before closing, this can be changed by changing the value in the plotting functions.

## Other changes

- Removed Plot outputs folder to declutter, example outputs will be provided in a future README update
- Removed leap_year.py as all functions within have been moved to Corollary.py
- Merged scatterplots, timeseries plots and efficiency plots functions in a single script called Plots.py.

# Known issues

- Currently the RMSD in the Taylor diagrams is not displayed correctly as it is based upon the option .csv files provided in the folder, the aim is to make the ranges dynamic for a better display of the graphs.

--------------------------------------------------------------------------------------

# **Version:** 2.6  
**Date:** 23/04/2025  

## Summary

Introduced basin average computation and performed significant cleanup of old scripts.

## Data Setupper Complete

- The data reading and setup script is now fully complete, with the addition of functionality to compute the daily mean basin average time series for SST datasets (both satellite and model).
- The data saving script has been updated to support saving the new basin average data.
- The CHL dataset still requires processing through the interpolator to compute its basin average.

## Other Changes

- Conducted a major cleanup of the functions folder. All outdated functions have been removed, with the exception of the `leap_year` function, which remains in use for analysis scripts.

--------------------------------------------------------------------------------------

# **Version:** 2.5  
**Date:** 23/04/2025  

## Summary

Expanded data saving functionality to support model datasets.

## Data Saver Expanded

- The `Data_saver.py` script has been extended with new functions to handle saving model datasets.
- Due to the large size of the model data, it has been split into multiple files, with each file corresponding to a specific year.

## Other Changes

- Added assertions for enhanced code stability and error handling.
- Reorganized files and functions to improve code structure and maintainability.

--------------------------------------------------------------------------------------

# **Version:** 2.4  
**Date:** 23/04/2025  

## Summary

Introduced functions for reading model data and expanded the main script to support them.

## Model Data

- Added necessary functions for reading model data, now integrated into a new script.
- The main code has been updated to handle the newly added model data reading functionality.

## Other Changes

- Reorganized functions for improved clarity and maintainability.
- Added assertions to enhance code stability and error handling.

--------------------------------------------------------------------------------------

# **Version:** 2.3  
**Date:** 23/04/2025  

## Summary

Introduced the ability to import and apply a mask to align satellite and model data.

## Mask Handling

- The model data contains `NaN` values for landmasses, so a mask has been implemented to ensure proper alignment between satellite and model datasets.
- The mask is essential for the interpolation process, and the one used in the current test case will be provided at a later stage.

## Other Changes

- Reorganized functions for improved code structure.
- Created a corollary file to house additional functions that are not directly related to missing data handling or data reading.

--------------------------------------------------------------------------------------

# **Version:** 2.2  
**Date:** 23/04/2025  

## Summary

Introduced the `Data_saver.py` script to enhance data management and facilitate the transition to the interpolator.

## Data Saving

- Two new steps have been implemented to allow users to save manipulated satellite datasets for validation or later processing.
- Users can now save datasets in a dedicated folder in either `.mat` or `.nc` formats:
  - **.mat files** contain all necessary variables for interpolation.
  - **.nc files** offer flexibility for future expansion if needed.
- Each saved dataset is timestamped with the date of the run for tracking purposes.

## Other Changes

- Reorganized and moved functions to continue restructuring the codebase for improved modularity and maintainability.

--------------------------------------------------------------------------------------

# **Version:** 2.1  
**Date:** 23/04/2025  

## Summary

Introduced a dedicated script for reading satellite SST (Sea Surface Temperature) data.

## Code Structure

- Added functions for reading satellite SST data, now integrated into the main script.
- The overall structure remains streamlined, with new functionality focused on SST data handling.

## Other Changes

- Refined and reorganized functions for improved clarity and maintainability.


--------------------------------------------------------------------------------------

# **Version:** 2.0  
**Date:** 22/04/2025  

## Summary

Reworked the **Data_reader_setupper** script from the ground up, with the script now moved to the home directory.

## Code Structure

- Currently works only with the satellite CHL datasets.
- The structure of the functions remains unchanged, but they have been moved to a dedicated SAT data script.
- The user can now define which level of data to handle via an input line.

## Other Changes

- Added assertions to improve code stability.

--------------------------------------------------------------------------------------

# Author: Alessandro Gozzoli  
**Version:** 1.0  
**Date:** 20/04/2025  

## Summary

Initial implementation of core functionality up to the *Data Analysis and Efficiency Metrics* stage. The codebase is structured to handle both satellite and model data through two distinct scripts, with support for saving outputs in `.nc` and `.mat` formats.

## Data Handling

- **Satellite Data**
  - **CHL (Chlorophyll):** Loads data into a single array and checks for missing fields. Requires the data level to be specified via a variable in the code.
  - **SST (Sea Surface Temperature):** Reads data and converts temperature values to Celsius.

- **Model Data**
  - **CHL:** Retrieves and reads chlorophyll data from model outputs.
  - **SST:** Retrieves and reads sea surface temperature data; also computes basin-averaged SST values.

## Code Structure

- Core functions are distributed across multiple files.
- Interpolation is handled using a MATLAB script due to Python limitations in managing grids with adjacent cells of identical values.
  - This MATLAB script also computes the basin average for CHL data.

## Analysis

- SST and CHL analyses are performed in separate scripts to mitigate RAM limitations.
- Implemented analyses include:
  - Time series plots
  - Scatter plots
  - Taylor diagrams
  - Target diagrams
  - Multiple efficiency metrics (refer to Krause et al., 2005, as listed in the Bibliography)