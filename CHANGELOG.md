# **Version:** 2.10  
**Date:** 24/04/2025  

## Summary

Initial version (v1.0) of the **Benthic Geochemical Analysis** script completed.

## Benthic Geochemical Analysis Script

This first iteration of the **Benthic Layer Analysis** script introduces foundational functionality for exploring geochemical dynamics at the sediment-water interface using output from the **BFM-NEMO coupled model**.

### Features:
- Computes the deepest active layer (out of 48 vertical layers) in each model grid cell across the domain.
- Enables visualization of the model basin bathymetry and deepest layer distribution.
- Allows users to select a chemical species from the simulation for spatial plotting.
- Generates **georeferenced 2D contour maps** of selected species at the benthic interface, enriched with coastlines.
- Default contour resolution is 51 levels (configurable in code) for the geochemical species, Benthic Depth plot uses 26.

## Future Development

This script represents the first half of the full analysis pipeline. Future updates will introduce:
- Computation and visualization of the **pressure field** within the water column.
- Diagnostic tools for investigating **deep water formation processes** in the Northern Adriatic Sea.

## Known Issues

- **Taylor Diagrams** still use static RMSD ranges — dynamic scaling is planned.
- **Taylor and Target plots** continue to depend on pre-defined `.csv` configuration files.
- **Chlorophyll regression analysis** occasionally produces anomalous values — further investigation is underway.
- **LaTeX rendering** in colorbar labels may break under certain conditions.
- While **dynamic colorbar scaling** may improve usability, the current fixed scaling highlights extremes effectively; additional testing is ongoing.

--------------------------------------------------------------------------------------

# **Version:** 2.9  
**Date:** 24/04/2025  

## Summary

Introduced seasonal scatterplots for both Sea Surface Temperature (SST) and Chlorophyll (CHL) datasets to support more detailed seasonal analysis.

## Seasonal Scatterplots

- Based on the insights from previous scatterplot analyses, new plots have been developed to break down basin-averaged values by season.
- The data is first decomposed into seasonal subsets and visualized in individual season-specific scatterplots.
- A combined scatterplot is also generated, consolidating all seasonal data and color-coding points according to their respective seasons.
- Each plot includes:
  - A best-fit line
  - A **Huber regression line** for robust linear fitting (less sensitive to outliers)
  - A **LOWESS (Locally Weighted Scatterplot Smoothing)** non-linear regression line to highlight trends in densely clustered areas.

## Known Issues

- The Taylor diagrams still use a fixed RMSD range; dynamic scaling is planned for a future update.
- Taylor and Target plots continue to rely on static `.csv` files, which limits flexibility.
- Some anomalous values have been observed in the CHL regression fits; further investigation is required to determine whether these are data artifacts or bugs.

--------------------------------------------------------------------------------------

# **Version:** 2.8  
**Date:** 24/04/2025  

## Summary

Updated the CHL analysis script and performed minor cleanup in the SST script.

## Chlorophyll Analysis Script Updated to Version 2.0

- The CHL analysis script has been updated to align with the improvements made in the SST script.
- Comments have been revised to more clearly identify level 3 and level 4 analysis sections.
- Plots are now displayed for 3 seconds before closing and are saved in dynamically created folders.

## Other Changes

- Added a print statement to the SST analysis script to inform the user when the BIAS has been computed.

## Known Issues

- Similar to the SST plots, the CHL plots have a fixed range for the RMSD in the Taylor diagrams. Future updates aim to make the RMSD range dynamic for improved graph display.

--------------------------------------------------------------------------------------

# **Version:** 2.7  
**Date:** 24/04/2025  

## Summary

Updated the SST analysis script and fixed minor path-related issues.

## Sea Surface Temperature Data Analysis Script Version 2.0

- The SST analysis script has been updated to version 2.0, enhancing user interaction through improved print statements and more effective plot handling.
- Implemented dynamic path creation, enabling the automatic creation of ad-hoc folders to save plots, as seen in previous scripts.
- Plots are now displayed for 3 seconds before automatically closing; this duration can be adjusted by modifying the value in the plotting functions.

## Other Changes

- Removed the `Plot outputs` folder to declutter the project structure. Example outputs will be provided in a future update to the README.
- Removed `leap_year.py`, as its functions have been migrated to `Corollary.py`.
- Merged scatter plots, time series plots, and efficiency plots functions into a single script, `Plots.py`.

## Known Issues

- The RMSD (Root Mean Square Deviation) in the Taylor diagrams is not currently displayed correctly. This issue stems from the `.csv` files provided in the folder. Future updates will aim to make the ranges dynamic for improved graph display.

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