# **Version:** 3.0.1  
**Date:** 18/05/2025  

## Summary

This is a minor update focused on expanding DataFrame usability within the SST and CHL analysis scripts and improving dataset loading performance.

---

## Expanded Use of Pandas DataFrames

- SST and CHL analysis scripts now fully leverage **`pandas` DataFrames**, enabling:
  - Seamless integration of the `datetime` dimension.
  - More efficient time-based slicing into **monthly** and **yearly** datasets using native `pandas` methods.
- Enhances clarity and performance for long-term and seasonal trend analysis.

---

## Faster Dataset Loading

- Introduced **parallel loading** of SST datasets using `ThreadPoolExecutor` from Python’s [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) module.
- Significantly improves script runtime when dealing with large temporal datasets.

--------------------------------------------------------------------------------------

# **Version:** 3.0  
**Date:** 18/05/2025  

## Summary

This version introduces a major rework and optimization of the plotting functions used for model validation and comparison. It focuses on improving clarity, maintainability, and performance in both visual output and computational workflow.

## New Taylor Diagrams and Target Plots

- **Normalization**: All monthly validation parameters are now normalized by their respective standard deviations, allowing for the unified display of all markers in a single diagram.
- **Marker Logic Update**: Marker representations have been reworked based on a consistent logic [insert table when available].
- **Enhanced Visualization**:
  - **Taylor Diagrams**: Now include RMSD arcs and repositioned RMSD labels outside the plot area to avoid marker overlap.
  - **Target Plots**: Include color-coded performance zones to quickly assess model accuracy and bias.

## Violin Plots

- Introduced **violin plots** as an alternative to **whisker-box plots**.
- Violin plots offer a smoother visual of data distribution but are less informative regarding outliers.
- This plot type is included for completeness and comparative analysis.

## Seaborn Integration

- Most plotting functions now utilize the [**Seaborn**](https://seaborn.pydata.org/) library.
- Advantages include:
  - Better integration with `pandas` DataFrames.
  - More expressive and customizable visualizations.
  - Improved consistency across plots.

## Separation of Computations

A significant refactor has begun to modularize core functionality:

- Extracted key routines from plotting scripts into a new **Auxiliary script**:
  - **Label formatting** (e.g., variable names, units).
  - **Key identification** from datasets.
  - **Seasonal masks** and data groupings.
  - **Statistical calculations** required for Taylor and Target diagrams.
  - **Regression line generation** (Huber, LOWESS, etc.).

This modularization paves the way for cleaner, more testable code in preparation for the final **pytest** integration.

## Future Direction

- Further optimize plotting routines for speed and clarity.
- Begin reworking data loading and interpolation functions for faster runtime.
- Explore full Python replacement of the current MATLAB `Interpolato.m` script.

## Known Issues

- **RMSD Label Placement**: Labels are currently tied to a fixed first arc value. Further investigation into the `SkillMetrics` library is ongoing to determine how arc ranges are defined and whether label placement can be dynamically bound to them.
- **Static RMSD Arc Ticks**: Taylor diagrams use the same arc ticks across plots. While this helps with consistency in test cases, dynamic adjustment would improve generality. Removing the `tickrms` override may solve this, but could also interfere with label alignment (see above).
- **Unexpected Target Plot Results**: Initial performance scores from Target plots appear lower than anticipated. Ongoing testing will determine if this is a bug, data artifact, or an accurate model assessment.
- **Chlorophyll regression analysis** occasionally produces anomalous values — further investigation is underway.

--------------------------------------------------------------------------------------

# **Version:** 2.11  
**Date:** 14/05/2025  

## Summary

Whisker-box plots have been implemented for satellite Basin Average SST and CHL datasets.

## Whisker Plots

A new visualization tool — the **whisker-box plot** — has been added to both the SST and CHL analysis scripts. These plots provide a clearer view of statistical distributions, highlighting mean values and outliers in the Basin Average datasets. Their primary purpose is to enhance model performance evaluation by offering a more nuanced look at dataset variability.

## Future Developments

With the implementation of this feature, the **2.x** development cycle is considered **complete**.

The next major update will initiate **Version 3.0**, which will focus on:
- **Refactoring all functions** to improve computational efficiency and streamline logic.
- **Introducing new libraries**, such as [**Seaborn**](https://seaborn.pydata.org/), for more advanced and elegant plotting.
- **Ensuring result consistency**, with side-by-side testing to confirm output reliability compared to previous versions.
- **Resolving known issues and bugs** from earlier versions.
- **Expanding documentation**, including:
  - Clearer comments and structure within the codebase.
  - A **step-by-step guide** for running the test case.
  - A **pytest** module to automate testing of computational functions.

Version 3.0 will mark a shift toward a more maintainable, scalable, and user-friendly project structure.

## Known Issues

- **Taylor Diagrams** still use static RMSD ranges — dynamic scaling is planned.
- **Taylor and Target plots** continue to depend on pre-defined `.csv` configuration files.
- **Chlorophyll regression analysis** occasionally produces anomalous values — further investigation is underway.
- **LaTeX rendering** in colorbar labels may break under certain conditions.
- While **dynamic colorbar scaling** may improve usability, the current fixed scaling highlights extremes effectively; additional testing is ongoing.

--------------------------------------------------------------------------------------

# **Version:** 2.10  
**Date:** 13/05/2025  

## Summary

The **Benthic Layer Analysis** script has been expanded to **version 2.0** with significant enhancements to data extraction, analysis, and visualization.

This update builds upon the initial version, adding functionality for the extraction and plotting of **temperature** and **salinity** data at the benthic layer. Additionally, the script now computes and visualizes the **density field** using three distinct **equations of state**, providing more accurate insights into deep water formation. As a result, the pressure field will no longer be included in the project, as the density field is deemed a more reliable representation of the evolution of dense water formation.

### Key Enhancements:

- **Temperature and Salinity Maps:**  
  The temperature and salinity values at the benthic layer are now extracted using a method similar to that employed for biogeochemical fields. These values are georeferenced and plotted using the same function as used for the biogeochemical species, ensuring consistent map generation.

- **Density Computation & Plotting:**  
  Temperature and salinity data are now processed to compute the **density field**, using the following three equations of state:
  - Simplified equation of state
  - Equation of State for Seawater (1980)
  - Thermodynamic Equation of State (2010)

  All three density fields are plotted using a fixed color range to allow for easy comparison of the differences between the equations of state.

Paper are provided in the **Bibliography** section of the README for the user to read to better understand differences in these 3 different **Equations of State**.

### Visual Enhancements:

- Plots now feature **fixed color ranges**, making it easier to identify and interpret the phenomena illustrated by the maps and plots.

## Future Developments:

- Ongoing improvements to the density computation.
- Additional enhancements to data visualization and analytical functions.

## Known Issues

- **Taylor Diagrams** still use static RMSD ranges — dynamic scaling is planned.
- **Taylor and Target plots** continue to depend on pre-defined `.csv` configuration files.
- **Chlorophyll regression analysis** occasionally produces anomalous values — further investigation is underway.
- **LaTeX rendering** in colorbar labels may break under certain conditions.
- While **dynamic colorbar scaling** may improve usability, the current fixed scaling highlights extremes effectively; additional testing is ongoing.

--------------------------------------------------------------------------------------

# **Version:** 2.10  
**Date:** 10/05/2025  

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
**Date:** 06/05/2025  

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