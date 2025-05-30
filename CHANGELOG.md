# **Version:** 4.2.7
**Date:** 30/05/2025

## Summary

This release includes a refactoring of submodules functions to enhance code clarity, stability, and maintainability, alongside minor performance optimizations.

## General Enhancements

- **Refactoring – `stats_math_utils.py`:**  
  Cleaned and reorganized function implementations for improved stability and readability.  
  Variable names have been updated to remove assumptions tied to specific data sources (e.g., model vs. satellite).  
  Comments and docstrings have been revised to improve clarity and documentation quality.

- **Optimization – `compute_coverage_stats`:**  
  Slight improvements to input handling for better robustness and reduced likelihood of runtime errors.

- **Cleanup – `time_utils.py`:**  
  Function declarations have been cleaned and input validation messages rewritten for clarity.  
  Optimizations were applied to reduce nested loops and improve performance.

- **Improvements – `utils.py`:**  
  Enhanced error messages and streamlined function parameters.  
  Refactored internal logic to eliminate unnecessary nested loops and improve efficiency.

--------------------------------------------------------------------------------------

# **Version:** 4.2.6
**Date:** 30/05/2025

## Summary

This release includes cleanups and performance improvements for data reading modules, enhancing readability, reliability, and execution speed.

## General Enhancements

- **Refactoring – `MOD_data_reader.py` and `SAT_data_reader.py`:**  
  Functions in these submodules have been refactored to improve code clarity and reduce computational overhead.

- **Input Validation Improvements:**  
  Replaced outdated `assert` statements with explicit `RaiseErrors` for more robust input validation and clearer error handling.

--------------------------------------------------------------------------------------

# **Version:** 4.2.5
**Date:** 30/05/2025

## Summary

This release provides cleanup and performance improvements in the `Efficiency_metrics.py` submodule, as well as bug fixes affecting test cases in `Benthic_layer.py`.

## General Enhancements

- **Refactoring – `Efficiency_metrics.py`:**  
  Functions within this submodule have been cleaned up for improved readability and maintainability.  
  Optimization efforts targeted reducing computational time, particularly by minimizing nested loops.

## Bug Fixes

- **`Benthic_physical_plot` Bug:**  
  Fixed an issue where an incorrect number of variables was extracted during the execution of the `get_benthic_plot_parameters` helper function.

- **`compute_dense_water_volume` Bug:**  
  Resolved a problem where the `valid_mask` parameter was not being passed correctly to the `calc_density` helper function.

--------------------------------------------------------------------------------------

# **Version:** 4.2.4
**Date:** 29/05/2025

## Summary

This release focuses on improved documentation and minor optimizations within the `file_io.py` module.

## General Enhancements

- **Expanded Docstrings:**  
  Function docstrings within `file_io.py` have been updated to provide clearer, more exhaustive information for end users and developers.

- **`mask_reader` Optimization:**  
  Refactored to use `slice` instead of `squeeze` for removing extra dimensions, ensuring safer array handling.

- **`call_interpolator` Cleanup:**  
  General code cleanup and minor optimizations to improve readability and maintainability.

--------------------------------------------------------------------------------------

# **Version:** 4.2.3 
**Date:** 29/05/2025

## Summary

This update refactors the `Density.py` submodule to enhance performance, maintainability, and numerical stability in density-related computations.

## Enhancements

- **Code Cleanup and Typing Improvements:**  
  Functions within the `Density.py` submodule have undergone general cleanup. Input types are now explicitly defined using `typing` annotations, and docstrings have been expanded for improved clarity and documentation.

- **NaN Value Handling:**  
  A general masking step has been added to the `calc_density` function to filter out `NaN` values during computation, increasing reliability.

## Dense Water Mass Computation

- **Function Rework – `compute_dense_water_volume`:**  
  The function has been restructured to utilize existing utility functions for reading `.gz` datasets, promoting code reuse and modularity.

- **Loop Optimization:**  
  Deprecated nested `for` loops have been replaced with efficient `NumPy` operations such as `where` and `broadcast`, significantly improving performance.

> **Note:**  
> Since the recent changes primarily involve refactoring, minor fixes, and performance optimizations, with no major new features, the versioning has been adjusted to reflect these as patch-level updates. The old versions 4.3.0 and 4.4.0 have been renamed to 4.2.1 and 4.2.2.

--------------------------------------------------------------------------------------

# **Version:** 4.2.2  
**Date:** 29/05/2025

## Summary

This patch includes refactoring of the `Data_saver.py` submodule to enhance performance, stability, and input flexibility.

## Enhancements

- **Refactored Data Saving Functions:**  
  Functions in the `Data_saver.py` submodule have been optimized for improved path handling. File paths can now be provided as either `Path` objects or `str`, offering greater flexibility.

- **Improved Error Handling:**  
  Replaced legacy `assert` statements with explicit `RaiseErrors` to provide clearer and more reliable error reporting.

--------------------------------------------------------------------------------------

# **Version:** 4.2.1 
**Date:** 29/05/2025

## Summary

This update focuses on performance and reliability improvements within the `data_alignment.py` submodule.

## Enhancements

- **Refactored `apply_3d_mask` Function:**  
  The `apply_3d_mask` function has been reworked to simplify broadcasting across datasets, improving clarity and maintainability.

- **Improved Input Validation:**  
  Added `RaiseErrors` to enforce correct input types and shapes, ensuring more robust error handling and early failure detection.

--------------------------------------------------------------------------------------

# **Version:** 4.2.0  
**Date:** 29/05/2025

## Summary

This release introduces a new computation feature: the percentage of cloud coverage and the percentage of available data within a basin. These enhancements lay the groundwork for more detailed analyses in future updates.

## New Features

- **Coverage Statistics Calculation:**  
  A new function, `compute_coverage_stats`, has been added to the `stats_math_utils.py` submodule. It calculates:
  - Percentage of available data in a basin  
  - Percentage of cloud coverage  

  These metrics are intended for future visualization alongside time series data plotted using the `timeseries` function.

## Fixes & Improvements

- **Dependency Cleanup:**  
  Removed deprecated libraries and modules from the `SST_data_analyser.py` test script.

- **Bug Fix – Dataset Selection:**  
  Resolved an issue in the `read_model_data` function that caused incorrect dataset selection under certain conditions.

--------------------------------------------------------------------------------------

# **Version:** 4.1.0  
**Date:** 28/05/2025

## Summary

Version 4.1.0 introduces a full refactor of the `Data_reader_setupper.py` test case, integrating the `Interpolator_v2.m` MATLAB script and consolidating data reading and saving functions for both satellite and model datasets. This release also marks the formal deprecation of `L4` data handling, shifting all analysis to `L3s` datasets.

## Major Changes

### `Data_reader_setupper.py`

- Complete refactor of the test case script
- Integrated `Interpolator_v2.m` using `matlab.engine`
- Unified data reading and saving logic for satellite and model datasets
- All previously used reader functions are now **deprecated**
- File management is critical: avoid moving files while the interpolator is running

### Unified Data Reading Functions

#### `MOD_data_reader.py` & `SAT_data_reader.py`

- Each script now contains a **single function** for reading model or satellite datasets
- Handles both `chl` and `sst` variables via the new `varname` argument
- Future work will focus on improved robustness for irregular key names (e.g., `adjusted_sea_surface_temperature` in CMEMS)

### `Missing_data.py`

- Rewritten to eliminate dependency on external constants
- Now runs autonomously and prepares for future optimizations

### `Data_saver.py`

- Merged functions into dedicated `save_model_data()` and `save_satellite_data()` routines
- Simplifies saving of processed and interpolated data

### `Interpolator_v2.m` Integration

- Executed from Python using `matlab.engine`
- New helper: `call_interpolator()`
- File structure and paths must remain unchanged during execution
- `setup.py` and `MANIFEST.in` updated to include required MATLAB files during installation

## Deprecations

- **L4 Data**: Now officially deprecated. Support removed from both `CHL_data_analyzer.py` and `SST_data_analyzer.py`
- **Legacy Reader Functions**: Replaced by new unified readers

## Minor Fixes & Adjustments

- Adjusted file names and dictionary keys in:
  - `CHL_data_analyzer.py`
  - `SST_data_analyzer.py`
- Minor typo corrections in `Benthic_layer.py`
- Fixed label positioning in `Taylor_diagrams.py` monthly plot function

## Future Work

- Further optimization and cleanup of `Data_reader_setupper.py` functions
- General cleanup of unused files in the GitHub repository
- Begin development of **spatial performance analysis** modules

--------------------------------------------------------------------------------------

# **Version:** 4.0.0 - OFFICIAL RELEASE  
**Date:** 25/05/2025

## Summary

This marks the official release of version 4.0.0. The `Benthic_layer.py` test case and all related `BFM` function scripts have been fully overhauled and are now operational. Additionally, `setup.py` has been updated to include all necessary dependencies for full functionality.

## `Benthic_layer.py` Functionality

While the high-level purpose of the `Benthic_layer.py` script remains the same, its underlying functions have been entirely restructured. All previous versions are now deprecated.

## Modular `BFM` Function Scripts

All functions related to BFM simulation variable handling are now organized in a dedicated module folder. This improves clarity, reuse, and integration with the rest of the project. Below is a breakdown of the updated logic and implementations.

### Geolocalization

- Introduced `geo_coords` function to convert raw Cartesian coordinates into geodetic coordinates (longitude, latitude) and Eulerian angles (φ and λ).
- Uses input horizontal resolution in degrees.

### Bottom Layer Computation

- New functions: `compute_Bmost()` and `compute_Bleast()` to extract benthic and surface layers, respectively.
- Visualization enhancements include:
  - `deep` colormap from `cmocean`
  - Optional 3D rendering using Plotly (`surface` and `3dmesh`), paving the way for advanced deep water mass analysis.

### Temperature and Salinity

- Functions now support parallel loading for improved performance.
- Datasets are cached for reuse in subsequent density calculations.
- Plotting options externalized for customization.
- Colormaps:
  - Temperature: `cmocean.thermal`
  - Salinity: `cmocean.haline`

### Density

- Computation logic extracted into a dedicated function.
- EOS-80 remains the default method, aligning with simulation standards.
- Optional methods retained for future flexibility.
- Plotting colormap: `cmocean.dense`

### Dense Water Masses

- New functionality added for detection and visualization of dense water masses (threshold: 1029.2 kg/m³, per Oddo et al.).
- Enhancements:
  - 2D maps with black contour overlays
  - 3D volume estimation via cell counting (800×800×2 m³)
  - Timeline plotting of deep water formation across all methods

### Chemical Species

- Data loading and plotting routines are now fully separated.
- Parallel loading avoided due to memory limitations of large `.nc` files.
- All plots use logarithmic color scaling with `cmocean` and `matplotlib` colormaps.
- Enhanced colorbars and axis labeling for clarity.

#### Oxygen

- Uses scalar steps and `cmocean.oxy` colormap.
- Thresholds:
  - Hypoxia: < 62.5 mmol/m³
  - Hyperoxia: > 312.5 mmol/m³
- Thresholds adjustable via the `formatting.py` module.

#### Chlorophyll-a

- Logarithmic steps
- Colormap: `viridis`

#### N-family (Nutrients)

- Logarithmic steps
- Colormap: `YlGnBu`

#### P-family (Primary Producers)

- Logarithmic steps
- Colormap: `cmocean.algae`

#### Z-family (Secondary Producers)

- Logarithmic steps
- Colormap: `cmocean.turbid`

#### R-family (Particulate Organic Matter)

- Logarithmic steps
- Colormap: `cmocean.matter`

## Future Work

The refactor of the `Benthic_layer.py` suite is now complete. Planned next steps include:

- **Integration of L3S Sea Surface Temperature data**, replacing L4 due to underperformance.
- Enhancements to the SST reading functions, with improved handling of missing or invalid data.
- General performance optimization and code cleanup across all test case scripts.

--------------------------------------------------------------------------------------

# **Version:** 4.0.0-δ - UNSTABLE  
**Date:** 23/05/2025

## Summary

This update brings the `Data_reader_setupper.py` test case script back online, resolving previous issues related to function imports and path handling.

## `Data_reader_setupper.py` Reactivation

- The script `Data_reader_setupper.py` is now functional again.
- Legacy hardcoded paths have been replaced with dynamic internal paths, removing the need for manual path extensions via `sys.path.append`.
- This change enhances modularity and reduces the risk of import errors during execution or testing.

> **Note:**  
> While the script is operational, **optimization of the functions** used within it has been **deferred to a future update**. The focus will shift to performance improvements once the `L3s` Sea Surface Temperature data integration is underway.

--------------------------------------------------------------------------------------

# **Version:** 4.0.0-γ - UNSTABLE  
**Date:** 23/05/2025

## Summary

This incremental update completes the rework of plotting functions used in the `SST` and `CHL` analysis workflows by updating the `Taylor Diagrams` and `Target Plots` plotting and computational scripts following the same conventions introduced in the previous release.

## Function Headers and Documentation

- Both `Taylor` and `Target` plotting functions now include detailed headers and inline comments designed to enhance code readability and provide clear guidance on usage.
- These headers document function purpose, input arguments, return values, and expected keyword arguments.

## Computational Function Improvements

- The underlying computation functions supporting these plots have been refined to:
  - Employ `itertools` for efficient iteration where applicable.
  - Replace all `assert` statements with explicit `RaiseErrors` to ensure robustness, even in optimized Python execution modes.
  - Include standardized function headers; comprehensive inline comments will be introduced in a subsequent update.

## Future Work

- The refactoring of analysis test case functions will be temporarily paused, with plans to revisit optimization efforts at a later stage.
- Next steps include fixing the `Data_reader_setupper.py` test case script to align with updated function paths; however, function re-optimization will wait until integration of the `Sea Surface Temperature` `L3s` data is complete.
- Subsequently, the `Benthic_layer` scripts will be corrected and overhauled to improve computational performance and extend functionality, including the planned calculation of deep water formation volumes.

--------------------------------------------------------------------------------------

# **Version:** 4.0.0-β - UNSTABLE  
**Date:** 22/05/2025

## Summary

This beta release continues the structural and functional overhaul of the project. Key updates include replacing `assert` statements with explicit `RaiseErrors` for robustness, the full integration of default plotting options, and improved documentation through consistent function headers. Additionally, the `SST` and `CHL` analysis test cases are now operational again following updates to the internal function paths.

> This version is still **UNSTABLE**. While core scripts for SST and CHL analysis are functioning, most other scripts remain incompatible due to unrefactored paths and legacy syntax. Use is advised only for testing specific updated modules.

## Major Changes

### `RaiseErrors` Replace `assert` Statements

- All validation previously handled through `assert` statements has been replaced with `raise ValueError(...)` or appropriate exceptions.
- This ensures checks remain active even when scripts are executed in optimized (`-O`) mode, increasing the robustness and reliability of the library at the cost of some runtime performance.

### Default Plotting Options Refactored

- Plotting functions used in `SST_data_analyzer.py` and `CHL_data_analyzer.py` now fully rely on centralized **default options**.
- Legacy hardcoded options have been moved to a dedicated defaults file, allowing users to override or extend behavior more flexibly.
- The default `dpi` remains set at **300** to maintain publication-quality output, but this will be lowered in the final release for faster rendering.

### Function Headers and Documentation

- All plotting functions (excluding Taylor and Target plots) and newly added scripts now include comprehensive headers:
  - Function purpose
  - Expected inputs and return types
  - Supported keyword arguments (`kwargs`)
  - Example usage
- This marks the beginning of a broader documentation effort to improve code clarity and onboarding for new contributors.

### Reactivated Test Cases

- Both `SST` and `CHL` test case scripts are now functional again after internal path corrections.
- The `setup.py` file has been updated accordingly, though users are still advised to install missing dependencies manually for full compatibility.

## Fixed Issues

- **Taylor Diagram Tick Labeling**
  - RMSD ticks are now configurable via a `tickRMS` parameter.
  - The first tick value determines both the tick spacing and the RMSD label position, resolving longstanding issues of fixed/static placement.

- **Validation of Target and Regression Plot Behavior**
  - Following extensive review and expert consultation, the anomalous behavior observed in `Target Plots` and `Regression Lines` for `L4 CHL` data is confirmed to be **data-driven**, not a bug.
  - Validation artifacts are present in the dataset itself; a `pytest` test suite will be released in the near future to systematically verify these findings.

## Future Work

### Near-Term Roadmap

- Add headers and docstrings to `Taylor` and `Target` plot functions, improving readability and consistency.
- Refactor internal `for` loops using `itertools` to reduce redundancy and optimize performance.

### Upcoming Feature Development

- Begin refactoring of the `Benthic_layer.py` test script:
  - Modularize computation and plotting functions
  - Implement monthly volume calculations for deep water formation  
    (_based on upcoming work from Oddo et al._)

- Add functionality to export plotting data as both `.csv` and `.nc` files (currently postponed due to priority conflicts).

- Launch support for `L3s` data in Sea Surface Temperature analysis.
  - This will **deprecate** support for `L4` data due to unsatisfactory reliability and quality of results.

--------------------------------------------------------------------------------------

# **Version:** 4.0.0-α - UNSTABLE  
**Date:** 20/05/2025

## Summary

This release marks a major overhaul of the project, transitioning it from standalone scripts into a fully modular Python package. The deprecated `Corollary.py` and `Auxilliary.py` scripts have been restructured and their functions relocated to more logically organized modules. Several changes to the package structure and default plotting options are introduced to enhance usability and maintainability.

## Major Changes

### **Hydrological_model_validator as a Python Package**

- The core structure of the `Hydrological_model_validator` project has been reworked into a Python package, making it easier to install and use as a library.
- A new `Setup.py` script has been introduced, enabling installation of necessary dependencies. However, as not all dependencies are included, manual installation of additional libraries (as listed in the README) is still required.
- The `Processing` and `Plotting` modules have been reorganized as submodules, allowing users to import specific functions from their respective scripts.
- The `Path` command from the `pathlib` Python library has been deprecated across the codebase, though it will remain in test case scripts for accessing data directories.

## Deprecations

- The `Corollary.py` and `Auxilliary.py` scripts are now officially deprecated due to the complexity and overabundance of functions. These functionalities have been moved to more specialized scripts to improve organization and maintainability.

## New Functionality: Modularized Script Collections

To better organize the deprecated functions from `Corollary.py` and `Auxilliary.py`, new themed scripts have been introduced within the `Processing` and `Plotting` modules. These changes ensure better modularity and improve the user experience by grouping related functions. The new scripts are as follows:

### **Processing Module:**
- `time_utils.py`:  
  - `leapyear`  
  - `true_true_time_series_length`  
  - `split_to_monthly`, `split_to_yearly`  
  - `get_common_years`  
  - `get_season_mask`

- `data_alignment.py`:  
  - `get_valid_mask`, `get_valid_mask_pandas`  
  - `align_pandas_series`, `align_numpy_series`  
  - `get_common_series_by_year`, `get_common_series_by_year_month`  
  - `extract_mod_sat_key`  
  - `gather_monthly_data_across_years`

- `file_io.py`:  
  - `mask_reader`  
  - `load_dataset`

- `stat_math_utils.py`:  
  - `fit_huber`  
  - `fit_lowess`  
  - `round_up_to_nearest`

- `utils.py`:  
  - `find_key`

### **Plotting Module:**
- `formatting.py`:  
  - `format_unit`  
  - `get_variable_from_label_unit`  
  - `fill_anular_region`  
  - `get_min_max_for_identity_line`  
  - `_style_axis_custom`

These scripts will be expanded as necessary to accommodate additional functions and improve usability.

## Default Plotting Options

- All previous options used in the plotting functions are now set as defaults. If the user does not provide custom options, the package will automatically apply these default settings, improving ease of use and flexibility for customizations. This will be further enhanced in the upcoming test case update.

## Test Case Scripts

- Test case scripts have been relocated to a dedicated folder alongside the data folder. This structure allows for better organization and easier management of test data moving forward.

## UNSTABLE

- **Important Notice**: This release is **extremely unstable** due to the fundamental changes in file paths and the overall structure of the package. Many old paths used to fetch functions have been broken, and some functions are still in the process of being integrated into the new structure.
- It is advised to **avoid using this release for anything beyond basic plotting functions**. Upcoming updates will address these issues and re-implement missing functionalities, restoring full compatibility.

## Future Work

- The immediate focus is to re-enable all core functions within the new package structure and restore their usability as quickly as possible.
- Future updates will:
  - Move default options for `Target` and `Taylor` plots into a centralized configuration file.
  - Rework the `Benthic_layer.py` test case script to separate computational and plotting functions, which are currently intertwined.
  - Provide the option to save plot data in both `.csv` and `.nc` formats.
  - Optimize the data reading/setup scripts, with a focus on improving performance and finally integrating the interpolator into the Python ecosystem.

--------------------------------------------------------------------------------------

# **Version:** 3.1.1
**Date:** 19/05/2025 

## Summary

Small patch for both analyser scripts regarding typos and a patch for both `Target_plot.py` and `Taylor_diagrams.py` functions' scripts regarding a couple of bugs.

## `Taylor_diagrams.py`

Fixed a visualization issue for the `Taylor_diagrams.py` scripts for which the title would not be properly displayed in the saved image, the extension of the plot is extended to accommodate more space for the text.

## `Target_plots.py`

Fixed a bug due to which the yearly Target plot would be saved as a white image with nothing inside

--------------------------------------------------------------------------------------

# **Version:** 3.1 
**Date:** 18/05/2025  

## Summary

This update introduces a rework of the **Whiskerbox** and **Violin** plot functions and adds a new utility for streamlined variable extraction.

## Plot Enhancements

- **Whiskerbox** and **Violin** plots have been restructured to follow the same logic and structure used in the other plotting functions.
- These plots now support:
  - Automatic key extraction from nested dictionaries.
  - Dynamic title and label formatting via existing auxiliary functions.

## New Helper Function: `gather_monthly_data_across_years`

- A new utility function, `gather_monthly_data_across_years`, has been implemented to facilitate data extraction across multiple years.
- Currently tailored for **box/violin plot** input, but will be tested and adapted for wider use across additional plotting and computation workflows.

## Future Direction

- Further optimize plotting routines for speed and clarity.
- Begin reworking data loading and interpolation functions for faster runtime.
- Explore full Python replacement of the current MATLAB `Interpolato.m` script.
- Improvements of changelogs listing the new functions that are added in each update. The added function in the previous 3.x.x updates are:
	- ver 3.0.1:
		- In `Corollary.py`:
			- `get_common_series_by_year` (slices dataset based on years)
			- `get_common_series_by_year_month` (slices dataset based on years and months)
	- ver 3.0
		- All of `Auxiliary.py` (functions to aid for the necessary computations regarding the plotting function, contains statistics and other)
		- All of `Target_computations-py` (computations and normalisations necessary for the correct plotting of the Target Plots)
		- All of `Taylor_computations.py` (computations and normalisations necessary for the correct plotting of the Taylor Diagrams)
		- All of `Density.py` (bundles necessary density computations)
		- In `Corollary.py`:
			- `extract_mod_sat_keys` (allows for the identification/extraction of model and satellite dictionary keys)

## Known Issues

- **RMSD Label Placement**: Labels are currently tied to a fixed first arc value. Further investigation into the `SkillMetrics` library is ongoing to determine how arc ranges are defined and whether label placement can be dynamically bound to them.
- **Static RMSD Arc Ticks**: Taylor diagrams use the same arc ticks across plots. While this helps with consistency in test cases, dynamic adjustment would improve generality. Removing the `tickrms` override may solve this, but could also interfere with label alignment (see above).
- **Unexpected Target Plot Results**: Initial performance scores from Target plots appear lower than anticipated. Ongoing testing will determine if this is a bug, data artifact, or an accurate model assessment.
- **Chlorophyll regression analysis** occasionally produces anomalous values — further investigation is underway.

--------------------------------------------------------------------------------------

# **Version:** 3.0.1  
**Date:** 18/05/2025  

## Summary

This is a minor update focused on expanding DataFrame usability within the SST and CHL analysis scripts and improving dataset loading performance.

## Expanded Use of Pandas DataFrames

- SST and CHL analysis scripts now fully leverage **`pandas` DataFrames**, enabling:
  - Seamless integration of the `datetime` dimension.
  - More efficient time-based slicing into **monthly** and **yearly** datasets using native `pandas` methods.
- Enhances clarity and performance for long-term and seasonal trend analysis.

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