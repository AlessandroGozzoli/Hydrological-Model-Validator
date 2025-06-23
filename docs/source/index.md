# Hydrological Model Validator

This project provides a set of tools for evaluating the performance of Bio-Geo-Hydrological simulations and analyzing their outputs.

The focus of this repository is on **post-processing**, offering utilities to:

- Clean and pre-process relevant dataframes
- Interpolate observed datasets to handle missing values and apply proper masking
- Compare observed and simulated outputs for validation and performance assessment
- Analyse the simulation outputs for further insights

---

## Features

- Data cleaning and transformation
- Missing data interpolation and masking
- Automated validation metrics and visual comparison
- Optional PDF report generation
- Modular structure for customizable analysis workflows

---

## Project Structure

The project is organized into two main objectives:

1. **Quick-Use Toolkit**  
   A high-level interface that allows users to input datasets and automatically generate:
   - Validation plots
   - Summary dataframes
   - A PDF report (optional)

2. **Modular Subcomponents**  
   A collection of standalone functions and classes for users who prefer to build custom analysis pipelines or integrate specific components into other projects.
   These are collected into 3 submodules, which can both be used as standalones or combined:
    - **`Processing/`**: Functions for reading, cleaning, transforming, and analyzing input datasets.
    - **`Plotting/`**: Tools for generating a variety of plots from the processed results, including time series, scatter plots, and performance metrics.
    - **`Report/`**: Utilities for creating structured PDF reports, incorporating plots, summary statistics, and metadata.

> **Note:** This repository is part of a **Physics of the Earth System** thesis and may be expanded in the future to include additional variables and more advanced analytical features.

---

### Model/Simulation Evaluation

The current evaluation approach is based on a **direct comparison** between simulated and observed datasets over the same time window. The results are presented through a variety of plots and statistical performance metrics.

#### Analytical Tools

The following visualization and statistical tools are used to evaluate model performance:

- **Time Series & Scatter Plots**
  - General time series plots for visual inspection
  - Seasonal scatter plots for intra-annual trends

- **Distribution Plots**
  - Box-and-whisker plots
  - Violin plots

- **Multivariate Performance Plots**
  - Target diagrams
  - Taylor diagrams

- **Efficiency Metrics**
  A wide set of statistical coefficients is implemented to evaluate model accuracy:

  - **Coefficient of Determination (R²)**
    - Standard
    - Weighted

  - **Index of Agreement (d)**
    - Standard
    - Modified
    - Relative

  - **Nash–Sutcliffe Efficiency (NSE)**
    - Standard
    - Logarithmic
    - Modified
    - Relative

- **Error Decomposition**
  - **Time-series and Spectral Analysis**
    - Compared against cloud coverage patterns
  - **Spatial Performance Mapping**
    - Annual and monthly resolution maps showing model performance across geographic regions

---

### Expansion of the Analysis: Bottom (Benthic) σ-Layer

As the **first direction for expanded analysis**, this repository introduces tools focused on the **extraction and study of the bottom σ-layer** of the simulation grid. This layer is particularly relevant for investigating the **formation of deep water masses** and the **distribution of bio-geochemical variables** near the seabed.

Once the model has been validated using the core evaluation tools, users can apply these modules to explore processes such as:

- Stratification and mixing at depth  
- Tracer evolution in deep layers (e.g., nutrients, oxygen, carbon compounds)  
- Temporal variability in bottom water properties

> For implementation details and example workflows, refer to the test cases provided in the `Test_cases/` directory.

--- 

# Installation Guide

This project can be installed using `conda` (recommended) or `pip` across all major operating systems. Below you’ll also find guidance for optional tools like **MATLAB** and **CDO** (Linux only) which are integrated in some of the functions/routines

---

## **Python Environment Setup**

Python version supported : ![Python version](https://img.shields.io/badge/python-3.9|3.10|3.11|3.12|3.13-blue.svg)

<details>
<summary><strong>Conda (Recommended)</strong></summary>
<p><strong>All Systems</strong></p>

```bash
# Create a new conda environment
conda create -n hydroval python=3.10

# Activate the environment
conda activate hydroval

# Install the package and dependencies in editable mode
pip install -e .
```
</details>

<details>
<summary><strong>Pip Only (Without Conda)</strong></summary>
<p><strong>All Systems</strong></p>

```bash
# Optionally create and activate a virtual environment (recommended)
python -m venv env
source env/bin/activate      # Windows: env\Scripts\activate

# Install the package and dependencies in editable mode
pip install -e .
```
</details>

<details>
<summary><strong>Alternative Pip Options</strong></summary>
<p><strong>--user (No admin rights)</strong></p>

```bash
pip install --user -e .
```

<p><strong>-e (Editable/development mode)<p><strong>

```bash
pip install -e .
```
Use -e when actively developing or modifying the source code.
</details>

---

## MATLAB (Optional but needed for the interpolator script)

<details>
<summary><strong>MATLAB Setup (All Systems)</strong></summary>

### Description

Some test cases or post-analysis steps may require MATLAB. Make sure it's installed and available via your system's PATH.

🔗 [Official MATLAB Installation Guide](https://www.mathworks.com/help/install/)
</details>

To correctly run the interpolator, the toolboxes **m_map**, **mexcdf**, and **nctoolbox** need to be accessible by the script. Please make sure that their paths are reachable by your MATLAB installation. For a guide on how to add paths in MATLAB, please refer to [MATLAB Add Folder to Path Documentation](https://www.mathworks.com/help/matlab/ref/addpath.html).

The usage of a MATLAB interpolator is to make the process NOAA compliant by using their same tools, allowing future integration of this repository with other [NOAA](https://www.noaa.gov/) tools.

- **m_map** toolbox: [https://www.eoas.ubc.ca/~rich/map.html](https://www.eoas.ubc.ca/~rich/map.html)  
- **mexcdf** toolbox: [https://www.mathworks.com/matlabcentral/fileexchange/26310-netcdf-interface-for-matlab-mexcdf](https://www.mathworks.com/matlabcentral/fileexchange/26310-netcdf-interface-for-matlab-mexcdf)  
- **nctoolbox**: [https://github.com/nctoolbox/nctoolbox](https://github.com/nctoolbox/nctoolbox)

---

## CDO - Climate Data Operators (Linux Only)

<details>
<summary><strong>CDO Setup (Linux Only)</strong></summary>

### ⚠️ CDO is supported **only on Linux-based systems**.

```bash
# Ubuntu/Debian
sudo apt install cdo

# Or use conda
conda install -c conda-forge cdo
```

🔗 [Official CDO Installation Guide](https://code.mpimet.mpg.de/projects/cdo/wiki/Cdo)
</details>

---

## Helpful Links

<details>
<summary><strong>Official Documentation</strong></summary>

- 🐍 [Python Installation](https://www.python.org/downloads/)
- 📦 [Anaconda/Miniconda Installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- 🪟 [WSL for Windows Users](https://learn.microsoft.com/en-us/windows/wsl/install)
- 🌐 [MATLAB Installation](https://www.mathworks.com/help/install/)
- 🌊 [CDO (Linux only)](https://code.mpimet.mpg.de/projects/cdo/wiki/Cdo)
</details>

---

# Usage Guide: `GenerateReport` CLI

The `GenerateReport` command-line interface (CLI) allows users to generate evaluation reports from observed and simulated Bio-Geo-Hydrological datasets.

## Basic Command

```bash
GenerateReport [input_folder_or_dict] [OPTIONS]
```

---

## Positional Argument

```bash
usage: GenerateReport [-h] [--output-dir path] [--check] [--no-pdf] [--verbose] [--open-report]
                      [--variable var_name] [--unit unit_str] [--no-banner] [--info] [--version]
                      [input]

Generate a comprehensive evaluation report from observed and simulated Bio-Geo-Hydrological datasets.

positional arguments:
  input                 Path to the input data directory or a dictionary of file paths.
                        You can pass:
                          - a folder containing: obs_spatial, sim_spatial, obs_ts, sim_ts, and mask
                          - or a stringified dictionary (JSON or Python format) mapping these keys:
                            {
                              "obs_spatial": "obs_spatial.nc",
                              "sim_spatial": "sim_spatial.nc",
                              "obs_ts": "obs_timeseries.csv",
                              "sim_ts": "sim_timeseries.csv",
                              "mask": "mask.nc"
                            }

options:
  -h, --help            Show this help message and exit
  --output-dir path     Destination folder for report and plots (default: ./REPORT)
  --check               Validate input files and structure only, no report generation
  --no-pdf              Skip PDF generation, only output plots and dataframes
  --verbose             Enable detailed logging
  --open-report         Automatically open the PDF report if generated
  --variable var_name   Name of the target variable (e.g. "Chlorophyll-a")
  --unit unit_str       Unit of the variable (e.g. "mg/L", "m3/s"), LaTeX-ready
  --no-banner           Suppress ASCII banner (useful for batch jobs)
  --info                Show program description and exit
  --version             Show version and exit
```

---

## Examples

<details>
<summary><strong>Minimal Run (Interactive)</strong></summary>

```bash
GenerateReport ./data
```

</details>

<details>
<summary><strong>With Output Directory & No PDF</strong></summary>

```bash
GenerateReport ./data --output-dir ./results --no-pdf
```

</details>

<details>
<summary><strong>Using a JSON-Style Dictionary</strong></summary>

```bash
GenerateReport "{ \"obs_spatial\": \"obs.nc\", \"sim_spatial\": \"sim.nc\", \"obs_ts\": \"obs.csv\", \"sim_ts\": \"sim.csv\", \"mask\": \"mask.nc\" }"
```

</details>

<details>
<summary><strong>Quiet Batch Run (No Banner, Auto Open Report)</strong></summary>

```bash
GenerateReport ./data --no-banner --open-report
```

</details>

---

> For example usage of the singular functions (sans the report generation ones) availbale in the repository, and generally for in-script import and usage, please refer to the test cases available in the **`Test_cases/`** folder and their respective TEST_CASES_README file.

---

# Test Cases and Pytests

This repository includes a suite of **example routines** and **automated tests** to ensure the correct functionality of its components. All tests are located in the **`Test_cases/`** directory.

---

## Test Case Scripts

These are **step-by-step, verbose scripts** that demonstrate how to apply the tools for data cleaning, analysis, and reporting. They are ideal for understanding the intended usage.

### Available Test Cases

- **`Data_cleaner_setupper.py`**  
  Demonstrates how to clean and prepare datasets for analysis.  
  Includes the MATLAB script **`Interpolator_v2.m`** to perform bilinear interpolation on observed datasets.

- **`SST_data_analyzer.py` & `CHL_data_analyzer.py`**  
  Practical examples of analysis workflows using **Sea Surface Temperature** and **Chlorophyll-a** datasets.  
  These are simplified and didactical illustrations of what the `Report_generator` submodule automates.

- **`Benthic_layer.py`**  
  Focuses on extracting and analyzing **bottom σ-layers**, emphasizing **dense water formation** and **bio-geochemical tracers** near the seabed.

---

## Pytests and Code Quality

Automated testing ensures reliability and stability of the modules, using:

- [`pytest`](https://docs.pytest.org/en/stable/)
- [`flake8`](https://flake8.pycqa.org/en/latest/) (for linting and style enforcement)

You can run them via:

```bash
pytest
flake8 src/
```

These tools verify logic correctness, class behavior, and code style compliance.

---

## Code Quality Reports

This project is continuously monitored with external quality and coverage tools:

| Codacy | Codebeat | Codecov | Documentation|
|--------|----------|---------|--------------|
| [![Codacy Badge](https://app.codacy.com/project/badge/Grade/78c1d747de5f4f0f9abb1e12af4b4f5a)](https://app.codacy.com/gh/AlessandroGozzoli/Hydrological-Model-Validator/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) | [![codebeat badge](https://codebeat.co/badges/7416d105-cd9a-4f3d-b5f2-16f311ba4de4)](https://codebeat.co/projects/github-com-alessandrogozzoli-hydrological-model-validator-master) | [![codecov](https://codecov.io/gh/AlessandroGozzoli/Hydrological-Model-Validator/graph/badge.svg?token=JYO0BFY7OX)](https://codecov.io/gh/AlessandroGozzoli/Hydrological-Model-Validator) | [![Documentation Status](https://readthedocs.org/projects/hydrological-model-validator/badge/?version=latest)](https://hydrological-model-validator.readthedocs.io/en/latest/?badge=latest)

---

# Bibliography

[**The Northern Adriatic Forecasting System for Circulation and Biogeochemistry: Implementation and Preliminary Results (Scroccaro I et al., 2022)**](https://www.researchgate.net/publication/363410921_The_Northern_Adriatic_Forecasting_System_for_Circulation_and_Biogeochemistry_Implementation_and_Preliminary_Results)

[**Comparison of different efficiency criteria for hydrological model assessment (Krause P. et al., 2005)**](https://www.researchgate.net/publication/26438340_Comparison_of_Different_Efficiency_Criteria_for_Hydrologic_Models)

[**Summary diagrams for coupled hydrodynamic-ecosystem model skill assessment (Jolliff et al., 2008)**](https://www.researchgate.net/publication/222660103_Summary_diagrams_for_coupled_hydrodynamic-ecosystem_model_skill_assessment)

[**The International Thermodynamic Equation of Seawater 2010 (TEOS-10): Calculation and Use of Thermodynamic Properties (McDougall et al., 2010)**](https://www.researchgate.net/publication/216028042_The_International_Thermodynamic_Equation_of_Seawater_2010_TEOS-10_Calculation_and_Use_of_Thermodynamic_Properties)

[**Defining a Simplified Yet “Realistic” Equation of State for Seawater (Roquet et al., 2015)**](https://journals.ametsoc.org/view/journals/phoc/45/10/jpo-d-15-0080.1.xml?utm_source=chatgpt.com)

[**Climatological analysis of the Adriatic Sea thermohaline characteristics (Giorgietti A., 1998)**](https://bgo.ogs.it/sites/default/files/2023-08/bgta40.1_GIORGETTI.pdf)

[**Evaluation of different Maritime rapid environmental assessment procedures with a focus on acoustic performance (Oddo et al., 2022)**](https://pubs.aip.org/asa/jasa/article/152/5/2962/2840159/Evaluation-of-different-Maritime-rapid)

[**A study of the hydrographic conditions in the Adriatic Sea from numerical modelling and direct observations (2000–2008) (Oddo et al., 2011)**](https://os.copernicus.org/articles/7/549/2011/)


```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:

api/api_index
test_case_readme_wrapped
changelog_wrapper