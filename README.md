# Hydrological Model Validator

## Overview

This project is designed to **read, interpolate, and analyze** Hydrological-Geochemical model data, enabling **validation against satellite observations**.

It is developed as part of a thesis project and may be extended with additional analysis methods in the future.

---

## Project Structure

The project is divided into two main components:

### 1. Data Setup
- Identification of missing data
- Interpolation using either:
  - Satellite data
    - Model data
    - Computation of Basin Averages

    ### 2. Data Analysis & Validation

    #### Analysis Tools:
    - Time series plotting
    - Bias plotting
    - Scatter plots

    #### Validation Tools:
    - Taylor diagrams
    - Target plots
    - Coefficient of Determination
    - **Weighted** Coefficient of Determination
    - Index of Agreement
    - Nash-Sutcliffe Efficiency

    ---

## Installation

All of the codes required for the cleaning, setupping and analysis of the data.
The main analyzers are divided in either a Sea Surface Temperature script or a Chlorofille script.
The cleaning and data setupping scripts are contained within the "Processing" folder.
Each folder contains the main functions to run the scripts.

## The Test Case

The project has been mostly taylored for the analysis of a North-Adriatic simulation of a Bio-Physical model developed combining the BFM and NEMO models.
The simulation revolves around the Sea Surface Temperature Fields and the Chlorofille values simulated between the year 2000 up to 2009.
Due to the rough size of the datasets (~ 22.2 Gb of data for the model and ~2 Gb satellite data) they will not be provided here due to the size file limit of GitHub.
These datas will be provided only upon request via mail.

# Bibliography

The Northern Adriatic Forecasting System for Circulation and Biogeochemistry: Implementation and Preliminary Results (Scroccaro I et al., 2022)

Comparison of different efficiency criteria for hydrological model assessment (Krause P. et al., 2005)

Summary diagrams for coupled hydrodynamic-ecosystem model skill assessment (Jolliff et al., 2008)
