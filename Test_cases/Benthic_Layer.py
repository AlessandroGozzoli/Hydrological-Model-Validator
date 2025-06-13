#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################
##                     Author: Gozzoli Alessandro                            ##
##              email: alessandro.gozzoli4@studio.unibo.it                   ##
##                        UniBO id: 0001126381                               ##
###############################################################################

def main():

    # Ignoring a depracation warning to ensure a better console run
    import warnings
    from cryptography.utils import CryptographyDeprecationWarning

    # Ignore specific deprecation warning from cryptography (used by paramiko)
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

###############################################################################
###############################################################################
##                               EXPLANATION                                 ##
###############################################################################
###############################################################################
###############################################################################

    title = "WELCOME TO THE BFM DATA ANALYSIS TEST CASE SCRIPT"
    border = "#" * 60
    print(border)
    print(title.center(60))
    print(border)


    print("""\nThis is a test case which illustrated the possible analysis that
can be done using the tools provided by the package.
This code retrieve the raw case data obtained  via a simulation of BFM 
(Bio-Geochemical Model) and extracts it for further computation/plotting
         
This test case is splin into 2 different objectives/metrcis analysed.

Firstly an analysis about the density will take place by extracting the bottom
layer physical parameters (temperature and salinity) then an analysis of the 
biogeochemical components will take place by plotting them.""")

    input("Please press any key to confirm and move on: \n")

###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

    # ----- LIBRARIED FOR PATHS -----
    import os
    from pathlib import Path

    # ----- UTILITY LIBRARIES -----
    import numpy as np
    import xarray as xr
    from datetime import datetime
    
###############################################################################
##                                                                           ##
##                                MODULES                                    ##
##                                                                           ##
###############################################################################

    print(border)
    print("\nLoading the necessary modules provided by the package...")

    print("Loading the Processing modules and constants...")
    from Hydrological_model_validator.Processing.BFM_data_reader import (read_benthic_parameter,
                                                                         read_bfm_chemical)

    from Hydrological_model_validator.Plotting.formatting import compute_geolocalized_coords
    print("\033[92m✅ Processing modules have been loaded!\033[0m")
    print("-"*45)

    print("Loading the density computation adjacent functions...")
    from Hydrological_model_validator.Processing.Density import (compute_density_bottom,
                                                             compute_Bmost, 
                                                             compute_Bleast,
                                                             filter_dense_water_masses,
                                                             compute_dense_water_volume)
    print("\033[92m✅ File I/O modules have been loaded!\033[0m")
    print("-"*45)

    print("Loading the plotting modules...")
    from Hydrological_model_validator.Plotting.bfm_plots import (Benthic_physical_plot,
                                                             Benthic_chemical_plot,
                                                             Benthic_depth,
                                                             plot_benthic_3d_mesh,
                                                             dense_water_timeseries)

    print("\033[92m✅ The plotting modules have been loaded!\033[0m")
    print('-'*45)

###############################################################################
##                                                                           ##
##                             DATA LOADING                                  ##
##                                                                           ##
###############################################################################

    # ----- SETTING UP THE WORKING DIRECTOTY -----
    print("Resetting the working directory...")
    WDIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(WDIR)  # Set the working directory
    print('*'*45)

    # ----- BASE DATA DIRECTORY -----
    BDIR = Path(WDIR, "Data")

    # ----- MODEL DATA DIRECTORY (CONTAINS BFM DATA) -----
    IDIR = Path(BDIR, "MODEL")
    print(f"""\033[91m⚠️ The input data needs to be located in the {IDIR} folder ⚠️
⚠️ Make sure that it contains all of the necessary datasets ⚠️\033[0m
An example dataset is also provided alongside the repository and it 
contains both the data and a mask...
Make sure to download it!""")

    # ----- RETRIEVING THE MASK -----
    print("Retrieving the mask...")
    mask_path = Path(BDIR, "mesh_mask.nc")
    ds = xr.open_dataset(mask_path)
    mask3d = ds['tmask'].values
    mask3d = np.squeeze(mask3d)
    print("\033[92m✅ Mask succesfully imported! \033[0m\n")
    print(border)

###############################################################################
##                                                                           ##
##                         BENTHIC MASK COMPUTATION                          ##
##                                                                           ##
###############################################################################

    print("""\nTo correctly analyse and plot the results some preliminray 
computations need to be done: the geolocalization of the dataset (coversion
from grid cell to eulerian coordinates) and the extraction of the deepest sigma
layer within the water column (Bmost), this is the deepest water cell for
each grid point and represents the phenomena right above the benthic layer.""")

    input("Please press any key to confirm and move on: \n")

    # ----- GEOLOCALIZE THE DATASET -----
    print("Computing the geolocalization...")
    # Known values from the dataset, need to be changed if the area of analysis is changed
    grid_shape = (278, 315)
    epsilon = 0.06  # Correction factor linked to the resolution of the dataset
    x_start = 12.200
    x_step = 0.0100
    y_start = 43.774
    y_step = 0.0073

    geo_coords = compute_geolocalized_coords(grid_shape, epsilon, x_start, x_step, y_start, y_step)
    print("\033[92m✅ Geolocalization complete! \033[0m")
    print('*'*45)

    # ----- RETRIEVE DEEPEST POINTS -----
    print("Extracting the coordinates of the Benthic Layer...")
    Bmost = compute_Bmost(mask3d)
    print("\033[92m✅ Benthic coordinates obatined! \033[0m")
    print("-"*45)

    print("Plotting the Benthic layer...")
    
    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("run_%Y-%m-%d")
    output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Depth", timestamp)
    os.makedirs(output_path, exist_ok=True)

    Benthic_depth(Bmost, geo_coords, output_path)
    plot_benthic_3d_mesh(Bmost, 
                     geo_coords, 
                     layer_thickness=2, 
                     plot_type='surface',
                     save_path=output_path)
    print("\033[92m✅ Benthic Layer depth plotted! \033[0m")
    print('-'*45)

    print("Creating a 2D mask for surface plots...")
    Bleast = compute_Bleast(mask3d)
    print("\033[92m✅ Surface coordinates obatined! \033[0m\n")
    print(border)

###############################################################################
##                                                                           ##
##                        DENSITY FIELD COMPUTATION                          ##
##                                                                           ##
###############################################################################

    print("""\nNow that the prelimary work has been concluded the 1st analysis
section will begin.

This section will revolve around the analysis of the water density.

By extracting the physical parameters (temperature and salinity) the density
can be computed using three different kinds of equation. All of the results are
plotted to see the difference.

Afterwards a threshold of 1029.2 is imposed to isolate the densest water masses
to see their formation and to compute their respective volumes.\n""")

    input("Please press any key to confirm and move on: \n")

    print("Beginning to process the temperature and salinity fields to compute the density field...")

    # ----- DEFINE FILE NAME FRAGMENTS FOR THE FUNCTIONS -----
    fragments = {
            'ffrag1': "new15_1m_",
            'ffrag2': "0101_",
            'ffrag3': "1231_grid_T"
        }

    print("Retrieving the temperature data and building the dataframe...")
    temperature_data = read_benthic_parameter(IDIR, mask3d, Bmost, fragments, variable_key='votemper')
    print("Bottom temperature dataframe created succesfully!")
    print('-'*45)

    print("Plotting the temperature data at the bottom layer...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Temperature")
    os.makedirs(outdir, exist_ok=True)

    Benthic_physical_plot(
        temperature_data,
        geo_coords,
        output_path=outdir,
        bfm2plot='votemper',
        unit='°C',
        description='Bottom Temperature'
        )
    print("Temperature data plotted!")
    print('-'*45)

    print("Retrieving the salinity data and building the dataframe...")
    salinity_data = read_benthic_parameter(IDIR, mask3d, Bmost, fragments, variable_key='vosaline')
    print("Bottom salinity dataframe created succesfully!")
    print('-'*45)

    print("Plotting the salinity data at the bottom layer...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Salinity")
    os.makedirs(outdir, exist_ok=True)

    Benthic_physical_plot(
        salinity_data,
        geo_coords,
        output_path=outdir,
        bfm2plot='vosaline',
        unit='psu',
        description='Bottom Salinity'
        )
    print("Salinity data plotted!")
    print('-'*45)

    print("Temperature and salinity field at the Benthic Layer have been plotted!")
    print('*'*45)

    print("Computing the density field using all methods...")
    density_data_SEOS = compute_density_bottom(
        temperature_data,
        salinity_data,
        Bmost,
        method="EOS" 
        )

    density_data_EOS80 = compute_density_bottom(
        temperature_data,
        salinity_data,
        Bmost,
        method="EOS80"  
        )

    density_data_TEOS10 = compute_density_bottom(
        temperature_data,
        salinity_data,
        Bmost,
        method="TEOS10"  
        )
    print("Density field has been computed using all methods!")
    print('-'*45)

    print("Proceeding to plot the results...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Density")
    os.makedirs(outdir, exist_ok=True)

    print("Starting from the linearized equation of state...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Density", "SEOS")
    os.makedirs(outdir, exist_ok=True)

    Benthic_physical_plot(
        density_data_SEOS,
        geo_coords,
        output_path=outdir,
        bfm2plot='density',
        unit='kg/m3',
        description='Bottom Density - SEOS'
        )
    print("Linearized EOS results plotted!")
    print("-"*45)
    
    print("Moving onto the EOS-80 results...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Density", "EOS-80")
    os.makedirs(outdir, exist_ok=True)

    Benthic_physical_plot(
        density_data_EOS80,
        geo_coords,
        output_path=outdir,
        bfm2plot='density',
        unit='kg/m3',
        description='Bottom Density - EOS-80'
        )
    print("EOS-80 results plotted!")
    print('-'*45)
    
    print("Finishing with the TEOS-10 results...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Density", "TEOS-10")
    os.makedirs(outdir, exist_ok=True)

    Benthic_physical_plot(
        density_data_TEOS10,
        geo_coords,
        output_path=outdir,
        bfm2plot='density',
        unit='kg/m3',
        description='Bottom Density - TEOS-10'
        )
    print("TEOS-10 reults plotted!")
    print('-'*45)

    print("Proceeding with the computation of the dense water at the bottom layer...")
    dense_water_SEOS=filter_dense_water_masses(density_data_SEOS)
    dense_water_EOS80=filter_dense_water_masses(density_data_EOS80)
    dense_water_TEOS10=filter_dense_water_masses(density_data_TEOS10)
    print("Dense water dataframes have been created")

    print("Proceeding to plot them...")
    print("Starting from the linearised SEOS dense water results...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Density", "SEOS", "Dense_Water")
    os.makedirs(outdir, exist_ok=True)

    Benthic_physical_plot(
        dense_water_SEOS,
        geo_coords,
        output_path=outdir,
        bfm2plot='dense_water',
        unit='kg/m3',
        description='Dense Water mass - SEOS'
        )
    print("SEOS dense water results plotted!")

    print("Proceeding with the EOS-80 Dense Water results...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Density", "EOS-80", "Dense_Water")
    os.makedirs(outdir, exist_ok=True)

    Benthic_physical_plot(
        dense_water_EOS80,
        geo_coords,
        output_path=outdir,
        bfm2plot='dense_water',
        unit='kg/m3',
        description='Dense Water mass - EOS-80'
        )
    print("EOS-80 Dense Water results plotted!")

    print("Moving onto the TEOS-10 Dense Water results...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Density", "TEOS-10", "Dense_Water")
    os.makedirs(outdir, exist_ok=True)
    
    Benthic_physical_plot(
        dense_water_TEOS10,
        geo_coords,
        output_path=outdir,
        bfm2plot='dense_water',
        unit='kg/m3',
        description='Dense Water mass - TEOS-10'
        )
    print("TEOS-10 Densa Water results plotted!")
    print("The dense water masses results computed with each method have been succesully plotted!")
    print('-'*45)

    print("Attempting to compute the volume of dense water...")
    print("SEOS...")
    dense_water_volume_SEOS = compute_dense_water_volume(IDIR, mask3d, fragments, density_method='EOS')
    print("EOS-80...")
    dense_water_volume_EOS80 = compute_dense_water_volume(IDIR, mask3d, fragments, density_method='EOS80')
    print("TEOS-10...")
    dense_water_volume_TEOS10 = compute_dense_water_volume(IDIR, mask3d, fragments, density_method='TEOS10')
    print("Dense water volume has been computed!")
    
    print("Proceeding to plot it...")
    outdir = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "Density")
    os.makedirs(outdir, exist_ok=True)
    
    dense_water_timeseries({
        "EOS": dense_water_volume_SEOS,
        "EOS80": dense_water_volume_EOS80,
        "TEOS10": dense_water_volume_TEOS10,
        },
        output_path=output_path)
    print("Dense water volume plotted!\n")
    print(border)
        
###############################################################################
##                                                                           ##
##                       GEOCHEMICAL FIELD PLOTTING                          ##
##                                                                           ##
###############################################################################

    print("""\nFor the 2nd part of the test case script the geochemical 
parameters will be extracted and plotted\n""")

    input("Please press any key to confirm and move on: \n")

    # ----- ASK THE USER WHICH VARIABLE TO PLOT -----
    # Dictionary of valid variable codes and their descriptions
    bfm__chem_variables = {
        "Chla": ("Chlorophyll-a", "mg Chl/m3"),
        "O2o": ("Oxygen", "mmol O2/m3"),
        "N1p": ("Phosphate", "mmol P/m3"),
        "N3n": ("Nitrate", "mmol N/m3"),
        "N4n": ("Ammonium", "mmol N/m3"),
        "P1c": ("Diatoms (C)", "mg C/m3"),
        "P2c": ("Flagellates (C)", "mg C/m3"),
        "P3c": ("PicoPhytoplankton (C)", "mg C/m3"),
        "P4c": ("Large Phytoplankton (C)", "mg C/m3"),
        "Z3c": ("Carnivorous Mesozooplankton (C)", "mg C/m3"),
        "Z4c": ("Omnivorous Mesozooplankton (C)", "mg C/m3"),
        "Z5c": ("Microzooplankton (C)", "mg C/m3"),
        "Z6c": ("Heterotrophic Nanoflagellates (HNAN) (C)", "mg C/m3"),
        "R6c": ("Particulate Organic Matter (C)", "mg C/m3"),
        }

    print("The chemical species contained in the BFM model are: ")
    for code, (description, unit) in sorted(bfm__chem_variables.items()):
        print(f"{code:5} {description}")

    while True:
        bfm2plot = input("Enter the chemical species you'd like to plot (e.g., 'Chla', 'N3n', 'P1c'): ").strip()
        if bfm2plot in bfm__chem_variables:
            # Retrieve the description and unit
            description, unit = bfm__chem_variables[bfm2plot]
            print(f"✅ Selected: {bfm2plot} - {description} [{unit}]")
        
            # Store them in a variable if you need them for further use
            selected_description = description
            selected_unit = str(unit)
            
            break
        else:
            print(f"⚠️ '{bfm2plot}' is not a valid BFM variable code. Please try again.\n")

    print('-'*45)

    # ----- DEFINE NEW FILENAME FRAGMENTS -----

    fragments = {
                'ffrag1': "new15_1m_",
                'ffrag2': "0101_",
                'ffrag3': "1231_grid_bfm"
                }

    # ----- READ AND PLOTS THE VARIABLE -----
    output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", bfm2plot, "BENTHIC")
    os.makedirs(output_path, exist_ok=True)

    print(f"Beginning the retrieval of the {selected_description} bottom data...")

    bfm_bottom_data = read_bfm_chemical(IDIR, mask3d, Bmost, fragments, variable_key=bfm2plot)

    Benthic_chemical_plot(
        bfm_bottom_data,
        geo_coords,
        output_path=output_path,
        bfm2plot=bfm2plot,
        unit=selected_unit,
        description=selected_description,
        location='Bottom'
        )

    print(f"The {selected_description} monthly mean values for the whole bottom dataset have been plotted!")   
    print('-'*45) 

    print(f"Moving onto the plotting of the surface {selected_description} data...")
    output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", bfm2plot, "SURFACE")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Beginning the retrieval of the {selected_description} surface data...")
         
    bfm_surface_data = read_bfm_chemical(IDIR, mask3d, Bleast, fragments, variable_key=bfm2plot)
    
    Benthic_chemical_plot(
        bfm_surface_data,
        geo_coords,
        output_path=output_path,
        bfm2plot=bfm2plot,
        unit=selected_unit,
        description=selected_description
        )

    print(f"The {selected_description} monthly mean values for the whole surface dataset have been plotted!\n")   
    print(border)
    
if __name__ == "__main__":
    main()