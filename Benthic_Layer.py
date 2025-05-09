###############################################################################
##                     Author: Gozzoli Alessandro                            ##
##              email: alessandro.gozzoli4@studio.unibo.it                   ##
##                        UniBO id: 0001126381                               ##
###############################################################################

# Ignoring a depracation warning to ensure a better console run
import warnings
from cryptography.utils import CryptographyDeprecationWarning

# Ignore specific deprecation warning from cryptography (used by paramiko)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

###############################################################################
###############################################################################
##    This reads the monthly average BFM model data and extract the data     ##
##     for further computations. The code is split into 2 main sections.     ##
##     The 1st extract the sea surface temperature field and the salinity    ##
##       field and combines them using and ESO equation to compute the       ##
##     pressure field. The values at the Benthic Layer are then extracted    ##
##   The 2nd part of the code extracts all of the specified chemical values  ##
##           at the Benthic Layer and plots them using a 2D map.             ##
###############################################################################
###############################################################################

print("#### WELCOME TO THE BENTHIC LAYER ANALYSIS SCRPT ####")

###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# ----- LIBRARIED FOR PATHS -----
import os
from pathlib import Path
import sys

# ----- UTILITY LIBRARIES -----
import numpy as np
import xarray as xr
from datetime import datetime

###############################################################################
##                                                                           ##
##                                MODULES                                    ##
##                                                                           ##
###############################################################################

# ANY FUNCTION NEEDS TO BE PUT HERE
print("Loading the necessary modules...")
WDIR = os.getcwd()
os.chdir(WDIR)  # Set the working directory

print("Loading the Pre-Processing modules and constants...")
ProcessingDIR = Path(WDIR, "Processing/")
sys.path.append(str(ProcessingDIR))  # Add the folder to the system path

from Costants import Ybeg, Yend, ysec

from MOD_data_reader import read_bfm

print("\033[92m✅ Pre-processing modules have been loaded!\033[0m")
print("-"*45)

print("Loading the plotting modules...")
PlottingDIR = Path(WDIR, "Plotting")
sys.path.append(str(PlottingDIR))  # Add the folder to the system path

from Plots import Benthic_depth

print("\033[92m✅ The plotting modules have been loaded!\033[0m")
print('-'*45)

###############################################################################
##                                                                           ##
##                             DATA LOADING                                  ##
##                                                                           ##
###############################################################################

# ----- SETTING UP THE WORKING DIRECTOTY -----
print("Resetting the working directory...")
WDIR = os.getcwd()
os.chdir(WDIR)  # Set the working directory
print('*'*45)

# ----- BASE DATA DIRECTORY -----
BDIR = Path(WDIR, "Data")

# ----- ACCESSING THE MODEL DATA FOLDER -----
print("Accessing the Model Data folder...")
MDIR = Path(BDIR, "MODEL")
print(f"Model data folder is {MDIR}")

FDIR = Path(BDIR, "OUTPUT/PLOTS/BFM")

# ----- RETRIEVING THE MASK -----
print("Retrieving the mask...")
mask_path = Path(BDIR, "mesh_mask.nc")
ds = xr.open_dataset(mask_path)
mask3d = ds['tmask'].values
mask3d = np.squeeze(mask3d)
print("\033[92m✅ Mask succesfully imported! \033[0m")
print('*'*45)

###############################################################################
##                                                                           ##
##                       PRESSURE FIELD COMPUTATION                          ##
##                                                                           ##
###############################################################################

# ----- RETRIEVE DEEPEST POINTS -----
print("Extracting the coordinates of the Benthic Layer...")
Bmost = np.zeros((mask3d.shape[1], mask3d.shape[2]))

for j in range(mask3d.shape[1]):
    for i in range(mask3d.shape[2]):  # Notice: should be shape[2] not shape[1]
        Bmost[j, i] = np.sum(mask3d[:, j, i])

# Squeeze in case it's needed
Bmost = np.squeeze(Bmost)
print("\033[92m✅ Benthic coordinates obatines! \033[0m")

print("Plotting the Benthic layer...")

# Create a timestamped folder for this run
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", "DEPTH", timestamp)
os.makedirs(output_path, exist_ok=True)

Benthic_depth(Bmost, output_path)
print("\033[92m✅ Benthic Layer depth plotted! \033[0m")
print('*'*45)

###############################################################################
##                                                                           ##
##                       PRESSURE FIELD COMPUTATION                          ##
##                                                                           ##
###############################################################################

# ----- ASK THE USER WHICH VARIABLE TO PLOT -----
# Dictionary of valid variable codes and their descriptions
bfm_variables = {
    "Chl": ("Chlorophyll", "mg Chl/m3"),
    "O2o": ("Oxygen", "mmol O2/m3"),
    "N1p": ("Phosphate", "mmol P/m3"),
    "N3n": ("Nitrate", "mmol N/m3"),
    "N4n": ("Ammonium", "mmol N/m3"),
    "O4n": ("NitrogenSink", "mmol N/m3"),
    "N5s": ("Silicate", "mmol Si/m3"),
    "N6r": ("Reduction Equivalents", "mmol S--/m3"),
    "B1c": ("Aerobic and Anaerobic Bacteria (C)", "mg C/m3"),
    "B1n": ("Aerobic and Anaerobic Bacteria (N)", "mmol N/m3"),
    "B1p": ("Aerobic and Anaerobic Bacteria (P)", "mmol P/m3"),
    "P1c": ("Diatoms (C)", "mg C/m3"),
    "P1n": ("Diatoms (N)", "mmol N/m3"),
    "P1p": ("Diatoms (P)", "mmol P/m3"),
    "P1l": ("Diatoms (Chl)", "mg Chl/m3"),
    "P1s": ("Diatoms (Si)", "mmol Si/m3"),
    "P2c": ("Flagellates (C)", "mg C/m3"),
    "P2n": ("Flagellates (N)", "mmol N/m3"),
    "P2p": ("Flagellates (P)", "mmol P/m3"),
    "P2l": ("Flagellates (Chl)", "mg Chl/m3"),
    "P3c": ("PicoPhytoplankton (C)", "mg C/m3"),
    "P3n": ("PicoPhytoplankton (N)", "mmol N/m3"),
    "P3p": ("PicoPhytoplankton (P)", "mmol P/m3"),
    "P3l": ("PicoPhytoplankton (Chl)", "mg Chl/m3"),
    "P4c": ("Large Phytoplankton (C)", "mg C/m3"),
    "P4n": ("Large Phytoplankton (N)", "mmol N/m3"),
    "P4p": ("Large Phytoplankton (P)", "mmol P/m3"),
    "P4l": ("Large Phytoplankton (Chl)", "mg Chl/m3"),
    "Z3c": ("Carnivorous Mesozooplankton (C)", "mg C/m3"),
    "Z3n": ("Carnivorous Mesozooplankton (N)", "mmol N/m3"),
    "Z3p": ("Carnivorous Mesozooplankton (P)", "mmol P/m3"),
    "Z4c": ("Omnivorous Mesozooplankton (C)", "mg C/m3"),
    "Z4n": ("Omnivorous Mesozooplankton (N)", "mmol N/m3"),
    "Z4p": ("Omnivorous Mesozooplankton (P)", "mmol P/m3"),
    "Z5c": ("Microzooplankton (C)", "mg C/m3"),
    "Z5n": ("Microzooplankton (N)", "mmol N/m3"),
    "Z5p": ("Microzooplankton (P)", "mmol P/m3"),
    "Z6c": ("Heterotrophic Nanoflagellates (HNAN) (C)", "mg C/m3"),
    "Z6n": ("Heterotrophic Nanoflagellates (HNAN) (N)", "mmol N/m3"),
    "Z6p": ("Heterotrophic Nanoflagellates (HNAN) (P)", "mmol P/m3"),
    "R1c": ("Labile Dissolved Organic Matter (C)", "mg C/m3"),
    "R1n": ("Labile Dissolved Organic Matter (N)", "mmol N/m3"),
    "R1p": ("Labile Dissolved Organic Matter (P)", "mmol P/m3"),
    "R2c": ("Semi-labile Dissolved Organic Carbon", "mg C/m3"),
    "R3c": ("Semi-refractory Dissolved Organic Carbon", "mg C/m3"),
    "R6c": ("Particulate Organic Matter (C)", "mg C/m3"),
    "R6n": ("Particulate Organic Matter (N)", "mmol N/m3"),
    "R6p": ("Particulate Organic Matter (P)", "mmol P/m3"),
    "R6s": ("Particulate Organic Matter (Si)", "mmol Si/m3"),
    "O3c": ("Dissolved Inorganic Carbon (C)", "mg C/m3"),
    "O3h": ("Dissolved Inorganic Carbon (eq)", "mmol eq/m3"),
    "O5c": ("Calcite (C)", "mg C/m3")
}

print("The chemical species contained in the BFM model are: ")
for code, (description, unit) in sorted(bfm_variables.items()):
    print(f"{code:5} {description}")

while True:
    bfm2plot = input("Enter the chemical species you'd like to plot (e.g., 'Chl', 'N3n', 'P1c'): ").strip()
    if bfm2plot in bfm_variables:
        # Retrieve the description and unit
        description, unit = bfm_variables[bfm2plot]
        print(f"✅ Selected: {bfm2plot} - {description} [{unit}]")
        
        # Store them in a variable if you need them for further use
        selected_description = description
        selected_unit = str(unit)
        
        break
    else:
        print(f"⚠️  '{bfm2plot}' is not a valid BFM variable code. Please try again.\n")

print('-'*45)

# ----- READ AND PLOTS THE VARIABLE -----
timestamp = datetime.now().strftime("run_%Y-%m-%d")
output_path = os.path.join(BDIR, "OUTPUT", "PLOTS", "BFM", bfm2plot)
os.makedirs(output_path, exist_ok=True)

print(f"Beginning the retrieval of the {selected_description} data...")
     
read_bfm(MDIR, Ybeg, Yend, ysec, bfm2plot, mask3d, Bmost, output_path, selected_unit, selected_description) 

print(f"The {selected_description} monthly mean values for the whole dataset have been plotted!")   
print('*'*45) 