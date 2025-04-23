import scipy.io
import xarray as xr
import os

def save_satellite_data(output_path, Truedays, Slon, Slat, Schl_complete):

    os.chdir(output_path)

    print("Saving the data in the folder...")
    
    data = {
        'Truedays': Truedays,
        'Slon': Slon,
        'Slat': Slat,
        'Schl_complete': Schl_complete
    }

    # File format input
    print("Choose a file format to save the data:")
    print("1. MAT-File (.mat)")
    print("2. NetCDF (.nc)")
    print("3. Both MAT and NetCDF")
    choice = input("Enter the number corresponding to your choice: ").strip()
    print('-'*45)

    if choice == '1' or choice == '3':
        print("Saving data as a single .mat file...")
        scipy.io.savemat("chl_clean.mat", data)
        print("Data saved as chl_clean.mat")
        print("-"*45)

    if choice == '2' or choice == '3':

        print("Saving the data as separate .nc files...")
        
        xr.DataArray(Slon).to_netcdf("Slon.nc")
        print("Slon saved as Slon.nc")

        xr.DataArray(Slat).to_netcdf("Slat.nc")
        print("Slat saved as Slat.nc")

        xr.DataArray(Schl_complete).to_netcdf("Schl_complete.nc")
        print("Schl_complete saved as Schl_complete.nc")
        print("-"*45)

    else:
        print("Invalid choice. Please run the script again and select a valid option.")

    print("✅ The requested clean data has been saved!")
    print("*" * 45)
