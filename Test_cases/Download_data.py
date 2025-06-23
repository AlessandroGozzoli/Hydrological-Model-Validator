#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import zipfile
import gdown

def download_and_extract():
    file_id = "https://drive.google.com/file/d/1uSfCRFar3n_GwFOpQ0htNzyGid69zkjt/view?usp=drive_link"
    zip_filename = "Data.zip"
    extracted_folder = "Data"

    # Only download/unzip if folder doesn't already exist
    if not os.path.exists(extracted_folder):
        print("Test data not found. Downloading...")

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_filename, quiet=False)

        print("Extracting zip file...")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(".")  # extract into current directory

        print("Extraction complete.")

        # Optional: clean up zip file after extracting
        os.remove(zip_filename)
        print("Removed zip file.")
    else:
        print("Test data already present. Skipping download.")

if __name__ == "__main__":
    download_and_extract()