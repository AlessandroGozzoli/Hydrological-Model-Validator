#!/usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__author__ = ['Alessandro Gozzoli']
__email__ = ['alessandro.gozzoli4@studio.unibo.it']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='Hydrological_model_validator',
    version='4.10.4',
    description='Tools for the analysis and validation of Bio-Geo-Hydrological simulations and other climatological data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alessandro Gozzoli',
    author_email='alessandro.gozzoli4@studio.unibo.it',
    url='https://github.com/AlessandroGozzoli/Hydrological-Model-Validator',
    download_url='https://github.com/AlessandroGozzoli/Hydrological-Model-Validator',
    packages=find_packages(include=["Hydrological_model_validator", 
                                    "Hydrological_model_validator.*", 
                                    "Test_cases", "Test_cases.*"]),
    include_package_data=True,
    package_data={
        'Hydrological_model_validator': ['Processing/*.m'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'matplotlib',
        'scikit-learn',
        'statsmodels',
        'netCDF4',
        'SkillMetrics',  
        'gsw',
        'cmocean',
        'seaborn',
        'plotly',
        'cartopy',
        'cryptography',
        'scipy',
        'dask',
        'eliot',
        'eliot-tree',
        'reportlab',
        'rasterio',
    ],
    python_requires='>=3.7',
    platforms='any',
    classifiers=[
      'Natural Language :: English',
      'Operating System :: MacOS :: MacOS X',
      'Operating System :: POSIX',
      'Operating System :: POSIX :: Linux',
      'Operating System :: Microsoft :: Windows',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Programming Language :: Python :: 3.9',
      'Programming Language :: Python :: 3.10',
      'Programming Language :: Python :: 3.11',
      'Programming Language :: Python :: Implementation :: CPython',
      'Programming Language :: Python :: Implementation :: PyPy'
    ],
    entry_points={
        "console_scripts": [
            "sst-analyze = Test_cases.SST_data_analyzer:main",
            "chl-analyze = Test_cases.CHL_data_analyzer:main",
            "bfm-analyze = Test_cases.Benthic_Layer:main",
            "data-setupper = Test_cases.Data_reader_setupper:main",
            "GenerateReport = Hydrological_model_validator.__main__:main",
            ]
        }
)