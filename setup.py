from setuptools import setup, find_packages

setup(
    name='Hydrological_model_validator',
    version='4.7.1',
    packages=find_packages(),
    include_package_data=True,  # Include package data files specified in package_data or MANIFEST.in
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
        'dask'
    ],
    python_requires='>=3.7',
)