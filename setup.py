from setuptools import setup, find_packages

setup(
    name='Hydrological_model_validator',
    version='4.0.0',
    packages=find_packages(),
    include_package_data=True,  # Include package data files specified in package_data or MANIFEST.in
    package_data={
        # Assuming 'Hydrological_model_validator' is your main package directory name
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
        'SkillMetrics',   # note lowercase per PyPI
        'gsw',
        'cmocean',
        'seaborn',
        'plotly',
        'cartopy',
        'cryptography',
        'scipy'
        # add any additional dependencies you might identify
    ],
    python_requires='>=3.7',
)