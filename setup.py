from setuptools import setup, find_packages

setup(
    name='Hydrological_model_validator',
    version='4.0.0',
    packages=find_packages(),
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