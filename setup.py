from setuptools import setup, find_packages

setup(
    name='Hydrological_model_validator',
    version='3.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'matplotlib',
        'scikit-learn',
        'statsmodels',
        'netCDF4',
        'SkillMetrics'
        # Add any other dependencies you use
    ],
    python_requires='>=3.7',
)