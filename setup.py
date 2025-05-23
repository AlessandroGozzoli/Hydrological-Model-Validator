from setuptools import setup, find_packages

setup(
    name='Hydrological_model_validator',
    version='4.0.0-δ',
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