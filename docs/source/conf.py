import os
import sys

# Adjust path to Hydrological_model_validator for autodoc
sys.path.insert(0, os.path.abspath('../..'))

project = 'Hydrological Model Validator'
author = 'Alessandro Gozzoli'
release = '4.10.2'

extensions = [
    'myst_parser',        # For Markdown support
    'sphinx.ext.autodoc', # For API docs
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Enable MyST parser extensions for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
]

master_doc = 'index'

html_theme = "sphinx_book_theme"
html_static_path = ['_static']