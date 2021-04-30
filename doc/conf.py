# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import matplotlib
matplotlib.use('agg')

curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, '..')))
sys.path.append(os.path.abspath(os.path.join(curdir, '..', 'meegkit')))

import meegkit # noqa

# -- Project information -----------------------------------------------------

project = f'MEEGkit v{meegkit.__version__}'
copyright = '2021, Nicolas Barascud'
author = 'Nicolas Barascud'
release = meegkit.__version__
version = meegkit.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'numpydoc',
    'jupyter_sphinx',
    'sphinx_gallery.gen_gallery',
    'sphinxemoji.sphinxemoji',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'config.py']

# generate autosummary even if no references
# autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
    'exclude-members': '__weakref__'
}
numpydoc_show_class_members = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_theme_options = {
    "show_toc_level": 1,
    "external_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nbara/python-meegkit",
        }
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nbara/python-meegkit",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/lebababa",
            "icon": "fab fa-twitter-square",
        },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "nbara",
    "github_repo": "python-meegkit",
    "github_version": "master",
    "doc_path": "doc",
}

# -- Options for Sphinx-gallery HTML ------------------------------------------

sphinx_gallery_conf = {
    'doc_module': ('meegkit',),
    'examples_dirs': '../examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'filename_pattern': '/example_',
    'ignore_pattern': 'config.py',
    'run_stale_examples': False,
}
