# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'interpret-community'
copyright = '2020, Microsoft'
author = 'Microsoft'

with open(os.path.join('..', 'interpret_community', 'version.py')) as f:
    code = compile(f.read(), f.name, 'exec')
    exec(code)

# The full version, including alpha/beta/rc tags
release = version  # noqa: F821


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks'
]
# eventually we may be able to use intersphinx for these! TODO
autodoc_mock_imports = ['interpret', 'interpret.utils', 'ml_wrappers']

# enable links to objects in the other standard libraries, e.g., list and str in the Python standard library
intersphinx_mapping = {
    'Python': ('https://docs.python.org/3', None),
    'NumPy': ('https://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'SciPy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'shap': ('https://shap.readthedocs.io/en/stable', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# http://www.sphinx-doc.org/en/master/usage/configuration.html
# master doctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [os.path.abspath(os.path.join('.', '_static'))]

# Include images in the documentation
imgpath = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'img'))

html_extra_path = [imgpath]


def run_apidoc(_):
    package_path = os.path.join('..', 'interpret_community')
    api_reference_path = os.path.join('.', 'api_reference')
    argv = ["-f", "-T", "-e", "-M", "-o", api_reference_path, package_path]

    from sphinx.ext import apidoc
    apidoc.main(argv)


def setup(app):
    app.connect('builder-inited', run_apidoc)
    app.add_css_file('theme_overrides.css')
