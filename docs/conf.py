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
import sphinx_rtd_theme
import virga 

#sys.path.insert(0, os.path.abspath('/home/nbatalh1/codes/virga'))
sys.path.insert(0, os.path.abspath('/Users/nbatalh1/Documents/codes/VIRGA/virga/'))
#/Users/nbatalh1/Documents/codes/VIRGA/virga/'))


# -- Project information -----------------------------------------------------

project = 'virga'
author = 'Batalha, Rooney, Marley'

# The full version, including alpha/beta/rc tags
version = virga.__version__
release = 'v'+virga.__version__



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'nbsphinx']

nbsphinx_allow_errors = False
nbsphinx_execute = 'always'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','**.ipynb_checkpoints']

pygments_style = 'sphinx'
todo_include_todos = True
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

htmlhelp_basename = 'virgadoc'

nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. note::  `Download full notebook here <https://github.com/natashabatalha/virga/tree/master/docs/{{ docname }}>`_
"""


#def setup(app):
#    app.add_stylesheet('https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css')
#    app.add_javascript('https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js')
#    app.add_javascript('https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js')


