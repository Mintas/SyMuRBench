"""Configuration file for the Sphinx documentation builder."""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib

symurbench_src = os.path.abspath("../../")  # noqa: PTH100
sys.path.insert(0, symurbench_src)


with (Path(__file__).parent.parent.parent / "pyproject.toml").open("rb") as f:
    pyproject_toml = tomllib.load(f)

project = pyproject_toml["project"]["name"]  #'SyMuRBench'
project_copyright = "2024"
author = pyproject_toml["project"]["authors"][0]["name"]
release = pyproject_toml["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_markdown_builder",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = [] #["_static"]
