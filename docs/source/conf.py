    # Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "AI/ML Workflows on the National CI"
copyright = "2026, PEARC26 Tutorial Team"
author = "PEARC26 Tutorial Team"

release = "0.1"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
]

# Show .. todo:: notes in the rendered docs while the guide is being authored.
# Set to False (or override on Read the Docs) for a clean published build.
todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "AI/ML Workflows on the National CI"

html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# -- Options for EPUB output -------------------------------------------------

epub_show_urls = "footnote"
