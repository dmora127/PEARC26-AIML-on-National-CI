    # Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "AI/ML Workflows on the National CI"
copyright = "2026, PEARC26 AI/ML Workflows on the National CI Tutorial Team"
author = "PEARC26 Tutorial Team"

release = "0.1"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_toolbox.collapse"
]

# Show .. todo:: notes in the rendered docs while the guide is being authored.
# Set to False (or override on Read the Docs) for a clean published build.
todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# Only the landing page and the setup section are published for now; the
# tutorial content stays hidden until it is ready for release.
exclude_patterns = [
    "introduction.rst",
    "part1-jetstream2/*",
    "part2-anvil/*",
    "part3-ospool/*",
    "part4-orchestrating-across-ci/*",
    "wrap-up.rst",
    "resources.rst",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "AI/ML Workflows on the National CI"

html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# Custom CSS — widens the content area (the RTD theme caps it at 800px).
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# -- Options for EPUB output -------------------------------------------------

epub_show_urls = "footnote"

# -- Options for Display output ------------------------------------------------
show_authors = True

