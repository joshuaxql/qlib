"""Sphinx/Read the Docs configuration; never initializes or downloads data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

from qlib import __version__

project = "Qlib"
author = "Qlib contributors"
copyright = "2026, Qlib contributors"
release = __version__
version = release
language = "zh_CN"
root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "api_reference",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_preserve_defaults = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
add_module_names = True
python_use_unqualified_type_names = True

html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release} 文档"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "style_external_links": True,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
