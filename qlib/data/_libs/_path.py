"""Locate ABI-independent C libraries beside their Python wrappers."""

from pathlib import Path
import sys


def library_path(name):
    """Return the bundled shared library for this operating system."""
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    return Path(__file__).with_name(f"{name}{suffix}")
