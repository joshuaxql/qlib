"""Locate ABI-independent C libraries beside their Python wrappers."""

from pathlib import Path
import sys


def library_path(name):
    """Return the bundled shared library for this operating system."""
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    # Python imports *.so extension modules before *.py wrappers on Linux.
    # The lib prefix keeps these ctypes libraries out of module resolution.
    prefix = "lib" if sys.platform.startswith("linux") else ""
    return Path(__file__).with_name(f"{prefix}{name}{suffix}")
