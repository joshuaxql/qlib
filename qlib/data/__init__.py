"""Data facade. Initialization never downloads data or requires a token."""

from .data import D, LocalProvider
__all__ = ["D", "LocalProvider"]
