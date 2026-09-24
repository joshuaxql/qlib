"""NumPy interface to the pure C expanding kernels (no Python/NumPy C ABI).

Build from the project root: python scripts/build_rolling.py --only expanding
The four functions return float64 arrays.
Only NaN is treated as missing by the low-level API.
"""

import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np


LIBRARY_PATH = Path(__file__).with_name("expanding.dll")
__all__ = ["expanding_mean", "expanding_slope", "expanding_rsquare", "expanding_resi"]


def is_available():
    """Whether the native library has been built in this package."""
    return LIBRARY_PATH.is_file()


@lru_cache(maxsize=1)
def _library():
    if not is_available():
        raise ImportError(
            "C expanding library is missing; run python scripts/build_rolling.py --only expanding with MinGW-w64"
        )
    library = ctypes.CDLL(str(LIBRARY_PATH))
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    for name in __all__:
        function = getattr(library, f"qlib_{name}")
        function.argtypes = [array, ctypes.c_size_t, array]
        function.restype = ctypes.c_int
    return library


def _expanding(name, a):
    values = np.asarray(a, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("expanding input must be one-dimensional")
    values = np.require(values, dtype=np.float64, requirements=["C", "A"])
    result = np.empty(values.size, dtype=np.float64)
    status = getattr(_library(), f"qlib_{name}")(values, values.size, result)
    if status != 0:
        raise ValueError(f"{name}: invalid C expanding arguments (status={status})")
    return result


def expanding_mean(a):
    """Cumulative mean, ignoring NaN; an empty valid prefix returns NaN."""
    return _expanding("expanding_mean", a)


def expanding_slope(a):
    """Cumulative OLS slope against time; requires two valid observations."""
    return _expanding("expanding_slope", a)


def expanding_rsquare(a):
    """Cumulative OLS R squared; undefined fits and zero variance return NaN."""
    return _expanding("expanding_rsquare", a)


def expanding_resi(a):
    """Current observation minus its cumulative OLS fit (NaN if current is NaN)."""
    return _expanding("expanding_resi", a)
