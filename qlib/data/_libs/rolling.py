"""NumPy interface to the pure C rolling kernels (no Python/NumPy C ABI).

Build from the project root: python scripts/build_rolling.py
The four functions return float64 arrays.
Only NaN is treated as missing by the low-level API.
"""

import ctypes
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np


LIBRARY_PATH = Path(__file__).with_name("rolling.dll")
__all__ = ["rolling_mean", "rolling_slope", "rolling_rsquare", "rolling_resi"]


def is_available():
    """Whether the native library has been built in this package."""
    return LIBRARY_PATH.is_file()


@lru_cache(maxsize=1)
def _library():
    if not is_available():
        raise ImportError("C rolling library is missing; run python scripts/build_rolling.py with MinGW-w64")
    library = ctypes.CDLL(str(LIBRARY_PATH))
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    for name in __all__:
        function = getattr(library, f"qlib_{name}")
        function.argtypes = [array, ctypes.c_size_t, ctypes.c_size_t, array]
        function.restype = ctypes.c_int
    return library


def _rolling(name, a, window):
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be an integer >= 1")
    if window > sys.maxsize:
        raise OverflowError("window is too large")
    values = np.asarray(a, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rolling input must be one-dimensional")
    values = np.require(values, dtype=np.float64, requirements=["C", "A"])
    result = np.empty(values.size, dtype=np.float64)
    status = getattr(_library(), f"qlib_{name}")(values, values.size, int(window), result)
    if status != 0:
        raise ValueError(f"{name}: invalid C rolling arguments (status={status})")
    return result


def rolling_mean(a, window):
    """Rolling mean, ignoring NaN; an empty valid window returns NaN."""
    return _rolling("rolling_mean", a, window)


def rolling_slope(a, window):
    """OLS slope against time positions; requires two valid observations."""
    return _rolling("rolling_slope", a, window)


def rolling_rsquare(a, window):
    """OLS R squared; undefined fits and zero variance return NaN."""
    return _rolling("rolling_rsquare", a, window)


def rolling_resi(a, window):
    """Last observation minus its OLS fitted value (NaN if last is NaN)."""
    return _rolling("rolling_resi", a, window)
