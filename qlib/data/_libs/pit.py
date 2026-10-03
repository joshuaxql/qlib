"""Batch as-of lookup, with an optional bundled C implementation."""

import ctypes
from functools import lru_cache

import numpy as np

from ._path import library_path

LIBRARY_PATH = library_path("pit")


def is_available():
    """Whether the native library is present in this package."""
    return LIBRARY_PATH.is_file()


@lru_cache(maxsize=1)
def _library():
    if not is_available():
        raise ImportError(
            "C PIT library is missing; install a platform wheel or run python scripts/build_rolling.py --only pit"
        )
    library = ctypes.CDLL(str(LIBRARY_PATH))
    array = lambda dtype: np.ctypeslib.ndpointer(dtype=dtype, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    library.qlib_pit_asof.argtypes = [array(np.uint32), ctypes.c_size_t, array(np.uint64),
                                    array(np.uint64), ctypes.c_size_t, ctypes.c_uint32, array(np.int64)]
    library.qlib_pit_asof.restype = ctypes.c_int
    return library


def asof_indices(dates, starts, counts, asof):
    dates = np.require(dates, dtype=np.uint32, requirements=["C", "A"])
    starts = np.require(starts, dtype=np.uint64, requirements=["C", "A"])
    counts = np.require(counts, dtype=np.uint64, requirements=["C", "A"])
    if any(a.ndim != 1 for a in (dates, starts, counts)) or len(starts) != len(counts):
        raise ValueError("Invalid PIT lookup arrays")
    if np.any(starts > len(dates)) or np.any(counts > np.uint64(len(dates)) - starts):
        raise ValueError("PIT group is outside the record array")
    if not isinstance(asof, (int, np.integer)) or not 0 <= asof <= 0xFFFFFFFF:
        raise ValueError("Invalid PIT observation date")
    result = np.full(len(starts), -1, dtype=np.int64)
    if is_available():
        if _library().qlib_pit_asof(dates, len(dates), starts, counts, len(starts), int(asof), result):
            raise ValueError("Invalid C PIT lookup arguments")
    else:
        for i, (start, count) in enumerate(zip(starts, counts)):
            start, count = int(start), int(count)
            position = np.searchsorted(dates[start:start + count], asof, side="right")
            if position:
                result[i] = start + position - 1
    return result
