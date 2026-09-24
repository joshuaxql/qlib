"""Check the MinGW C kernels against independent NumPy/pandas calculations.

Build first: python scripts/build_rolling.py
"""

import ctypes
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.data._libs import expanding, rolling
from qlib.data.ops import OPERATORS


FUNCTIONS = {
    "Mean": rolling.rolling_mean,
    "Slope": rolling.rolling_slope,
    "Rsquare": rolling.rolling_rsquare,
    "Resi": rolling.rolling_resi,
}


def reference(values, window, kind):
    values = np.asarray(values, dtype=np.float64)
    if kind == "Mean":
        return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()
    result = np.full(len(values), np.nan)
    for end in range(len(values)):
        y = values[max(0, end + 1 - window):end + 1]
        x = np.arange(len(y), dtype=float)
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue
        design = np.column_stack([x[valid], np.ones(valid.sum())])
        slope, intercept = np.linalg.lstsq(design, y[valid], rcond=None)[0]
        if kind == "Slope":
            result[end] = slope
        elif kind == "Resi":
            result[end] = y[-1] - (slope * x[-1] + intercept)
        else:
            variance = np.square(y[valid] - y[valid].mean()).sum()
            if variance > 0:
                error = y[valid] - (slope * x[valid] + intercept)
                result[end] = 1 - np.square(error).sum() / variance
    return result


@unittest.skipUnless(rolling.is_available(), "Build the DLL: python scripts/build_rolling.py")
class RollingTest(unittest.TestCase):
    def test_random_windows_against_independent_fit(self):
        random = np.random.default_rng(2026)
        values = random.normal(size=700)
        values[random.random(len(values)) < 0.25] = np.nan
        values[200:240] = np.nan
        for window in (1, 2, 5, 20, 1000):
            for kind, function in FUNCTIONS.items():
                with self.subTest(window=window, kind=kind):
                    np.testing.assert_allclose(
                        function(values, window), reference(values, window, kind),
                        rtol=1e-7, atol=1e-8, equal_nan=True,
                    )

    def test_degenerate_and_partial_windows(self):
        for values in ([], [np.nan], [3.0], [3.0] * 15, [np.nan] * 15,
                       [np.nan, 1, np.nan, 5, 7, np.nan, np.nan, np.nan, 2, 4]):
            values = np.array(values, dtype=float)
            for window in (1, 3, 30):
                for kind, function in FUNCTIONS.items():
                    with self.subTest(values=values, window=window, kind=kind):
                        np.testing.assert_allclose(
                            function(values, window), reference(values, window, kind),
                            atol=1e-12, equal_nan=True,
                        )

    def test_missing_positions_are_not_compressed(self):
        values = np.array([1, np.nan, 5, 7, np.nan], dtype=float)
        np.testing.assert_allclose(rolling.rolling_slope(values, 5),
                                   [np.nan, np.nan, 2, 2, 2], equal_nan=True)
        np.testing.assert_allclose(rolling.rolling_resi(values, 5),
                                   [np.nan, np.nan, 0, 0, np.nan], atol=1e-12, equal_nan=True)

    def test_input_layout_dtype_and_ownership(self):
        unaligned = np.ndarray((12,), dtype=np.float64, buffer=bytearray(97), offset=1)
        unaligned[:] = np.arange(12)
        for values in (np.arange(12, dtype=np.float32), np.arange(24)[::2],
                       np.arange(12)[::-1], np.arange(12, dtype=">f8"), unaligned):
            before = values.copy()
            values.flags.writeable = False
            for kind, function in FUNCTIONS.items():
                result = function(values, np.int64(4))
                self.assertEqual(result.dtype, np.dtype("float64"))
                self.assertFalse(np.shares_memory(result, values))
                np.testing.assert_allclose(result, reference(values, 4, kind),
                                           atol=1e-12, equal_nan=True)
            np.testing.assert_array_equal(values, before)

    def test_invalid_arguments(self):
        for function in FUNCTIONS.values():
            for window in (0, -1, 2.5, True, np.bool_(True), "3"):
                with self.assertRaises(ValueError):
                    function([1, 2, 3], window)
            for values in (1, [[1, 2], [3, 4]]):
                with self.assertRaises(ValueError):
                    function(values, 3)
            with self.assertRaises(OverflowError):
                function([1], 2**100)

    def test_c_argument_validation(self):
        library = ctypes.CDLL(str(rolling.LIBRARY_PATH))
        for kind in FUNCTIONS:
            function = getattr(library, f"qlib_rolling_{kind.lower()}")
            function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self.assertEqual(function(None, 0, 1, None), 0)
            self.assertEqual(function(None, 0, 0, None), 1)
            self.assertEqual(function(None, 1, 1, None), 1)

    def test_expression_integration_and_expanding(self):
        series = pd.Series([np.nan, 2, 5, np.inf, 4, -np.inf, 7, 9],
                           index=pd.date_range("2026-01-01", periods=8), name="close")
        for kind in FUNCTIONS:
            for window in (0, 1, 4, 20):
                with self.subTest(kind=kind, window=window):
                    actual = OPERATORS[kind](series, window)
                    with patch.object(rolling, "is_available", return_value=False), \
                            patch.object(expanding, "is_available", return_value=False):
                        expected = OPERATORS[kind](series, window)
                    pd.testing.assert_series_equal(actual, expected, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
