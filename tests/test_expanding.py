"""Check cumulative C statistics against independent prefix calculations.

Build first: python scripts/build_rolling.py --only expanding
"""

import ctypes
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.data._libs import expanding
from qlib.data.ops import OPERATORS


FUNCTIONS = {
    "Mean": expanding.expanding_mean,
    "Slope": expanding.expanding_slope,
    "Rsquare": expanding.expanding_rsquare,
    "Resi": expanding.expanding_resi,
}


def reference(values, kind):
    values = np.asarray(values, dtype=np.float64)
    if kind == "Mean":
        return pd.Series(values).expanding(min_periods=1).mean().to_numpy()
    result = np.full(len(values), np.nan)
    for end in range(len(values)):
        y = values[:end + 1]
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
        elif not np.all(y[valid] == y[valid][0]):
            variance = np.square(y[valid] - y[valid].mean()).sum()
            error = y[valid] - (slope * x[valid] + intercept)
            result[end] = 1 - np.square(error).sum() / variance
    return result


@unittest.skipUnless(expanding.is_available(), "Build the DLL: python scripts/build_rolling.py --only expanding")
class ExpandingTest(unittest.TestCase):
    def test_random_prefixes_against_independent_fit(self):
        random = np.random.default_rng(2026)
        values = random.normal(size=700)
        values[random.random(len(values)) < 0.25] = np.nan
        values[:5] = np.nan
        values[200:240] = np.nan
        for kind, function in FUNCTIONS.items():
            with self.subTest(kind=kind):
                np.testing.assert_allclose(function(values), reference(values, kind),
                                           rtol=1e-9, atol=1e-10, equal_nan=True)

    def test_degenerate_prefixes(self):
        for values in ([], [np.nan], [3.0], [np.nan, 2, np.nan], [0.1] * 20,
                       [np.nan] * 20, [np.nan, 3, np.nan, 3, 3, np.nan]):
            for kind, function in FUNCTIONS.items():
                with self.subTest(values=values, kind=kind):
                    np.testing.assert_allclose(function(values), reference(values, kind),
                                               atol=1e-12, equal_nan=True)

    def test_missing_positions_and_prefix_invariance(self):
        values = np.array([np.nan, 3, np.nan, 7, 9, np.nan, 13, 15], dtype=float)
        np.testing.assert_allclose(expanding.expanding_slope(values),
                                   [np.nan, np.nan, np.nan, 2, 2, 2, 2, 2], equal_nan=True)
        np.testing.assert_allclose(expanding.expanding_resi(values),
                                   [np.nan, np.nan, np.nan, 0, 0, np.nan, 0, 0],
                                   atol=1e-12, equal_nan=True)
        for function in FUNCTIONS.values():
            full = function(values)
            for end in (0, 1, 4, 7):
                np.testing.assert_array_equal(function(values[:end]), full[:end])

    def test_large_offset_linear_trend(self):
        values = 1e12 + 0.25 * np.arange(10000, dtype=float)
        self.assertTrue(np.isnan(expanding.expanding_slope(values)[0]))
        np.testing.assert_allclose(expanding.expanding_slope(values)[1:], 0.25, atol=1e-12)
        np.testing.assert_allclose(expanding.expanding_rsquare(values)[1:], 1, atol=1e-12)
        np.testing.assert_allclose(expanding.expanding_resi(values)[1:], 0, atol=1e-12)

    def test_input_layout_dtype_and_ownership(self):
        unaligned = np.ndarray((12,), dtype=np.float64, buffer=bytearray(97), offset=1)
        unaligned[:] = np.arange(12)
        for values in (np.arange(12, dtype=np.float32), np.arange(24)[::2],
                       np.arange(12)[::-1], np.arange(12, dtype=">f8"), unaligned):
            before = values.copy()
            values.flags.writeable = False
            for kind, function in FUNCTIONS.items():
                result = function(values)
                self.assertEqual(result.dtype, np.dtype("float64"))
                self.assertFalse(np.shares_memory(result, values))
                np.testing.assert_allclose(result, reference(values, kind), atol=1e-12, equal_nan=True)
            np.testing.assert_array_equal(values, before)

    def test_invalid_arguments(self):
        for function in FUNCTIONS.values():
            for values in (1, [[1, 2], [3, 4]], ["not a number"]):
                with self.assertRaises(ValueError):
                    function(values)

    def test_c_argument_validation(self):
        library = ctypes.CDLL(str(expanding.LIBRARY_PATH))
        value = ctypes.c_double(1)
        for kind in FUNCTIONS:
            function = getattr(library, f"qlib_expanding_{kind.lower()}")
            function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self.assertEqual(function(None, 0, None), 0)
            self.assertEqual(function(None, 1, None), 1)
            self.assertEqual(function(ctypes.byref(value), 1, None), 1)
            self.assertEqual(function(None, 1, ctypes.byref(value)), 1)

    def test_expression_dispatch_and_fallback(self):
        series = pd.Series([np.nan, 2, 5, np.inf, 4, -np.inf, 7, 9],
                           index=pd.date_range("2026-01-01", periods=8), name="close")
        for kind, function in FUNCTIONS.items():
            with self.subTest(kind=kind):
                with patch.object(expanding, function.__name__, wraps=function) as native:
                    actual = OPERATORS[kind](series, 0)
                    native.assert_called_once()
                with patch.object(expanding, "is_available", return_value=False):
                    expected = OPERATORS[kind](series, 0)
                pd.testing.assert_series_equal(actual, expected, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
