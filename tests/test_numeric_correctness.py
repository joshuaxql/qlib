"""Independent Decimal oracles for expression and native numerical regressions."""

from contextlib import contextmanager
from decimal import Decimal, localcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.data._libs import expanding, rolling
from qlib.data.base import ExpressionEngine
from qlib.data.ops import OPERATORS


def reference(values, window, kind, left=None):
    """Recompute each window with exact float inputs and 400 decimal digits."""
    result = np.full(len(values), np.nan)
    with localcontext() as context:
        context.prec = 400
        for end in range(len(values)):
            start = 0 if window == 0 else max(0, end + 1 - window)
            positions = [i for i in range(start, end + 1)
                         if np.isfinite(values[i]) and (left is None or np.isfinite(left[i]))]
            if not positions:
                continue
            y = [Decimal.from_float(float(values[i])) for i in positions]
            n = Decimal(len(y))
            mean_y = sum(y) / n
            if kind == "Mean":
                result[end] = float(mean_y)
                continue
            if kind == "WMA":
                weights = [Decimal(i - start + 1) for i in positions]
                result[end] = float(sum(a * b for a, b in zip(y, weights)) / sum(weights))
                continue
            if len(y) < 2:
                continue
            x = [Decimal(i - start) if left is None else Decimal.from_float(float(left[i]))
                 for i in positions]
            mean_x = sum(x) / n
            xx = sum((a - mean_x) ** 2 for a in x)
            yy = sum((a - mean_y) ** 2 for a in y)
            xy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
            if kind == "Cov":
                result[end] = float(xy / (n - 1))
            elif xx:
                if kind == "Slope":
                    result[end] = float(xy / xx)
                elif kind == "Resi" and np.isfinite(values[end]):
                    current = Decimal.from_float(float(values[end]))
                    result[end] = float(current - mean_y - xy / xx * (Decimal(end - start) - mean_x))
                elif kind in ("Corr", "Rsquare") and yy:
                    value = xy / xx.sqrt() / yy.sqrt() if kind == "Corr" else xy * xy / (xx * yy)
                    result[end] = float(value)
    return result


@contextmanager
def fallback():
    with patch.object(rolling, "is_available", return_value=False), \
            patch.object(expanding, "is_available", return_value=False):
        yield


class NumericCorrectnessTest(unittest.TestCase):
    def engine(self):
        dates = pd.date_range("2024-01-01", periods=4, name="datetime")

        class Provider:
            def _field(self, instrument, field):
                return pd.Series([1.0, 2.0, 3.0, 4.0], index=dates, name=field)

        return ExpressionEngine(Provider(), "TEST", dates, allow_future=False)

    def evaluate_operator(self, kind, values, n, left=None):
        series = pd.Series(values, name="value")
        return (OPERATORS[kind](series, n) if left is None else
                OPERATORS[kind](pd.Series(left), series, n)).to_numpy()

    def test_if_preserves_index_and_composes(self):
        for index in (pd.date_range("2024-01-01", periods=4), pd.Index([20231, 20232, 20233, 20234])):
            values = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="close")
            conditional = OPERATORS["If"](values > 1, values, 0)
            pd.testing.assert_series_equal(conditional, pd.Series([0.0, 2.0, 3.0, 4.0], index=index, name="close"))
            pd.testing.assert_series_equal(OPERATORS["Ref"](conditional, 1), conditional.shift(1))
        engine = self.engine()
        with fallback():
            np.testing.assert_allclose(engine.evaluate("Mean(If($close > 1, $close, 0), 2)"), [0, 1, 2.5, 3.5])
        np.testing.assert_allclose(engine.evaluate("EMA(If($close > 1, $close, 0), 2)"), [0, 1.5, 33 / 13, 3.525])
        np.testing.assert_allclose(engine.evaluate("Ref(If($close > 1, $close, 0), 1)"),
                                   [np.nan, 0, 2, 3], equal_nan=True)

    def test_named_not_matches_unary_for_scalars_and_shifted_conditions(self):
        engine = self.engine()
        for named, unary in (("Not(True)", "~True"), ("Not(False)", "~False"),
                             ("Not(Ref($close > 1, 1))", "~Ref($close > 1, 1)")):
            pd.testing.assert_series_equal(engine.evaluate(named), engine.evaluate(unary))
        np.testing.assert_array_equal(engine.evaluate("If(Not(True), $close, 0)"), [0, 0, 0, 0])
        np.testing.assert_array_equal(engine.evaluate("Not(Ref($close > 1, 1))"), [False, True, False, False])

    def test_large_baseline_statistics_against_decimal(self):
        values = 1e8 + np.arange(30, dtype=float)
        left = values.copy()
        for n in (0, 3, 20, 100):
            for kind in ("Mean", "Slope", "Rsquare", "Resi", "Corr", "Cov"):
                pair = left if kind in ("Corr", "Cov") else None
                expected = reference(values, n, kind, pair)
                with self.subTest(n=n, kind=kind), fallback():
                    np.testing.assert_allclose(self.evaluate_operator(kind, values, n, pair), expected,
                                               rtol=1e-10, atol=1e-10, equal_nan=True)
        float32_values = (1e8 + 8 * np.arange(100)).astype(np.float32)
        with fallback():
            np.testing.assert_allclose(self.evaluate_operator("Rsquare", float32_values, 20)[1:], 1, atol=1e-12)

    def test_large_value_eviction_restores_current_window(self):
        for outlier in (1e16, 1e308, -1e308):
            values = np.r_[outlier, np.ones(20)]
            for kind, expected in (("Mean", 1), ("Slope", 0), ("Rsquare", np.nan), ("Resi", 0)):
                with self.subTest(outlier=outlier, kind=kind), fallback():
                    np.testing.assert_allclose(self.evaluate_operator(kind, values, 3)[3:], expected,
                                               atol=1e-12, equal_nan=True)

    def test_nonlinear_fits_keep_precision_under_large_translations(self):
        # Irregular increments force fractional centered means. A stored
        # absolute mean would lose these at 1e12/1e15, even after binary scaling.
        increments = np.array([0.0, 1.0, 4.0, 10.0])
        left_increments = np.array([0.0, 1.0, 2.0, 4.0])
        modes = [False, True] if rolling.is_available() and expanding.is_available() else [False]
        for native in modes:
            for baseline in (0.0, 1e8, 1e12, 1e15, -1e15):
                values, left = baseline + increments, baseline + left_increments
                for n in (0, 2, 3, 4):
                    for kind in ("Mean", "Slope", "Rsquare", "Resi", "Corr", "Cov"):
                        pair = left if kind in ("Corr", "Cov") else None
                        with self.subTest(native=native, baseline=baseline, n=n, kind=kind):
                            if native:
                                actual = self.evaluate_operator(kind, values, n, pair)
                            else:
                                with fallback():
                                    actual = self.evaluate_operator(kind, values, n, pair)
                            np.testing.assert_allclose(actual, reference(values, n, kind, pair),
                                                       rtol=1e-13, atol=1e-13, equal_nan=True)
                            if kind != "Mean":
                                # Translation changes no regression moment or residual.
                                base_pair = left_increments if pair is not None else None
                                np.testing.assert_allclose(actual, reference(increments, n, kind, base_pair),
                                                           rtol=1e-13, atol=1e-13, equal_nan=True)

    def test_constant_rsquare_is_undefined_in_both_window_modes(self):
        for constant in (0.1, 1e8, 1e308, 1e-300):
            values = np.full(30, constant)
            values[[2, 8]] = np.nan
            for n in (0, 3, 20):
                with self.subTest(constant=constant, n=n), fallback():
                    self.assertTrue(np.isnan(self.evaluate_operator("Rsquare", values, n)).all())

    def test_finite_extreme_means_and_weighted_means(self):
        for values in ([1e308, 1e308], [1e308, -1e308, 1e308],
                       [np.finfo(float).max] * 5, [1e308, np.nan, -1e308],
                       [0, 1e-300, 2e-300, 3e-300]):
            for n in (0, 2, 3):
                for kind in ("Mean", "WMA"):
                    with self.subTest(values=values, n=n, kind=kind), fallback():
                        actual = self.evaluate_operator(kind, values, n)
                        np.testing.assert_allclose(actual, reference(values, n, kind), rtol=2e-14, atol=0,
                                                   equal_nan=True)

    def test_pairwise_missing_and_general_fits_have_independent_oracles(self):
        random = np.random.default_rng(901)
        values = random.normal(size=90)
        left = random.normal(size=90)
        values[[0, 10, 17, 18, 70]] = np.nan
        left[[2, 8, 15, 25, 72]] = np.nan
        values[40:46] = np.nan
        for n in (0, 1, 2, 7, 30, 100):
            for kind in ("Mean", "Slope", "Rsquare", "Resi", "Corr", "Cov"):
                pair = left if kind in ("Corr", "Cov") else None
                with self.subTest(n=n, kind=kind), fallback():
                    np.testing.assert_allclose(self.evaluate_operator(kind, values, n, pair),
                                               reference(values, n, kind, pair), rtol=1e-10, atol=1e-11,
                                               equal_nan=True)

    def test_covariance_scaling_avoids_intermediate_overflow(self):
        left = np.array([1e308, -1e308, 1e308])
        values = np.array([1e-308, -1e-308, 1e-308])
        for n in (0, 2, 3):
            for kind in ("Corr", "Cov"):
                with self.subTest(n=n, kind=kind), fallback():
                    np.testing.assert_allclose(self.evaluate_operator(kind, values, n, left),
                                               reference(values, n, kind, left), rtol=1e-13, equal_nan=True)

    @unittest.skipUnless(rolling.is_available() and expanding.is_available(), "Native libraries are unavailable")
    def test_native_and_fallback_agree_including_extreme_scales(self):
        samples = [1e12 + 0.25 * np.arange(60), np.r_[1e308, np.ones(59)],
                   np.full(60, 0.1), np.arange(60) * 1e-300]
        for values in samples:
            values = values.copy()
            values[[2, 8, 21]] = np.nan
            for n in (0, 1, 3, 20, 100):
                for kind in ("Mean", "Slope", "Rsquare", "Resi", "Corr", "Cov"):
                    pair = values if kind in ("Corr", "Cov") else None
                    with self.subTest(n=n, kind=kind, first=values[0]):
                        actual = self.evaluate_operator(kind, values, n, pair)
                        with fallback():
                            expected = self.evaluate_operator(kind, values, n, pair)
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    @unittest.skipUnless(rolling.is_available(), "Native library is unavailable")
    def test_low_level_infinity_leaves_without_poisoning_finite_windows(self):
        np.testing.assert_allclose(rolling.rolling_mean([np.inf, 1, 2, 3], 2), [np.inf, np.inf, 1.5, 2.5])
        np.testing.assert_allclose(rolling.rolling_mean([np.inf, -np.inf, 2, 3], 2),
                                   [np.inf, np.nan, -np.inf, 2.5], equal_nan=True)
        for kind, expected in (("slope", 1), ("rsquare", 1), ("resi", 0)):
            result = getattr(rolling, f"rolling_{kind}")([np.inf, 1, 2, 3], 2)
            np.testing.assert_allclose(result[2:], expected, atol=1e-12)

    def test_old_native_libraries_can_fall_back_for_new_pair_kernels(self):
        for module, prefix in ((rolling, "rolling"), (expanding, "expanding")):
            library = SimpleNamespace(**{f"qlib_{prefix}_mean": lambda *args: 0})
            with patch.object(module, "_library", return_value=library), \
                    patch.object(module, "is_available", return_value=True):
                self.assertTrue(module.supports("Mean"))
                self.assertFalse(module.supports("Corr"))
                self.assertFalse(module.supports("Cov"))
                n = 2 if prefix == "rolling" else 0
                values = pd.Series([1e8, 1e8 + 1, 1e8 + 2])
                np.testing.assert_allclose(OPERATORS["Corr"](values, values, n), [np.nan, 1, 1], equal_nan=True)

    def test_pair_input_validation_and_allocation_failure(self):
        for function in (rolling.rolling_corr, rolling.rolling_cov):
            with self.assertRaises(ValueError):
                function([1, 2], [1], 2)
        for function in (expanding.expanding_corr, expanding.expanding_cov):
            with self.assertRaises(ValueError):
                function([[1, 2]], [1, 2])
        with patch.object(rolling, "_library", return_value=SimpleNamespace(qlib_rolling_mean=lambda *args: 2)):
            with self.assertRaises(MemoryError):
                rolling.rolling_mean([1, 2], 2)


if __name__ == "__main__":
    unittest.main()
