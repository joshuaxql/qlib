"""Price adjustment is applied to source fields, before expression evaluation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

import qlib
from qlib.data import D, LocalProvider
from qlib.data.filter import ExpressionFilter


A, B = "000001.SZ", "600000.SH"
PRICES = ("open", "high", "low", "close", "vwap", "pre_close", "up_limit", "down_limit")


class AdjustmentTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dates = pd.bdate_range("2024-01-02", periods=8, name="datetime")
        for folder in ("calendars", "instruments", f"features/{A}", f"features/{B}"):
            (self.root / folder).mkdir(parents=True)
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")), encoding="utf-8")
        (self.root / "instruments/all.txt").write_text(
            f"{A} 2024-01-02 2024-01-11\n{B} 2024-01-02 2024-01-11\n", encoding="utf-8")
        self.close = np.array([10, 11, 6, np.nan, 7, 8, 5, 6], dtype=float)
        self.factors = {
            A: np.array([2, 2, 4, 4, 4, 4, 8, 8], dtype=float),
            B: np.array([5, 5, 5, 5, 10, 10, 10, 10], dtype=float),
        }
        for code in (A, B):
            for field, offset in zip(PRICES, (-0.5, 1, -1, 0, 0.25, 2, 3, -3)):
                self.write_daily(code, field, self.close + offset)
            self.write_daily(code, "factor", self.factors[code])
            for field in ("volume", "amount", "total_mv", "circ_mv", "extra_field"):
                self.write_daily(code, field, np.arange(100, 108))
        self.provider = LocalProvider(self.root)

    def write_daily(self, code, field, values):
        np.asarray([0, *values], dtype="<f4").tofile(self.root / "features" / code / f"{field}.day.bin")

    def test_default_hfq_and_all_price_fields(self):
        actual = self.provider.daily([A, B], PRICES)
        raw = self.provider.daily([A, B], PRICES, adjust="none")
        for code in (A, B):
            expected = raw.loc[code].mul(self.factors[code], axis=0)
            pd.testing.assert_frame_equal(actual.loc[code], expected)
        # The first factor is 2, so hfq must not normalize to the query start.
        self.assertEqual(actual.loc[(A, self.dates[0]), "close"], 20)

    def test_qfq_uses_each_instruments_query_end_factor(self):
        for end, anchor in ((self.dates[2], 2), ("2024-01-07", 3), (None, 7)):
            with self.subTest(end=end):
                raw = self.provider.daily([A, B], PRICES, end_time=end, adjust="none")
                actual = self.provider.daily([A, B], PRICES, end_time=end, adjust="qfq")
                for code in (A, B):
                    scale = self.factors[code][:len(raw.loc[code])] / self.factors[code][anchor]
                    pd.testing.assert_frame_equal(actual.loc[code], raw.loc[code].mul(scale, axis=0))
        early = self.provider.daily([A], "close", self.dates[2], self.dates[2], adjust="qfq")
        self.assertEqual(early.iloc[0, 0], 6)  # Future factor 8 must not enter this anchor.

    def test_adjustment_precedes_nested_rolling_and_expanding(self):
        expressions = ["$close", "Mean($close, 3)", "Mean(Ref($close, 1), 2)",
                       "$close / Ref($close, 1) - 1", "Sum($close, 0)",
                       "Slope($close, 3)", "Rsquare($close, 0)", "Resi($close, 3)",
                       "$high - $low"]
        for mode, scale in (("hfq", 1), ("qfq", 0.25)):
            for allow_future in (True, False):
                with self.subTest(mode=mode, allow_future=allow_future):
                    result = self.provider.features([A], expressions, self.dates[2], self.dates[2],
                                                    adjust=mode, allow_future=allow_future)
                    expected = [24 * scale, 22 * scale, 21 * scale, 24 / 22 - 1, 66 * scale,
                                2 * scale, 1, 0, 8 * scale]
                    np.testing.assert_allclose(result.iloc[0], expected, atol=1e-12)
                    full = self.provider.features([A], expressions, end_time=self.dates[2],
                                                  adjust=mode, allow_future=allow_future)
                    pd.testing.assert_frame_equal(result, full.iloc[-1:])
        raw = self.provider.features([A], ["Mean($close, 3)"], self.dates[2], self.dates[2], adjust="none")
        self.assertEqual(raw.iloc[0, 0], 9)

    def test_nonprice_fields_stay_raw(self):
        expressions = ["$factor", "$volume", "$amount", "$total_mv", "$circ_mv", "$extra_field"]
        raw = self.provider.features([A], expressions, adjust="none")
        for mode in ("hfq", "qfq"):
            pd.testing.assert_frame_equal(self.provider.features([A], expressions, adjust=mode), raw)

    def test_init_defaults_overrides_and_raw_cache_isolation(self):
        raw_before = self.provider._daily(A, "close").copy()
        expected = {"hfq": 20, "qfq": 2.5, "none": 10}
        for mode in ("hfq", "qfq", "none", "qfq", "hfq"):
            result = self.provider.daily([A], "close", adjust=mode)
            self.assertEqual(result.iloc[0, 0], expected[mode])
            result.iloc[0, 0] = -999
            self.assertEqual(self.provider.daily([A], "close", adjust=mode).iloc[0, 0], expected[mode])
        pd.testing.assert_series_equal(self.provider._daily(A, "close"), raw_before)
        self.assertEqual(self.provider.adjust, "hfq")
        self.assertIsNone(self.provider._adjustment_end)
        for mode in expected:
            provider = qlib.init(self.root, adjust=mode)
            self.assertEqual(D.daily([A], "close").iloc[0, 0], expected[mode])
            self.assertEqual(provider.features([A], ["$close"], adjust=None).iloc[0, 0], expected[mode])
            self.assertEqual(D.daily([A], "close", adjust="none").iloc[0, 0], 10)
            self.assertEqual(provider.adjust, mode)

    def test_filters_share_query_adjustment(self):
        pool = self.provider.instruments([A], [ExpressionFilter("$close > 15") &
                                               ExpressionFilter("Mean($close, 2) > 15")])
        for mode in ("hfq", "qfq", "none"):
            with self.subTest(mode=mode):
                actual = self.provider.daily(pool, "close", self.dates[2], self.dates[2], adjust=mode)
                self.assertEqual(len(actual), 1 if mode == "hfq" else 0)
                mask = self.provider.universe(pool, self.dates[2], self.dates[2], adjust=mode)
                self.assertEqual(mask.iloc[0, 0], mode == "hfq")
                codes = self.provider.list_instruments(pool, self.dates[2], self.dates[2],
                                                       as_list=True, adjust=mode)
                self.assertEqual(codes, [A] if mode == "hfq" else [])
        pool = self.provider.instruments([A], [ExpressionFilter("$close > 5.5")])
        actual = self.provider.features(pool, ["$close"], self.dates[2], self.dates[2], adjust="qfq")
        self.assertEqual(actual.iloc[0, 0], 6)

    def test_invalid_factors_and_last_valid_anchor(self):
        factors = [2, 2, 4, np.nan, 4, 0, np.inf, -1]
        self.write_daily(A, "factor", factors)
        for mode in ("hfq", "qfq"):
            actual = self.provider.daily([A], "close", adjust=mode).iloc[:, 0]
            expected = np.array([20, 22, 24, np.nan, 28, np.nan, np.nan, np.nan])
            if mode == "qfq":
                expected /= 4
            np.testing.assert_allclose(actual, expected, equal_nan=True)
        np.testing.assert_allclose(self.provider.daily([A], "close", adjust="none").iloc[:, 0],
                                   self.close, equal_nan=True)
        self.write_daily(A, "factor", [np.nan, np.nan, 4, 4, 4, 4, 8, 8])
        self.provider.clear_cache()
        result = self.provider.features([A], ["$close"], end_time=self.dates[1], adjust="qfq")
        self.assertTrue(result.iloc[:, 0].isna().all())

    def test_missing_factor_file_and_missing_policy(self):
        (self.root / "features" / A / "factor.day.bin").unlink()
        self.assertTrue(self.provider.daily([A], "close").iloc[:, 0].isna().all())
        self.assertEqual(self.provider.daily([A], "close", adjust="none").iloc[0, 0], 10)
        strict = LocalProvider(self.root, missing="raise")
        with self.assertRaises(FileNotFoundError):
            strict.daily([A], "close")
        self.assertEqual(strict.daily([A], "close", adjust="none").iloc[0, 0], 10)
        (self.root / "features" / B / "factor.day.bin").unlink()
        self.provider.clear_cache()
        with self.assertRaisesRegex(KeyError, "factor"):
            self.provider.daily([A], "close")
        self.assertEqual(self.provider.daily([A], "volume").iloc[0, 0], 100)

    def test_invalid_adjustment(self):
        for mode in ("bad", "", 0, False, []):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    LocalProvider(self.root, adjust=mode)
                with self.assertRaises(ValueError):
                    self.provider.daily([A], "close", adjust=mode)
                with self.assertRaises(ValueError):
                    self.provider.features([], ["$close"], adjust=mode)
                with self.assertRaises(ValueError):
                    self.provider.universe([A], adjust=mode)


if __name__ == "__main__":
    unittest.main()
