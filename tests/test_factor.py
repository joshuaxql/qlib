"""Independent cross-sectional examples and provider-backed label timing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from qlib.data import LocalProvider
from qlib.contrib.report.analysis_model import analyze_factors, calculate_factors, calculate_forward_returns, factor_analysis


class FactorMetricsTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-02", periods=4)
        self.index = pd.MultiIndex.from_product([list("ABCD"), self.dates], names=["instrument", "datetime"])
        self.values = pd.DataFrame({"alpha": np.repeat([1., 2., 3., 4.], 4)}, index=self.index)
        self.labels = pd.DataFrame({1: np.array([[1, -1, 1, 1], [2, -2, 1, 4],
                                               [3, -3, 1, 9], [4, -4, 1, 16]]).ravel() / 100}, index=self.index)

    def test_ic_rank_ic_and_summary(self):
        result = analyze_factors(self.values, self.labels, quantiles=2)
        daily = result.daily.loc[("alpha", 1)]
        expected_ic = np.corrcoef([1, 2, 3, 4], [1, 4, 9, 16])[0, 1]
        np.testing.assert_allclose(daily.ic, [1, -1, np.nan, expected_ic], equal_nan=True)
        np.testing.assert_allclose(daily.rank_ic, [1, -1, np.nan, 1], equal_nan=True)
        summary = result.summary.loc[("alpha", 1)]
        self.assertEqual(summary.ic_count, 3)
        self.assertAlmostEqual(summary.rank_ic_positive_rate, 2 / 3)
        self.assertAlmostEqual(summary.rank_ic_mean, 1 / 3)
        self.assertAlmostEqual(summary.rank_ic_ir, np.mean([1, -1, 1]) / np.std([1, -1, 1], ddof=1))
        np.testing.assert_allclose(daily.long_short_return, [.01, -.01, 0, .05])
        np.testing.assert_allclose(result.turnover.turnover.dropna(), 0)
        np.testing.assert_allclose(result.autocorrelation.autocorrelation.dropna(), 1)

    def test_missing_and_infinite_coverage_no_filling(self):
        self.values.loc[("A", self.dates[0]), "alpha"] = np.inf
        self.labels.loc[("B", self.dates[0]), 1] = np.nan
        result = analyze_factors(self.values, self.labels, quantiles=2, min_samples=3)
        row = result.daily.loc[("alpha", 1, self.dates[0])]
        self.assertEqual(row.factor_count, 3)
        self.assertEqual(row.pair_count, 2)
        self.assertEqual(row.coverage, .75)
        self.assertEqual(row.pair_coverage, .5)
        self.assertTrue(np.isnan(row.ic))
        # Grouping must be independent of future-label availability.
        other = analyze_factors(self.values, self.labels * np.nan, quantiles=2)
        pd.testing.assert_frame_equal(result.quantile_membership, other.quantile_membership)

    def test_ties_constant_factor_and_small_cross_section(self):
        self.values["ties"] = np.repeat([1, 1, 3, 4], 4)
        self.values["constant"] = 1.
        result = analyze_factors(self.values, self.labels, quantiles=2)
        members = result.quantile_membership
        np.testing.assert_array_equal(members.loc["A", "ties"], members.loc["B", "ties"])
        self.assertTrue(members.constant.isna().all())
        self.assertTrue(result.daily.loc["constant"].ic.isna().all())
        # Hand-ranked Spearman, including averaged ties.
        expected = np.corrcoef([1.5, 1.5, 3, 4], [1, 2, 3, 4])[0, 1]
        self.assertAlmostEqual(result.daily.loc[("ties", 1, self.dates[0]), "rank_ic"], expected)
        small = analyze_factors(self.values[["alpha"]], self.labels, quantiles=5)
        self.assertTrue(small.quantile_membership.isna().all().all())

    def test_turnover_and_autocorrelation_use_only_factor_data(self):
        self.values.loc[(slice(None), self.dates[1]), "alpha"] = [4, 3, 2, 1]
        result = analyze_factors(self.values, self.labels, quantiles=2)
        self.assertEqual(result.turnover.loc[("alpha", self.dates[1], 2), "turnover"], 1)
        self.assertEqual(result.autocorrelation.loc[("alpha", self.dates[1]), "autocorrelation"], -1)
        lagged = analyze_factors(self.values, self.labels, quantiles=2, turnover_lag=2)
        self.assertEqual(lagged.turnover.loc[("alpha", self.dates[2], 2), "turnover"], 0)

    def test_alignment_multiple_horizons_and_export(self):
        labels = self.labels.copy()
        labels[5] = -labels[1]
        result = analyze_factors(self.values.sample(frac=1), labels.swaplevel().sort_index(), quantiles=2)
        self.assertEqual(len(result.summary), 2)
        self.assertAlmostEqual(result.summary.loc[("alpha", 5), "ic_mean"],
                               -result.summary.loc[("alpha", 1), "ic_mean"])
        with TemporaryDirectory() as temporary:
            result.save(temporary)
            self.assertEqual(len(list(Path(temporary).glob("*.csv"))), 8)
            self.assertEqual(json.loads((Path(temporary) / "config.json").read_text())["horizons"], [1, 5])

    def test_invalid_panels_and_options(self):
        for values in (self.values.reset_index(), pd.concat([self.values, self.values]), self.values.iloc[:0]):
            with self.assertRaises(ValueError):
                analyze_factors(values, self.labels)
        for kwargs in ({"quantiles": 1}, {"min_samples": 1}, {"turnover_lag": 0}, {"quantiles": True}):
            with self.assertRaises(ValueError):
                analyze_factors(self.values, self.labels, **kwargs)


class FactorProviderTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dates = pd.bdate_range("2025-01-02", periods=9)
        self.codes = ["000001.SZ", "600000.SH", "600519.SH"]
        (self.root / "calendars").mkdir()
        (self.root / "instruments").mkdir()
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        (self.root / "instruments/all.txt").write_text("\n".join(
            f"{code} {self.dates[0]:%Y-%m-%d} {self.dates[-1]:%Y-%m-%d}" for code in self.codes))
        # All stocks leave this research pool after signal day 2.
        (self.root / "instruments/sample.txt").write_text("\n".join(
            f"{code} {self.dates[0]:%Y-%m-%d} {self.dates[1]:%Y-%m-%d}" for code in self.codes))
        for i, code in enumerate(self.codes):
            directory = self.root / "features" / code
            directory.mkdir(parents=True)
            raw = np.arange(10., 19.) + i
            factors = np.ones(9)
            raw[3:] /= 2
            factors[3:] = 2
            if i == 1:
                raw[2] = np.nan
            for field, values in (("open", raw), ("close", raw + 1), ("factor", factors)):
                np.array([0, *values], dtype="<f4").tofile(directory / f"{field}.day.bin")
        self.provider = LocalProvider(self.root)

    def test_factor_future_rejected_aliases_and_warmup(self):
        with self.assertRaises(ValueError):
            calculate_factors(self.codes, {"bad": "Ref($close, -1)"}, provider=self.provider)
        values = calculate_factors(self.codes, {"a": "Mean($close, 3)", "b": "Mean($close, 3)"},
                                   self.dates[1], self.dates[1], provider=self.provider)
        self.assertEqual(values.loc[(self.codes[0], self.dates[1]), "a"], 11.5)
        np.testing.assert_array_equal(values.a, values.b)

    def test_calendar_shift_adjustment_future_tail_and_universe_exit(self):
        values = calculate_factors("sample", {"alpha": "$close"}, provider=self.provider)
        labels = calculate_forward_returns(values, [1, 2], provider=self.provider)
        self.assertAlmostEqual(labels.loc[(self.codes[0], self.dates[0]), 1], 12 / 11 - 1)
        self.assertAlmostEqual(labels.loc[(self.codes[0], self.dates[1]), 1], 13 / 12 - 1)
        # A missing t+2 price stays missing; do not shift to the next valid quote.
        self.assertTrue(np.isnan(labels.loc[(self.codes[1], self.dates[0]), 1]))
        self.assertAlmostEqual(labels.loc[(self.codes[1], self.dates[0]), 2], 14 / 12 - 1)
        raw = calculate_forward_returns(values, [1], provider=self.provider, adjust="none")
        self.assertAlmostEqual(raw.loc[(self.codes[0], self.dates[1]), 1], 6.5 / 12 - 1)
        qfq = calculate_forward_returns(values, [1, 2], provider=self.provider, adjust="qfq")
        pd.testing.assert_frame_equal(qfq, labels)

    def test_lag_zero_is_current_price_not_ref_zero(self):
        values = calculate_factors(self.codes, "$close", provider=self.provider)
        labels = calculate_forward_returns(values, [1], provider=self.provider, entry_lag=0)
        self.assertAlmostEqual(labels.loc[(self.codes[0], self.dates[1]), 1], 12 / 11 - 1)
        self.assertTrue(labels.xs(self.dates[-1], level="datetime").isna().all().all())

    def test_end_to_end_and_validation(self):
        result = factor_analysis("sample", {"level": "$close", "momentum": "$close / Ref($close, 1) - 1"},
                                 provider=self.provider, horizons=[1, 2], quantiles=2, min_samples=2)
        self.assertEqual(len(result.summary), 4)
        self.assertEqual(result.config["adjust"], "hfq")
        self.assertEqual(result.config["entry_lag"], 1)
        for kwargs in ({"horizons": [0]}, {"horizons": [1, 1]}, {"price": "volume"}, {"entry_lag": -1}):
            with self.assertRaises(ValueError):
                calculate_forward_returns(result.factors, provider=self.provider, **kwargs)


if __name__ == "__main__":
    unittest.main()
