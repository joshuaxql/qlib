"""Independent selection, daily aggregation and numerical-scale regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from qlib.contrib.eva.alpha import calc_ic, calc_long_short_prec, calc_long_short_return, pred_autocorr
from qlib.contrib.report.analysis_model import (
    analyze_factors, calculate_factors, calculate_forward_returns, neutralize_factors,
    standardize_factors, winsorize_factors,
)
from qlib.data import LocalProvider


class ExposureProvider:
    """Minimal historical exposures, independent of the binary-data provider."""

    def __init__(self, index, industry, log_cap):
        self.index = index
        self.codes = index.get_level_values("instrument").unique()
        self.dates = index.get_level_values("datetime").unique().sort_values()
        self.industry = pd.Series(industry, index=index)
        self.cap = pd.DataFrame({"total_mv": np.exp(log_cap)}, index=index)

    def calendar(self):
        return self.dates

    def industries(self):
        return sorted(self.industry.unique())

    def universe(self, name, *args):
        membership = self.industry.eq(name.split("/")[1]).unstack("instrument")
        return membership.reindex(index=self.dates, columns=self.codes)

    def daily(self, codes, fields, *args, **kwargs):
        return self.cap.loc[codes, fields]


class FactorCorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.date = pd.Timestamp("2025-01-02")
        self.index = pd.MultiIndex.from_product([[self.date], list("ABCDEFGH")],
                                                names=["datetime", "instrument"])

    def test_single_date_precision_is_daily_scalar_with_correct_denominator(self):
        predictions = pd.Series(range(8), index=self.index)
        labels = pd.Series([-.01, .01, .01, .01, .01, .01, -.01, .01], index=self.index)
        long, short = calc_long_short_prec(predictions, labels, quantile=.25)
        expected = pd.Series([.5], index=pd.DatetimeIndex([self.date], name="datetime"))
        pd.testing.assert_series_equal(long, expected)
        pd.testing.assert_series_equal(short, expected)
        # The same scalar daily aggregation must work with a custom date name.
        renamed = calc_long_short_prec(predictions.rename_axis(index={"datetime": "date"}),
                                       labels.rename_axis(index={"datetime": "date"}),
                                       date_col="date", quantile=.25)
        for actual in renamed:
            pd.testing.assert_series_equal(actual, expected.rename_axis("date"))

    def test_finite_prediction_count_controls_floor_and_nan_inf_never_enter(self):
        predictions = pd.Series([1, 2, 3, 4, 5, np.nan, np.inf, -np.inf], index=self.index)
        labels = pd.Series([-.05, -.04, -.03, .02, .05, .9, .8, -.8], index=self.index)
        # There are five valid predictions, so floor(5 * .25) selects one side.
        spread, universe = calc_long_short_return(predictions, labels, quantile=.25)
        self.assertAlmostEqual(spread.iloc[0], (.05 - (-.05)) / 2)
        self.assertAlmostEqual(universe.iloc[0], np.mean(labels.to_numpy()))
        long, short = calc_long_short_prec(predictions, labels, quantile=.25)
        self.assertEqual(long.iloc[0], 1)
        self.assertEqual(short.iloc[0], 1)

    def test_missing_future_label_never_replaces_a_selected_prediction_by_default(self):
        predictions = pd.Series(range(8), index=self.index)
        labels = pd.Series([-.03, .01, .02, .02, .02, .5, -.02, np.nan], index=self.index)
        spread, universe = calc_long_short_return(predictions, labels, quantile=.25)
        # The top is G/H, with H's unavailable label ignored rather than replaced by F.
        self.assertAlmostEqual(spread.iloc[0], (-.02 - np.mean([-.03, .01])) / 2)
        self.assertAlmostEqual(universe.iloc[0], np.mean(labels.dropna().to_numpy()))
        long, short = calc_long_short_prec(predictions, labels, quantile=.25)
        self.assertEqual(long.iloc[0], 0)
        self.assertEqual(short.iloc[0], .5)
        # Explicit pair filtering remains the opt-in dropna=True API contract.
        paired_spread, _ = calc_long_short_return(predictions, labels, quantile=.25, dropna=True)
        self.assertAlmostEqual(paired_spread.iloc[0], (-.02 - (-.03)) / 2)
        paired_long, paired_short = calc_long_short_prec(predictions, labels, quantile=.25, dropna=True)
        self.assertEqual(paired_long.iloc[0], 0)
        self.assertEqual(paired_short.iloc[0], 1)

    def test_no_signal_date_is_undefined_and_does_not_count_as_return_observation(self):
        predictions = pd.Series(np.nan, index=self.index, name="alpha")
        labels = pd.Series(np.arange(1, 9) / 100, index=self.index, name=1)
        spread, universe = calc_long_short_return(predictions, labels, quantile=.25)
        self.assertTrue(np.isnan(spread.iloc[0]))
        self.assertAlmostEqual(universe.iloc[0], .045)
        for precision in calc_long_short_prec(predictions, labels, quantile=.25):
            self.assertIsInstance(precision, pd.Series)
            self.assertTrue(np.isnan(precision.iloc[0]))
        report = analyze_factors(predictions, labels, quantiles=4)
        row = report.daily.iloc[0]
        self.assertEqual(row.factor_count, 0)
        self.assertEqual(row.pair_count, 0)
        self.assertTrue(np.isnan(row.long_short_return))
        self.assertEqual(report.summary.iloc[0].long_short_count, 0)
        self.assertTrue(np.isnan(report.summary.iloc[0].long_short_mean))

    def test_empty_selected_sides_and_explicit_all_pair_filtering_return_series(self):
        predictions = pd.Series([1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                index=self.index)
        labels = pd.Series(.01, index=self.index)
        # floor(1 * .25) == 0; the other seven rows cannot expand the selection.
        self.assertTrue(np.isnan(calc_long_short_return(predictions, labels, quantile=.25)[0].iloc[0]))
        for precision in calc_long_short_prec(predictions, labels, quantile=.25):
            self.assertIsInstance(precision, pd.Series)
            self.assertTrue(np.isnan(precision.iloc[0]))
        for function in (calc_long_short_prec, calc_long_short_return):
            for result in function(predictions * np.nan, labels, quantile=.25, dropna=True):
                self.assertIsInstance(result, pd.Series)
                self.assertTrue(result.empty)

    def test_tie_selection_and_half_spread_are_preserved(self):
        predictions = pd.Series([1, 1, 2, 3, 4, 5, 6, 6], index=self.index)
        labels = pd.Series([-.1, .4, .03, .04, .05, .06, .7, -.2], index=self.index)
        # floor(8 * .2) == 1; pandas' first tied occurrence selects G and A.
        spread, _ = calc_long_short_return(predictions, labels)
        self.assertAlmostEqual(spread.iloc[0], (.7 - (-.1)) / 2)
        long, short = calc_long_short_prec(predictions, labels)
        self.assertEqual(long.iloc[0], 1)
        self.assertEqual(short.iloc[0], 1)

    def test_pearson_ic_is_invariant_to_independent_positive_rescaling(self):
        index = self.index[:4]
        predictions = np.array([-1., 0., 0., 1.])
        labels = np.array([-.015, -.005, .005, .015])
        # Hand centered-inner-product oracle: 3 / sqrt(2 * 5).
        expected = 3 / np.sqrt(10)
        for pred_scale, label_scale in ((1., 1.), (1e160, 1.), (1e-200, 1.),
                                        (1., 1e160), (1., 1e-200), (1e300, 1e-300)):
            with self.subTest(pred_scale=pred_scale, label_scale=label_scale):
                ic, ric = calc_ic(pd.Series(predictions * pred_scale, index=index),
                                 pd.Series(labels * label_scale, index=index))
                self.assertAlmostEqual(ic.iloc[0], expected, places=14)
                self.assertAlmostEqual(ric.iloc[0], expected, places=14)

    def test_pearson_autocorrelation_and_missing_pairing_are_scale_stable(self):
        dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
        index = pd.MultiIndex.from_product([dates, list("ABCD")], names=["datetime", "instrument"])
        values = np.array([-1., 0., 0., 1., -1.5, -.5, .5, 1.5])
        for scale in (1., 1e160, 1e-200):
            result = pred_autocorr(pd.Series(values * scale, index=index))
            self.assertTrue(np.isnan(result.iloc[0]))
            self.assertAlmostEqual(result.iloc[1], 3 / np.sqrt(10), places=14)
        predictions = pd.Series([1e160, 2e160, np.nan, 4e160], index=self.index[:4])
        labels = pd.Series([.01, np.nan, 999., .04], index=self.index[:4])
        ic, _ = calc_ic(predictions, labels)
        self.assertAlmostEqual(ic.iloc[0], 1., places=14)

    def test_pearson_preserves_small_variation_around_large_common_offsets(self):
        index = self.index[:4]
        predictions = np.array([0., 1., 2., 4.])
        labels = np.array([0., 1., 4., 9.])
        # Exact integer multiples of the independently centered observations.
        x = np.array([-7., -3., 1., 9.])
        y = np.array([-7., -5., 1., 11.])
        expected = np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
        for pred_offset, label_offset in ((1e8, 0.), (0., 1e8), (1e8, -1e8)):
            with self.subTest(pred_offset=pred_offset, label_offset=label_offset):
                ic, _ = calc_ic(pd.Series(predictions + pred_offset, index=index),
                               pd.Series(labels + label_offset, index=index))
                self.assertAlmostEqual(ic.iloc[0], expected, places=14)

    @staticmethod
    def within_industry_oracle(y, log_cap, industries):
        # Frisch-Waugh-Lovell: remove group means, then regress centered y on x.
        y, x = y.copy(), log_cap.copy()
        for group in np.unique(industries):
            rows = industries == group
            y[rows] -= y[rows].mean()
            x[rows] -= x[rows].mean()
        return y - x * np.dot(x, y) / np.dot(x, x)

    def test_neutralization_matches_independent_oracle_across_extreme_scales(self):
        index = self.index.swaplevel().sort_values()
        industry = np.repeat(["one", "two"], 4)
        log_cap = np.tile(np.arange(4.), 2)
        values = np.array([-2., 1., 3., -2., 4., -1., -2., -1.])
        provider = ExposureProvider(index, industry, log_cap)
        expected = self.within_industry_oracle(values, np.log(np.exp(log_cap)), industry)
        for scale in (1., 1e160, 1e-200, 1e300):
            with self.subTest(scale=scale):
                factors = pd.DataFrame({"alpha": values * scale}, index=index)
                result = neutralize_factors(factors, provider=provider)
                np.testing.assert_allclose(result.alpha / scale, expected, rtol=1e-12, atol=1e-12)

    def test_neutralization_preserves_extreme_centered_signal_and_large_constant(self):
        index = self.index[:4].swaplevel().sort_values()
        provider = ExposureProvider(index, ["one"] * 4, np.zeros(4))
        base = np.array([-1., 0., 0., 1.])
        for scale in (1e308, 1e-300):
            result = neutralize_factors(pd.DataFrame({"alpha": base * scale}, index=index), provider=provider)
            np.testing.assert_allclose(result.alpha / scale, base, atol=1e-14)
        result = neutralize_factors(pd.DataFrame({"alpha": 1e308}, index=index), provider=provider)
        np.testing.assert_array_equal(result.alpha, 0.)

    def test_neutralization_preserves_small_variation_around_large_common_offsets(self):
        index = self.index.swaplevel().sort_values()
        industry = np.repeat(["one", "two"], 4)
        log_cap = np.tile(np.arange(4.), 2)
        values = np.array([-2., 1., 3., -2., 4., -1., -2., -1.])
        provider = ExposureProvider(index, industry, log_cap)
        expected = self.within_industry_oracle(values, np.log(np.exp(log_cap)), industry)
        for offset in (1e8, -1e8):
            with self.subTest(offset=offset):
                factors = pd.DataFrame({"alpha": values + offset}, index=index)
                result = neutralize_factors(factors, provider=provider)
                np.testing.assert_allclose(result.alpha, expected, rtol=1e-12, atol=1e-12)

    def test_standardization_preserves_small_variation_after_large_translation(self):
        index = self.index[:4].swaplevel().sort_values()
        values = np.array([0., 1., 4., 10.])
        for ddof in (0, 1):
            expected = (values - values.mean()) / values.std(ddof=ddof)
            for offset in (1e8, -1e8, 1e15):
                with self.subTest(ddof=ddof, offset=offset):
                    factors = pd.DataFrame({"alpha": values + offset}, index=index)
                    result = standardize_factors(factors, ddof=ddof)
                    np.testing.assert_allclose(result.alpha, expected, atol=1e-14)

    def test_winsorization_uses_translated_bounds_and_preserves_unclipped_values(self):
        index = self.index[:4].swaplevel().sort_values()
        values = np.array([0., 1., 4., 10.])
        for offset in (1e8, -1e8, 1e15):
            for method in ("std", "mad"):
                for n in (.1, .3, .8, 1.2):
                    with self.subTest(offset=offset, method=method, n=n):
                        center = values.mean() if method == "std" else np.median(values)
                        spread = values.std() if method == "std" else np.median(np.abs(values - center))
                        # Add the large baseline only once, after the independent clip.
                        expected = offset + np.clip(values, center - n * spread, center + n * spread)
                        result = winsorize_factors(pd.DataFrame({"alpha": values + offset}, index=index),
                                                  method=method, n=n, mad_scale=1)
                        np.testing.assert_array_equal(result.alpha, expected)
        mixed = np.array([-1e308, 0., 1., 2.])
        result = winsorize_factors(pd.DataFrame({"alpha": mixed}, index=index), n=3)
        np.testing.assert_array_equal(result.alpha, mixed)
        result = winsorize_factors(pd.DataFrame({"alpha": mixed}, index=index), method="mad", n=1, mad_scale=1)
        np.testing.assert_allclose(result.alpha, [-.5, 0., 1., 1.5], atol=1e-14)


class CausalFactorAdjustmentTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dates = pd.bdate_range("2025-01-02", periods=5)
        self.codes = ["000001.SZ", "600000.SH"]
        (self.root / "calendars").mkdir()
        (self.root / "instruments").mkdir()
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        (self.root / "instruments/all.txt").write_text("\n".join(
            f"{code} {self.dates[0]:%Y-%m-%d} {self.dates[-1]:%Y-%m-%d}" for code in self.codes))
        # A's two-for-one split occurs after the short signal interval; B never splits.
        for code, prices, factors in (
            (self.codes[0], [20., 22., 24., 12., 13.], [1., 1., 1., 2., 2.]),
            (self.codes[1], [15., 16., 17., 18., 19.], [1.] * 5),
        ):
            directory = self.root / "features" / code
            directory.mkdir(parents=True)
            for field, values in (("open", prices), ("close", prices), ("factor", factors)):
                np.asarray([0, *values], dtype="<f4").tofile(directory / f"{field}.day.bin")

    def test_qfq_signal_factors_and_rankings_have_identical_short_long_prefixes(self):
        definitions = {"level": "$close", "average": "Mean($close,2)", "threshold": "$close>18"}
        for provider_adjust, override in (("qfq", None), ("hfq", "qfq")):
            with self.subTest(provider_adjust=provider_adjust, override=override):
                provider = LocalProvider(self.root, adjust=provider_adjust)
                short = calculate_factors(self.codes, definitions, end_time=self.dates[1],
                                          provider=provider, adjust=override)
                long = calculate_factors(self.codes, definitions, end_time=self.dates[-1],
                                         provider=provider, adjust=override)
                pd.testing.assert_frame_equal(short, long.reindex(short.index))
                pd.testing.assert_series_equal(short.level.groupby(level="datetime").rank(),
                                               long.reindex(short.index).level.groupby(level="datetime").rank())
                self.assertEqual(short.loc[(self.codes[0], self.dates[1]), "level"], 22.)
                self.assertEqual(short.loc[(self.codes[0], self.dates[1]), "average"], 21.)
                self.assertEqual(long.loc[(self.codes[0], self.dates[3]), "average"], 12.)
                # Ordinary feature queries retain their historical query-end qfq convention.
                ordinary_short = provider.features(self.codes, ["$close"], end_time=self.dates[1], adjust="qfq")
                ordinary_long = provider.features(self.codes, ["$close"], end_time=self.dates[-1], adjust="qfq")
                self.assertEqual(ordinary_short.loc[(self.codes[0], self.dates[1]), "$close"], 22.)
                self.assertEqual(ordinary_long.loc[(self.codes[0], self.dates[1]), "$close"], 11.)
                returns = calculate_forward_returns(short, horizons=(1,), provider=provider, adjust="qfq")
                self.assertAlmostEqual(returns.loc[(self.codes[0], self.dates[0]), 1], 24 / 22 - 1)
                self.assertEqual(returns.loc[(self.codes[0], self.dates[1]), 1], 0.)


if __name__ == "__main__":
    unittest.main()
