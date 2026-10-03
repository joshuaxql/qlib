"""Independent arithmetic oracles for the report's 14 performance metrics."""

from copy import deepcopy
import json
import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

from qlib.contrib.report.analysis_model import analyze_factors
from qlib.contrib.report.analysis_model._factor_report_data import (
    _build_report_data, _performance_metrics, _risk_metrics,
)

METRICS = {"factor_return", "sharpe", "annualized_return", "max_drawdown", "ic_mean", "rank_ic_mean",
           "ic_std", "ic_ir", "rank_ic_ir", "ic_negative_rate", "ic_positive_rate", "ic_t_stat",
           "ic_p_value", "monotonicity"}


class FactorReportMetricsTest(unittest.TestCase):
    def make_result(self, returns, horizon=3):
        self.dates = pd.bdate_range("2024-01-02", periods=len(returns), name="datetime")
        codes = [f"{number:06d}.SZ" for number in range(1, 11)]
        index = pd.MultiIndex.from_product([codes, self.dates], names=["instrument", "datetime"])
        factors = pd.DataFrame({"score": np.repeat(np.arange(1, 11, dtype=float), len(returns))}, index=index)
        labels = np.concatenate([np.full(len(returns), (number // 2 + 1) / 100) if number < 8 else
                                 np.asarray(returns, dtype=float) for number in range(10)])
        return analyze_factors(factors, pd.DataFrame({horizon: labels}, index=index), quantiles=5, min_samples=3)

    def performance(self, result, horizon=3, **kwargs):
        data = _build_report_data(result, **kwargs)
        return data["factors"][0]["horizons"][str(horizon)]["performance"], data

    def test_fixed_nonoverlapping_periods_use_top_group_and_manual_compounding(self):
        result = self.make_result([.1, .9, .8, -.2, .7, .6])
        performance, data = self.performance(result)
        periods = np.array([.1, -.2])
        self.assertEqual(set(performance), METRICS)
        self.assertAlmostEqual(performance["factor_return"], 1.1 * .8 - 1, places=14)
        self.assertAlmostEqual(performance["annualized_return"], (1.1 * .8) ** (84 / 2) - 1, places=14)
        self.assertAlmostEqual(performance["sharpe"], periods.mean() / periods.std(ddof=1) * np.sqrt(84), places=14)
        self.assertAlmostEqual(performance["max_drawdown"], .2, places=14)
        self.assertEqual(data["meta"]["performance_quantile"], 5)
        self.assertEqual(data["meta"]["performance_calendar"], "report_dates")

    def test_missing_anchored_period_skips_without_reselecting_the_next_signal(self):
        result = self.make_result([np.nan, .3, .4, .2, .9, .9])
        performance, _ = self.performance(result)
        self.assertAlmostEqual(performance["factor_return"], .2, places=14)
        self.assertAlmostEqual(performance["annualized_return"], 1.2 ** 84 - 1, delta=1.2 ** 84 * 2e-14)
        self.assertEqual(performance["max_drawdown"], 0)
        self.assertIsNone(performance["sharpe"])
        result.forward_returns.loc[("000010.SZ", self.dates[3]), 3] = np.nan
        # Group means retain their existing available-label diagnostic meaning.
        refreshed = analyze_factors(result.factors, result.forward_returns, quantiles=5, min_samples=3)
        refreshed_performance, _ = self.performance(refreshed)
        self.assertAlmostEqual(refreshed_performance["factor_return"], .2, places=14)

    def test_first_report_date_anchors_sampling_inside_the_full_calendar(self):
        result = self.make_result([.1, .9, .8, -.2, .7, .6])
        calendar = pd.bdate_range(self.dates[0] - pd.offsets.BDay(2), periods=8)
        performance, data = self.performance(result, calendar=calendar)
        self.assertAlmostEqual(performance["factor_return"], 1.1 * .8 - 1, places=14)
        self.assertEqual(data["meta"]["performance_calendar"], "trading_calendar")
        self.assertEqual(data["meta"]["performance_sampling_anchor"], str(self.dates[0].date()))
        for invalid in (calendar[::-1], calendar.append(calendar[-1:]), calendar[:-1],
                        pd.DatetimeIndex([pd.NaT, *self.dates])):
            with self.subTest(calendar=invalid), self.assertRaisesRegex(ValueError, "Performance calendar"):
                _build_report_data(result, calendar=invalid)

    def test_missing_factor_dates_do_not_compress_the_full_trading_calendar(self):
        result = self.make_result([.1, .9, .8, -.2, .7, .6, .5, .4])
        full_calendar = pd.bdate_range(self.dates[0] - pd.offsets.BDay(2), periods=10)
        retained = self.dates[[0, 2, 3, 4, 6, 7]]
        factors = result.factors.loc[pd.IndexSlice[:, retained], :]
        labels = result.forward_returns.reindex(factors.index)
        result = analyze_factors(factors, labels, quantiles=5, min_samples=3)
        performance, _ = self.performance(result, calendar=full_calendar)
        self.assertAlmostEqual(performance["factor_return"], 1.1 * .8 * 1.5 - 1, places=14)
        fallback, _ = self.performance(result)
        self.assertAlmostEqual(fallback["factor_return"], 1.1 * 1.7 - 1, places=14)

    def test_initial_wealth_is_part_of_drawdown_and_complete_loss_stays_zero(self):
        risk = _risk_metrics([-.2, .25, .1], 1)
        self.assertAlmostEqual(risk["factor_return"], .1, places=14)
        self.assertAlmostEqual(risk["max_drawdown"], .2, places=14)
        for sample in ([.1, -1, .5], [1e200, 1e200, -1, .2]):
            with self.subTest(sample=sample):
                risk = _risk_metrics(sample, 1)
                self.assertEqual(risk["factor_return"], -1)
                self.assertEqual(risk["annualized_return"], -1)
                self.assertEqual(risk["max_drawdown"], 1)

    def test_below_complete_loss_is_unknown_and_is_not_clipped_or_dropped(self):
        for sample in ([-1.01, .2], [np.nan, np.inf], []):
            with self.subTest(sample=sample):
                self.assertTrue(all(np.isnan(value) for value in _risk_metrics(sample, 1).values()))
        result = self.make_result([.1, .9, .8, -1.01, .7, .6])
        performance, _ = self.performance(result)
        self.assertTrue(all(performance[field] is None for field in
                            ("factor_return", "sharpe", "annualized_return", "max_drawdown")))

    def test_ic_sample_statistics_ttest_and_strict_threshold_ties(self):
        result = self.make_result([.1, .2, .3, .4, .5, .6], 1)
        ic = np.array([.02, -.02, .04, -.08, np.nan, 0])
        ric = np.array([.3, -.1, np.nan, .2, 0, np.nan])
        result.daily.loc[("score", 1), "ic"] = ic
        result.daily.loc[("score", 1), "rank_ic"] = ric
        performance, _ = self.performance(result, 1)
        valid, rank_valid = ic[np.isfinite(ic)], ric[np.isfinite(ric)]
        oracle = ttest_1samp(valid, 0)
        self.assertAlmostEqual(performance["ic_mean"], valid.mean(), places=15)
        self.assertAlmostEqual(performance["rank_ic_mean"], rank_valid.mean(), places=15)
        self.assertAlmostEqual(performance["ic_std"], valid.std(ddof=1), places=15)
        self.assertAlmostEqual(performance["ic_ir"], valid.mean() / valid.std(ddof=1), places=15)
        self.assertAlmostEqual(performance["rank_ic_ir"], rank_valid.mean() / rank_valid.std(ddof=1), places=15)
        self.assertAlmostEqual(performance["ic_t_stat"], oracle.statistic, places=14)
        self.assertAlmostEqual(performance["ic_p_value"], oracle.pvalue, places=14)
        self.assertEqual(performance["ic_negative_rate"], 1 / 5)
        self.assertEqual(performance["ic_positive_rate"], 1 / 5)

    def test_constant_and_insufficient_samples_keep_ir_sharpe_and_significance_unknown(self):
        result = self.make_result([.1] * 6, 1)
        for ic, ric in (([.1] * 6, [-.2] * 6), ([.1] + [np.nan] * 5, [np.nan] * 6)):
            result.daily.loc[("score", 1), "ic"] = ic
            result.daily.loc[("score", 1), "rank_ic"] = ric
            performance, _ = self.performance(result, 1)
            self.assertAlmostEqual(performance["ic_mean"], .1)
            for field in ("ic_ir", "rank_ic_ir", "ic_t_stat", "ic_p_value", "sharpe"):
                self.assertIsNone(performance[field], field)

    def test_monotonicity_spearman_retains_ties_missing_groups_and_direction(self):
        for means in ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 2, np.nan, 1, 4], [1] * 5):
            valid = np.isfinite(means)
            with self.subTest(means=means):
                groups = [np.repeat(value, 3) for value in means]
                metrics = _performance_metrics([.1, .2, .3], [.1, .2, .3], groups, [True, False, False], 1)
                if np.ptp(np.asarray(means)[valid]) == 0:
                    self.assertTrue(np.isnan(metrics["monotonicity"]))
                else:
                    oracle = spearmanr(np.arange(1, 6)[valid], np.asarray(means)[valid]).statistic
                    self.assertAlmostEqual(metrics["monotonicity"], oracle, places=14)

    def test_ic_arithmetic_cumulative_preserves_null_gaps_and_all_input_tables(self):
        result = self.make_result([.1, .2, .3, .4, .5, .6], 1)
        ic = np.array([1 / 3, np.nan, -.11, np.inf, .25, -np.inf])
        result.daily.loc[("score", 1), "ic"] = ic
        result.daily.loc[("score", 1), "rank_ic"] = ic[::-1]
        before = {name: getattr(result, name).copy(deep=True) for name in
                  ("factors", "forward_returns", "daily", "summary", "quantile_returns", "quantile_membership",
                   "turnover", "autocorrelation")}
        config = deepcopy(result.config)
        _, data = self.performance(result, 1)
        horizon = data["factors"][0]["horizons"]["1"]
        self.assertEqual(horizon["daily"]["ic_cumulative"],
                         [1 / 3, None, 1 / 3 - .11, None, 1 / 3 - .11 + .25, None])
        self.assertEqual(horizon["daily"]["rank_ic_cumulative"],
                         [None, .25, None, .25 - .11, None, .25 - .11 + 1 / 3])
        self.assertEqual(set(data["factors"][0]["summary"][0]),
                         {"horizon", "top_turnover", "bottom_turnover", "autocorrelation", "coverage", "pair_coverage", "dates"})
        for name, frame in before.items():
            pd.testing.assert_frame_equal(getattr(result, name), frame, check_exact=True)
        self.assertEqual(result.config, config)
        self.assertIn("long_short_return", result.daily)
        self.assertIn("long_short_mean", result.summary)
        json.dumps(data, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
