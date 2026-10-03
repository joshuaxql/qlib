"""Hand-calculated HTML diagnostics and grouped-sector numerical oracles."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings

import numpy as np
import pandas as pd

from qlib.contrib.eva.alpha import _array_correlations
from qlib.contrib.report.analysis_model import analyze_factors
from qlib.contrib.report.analysis_model._factor_report_data import (
    _build_report_data, _cumulative, _distribution,
    _group_correlations, _group_ranks, _historical_industries, _sector_payload,
)


TABLES = ("factors", "forward_returns", "summary", "daily", "quantile_returns",
          "quantile_membership", "turnover", "autocorrelation")


def analyze(values, returns, **kwargs):
    # Perfect IC has infinite IR; this deliberately exercises strict-JSON
    # missing-value handling without cluttering the test output with a warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return analyze_factors(values, returns, **kwargs)


class HistoricalProvider:
    def __init__(self, dates, conflict=False):
        self.dates = dates
        self.conflict = conflict
        self.calls = []

    def industries(self):
        return ["first", "second"]

    def universe(self, market, start, end):
        self.calls.append((market, start, end))
        mask = pd.DataFrame(False, index=self.dates, columns=list("ABCD"))
        if market == "industry/first":
            mask.loc[self.dates[:2], "A"] = True
        else:
            mask.loc[self.dates[2:], "A"] = True
            mask.loc[:, "B"] = True
            if self.conflict:
                mask.loc[self.dates[0], "A"] = True
        return mask

    def stock_basic(self, *args, **kwargs):
        raise AssertionError("Current industry snapshots must not be consulted")


class FactorHTMLDataTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-02", periods=6)
        self.index = pd.MultiIndex.from_product([list("ABCD"), self.dates], names=["instrument", "datetime"])
        self.values = pd.DataFrame({"alpha": np.repeat([1., 2., 3., 4.], 6)}, index=self.index)
        self.labels = pd.DataFrame({1: np.repeat([.01, .02, .03, .04], 6)}, index=self.index)
        self.result = analyze(self.values, self.labels, quantiles=2)

    def data(self, result=None, **kwargs):
        return _build_report_data(self.result if result is None else result, **kwargs)

    def test_quantile_returns_keep_original_units_and_no_removed_report_calculations(self):
        factor = self.data()["factors"][0]
        horizon = factor["horizons"]["1"]
        np.testing.assert_allclose(horizon["quantiles"][0]["daily_bps"], 150.)
        np.testing.assert_allclose(horizon["quantiles"][1]["daily_bps"], 350.)
        np.testing.assert_allclose(horizon["quantiles"][0]["cumulative_bps"], np.arange(1, 7) * 150.)
        removed = {"factor_weighted", "spread", "returns_metrics", "quantile_stats", "alpha_bps", "beta",
                   "missing_gross_weight", "undefined_weights", "cumulative_display_bps"}

        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse(removed.intersection(value))
                for item in value.values():
                    check_keys(item)
            elif isinstance(value, list):
                for item in value:
                    check_keys(item)

        check_keys(factor)
        self.assertEqual(set(factor["summary"][0]), {"horizon", "top_turnover", "bottom_turnover",
                                                    "autocorrelation", "coverage", "pair_coverage", "dates"})

    def test_missing_labels_preserve_global_memberships_and_coverage_denominators(self):
        labels = self.labels.copy()
        labels.loc[("A", self.dates[1]), 1] = np.nan
        labels.loc[("B", self.dates[1]), 1] = np.nan
        labels.loc[("D", self.dates[2]), 1] = np.inf
        changed = analyze(self.values, labels, quantiles=2)
        pd.testing.assert_frame_equal(changed.quantile_membership, self.result.quantile_membership)
        daily = self.data(changed)["factors"][0]["horizons"]["1"]["daily"]
        self.assertEqual(daily["universe_count"], [4] * 6)
        self.assertEqual(daily["factor_count"], [4] * 6)
        self.assertEqual(daily["label_count"], [4, 2, 3, 4, 4, 4])
        self.assertEqual(daily["pair_count"], [4, 2, 3, 4, 4, 4])
        self.assertEqual(daily["missing_label_count"], [0, 2, 1, 0, 0, 0])
        np.testing.assert_allclose(daily["coverage"], 1.)
        np.testing.assert_allclose(daily["pair_coverage"], [1., .5, .75, 1., 1., 1.])

    def test_missing_factor_and_label_counts_are_independent(self):
        index = pd.MultiIndex.from_product([list("ABC"), self.dates], names=["instrument", "datetime"])
        values = pd.DataFrame({"score": np.repeat([1., 2., 3.], 6)}, index=index)
        labels = pd.DataFrame({1: np.repeat([.01, np.nan, .03], 6)}, index=index)
        first = self.data(analyze(values, labels, quantiles=2))["factors"][0]["horizons"]["1"]["daily"]
        self.assertEqual(first["universe_count"], [3] * 6)
        self.assertEqual(first["factor_count"], [3] * 6)
        self.assertEqual(first["label_count"], [2] * 6)
        self.assertEqual(first["pair_count"], [2] * 6)
        self.assertEqual(first["missing_label_count"], [1] * 6)
        values.loc["B", "score"] = np.nan
        second = self.data(analyze(values, labels, quantiles=2))["factors"][0]["horizons"]["1"]["daily"]
        self.assertEqual(second["universe_count"], [3] * 6)
        self.assertEqual(second["factor_count"], [2] * 6)
        self.assertEqual(second["label_count"], [2] * 6)
        self.assertEqual(second["pair_count"], [2] * 6)
        self.assertEqual(second["missing_label_count"], [0] * 6)

    def test_ic_cumulative_is_arithmetic_with_null_prefix_internal_gaps_and_tail(self):
        result = copy.deepcopy(self.result)
        result.daily.loc[("alpha", 1), "ic"] = [np.nan, .1, np.nan, -.2, 0., np.nan]
        result.daily.loc[("alpha", 1), "rank_ic"] = [np.nan, -.25, .5, np.nan, .25, np.nan]
        before = {name: getattr(result, name).copy(deep=True) for name in TABLES}
        config = copy.deepcopy(result.config)
        daily = self.data(result)["factors"][0]["horizons"]["1"]["daily"]
        self.assertEqual(daily["ic"], [None, .1, None, -.2, 0., None])
        self.assertEqual(daily["rank_ic"], [None, -.25, .5, None, .25, None])
        self.assertEqual(daily["ic_cumulative"], [None, .1, None, -.1, -.1, None])
        self.assertEqual(daily["rank_ic_cumulative"], [None, -.25, .25, None, .5, None])
        for name in TABLES:
            pd.testing.assert_frame_equal(getattr(result, name), before[name], check_exact=True)
        self.assertEqual(result.config, config)

    def test_constant_and_all_missing_samples(self):
        values = self.values.assign(constant=1., unavailable=np.nan)
        result = analyze(values, self.labels * np.nan, quantiles=2)
        payload = self.data(result)
        for factor in payload["factors"]:
            horizon = factor["horizons"]["1"]
            for metric in ("ic", "rank_ic"):
                self.assertEqual(horizon["daily"][metric], [None] * 6)
                self.assertEqual(horizon["daily"][metric + "_cumulative"], [None] * 6)
                self.assertEqual(horizon[metric + "_distribution"]["histogram"], [])
                self.assertEqual(horizon[metric + "_distribution"]["qq"], [])
            self.assertEqual(horizon["daily"]["label_count"], [0] * 6)
            for quantile in horizon["quantiles"]:
                self.assertIsNone(quantile["box_bps"])
                self.assertEqual(quantile["violin"], [])
                self.assertIsNone(quantile["mean_return_bps"])

    def test_constant_ic_performance_is_null_safe_and_keeps_original_public_summary(self):
        before = {name: getattr(self.result, name).copy(deep=True) for name in TABLES}
        performance = self.data()["factors"][0]["horizons"]["1"]["performance"]
        self.assertEqual(set(performance), {
            "factor_return", "sharpe", "annualized_return", "max_drawdown", "ic_mean", "rank_ic_mean",
            "ic_std", "ic_ir", "rank_ic_ir", "ic_negative_rate", "ic_positive_rate", "ic_t_stat",
            "ic_p_value", "monotonicity"})
        self.assertEqual(performance["ic_std"], 0.)
        for field in ("ic_ir", "rank_ic_ir", "ic_t_stat", "ic_p_value"):
            self.assertIsNone(performance[field])
        self.assertTrue(np.isinf(self.result.summary.loc[("alpha", 1), "ic_ir"]))
        self.assertEqual(json.loads(json.dumps(performance, allow_nan=False)), performance)
        for name in TABLES:
            pd.testing.assert_frame_equal(getattr(self.result, name), before[name], check_exact=True)

    def test_strict_json_preserves_nonfinite_as_null_without_changing_source(self):
        result = copy.deepcopy(self.result)
        result.daily.loc[("alpha", 1), "ic"] = [np.inf, -np.inf, np.nan, .125, .25, 0.]
        before = result.daily.copy(deep=True)
        restored = json.loads(json.dumps(self.data(result), allow_nan=False))
        daily = restored["factors"][0]["horizons"]["1"]["daily"]
        self.assertEqual(daily["ic"], [None, None, None, .125, .25, 0.])
        self.assertEqual(daily["ic_cumulative"], [None, None, None, .125, .375, .375])
        self.assertTrue(any("非有限" in note for note in restored["meta"]["notes"]))
        pd.testing.assert_frame_equal(result.daily, before, check_exact=True)

    def test_arithmetic_accumulation_keeps_gaps_and_horizon_does_not_rescale_ic(self):
        np.testing.assert_allclose(_cumulative([.1, np.nan, -.2, np.inf, 0.]),
                                   [.1, np.nan, -.1, np.nan, -.1], equal_nan=True)
        labels = self.labels.copy()
        labels[5] = labels[1]
        factor = self.data(analyze(self.values, labels, quantiles=2))["factors"][0]
        first, fifth = factor["horizons"]["1"]["daily"], factor["horizons"]["5"]["daily"]
        for metric in ("ic", "rank_ic"):
            self.assertEqual(first[metric + "_cumulative"], fifth[metric + "_cumulative"])
            np.testing.assert_allclose(fifth[metric + "_cumulative"], np.arange(1, 7))

    def test_group_mean_weights_dates_equally(self):
        labels = self.labels.copy()
        labels.loc[("A", slice(None)), 1] = [.01, .10, np.nan, np.nan, np.nan, np.nan]
        labels.loc[("B", slice(None)), 1] = [.03, np.nan, np.nan, np.nan, np.nan, np.nan]
        group = self.data(analyze(self.values, labels, quantiles=2))["factors"][0]["horizons"]["1"]["quantiles"][0]
        # Daily means .02 and .10 give .06, rather than pooling three labels.
        self.assertAlmostEqual(group["mean_return_bps"], 600.)
        self.assertAlmostEqual(group["standard_error_bps"], 400.)
        np.testing.assert_allclose(group["box_bps"], [200., 400., 600., 800., 1000.])
        self.assertEqual(group["daily_bps"][2:], [None] * 4)
        self.assertEqual(len(group["violin"]), 64)

    def test_ic_monthly_histogram_and_rolling_missingness(self):
        dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-02-03", "2025-02-04"])
        distribution = _distribution([1., np.nan, -1., .5], dates)
        monthly = distribution["monthly"]
        self.assertEqual(monthly["years"], [2025])
        self.assertEqual(monthly["values"][0], [0, 0, 1.])
        self.assertEqual(monthly["values"][1], [1, 0, -.25])
        self.assertTrue(np.isnan(monthly["values"][2][2]))
        self.assertEqual(sum(row["count"] for row in distribution["histogram"]), 3)
        np.testing.assert_array_equal(np.asarray(distribution["qq"])[:, 1], [-1., .5, 1.])
        rolling = self.data()["factors"][0]["horizons"]["1"]["daily"]["ic_rolling"]
        self.assertEqual(rolling[:4], [None] * 4)
        np.testing.assert_allclose(rolling[4:], 1.)

    def test_almost_constant_ic_histogram_retains_observations_without_zero_width_bins(self):
        first = np.nextafter(1., 0.)
        second = np.nextafter(first, 0.)
        values = [1., first, 1., second, np.nan, first]
        distribution = _distribution(values, self.dates)
        self.assertEqual(sum(row["count"] for row in distribution["histogram"]), 5)
        self.assertTrue(all(np.isfinite(row["center"]) and np.isfinite(row["width"])
                            and row["width"] > 0 for row in distribution["histogram"]))
        # Histogram geometry may use a wider representable interval, while the
        # underlying sample, QQ positions and monthly mean keep full precision.
        self.assertEqual([point[1] for point in distribution["qq"]], sorted([1., first, 1., second, first]))
        self.assertEqual(distribution["monthly"]["values"][0][2], np.nanmean(values))

    def test_historical_industries_date_changes_and_conflicts(self):
        provider = HistoricalProvider(self.dates)
        sectors = _historical_industries(self.result, provider)
        self.assertEqual(sectors.loc[("A", self.dates[0])], "first")
        self.assertEqual(sectors.loc[("A", self.dates[2])], "second")
        self.assertEqual(sectors.loc[("B", self.dates[0])], "second")
        self.assertIsNone(sectors.loc[("C", self.dates[0])])
        self.assertEqual(len(provider.calls), 2)
        with self.assertRaisesRegex(ValueError, "Multiple historical industry memberships"):
            _historical_industries(self.result, HistoricalProvider(self.dates, conflict=True))
        self.assertIsNone(_historical_industries(self.result, object()))

    def test_sector_groups_keep_global_memberships(self):
        industries = {"A": "first", "B": "first", "C": "second", "D": "second"}
        sector = self.data(industries=industries)["factors"][0]["horizons"]["1"]["sector"]
        self.assertTrue(sector["available"])
        first, second = sector["quantile_returns"]
        self.assertEqual(first["name"], "first")
        self.assertAlmostEqual(first["groups"][0]["mean_return_bps"], 150.)
        self.assertIsNone(first["groups"][1]["mean_return_bps"])
        self.assertIsNone(second["groups"][0]["mean_return_bps"])
        self.assertAlmostEqual(second["groups"][1]["mean_return_bps"], 350.)
        self.assertEqual(first["groups"][0]["count"], 6)
        np.testing.assert_allclose([row["ic_mean"] for row in sector["overview"]], 1.)

    def test_industry_series_alignment_and_invalid_indices(self):
        industries = pd.Series("sector", index=self.index, dtype=object)
        industries.loc[("A", self.dates[0])] = "changed"
        industries.loc[("A", self.dates[1])] = None
        first = self.data(industries=industries)
        second = self.data(industries=industries.swaplevel().sort_index(ascending=False))
        self.assertEqual(first, second)
        sector = first["factors"][0]["horizons"]["1"]["sector"]
        changed = next(row for row in sector["quantile_returns"] if row["name"] == "changed")
        self.assertEqual(changed["groups"][0]["count"], 1)
        invalid = [industries.reset_index(drop=True), pd.concat([industries, industries]),
                   industries.rename_axis(["code", "date"])]
        wrong_dates = self.index.to_frame(index=False)
        wrong_dates["datetime"] = wrong_dates["datetime"].astype(str)
        invalid.append(pd.Series("sector", index=pd.MultiIndex.from_frame(wrong_dates)))
        null_dates = self.index.to_frame(index=False)
        null_dates.loc[0, "datetime"] = pd.NaT
        invalid.append(pd.Series("sector", index=pd.MultiIndex.from_frame(null_dates)))
        for item in invalid:
            with self.subTest(index=item.index), self.assertRaises(ValueError):
                self.data(industries=item)
        self.assertFalse(self.data(industries={})["factors"][0]["horizons"]["1"]["sector"]["available"])

    def test_export_preserves_every_table_and_configuration(self):
        before = {name: getattr(self.result, name).copy(deep=True) for name in TABLES}
        config = copy.deepcopy(self.result.config)
        self.data(industries={"A": "sector"})
        with TemporaryDirectory() as temporary:
            destination = Path(temporary)
            path = self.result.to_html(destination / "nested" / "factor.html", title="example")
            self.assertEqual(path, destination / "nested" / "factor.html")
            document = path.read_text(encoding="utf-8")
            self.assertIn("echarts", document.lower())
            self.assertIn('"title":"example"', document)
            self.result.save(destination / "tables", html=False)
            self.assertEqual(len(list((destination / "tables" / "alpha").glob("*.csv"))), 8)
            self.assertFalse((destination / "tables" / "alpha" / "report.html").exists())
            self.assertEqual(json.loads((destination / "tables" / "alpha" / "config.json").read_text()), config)
            self.result.save(destination / "default")
            self.assertTrue((destination / "default" / "alpha" / "report.html").exists())
        for name in TABLES:
            pd.testing.assert_frame_equal(getattr(self.result, name), before[name], check_exact=True)
        self.assertEqual(self.result.config, config)


class FactorHTMLSectorOracleTest(unittest.TestCase):
    def test_batched_sectors_against_individual_stable_evaluators(self):
        rng = np.random.default_rng(517)
        for size in (1, 2, 17, 139, 1777):
            for style in ("random", "ties", "constant", "large", "tiny"):
                with self.subTest(size=size, style=style):
                    dates, sectors = rng.integers(0, 11, size), rng.integers(-1, 4, size)
                    x, y = rng.normal(size=size), rng.normal(size=size)
                    if style == "ties":
                        x = rng.integers(0, 4, size).astype(float)
                        y = rng.integers(0, 5, size).astype(float)
                    elif style == "constant":
                        x[:] = 1
                    elif style == "large":
                        x, y = x * 1e150 + 1e160, y * 1e200
                    elif style == "tiny":
                        x, y = x * 1e-200, y * 1e-200
                    x[rng.random(size) < .1] = np.nan
                    y[rng.random(size) < .1] = np.nan
                    group_ids = dates * 4 + sectors
                    valid = (sectors >= 0) & np.isfinite(x) & np.isfinite(y)
                    ids, left, right = group_ids[valid], x[valid], y[valid]
                    ic = _group_correlations(left, right, ids, 44, 2)
                    ric = _group_correlations(_group_ranks(left, ids), _group_ranks(right, ids), ids, 44, 2)
                    expected = np.full((44, 2), np.nan)
                    for group in range(44):
                        rows = ids == group
                        if rows.sum() >= 2:
                            expected[group] = _array_correlations(left[rows], right[rows])
                    np.testing.assert_allclose(ic, expected[:, 0], atol=1e-14, rtol=1e-13, equal_nan=True)
                    np.testing.assert_allclose(ric, expected[:, 1], atol=1e-14, rtol=1e-13, equal_nan=True)
                    memberships = rng.integers(1, 6, size).astype(float)
                    memberships[~np.isfinite(x)] = np.nan
                    actual = _sector_payload(x, y, memberships, sectors, dates, list("abcd"), 11, 5, 2)
                    for sector, item in enumerate(actual["quantile_returns"]):
                        for group, row in enumerate(item["groups"], 1):
                            daily = []
                            for date in range(11):
                                rows = (sectors == sector) & (dates == date) & (memberships == group) & np.isfinite(y)
                                daily.append(y[rows].mean() if rows.any() else np.nan)
                            expected_mean = np.nanmean(daily) if np.isfinite(daily).any() else np.nan
                            np.testing.assert_allclose(row["mean_return_bps"], expected_mean * 10000,
                                                       atol=1e-12, rtol=1e-13, equal_nan=True)
                            self.assertEqual(row["count"], np.isfinite(daily).sum())


if __name__ == "__main__":
    unittest.main()
