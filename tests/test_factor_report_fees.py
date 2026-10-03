"""Cash-budget fee oracles, diagnostic units and unchanged public exports."""

from copy import deepcopy
from decimal import Decimal, localcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings

import numpy as np
import pandas as pd

from qlib.contrib.report.analysis_model import analyze_factors
from qlib.contrib.report.analysis_model._factor_report_charts import _add_chart_options
from qlib.contrib.report.analysis_model._factor_report_data import (
    _build_report_data, _json_safe, _net_period_returns, _risk_metrics,
)


COMMISSION, STAMP_TAX = .0003, .001
RISK_FIELDS = {"factor_return", "sharpe", "annualized_return", "max_drawdown"}
TABLES = ("factors", "forward_returns", "summary", "daily", "quantile_returns",
          "quantile_membership", "turnover", "autocorrelation")


def cash_return(value, commission="0.0003", stamp_tax="0.001"):
    """An independent cash ledger: budget -> purchase -> sale -> cash."""
    if not np.isfinite(value):
        return np.nan
    with localcontext() as context:
        context.prec = 60
        budget = Decimal("1000000")
        buy_notional = budget / (1 + Decimal(commission))
        sale_notional = buy_notional * (1 + Decimal(str(value)))
        final_cash = sale_notional - sale_notional * Decimal(commission) - sale_notional * Decimal(stamp_tax)
        return float(final_cash / budget - 1)


def floats(values):
    return np.asarray(values, dtype=float)


class FactorReportFeesTest(unittest.TestCase):
    def make_result(self, top=None, horizons=(1, 3, 20)):
        top = [0., .1, np.nan, -.2, .04, .03, .02, np.nan] if top is None else top
        self.dates = pd.bdate_range("2025-01-02", periods=len(top))
        codes = list("ABCDEFGHIJ")
        index = pd.MultiIndex.from_product([codes, self.dates], names=["instrument", "datetime"])
        values = pd.DataFrame({"score": np.repeat(np.arange(1., 11.), len(top))}, index=index)
        label = np.concatenate([np.asarray(top, dtype=float) - max(8 - stock, 0) * .001
                                for stock in range(10)])
        labels = pd.DataFrame({horizon: label for horizon in horizons}, index=index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = analyze_factors(values, labels, quantiles=5)
        return result

    def data(self, result, **kwargs):
        return _build_report_data(result, **kwargs)

    def test_complete_round_trip_matches_cash_budget_and_charges_the_first_period(self):
        gross = np.array([0., .1, -.2, -1., 1e-20])
        before = gross.copy()
        actual = _net_period_returns(gross, COMMISSION, STAMP_TAX)
        expected = np.array([cash_return(value) for value in gross])
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-16)
        self.assertAlmostEqual(actual[0], -.0016 / 1.0003, places=17)
        self.assertNotEqual(actual[0], -.0016)
        self.assertEqual(actual[3], -1.)
        np.testing.assert_array_equal(gross, before)
        result = self.make_result([0.], horizons=(20,))
        horizon = self.data(result)["factors"][0]["horizons"]["20"]
        self.assertEqual(horizon["performance"]["factor_return"], 0.)
        self.assertAlmostEqual(horizon["fee_scenarios"]["commission_stamp"]["performance"]["factor_return"],
                               expected[0], places=17)

    def test_zero_fee_is_an_exact_identity_without_aliasing_input(self):
        gross = np.array([1e-300, -.1, -1., np.nextafter(1., 2.), 1e308, np.nan, np.inf, -np.inf])
        before = gross.copy()
        actual = _net_period_returns(gross, 0, 0)
        np.testing.assert_array_equal(actual, before)
        self.assertFalse(np.shares_memory(actual, gross))
        actual[0] = 123.
        np.testing.assert_array_equal(gross, before)

    def test_net_quantile_statistics_use_bps_once_without_dividing_fees_by_horizon(self):
        result = self.make_result()
        payload = self.data(result)
        self.assertEqual(payload["meta"]["fee_options"], [
            {"value": "none", "label": "无", "commission": 0, "stamp_tax": 0},
            {"value": "commission_stamp", "label": "3‱佣金 + 1‰印花税",
             "commission": COMMISSION, "stamp_tax": STAMP_TAX}])
        expected = np.array([cash_return(value) for value in [0., .1, np.nan, -.2, .04, .03, .02, np.nan]]) * 10000
        finite = expected[np.isfinite(expected)]
        cumulative = np.array([finite[0], finite[:2].sum(), np.nan, finite[:3].sum(),
                               finite[:4].sum(), finite[:5].sum(), finite.sum(), np.nan])
        first_net = None
        for horizon in payload["factors"][0]["horizons"].values():
            net = horizon["fee_scenarios"]["commission_stamp"]["quantiles"][-1]
            self.assertEqual(set(net), set(horizon["quantiles"][-1]))
            np.testing.assert_allclose(floats(net["daily_bps"]), expected, rtol=2e-14, atol=2e-12, equal_nan=True)
            np.testing.assert_allclose(floats(net["cumulative_bps"]), cumulative, rtol=2e-14, atol=2e-12, equal_nan=True)
            self.assertAlmostEqual(net["mean_return_bps"], finite.mean(), places=11)
            self.assertAlmostEqual(net["standard_error_bps"], finite.std(ddof=1) / np.sqrt(len(finite)), places=11)
            np.testing.assert_allclose(net["box_bps"], np.quantile(finite, [0, .25, .5, .75, 1]), atol=2e-12)
            gross_violin = floats(horizon["quantiles"][-1]["violin"])
            net_violin = floats(net["violin"])
            multiplier = float(Decimal("0.9987") / Decimal("1.0003"))
            np.testing.assert_allclose(net_violin[:, 0], gross_violin[:, 0] * multiplier + cash_return(0.) * 10000,
                                       rtol=2e-14, atol=2e-12)
            np.testing.assert_allclose(net_violin[:, 1], gross_violin[:, 1] / multiplier, rtol=2e-13, atol=1e-16)
            if first_net is not None:
                self.assertEqual(net, first_net)
            first_net = net

    def test_net_risk_uses_original_anchored_periods_and_copies_all_ten_ic_statistics(self):
        result = self.make_result()
        horizon = self.data(result)["factors"][0]["horizons"]["3"]
        net = horizon["fee_scenarios"]["commission_stamp"]["performance"]
        periods = np.array([cash_return(value) for value in (0., -.2, .02)])
        wealth = np.cumprod(1 + periods)
        self.assertEqual(set(net), set(horizon["performance"]))
        self.assertAlmostEqual(net["factor_return"], wealth[-1] - 1, places=14)
        self.assertAlmostEqual(net["annualized_return"], wealth[-1] ** (252 / (3 * 3)) - 1, places=14)
        self.assertAlmostEqual(net["sharpe"], periods.mean() / periods.std(ddof=1) * np.sqrt(252 / 3), places=14)
        self.assertAlmostEqual(net["max_drawdown"], 1 - wealth.min(), places=14)
        self.assertLess(net["factor_return"], horizon["performance"]["factor_return"])
        ic_fields = set(net) - RISK_FIELDS
        self.assertEqual(len(ic_fields), 10)
        for field in ic_fields:
            self.assertEqual(net[field], horizon["performance"][field], field)

    def test_missing_sampled_period_does_not_shift_sampling_or_compress_calendar(self):
        result = self.make_result([0., .9, .8, np.nan, .7, .6, .02, .4], horizons=(3,))
        calendar = pd.bdate_range(self.dates[0] - pd.offsets.BDay(2), periods=10)
        retained = self.dates[[0, 2, 3, 5, 6, 7]]
        values = result.factors.loc[pd.IndexSlice[:, retained], :]
        labels = result.forward_returns.reindex(values.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = analyze_factors(values, labels, quantiles=5)
        horizon = self.data(result, calendar=calendar)["factors"][0]["horizons"]["3"]
        net = horizon["fee_scenarios"]["commission_stamp"]["performance"]
        wealth = (1 + cash_return(0.)) * (1 + cash_return(.02))
        self.assertAlmostEqual(net["factor_return"], wealth - 1, places=14)
        self.assertAlmostEqual(net["annualized_return"], wealth ** (252 / (2 * 3)) - 1, places=13)
        self.assertIsNone(horizon["fee_scenarios"]["commission_stamp"]["quantiles"][-1]["daily_bps"][2])

    def test_nonfinite_complete_loss_invalid_negative_wealth_and_long_overflow(self):
        values = np.array([np.nan, np.inf, -np.inf, -1., np.nextafter(-1., -np.inf)])
        net = _net_period_returns(values, COMMISSION, STAMP_TAX)
        self.assertTrue(np.isnan(net[:3]).all())
        self.assertEqual(net[3], -1.)
        self.assertLess(net[4], -1.)
        # High exit fees make sub-ULP negative wealth round toward zero without
        # the explicit boundary preservation; the invalid return must survive.
        almost_loss = _net_period_returns([np.nextafter(-1., -np.inf)], .5, .4)
        self.assertLess(almost_loss[0], -1.)
        self.assertTrue(all(np.isnan(value) for value in _risk_metrics(almost_loss, 1).values()))
        large = np.finfo(float).max
        net = _net_period_returns([large, large, -1., .2], COMMISSION, STAMP_TAX)
        self.assertTrue(np.isfinite(net).all())
        risk = _risk_metrics(net, 1)
        self.assertEqual(risk["factor_return"], -1.)
        self.assertEqual(risk["annualized_return"], -1.)
        self.assertEqual(risk["max_drawdown"], 1.)
        overflow = _json_safe(_risk_metrics(net[:2], 1))
        self.assertIsNone(overflow["factor_return"])
        self.assertIsNone(overflow["annualized_return"])
        json.dumps(overflow, allow_nan=False)

    def test_all_missing_fee_scenario_keeps_nulls_and_no_sector_overview(self):
        result = self.make_result([np.nan] * 6)
        payload = self.data(result, industries=dict.fromkeys(list("ABCDEFGHIJ"), "sector"))
        for horizon in payload["factors"][0]["horizons"].values():
            net = horizon["fee_scenarios"]["commission_stamp"]
            for group in net["quantiles"]:
                self.assertEqual(group["daily_bps"], [None] * 6)
                self.assertEqual(group["cumulative_bps"], [None] * 6)
                self.assertIsNone(group["mean_return_bps"])
                self.assertIsNone(group["box_bps"])
                self.assertEqual(group["violin"], [])
            self.assertTrue(all(net["performance"][field] is None for field in RISK_FIELDS))
            self.assertEqual(set(net["sector"]), {"quantile_returns"})
            for group in net["sector"]["quantile_returns"][0]["groups"]:
                self.assertIsNone(group["mean_return_bps"])
                self.assertEqual(group["count"], 0)
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)

    def test_sector_affine_means_keep_counts_and_use_independent_rows(self):
        result = self.make_result()
        industries = {code: "first" if stock % 2 == 0 else "second"
                      for stock, code in enumerate("ABCDEFGHIJ")}
        horizon = self.data(result, industries=industries)["factors"][0]["horizons"]["1"]
        gross = deepcopy(horizon["sector"])
        net = horizon["fee_scenarios"]["commission_stamp"]["sector"]
        self.assertEqual(set(net), {"quantile_returns"})
        for base, changed in zip(horizon["sector"]["quantile_returns"], net["quantile_returns"]):
            self.assertIsNot(base, changed)
            self.assertIsNot(base["groups"], changed["groups"])
            self.assertEqual(base["name"], changed["name"])
            for old, new in zip(base["groups"], changed["groups"]):
                self.assertEqual(old["count"], new["count"])
                self.assertEqual(old["quantile"], new["quantile"])
                self.assertAlmostEqual(new["mean_return_bps"], cash_return(old["mean_return_bps"] / 10000) * 10000,
                                       places=11)
        net["quantile_returns"][0]["name"] = "changed"
        net["quantile_returns"][0]["groups"][0]["count"] = 999
        self.assertEqual(horizon["sector"], gross)

    def test_all_chart_values_keep_bps_and_ic_units_without_rounding(self):
        result = self.make_result()
        payload = self.data(result, industries=dict.fromkeys("ABCDEFGHIJ", "sector"))
        _add_chart_options(payload)
        factor = payload["factors"][0]
        for position, (name, horizon) in enumerate(factor["horizons"].items()):
            net = horizon["fee_scenarios"]["commission_stamp"]
            bar = factor["fee_chart_options"]["commission_stamp"]["mean-quantile"]
            self.assertEqual(bar["series"][position]["data"], [item["mean_return_bps"] for item in net["quantiles"]])
            cumulative = net["chart_options"]["common"]["quantile-cumulative"]
            for group, series in zip(net["quantiles"], cumulative["series"]):
                self.assertEqual(series["data"], list(map(list, zip(horizon["dates"], group["cumulative_bps"]))))
            sector_bar = net["chart_options"]["common"]["sector-0"]["series"][0]["data"]
            self.assertEqual([point.get("value") for point in sector_bar],
                             [group["mean_return_bps"] for group in net["sector"]["quantile_returns"][0]["groups"]])
            for metric in ("ic", "rank_ic"):
                self.assertEqual(horizon["chart_options"][metric]["ic-cumulative"]["series"][0]["data"],
                                 list(map(list, zip(horizon["dates"], horizon["daily"][metric + "_cumulative"]))))
            # Ratios used by performance remain raw ratios, not basis points.
            self.assertLess(abs(net["performance"]["factor_return"]), 1)
            self.assertNotIn("ic", net["chart_options"])
        json.dumps(payload, allow_nan=False)

    def test_gross_values_all_tables_configuration_and_csv_bytes_remain_unchanged(self):
        result = self.make_result()
        before = {name: getattr(result, name).copy(deep=True) for name in TABLES}
        config = deepcopy(result.config)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result.save(root / "before", html=False)
            payload = self.data(result)
            result.save(root / "after", html=False)
            for path in (root / "before" / "score").iterdir():
                self.assertEqual(path.read_bytes(), (root / "after" / "score" / path.name).read_bytes(), path.name)
            self.assertEqual(len(list((root / "after" / "score").glob("*.csv"))), 8)
        for horizon in (1, 3, 20):
            data = payload["factors"][0]["horizons"][str(horizon)]
            for group in range(1, 6):
                source = result.quantile_returns.xs(("score", horizon, group), level=("factor", "horizon", "quantile"))["mean"]
                expected = (source * 10000).where(source.notna(), None).astype(object)
                expected[source.isna()] = None
                self.assertEqual(data["quantiles"][group - 1]["daily_bps"], expected.tolist())
            for field in ("ic", "rank_ic", "coverage", "pair_coverage"):
                source = result.daily.loc[("score", horizon), field]
                expected = [None if not np.isfinite(value) else value for value in source]
                self.assertEqual(data["daily"][field], expected)
        for name in TABLES:
            pd.testing.assert_frame_equal(getattr(result, name), before[name], check_exact=True)
        self.assertEqual(result.config, config)


if __name__ == "__main__":
    unittest.main()
