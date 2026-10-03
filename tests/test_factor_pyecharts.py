"""pyecharts rendering preserves diagnostic values, units and missing dates."""

import copy
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from pyecharts.charts import Bar, Custom, HeatMap, Line, Scatter

from qlib.contrib.report.analysis_model import analyze_factors
from qlib.contrib.report.analysis_model import _factor_report_charts as charts
from qlib.contrib.report.analysis_model._factor_report_data import _build_report_data


def analyze(values, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return analyze_factors(values, labels, quantiles=2)


def line_values(option, series=0):
    return option["series"][series]["data"]


def item_value(item):
    # pyecharts removes None-valued object keys; an absent BarItem.value remains
    # a missing ECharts observation, just like a literal null in a Line array.
    return item.get("value") if isinstance(item, dict) else item


class FactorPyechartsTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-02", periods=6)
        index = pd.MultiIndex.from_product([list("ABCD"), self.dates], names=["instrument", "datetime"])
        self.values = pd.DataFrame({"alpha": np.repeat([1., 2., 4., 7.], 6)}, index=index)
        daily = np.arange(6) * .000123456789
        returns = np.concatenate([base + daily for base in (.01123456789, .027234567891,
                                                           .019234567892, .053345678913)])
        self.labels = pd.DataFrame({1: returns, 5: returns * 2.171717171717}, index=index)
        self.labels.loc[("B", self.dates[1]), 1] = np.nan
        self.labels.loc[("A", self.dates[-1]), 5] = np.nan
        self.result = analyze(self.values, self.labels)
        self.industries = {"A": "lower", "B": "lower", "C": "upper", "D": "upper"}
        self.original = _build_report_data(self.result, industries=self.industries)
        self.payload = copy.deepcopy(self.original)
        charts._add_chart_options(self.payload)
        self.factor = self.payload["factors"][0]

    def test_quantile_return_and_stability_series_preserve_dates_bps_and_nulls(self):
        factor = self.factor
        for name, horizon in factor["horizons"].items():
            with self.subTest(horizon=name):
                dates, options = horizon["dates"], horizon["chart_options"]["common"]
                self.assertFalse({"weighted-cumulative", "quantile-spread", "spread-cumulative"}.intersection(options))
                self.assertNotIn("factor_weighted", horizon)
                self.assertNotIn("spread", horizon)
                self.assertNotIn("returns_metrics", factor)
                for column, quantile in enumerate(horizon["quantiles"]):
                    self.assertEqual(line_values(options["quantile-cumulative"], column),
                                     list(map(list, zip(dates, quantile["cumulative_bps"]))))
                for column, field in enumerate(("coverage", "pair_coverage")):
                    self.assertEqual(line_values(options["coverage"], column),
                                     list(map(list, zip(dates, horizon["daily"][field]))))
                self.assertEqual(line_values(options["factor-autocorrelation"]),
                                 list(map(list, zip(factor["autocorrelation"]["dates"], factor["autocorrelation"]["values"]))))
                for column, field in enumerate(("top", "bottom")):
                    self.assertEqual(line_values(options["quantile-turnover"], column),
                                     list(map(list, zip(factor["turnover"]["dates"], factor["turnover"][field]))))
                # Group return values already use bps; the chart preserves units.
                bar_series = factor["chart_options"]["mean-quantile"]["series"]
                position = self.payload["meta"]["horizons"].index(int(name))
                self.assertEqual(bar_series[position]["data"],
                                 [item["mean_return_bps"] for item in horizon["quantiles"]])

    def test_ic_cumulative_chart_keeps_null_gaps_without_smoothing_or_range_clipping(self):
        payload = copy.deepcopy(self.original)
        horizon = payload["factors"][0]["horizons"]["1"]
        horizon["daily"]["ic_cumulative"] = [None, .1, None, -.1, -.1, None]
        horizon["daily"]["rank_ic_cumulative"] = [None, -.25, .25, None, .5, None]
        charts._add_chart_options(payload)
        for metric in ("ic", "rank_ic"):
            option = horizon["chart_options"][metric]["ic-cumulative"]
            self.assertEqual(line_values(option),
                             list(map(list, zip(horizon["dates"], horizon["daily"][metric + "_cumulative"]))))
            self.assertEqual(len(option["series"]), 1)
            self.assertFalse(option["series"][0]["connectNulls"])
            self.assertFalse(option["series"][0]["smooth"])
            self.assertIsNone(option["yAxis"][0].get("min"))
            self.assertIsNone(option["yAxis"][0].get("max"))

    def test_both_ic_modes_keep_original_units_and_separate_series(self):
        for horizon in self.factor["horizons"].values():
            for metric in ("ic", "rank_ic"):
                with self.subTest(metric=metric):
                    mode = horizon["chart_options"][metric]
                    self.assertTrue({"ic-timeline", "ic-cumulative", "ic-histogram", "ic-qq", "ic-monthly"}.issubset(mode))
                    self.assertEqual(line_values(mode["ic-timeline"]),
                                     list(map(list, zip(horizon["dates"], horizon["daily"][metric]))))
                    self.assertEqual(line_values(mode["ic-timeline"], 1),
                                     list(map(list, zip(horizon["dates"], horizon["daily"][metric + "_rolling"]))))
                    self.assertEqual(line_values(mode["ic-cumulative"]),
                                     list(map(list, zip(horizon["dates"], horizon["daily"][metric + "_cumulative"]))))
                    distribution = horizon["ic_distribution" if metric == "ic" else "rank_ic_distribution"]
                    self.assertEqual(mode["ic-qq"]["series"][0]["data"], distribution["qq"])
                    self.assertEqual(mode["ic-monthly"]["series"][0]["data"],
                                     [item for item in distribution["monthly"]["values"] if item[2] is not None])
                    self.assertEqual(mode["ic-timeline"]["yAxis"][0]["min"], -1)
                    self.assertEqual(mode["ic-timeline"]["yAxis"][0]["max"], 1)
        self.assertNotEqual(self.factor["horizons"]["1"]["daily"]["ic"],
                            self.factor["horizons"]["1"]["daily"]["rank_ic"])

    def test_histogram_qq_and_monthly_hand_calculated_examples(self):
        payload = copy.deepcopy(self.original)
        horizon = payload["factors"][0]["horizons"]["1"]
        horizon["ic_distribution"] = {
            "histogram": [{"center": -.2, "count": 3, "width": .1},
                          {"center": 0., "count": 2, "width": .1}],
            "qq": [[-1., .1], [0., .4], [1., .6]],
            "monthly": {"years": [2025], "months": [1, 2, 3],
                        "values": [[0, 0, .123456789], [1, 0, None], [2, 0, -.987654321]]},
        }
        charts._add_chart_options(payload)
        options = horizon["chart_options"]["ic"]
        np.testing.assert_allclose(options["ic-histogram"]["series"][0]["data"],
                                   [[-.2, 3, -.25, -.15], [0., 2, -.05, .05]], rtol=1e-14, atol=1e-15)
        self.assertEqual(options["ic-histogram"]["series"][0]["type"], "custom")
        # Least-squares slope=.25, mean(y)=11/30, giving these two endpoints.
        reference = [item_value(item) for item in options["ic-qq"]["series"][1]["data"]]
        np.testing.assert_allclose(reference, [[-1., 7 / 60], [1., 37 / 60]], rtol=1e-14, atol=1e-15)
        self.assertEqual(options["ic-monthly"]["series"][0]["data"],
                         [[0, 0, .123456789], [2, 0, -.987654321]])
        self.assertEqual(horizon["ic_distribution"]["monthly"]["values"][1], [1, 0, None])

    def test_full_precision_is_independent_of_display_formatting(self):
        payload = copy.deepcopy(self.original)
        horizon = payload["factors"][0]["horizons"]["1"]
        horizon["quantiles"][0]["mean_return_bps"] = 123.456789012345
        horizon["daily"]["ic_cumulative"][0] = 999.123456789012345
        horizon["daily"]["rank_ic_cumulative"][0] = -123.98765432109876
        horizon["daily"]["coverage"][0] = .6543210987654321
        horizon["daily"]["ic"][0] = .1234567890123456
        before = copy.deepcopy(payload)
        charts._add_chart_options(payload)
        factor, common = payload["factors"][0], horizon["chart_options"]["common"]
        self.assertEqual(factor["chart_options"]["mean-quantile"]["series"][0]["data"][0], 123.456789012345)
        self.assertEqual(line_values(horizon["chart_options"]["ic"]["ic-cumulative"])[0][1], 999.123456789012345)
        self.assertEqual(line_values(horizon["chart_options"]["rank_ic"]["ic-cumulative"])[0][1], -123.98765432109876)
        self.assertEqual(line_values(common["coverage"])[0][1], .6543210987654321)
        self.assertEqual(line_values(horizon["chart_options"]["ic"]["ic-timeline"])[0][1], .1234567890123456)
        # Decimal presentation is a fixed template callback, while source JSON
        # and pyecharts serialization continue to contain the original floats.
        for item in payload["factors"]:
            item.pop("chart_options")
            item.pop("fee_chart_options", None)
            for data in item["horizons"].values():
                data.pop("chart_options")
                for scenario in data.get("fee_scenarios", {}).values():
                    scenario.pop("chart_options", None)
        self.assertEqual(payload, before)

    def test_fee_return_options_preserve_net_values_and_do_not_override_other_charts(self):
        payload = copy.deepcopy(self.original)
        payload["meta"]["fee_options"] = [
            {"value": "none", "label": "无", "commission": 0, "stamp_tax": 0},
            {"value": "commission_stamp", "label": "3‱佣金 + 1‰印花税",
             "commission": .0003, "stamp_tax": .001},
        ]
        factor = payload["factors"][0]
        for horizon in factor["horizons"].values():
            quantiles = copy.deepcopy(horizon["quantiles"])
            for position, quantile in enumerate(quantiles):
                quantile["mean_return_bps"] = -12.123456789012345 * (position + 1)
                quantile["cumulative_bps"] = [None, -12.3456789012345, None, -.5, .00000123456789, None]
                quantile["box_bps"] = [-123.456789012345, -50., -12.3, -.5, .123456789012345]
                quantile["violin"] = [[-123.456789012345, .1], [.123456789012345, .2]]
            sector = copy.deepcopy(horizon["sector"]["quantile_returns"])
            for item in sector:
                for position, group in enumerate(item["groups"]):
                    group["mean_return_bps"] = -.987654321098765 if position else None
            horizon["fee_scenarios"] = {"commission_stamp": {
                "quantiles": quantiles, "performance": copy.deepcopy(horizon["performance"]),
                "sector": {"quantile_returns": sector}}}
        before = copy.deepcopy(payload)
        baseline = copy.deepcopy(payload)
        baseline["meta"].pop("fee_options")
        charts._add_chart_options(baseline)
        charts._add_chart_options(payload)
        self.assertEqual(factor["chart_options"], baseline["factors"][0]["chart_options"])
        self.assertEqual(set(factor["fee_chart_options"]["commission_stamp"]), {"mean-quantile"})
        for column, horizon_name in enumerate(payload["meta"]["horizons"]):
            horizon = factor["horizons"][str(horizon_name)]
            scenario = horizon["fee_scenarios"]["commission_stamp"]
            fee = scenario["chart_options"]["common"]
            self.assertEqual(horizon["chart_options"], baseline["factors"][0]["horizons"][str(horizon_name)]["chart_options"])
            self.assertEqual(set(fee), {"quantile-distribution", "quantile-cumulative",
                                       *[f"sector-{index}" for index in range(len(scenario["sector"]["quantile_returns"]))]})
            mean = factor["fee_chart_options"]["commission_stamp"]["mean-quantile"]
            self.assertEqual(mean["series"][column]["data"], [item["mean_return_bps"] for item in scenario["quantiles"]])
            for position, quantile in enumerate(scenario["quantiles"]):
                self.assertEqual(line_values(fee["quantile-cumulative"], position),
                                 list(map(list, zip(horizon["dates"], quantile["cumulative_bps"]))))
                self.assertFalse(fee["quantile-cumulative"]["series"][position]["connectNulls"])
            self.assertEqual(fee["quantile-distribution"]["series"][0]["data"],
                             [[index, -123.456789012345, .123456789012345] for index in range(len(scenario["quantiles"]))])
            for index, item in enumerate(scenario["sector"]["quantile_returns"]):
                self.assertEqual([item_value(value) for value in fee[f"sector-{index}"]["series"][0]["data"]],
                                 [group["mean_return_bps"] for group in item["groups"]])
            horizon.pop("chart_options")
            scenario.pop("chart_options")
        factor.pop("chart_options")
        factor.pop("fee_chart_options")
        self.assertEqual(payload, before)

    def test_fee_return_options_keep_empty_quantiles_and_sectors(self):
        payload = copy.deepcopy(self.original)
        payload["meta"]["fee_options"] = [{"value": "none"}, {"value": "commission_stamp"}]
        factor = payload["factors"][0]
        for horizon in factor["horizons"].values():
            quantiles = copy.deepcopy(horizon["quantiles"])
            for quantile in quantiles:
                quantile.update(mean_return_bps=None, box_bps=None, violin=[],
                                cumulative_bps=[None] * len(horizon["dates"]))
            horizon["fee_scenarios"] = {"commission_stamp": {
                "quantiles": quantiles, "performance": {}, "sector": {"quantile_returns": []}}}
        charts._add_chart_options(payload)
        for series in factor["fee_chart_options"]["commission_stamp"]["mean-quantile"]["series"]:
            self.assertEqual(series["data"], [None, None])
        for horizon in factor["horizons"].values():
            common = horizon["fee_scenarios"]["commission_stamp"]["chart_options"]["common"]
            self.assertEqual(set(common), {"quantile-distribution", "quantile-cumulative"})
            self.assertEqual(common["quantile-distribution"]["series"][0]["data"], [[0, None, None], [1, None, None]])
            for series in common["quantile-cumulative"]["series"]:
                self.assertEqual(series["data"], [[date, None] for date in horizon["dates"]])
        json.dumps(payload, allow_nan=False)

    def test_real_components_strict_json_and_no_external_custom_plugins(self):
        payload = copy.deepcopy(self.original)
        constructed, dependencies = set(), set()
        original_dump = charts._dump

        def capture(chart):
            constructed.add(type(chart))
            dependencies.update(chart.js_dependencies.items)
            return original_dump(chart)

        with patch.object(charts, "_dump", side_effect=capture):
            charts._add_chart_options(payload)
        self.assertTrue({Bar, Line, Scatter, HeatMap, Custom}.issubset(constructed))
        self.assertEqual(dependencies, {"echarts"})
        encoded = json.dumps(payload, allow_nan=False)
        self.assertNotIn("--x_x--0_0--", encoded)
        self.assertNotIn("echarts-x-", encoded)
        self.assertNotIn("function(", encoded)
        self.assertNotIn("function (", encoded)
        self.assertEqual(json.loads(encoded), payload)
        custom = payload["factors"][0]["horizons"]["1"]["chart_options"]["common"]["quantile-distribution"]
        self.assertEqual(custom["series"][0]["type"], "custom")
        self.assertEqual(custom["series"][0]["encode"], {"x": 0, "y": [1, 2]})
        self.assertNotIn("renderItem", custom["series"][0])

    def test_common_sector_overview_contains_both_ic_series(self):
        for horizon in self.factor["horizons"].values():
            common = horizon["chart_options"]["common"]
            overview = horizon["sector"]["overview"]
            self.assertEqual([series["name"] for series in common["sector-ic"]["series"]],
                             ["Pearson IC", "Rank IC"])
            self.assertEqual(common["sector-ic"]["series"][0]["data"], [item["ic_mean"] for item in overview])
            self.assertEqual(common["sector-ic"]["series"][1]["data"], [item["rank_ic_mean"] for item in overview])
            for position, sector in enumerate(horizon["sector"]["quantile_returns"]):
                series = common[f"sector-{position}"]["series"][0]
                self.assertEqual([item_value(item) for item in series["data"]],
                                 [item["mean_return_bps"] for item in sector["groups"]])

    def test_all_missing_constant_factor_keeps_empty_and_null_chart_data(self):
        values = self.values.assign(alpha=1.)
        payload = _build_report_data(analyze(values, self.labels * np.nan))
        charts._add_chart_options(payload)
        factor = payload["factors"][0]
        self.assertEqual(factor["chart_options"]["mean-quantile"]["series"][0]["data"], [None, None])
        for horizon in factor["horizons"].values():
            common = horizon["chart_options"]["common"]
            self.assertEqual(common["quantile-distribution"]["series"][0]["data"], [[0, None, None], [1, None, None]])
            for metric in ("ic", "rank_ic"):
                options = horizon["chart_options"][metric]
                self.assertEqual(line_values(options["ic-cumulative"]), [[date, None] for date in horizon["dates"]])
                self.assertEqual(options["ic-histogram"]["series"][0]["data"], [])
                self.assertEqual(options["ic-qq"]["series"][0]["data"], [])
                self.assertEqual(len(options["ic-qq"]["series"]), 1)
                self.assertEqual(options["ic-monthly"]["series"][0]["data"], [])
        json.dumps(payload, allow_nan=False)

    def test_single_factor_public_html_contains_pyecharts_generated_options(self):
        with TemporaryDirectory() as temporary:
            path = self.result.to_html(Path(temporary) / "report.html", industries=self.industries)
            document = path.read_text(encoding="utf-8")
        match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', document, re.S)
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertEqual(payload["meta"]["renderer"], "pyecharts")
        self.assertEqual(payload["meta"]["pyecharts_version"], "2.1.0")
        self.assertEqual(len(payload["factors"]), 1)
        self.assertEqual(payload["factors"][0]["name"], "alpha")
        for horizon in payload["factors"][0]["horizons"].values():
            self.assertEqual(set(horizon["chart_options"]), {"common", "ic", "rank_ic"})
        self.assertEqual(payload["factors"][0]["chart_options"], self.factor["chart_options"])
        self.assertFalse(re.search(r'<script\b[^>]*\bsrc\s*=', document, re.I))


if __name__ == "__main__":
    unittest.main()
