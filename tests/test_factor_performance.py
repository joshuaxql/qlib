"""Independent array-evaluator equivalence: pandas selection/ranks, Decimal IC."""

from decimal import Decimal, localcontext
import unittest
import warnings

import numpy as np
import pandas as pd

from qlib.contrib.eva.alpha import (
    calc_ic, calc_long_short_prec, calc_long_short_return, pred_autocorr, _selected_positions,
)
from qlib.contrib.report.analysis_model import analyze_factors


def pearson_oracle(left, right):
    left, right = left.align(right, join="inner")
    mask = left.notna() & right.notna()
    x, y = left[mask].to_numpy(dtype=float), right[mask].to_numpy(dtype=float)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan
    with localcontext() as context:
        context.prec = 400
        x, y = ([Decimal.from_float(float(v)) for v in a] for a in (x, y))
        x = [v - sum(x) / len(x) for v in x]
        y = [v - sum(y) / len(y) for v in y]
        xx, yy = sum(v * v for v in x), sum(v * v for v in y)
        return float(sum(a * b for a, b in zip(x, y)) / xx.sqrt() / yy.sqrt()) if xx and yy else np.nan


def selected(frame, fraction, largest):
    eligible = frame[np.isfinite(frame.pred)]
    n = int(len(eligible) * fraction)
    return (eligible.nlargest(n, "pred") if largest else eligible.nsmallest(n, "pred")).label


class FactorPerformanceTest(unittest.TestCase):
    def setUp(self):
        self.days = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-08", "2025-01-13"], utc=True)
        self.index = pd.MultiIndex.from_product([self.days, list("ABCDEFGHIJKL")], names=["datetime", "instrument"])
        random = np.random.default_rng(38)
        self.pred = pd.Series(random.integers(-2, 4, len(self.index)).astype(float), index=self.index)
        self.label = pd.Series(random.normal(size=len(self.index)), index=self.index)
        self.pred.iloc[[0, 9, 13, 19]] = [np.nan, np.inf, -np.inf, np.nan]
        self.label.iloc[[4, 8, 18, 29]] = np.nan

    def test_selection_matches_pandas_at_tie_boundary_and_extreme_values(self):
        random = np.random.default_rng(93)
        values = random.choice([np.nan, np.inf, -np.inf, -0.0, 0.0, 1, 2, 3, -1e308, 1e308, 1e-300], 257)
        integers = np.array([np.iinfo("int64").min, 2**62, 2**62 + 1, 2**62 + 2,
                             np.iinfo("int64").max, 0, 0, -1], dtype="int64")
        with np.errstate(over="ignore"):
            float32 = values.astype("float32")
        for sample in (values, float32, integers):
            eligible = pd.Series(sample)[np.isfinite(sample)]
            for fraction in (0, .01, .2, .5, 1, 1.2, -.2):
                for largest in (False, True):
                    n = int(len(eligible) * fraction)
                    expected = eligible.nlargest(n) if largest else eligible.nsmallest(n)
                    with self.subTest(dtype=sample.dtype, fraction=fraction, largest=largest):
                        np.testing.assert_array_equal(_selected_positions(sample, fraction, largest), expected.index)

    def test_ic_uses_horizon_specific_pairs_and_orders_infinity_for_ranks(self):
        pred = self.pred.copy()
        label = self.label.copy()
        label.iloc[0] = np.inf
        label.iloc[1] = np.inf
        extra = pd.Series([1., 2.], index=pd.MultiIndex.from_product(
            [[self.days[-1] + pd.Timedelta(days=20)], list("AB")], names=pred.index.names))
        label = pd.concat([label, extra]).sample(frac=1, random_state=4)
        frame = pd.DataFrame({"pred": pred, "label": label})
        expected_ic, expected_rank = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _, day in frame.groupby(level="datetime"):
                expected_ic.append(pearson_oracle(day.pred, day.label))
                expected_rank.append(day.pred.corr(day.label, method="spearman"))
            actual = calc_ic(pred, label)
        dates = frame.groupby(level="datetime").size().index
        for output, values in zip(actual, (expected_ic, expected_rank)):
            pd.testing.assert_series_equal(output, pd.Series(values, index=dates), rtol=1e-13, atol=1e-13)
        clean = calc_ic(pred, label, dropna=True)
        for output, full in zip(clean, actual):
            pd.testing.assert_series_equal(output, full.dropna())

    def test_nullable_numeric_inputs_retain_pairing_and_missing_behavior(self):
        pred, label = self.pred.astype("Float64"), self.label.astype("Float64")
        frame = pd.DataFrame({"pred": pred, "label": label})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_ic, actual_ric = calc_ic(pred, label)
            wanted_ic = frame.groupby(level="datetime").apply(lambda day: pearson_oracle(day.pred, day.label))
            wanted_ric = frame.groupby(level="datetime").apply(lambda day: day.pred.corr(day.label, method="spearman"))
        pd.testing.assert_series_equal(actual_ic, wanted_ic, rtol=1e-13, atol=1e-13)
        pd.testing.assert_series_equal(actual_ric, wanted_ric, rtol=1e-13, atol=1e-13)
        pd.testing.assert_series_equal(pred_autocorr(pred), pred_autocorr(self.pred))
        for function in (calc_long_short_return, calc_long_short_prec):
            for actual, expected in zip(function(pred, label), function(self.pred, self.label)):
                pd.testing.assert_series_equal(actual, expected, check_dtype=False)

    def test_ic_and_autocorr_keep_large_offsets_and_tiny_scales(self):
        base = np.array([0., 1., 4., 10.])
        index = pd.MultiIndex.from_product([self.days, list("ABCD")], names=self.index.names)
        for scale, offset in ((1, 1e15), (1e-300, 0), (1e306, 0)):
            values = np.array([base, base[::-1], base ** 2, base + 4]) * scale + offset
            pred = pd.Series(values.ravel(), index=index)
            labels = pd.Series(np.tile(base ** 2 * 1e-300, 4), index=index)
            actual_ic, _ = calc_ic(pred, labels)
            expected_ic = [pearson_oracle(pred.xs(d), labels.xs(d)) for d in self.days]
            np.testing.assert_allclose(actual_ic, expected_ic, atol=1e-13, rtol=1e-13, equal_nan=True)
            wide = pred.unstack("instrument")
            for lag in (-7, -1, 0, 1, 2, 7):
                shifted = wide.shift(lag)
                expected = pd.Series({d: pearson_oracle(wide.loc[d], shifted.loc[d]) for d in self.days})
                pd.testing.assert_series_equal(pred_autocorr(pred.sample(frac=1, random_state=9), lag=lag),
                                               expected, rtol=1e-13, atol=1e-13)

    def test_returns_precision_and_alpha_match_selection_oracle(self):
        for dropna in (False, True):
            frame = pd.DataFrame({"pred": self.pred, "label": self.label})
            if dropna:
                frame = frame.dropna()
            groups = frame.groupby(level="datetime")
            expected_spread = groups.apply(lambda day: (selected(day, .2, True).mean() - selected(day, .2, False).mean()) / 2)
            spread, average = calc_long_short_return(self.pred, self.label, dropna=dropna)
            pd.testing.assert_series_equal(spread, expected_spread)
            pd.testing.assert_series_equal(average, groups.label.mean())
            for is_alpha in (False, True):
                labels = self.label - self.label.groupby(level="datetime").mean() if is_alpha else self.label
                frame = pd.DataFrame({"pred": self.pred, "label": labels})
                if dropna:
                    frame = frame.dropna()
                expected = [frame.groupby(level="datetime").apply(
                    lambda day: (selected(day, .2, True).dropna() > 0).mean()),
                    frame.groupby(level="datetime").apply(
                    lambda day: (selected(day, .2, False).dropna() < 0).mean())]
                for actual, wanted in zip(calc_long_short_prec(self.pred, self.label, dropna=dropna, is_alpha=is_alpha), expected):
                    pd.testing.assert_series_equal(actual, wanted)

    def test_report_all_daily_group_membership_and_turnover_tables(self):
        index = self.index.swaplevel().sort_values()
        factors = pd.DataFrame({"ties": self.pred.reindex(index), "other": -self.pred.reindex(index)}, index=index)
        factors["constant"] = 1.
        labels = pd.DataFrame({1: self.label.reindex(index), 5: -self.label.reindex(index)}, index=index)
        labels.loc[("L", self.days[0]), 5] = np.nan
        labels.loc[("K", self.days[1]), 1] = np.nan
        # A changing universe must retain set-based turnover and per-date counts.
        factors = factors.drop(index=[("A", self.days[2]), ("B", self.days[3])])
        labels = labels.reindex(factors.index)
        clean = factors.replace([np.inf, -np.inf], np.nan)
        for quantiles, min_samples, lag in ((3, 2, 1), (8, 5, 2)):
            actual = analyze_factors(factors.sample(frac=1, random_state=4), labels, quantiles=quantiles,
                                     min_samples=min_samples, turnover_lag=lag)
            members = pd.DataFrame(np.nan, index=clean.index, columns=clean.columns)
            daily, grouped, turns = [], [], []
            for name in clean:
                history = []
                for date in self.days:
                    values = clean[name].xs(date, level="datetime")
                    valid = values.dropna()
                    groups = pd.Series(np.nan, index=values.index)
                    if valid.nunique() >= quantiles:
                        groups.loc[valid.index] = np.floor((valid.rank(method="average") - 1) * quantiles / len(valid)) + 1
                    for code, group in groups.items():
                        members.loc[(code, date), name] = group
                    previous = history[-lag] if len(history) >= lag else None
                    for group in range(1, quantiles + 1):
                        current = set(groups.index[groups == group])
                        old = set() if previous is None else set(previous.index[previous == group])
                        turns.append((name, date, group, len(current - old) / len(current) if current and old else np.nan))
                    history.append(groups)
                    for horizon in labels:
                        target = labels[horizon].xs(date, level="datetime")
                        paired = values.notna() & target.notna()
                        count = int(paired.sum())
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ic = pearson_oracle(values, target) if count >= min_samples else np.nan
                            ric = values.corr(target, method="spearman") if count >= min_samples else np.nan
                        selection = pd.DataFrame({"pred": values, "label": target})
                        spread = (selected(selection, 1 / quantiles, True).mean() - selected(selection, 1 / quantiles, False).mean()) / 2
                        daily.append((name, horizon, date, len(values), int(values.notna().sum()), count,
                                      values.notna().mean(), count / len(values), ic, ric, target.mean(), spread))
                        for group in range(1, quantiles + 1):
                            sample = target[(groups == group) & paired]
                            grouped.append((name, horizon, date, group, len(sample), sample.mean(), sample.std(ddof=1)))
            daily = pd.DataFrame(daily, columns=["factor", "horizon", "datetime", *actual.daily.columns]).set_index(
                ["factor", "horizon", "datetime"]).sort_index()
            grouped = pd.DataFrame(grouped, columns=["factor", "horizon", "datetime", "quantile", "count", "mean", "std"]).set_index(
                ["factor", "horizon", "datetime", "quantile"]).sort_index()
            turns = pd.DataFrame(turns, columns=["factor", "datetime", "quantile", "turnover"]).set_index(
                ["factor", "datetime", "quantile"]).sort_index()
            pd.testing.assert_frame_equal(actual.daily, daily, rtol=1e-13, atol=1e-13)
            pd.testing.assert_frame_equal(actual.quantile_returns, grouped, rtol=1e-13, atol=1e-13)
            pd.testing.assert_frame_equal(actual.quantile_membership, members)
            pd.testing.assert_frame_equal(actual.turnover, turns)
            # Future labels influence returns and paired IC only, never membership/turnover.
            missing = analyze_factors(factors, labels * np.nan, quantiles=quantiles,
                                      min_samples=min_samples, turnover_lag=lag)
            pd.testing.assert_frame_equal(actual.quantile_membership, missing.quantile_membership)
            pd.testing.assert_frame_equal(actual.turnover, missing.turnover)
            pd.testing.assert_frame_equal(actual.autocorrelation, missing.autocorrelation)

    def test_perfect_rank_series_preserves_infinite_ir(self):
        values = pd.Series(np.tile(np.arange(12, dtype=float), 4), index=self.index)
        factors = values.rename("ordered").swaplevel().sort_index()
        labels = (values ** 2).rename(1).swaplevel().sort_index()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_factors(factors, labels)
        expected = values.xs(self.days[0]).corr((values ** 2).xs(self.days[0]), method="spearman")
        np.testing.assert_array_equal(result.daily.rank_ic, expected)
        self.assertTrue(np.isposinf(result.summary.rank_ic_ir.iloc[0]))


if __name__ == "__main__":
    unittest.main()
