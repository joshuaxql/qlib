"""Qlib alpha API contract, edge cases and integration with the report wrapper."""

import inspect
import unittest
import warnings

import numpy as np
import pandas as pd

from qlib.contrib.eva.alpha import (
    calc_ic, calc_all_ic, calc_long_short_prec, calc_long_short_return,
    pred_autocorr, pred_autocorr_all,
)
from qlib.contrib.report.analysis_model import analyze_factors


class OfficialAlphaTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-08"])
        self.index = pd.MultiIndex.from_product([self.dates, list("ABCDEF")], names=["datetime", "instrument"])
        self.pred = pd.Series(np.tile([1, 2, 3, 4, 5, 6], 3), index=self.index, dtype=float)
        self.label = pd.Series(np.tile([.01, .02, .03, .04, .05, .06], 3), index=self.index)

    def test_public_parameter_names_and_defaults(self):
        expected = {
            calc_ic: ("pred", "label", "date_col", "dropna"),
            calc_all_ic: ("pred_dict_all", "label", "date_col", "dropna", "n_jobs"),
            calc_long_short_return: ("pred", "label", "date_col", "quantile", "dropna"),
            calc_long_short_prec: ("pred", "label", "date_col", "quantile", "dropna", "is_alpha"),
            pred_autocorr: ("pred", "lag", "inst_col", "date_col"),
            pred_autocorr_all: ("pred_dict", "n_jobs", "kwargs"),
        }
        for function, names in expected.items():
            self.assertEqual(tuple(inspect.signature(function).parameters), names)
        self.assertEqual(inspect.signature(calc_long_short_return).parameters["quantile"].default, .2)
        self.assertFalse(inspect.signature(calc_ic).parameters["dropna"].default)
        self.assertEqual(inspect.signature(calc_all_ic).parameters["n_jobs"].default, -1)

    def test_ic_alignment_ties_two_samples_dropna_and_custom_date(self):
        pred = self.pred.copy()
        pred.loc[self.dates[0]] = [1, 1, 3, 4, np.nan, 6]
        pred.loc[self.dates[1]] = [1, 1, 1, 1, 1, 1]
        pred.loc[self.dates[2]] = [1, 2, np.nan, np.nan, np.nan, np.nan]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic, ric = calc_ic(pred, self.label.sample(frac=1, random_state=5))
        self.assertAlmostEqual(ric.iloc[0], np.corrcoef([1.5, 1.5, 3, 4, 5], [1, 2, 3, 4, 5])[0, 1])
        self.assertTrue(np.isnan(ic.iloc[1]))
        self.assertAlmostEqual(ic.iloc[2], 1)
        self.assertAlmostEqual(ric.iloc[2], 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean_ic, clean_ric = calc_ic(pred, self.label, dropna=True)
            renamed = calc_ic(pred.rename_axis(index={"datetime": "date"}),
                              self.label.rename_axis(index={"datetime": "date"}), date_col="date")
        pd.testing.assert_series_equal(clean_ic, ic.dropna())
        pd.testing.assert_series_equal(clean_ric, ric.dropna())
        pd.testing.assert_series_equal(renamed[0].rename_axis("datetime"), ic)
        # Outer alignment retains label-only dates, then returns NaN for those dates.
        extra = self.label.copy()
        extra.index = pd.MultiIndex.from_arrays([self.index.get_level_values(0) + pd.Timedelta(days=100),
                                                 self.index.get_level_values(1)], names=self.index.names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unaligned, _ = calc_ic(self.pred, pd.concat([self.label, extra]))
        self.assertEqual(len(unaligned), 6)
        self.assertTrue(unaligned.iloc[3:].isna().all())

    def test_long_short_half_spread_floor_count_ties_and_missing(self):
        spread, average = calc_long_short_return(self.pred, self.label)
        np.testing.assert_allclose(spread, .025)  # floor(6 * .2) == 1
        np.testing.assert_allclose(average, .035)
        pred, label = self.pred.copy(), self.label.copy()
        pred.loc[self.dates[0]] = [1, 2, 3, 4, 6, 6]
        label.loc[(self.dates[0], "E")] = .5
        # With top ties, nlargest keeps the first occurrence (E), not both.
        self.assertAlmostEqual(calc_long_short_return(pred, label)[0].iloc[0], (.5 - .01) / 2)
        pred.loc[(self.dates[0], "F")] = np.nan
        full_average = calc_long_short_return(pred, label)[1].iloc[0]
        paired_average = calc_long_short_return(pred, label, dropna=True)[1].iloc[0]
        self.assertAlmostEqual(full_average, (.01 + .02 + .03 + .04 + .5 + .06) / 6)
        self.assertAlmostEqual(paired_average, (.01 + .02 + .03 + .04 + .5) / 5)
        empty_sides, _ = calc_long_short_return(self.pred, self.label, quantile=.1)
        self.assertTrue(empty_sides.isna().all())

    def test_long_short_precision_and_alpha_demeaning(self):
        long, short = calc_long_short_prec(self.pred, self.label)
        np.testing.assert_allclose(long, 1)
        np.testing.assert_allclose(short, 0)
        long, short = calc_long_short_prec(self.pred, self.label, is_alpha=True)
        np.testing.assert_allclose(long, 1)
        np.testing.assert_allclose(short, 1)
        pred = self.pred[self.pred.index.get_level_values("instrument") != "F"]
        with self.assertRaisesRegex(ValueError, "more instruments"):
            calc_long_short_prec(pred, self.label.reindex(pred.index))

    def test_autocorr_pearson_not_rank_sparse_dates_and_frame(self):
        pred = self.pred.copy()
        pred.loc[self.dates[1]] = [1, 4, 9, 16, 25, 36]
        actual = pred_autocorr(pred)
        expected = np.corrcoef([1, 2, 3, 4, 5, 6], [1, 4, 9, 16, 25, 36])[0, 1]
        self.assertTrue(np.isnan(actual.iloc[0]))
        self.assertAlmostEqual(actual.iloc[1], expected)
        self.assertAlmostEqual(actual.iloc[2], expected)
        self.assertNotAlmostEqual(actual.iloc[1], 1)
        self.assertAlmostEqual(pred_autocorr(pred, lag=2).iloc[2], 1)
        with self.assertLogs("pred_autocorr", level="WARNING"):
            frame_result = pred_autocorr(pd.DataFrame({"score": pred, "ignored": -pred}))
        pd.testing.assert_series_equal(actual, frame_result)
        renamed = pred.rename_axis(index={"instrument": "asset", "datetime": "date"})
        pd.testing.assert_series_equal(pred_autocorr(renamed, inst_col="asset", date_col="date"), actual)

    def test_batch_outputs_serial_and_parallel(self):
        predictions = {"positive": self.pred, "negative": -self.pred}
        for n_jobs in (1, 2):
            results = calc_all_ic(predictions, self.label, n_jobs=n_jobs)
            autocorr = pred_autocorr_all(predictions, n_jobs=n_jobs, lag=2)
            self.assertEqual(list(results), list(predictions))
            for name, pred in predictions.items():
                expected_ic, expected_ric = calc_ic(pred, self.label)
                pd.testing.assert_series_equal(results[name]["ic"], expected_ic)
                pd.testing.assert_series_equal(results[name]["ric"], expected_ric)
                pd.testing.assert_series_equal(autocorr[name], pred_autocorr(pred, lag=2))
        self.assertEqual(calc_all_ic({}, self.label, n_jobs=1), {})

    def test_report_uses_official_functions(self):
        pred = self.pred.copy()
        pred.loc[self.dates[1]] = [1, 4, 9, 16, 25, 36]
        pred.loc[self.dates[2]] = [6, 5, 4, 3, 2, 1]
        # Report panels use instrument-first sorted order; tie ordering is significant.
        values = pred.rename("alpha").swaplevel().sort_index()
        labels = self.label.rename(1).swaplevel().sort_index()
        result = analyze_factors(values, labels)
        ic, ric = calc_ic(values, labels)
        spread, average = calc_long_short_return(values, labels)
        daily = result.daily.loc[("alpha", 1)]
        for column, expected in (("ic", ic), ("rank_ic", ric), ("long_short_return", spread), ("universe_return", average)):
            np.testing.assert_allclose(daily[column], expected, equal_nan=True)
        np.testing.assert_allclose(result.autocorrelation.loc["alpha"].autocorrelation,
                                   pred_autocorr(values), equal_nan=True)
        self.assertAlmostEqual(result.summary.loc[("alpha", 1), "ic_ir"], ic.mean() / ic.std())


if __name__ == "__main__":
    unittest.main()
