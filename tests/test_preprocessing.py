"""Independent cross-sectional clipping/Z-score examples and edge cases."""

import unittest

import numpy as np
import pandas as pd

from qlib.contrib.report.analysis_model import preprocess_factors, standardize_factors, winsorize_factors


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-02", periods=2)
        self.index = pd.MultiIndex.from_product([list("ABCDE"), self.dates], names=["instrument", "datetime"])
        self.frame = pd.DataFrame({"a": np.array([[0, 1000], [1, 1100], [2, 1200], [3, 1300], [100, 1400]]).ravel(),
                                   "b": np.array([[2, np.nan], [2, np.inf], [2, -np.inf], [2, 7], [100, 7]]).ravel()},
                                  index=self.index)

    def test_mean_std_clipping_with_explicit_ddof(self):
        original = self.frame.copy()
        x = np.array([0., 1., 2., 3., 100.])
        for ddof in (0, 1):
            result = winsorize_factors(self.frame, method="std", n=1, ddof=ddof)
            expected = np.clip(x, x.mean() - x.std(ddof=ddof), x.mean() + x.std(ddof=ddof))
            np.testing.assert_allclose(result.xs(self.dates[0], level="datetime").a, expected, atol=1e-12)
            self.assertEqual(result.notna().sum().a, 10)  # cap, not discard
        pd.testing.assert_frame_equal(self.frame, original)

    def test_median_mad_scaling_and_zero_mad(self):
        result = winsorize_factors(self.frame, method="mad", n=3)
        np.testing.assert_allclose(result.xs(self.dates[0], level="datetime").a, [0, 1, 2, 3, 6.4478])
        np.testing.assert_array_equal(result.xs(self.dates[0], level="datetime").b, [2] * 5)
        unscaled = winsorize_factors(self.frame, method="mad", n=3, mad_scale=1)
        np.testing.assert_allclose(unscaled.xs(self.dates[0], level="datetime").a, [0, 1, 2, 3, 5])
        # ddof does not affect median/MAD.
        pd.testing.assert_frame_equal(result, winsorize_factors(self.frame, method="mad", ddof=100))

    def test_zscore_mean_variance_and_columnwise_missing_values(self):
        for ddof in (0, 1):
            result = standardize_factors(self.frame, ddof=ddof)
            for date in self.dates:
                x = self.frame.xs(date, level="datetime").a.to_numpy(dtype=float)
                actual = result.xs(date, level="datetime").a
                np.testing.assert_allclose(actual, (x - x.mean()) / x.std(ddof=ddof), atol=1e-14)
                self.assertAlmostEqual(actual.mean(), 0)
                self.assertAlmostEqual(actual.var(ddof=ddof), 1)
            b = result.xs(self.dates[1], level="datetime").b
            self.assertTrue(b.iloc[:3].isna().all())
            np.testing.assert_array_equal(b.iloc[3:], [0, 0])

    def test_singleton_all_missing_empty_and_insufficient_samples(self):
        frame = self.frame.loc[["A"]]
        np.testing.assert_array_equal(standardize_factors(frame).a, [0, 0])
        self.assertTrue(standardize_factors(frame, ddof=1).isna().all().all())
        self.assertTrue(winsorize_factors(frame, method="std", ddof=1).isna().all().all())
        for function in (winsorize_factors, standardize_factors, preprocess_factors):
            options = {"neutralize": False} if function is preprocess_factors else {}
            self.assertTrue(function(self.frame * np.nan, **options).isna().all().all())
            empty = function(self.frame.iloc[:0], **options)
            self.assertTrue(empty.empty)
            self.assertEqual(empty.columns.tolist(), ["a", "b"])

    def test_order_series_and_date_locality(self):
        shuffled = self.frame.a.sample(frac=1, random_state=2).swaplevel()
        original = shuffled.copy()
        result = preprocess_factors(shuffled, winsorize="mad", neutralize=False)
        pd.testing.assert_frame_equal(result, preprocess_factors(self.frame[["a"]], winsorize="mad", neutralize=False))
        pd.testing.assert_series_equal(shuffled, original)
        x = np.array([0, 1, 2, 3, 6.4478])
        np.testing.assert_allclose(result.xs(self.dates[0], level="datetime").a, (x - x.mean()) / x.std())
        future = self.frame.copy()
        future.loc[(slice(None), self.dates[1]), :] = 1e9
        changed = preprocess_factors(future, winsorize="mad", neutralize=False)
        pd.testing.assert_frame_equal(result.xs(self.dates[0], level="datetime"),
                                      changed[["a"]].xs(self.dates[0], level="datetime"))

    def test_large_finite_values_do_not_overflow_moments(self):
        frame = self.frame[["a"]].astype(float)
        frame.loc[(slice(None), self.dates[0]), "a"] = [-1e308, -5e307, 0, 5e307, 1e308]
        result = standardize_factors(frame)
        np.testing.assert_allclose(result.xs(self.dates[0], level="datetime").a,
                                   np.array([-2, -1, 0, 1, 2]) / np.sqrt(2))
        clipped = winsorize_factors(frame, method="std", n=1)
        self.assertTrue(np.isfinite(clipped).all().all())
        np.testing.assert_allclose(clipped.xs(self.dates[0], level="datetime").a / 1e308,
                                   [-1 / np.sqrt(2), -.5, 0, .5, 1 / np.sqrt(2)])

    def test_invalid_options_and_panels(self):
        for kwargs in ({"method": "unknown"}, {"n": 0}, {"n": True}, {"n": np.inf},
                       {"n": "3"}, {"mad_scale": -1}, {"mad_scale": np.nan}, {"ddof": -1}, {"ddof": True}):
            with self.assertRaises(ValueError):
                winsorize_factors(self.frame, **kwargs)
        for value in (-1, True, .5, np.nan):
            with self.assertRaises(ValueError):
                standardize_factors(self.frame, ddof=value)
        for kwargs in ({"winsorize": True}, {"standardize": "yes"}, {"neutralize": 1}, {"winsorize_n": 0}):
            with self.assertRaises(ValueError):
                preprocess_factors(self.frame, **kwargs)
        for frame in (self.frame.reset_index(), pd.concat([self.frame, self.frame])):
            with self.assertRaises(ValueError):
                standardize_factors(frame)


if __name__ == "__main__":
    unittest.main()
