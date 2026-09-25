"""Historical exposure alignment and independent within-industry OLS checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from qlib.data import LocalProvider
from qlib.contrib.report.analysis_model import factor_analysis, neutralize_factors


class NeutralizationTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dates = pd.bdate_range("2025-01-02", periods=4)
        self.codes = [f"{i:06d}.SZ" for i in range(1, 9)]
        for folder in ("calendars", "instruments", "industry"):
            (self.root / folder).mkdir()
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        (self.root / "instruments/all.txt").write_text("\n".join(
            f"{code} {self.dates[0]:%Y-%m-%d} {self.dates[-1]:%Y-%m-%d}" for code in self.codes))
        first, second = [], []
        for i, code in enumerate(self.codes):
            end = self.dates[1] if i == 3 else self.dates[-1]
            (first if i < 4 else second).append(f"{code} {self.dates[0]:%Y-%m-%d} {end:%Y-%m-%d}")
        second.append(f"{self.codes[3]} {self.dates[2]:%Y-%m-%d} {self.dates[-1]:%Y-%m-%d}")
        (self.root / "industry/801010.SI.txt").write_text("\n".join(first))
        (self.root / "industry/801020.SI.txt").write_text("\n".join(second))
        self.noise = np.array([1, -2, 1, 0, 2, -1, -4, 3], dtype=float)
        self.log_caps = np.log(np.exp(np.tile([1., 2., 3., 4.], 2)).astype("<f4").astype(float))
        self.groups = np.tile(np.repeat([0, 1], 4)[:, None], (1, 4))
        self.groups[3, 2:] = 1
        for i, code in enumerate(self.codes):
            for field, values in (
                ("total_mv", np.exp(self.log_caps[i]) * np.ones(4)),
                ("circ_mv", np.exp(self.log_caps[i]) * np.ones(4) / 2),
                ("open", np.arange(10., 14.) + i), ("close", np.arange(11., 15.) + i),
                ("factor", np.ones(4)), ("score", 10 * self.groups[i] + 2 * self.log_caps[i] + self.noise[i]),
            ):
                self.write(code, field, values)
        self.provider = LocalProvider(self.root)
        index = pd.MultiIndex.from_product([self.codes, self.dates], names=["instrument", "datetime"])
        values = 10 * self.groups + 2 * self.log_caps[:, None] + self.noise[:, None]
        self.factors = pd.DataFrame({"alpha": values.ravel(), "other": values.ravel() ** 2}, index=index)

    def write(self, code, field, values):
        directory = self.root / "features" / code
        directory.mkdir(parents=True, exist_ok=True)
        np.asarray([0, *values], dtype="<f4").tofile(directory / f"{field}.day.bin")

    @staticmethod
    def oracle(y, x, groups):
        # Frisch-Waugh-Lovell: remove industry means before a one-variable slope.
        y, x = y.copy(), x.copy()
        for group in np.unique(groups):
            rows = groups == group
            y[rows] -= y[rows].mean()
            x[rows] -= x[rows].mean()
        return y - x * (np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else y

    def test_joint_residual_matches_independent_oracle_and_orthogonality(self):
        original = self.factors.copy()
        result = neutralize_factors(self.factors, provider=self.provider)
        for day, date in enumerate(self.dates):
            for field in self.factors:
                y = self.factors.xs(date, level="datetime")[field].to_numpy()
                actual = result.xs(date, level="datetime")[field].to_numpy()
                expected = self.oracle(y, self.log_caps, self.groups[:, day])
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                self.assertAlmostEqual(np.dot(actual, self.log_caps), 0, places=10)
                for group in (0, 1):
                    self.assertAlmostEqual(actual[self.groups[:, day] == group].sum(), 0, places=10)
        np.testing.assert_allclose(result.xs(self.dates[0], level="datetime").alpha, self.noise, atol=1e-7)
        pd.testing.assert_frame_equal(self.factors, original)

    def test_historical_membership_series_and_order_alignment(self):
        shuffled = self.factors.alpha.sample(frac=1, random_state=7).swaplevel()
        actual = neutralize_factors(shuffled, provider=self.provider)
        expected = neutralize_factors(self.factors[["alpha"]], provider=self.provider)
        pd.testing.assert_frame_equal(actual, expected)
        early = neutralize_factors(self.factors.loc[(slice(None), self.dates[:2]), :], provider=self.provider)
        pd.testing.assert_frame_equal(early, expected.loc[early.index].join(
            neutralize_factors(self.factors[["other"]], provider=self.provider).loc[early.index]))
        # Change only future caps; earlier residuals must remain identical.
        self.write(self.codes[0], "total_mv", [np.exp(self.log_caps[0])] * 2 + [1e6, 1e9])
        self.provider.clear_cache()
        changed = neutralize_factors(self.factors, provider=self.provider)
        pd.testing.assert_frame_equal(early, changed.loc[early.index])

    def test_missing_exposures_and_factor_specific_samples(self):
        (self.root / "industry/801020.SI.txt").write_text("\n".join(
            line for line in (self.root / "industry/801020.SI.txt").read_text().splitlines()
            if not line.startswith(self.codes[-1])))
        self.write(self.codes[0], "total_mv", [0, -1, np.nan, np.inf])
        self.factors.loc[(self.codes[1], self.dates[0]), "alpha"] = np.nan
        self.factors.loc[(self.codes[2], self.dates[0]), "alpha"] = np.inf
        result = neutralize_factors(self.factors, provider=self.provider)
        self.assertTrue(result.loc[[self.codes[0], self.codes[-1]]].isna().all().all())
        day = result.xs(self.dates[0], level="datetime")
        self.assertTrue(day.loc[self.codes[1:3], "alpha"].isna().all())
        self.assertTrue(day.loc[self.codes[1:3], "other"].notna().all())
        valid = [3, 4, 5, 6]
        expected = self.oracle(self.factors.xs(self.dates[0], level="datetime").alpha.iloc[valid].to_numpy(),
                               self.log_caps[valid], self.groups[valid, 0])
        np.testing.assert_allclose(day.alpha.iloc[valid], expected, atol=1e-12)

    def test_constant_and_collinear_controls_and_explained_factors(self):
        constant = self.factors.assign(constant=7., explained=np.repeat(self.log_caps, 4))
        result = neutralize_factors(constant, provider=self.provider)
        np.testing.assert_array_equal(result.constant, 0.)
        np.testing.assert_array_equal(result.explained, 0.)
        for i, code in enumerate(self.codes):
            self.write(code, "total_mv", np.where(self.groups[i] == 0, 100, 200))
        self.provider.clear_cache()
        result = neutralize_factors(self.factors, provider=self.provider)
        for day, date in enumerate(self.dates):
            y = self.factors.xs(date, level="datetime").alpha.to_numpy()
            expected = y.copy()
            for group in (0, 1):
                mask = self.groups[:, day] == group
                expected[mask] -= y[mask].mean()
            np.testing.assert_allclose(result.xs(date, level="datetime").alpha, expected, atol=1e-12)

    def test_small_universe_degrees_of_freedom_and_minimum(self):
        subset = self.factors.loc[self.codes[:2] + self.codes[4:5]]
        result = neutralize_factors(subset, provider=self.provider, min_samples=2)
        self.assertTrue(result.isna().all().all())  # N=3, rank=3.
        self.assertTrue(neutralize_factors(self.factors, provider=self.provider, min_samples=9).isna().all().all())
        one_industry = self.factors.loc[self.codes[:3]]
        result = neutralize_factors(one_industry, provider=self.provider)
        self.assertTrue(result.notna().all().all())
        expected = self.oracle(one_industry.xs(self.dates[0], level="datetime").alpha.to_numpy(),
                               self.log_caps[:3], np.zeros(3))
        np.testing.assert_allclose(result.xs(self.dates[0], level="datetime").alpha, expected, atol=1e-12)

    def test_cap_choice_empty_and_invalid_inputs(self):
        total = neutralize_factors(self.factors, provider=self.provider)
        circ = neutralize_factors(self.factors, provider=self.provider, market_cap="circ_mv")
        np.testing.assert_allclose(total, circ, atol=1e-12)
        self.assertTrue(neutralize_factors(self.factors.iloc[:0], provider=self.provider).empty)
        for kwargs in ({"min_samples": True}, {"min_samples": 1}, {"market_cap": "close"}):
            with self.assertRaises(ValueError):
                neutralize_factors(self.factors, provider=self.provider, **kwargs)
        for frame in (self.factors.reset_index(), pd.concat([self.factors, self.factors]),
                      self.factors.rename(index={self.dates[0]: pd.Timestamp("2025-01-01")})):
            with self.assertRaises(ValueError):
                neutralize_factors(frame, provider=self.provider)

    def test_ambiguous_industry_and_missing_sources_raise(self):
        path = self.root / "industry/801020.SI.txt"
        original = path.read_text()
        path.write_text(original + f"\n{self.codes[0]} 2025-01-02 2025-01-02")
        with self.assertRaisesRegex(ValueError, "Multiple historical industry"):
            neutralize_factors(self.factors, provider=self.provider)
        for path in (self.root / "industry").glob("*.txt"):
            path.unlink()
        self.provider.clear_cache()
        with self.assertRaisesRegex(ValueError, "requires historical"):
            neutralize_factors(self.factors, provider=self.provider)

    def test_report_integration_labels_and_saved_config(self):
        options = dict(provider=self.provider, horizons=(1,), quantiles=2)
        raw = factor_analysis(self.codes, {"alpha": "$score"}, **options)
        result = factor_analysis(self.codes, {"alpha": "$score"}, neutralize=True, **options)
        expected = neutralize_factors(raw.factors, provider=self.provider)
        pd.testing.assert_frame_equal(result.factors, expected)
        pd.testing.assert_frame_equal(result.forward_returns, raw.forward_returns)
        self.assertIsNone(raw.config["neutralization"])
        result.save(self.root / "report")
        config = json.loads((self.root / "report/config.json").read_text())
        self.assertEqual(config["neutralization"], {"method": "industry_log_market_cap", "market_cap": "total_mv", "min_samples": 3})
        with self.assertRaises(ValueError):
            factor_analysis(self.codes, "$score", neutralize="yes", **options)

    def test_missing_cap_source_raises_and_missing_stock_cap_is_nan(self):
        (self.root / "features" / self.codes[0] / "total_mv.day.bin").unlink()
        result = neutralize_factors(self.factors, provider=self.provider)
        self.assertTrue(result.loc[self.codes[0]].isna().all().all())
        for path in (self.root / "features").glob("*/total_mv.day.bin"):
            path.unlink()
        self.provider.clear_cache()
        with self.assertRaises(KeyError):
            neutralize_factors(self.factors, provider=self.provider)


if __name__ == "__main__":
    unittest.main()
