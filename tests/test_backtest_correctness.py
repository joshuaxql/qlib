"""Strategy signal causality and valid risk annualization parameters."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from qlib.backtest import BacktestEngine, ExchangeConfig
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy, TopkStrategy
from qlib.data import LocalProvider
from qlib.data.filter import ExpressionFilter

A, B = "000001.SZ", "000002.SZ"


class BacktestCorrectnessTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dates = pd.bdate_range("2024-01-02", periods=7, name="datetime")
        for directory in ("calendars", "instruments", f"features/{A}", f"features/{B}"):
            (self.root / directory).mkdir(parents=True)
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        (self.root / "instruments/all.txt").write_text(
            f"{A} 2024-01-02 2024-01-31\n{B} 2024-01-02 2024-01-31"
        )
        pd.DataFrame({"ts_code": [A, B], "list_date": ["2020-01-01"] * 2}).to_csv(
            self.root / "stock_basic.csv", index=False,
        )
        for field in ("open", "close"):
            self.write(A, field, [10, 10, 10, 5, 5, 2.5, 2.5])
            self.write(B, field, [8] * 7)
        self.write(A, "factor", [1, 1, 1, 2, 2, 4, 4])
        self.write(B, "factor", [1] * 7)
        for code in (A, B):
            self.write(code, "volume", [10000] * 7)
        self.provider = LocalProvider(self.root, adjust="qfq")
        self.engine = BacktestEngine(
            self.provider, initial_cash=10000,
            exchange=ExchangeConfig(lot_size=1, buy_cost=0, sell_cost=0, min_cost=0, sell_tax=0),
        )

    def write(self, code, field, values):
        np.asarray([0, *values], dtype="<f4").tofile(self.root / "features" / code / f"{field}.day.bin")

    def assert_prefix(self, strategy):
        short = self.engine.run(strategy, self.dates[0], self.dates[2])
        long = self.engine.run(strategy, self.dates[0], self.dates[-1])
        self.assertEqual(short.trades.iloc[0].instrument, A)
        self.assertEqual(long.trades.iloc[0].instrument, A)
        pd.testing.assert_frame_equal(short.report, long.report.loc[:self.dates[2]])
        # Later factor adjustments can promote the complete quantity column to
        # float; the historical trade values themselves must remain identical.
        pd.testing.assert_frame_equal(
            short.trades, long.trades[long.trades.datetime <= self.dates[2]], check_dtype=False,
        )

    def test_qfq_expression_strategies_ignore_future_splits(self):
        for strategy in (
            TopkStrategy(score="$close", topk=1),
            TopkDropoutStrategy(score="Mean($close + 1, 3)", topk=1, n_drop=1),
        ):
            with self.subTest(strategy=type(strategy).__name__):
                self.assert_prefix(strategy)

    def test_qfq_filters_preserve_expression_and_external_signal_prefixes(self):
        pool = self.provider.instruments("all", [ExpressionFilter("Mean($close + 1, 3) > 7")])
        external = pd.DataFrame({A: [2.] * 7, B: [1.] * 7}, index=self.dates)
        for strategy in (
            TopkStrategy(score="$close", topk=1, instruments=pool),
            TopkDropoutStrategy(score="$close", topk=1, n_drop=1, instruments=pool),
            TopkDropoutStrategy(signal=external, topk=1, n_drop=1, instruments=pool),
            TopkDropoutStrategy(signal=external.rename_axis(columns="instrument").stack(),
                                topk=1, n_drop=1, instruments=pool),
        ):
            with self.subTest(strategy=type(strategy).__name__, external=strategy.signal is not None
                             if isinstance(strategy, TopkDropoutStrategy) else False):
                self.assert_prefix(strategy)

    def test_signal_anchors_keep_nested_history_and_ordinary_queries_compatible(self):
        end = self.dates[-1]
        ordinary = self.provider.features([A], ["$close"], self.dates[0], end, allow_future=False)
        self.assertTrue(ordinary.iloc[:, 0].eq(2.5).all())
        signals = self.provider._signal_view().features(
            [A], ["$close", "Mean($close + 1, 3)", "Sum($close > 7, 0)"],
            self.dates[0], end, allow_future=False,
        ).loc[A]
        np.testing.assert_allclose(signals["$close"], [10, 10, 10, 5, 5, 2.5, 2.5])
        np.testing.assert_allclose(signals["Mean($close + 1, 3)"], [11, 11, 11, 6, 6, 3.5, 3.5])
        np.testing.assert_allclose(signals["Sum($close > 7, 0)"], [1, 2, 3, 0, 0, 0, 0])
        # A sliced query must still warm all preceding history with its signal-date anchor.
        sliced = self.provider._signal_view().features(
            [A], ["Mean($close + 1, 3)", "Sum($close > 7, 0)"], end, end, allow_future=False,
        )
        np.testing.assert_allclose(sliced.iloc[0], [3.5, 0])
        pd.testing.assert_frame_equal(
            self.provider.features([A], ["$close"], self.dates[0], end, allow_future=False), ordinary,
        )
        self.assertEqual(self.provider.adjust, "qfq")
        self.assertIsNone(self.provider._adjustment_end)

    def test_annualization_requires_finite_positive_value(self):
        for value in (0, -1, np.nan, np.inf, -np.inf):
            with self.subTest(periods_per_year=value):
                with self.assertRaises(ValueError):
                    risk_analysis([0.01, 0.02], value)
                with self.assertRaises(ValueError):
                    BacktestEngine(self.provider, periods_per_year=value)
        self.assertAlmostEqual(risk_analysis([-0.1, 0.2], 2).annualized_return, 0.08)
        self.assertAlmostEqual(risk_analysis([-0.1, 0.2], 2).annualized_volatility, 0.3)

    def test_nonprice_qfq_signals_and_filters_do_not_require_factors(self):
        for code in (A, B):
            (self.root / "features" / code / "factor.day.bin").unlink()
        self.provider.clear_cache()
        pool = self.provider.instruments("all", [ExpressionFilter("$volume > 0")])
        weights = TopkStrategy(score="$volume", topk=1, instruments=pool).target_weights(
            self.provider, self.dates[0], self.dates[-1],
        )
        self.assertTrue(weights[A].eq(0.95).all())
        self.assertTrue(weights[B].eq(0).all())

    def test_signal_anchors_use_last_valid_factor_through_missing_values(self):
        self.write(A, "factor", [np.nan, 1, 2, np.nan, 4, 0, 2])
        self.provider.clear_cache()
        fields = ["$close", "Mean($close + 1, 3)", "Sum($close > 7, 0)", "IsNull($close)"]
        actual = self.provider._signal_view().features(
            [A], fields, self.dates[0], self.dates[-1], allow_future=False,
        )
        for date in self.dates:
            with self.subTest(date=date):
                # An ordinary single-date qfq query supplies an independent
                # reference for the latest valid anchor observable that day.
                expected = self.provider.features([A], fields, date, date, allow_future=False)
                pd.testing.assert_series_equal(actual.loc[(A, date)], expected.iloc[0])


if __name__ == "__main__":
    unittest.main()
