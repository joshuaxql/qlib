"""Stateful dropout decisions, actual-fill feedback and signal timing."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from qlib.backtest import BacktestEngine, ExchangeConfig
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import LocalProvider

A, B, C, E = "000001.SZ", "000002.SZ", "600000.SH", "600519.SH"
CODES = [A, B, C, E]


class TopkDropoutTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dates = pd.bdate_range("2024-01-02", periods=7, name="datetime")
        for directory in ("calendars", "instruments", *[f"features/{code}" for code in CODES]):
            (self.root / directory).mkdir(parents=True)
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        (self.root / "instruments/all.txt").write_text("\n".join(f"{c} 2024-01-02 2024-01-31" for c in CODES))
        pd.DataFrame({"ts_code": CODES, "list_date": ["2020-01-01"] * 4}).to_csv(self.root / "stock_basic.csv", index=False)
        for code in CODES:
            for field, value in (("open", 10), ("close", 10), ("volume", 10000), ("factor", 1)):
                self.write(code, field, [value] * 7)
        self.provider = LocalProvider(self.root)
        self.scores = pd.DataFrame([[4, 3, 2, 1], *[[4, 1, 3, 2]] * 6], index=self.dates, columns=CODES, dtype=float)

    def write(self, code, field, values):
        np.asarray([0, *values], dtype="<f4").tofile(self.root / "features" / code / f"{field}.day.bin")

    def run_strategy(self, strategy=None, *, end=6, start=0, **exchange):
        config = dict(lot_size=1, buy_cost=0, sell_cost=0, min_cost=0)
        config.update(exchange)
        self.provider.clear_cache()
        engine = BacktestEngine(self.provider, initial_cash=10000, exchange=ExchangeConfig(**config))
        return engine.run(strategy or TopkDropoutStrategy(topk=2, n_drop=1, signal=self.scores),
                          self.dates[start], self.dates[end])

    def test_dropout_does_not_resize_retained_holdings(self):
        self.write(A, "open", [10, 10, 20, 20, 20, 20, 20])
        self.write(A, "close", [10, 10, 20, 20, 20, 20, 20])
        result = self.run_strategy()
        self.assertEqual(result.trades.instrument.tolist(), [A, B, B, C])
        self.assertEqual(result.trades.side.tolist(), ["buy", "buy", "sell", "buy"])
        self.assertEqual(result.trades.quantity.tolist(), [475, 475, 475, 498])
        self.assertEqual(result.report.iloc[2].cash, 270)
        self.assertTrue(result.positions.xs(A, level="instrument").quantity.eq(475).all())
        self.assertTrue((result.trades.signal_date < result.trades.datetime).all())

    def test_no_downgrade_and_zero_dropout(self):
        for n_drop in (0, 1):
            signal = self.scores.copy()
            signal.loc[:] = [4, 3, 2, 1]
            result = self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=n_drop, signal=signal))
            self.assertEqual(result.trades.side.tolist(), ["buy", "buy"])
        frozen = self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=0, signal=self.scores))
        self.assertEqual(frozen.trades.instrument.tolist(), [A, B])

    def test_hold_threshold_and_corporate_action_do_not_reset_age(self):
        signal = self.scores.copy()
        signal.iloc[1:] = [1, 4, 3, 2]
        self.write(A, "open", [10, 10, 5, 5, 5, 5, 5])
        self.write(A, "close", [10, 10, 5, 5, 5, 5, 5])
        self.write(A, "factor", [1, 1, 2, 2, 2, 2, 2])
        result = self.run_strategy(TopkDropoutStrategy(topk=1, n_drop=1, hold_thresh=2, risk_degree=1, signal=signal))
        sales = result.trades[result.trades.side == "sell"]
        self.assertEqual(sales.datetime.tolist(), [self.dates[3]])
        self.assertEqual(sales.quantity.tolist(), [2000])
        self.assertTrue(result.report.equity.eq(10000).all())

    def test_failed_sale_uses_actual_cash_and_actual_next_day_holdings(self):
        self.write(B, "volume", [10000, 10000, 0, 10000, 10000, 10000, 10000])
        result = self.run_strategy()
        day = result.trades[result.trades.datetime == self.dates[2]]
        self.assertEqual(day.instrument.tolist(), [C])
        self.assertEqual(day.quantity.tolist(), [47])  # Only the 500 uninvested cash is available.
        self.assertEqual(len(result.positions.loc[self.dates[2]]), 3)
        self.assertEqual(result.orders[result.orders.side == "sell"].iloc[0].reason, "suspended_or_missing_quote")
        later = result.trades[result.trades.datetime == self.dates[3]]
        self.assertEqual(later.instrument.tolist(), [B])
        self.assertEqual(later.side.tolist(), ["sell"])

    def test_partial_sales_and_fees_fund_purchases_without_negative_cash(self):
        self.write(B, "volume", [10000, 10000, 1, 10000, 10000, 10000, 10000])
        result = self.run_strategy(end=2, volume_limit=0.5, volume_unit=100, min_cost=5, sell_tax=0.01)
        sale = result.trades[result.trades.side == "sell"].iloc[0]
        self.assertEqual(sale.quantity, 50)
        self.assertEqual(sale.cost, 10)
        self.assertEqual(result.positions.loc[(self.dates[2], B), "quantity"], 425)
        self.assertEqual(result.trades.iloc[-1].quantity, 93)  # (490 + 500 - 10) * .95 / 10.
        self.assertTrue(result.report.cash.ge(0).all())

    def test_only_tradable_filters_ranking_candidates(self):
        self.write(A, "volume", [10000, 0, 10000, 10000, 10000, 10000, 10000])
        filtered = self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=1, signal=self.scores, only_tradable=True), end=1)
        unfiltered = self.run_strategy(end=1)
        self.assertEqual(filtered.trades.instrument.tolist(), [B, C])
        self.assertEqual(unfiltered.trades.instrument.tolist(), [B])

    def test_sell_at_limit_up_and_buy_at_limit_down(self):
        signal = self.scores.copy()
        signal.iloc[1:] = [1, 4, 3, 2]
        self.write(A, "up_limit", [100, 100, 10, 100, 100, 100, 100])
        for forbid, expected in ((True, ["buy"]), (False, ["buy", "sell", "buy"])):
            strategy = TopkDropoutStrategy(topk=1, n_drop=1, risk_degree=1, signal=signal,
                                           forbid_all_trade_at_limit=forbid)
            self.assertEqual(self.run_strategy(strategy, end=2).trades.side.tolist(), expected)
        self.write(A, "down_limit", [0, 10, 0, 0, 0, 0, 0])
        for forbid in (True, False):
            strategy = TopkDropoutStrategy(topk=1, n_drop=1, signal=signal, forbid_all_trade_at_limit=forbid)
            self.assertEqual(self.run_strategy(strategy, end=1).trades.empty, forbid)

    def test_random_selection_repeatability_and_prefix_invariance(self):
        for sell in ("random", "bottom"):
            for buy in ("random", "top"):
                strategy = TopkDropoutStrategy(topk=2, n_drop=1, signal=self.scores,
                                               method_sell=sell, method_buy=buy, random_seed=7)
                long = self.run_strategy(strategy)
                short = self.run_strategy(strategy, end=3)
                again = self.run_strategy(strategy)
                pd.testing.assert_frame_equal(long.trades, again.trades)
                pd.testing.assert_frame_equal(short.report, long.report.loc[:self.dates[3]])

    def test_multiindex_signal_first_column_and_prior_session(self):
        series = self.scores.rename_axis(columns="instrument").stack()
        frame = pd.DataFrame({"score": series, "ignored": -series})
        expected = self.run_strategy(start=1)
        for signal in (series, frame, series.reorder_levels(["instrument", "datetime"])):
            actual = self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=1, signal=signal), start=1)
            pd.testing.assert_frame_equal(actual.trades, expected.trades)
        self.assertEqual(expected.trades.iloc[0].signal_date, self.dates[0])

    def test_expression_scores_membership_and_missing_signal_dates(self):
        for code in CODES:
            self.write(code, "score", self.scores[code])
        strategy = TopkDropoutStrategy(topk=2, n_drop=1, score="$score")
        pd.testing.assert_frame_equal(self.run_strategy(strategy).trades, self.run_strategy().trades)
        sparse = self.scores.iloc[[0]].copy()
        result = self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=1, signal=sparse))
        self.assertEqual(len(result.trades), 2)
        missing = self.scores.copy()
        missing.iloc[1:] = np.nan
        result = self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=1, signal=missing))
        self.assertEqual(len(result.trades), 2)
        (self.root / "instruments/csi300.txt").write_text(f"{C} 2024-01-02 2024-01-31")
        filtered = self.run_strategy(TopkDropoutStrategy(topk=1, n_drop=1, signal=self.scores, instruments="csi300"))
        self.assertEqual(set(filtered.trades.instrument), {C})

    def test_validation_and_missing_scores_do_not_create_positions(self):
        for config in ({"topk": 0}, {"n_drop": -1}, {"hold_thresh": 1.5}, {"risk_degree": 1.1},
                       {"method_buy": "bottom"}, {"method_sell": "top"}, {"score": "Ref($close, -1)"},
                       {"only_tradable": "yes"}, {"random_seed": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                TopkDropoutStrategy(**{**dict(topk=2, n_drop=1), **config})
        scores = self.scores.copy()
        scores.loc[:] = np.nan
        self.assertTrue(self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=1, signal=scores)).trades.empty)
        scores = self.scores.copy()
        scores[A] = np.inf
        self.assertNotIn(A, self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=1, signal=scores)).trades.instrument.tolist())
        duplicate = self.scores.rename(columns={B: "SZ000001"})
        with self.assertRaises(ValueError):
            self.run_strategy(TopkDropoutStrategy(topk=2, n_drop=1, signal=duplicate))


if __name__ == "__main__":
    unittest.main()
