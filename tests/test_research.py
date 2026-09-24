"""Offline regressions for daily data, expressions, filters and backtesting."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

import qlib
from qlib.data import D, LocalProvider
from qlib.data.filter import (ExpressionFilter, IndustryFilter, ListingDaysFilter,
                              MembershipFilter, STFilter, TradableFilter)
from qlib.backtest import (BacktestEngine, ExchangeConfig, TopkStrategy,
                           WeightStrategy, risk_analysis)

A, B = "000001.SZ", "600000.SH"


class ResearchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dates = pd.bdate_range("2024-01-02", periods=8, name="datetime")
        for folder in ("calendars", "instruments", "industry", f"features/{A}",
                       f"features/{B}", "cache/daily"):
            (self.root / folder).mkdir(parents=True)
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")), encoding="utf-8")
        future = self.dates.append(pd.DatetimeIndex(["2024-01-12"]))
        (self.root / "calendars/day_future.txt").write_text("\n".join(future.strftime("%Y-%m-%d")), encoding="utf-8")
        (self.root / "instruments/all.txt").write_text(
            f"{A}\t2024-01-02\t2024-01-11\n{B}\t2024-01-04\t2024-01-11\n", encoding="utf-8")
        (self.root / "instruments/st.txt").write_text(
            f"{A}\t2024-01-03\t2024-01-04\n{A}\t2024-01-09\t2024-01-09\n", encoding="utf-8")
        (self.root / "instruments/csi300.txt").write_text(
            f"{A}\t2024-01-02\t2024-01-04\n{B}\t2024-01-08\t2024-01-11\n", encoding="utf-8")
        (self.root / "industry/801780.SI.txt").write_text(f"{A}\t2024-01-04\t2024-01-09\n", encoding="utf-8")
        pd.DataFrame({"ts_code": [A, B], "symbol": ["000001", "600000"],
                      "name": ["Current name", "Other"], "list_date": ["2024-01-01", "2024-01-04"],
                      "delist_date": [None, None], "list_status": ["L", "L"]}).to_csv(self.root / "stock_basic.csv", index=False)
        a = [10, 11, 12, np.nan, 6, 7, 7, 8]
        b = [20, 21, 22, 23, 24, 25]
        for field in ("open", "close", "high", "low", "vwap"):
            self.write_daily(A, field, a)
            self.write_daily(B, field, b, offset=2)
        self.write_daily(A, "volume", [1000, 1000, 1000, np.nan, 1000, 1000, 1000, 1000])
        self.write_daily(B, "volume", [1000] * 6, offset=2)
        self.write_daily(A, "factor", [1, 1, 1, np.nan, 2, 2, 2, 2])
        self.write_daily(B, "factor", [1] * 6, offset=2)
        self.write_daily(A, "extra_field", list(range(8)))
        pd.DataFrame({"ts_code": [A], "trade_date": ["20240102"], "pe": [8.5]}).to_csv(
            self.root / "cache/daily/20240102.csv", index=False)
        self.provider = LocalProvider(self.root, cache_uri=self.root / "cache")

    def write_daily(self, code, field, values, offset=0):
        np.asarray([offset, *values], dtype="<f4").tofile(self.root / "features" / code / f"{field}.day.bin")

    def weights(self, rows):
        return WeightStrategy(pd.DataFrame(rows).T.rename_axis("datetime"))

    def engine(self, **kwargs):
        defaults = dict(lot_size=100, buy_cost=0, sell_cost=0, min_cost=0)
        defaults.update(kwargs)
        return BacktestEngine(self.provider, initial_cash=10000, exchange=ExchangeConfig(**defaults))

    def test_default_facade_and_calendar(self):
        qlib.init(self.root)
        self.assertEqual(len(D.calendar()), 8)
        self.assertEqual(len(D.calendar(future=True)), 9)
        self.assertEqual(len(D.calendar("2024-01-06", "2024-01-09")), 2)
        with self.assertRaises(ValueError):
            D.calendar(freq="1min")
        with self.assertRaises(ValueError):
            D.calendar("2024-02-01", "2024-01-01")

    def test_daily_offsets_aliases_and_arbitrary_fields(self):
        data = self.provider.daily(["sz000001", "SH600000"], ["close", "extra_field"])
        self.assertEqual(data.loc[(B, "2024-01-04"), "close"], 20)
        self.assertEqual(data.loc[(A, "2024-01-11"), "extra_field"], 7)
        self.assertTrue(data.loc[B, "extra_field"].isna().all())
        self.assertTrue(np.isnan(data.loc[(A, "2024-01-05"), "close"]))
        with self.assertRaises(KeyError):
            self.provider.features([A], ["$typo"])
        with self.assertRaises(KeyError):
            self.provider.daily(["999999.SZ"])

    def test_corrupt_binary_and_strict_missing(self):
        (self.root / "features" / A / "close.day.bin").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            self.provider.daily([A], ["close"])
        provider = LocalProvider(self.root, missing="raise")
        with self.assertRaises(FileNotFoundError):
            provider.daily([B], ["extra_field"])

    def test_warmup_nested_and_expanding(self):
        expressions = ["Mean($close, 3)", "Mean(Ref($close, 1), 2)", "Sum($close, 0)", "Ref($close, 0)"]
        result = self.provider.features([A], expressions, "2024-01-04", "2024-01-04")
        self.assertEqual(result.iloc[0].tolist(), [11, 10.5, 33, 10])

    def test_operators(self):
        expressions = ["Slope($extra_field, 3)", "Rsquare($extra_field, 3)", "Resi($extra_field, 3)",
                       "Rank($extra_field, 3)", "IdxMax($extra_field, 3)", "Quantile($extra_field, 3, 0.5)",
                       "Corr($close, $close, 3)", "If($close > 10, $close, 0)", "1 / ($close - $close)"]
        result = self.provider.features([A], expressions, "2024-01-04", "2024-01-04").iloc[0]
        np.testing.assert_allclose(result.iloc[:8].to_numpy(dtype=float), [1, 1, 0, 1, 3, 1, 1, 12])
        self.assertTrue(np.isnan(result.iloc[-1]))

    def test_expanding_operators_with_historical_warmup(self):
        expressions = [f"{kind}($extra_field, 0)" for kind in ("Mean", "Slope", "Rsquare", "Resi")]
        result = self.provider.features([A], expressions, "2024-01-10", "2024-01-10")
        np.testing.assert_allclose(result.iloc[0].to_numpy(dtype=float), [3, 1, 1, 0], atol=1e-12)

    def test_expression_security_and_lookahead(self):
        for expression in ("__import__('os')", "$close.__class__", "[x for x in $close]", "open('x')"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                self.provider.features([A], [expression])
        label = self.provider.features([A], ["Ref($close, -1)"], "2024-01-02", "2024-01-02")
        self.assertEqual(label.iloc[0, 0], 11)
        for expression in ("Ref($close, -1)", "Delta($close, -1)"):
            with self.assertRaises(ValueError):
                self.provider.features([A], [expression], allow_future=False)
            with self.assertRaises(ValueError):
                TopkStrategy(score=expression)

    def test_historical_universe_and_composed_filters(self):
        pool = self.provider.instruments("all", [~STFilter(exclude=False) & ListingDaysFilter(min_days=3),
                                                  ExpressionFilter("$close > 0")])
        mask = self.provider.universe(pool)
        self.assertFalse(mask.loc["2024-01-03", A])
        self.assertFalse(mask.loc["2024-01-05", A])
        self.assertTrue(mask.loc["2024-01-08", A])
        self.assertTrue(mask.loc["2024-01-08", B])
        self.assertFalse(mask.loc["2024-01-09", A])
        self.assertFalse(self.provider.universe("csi300").loc["2024-01-05"].any())
        industry = self.provider.universe(self.provider.instruments("all", [IndustryFilter("801780.SI")]))
        self.assertTrue(industry.loc["2024-01-04", A])
        self.assertFalse(industry.loc["2024-01-10", A])

    def test_filter_configuration_and_discontinuous_intervals(self):
        pool = self.provider.instruments("all", [{"filter_type": "STFilter"}])
        spans = self.provider.list_instruments(pool)[A]
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0], (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")))
        selected = self.provider.instruments("all", [ExpressionFilter("$close > Mean($close, 3)")])
        self.assertIn(A, self.provider.list_instruments(selected, "2024-01-04", "2024-01-04", as_list=True))
        with self.assertRaises(ValueError):
            self.provider.universe(self.provider.instruments("all", [ExpressionFilter("Ref($close, -1) > 0")]))

    def test_listing_days_zero_based_and_incomplete_trading_calendar(self):
        mask = self.provider.universe(self.provider.instruments("all", [ListingDaysFilter(0, 0)]))
        self.assertTrue(mask.loc["2024-01-04", B])
        self.assertFalse(mask.loc["2024-01-05", B])
        with self.assertRaisesRegex(ValueError, "Calendar starts"):
            self.provider.universe(self.provider.instruments("all", [ListingDaysFilter(1, trading_days=True)]))

    def test_stock_basic_raw_tables_and_path_safety(self):
        self.assertEqual(self.provider.stock_basic([A]).iloc[0].symbol, "000001")
        raw = self.provider.read_table("daily/20240102.csv", ts_code=A)
        self.assertEqual(raw.iloc[0].trade_date, "20240102")
        self.assertEqual(raw.iloc[0].pe, 8.5)
        with self.assertRaises(ValueError):
            self.provider.read_table("../stock_basic.csv")
        with self.assertRaises(ValueError):
            self.provider.universe("../all")

    def test_next_day_execution_suspension_and_split(self):
        strategy = self.weights({"2024-01-02": {A: 1}, "2024-01-08": {A: 0}})
        result = self.engine().run(strategy, "2024-01-02", "2024-01-11")
        self.assertEqual(result.trades.datetime.tolist(), [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-09")])
        self.assertEqual(result.trades.quantity.tolist(), [900, 1800])
        self.assertEqual(result.report.loc["2024-01-05", "stale_positions"], 1)
        self.assertEqual(result.report.loc["2024-01-04", "equity"], 10900)
        self.assertEqual(result.report.loc["2024-01-08", "equity"], 10900)
        self.assertEqual(result.report.iloc[-1].equity, 12700)
        self.assertEqual(result.report.iloc[0].equity, 10000)
        self.assertTrue((result.report.cash >= 0).all())

    def test_execution_uses_raw_prices_for_all_read_adjustments(self):
        strategy = self.weights({"2024-01-02": {A: 1}, "2024-01-08": {A: 0}})
        expected = self.engine().run(strategy, "2024-01-02", "2024-01-11")
        for mode in ("hfq", "qfq", "none"):
            provider = LocalProvider(self.root, adjust=mode)
            engine = BacktestEngine(provider, initial_cash=10000,
                                    exchange=ExchangeConfig(lot_size=100, buy_cost=0, sell_cost=0, min_cost=0))
            actual = engine.run(strategy, "2024-01-02", "2024-01-11")
            pd.testing.assert_frame_equal(actual.report, expected.report)
            pd.testing.assert_frame_equal(actual.trades, expected.trades)

    def test_commission_minimum_sell_tax_and_cash_conservation(self):
        strategy = self.weights({"2024-01-02": {A: 1}, "2024-01-03": {A: 0}})
        result = self.engine(min_cost=5, sell_tax=0.001).run(strategy, "2024-01-02", "2024-01-04")
        self.assertEqual(result.trades.quantity.tolist(), [900, 900])
        self.assertAlmostEqual(result.report.iloc[-1].equity, 10879.2)
        self.assertAlmostEqual(result.trades.cost.sum(), 20.8)
        self.assertAlmostEqual((1 + result.report["return"]).prod(), result.report.iloc[-1].net_value)

    def test_limit_up_and_volume_partial_fills(self):
        strategy = self.weights({"2024-01-02": {A: 1}})
        blocked = self.engine(limit_threshold=0.05).run(strategy, "2024-01-02", "2024-01-04")
        self.assertTrue(blocked.trades.empty)
        self.assertEqual(blocked.orders.iloc[0].reason, "limit_up")
        partial = self.engine(volume_limit=0.001, lot_size=10).run(strategy, "2024-01-02", "2024-01-04")
        self.assertEqual(partial.trades.iloc[0].quantity, 100)
        self.assertEqual(partial.orders.iloc[0].status, "partial")

    def test_explicit_price_bands_override_scalar_threshold(self):
        self.write_daily(A, "up_limit", [100, 11, 100, 100, 100, 100, 100, 100])
        strategy = self.weights({"2024-01-02": {A: 1}})
        result = self.engine().run(strategy, "2024-01-02", "2024-01-04")
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.orders.iloc[0].reason, "limit_up")

    def test_suspended_sell_keeps_position(self):
        strategy = self.weights({"2024-01-02": {A: 1}, "2024-01-04": {A: 0}})
        result = self.engine().run(strategy, "2024-01-02", "2024-01-08")
        self.assertEqual(result.orders.iloc[-1].reason, "suspended_or_missing_quote")
        self.assertEqual(result.positions.loc[(pd.Timestamp("2024-01-08"), A), "quantity"], 1800)

    def test_missing_factor_rejects_buy_and_errors_on_held_quote(self):
        self.write_daily(A, "factor", [1, np.nan, 1, np.nan, 2, 2, 2, 2])
        strategy = self.weights({"2024-01-02": {A: 1}})
        result = self.engine().run(strategy, "2024-01-02", "2024-01-04")
        self.assertEqual(result.orders.iloc[0].reason, "missing_adjustment_factor")
        self.write_daily(A, "factor", [1, 1, np.nan, np.nan, 2, 2, 2, 2])
        self.provider.clear_cache()
        with self.assertRaisesRegex(ValueError, "factor for held"):
            self.engine().run(strategy, "2024-01-02", "2024-01-04")

    def test_delisted_holdings_do_not_silently_keep_stale_value(self):
        frame = pd.read_csv(self.root / "stock_basic.csv", dtype=str)
        frame.loc[0, "delist_date"] = "2024-01-04"
        frame.to_csv(self.root / "stock_basic.csv", index=False)
        with self.assertRaisesRegex(ValueError, "delisted"):
            self.engine().run(self.weights({"2024-01-02": {A: 1}}), "2024-01-02", "2024-01-08")

    def test_delisted_last_close_settlement_after_suspension(self):
        frame = pd.read_csv(self.root / "stock_basic.csv", dtype=str)
        frame.loc[0, "delist_date"] = "2024-01-07"  # First following session is January 8.
        frame.to_csv(self.root / "stock_basic.csv", index=False)
        strategy = self.weights({"2024-01-02": {A: 1}, "2024-01-04": {A: 0}})
        engine = self.engine(delist_policy="last_close", sell_cost=0.03, sell_tax=0.01, volume_limit=0.01)
        short = engine.run(strategy, "2024-01-02", "2024-01-05")
        result = engine.run(strategy, "2024-01-02", "2024-01-11")
        pd.testing.assert_frame_equal(short.report, result.report.loc[:"2024-01-05"])
        self.assertEqual(result.orders.iloc[1].reason, "suspended_or_missing_quote")
        self.assertEqual(result.report.loc["2024-01-05", "stale_positions"], 1)
        settlements = result.trades[result.trades.side == "settle"]
        self.assertEqual(len(settlements), 1)
        settlement = settlements.iloc[0]
        self.assertEqual(settlement.datetime, pd.Timestamp("2024-01-08"))
        self.assertTrue(pd.isna(settlement.signal_date))
        # Use the pre-suspension close, not the fixture's post-delisting quote/factor.
        self.assertEqual(settlement.quantity, 900)
        self.assertEqual(settlement.price, 12)
        self.assertEqual(settlement.notional, 10800)
        self.assertEqual(settlement.cost, 0)
        self.assertEqual(result.orders.iloc[-1].reason, "delisted_last_close")
        after = result.report.loc["2024-01-08":]
        self.assertTrue(after.cash.eq(10900).all())
        self.assertTrue(after.equity.eq(10900).all())
        self.assertTrue(after.market_value.eq(0).all())
        self.assertTrue(after.stale_positions.eq(0).all())
        self.assertTrue(after.cost.eq(0).all())
        self.assertTrue(after.turnover.eq(0).all())
        self.assertTrue((result.positions.index.get_level_values("datetime") < "2024-01-08").all())

    def test_delisted_settlement_cash_can_fund_other_positions(self):
        frame = pd.read_csv(self.root / "stock_basic.csv", dtype=str)
        frame.loc[0, "delist_date"] = "2024-01-04"
        frame.to_csv(self.root / "stock_basic.csv", index=False)
        strategy = self.weights({"2024-01-02": {A: 1}, "2024-01-03": {B: 1},
                                 "2024-01-04": {A: 1}})
        result = self.engine(delist_policy="last_close").run(strategy, "2024-01-02", "2024-01-08")
        day = result.trades[result.trades.datetime == pd.Timestamp("2024-01-04")]
        self.assertEqual(day.side.tolist(), ["settle", "buy"])
        self.assertEqual(day.instrument.tolist(), [A, B])
        self.assertEqual(day.quantity.tolist(), [900, 500])
        self.assertEqual(result.report.loc["2024-01-04", "equity"], 10000)
        self.assertEqual(len(result.trades[(result.trades.instrument == A) & (result.trades.side == "buy")]), 1)
        after = result.positions[result.positions.index.get_level_values("datetime") >= "2024-01-04"]
        self.assertNotIn(A, after.index.get_level_values("instrument"))

    def test_topk_uses_historical_membership_and_filters(self):
        pool = self.provider.instruments("csi300", [STFilter(), TradableFilter()])
        strategy = TopkStrategy(score="$close", topk=1, instruments=pool, risk_degree=0.9)
        weights = strategy.target_weights(self.provider, "2024-01-02", "2024-01-11")
        self.assertEqual(weights.loc["2024-01-02", A], 0.9)
        self.assertEqual(weights.loc["2024-01-03"].sum(), 0)
        self.assertEqual(weights.loc["2024-01-08", B], 0.9)
        result = self.engine().run(strategy, "2024-01-02", "2024-01-11")
        self.assertTrue((result.trades.signal_date < result.trades.datetime).all())

    def test_prefix_invariance_and_prior_session_signal(self):
        strategy = TopkStrategy(score="Mean($close, 2)", topk=1, instruments=[A])
        short = self.engine().run(strategy, "2024-01-03", "2024-01-08")
        long = self.engine().run(strategy, "2024-01-03", "2024-01-11")
        pd.testing.assert_frame_equal(short.report, long.report.loc[:"2024-01-08"])
        self.assertEqual(short.trades.iloc[0].signal_date, pd.Timestamp("2024-01-02"))

    def test_empty_universe_cash_benchmark_and_exports(self):
        strategy = TopkStrategy(score="$close", instruments=self.provider.instruments("all", [ExpressionFilter("$close < 0")]))
        benchmark = pd.Series(0.01, index=self.dates)
        result = self.engine().run(strategy, "2024-01-02", "2024-01-11", benchmark=benchmark)
        self.assertTrue(result.trades.empty)
        self.assertTrue(result.report.equity.eq(10000).all())
        self.assertAlmostEqual(result.metrics.benchmark_total_return, 1.01**8 - 1)
        result.save(self.root / "result")
        self.assertEqual(len(list((self.root / "result").glob("*.csv"))), 5)

    def test_invalid_weights_configuration_and_benchmark(self):
        for value in (-0.1, 1.1, np.inf):
            with self.assertRaises(ValueError):
                self.engine().run(self.weights({"2024-01-02": {A: value}}), "2024-01-02", "2024-01-04")
        with self.assertRaises(ValueError):
            ExchangeConfig(lot_size=0)
        with self.assertRaises(ValueError):
            ExchangeConfig(delist_policy="invalid")
        with self.assertRaises(ValueError):
            self.engine().run(self.weights({"2024-01-02": {A: 1}}), "2024-01-02", "2024-01-04", benchmark=pd.Series(dtype=float))

    def test_drawdown_includes_initial_capital(self):
        metrics = risk_analysis([-0.1, 0.0])
        self.assertAlmostEqual(metrics.max_drawdown, -0.1)


if __name__ == "__main__":
    unittest.main()
