"""Equivalence and bounded-work regressions for daily execution optimizations."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.backtest import BacktestEngine, ExchangeConfig
from qlib.backtest.executor import _ExecutionQuotes, _QuoteRows
from qlib.contrib.strategy import TopkDropoutStrategy, TopkStrategy, WeightStrategy


class ArrayProvider:
    """A custom provider with no LocalProvider internals or filesystem cache."""

    def __init__(self, stocks=10):
        self.dates = pd.bdate_range("2024-01-02", periods=30, name="datetime")
        self.codes = [f"{i + 1:06d}.SZ" for i in range(stocks)]
        self.reads = Counter()
        self.values = {}
        for i, code in enumerate(self.codes):
            prices = 10 + i + np.arange(30) / 10
            for field in ("open", "close", "vwap"):
                self.values[code, field] = pd.Series(prices.copy(), index=self.dates, name=field)
            for field, value in (("volume", 1000), ("factor", 1), ("up_limit", 1000), ("down_limit", 0)):
                self.values[code, field] = pd.Series(float(value), index=self.dates, name=field)
        self.basic = pd.DataFrame({"ts_code": self.codes, "list_date": self.dates[0], "delist_date": pd.NaT})
        self.scores = pd.DataFrame(np.tile(np.arange(stocks, 0, -1), (30, 1)),
                                   index=self.dates, columns=self.codes, dtype=float)

    def calendar(self, start_time=None, end_time=None):
        return self.dates[(self.dates >= (pd.Timestamp(start_time) if start_time is not None else self.dates[0])) &
                          (self.dates <= (pd.Timestamp(end_time) if end_time is not None else self.dates[-1]))].copy()

    def _daily(self, code, field):
        self.reads[code, field] += 1
        return self.values[code, field]

    def fields(self, kind):
        return sorted({field for code, field in self.values})

    def stock_basic(self):
        return self.basic.copy()

    def list_instruments(self, instruments="all", as_list=False):
        return self.codes.copy()

    def universe(self, instruments, start_time=None, end_time=None):
        dates = self.calendar(start_time, end_time)
        result = pd.DataFrame(True, index=dates, columns=self.codes)
        for row in self.basic.itertuples():
            result[row.ts_code] = (dates >= row.list_date) & (pd.isna(row.delist_date) | (dates < row.delist_date))
        return result

    def features(self, instruments, expressions, start_time=None, end_time=None, **kwargs):
        values = self.scores.loc[start_time:end_time]
        index = pd.MultiIndex.from_product([values.columns, values.index], names=["instrument", "datetime"])
        series = pd.Series(values.to_numpy().T.ravel(), index=index)
        return pd.DataFrame({expression: series for expression in expressions})


class FullHistoryQuotes:
    """Independent oracle using the pre-optimization full-history operations."""

    def __init__(self, provider, codes, fields, calendar, dates):
        self.frames = {code: pd.DataFrame({field: provider._daily(code, field).reindex(calendar)
                                          for field in fields}) for code in codes}
        self.calendar_positions = calendar.get_indexer(dates)
        self.dates = dates

    def bars(self, position):
        return {code: frame.loc[self.dates[position]] for code, frame in self.frames.items()}

    def previous_close(self, code, position, factor, adjust_positions):
        previous = self.frames[code].iloc[:self.calendar_positions[position]]
        valid = previous.close.dropna()
        price = valid.iloc[-1] if len(valid) else np.nan
        if adjust_positions and len(valid) and np.isfinite(factor) and factor > 0:
            price *= previous.loc[valid.index[-1], "factor"] / factor
        return price


class BacktestPerformanceTest(unittest.TestCase):
    def test_references_match_full_history_with_suspensions_and_nonfinite_values(self):
        provider = ArrayProvider(3)
        a, b, c = provider.codes
        provider.values[a, "close"].iloc[18:22] = np.nan
        provider.values[a, "factor"].iloc[19:22] = 8
        provider.values[b, "close"].iloc[17:22] = [np.inf, np.nan, np.nan, -np.inf, np.nan]
        provider.values[b, "factor"].iloc[17:22] = [0, 5, 6, np.nan, 2]
        provider.values[c, "close"].iloc[:22] = np.nan
        dates = provider.dates[18:25]
        fields = provider.fields("daily")
        actual = _ExecutionQuotes(provider, provider.codes, fields, provider.dates, dates)
        expected = FullHistoryQuotes(provider, provider.codes, fields, provider.dates, dates)
        with np.errstate(invalid="ignore"):
            for position, date in enumerate(dates):
                for code in provider.codes:
                    for factor in (np.nan, 0, np.inf, 1, 2):
                        for adjust in (False, True):
                            np.testing.assert_equal(actual.previous_close(code, position, factor, adjust),
                                                    expected.previous_close(code, position, factor, adjust))
        # The suspension's newer factor must not replace the last quote's factor.
        self.assertEqual(actual.previous_close(a, 3, 8, True), provider.values[a, "close"].iloc[17] / 8)

    def test_window_quotes_keep_real_lazy_series_and_read_each_field_once(self):
        provider = ArrayProvider(3)
        dates = provider.dates[-4:]
        fields = provider.fields("daily")
        actual = _ExecutionQuotes(provider, provider.codes, fields, provider.dates, dates)
        self.assertTrue(all(frame.index.equals(dates) for frame in actual.frames.values()))
        self.assertEqual(sum(provider.reads.values()), len(fields) * len(provider.codes))
        self.assertTrue(all(count == 1 for count in provider.reads.values()))
        bars = actual.bars(0)
        self.assertEqual(len(bars), 0)
        bar = bars[provider.codes[0]]
        self.assertIsInstance(bar, pd.Series)
        self.assertEqual(bar.name, dates[0])
        self.assertIs(bars[provider.codes[0]], bar)
        self.assertEqual(len(bars), 1)
        expected = pd.Series({field: provider.values[provider.codes[0], field].loc[dates[0]] for field in fields},
                             name=dates[0])
        pd.testing.assert_series_equal(bar, expected)

    def test_custom_provider_scalar_types_preserve_reference_rounding(self):
        for dtype in ("float32", "int64"):
            provider = ArrayProvider(1)
            code = provider.codes[0]
            for key, values in provider.values.items():
                provider.values[key] = values.astype(dtype)
            provider.values[code, "factor"].iloc[:20] = np.asarray(1.1 if dtype == "float32" else 2, dtype=dtype)
            provider.values[code, "factor"].iloc[20:] = np.asarray(1.7 if dtype == "float32" else 3, dtype=dtype)
            dates = provider.dates[20:]
            actual = _ExecutionQuotes(provider, provider.codes, provider.fields("daily"), provider.dates, dates)
            expected = FullHistoryQuotes(provider, provider.codes, provider.fields("daily"), provider.dates, dates)
            for position, date in enumerate(dates):
                factor = expected.bars(position)[code].factor
                for adjust in (False, True):
                    np.testing.assert_equal(actual.previous_close(code, position, factor, adjust),
                                            expected.previous_close(code, position, factor, adjust))
                pd.testing.assert_series_equal(actual.bars(position)[code], expected.bars(position)[code])

            for field in ("open", "vwap"):
                provider.values[code, field] = pd.Series(2**53 + 1, index=provider.dates, dtype=dtype)
            provider.values[code, "up_limit"] = pd.Series(2**54, index=provider.dates, dtype=dtype)
            strategy = WeightStrategy(pd.DataFrame({code: [0.5, 0]}, index=provider.dates[[19, 25]]))
            for adjust in (False, True):
                exchange = ExchangeConfig(lot_size=1, min_cost=0, buy_cost=0, sell_cost=0, sell_tax=0,
                                          adjust_positions=adjust)
                engine = BacktestEngine(provider, initial_cash=2**60, exchange=exchange)
                result = engine.run(strategy, dates[0], dates[-1])
                with patch("qlib.backtest.executor._ExecutionQuotes", FullHistoryQuotes):
                    reference = engine.run(strategy, dates[0], dates[-1])
                for name in ("report", "positions", "orders", "trades"):
                    pd.testing.assert_frame_equal(getattr(result, name), getattr(reference, name), check_exact=True)
                pd.testing.assert_series_equal(result.metrics, reference.metrics, check_exact=True)

    def test_nullable_quote_reference_scalars_preserve_large_integer_bits(self):
        for dtype in ("Int64", "UInt64"):
            provider = ArrayProvider(1)
            code = provider.codes[0]
            for key, values in provider.values.items():
                provider.values[key] = values.round().astype(dtype)
            provider.values[code, "close"] = pd.Series(2**53 + 1, index=provider.dates, dtype=dtype)
            provider.values[code, "close"].iloc[12] = pd.NA
            provider.values[code, "factor"] = pd.Series(2**53 + 1, index=provider.dates, dtype=dtype)
            provider.values[code, "factor"].iloc[9] = pd.NA
            provider.values[code, "factor"].iloc[20:] = 2**53 + 3
            dates = provider.dates[20:]
            actual = _ExecutionQuotes(provider, provider.codes, provider.fields("daily"), provider.dates, dates)
            expected = FullHistoryQuotes(provider, provider.codes, provider.fields("daily"), provider.dates, dates)
            self.assertEqual(int(actual.previous_close(code, 0, 1, False)), 2**53 + 1)
            self.assertEqual(int(actual.previous[code][1][0]), 2**53 + 1)
            for position, date in enumerate(dates):
                factor = expected.bars(position)[code].factor
                for adjust in (False, True):
                    np.testing.assert_equal(actual.previous_close(code, position, factor, adjust),
                                            expected.previous_close(code, position, factor, adjust))
                pd.testing.assert_series_equal(actual.bars(position)[code], expected.bars(position)[code])

            for field in ("open", "vwap"):
                provider.values[code, field] = pd.Series(2**53 + 1, index=provider.dates, dtype=dtype)
            provider.values[code, "up_limit"] = pd.Series(2**54, index=provider.dates, dtype=dtype)
            strategy = WeightStrategy(pd.DataFrame({code: [0.5, 0]}, index=provider.dates[[19, 25]]))
            for adjust in (False, True):
                exchange = ExchangeConfig(lot_size=1, min_cost=0, buy_cost=0, sell_cost=0, sell_tax=0,
                                          adjust_positions=adjust)
                engine = BacktestEngine(provider, initial_cash=2**60, exchange=exchange)
                result = engine.run(strategy, dates[0], dates[-1])
                with patch("qlib.backtest.executor._ExecutionQuotes", FullHistoryQuotes):
                    reference = engine.run(strategy, dates[0], dates[-1])
                for name in ("report", "positions", "orders", "trades"):
                    pd.testing.assert_frame_equal(getattr(result, name), getattr(reference, name), check_exact=True)
                pd.testing.assert_series_equal(result.metrics, reference.metrics, check_exact=True)

    def test_five_outputs_equal_full_history_execution_oracle(self):
        provider = ArrayProvider(4)
        a, b, c, d = provider.codes
        provider.values[a, "close"].iloc[21] = np.nan
        provider.values[a, "open"].iloc[21] = np.nan
        provider.values[a, "factor"].iloc[22:] = 2
        for field in ("open", "close", "vwap"):
            provider.values[a, field].iloc[22:] /= 2
        provider.values[b, "volume"].iloc[23] = 0
        provider.values[c, "volume"].iloc[24] = 0.1
        provider.values[c, "up_limit"].iloc[25] = provider.values[c, "open"].iloc[25]
        provider.basic.loc[3, "delist_date"] = provider.dates[27]
        provider.scores.iloc[20:] = np.array([[4, 3, 2, 1], [1, 2, 4, 3], [2, 4, 3, 1],
                                             [4, 3, 1, 2], [1, 2, 3, 4]] * 2)
        for deal_price in ("open", "close", "vwap"):
            for only_tradable in (False, True):
                for method in ("top", "random"):
                    exchange = ExchangeConfig(deal_price=deal_price, delist_policy="last_close", lot_size=1,
                                              volume_limit=0.3, limit_threshold=0.1, slippage=0.003,
                                              buy_cost=0.01, sell_cost=0.02, sell_tax=0.003, min_cost=3)
                    strategy = TopkDropoutStrategy(topk=2, n_drop=1, signal=provider.scores, random_seed=7,
                                                   only_tradable=only_tradable, method_buy=method,
                                                   method_sell="random" if method == "random" else "bottom")
                    with self.subTest(price=deal_price, tradable=only_tradable, method=method):
                        engine = BacktestEngine(provider, initial_cash=10000, exchange=exchange)
                        actual = engine.run(strategy, provider.dates[20], provider.dates[-1])
                        with patch("qlib.backtest.executor._ExecutionQuotes", FullHistoryQuotes):
                            expected = engine.run(strategy, provider.dates[20], provider.dates[-1])
                        for name in ("report", "positions", "orders", "trades"):
                            pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
                        pd.testing.assert_series_equal(actual.metrics, expected.metrics, check_exact=True)

    def test_zero_weight_liquidation_and_sparse_rebalances_equal_oracle(self):
        provider = ArrayProvider(3)
        dates = provider.dates
        weights = pd.DataFrame([[0.8, 0, 0], [0.2, 0.6, 0], [0, 0, 0]],
                               index=dates[[19, 22, 25]], columns=provider.codes)
        strategy = WeightStrategy(weights)
        engine = BacktestEngine(provider, initial_cash=10000, exchange=ExchangeConfig(lot_size=1))
        actual = engine.run(strategy, dates[20], dates[-1])
        with patch("qlib.backtest.executor._ExecutionQuotes", FullHistoryQuotes):
            expected = engine.run(strategy, dates[20], dates[-1])
        for name in ("report", "positions", "orders", "trades"):
            pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
        pd.testing.assert_series_equal(actual.metrics, expected.metrics, check_exact=True)

    def test_inactive_signal_columns_do_not_materialize_daily_bars(self):
        provider = ArrayProvider(10)
        reads, original = Counter(), _QuoteRows.__missing__

        def observe(rows, code):
            reads[rows.position, code] += 1
            return original(rows, code)

        with patch.object(_QuoteRows, "__missing__", observe):
            result = BacktestEngine(provider).run(TopkDropoutStrategy(topk=1, n_drop=0, signal=provider.scores),
                                                 provider.dates[-5], provider.dates[-1])
        self.assertEqual(set(result.positions.index.get_level_values("instrument")), {provider.codes[0]})
        self.assertEqual(len(reads), 5)
        self.assertTrue(all(count == 1 for count in reads.values()))

    def test_topk_array_ranking_equals_pandas_with_ties_and_missing_scores(self):
        provider = ArrayProvider(4)
        provider.scores.iloc[:4] = [[np.nan, np.inf, 3, 3], [-np.inf, 1, 1, np.nan],
                                    [0, -0.0, 0, 0], [np.nan] * 4]
        provider.scores = provider.scores[provider.codes[::-1]]
        for ascending in (False, True):
            for topk in (1, 3, 99):
                for rebalance in (1, 3):
                    strategy = TopkStrategy(score="$score", topk=topk, rebalance=rebalance, ascending=ascending)
                    actual = strategy.target_weights(provider, provider.dates[0], provider.dates[5])
                    expected = pd.DataFrame(0.0, index=provider.dates[:6:rebalance], columns=provider.codes)
                    for date in expected.index:
                        values = provider.scores.loc[date].reindex(provider.codes).replace([np.inf, -np.inf], np.nan).dropna()
                        selected = values.sort_values(ascending=ascending, kind="stable").head(topk).index
                        if len(selected):
                            expected.loc[date, selected] = strategy.risk_degree / len(selected)
                    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_tradable_candidate_scan_stops_after_enough_valid_candidates(self):
        provider = ArrayProvider(50)
        calls = []

        def tradable(code):
            calls.append(code)
            return code != provider.codes[0]

        strategy = TopkDropoutStrategy(topk=2, n_drop=0, only_tradable=True)
        sell, buy = strategy.select_stocks(provider.scores.iloc[0], {}, is_tradable=tradable,
                                          rng=np.random.RandomState(7))
        self.assertEqual(sell, [])
        self.assertEqual(buy, provider.codes[1:3])
        self.assertEqual(calls, provider.codes[:3])

    def test_topk_integer_extremes_preserve_exact_rank_and_stable_ties(self):
        provider = ArrayProvider(4)
        provider.scores = provider.scores.astype("int64")
        provider.scores.iloc[:] = [np.iinfo(np.int64).min, 2**53, 2**53 + 1, np.iinfo(np.int64).max]
        for ascending, selected in ((False, provider.codes[2:]), (True, provider.codes[:2])):
            result = TopkStrategy(score="$score", topk=2, ascending=ascending).target_weights(
                provider, provider.dates[0], provider.dates[2])
            self.assertTrue(result[selected].eq(0.475).all().all())
            self.assertEqual(result.gt(0).sum(axis=1).tolist(), [2, 2, 2])

    def test_topk_nullable_custom_scores_keep_pandas_fallback(self):
        provider = ArrayProvider(4)
        provider.scores = provider.scores.astype("Int64")
        provider.scores.iloc[:] = [pd.NA, 2**53 + 1, 2**53, 1]
        result = TopkStrategy(score="$score", topk=1).target_weights(provider, provider.dates[0], provider.dates[2])
        self.assertTrue(result[provider.codes[1]].eq(0.95).all())
        self.assertEqual(result.gt(0).sum(axis=1).tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
