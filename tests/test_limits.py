"""Price-limit ingestion, binary alignment and execution from real field files."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from qlib.backtest import BacktestEngine, ExchangeConfig, WeightStrategy
from qlib.data import LocalProvider
from scripts import config as C
from scripts.build_limits import build_limits
from scripts.dump.bin import DAILY_FIELDS, build_daily, merge_daily, prepare_limits
from scripts.tushare.data import CsvClient, TushareClient, cache_csv, download_limit_cache

A, B = "000001.SZ", "600000.SH"


class LimitDataTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "cn_data"
        self.cache = Path(temporary.name) / "cache"
        self.dates = pd.bdate_range("2024-01-02", periods=4, name="datetime")
        for path in ("calendars", "instruments", f"features/{A}", f"features/{B}"):
            (self.root / path).mkdir(parents=True)
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        (self.root / "instruments/all.txt").write_text(f"{A} 2024-01-02 2024-01-05\n{B} 2024-01-03 2024-01-05\n")
        self.basic = pd.DataFrame({"ts_code": [A, B], "list_date": ["2020-01-01", "2024-01-03"]})
        self.basic.to_csv(self.root / "stock_basic.csv", index=False)
        for code, offset, prices in ((A, 0, [10, 11, np.nan, 9]), (B, 1, [20, 21, 22])):
            for field, values in (("open", prices), ("close", prices), ("volume", [1000] * len(prices)),
                                  ("factor", [2] * len(prices))):
                np.asarray([offset, *values], dtype="<f4").tofile(self.root / "features" / code / f"{field}.day.bin")
        for pos, date in enumerate(self.dates):
            day = date.strftime("%Y%m%d")
            codes = [A] if pos == 0 else [B] if pos == 2 else [A, B]
            prices = {A: [10, 11, np.nan, 9][pos], B: 19 + pos}
            daily = pd.DataFrame([dict(ts_code=code, trade_date=day, open=prices[code], high=prices[code],
                                       low=prices[code], close=prices[code], vol=1000, amount=prices[code] * 100)
                                  for code in codes])
            limits = pd.DataFrame({"ts_code": [A, B], "trade_date": [day] * 2,
                                   "up_limit": [11 if pos == 1 else 12, 25], "down_limit": [9, 18]})
            if pos == 3:
                limits = limits[limits.ts_code != A]  # No fabricated or forward-filled limit.
            frames = {"daily": daily, "stk_limit": limits,
                      "adj_factor": daily[["ts_code", "trade_date"]].assign(adj_factor=2),
                      "daily_basic": daily[["ts_code", "trade_date"]].assign(total_mv=100, circ_mv=80),
                      "stock_st": pd.DataFrame(columns=["ts_code", "trade_date"])}
            for api, frame in frames.items():
                path = self.cache / api / f"{day}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)

    def test_backfill_offsets_suspension_missing_and_read_adjustments(self):
        original = {p: p.read_bytes() for p in self.root.rglob("*.bin")}
        stats = build_limits(self.root, self.cache)
        self.assertEqual(stats["stocks"], 2)
        self.assertEqual(stats["daily_rows"], 6)
        self.assertEqual(stats["rows_missing_limits"], 1)
        for path, data in original.items():
            self.assertEqual(path.read_bytes(), data)
        np.testing.assert_allclose(np.fromfile(self.root / "features" / A / "up_limit.day.bin", dtype="<f4"),
                                   [0, 12, 11, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(np.fromfile(self.root / "features" / B / "down_limit.day.bin", dtype="<f4"),
                                   [1, 18, 18, 18])
        provider = LocalProvider(self.root)
        raw = provider.daily([A], ["up_limit", "down_limit"], adjust="none")
        adjusted = provider.daily([A], ["up_limit", "down_limit"])
        pd.testing.assert_frame_equal(adjusted, raw * 2)
        before = (self.root / "features" / A / "up_limit.day.bin").read_bytes()
        build_limits(self.root, self.cache)
        self.assertEqual((self.root / "features" / A / "up_limit.day.bin").read_bytes(), before)

    def test_full_builder_matches_backfilled_fields(self):
        build_limits(self.root, self.cache)
        target = self.root.parent / "full"
        target.mkdir()
        # Creating lowercase filesystem aliases is unrelated to binary correctness.
        def directory(root, kind, code):
            path = root / kind / code
            path.mkdir(parents=True, exist_ok=True)
            return path
        with patch("scripts.dump.bin.stock_dir", side_effect=directory):
            dates = build_daily(CsvClient(self.cache), target, self.basic, self.dates, self.dates[-1])
        self.assertEqual(list(dates), list(self.dates))
        for code in (A, B):
            for field in ("up_limit", "down_limit"):
                self.assertEqual((target / "features" / code / f"{field}.day.bin").read_bytes(),
                                 (self.root / "features" / code / f"{field}.day.bin").read_bytes())
        self.assertIn("up_limit", DAILY_FIELDS)
        self.assertIn("down_limit", DAILY_FIELDS)

    def test_invalid_source_aborts_before_publication(self):
        build_limits(self.root, self.cache)
        path = self.root / "features" / A / "up_limit.day.bin"
        before = path.read_bytes()
        source = self.cache / "stk_limit" / "20240105.csv"
        frame = pd.read_csv(source, dtype={"trade_date": str})
        frame["trade_date"] = "20240104"
        frame.to_csv(source, index=False)
        with self.assertRaises(ValueError):
            build_limits(self.root, self.cache)
        self.assertEqual(path.read_bytes(), before)

    def test_source_zero_null_infinity_and_inverted_bounds(self):
        source = pd.DataFrame({"ts_code": [A, B], "trade_date": ["20240102"] * 2,
                               "up_limit": [0, np.inf], "down_limit": [-1, np.nan]})
        self.assertTrue(prepare_limits(source)[["up_limit", "down_limit"]].isna().all().all())
        source.loc[0, ["up_limit", "down_limit"]] = [9, 10]
        with self.assertRaises(ValueError):
            prepare_limits(source)
        with self.assertRaises(ValueError):
            prepare_limits(pd.concat([source, source], ignore_index=True))
        reader = CsvClient(self.cache)
        merged = merge_daily(reader.fetch("daily", trade_date="20240102"), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.assertTrue(merged[["up_limit", "down_limit"]].isna().all().all())

    def test_download_requests_fields_reuses_cache_and_refreshes_today(self):
        client = Mock()
        client.fetch.return_value = pd.DataFrame({"ts_code": [A], "trade_date": ["20240102"],
                                                  "up_limit": [11], "down_limit": [9]})
        destination = self.root.parent / "download"
        dates = self.dates[:1]
        download_limit_cache(client, destination, dates, pd.Timestamp("2024-01-03"))
        self.assertEqual(client.fetch.call_args.args, ("stk_limit",))
        self.assertEqual(set(client.fetch.call_args.kwargs["fields"].split(",")),
                         {"ts_code", "trade_date", "up_limit", "down_limit"})
        download_limit_cache(client, destination, dates, pd.Timestamp("2024-01-03"))
        self.assertEqual(client.fetch.call_count, 1)
        download_limit_cache(client, destination, dates, dates[0])
        self.assertEqual(client.fetch.call_count, 2)
        self.assertNotIn("refresh", client.fetch.call_args.kwargs)

    def test_pagination_does_not_truncate_at_api_page_limit(self):
        first = pd.DataFrame({"ts_code": [f"{i:06d}.SZ" for i in range(5800)]})
        last = pd.DataFrame({"ts_code": ["999999.SZ"]})
        pro = Mock()
        pro.query.side_effect = [first, last]
        with patch.object(C, "REQUEST_INTERVAL", 0):
            frame = TushareClient(pro).fetch("stk_limit", trade_date="20240102")
        self.assertEqual(len(frame), 5801)
        self.assertEqual([call.kwargs["offset"] for call in pro.query.call_args_list], [0, 5800])
        self.assertTrue(all(call.kwargs["limit"] == 5800 for call in pro.query.call_args_list))

    def test_raw_limit_fields_automatically_block_backtest_buy(self):
        build_limits(self.root, self.cache)
        provider = LocalProvider(self.root)  # Default read adjustment is hfq.
        strategy = WeightStrategy(pd.DataFrame({A: [1.0]}, index=self.dates[:1]))
        result = BacktestEngine(provider, initial_cash=10000,
                                exchange=ExchangeConfig(lot_size=1, buy_cost=0, sell_cost=0, min_cost=0)).run(
                                    strategy, self.dates[0], self.dates[1])
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.orders.iloc[0].reason, "limit_up")

    def test_interrupted_pagination_keeps_csv_and_restarts_in_memory(self):
        path = self.cache / "stk_limit" / "20240102.csv"
        original = path.read_bytes()
        first = pd.DataFrame({"ts_code": [A], "trade_date": ["20240102"],
                              "up_limit": [12], "down_limit": [8]})
        second = first.assign(ts_code=B, up_limit=22, down_limit=18)
        client = TushareClient(Mock())
        with patch("scripts.tushare.data.API_PAGE_SIZES", {"stk_limit": 1}), \
                patch.object(client, "_request_page", side_effect=[first, RuntimeError("interrupted")]):
            with self.assertRaises(RuntimeError):
                cache_csv(path, lambda: client.fetch("stk_limit", trade_date="20240102"), refresh=True)
        self.assertEqual(path.read_bytes(), original)
        with patch("scripts.tushare.data.API_PAGE_SIZES", {"stk_limit": 1}), \
                patch.object(client, "_request_page", side_effect=[first, second, pd.DataFrame()]) as request:
            cache_csv(path, lambda: client.fetch("stk_limit", trade_date="20240102"), refresh=True)
        self.assertEqual([call.args[1] for call in request.call_args_list], [0, 1, 2])
        saved = CsvClient(self.cache).fetch("stk_limit", trade_date="20240102")
        self.assertEqual(saved.ts_code.tolist(), [A, B])
        self.assertEqual(saved.up_limit.tolist(), [12, 22])


if __name__ == "__main__":
    unittest.main()
