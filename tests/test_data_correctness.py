"""Regression checks for listing causality, version merges and complete pagination."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from qlib.backtest import BacktestEngine, ExchangeConfig
from qlib.contrib.strategy import TopkDropoutStrategy, TopkStrategy
from qlib.data import LocalProvider
from scripts import config as C
from scripts.dump.bin import build_daily, iso_date, write_lines
from scripts.dump.pit import prepare_financial
from scripts.tushare.data import CsvClient, TushareClient, download_financial_cache

A, B, NO_QUOTE = "000001.SZ", "000002.SZ", "000003.SZ"
DATES = pd.bdate_range("2024-01-02", periods=5, name="datetime")


class QuoteClient:
    def fetch(self, api, **params):
        day = params["trade_date"]
        position = DATES.get_loc(pd.Timestamp(day))
        codes = [B] if position in (2, 3) else [A, B]
        if api in ("stock_st", "stk_limit"):
            return pd.DataFrame()
        rows = []
        for code in codes:
            row = {"ts_code": code, "trade_date": day}
            if api == "daily":
                price = 10 if code == A else 8
                close = 9 if code == B and position == 3 else price
                row.update(open=price, high=max(price, close), low=price, close=close,
                           vol=10000, amount=price * 100000)
            elif api == "adj_factor":
                row["adj_factor"] = 1
            elif api == "daily_basic":
                row.update(total_mv=100, circ_mv=100)
            else:
                raise AssertionError(api)
            rows.append(row)
        return pd.DataFrame(rows)


def stock_directory(root, section, code):
    directory = root / section / code
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class DataCorrectnessTest(unittest.TestCase):
    def build(self, root, dates, basic):
        root.mkdir()
        basic.to_csv(root / "stock_basic.csv", index=False)
        with patch("scripts.dump.bin.stock_dir", side_effect=stock_directory):
            completed = build_daily(QuoteClient(), root, basic, dates, dates[-1])
        write_lines(root / "calendars/day.txt", map(iso_date, completed))
        return LocalProvider(root)

    def test_future_resumption_does_not_rewrite_historical_selection(self):
        basic = pd.DataFrame({"ts_code": [A, B], "list_date": ["2020-01-01"] * 2,
                              "delist_date": [pd.NaT] * 2})
        signals = pd.DataFrame({A: [2.] * 4, B: [1.] * 4}, index=DATES[:4])
        strategies = [
            TopkStrategy(score="Mean($close,3)", topk=1),
            TopkDropoutStrategy(score="Mean($close,3)", topk=1, n_drop=1),
            TopkDropoutStrategy(signal=signals, topk=1, n_drop=1),
        ]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = self.build(root / "short", DATES[:4], basic)
            long = self.build(root / "long", DATES, basic)
            pd.testing.assert_frame_equal(short.universe("all"), long.universe("all").loc[DATES[:4]])
            self.assertTrue(short.universe("all").loc[DATES[3], A])
            for strategy in strategies:
                with self.subTest(strategy=type(strategy).__name__, external=isinstance(strategy, TopkDropoutStrategy)
                                  and strategy.signal is not None):
                    results = []
                    for provider in (short, long):
                        engine = BacktestEngine(provider, initial_cash=10000, exchange=ExchangeConfig(
                            lot_size=1, buy_cost=0, sell_cost=0, min_cost=0, sell_tax=0))
                        results.append(engine.run(strategy, DATES[0], DATES[3]))
                    pd.testing.assert_frame_equal(results[0].report, results[1].report)
                    pd.testing.assert_frame_equal(results[0].trades, results[1].trades)
                    self.assertEqual(results[0].report.equity.iloc[-1], 10000)

    def test_listing_dates_control_membership_even_without_quotes(self):
        basic = pd.DataFrame({"ts_code": [A, B, NO_QUOTE],
                              "list_date": ["2020-01-01", "2020-01-01", "2024-01-03"],
                              "delist_date": [pd.NaT, pd.Timestamp("2024-01-05"), pd.NaT]})
        with TemporaryDirectory() as temporary:
            provider = self.build(Path(temporary) / "data", DATES[:4], basic)
            pool = provider.universe("all")
            self.assertTrue(pool[A].all())
            self.assertEqual(pool[B].tolist(), [True, True, True, False])
            self.assertEqual(pool[NO_QUOTE].tolist(), [False, True, True, True])
            self.assertTrue(provider.daily([A], "close", adjust="none").loc[A].close.iloc[-1:].isna().all())
            self.assertTrue(provider.daily([NO_QUOTE], "close", adjust="none").isna().all().all())

    def test_partial_refresh_preserves_values_and_explicit_nan_revisions(self):
        frame = pd.DataFrame({"ts_code": [A], "ann_date": ["20230420"], "end_date": ["20230331"],
                              "update_flag": ["1"], "eps": [1], "roe": [10]})
        client = Mock()
        with TemporaryDirectory() as temporary, patch.object(C, "START_DATE", "20230101"):
            root = Path(temporary)
            client.fetch.return_value = frame
            download_financial_cache(client, root, pd.Timestamp("2023-06-01"),
                                     {"fina_indicator": ["eps", "roe"]})
            client.fetch.return_value = frame.drop(columns="roe").assign(eps=1.1)
            download_financial_cache(client, root, pd.Timestamp("2023-06-01"), {"fina_indicator": ["eps"]})
            actual = CsvClient(root).fetch("fina_indicator_vip", period="20230331")
            self.assertEqual(actual.eps.tolist(), [1.1])
            self.assertEqual(actual.roe.tolist(), [10])
            # NaN in a requested field invalidates that exact version's old value.
            client.fetch.return_value = frame.drop(columns="roe").assign(eps=np.nan)
            download_financial_cache(client, root, pd.Timestamp("2023-06-01"), {"fina_indicator": ["eps"]})
            actual = CsvClient(root).fetch("fina_indicator_vip", period="20230331")
            self.assertTrue(actual.eps.isna().all())
            self.assertEqual(actual.roe.tolist(), [10])
            rows = prepare_financial(actual, "fina_indicator", ["eps", "roe"], {A}, "2023-06-01")
            self.assertEqual(rows.loc[rows.field.eq("roe"), "value"].tolist(), [10])
            self.assertTrue(rows.loc[rows.field.eq("eps"), "value"].isna().all())

    def test_new_versions_do_not_inherit_unrequested_old_values(self):
        original = pd.DataFrame({"ts_code": [A], "ann_date": ["20230420"], "end_date": ["20230331"],
                                 "update_flag": ["1"], "eps": [1.0], "roe": [10.0]})
        client = Mock()
        with TemporaryDirectory() as temporary, patch.object(C, "START_DATE", "20230101"):
            root = Path(temporary)
            client.fetch.return_value = original
            download_financial_cache(client, root, pd.Timestamp("2023-06-01"),
                                     {"fina_indicator": ["eps", "roe"]})
            client.fetch.return_value = original.drop(columns="roe").assign(ann_date="20230510", eps=2)
            download_financial_cache(client, root, pd.Timestamp("2023-06-01"), {"fina_indicator": ["eps"]})
            actual = CsvClient(root).fetch("fina_indicator_vip", period="20230331").set_index("ann_date")
            self.assertEqual(actual.loc["20230420", "roe"], 10)
            self.assertTrue(pd.isna(actual.loc["20230510", "roe"]))
            self.assertEqual(actual.loc["20230510", "eps"], 2)
            download_financial_cache(client, root, pd.Timestamp("2023-06-01"), {"fina_indicator": ["eps"]})
            replay = CsvClient(root).fetch("fina_indicator_vip", period="20230331").set_index("ann_date")
            pd.testing.assert_frame_equal(actual, replay)

    def test_csi1000_complete_pagination_with_small_pages_and_month_snapshots(self):
        pages = [pd.DataFrame({"trade_date": [date] * 2,
                               "con_code": [f"{start + i:06d}.SZ" for i in range(2)]})
                 for date, start in (("20240131", 1), ("20240131", 3), ("20240115", 1))]
        pro = Mock()
        pro.query.side_effect = pages + [pd.DataFrame()]
        with patch.object(C, "REQUEST_INTERVAL", 0), patch.object(C, "RETRIES", 0), \
                patch.object(C, "PAGE_SIZE", 2):
            result = TushareClient(pro).fetch("index_weight", index_code="000852.SH",
                                             start_date="20240101", end_date="20240131")
        self.assertEqual([call.kwargs["offset"] for call in pro.query.call_args_list], [0, 2, 4, 6])
        self.assertEqual(result.groupby("trade_date").size().to_dict(), {"20240115": 2, "20240131": 4})


if __name__ == "__main__":
    unittest.main()
