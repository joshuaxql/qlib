"""PIT v2 round trips, independent as-of oracle, event projections and builds."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from qlib.data import LocalProvider
from qlib.data.base import validate
from qlib.data.filter import ExpressionFilter
from qlib.data.pit import (HEADER, RECORD_DTYPE, INDEX_DTYPE, NO_NEXT, PITStore, StockPIT,
                           read_registry, register_fields, write_stock)
from qlib.data._libs import pit as native
from scripts import config as C
from scripts.build_pit import build_pit
from scripts.dump.pit import prepare_financial
from scripts.tushare.data import CsvClient, download_financial_cache
from scripts.tushare.fields import FINA_INDICATOR_FIELDS

A, B = "000001.SZ", "600000.SH"
PROFIT, ASSETS, EPS = "profit_dedt", "tangible_asset", "eps"


class PITTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.financial = self.root / "financial"
        self.dates = pd.bdate_range("2024-04-01", "2024-04-12", name="datetime")
        for folder in ("calendars", "instruments", f"features/{A}", f"features/{B}"):
            (self.root / folder).mkdir(parents=True)
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")), encoding="utf-8")
        (self.root / "instruments/all.txt").write_text(
            f"{A} 2024-04-01 2024-04-12\n{B} 2024-04-01 2024-04-12\n", encoding="utf-8")
        pd.DataFrame({"ts_code": [A, B], "list_date": ["2000-01-01"] * 2}).to_csv(self.root / "stock_basic.csv", index=False)
        for code in (A, B):
            for field, value in (("close", 10), ("factor", 2)):
                np.array([0, *([value] * len(self.dates))], dtype="<f4").tofile(
                    self.root / "features" / code / f"{field}.day.bin")
        self.ids = register_fields(self.financial, {
            name: {"unit": "source", "basis": "source", "source": "fina_indicator_vip"}
            for name in (PROFIT, ASSETS, EPS)
        })
        self.rows = pd.DataFrame([
            (20240402, 202303, PROFIT, 10), (20240402, 202304, PROFIT, 20),
            (20240404, 202401, PROFIT, 30), (20240408, 202303, PROFIT, 40),
            (20240409, 202401, PROFIT, np.nan), (20240410, 202401, PROFIT, 35),
            (20240403, 202304, ASSETS, 100), (20240411, 202401, ASSETS, 200),
            (20240402, 202301, EPS, 1), (20240404, 202303, EPS, 3),
        ], columns=["date", "period", "field", "value"])
        write_stock(self.financial, A, self.rows.sample(frac=1, random_state=7))
        self.provider = LocalProvider(self.root)

    def test_layout_indices_links_and_file_count(self):
        self.assertEqual(RECORD_DTYPE.itemsize, 28)
        self.assertEqual(INDEX_DTYPE.itemsize, 24)
        self.assertEqual(HEADER.size, 72)
        self.assertEqual(sorted(p.name for p in (self.financial / A).iterdir()), ["pit.data", "pit.index"])
        stock = StockPIT.read(self.financial / A)
        self.assertEqual(len(stock.records), len(self.rows))
        for group in stock.index:
            offset = int(group["offset"])
            count = 0
            while offset != NO_NEXT:
                row = stock.records[(offset - HEADER.size) // RECORD_DTYPE.itemsize]
                self.assertEqual(row["field_id"], group["field_id"])
                self.assertEqual(row["period"], group["period"])
                offset = int(row["next"])
                count += 1
            self.assertEqual(count, group["count"])

    def test_snapshot_matches_independent_dataframe_oracle(self):
        for date in self.dates:
            known = self.rows[self.rows.date <= int(date.strftime("%Y%m%d"))]
            expected = known.sort_values("date").drop_duplicates(["field", "period"], keep="last")
            actual = self.provider.financial([A], [PROFIT, ASSETS, EPS], date).droplevel("instrument")
            expected = expected.pivot(index="period", columns="field", values="value").reindex(columns=actual.columns)
            expected.columns.name = None
            pd.testing.assert_frame_equal(actual, expected, check_index_type=False)

    def test_event_projection_revisions_gaps_and_explicit_nan(self):
        expressions = [f"P($${PROFIT})", f"P(Mean($${PROFIT}, 3))", f"PRef($${PROFIT}, -1)",
                       f"$${PROFIT}", f"PRef($${EPS}, -1)", f"P($${PROFIT}) / $close"]
        actual = self.provider.features([A], expressions, allow_future=False).loc[A]
        latest = [np.nan, 20, 20, 30, 30, 30, np.nan, 35, 35, 35]
        np.testing.assert_allclose(actual.iloc[:, 0], latest, equal_nan=True)
        np.testing.assert_allclose(actual.iloc[:, 1], [np.nan, 15, 15, 20, 20, 30, 30, 95 / 3, 95 / 3, 95 / 3], equal_nan=True)
        np.testing.assert_allclose(actual.iloc[:, 2], [np.nan, 10, 10, 20, 20, 20, 20, 20, 20, 20], equal_nan=True)
        np.testing.assert_allclose(actual.iloc[:, 3], latest, equal_nan=True)
        self.assertTrue(actual.iloc[:, 4].isna().all())  # Missing Q2 stays missing.
        np.testing.assert_allclose(actual.iloc[:, 5], np.array(latest) / 20, equal_nan=True)
        short = self.provider.features([A], expressions, "2024-04-05", "2024-04-08", allow_future=False)
        pd.testing.assert_frame_equal(short.loc[A], actual.loc["2024-04-05":"2024-04-08"])

    def test_multiple_fields_and_projections_share_one_scan(self):
        stock = self.provider._pit.stock(A)
        expressions = [f"P($${PROFIT} / $${ASSETS})", f"P($${PROFIT})", f"P(Mean($${PROFIT}, 3))"]
        with patch.object(stock, "events", wraps=stock.events) as events:
            actual = self.provider.features([A], expressions).loc[A]
        self.assertEqual(events.call_count, 1)
        self.assertAlmostEqual(actual.loc["2024-04-03"].iloc[0], 0.2)
        self.assertTrue(np.isnan(actual.loc["2024-04-04"].iloc[0]))
        self.assertAlmostEqual(actual.loc["2024-04-11"].iloc[0], 35 / 200)

    def test_event_projection_against_daily_snapshot_oracle(self):
        expression = f"P(Mean($${PROFIT}, 3) / Ref($${PROFIT}, 1))"
        result = self.provider.features([A], [expression]).iloc[:, 0]
        expected = []
        for date in self.dates:
            rows = self.rows[(self.rows.field == PROFIT) & (self.rows.date <= int(date.strftime("%Y%m%d")))]
            rows = rows.sort_values("date").drop_duplicates("period", keep="last").sort_values("period")
            if len(rows) < 2:
                expected.append(np.nan)
                continue
            values = rows.set_index("period").value
            expected.append(values.tail(3).mean() / values.iloc[-2])
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_bare_field_names_filter_catalog_and_missing_stock(self):
        self.assertEqual(self.provider.fields("financial"), sorted((PROFIT, ASSETS, EPS)))
        self.assertEqual(self.provider.fields("financial", A), sorted((PROFIT, ASSETS, EPS)))
        self.assertEqual(self.provider.fields("financial", B), [])
        np.testing.assert_allclose(self.provider.features([A], ["P($$profit_dedt)"]).iloc[:, 0],
                                   self.provider.features([A], [f"P($${PROFIT})"]).iloc[:, 0], equal_nan=True)
        pool = self.provider.instruments([A], [ExpressionFilter("P($$profit_dedt) > 25")])
        selected = self.provider.features(pool, ["$close"])
        self.assertNotIn(pd.Timestamp("2024-04-09"), selected.index.get_level_values("datetime"))
        self.assertIn(pd.Timestamp("2024-04-10"), selected.index.get_level_values("datetime"))
        self.assertTrue(self.provider.features([B], ["P($$profit_dedt)"]).iloc[:, 0].isna().all())
        with self.assertRaises(FileNotFoundError):
            LocalProvider(self.root, missing="raise").features([B], ["P($$profit_dedt)"])
        for unknown in ("fina_indicator.eps", "eps_q"):
            with self.subTest(name=unknown), self.assertRaises(KeyError):
                self.provider.features([A], [f"P($${unknown})"])

    def test_incremental_merge_stable_ids_and_cache_invalidation(self):
        before = self.provider.features([A], [f"P($${PROFIT})"], end_time="2024-04-08", allow_future=False)
        cached = self.provider._pit.stock(A)
        register_fields(self.financial, {"roe": {"unit": "source"}})
        registry = read_registry(self.financial)
        for name, field_id in self.ids.items():
            self.assertEqual(registry["fields"][str(field_id)]["name"], name)
        additions = pd.DataFrame([(20240412, 202401, PROFIT, 90), (20240412, 202401, "roe", 7)],
                                 columns=self.rows.columns)
        write_stock(self.financial, A, additions)
        self.assertIsNot(cached, self.provider._pit.stock(A))
        self.assertEqual(len(self.provider.financial_records(A)), len(self.rows) + 2)
        self.assertEqual(self.provider.financial([A], [PROFIT], "2024-04-12").loc[(A, 202401), PROFIT], 90)
        pd.testing.assert_frame_equal(before, self.provider.features([A], [f"P($${PROFIT})"],
                                                                     end_time="2024-04-08", allow_future=False))
        write_stock(self.financial, A, additions)  # Idempotent replay.
        self.assertEqual(len(self.provider.financial_records(A)), len(self.rows) + 2)
        additions.loc[0, "value"] = 91  # Same-size update must invalidate the cache too.
        write_stock(self.financial, A, additions)
        self.assertEqual(self.provider.financial([A], [PROFIT], "2024-04-12").loc[(A, 202401), PROFIT], 91)

    def test_byte_bounded_cache_and_clear(self):
        write_stock(self.financial, B, self.rows)
        size = self.provider._pit.stock(A).nbytes
        store = PITStore(self.financial, cache_bytes=size)
        first = store.stock(A)
        self.assertIs(first, store.stock(A))
        store.stock(B)
        self.assertEqual(list(store.cache), [B])
        self.assertLessEqual(store.cached_bytes, size)
        store.clear()
        self.assertEqual(store.cached_bytes, 0)
        uncached = PITStore(self.financial, cache_bytes=0)
        self.assertIsNot(uncached.stock(A), uncached.stock(A))

    def test_corrupt_data_and_mismatched_pair_rejected(self):
        path = self.financial / A / "pit.data"
        original = path.read_bytes()
        for damaged in (original[:10], original[:-1], original[:-1] + bytes([original[-1] ^ 1])):
            path.write_bytes(damaged)
            with self.assertRaises(ValueError):
                StockPIT.read(path.parent)
        path.write_bytes(original)
        old_index = (path.parent / "pit.index").read_bytes()
        write_stock(self.financial, A, self.rows)
        (path.parent / "pit.index").write_bytes(old_index)
        with self.assertRaisesRegex(ValueError, "generation"):
            StockPIT.read(path.parent)

    def test_empty_pair_and_invalid_input(self):
        write_stock(self.financial, B, self.rows.iloc[:0])
        self.assertEqual(len(StockPIT.read(self.financial / B).records), 0)
        self.assertTrue(self.provider.financial([B], [PROFIT], "2024-04-12").empty)
        for column, value in (("period", 202405), ("period", 202401.5), ("date", None),
                              ("value", np.inf), ("field", "unknown_q")):
            rows = self.rows.iloc[:1].copy()
            rows[column] = value
            with self.assertRaises((ValueError, KeyError)):
                write_stock(self.financial, B, rows)
        for expression in ("P($close)", "P(P($$profit_q))", "PRef($$profit_q, 1)",
                           "PRef($$profit_q, -1.5)", "P(Ref($$profit_q, -1))", "P($$profit_q + $close)"):
            with self.assertRaises(ValueError):
                validate(expression)

    def test_native_and_numpy_asof_match(self):
        random = np.random.default_rng(99)
        lengths = random.integers(1, 12, size=30).astype(np.uint64)
        starts = np.r_[np.uint64(0), np.cumsum(lengths)[:-1]]
        dates = np.concatenate([np.sort(random.integers(1, 100, size=int(n))) for n in lengths]).astype(np.uint32)
        for asof in (0, 10, 45, 99, 100):
            actual = native.asof_indices(dates, starts, lengths, asof)
            expected = []
            for start, length in zip(starts, lengths):
                indices = np.arange(int(start), int(start + length))
                visible = indices[dates[indices] <= asof]
                expected.append(visible[-1] if len(visible) else -1)
            np.testing.assert_array_equal(actual, expected)
            with patch.object(native, "LIBRARY_PATH", self.root / "absent.dll"):
                np.testing.assert_array_equal(native.asof_indices(dates, starts, lengths, asof), expected)

    def source_frame(self):
        return pd.DataFrame({"ts_code": [A, A, A], "ann_date": ["20230420", "20230420", "20230510"],
                             "end_date": ["20230331"] * 3, "update_flag": ["0", "1", "1"],
                             "eps": [1.0, 1.1, np.nan], "unselected": [10, 20, 30]})

    def test_csv_normalization_priorities_and_announcement_dates(self):
        frame = self.source_frame()
        rows = prepare_financial(frame, "fina_indicator", ["eps"], {A}, "2023-06-01")
        self.assertEqual(rows.date.tolist(), [20230420, 20230510])
        np.testing.assert_allclose(rows.value, [1.1, np.nan], equal_nan=True)
        self.assertEqual(set(rows.field), {"eps"})
        frame.loc[1, "ann_date"] = "20230701"
        early = prepare_financial(frame, "fina_indicator", ["eps"], {A}, "2023-04-30")
        self.assertEqual(early.value.tolist(), [1.0])
        frame.loc[1, "ann_date"] = "20230420"
        frame.loc[0, "update_flag"] = "1"
        with self.assertRaises(ValueError):
            prepare_financial(frame, "fina_indicator", ["eps"], {A}, "2023-06-01")

    def test_independent_offline_build_and_rebuild(self):
        cache = self.root / "cache"
        directory = cache / "financial" / "fina_indicator"
        directory.mkdir(parents=True)
        self.source_frame().to_csv(directory / "202303.csv", index=False)
        (self.financial / A / "legacy.data").write_bytes(b"obsolete per-field output")
        register_fields(self.financial, {"income.basic_eps_q": {"unit": "source"}})
        with patch.object(C, "START_DATE", "20230101"), patch.object(C, "PIT_FIELDS", {"fina_indicator": ("eps",)}):
            self.assertEqual(len(CsvClient(cache).fetch("fina_indicator_vip", period="20230331")), 3)
            build_pit(self.root, cache, today="2023-06-01")
            self.assertEqual(sorted(p.name for p in (self.financial / A).iterdir()), ["pit.data", "pit.index"])
            provider = LocalProvider(self.root)
            self.assertEqual(provider.fields("financial"), ["eps"])
            self.assertEqual(provider._pit.resolve("eps"), self.ids[EPS])
            records = provider.financial_records(A, "eps")
            np.testing.assert_allclose(records.value, [1.1, np.nan], equal_nan=True)
            self.assertTrue(provider.financial([A], ["eps"], "2023-04-19").empty)
            self.assertEqual(provider.financial([A], ["eps"], "2023-04-20").iloc[0, 0], 1.1)
            self.assertTrue(pd.isna(provider.financial([A], ["eps"], "2023-05-10").iloc[0, 0]))
            ids_before = read_registry(self.financial)
            build_pit(self.root, cache, today="2023-06-01")
            self.assertEqual(read_registry(self.financial), ids_before)

    def test_publication_before_period_end_cannot_leak_future_report(self):
        source = self.source_frame()
        premature = source.iloc[:1].assign(ann_date="20230201", eps=999.0)
        rows = prepare_financial(pd.concat([premature, source], ignore_index=True),
                                 "fina_indicator", ["eps"], {A}, "2023-06-01")
        self.assertEqual(rows.date.tolist(), [20230420, 20230510])
        write_stock(self.financial, B, rows, update=False)
        self.assertTrue(self.provider.financial([B], ["eps"], "2023-03-31").empty)
        self.assertEqual(self.provider.financial([B], ["eps"], "2023-04-20").iloc[0, 0], 1.1)
        self.assertTrue(prepare_financial(premature, "fina_indicator", ["eps"], {A}, "2023-06-01").empty)

    def test_direct_csv_build_groups_stocks_quarters_and_revisions(self):
        cache = self.root / "cache"
        directory = cache / "financial" / "fina_indicator"
        directory.mkdir(parents=True)
        first = self.source_frame().drop(columns="unselected")
        first = pd.concat([first, first.iloc[:1].assign(ts_code=B, eps=7.0),
                           first.iloc[:1].assign(ts_code="999999.SZ", eps=999.0)], ignore_index=True)
        first["roe"] = 10.0
        first.to_csv(directory / "202303.csv", index=False)
        second = pd.DataFrame({"ts_code": [B, A, A], "ann_date": ["20230820", "20230810", "20231010"],
                               "end_date": ["20230630"] * 3, "update_flag": ["1"] * 3,
                               "eps": [8.0, 2.0, 99.0], "roe": [np.nan, 20.0, 99.0]})
        second.to_csv(directory / "202306.csv", index=False)
        with patch.object(C, "START_DATE", "20230101"), \
                patch.object(C, "PIT_FIELDS", {"fina_indicator": ("eps", "roe")}):
            build_pit(self.root, cache, today="2023-09-01")
        provider = LocalProvider(self.root)
        a = provider.financial_records(A, "eps")
        self.assertEqual(a.period.tolist(), [202301, 202301, 202302])
        np.testing.assert_allclose(a.value, [1.1, np.nan, 2.0], equal_nan=True)
        b = provider.financial_records(B, "eps")
        self.assertEqual(b.period.tolist(), [202301, 202302])
        np.testing.assert_array_equal(b.value, [7.0, 8.0])
        self.assertTrue(pd.isna(provider.financial([B], ["roe"], "2023-09-01").loc[(B, 202302), "roe"]))
        self.assertFalse((self.financial / "999999.SZ").exists())
        self.assertEqual(sorted(p.name for p in self.financial.iterdir()), [A, B, "fields.json"])

    def test_failed_build_keeps_existing_dataset(self):
        before = (self.financial / A / "pit.data").read_bytes()
        with patch.object(C, "START_DATE", "20230101"), patch.object(C, "PIT_FIELDS", {"fina_indicator": ("eps",)}):
            with self.assertRaises(FileNotFoundError):
                build_pit(self.root, self.root / "missing_cache", today="2023-06-01")
        self.assertEqual((self.financial / A / "pit.data").read_bytes(), before)

    def test_download_whitelist_and_revision_preservation(self):
        cache = self.root / "cache"
        client = Mock()
        frame = self.source_frame().drop(columns="unselected")
        client.fetch.return_value = frame.iloc[:1]
        with patch.object(C, "START_DATE", "20230101"):
            download_financial_cache(client, cache, pd.Timestamp("2023-06-01"), {"fina_indicator": ["eps"]})
            self.assertEqual(client.fetch.call_args.args, ("fina_indicator_vip",))
            self.assertIn("eps", client.fetch.call_args.kwargs["fields"])
            self.assertNotIn("unselected", client.fetch.call_args.kwargs["fields"])
            client.fetch.return_value = frame.iloc[2:]
            download_financial_cache(client, cache, pd.Timestamp("2023-06-01"), {"fina_indicator": ["eps"]})
        saved = CsvClient(cache).fetch("fina_indicator_vip", period="20230331")
        self.assertEqual(saved.ann_date.tolist(), ["20230420", "20230510"])

    def test_default_download_includes_nondefault_indicator_fields(self):
        self.assertEqual(len(FINA_INDICATOR_FIELDS), 163)
        self.assertEqual(C.PIT_FIELDS, {"fina_indicator": FINA_INDICATOR_FIELDS})
        frame = self.source_frame().iloc[:1].drop(columns="unselected")
        frame = frame.reindex(columns=[*frame.columns, *[name for name in FINA_INDICATOR_FIELDS if name != "eps"]])
        frame["rd_exp"] = 10.0
        client = Mock()
        client.fetch.return_value = frame
        with patch.object(C, "START_DATE", "20230101"):
            download_financial_cache(client, self.root / "cache", pd.Timestamp("2023-06-01"))
        requested = client.fetch.call_args.kwargs["fields"].split(",")
        self.assertEqual(set(requested), {*FINA_INDICATOR_FIELDS, "ts_code", "ann_date", "end_date", "update_flag"})
        rows = prepare_financial(frame, "fina_indicator", list(FINA_INDICATOR_FIELDS), {A}, "2023-06-01")
        self.assertEqual(len(rows), 163)
        self.assertEqual(rows.set_index("field").loc["rd_exp", "value"], 10.0)
        self.assertTrue(pd.isna(rows.set_index("field").loc["roe", "value"]))


if __name__ == "__main__":
    unittest.main()
