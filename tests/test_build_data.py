"""Offline checks for refresh scope and preparing an immutable release snapshot."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import pandas as pd

from scripts import build_data as builder
from scripts import config as C
from scripts.tushare.data import CsvClient, TushareClient, download_cache, download_data

A, B = "000001.SZ", "000002.SZ"
CSI1000 = "000852.SH"


class PaginationTailTest(unittest.TestCase):
    def test_successful_empty_tail_is_requested_once(self):
        pro = Mock()
        pro.query.side_effect = [pd.DataFrame({"value": [1, 2]}), pd.DataFrame()]
        with patch.object(C, "PAGE_SIZE", 2), patch.object(C, "RETRIES", 3), \
                patch("scripts.tushare.data.time.sleep"):
            result = TushareClient(pro).fetch("index_weight")
        self.assertEqual(result.value.tolist(), [1, 2])
        self.assertEqual([call.kwargs["offset"] for call in pro.query.call_args_list], [0, 2])

    def test_tail_exceptions_retry_before_accepting_empty_success(self):
        pro = Mock()
        pro.query.side_effect = [pd.DataFrame({"value": [1, 2]}), TimeoutError("transient"), pd.DataFrame()]
        with patch.object(C, "PAGE_SIZE", 2), patch.object(C, "RETRIES", 3), \
                patch("scripts.tushare.data.time.sleep") as sleep:
            result = TushareClient(pro).fetch("index_weight")
        self.assertEqual(result.value.tolist(), [1, 2])
        self.assertEqual([call.kwargs["offset"] for call in pro.query.call_args_list], [0, 2, 2])
        sleep.assert_any_call(3)

    def test_persistent_tail_exception_never_returns_partial_result(self):
        pro = Mock()
        pro.query.side_effect = [pd.DataFrame({"value": [1, 2]}), TimeoutError("failed"), TimeoutError("failed")]
        with patch.object(C, "PAGE_SIZE", 2), patch.object(C, "RETRIES", 1), \
                patch("scripts.tushare.data.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "offset=2"):
                TushareClient(pro).fetch("index_weight")
        self.assertEqual([call.kwargs["offset"] for call in pro.query.call_args_list], [0, 2, 2])

    def test_empty_first_page_retains_retry_behavior(self):
        pro = Mock()
        pro.query.side_effect = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame({"value": [1]})]
        with patch.object(C, "PAGE_SIZE", 2), patch.object(C, "RETRIES", 2), \
                patch("scripts.tushare.data.time.sleep"):
            result = TushareClient(pro).fetch("daily")
        self.assertEqual(result.value.tolist(), [1])
        self.assertEqual([call.kwargs["offset"] for call in pro.query.call_args_list], [0, 0, 0])


class IndexRefreshTest(unittest.TestCase):
    def test_csi1000_refresh_does_not_refresh_daily_or_other_historical_indices(self):
        today = pd.Timestamp("2024-04-01")
        calendar = pd.DatetimeIndex(["2024-01-02"])
        client = Mock()

        def fetch(api, **params):
            self.assertEqual(api, "index_weight")
            return pd.DataFrame({"index_code": [params["index_code"]], "con_code": [B],
                                 "trade_date": [params["start_date"]], "weight": [100.]})

        client.fetch.side_effect = fetch
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = pd.DataFrame({"ts_code": [A]})
            for name in ("stock_basic.csv", "trade_cal.csv", "industry.csv",
                         *(f"{api}/20240102.csv" for api in
                           ("daily", "adj_factor", "daily_basic", "stock_st", "stk_limit"))):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                common.to_csv(path, index=False)
            daily = root / "daily/20240102.csv"
            original_daily = daily.read_bytes()
            for code in ("000300.SH", CSI1000):
                for month in ("202401", "202402", "202403", "202404"):
                    path = root / "index_weight" / code / f"{month}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"index_code": [code], "con_code": [A],
                                  "trade_date": [month + "01"], "weight": [100.]}).to_csv(path, index=False)
            with patch.object(C, "START_DATE", "20240101"), patch.object(C, "BUILD_PIT", False), \
                    patch.object(C, "REFRESH_CACHE", False), \
                    patch.object(C, "INDEX_CODES", {"csi300": "000300.SH", "csi1000": CSI1000}), \
                    patch("scripts.tushare.data.fetch_calendar", return_value=(calendar, calendar)), \
                    patch("scripts.tushare.data.cache_day", return_value=today):
                download_cache(client, root, today, refresh_index_codes=(CSI1000,))
            requests = [(call.args[0], call.kwargs["index_code"], call.kwargs["start_date"][:6])
                        for call in client.fetch.call_args_list]
            self.assertEqual(requests, [("index_weight", "000300.SH", "202404"),
                                        *(('index_weight', CSI1000, month)
                                          for month in ("202401", "202402", "202403", "202404"))])
            self.assertEqual(daily.read_bytes(), original_daily)
            self.assertEqual(pd.read_csv(root / "index_weight/000300.SH/202401.csv").con_code.tolist(), [A])
            self.assertEqual(pd.read_csv(root / f"index_weight/{CSI1000}/202401.csv").con_code.tolist(), [B])

    def test_download_entry_forwards_selected_index_codes(self):
        sdk = Mock()
        with patch.dict("sys.modules", {"tushare": sdk}), patch.object(C, "TOKEN", "unit-test"), \
                patch("scripts.tushare.data.download_cache") as cache:
            download_data("cache", pd.Timestamp("2024-04-01"), refresh_index_codes=(CSI1000,))
        self.assertEqual(cache.call_args.kwargs, {"refresh_index_codes": (CSI1000,)})


class PreparedBuildTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        temporary = self.stack.enter_context(TemporaryDirectory())
        self.base = Path(temporary).resolve()
        self.output, self.cache, self.work = [self.base / name for name in ("live", "cache", "stage")]
        self.output.mkdir()
        self.cache.mkdir()
        (self.output / "original").write_bytes(b"old dataset")
        self.calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
        for name, value in (("OUTPUT_DIR", str(self.output)), ("CACHE_DIR", str(self.cache)),
                            ("BUILD_PIT", True), ("REFRESH_CACHE", False)):
            self.stack.enter_context(patch.object(C, name, value))
        self.download = self.stack.enter_context(patch.object(builder, "download_data", return_value=CsvClient(self.cache)))
        self.fetch_calendar = self.stack.enter_context(patch.object(builder, "fetch_calendar",
                                                                   return_value=(self.calendar, self.calendar)))

        def basic(client, root):
            frame = pd.DataFrame({"ts_code": [A]})
            frame.to_csv(root / "stock_basic.csv", index=False)
            return frame

        def daily(client, root, frame, calendar, today):
            (root / "daily-payload").write_bytes(b"stable daily bytes")
            return calendar

        def financial(client, root, codes, today):
            directory = root / "financial"
            directory.mkdir(exist_ok=True)
            (directory / "pit-generation").write_bytes(uuid4().bytes)

        self.builders = [self.stack.enter_context(patch.object(builder, name, side_effect=effect))
                         for name, effect in (("build_stock_basic", basic), ("build_daily", daily),
                                              ("build_indices", None), ("build_industry", None),
                                              ("build_financial", financial))]

    def tearDown(self):
        self.stack.close()

    @staticmethod
    def snapshot(root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()}

    def test_default_publish_removes_backup_and_work(self):
        result = builder.build_data(download=False, resume_dir=self.work)
        self.assertEqual(result, self.output)
        self.assertTrue((result / "daily-payload").exists())
        self.assertFalse((result / "original").exists())
        self.assertFalse(self.work.exists())
        self.assertEqual(list(self.base.glob("live.backup-*")), [])
        self.download.assert_not_called()

    def test_prepare_then_publish_preserves_exact_bytes_and_keeps_backup(self):
        with patch.object(Path, "rename", side_effect=AssertionError("preparation must not rename")):
            stage = builder.build_data(resume_dir=self.work, keep_backup=True,
                                       refresh_index_codes=(CSI1000,), publish=False)
        self.assertEqual(stage, self.work / "cn_data")
        self.assertEqual((self.output / "original").read_bytes(), b"old dataset")
        self.assertEqual(list(self.base.glob("live.backup-*")), [])
        state = json.loads((self.work / "build.json").read_text(encoding="utf-8"))
        self.assertTrue(state["prepared"])
        self.assertEqual(state["refresh_index_codes"], [CSI1000])
        self.assertEqual(self.download.call_args.kwargs, {"refresh_index_codes": (CSI1000,)})
        prepared = self.snapshot(stage)
        for mock in (self.download, self.fetch_calendar, *self.builders):
            mock.reset_mock()
        result = builder.build_data(download=False, resume_dir=self.work, keep_backup=True,
                                    refresh_index_codes=(CSI1000,))
        self.assertEqual(result, self.output)
        self.assertEqual(self.snapshot(result), prepared)
        for mock in (self.download, self.fetch_calendar, *self.builders):
            mock.assert_not_called()
        backups = list(self.base.glob("live.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "original").read_bytes(), b"old dataset")
        self.assertFalse(self.work.exists())

    def test_prepared_refresh_identity_cannot_change(self):
        stage = builder.build_data(resume_dir=self.work, refresh_index_codes=(CSI1000,), publish=False)
        prepared = self.snapshot(stage)
        with self.assertRaisesRegex(ValueError, "配置已改变"):
            builder.build_data(download=False, resume_dir=self.work)
        with patch.object(C, "REFRESH_CACHE", True), self.assertRaisesRegex(ValueError, "配置已改变"):
            builder.build_data(download=False, resume_dir=self.work, refresh_index_codes=(CSI1000,))
        self.assertEqual(self.snapshot(stage), prepared)
        self.assertEqual((self.output / "original").read_bytes(), b"old dataset")

    def test_publication_failure_restores_previous_dataset(self):
        stage = builder.build_data(download=False, resume_dir=self.work, publish=False)
        prepared = self.snapshot(stage)
        rename = Path.rename

        def fail_publication(path, target):
            if path == stage:
                raise OSError("publication failed")
            return rename(path, target)

        with patch.object(Path, "rename", fail_publication), self.assertRaisesRegex(OSError, "publication failed"):
            builder.build_data(download=False, resume_dir=self.work, keep_backup=True)
        self.assertEqual((self.output / "original").read_bytes(), b"old dataset")
        self.assertEqual(self.snapshot(stage), prepared)
        self.assertTrue((self.work / "build.json").exists())
        self.assertEqual(list(self.base.glob("live.backup-*")), [])

    def test_cli_forwards_prepare_refresh_and_backup_options(self):
        args = ["build_data.py", "--resume-dir", str(self.work), "--no-download",
                "--keep-backup", "--prepare-only", "--refresh-csi1000"]
        with patch("sys.argv", args), patch.object(builder, "build_data") as build:
            builder.main()
        build.assert_called_once_with(download=False, resume_dir=str(self.work), keep_backup=True,
                                      refresh_index_codes=(CSI1000,), publish=False)


if __name__ == "__main__":
    unittest.main()
