"""Sparse Loguru events, batch warning summaries and application-owned sinks."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from loguru import logger

from qlib.contrib.eva.alpha import calc_all_ic, pred_autocorr_all
from qlib.log import log_warning, summarize_warnings
from scripts import config as C
from scripts.dump.pit import prepare_financial
from scripts.tushare.data import TushareClient, cache_csv, download_limit_cache


class LoggingTest(unittest.TestCase):
    def setUp(self):
        self.messages = []
        handler = logger.add(self.messages.append, format="{message}", level="TRACE")
        self.addCleanup(logger.remove, handler)

    def test_nested_summaries_emit_once_per_category(self):
        with summarize_warnings():
            log_warning("missing records", 2)
            with summarize_warnings():
                for _ in range(100):
                    log_warning("missing records")
                log_warning("retained cache files", 3)
            self.assertEqual(self.messages, [])
        self.assertEqual([m.record["message"] for m in self.messages],
                         ["missing records；数量=102", "retained cache files；数量=3"])
        self.assertTrue(all(m.record["level"].name == "WARNING" for m in self.messages))

    def test_summaries_flush_on_failure_and_reset(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with summarize_warnings():
                log_warning("missing records", 5)
                raise RuntimeError("interrupted")
        with summarize_warnings():
            log_warning("missing records", 1)
        log_warning("standalone warning", 2)
        self.assertEqual([m.record["message"] for m in self.messages],
                         ["missing records；数量=5", "missing records；数量=1", "standalone warning；数量=2"])

    def test_financial_anomalies_are_counted_without_row_dumps(self):
        frame = pd.DataFrame({"ts_code": ["000001.SZ"], "ann_date": ["20230201"],
                              "end_date": ["20230331"], "update_flag": ["1"], "eps": [99.]})
        with summarize_warnings():
            for _ in range(4):
                rows = prepare_financial(frame, "fina_indicator", ["eps"], {"000001.SZ"}, "2023-06-01")
                self.assertTrue(rows.empty)
        self.assertEqual(len(self.messages), 1)
        message = self.messages[0].record["message"]
        self.assertIn("数量=4", message)
        self.assertNotIn("000001.SZ", message)
        self.assertNotIn("99", message)

    def test_empty_responses_and_cached_reads_are_quiet(self):
        pro = Mock()
        pro.query.return_value = pd.DataFrame()
        with patch.object(C, "RETRIES", 2), patch("scripts.tushare.data.time.sleep"):
            self.assertTrue(TushareClient(pro).fetch("stock_st", trade_date="20240102").empty)
        self.assertEqual(pro.query.call_count, 3)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "cached.csv"
            pd.DataFrame({"value": [1]}).to_csv(path, index=False)
            load = Mock(side_effect=AssertionError("cache should be reused"))
            with patch.object(C, "REFRESH_CACHE", False):
                self.assertEqual(cache_csv(path, load).value.tolist(), [1])
            load.assert_not_called()
        self.assertEqual(self.messages, [])

    def test_retry_recovery_does_not_log_request_or_exception_payload(self):
        pro = Mock()
        pro.query.side_effect = [RuntimeError("secret-token"), RuntimeError("secret-token"),
                                 pd.DataFrame({"value": [1]})]
        with patch.object(C, "RETRIES", 2), patch("scripts.tushare.data.time.sleep"):
            result = TushareClient(pro).fetch("daily", fields="secret-request")
        self.assertEqual(result.value.tolist(), [1])
        self.assertEqual(len(self.messages), 1)
        message = self.messages[0].record["message"]
        self.assertIn("daily", message)
        self.assertIn("数量=2", message)
        self.assertNotIn("secret", message)

    def test_failed_retries_propagate_without_success_log(self):
        pro = Mock()
        pro.query.side_effect = RuntimeError("network down")
        with patch.object(C, "RETRIES", 2), patch("scripts.tushare.data.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "daily 请求失败"):
                TushareClient(pro).fetch("daily", trade_date="20240102")
        self.assertEqual(pro.query.call_count, 3)
        self.assertEqual(self.messages, [])

    def test_download_has_stage_logs_not_per_date_progress(self):
        client = Mock()
        client.fetch.return_value = pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
        dates = pd.bdate_range("2024-01-02", periods=30)
        stdout, stderr = StringIO(), StringIO()
        with TemporaryDirectory() as temporary, redirect_stdout(stdout), redirect_stderr(stderr):
            download_limit_cache(client, temporary, dates, dates[-1])
        self.assertEqual(client.fetch.call_count, len(dates))
        self.assertEqual(len(self.messages), 2)
        self.assertTrue(all(m.record["level"].name == "INFO" for m in self.messages))
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("%|", stderr.getvalue())

    def test_batch_evaluation_has_no_joblib_progress(self):
        index = pd.MultiIndex.from_product([pd.date_range("2024-01-02", periods=2), list("ABC")],
                                          names=["datetime", "instrument"])
        pred = pd.Series([1., 2., 3.] * 2, index=index)
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            calc_all_ic({"alpha": pred}, pred, n_jobs=1)
            pred_autocorr_all({"alpha": pred}, n_jobs=1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(self.messages, [])

    def test_import_is_quiet_and_preserves_application_handler(self):
        code = """
from io import StringIO
from loguru import logger
stream = StringIO()
logger.remove()
handler = logger.add(stream, level='WARNING', format='{message}')
import qlib
import qlib.log
import qlib.backtest
import qlib.contrib.report.analysis_model
import scripts.build_data
import scripts.build_pit
import scripts.build_limits
import scripts.build_rolling
assert stream.getvalue() == ''
logger.info('filtered out')
assert stream.getvalue() == ''
logger.warning('application handler retained')
assert stream.getvalue() == 'application handler retained\\n'
logger.remove(handler)
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
