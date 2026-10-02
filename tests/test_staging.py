"""Publication staging keeps inherited access and preserves live data on failure."""

from contextlib import ExitStack
import ctypes
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.data.pit import StockPIT
from scripts import config as C
from scripts._staging import staging_directory
from scripts.build_instruments import _staging_directory
from scripts.build_limits import build_limits
from scripts.build_pit import build_pit

A = "000001.SZ"


def windows_acl(path):
    """Read DACL ACE bytes using the Windows API; no shell or extra dependency."""
    from ctypes import wintypes

    class ACL(ctypes.Structure):
        _fields_ = [("revision", wintypes.BYTE), ("reserved", wintypes.BYTE),
                    ("size", wintypes.WORD), ("count", wintypes.WORD), ("reserved2", wintypes.WORD)]

    api = ctypes.WinDLL("advapi32", use_last_error=True)
    api.GetNamedSecurityInfoW.argtypes = [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
                                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.POINTER(ctypes.c_void_p)]
    api.GetNamedSecurityInfoW.restype = wintypes.DWORD
    api.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    api.GetAce.restype = wintypes.BOOL
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    dacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p()
    status = api.GetNamedSecurityInfoW(str(path), 1, 4, None, None,
                                      ctypes.byref(dacl), None, ctypes.byref(descriptor))
    if status:
        raise OSError(status, "GetNamedSecurityInfoW failed")
    try:
        if not dacl:
            return []
        acl = ctypes.cast(dacl, ctypes.POINTER(ACL)).contents
        result = []
        for index in range(acl.count):
            ace = ctypes.c_void_p()
            if not api.GetAce(dacl, index, ctypes.byref(ace)):
                raise ctypes.WinError(ctypes.get_last_error())
            header = ctypes.string_at(ace, 4)
            size = int.from_bytes(header[2:4], "little")
            result.append(ctypes.string_at(ace, size))
        return result
    finally:
        kernel.LocalFree(descriptor)


class StagingTest(unittest.TestCase):
    def test_normal_access_inheritance_and_private_instrument_helper_compatibility(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ordinary = root / "ordinary"
            ordinary.mkdir(mode=0o755)
            for context in (staging_directory(root, prefix=".test-build-"), _staging_directory(root)):
                with context as stage:
                    self.assertEqual(stage.parent, root)
                    (stage / "payload").write_bytes(b"readable")
                    if os.name == "nt":
                        acl = windows_acl(stage)
                        self.assertTrue(acl)
                        self.assertTrue(all(ace[1] & 0x10 for ace in acl), "new stage must inherit every ACE")
                        self.assertEqual(acl, windows_acl(ordinary))
                    else:
                        self.assertEqual(stage.stat().st_mode & 0o777, ordinary.stat().st_mode & 0o777)
                self.assertFalse(stage.exists())

    def test_failure_cleans_only_its_stage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            marker = root / "unchanged"
            marker.write_bytes(b"live data")
            with self.assertRaisesRegex(ValueError, "generation failed"):
                with staging_directory(root, prefix=".test-build-") as stage:
                    (stage / "partial").write_bytes(b"partial")
                    raise ValueError("generation failed")
            self.assertFalse(stage.exists())
            self.assertEqual(marker.read_bytes(), b"live data")

    def test_prefix_cannot_escape_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "超出"):
                with staging_directory(root, prefix="../outside-"):
                    self.fail("invalid prefix created a stage")
            self.assertEqual(list(root.iterdir()), [])


class IndependentBuilderStagingTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        parent = Path(self.stack.enter_context(TemporaryDirectory())).resolve()
        self.root, self.cache = parent / "cn_data", parent / "cache"
        self.root.mkdir()
        self.cache.mkdir()
        pd.DataFrame({"ts_code": [A]}).to_csv(self.root / "stock_basic.csv", index=False)
        self.stack.enter_context(patch.object(C, "START_DATE", "20230101"))
        self.stack.enter_context(patch.object(C, "PIT_FIELDS", {"fina_indicator": ["eps"]}))
        path = self.cache / "financial/fina_indicator/202303.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"ts_code": [A], "ann_date": ["20230420"], "end_date": ["20230331"],
                      "update_flag": ["1"], "eps": [2.]}).to_csv(path, index=False)
        (self.root / "financial").mkdir()
        (self.root / "financial/original").write_bytes(b"previous financial data")

    def tearDown(self):
        self.stack.close()

    def test_pit_publishes_valid_files_with_normal_file_access(self):
        ordinary = self.root / "ordinary"
        ordinary.write_bytes(b"normal access")
        build_pit(self.root, self.cache, today="2023-06-01")
        pit = self.root / "financial" / A
        self.assertEqual(StockPIT.read(pit).records["value"].tolist(), [2.])
        self.assertFalse((self.root / "financial/original").exists())
        self.assertEqual(list(self.root.glob(".pit-build-*")), [])
        if os.name == "nt":
            for name in ("pit.data", "pit.index"):
                self.assertEqual(windows_acl(pit / name), windows_acl(ordinary))

    def test_pit_generation_failure_keeps_original_and_cleans_stage(self):
        with patch("scripts.build_pit.build_financial", side_effect=ValueError("invalid source")), \
                self.assertRaisesRegex(ValueError, "invalid source"):
            build_pit(self.root, self.cache, today="2023-06-01")
        self.assertEqual((self.root / "financial/original").read_bytes(), b"previous financial data")
        self.assertEqual(list(self.root.glob(".pit-build-*")), [])
        self.assertEqual(list(self.root.glob(".financial.backup-*")), [])

    def test_pit_publication_failure_restores_original_and_cleans_stage(self):
        rename = Path.rename

        def fail_publication(path, target):
            if path.name == "financial" and path.parent.name.startswith(".pit-build-"):
                raise OSError("publication failed")
            return rename(path, target)

        with patch.object(Path, "rename", fail_publication), self.assertRaisesRegex(OSError, "publication failed"):
            build_pit(self.root, self.cache, today="2023-06-01")
        self.assertEqual((self.root / "financial/original").read_bytes(), b"previous financial data")
        self.assertEqual(list(self.root.glob(".pit-build-*")), [])
        self.assertEqual(list(self.root.glob(".financial.backup-*")), [])

    def make_limits(self):
        (self.root / "calendars").mkdir()
        (self.root / "calendars/day.txt").write_text("2024-01-02\n2024-01-03\n")
        directory = self.root / "features" / A
        directory.mkdir(parents=True)
        np.asarray([0, 10, 11], dtype="<f4").tofile(directory / "close.day.bin")
        for day in ("20240102", "20240103"):
            for api, frame in (("daily", pd.DataFrame({"ts_code": [A], "trade_date": [day]})),
                               ("stk_limit", pd.DataFrame({"ts_code": [A], "trade_date": [day],
                                                           "up_limit": [12.], "down_limit": [9.]}))):
                path = self.cache / api / f"{day}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
        return directory

    def test_limits_publish_correct_values_with_normal_file_access(self):
        directory = self.make_limits()
        build_limits(self.root, self.cache)
        np.testing.assert_array_equal(np.fromfile(directory / "up_limit.day.bin", dtype="<f4"), [0, 12, 12])
        np.testing.assert_array_equal(np.fromfile(directory / "down_limit.day.bin", dtype="<f4"), [0, 9, 9])
        self.assertEqual(list(self.root.glob(".limit-build-*")), [])
        if os.name == "nt":
            for field in ("up_limit", "down_limit"):
                self.assertEqual(windows_acl(directory / f"{field}.day.bin"),
                                 windows_acl(directory / "close.day.bin"))

    def test_limits_invalid_source_keeps_published_files_and_cleans_stage(self):
        directory = self.make_limits()
        for field in ("up_limit", "down_limit"):
            np.asarray([0, 13, 13] if field == "up_limit" else [0, 8, 8], dtype="<f4").tofile(
                directory / f"{field}.day.bin")
        original = {path: path.read_bytes() for path in directory.glob("*.bin")}
        source = self.cache / "stk_limit/20240103.csv"
        frame = pd.read_csv(source, dtype={"trade_date": str})
        frame["up_limit"] = 7.
        frame.to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "涨停价低于跌停价"):
            build_limits(self.root, self.cache)
        self.assertEqual({path: path.read_bytes() for path in directory.glob("*.bin")}, original)
        self.assertEqual(list(self.root.glob(".limit-build-*")), [])


if __name__ == "__main__":
    unittest.main()
