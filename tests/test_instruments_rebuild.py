"""Offline pool rebuilding preserves data and recovers publication failures."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import config as C
from scripts.build_instruments import build_instruments


class InstrumentsRebuildTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        parent = Path(temporary.name)
        self.root, self.cache = parent / "cn_data", parent / "cache"
        for directory in (self.root / "calendars", self.root / "instruments",
                          self.root / "features", self.root / "financial",
                          self.cache / "index_weight/000300.SH"):
            directory.mkdir(parents=True)
        self.days = pd.bdate_range("2024-01-02", periods=4)
        (self.root / "calendars/day.txt").write_text("\n".join(self.days.strftime("%Y-%m-%d")))
        pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                      "list_date": ["2020-01-01", "2020-01-01", "2024-01-03"],
                      "delist_date": [None, "2024-01-05", None]}).to_csv(
                          self.root / "stock_basic.csv", index=False)
        (self.root / "instruments/all.txt").write_text("000001.SZ 2024-01-02 2024-01-03\n")
        (self.root / "instruments/st.txt").write_text("000002.SZ 2024-01-03 2024-01-04\n")
        (self.root / "instruments/custom.txt").write_text("000001.SZ 2024-01-02 2024-01-05\n")
        (self.root / "features/marker").write_bytes(b"daily data unchanged")
        (self.root / "financial/marker").write_bytes(b"financial data unchanged")
        pd.DataFrame({"index_code": ["000300.SH"] * 2, "trade_date": ["20240102"] * 2,
                      "con_code": ["000001.SZ", "000003.SZ"], "weight": [50, 50]}).to_csv(
                          self.cache / "index_weight/000300.SH/202401.csv", index=False)
        self.addCleanup(patch.stopall)
        patch.object(C, "START_DATE", "20240101").start()
        patch.object(C, "INDEX_CODES", {"csi300": "000300.SH"}).start()

    def content(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def test_rebuild_retains_other_files_and_complete_pool_backup(self):
        original = self.content()
        cache_file = self.cache / "index_weight/000300.SH/202401.csv"
        cached = cache_file.read_bytes()
        backup = build_instruments(self.root, self.cache)
        self.assertEqual(backup.parent, self.root.resolve())
        self.assertEqual((backup / "all.txt").read_bytes(), original["instruments/all.txt"])
        self.assertEqual((self.root / "instruments/all.txt").read_text().splitlines(), [
            "000001.SZ\t2024-01-02\t2024-01-05",
            "000002.SZ\t2024-01-02\t2024-01-04",
            "000003.SZ\t2024-01-03\t2024-01-05",
        ])
        self.assertEqual((self.root / "instruments/csi300.txt").read_text().splitlines(), [
            "000001.SZ\t2024-01-02\t2024-01-05",
            "000003.SZ\t2024-01-02\t2024-01-05",
        ])
        for relative, data in original.items():
            if relative != "instruments/all.txt":
                self.assertEqual((self.root / relative).read_bytes(), data)
        self.assertEqual(cache_file.read_bytes(), cached)

    def test_generation_failure_does_not_publish_partial_pools(self):
        original = self.content()
        with patch("scripts.build_instruments.build_indices", side_effect=FileNotFoundError("missing cache")):
            with self.assertRaises(FileNotFoundError):
                build_instruments(self.root, self.cache)
        self.assertEqual(self.content(), original)

    def test_failed_publication_restores_original_pools(self):
        original, rename = self.content(), Path.rename

        def fail_publish(path, target):
            if path.name == "instruments" and path.parent.name.startswith(".instruments-build-"):
                raise OSError("publication failure")
            return rename(path, target)

        with patch.object(Path, "rename", fail_publish):
            with self.assertRaises(OSError):
                build_instruments(self.root, self.cache)
        self.assertEqual(self.content(), original)

    def test_generated_file_links_are_rejected_before_any_source_is_modified(self):
        # Mock the predicate so Windows symlink privileges are not required.
        original, is_symlink = self.content(), Path.is_symlink

        def linked_output(path):
            return (path.name == "all.txt" and path.parent.parent.name.startswith(".instruments-build-")) \
                or is_symlink(path)

        with patch.object(Path, "is_symlink", linked_output):
            with self.assertRaises(ValueError):
                build_instruments(self.root, self.cache)
        self.assertEqual(self.content(), original)

    def test_compact_dates_are_parsed_as_dates_instead_of_nanoseconds(self):
        basic_path = self.root / "stock_basic.csv"
        frame = pd.read_csv(basic_path)
        for field in ("list_date", "delist_date"):
            frame[field] = pd.to_datetime(frame[field]).dt.strftime("%Y%m%d")
        frame.to_csv(basic_path, index=False)
        build_instruments(self.root, self.cache)
        self.assertIn("000003.SZ\t2024-01-03\t2024-01-05", (self.root / "instruments/all.txt").read_text())


if __name__ == "__main__":
    unittest.main()
