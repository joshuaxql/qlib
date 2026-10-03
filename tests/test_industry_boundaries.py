"""Industry source out_date is the final included membership date."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.data import LocalProvider
from scripts import config as C
from scripts.dump.bin import _validate_industry_intervals, build_industry
from scripts.rebuild_industry import main, prepare_industry
from scripts.tushare.data import CsvClient

A, FARM, CHEM = "000001.SZ", "801010.SI", "801030.SI"


class IndustryBoundariesTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.source, self.cache, self.stage = [self.base / name for name in ("source", "cache", "stage")]
        self.source.mkdir()
        self.cache.mkdir()
        (self.source / "calendars").mkdir()
        self.dates = pd.bdate_range("2024-01-05", periods=8, name="datetime")
        (self.source / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        self.basic = pd.DataFrame({"ts_code": [A], "list_date": [pd.Timestamp("2024-01-01")],
                                   "delist_date": [pd.NaT]})
        self.basic.to_csv(self.source / "stock_basic.csv", index=False)

    def write_cache(self, rows):
        frame = pd.DataFrame(rows, columns=["l1_code", "in_date", "out_date", "is_new"])
        frame["ts_code"] = A
        frame["l1_name"] = frame.l1_code.map({FARM: "农林牧渔", CHEM: "基础化工"})
        frame.to_csv(self.cache / "industry.csv", index=False)

    def build(self, rows):
        self.write_cache(rows)
        build_industry(CsvClient(self.cache), self.source, self.basic, self.dates)
        return LocalProvider(self.source)

    def test_last_member_day_and_next_trading_day_have_no_artificial_gap(self):
        provider = self.build([(FARM, "20240105", "20240108", "N"),
                               (CHEM, "20240109", None, "Y")])
        farm = provider.universe("industry/" + FARM)[A]
        chemical = provider.universe("industry/" + CHEM)[A]
        self.assertEqual(farm.tolist(), [True, True, False, False, False, False, False, False])
        self.assertEqual(chemical.tolist(), [False, False, True, True, True, True, True, True])
        self.assertTrue((farm | chemical).all())
        self.assertFalse((farm & chemical).any())

    def test_same_day_cross_industry_overlap_fails_before_any_file_changes(self):
        self.write_cache([(FARM, "20240105", "20240109", "N"),
                          (CHEM, "20240109", None, "Y")])
        industry = self.source / "industry"
        industry.mkdir()
        previous = industry / (FARM + ".txt")
        previous.write_bytes(b"existing membership must stay intact")
        catalog = self.source / "industry_names.json"
        catalog.write_bytes(b'{"schema_version":1,"names":{"801010.SI":"original label"}}\n')
        original_catalog = catalog.read_bytes()
        with self.assertRaisesRegex(ValueError, "Cross-industry membership overlap.*2024-01-09"):
            build_industry(CsvClient(self.cache), self.source, self.basic, self.dates)
        self.assertEqual(previous.read_bytes(), b"existing membership must stay intact")
        self.assertEqual(catalog.read_bytes(), original_catalog)
        self.assertFalse(catalog.with_suffix(".json.tmp").exists())
        self.assertEqual([path.name for path in industry.iterdir()], [FARM + ".txt"])

    def test_industry_names_json_is_sorted_validated_source_data(self):
        self.build([(CHEM, "20240109", None, "Y"),
                    (FARM, "20240105", "20240108", "N")])
        path = self.source / "industry_names.json"
        catalog = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(catalog, {"schema_version": 1,
            "source": {"kind": "index_classify", "level": "L1", "sources": ["SW2014", "SW2021"]},
            "names": {FARM: "农林牧渔", CHEM: "基础化工"}})
        self.assertEqual(list(catalog["names"]), [FARM, CHEM])
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        csv = pd.read_csv(self.source / "industry/names.csv")
        self.assertEqual(csv.to_dict("records"), [{"index_code": FARM, "industry_name": "农林牧渔"},
                                                 {"index_code": CHEM, "industry_name": "基础化工"}])

    def test_missing_labels_write_empty_json_without_using_or_changing_legacy_names(self):
        self.write_cache([(FARM, "20240105", None, "Y")])
        cache_file = self.cache / "industry.csv"
        pd.read_csv(cache_file).drop(columns="l1_name").to_csv(cache_file, index=False)
        legacy = self.source / "industry/names.csv"
        legacy.parent.mkdir()
        legacy.write_bytes(b"index_code,industry_name\n801010.SI,legacy label\n")
        original_legacy = legacy.read_bytes()
        source_cache = cache_file.read_bytes()
        build_industry(CsvClient(self.cache), self.source, self.basic, self.dates)
        catalog = json.loads((self.source / "industry_names.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(catalog["names"], {})
        self.assertEqual(legacy.read_bytes(), original_legacy)
        self.assertEqual(cache_file.read_bytes(), source_cache)

    def test_conflicting_labels_fail_before_old_catalog_or_memberships_change(self):
        self.write_cache([(FARM, "20240105", "20240108", "N"),
                          (FARM, "20240109", None, "Y")])
        cache_file = self.cache / "industry.csv"
        frame = pd.read_csv(cache_file)
        frame.loc[1, "l1_name"] = "conflicting label"
        frame.to_csv(cache_file, index=False)
        industry = self.source / "industry"
        industry.mkdir()
        previous = industry / (FARM + ".txt")
        previous.write_bytes(b"original membership")
        catalog = self.source / "industry_names.json"
        catalog.write_bytes(b'{"schema_version":1,"names":{}}\n')
        original = {path: path.read_bytes() for path in (previous, catalog, cache_file)}
        with self.assertRaisesRegex(ValueError, "Conflicting industry names"):
            build_industry(CsvClient(self.cache), self.source, self.basic, self.dates)
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        self.assertEqual([path.name for path in industry.iterdir()], [FARM + ".txt"])

    def test_same_l1_overlapping_and_adjacent_intervals_merge(self):
        provider = self.build([(FARM, "20240105", "20240109", "N"),
                               (FARM, "20240108", "20240111", "N"),
                               (FARM, "20240112", None, "Y")])
        self.assertTrue(provider.universe("industry/" + FARM)[A].all())
        lines = (self.source / "industry" / (FARM + ".txt")).read_text().splitlines()
        self.assertEqual(lines, [f"{A}\t2024-01-05\t2099-12-31"])

    def test_real_long_gap_is_preserved(self):
        provider = self.build([(FARM, "20240105", "20240108", "N"),
                               (CHEM, "20240115", None, "Y")])
        covered = provider.universe("industry/" + FARM)[A] | provider.universe("industry/" + CHEM)[A]
        self.assertEqual(covered.tolist(), [True, True, False, False, False, False, True, True])

    def test_nontrading_day_intersection_does_not_create_trading_day_overlap(self):
        provider = self.build([(FARM, "20240105", "20240107", "N"),
                               (CHEM, "20240106", None, "Y")])
        farm = provider.universe("industry/" + FARM)[A]
        chemical = provider.universe("industry/" + CHEM)[A]
        self.assertEqual(farm.tolist(), [True] + [False] * 7)
        self.assertEqual(chemical.tolist(), [False] + [True] * 7)
        self.assertFalse((farm & chemical).any())

    def test_interval_sweep_matches_independent_datewise_conflict_oracle(self):
        rng = np.random.RandomState(81)
        for _ in range(60):
            rows = {FARM: [], CHEM: []}
            covered = {industry: np.zeros(len(self.dates), dtype=bool) for industry in rows}
            for _ in range(8):
                industry = FARM if rng.randint(2) else CHEM
                start, end = sorted(rng.choice(len(self.dates), size=2, replace=False))
                rows[industry].append((A, self.dates[start], self.dates[end]))
                covered[industry] |= (self.dates >= self.dates[start]) & (self.dates <= self.dates[end])
            conflict = bool(np.any(covered[FARM] & covered[CHEM]))
            if conflict:
                with self.assertRaisesRegex(ValueError, "Cross-industry membership overlap"):
                    _validate_industry_intervals(rows, self.dates)
            else:
                _validate_industry_intervals(rows, self.dates)

    def test_prepare_cli_uses_default_csv_cache_and_never_publishes(self):
        self.write_cache([(FARM, "20240105", "20240108", "N"),
                          (CHEM, "20240109", None, "Y")])
        (self.source / "industry_names.json").write_text(
            json.dumps({"schema_version": 1, "names": {FARM: "existing dataset label"}}), encoding="utf-8")
        original = {path: path.read_bytes() for root in (self.source, self.cache) for path in root.rglob("*") if path.is_file()}
        args = ["rebuild_industry.py", "--source-root", str(self.source), "--output-root", str(self.stage)]
        with patch.object(C, "CACHE_DIR", str(self.cache)), patch("sys.argv", args):
            main()
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        self.assertFalse((self.source / "industry").exists())
        self.assertFalse((self.stage / "stock_basic.csv").exists())
        receipt = json.loads((self.stage / "rebuild_industry.json").read_text())
        self.assertTrue(receipt["prepared"])
        self.assertEqual(receipt["cross_industry_trading_day_overlaps"], 0)
        self.assertEqual(receipt["industry_codes"], [FARM, CHEM])
        self.assertIn("industry_names.json", receipt["files_sha256"])
        self.assertEqual(json.loads((self.stage / "industry_names.json").read_text(encoding="utf-8"))["names"],
                         {FARM: "农林牧渔", CHEM: "基础化工"})
        for name, sha in receipt["files_sha256"].items():
            self.assertEqual(hashlib.sha256((self.stage / name).read_bytes()).hexdigest(), sha)
        for output in (self.source, self.source / "industry-stage", self.cache, self.stage):
            with self.subTest(output=output), self.assertRaises(ValueError):
                prepare_industry(self.source, output, cache_root=self.cache)


if __name__ == "__main__":
    unittest.main()
