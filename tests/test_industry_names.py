"""Classification labels remain separate from historical stock membership."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from qlib.data import LocalProvider
from scripts.dump.bin import build_industry
from scripts.tushare.data import CsvClient

A, B = "000001.SZ", "600000.SH"
FARM, CHEM, UNKNOWN = "801010.SI", "801030.SI", "899999.SI"


class IndustryNamesTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.cache = self.base / "built", self.base / "cache"
        self.root.mkdir()
        self.cache.mkdir()
        (self.root / "calendars").mkdir()
        self.dates = pd.bdate_range("2024-01-02", periods=5, name="datetime")
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        self.basic = pd.DataFrame({"ts_code": [A, B], "list_date": [self.dates[0]] * 2,
                                   "delist_date": [pd.NaT] * 2, "industry": ["当前股票快照名称"] * 2})
        self.basic.to_csv(self.root / "stock_basic.csv", index=False)
        self.members = pd.DataFrame([
            (FARM, "农林牧渔", A, "20240102", "20240103", "N"),
            (CHEM, "基础化工", A, "20240104", None, "Y"),
            (FARM, "农林牧渔", B, "20240102", None, "Y"),
        ], columns=["l1_code", "l1_name", "ts_code", "in_date", "out_date", "is_new"])
        self.members.to_csv(self.cache / "industry.csv", index=False)

    def provider(self):
        return LocalProvider(self.root)

    def write_json(self, names, *, root=None, encoding="utf-8"):
        root = self.root if root is None else root
        (root / "industry_names.json").write_text(json.dumps({"schema_version": 1, "names": names},
                                                             ensure_ascii=False), encoding=encoding)

    def write_membership(self, root=None, codes=(FARM, CHEM, UNKNOWN)):
        root = self.root if root is None else root
        (root / "industry").mkdir(exist_ok=True)
        for code in codes:
            (root / "industry" / (code + ".txt")).write_text(f"{A} 2024-01-02 2099-12-31\n")

    def test_csv_client_retains_classification_names_and_requested_fields(self):
        client = CsvClient(self.cache)
        expected = pd.DataFrame({"index_code": [FARM, CHEM], "industry_name": ["农林牧渔", "基础化工"]})
        for source in ("SW2014", "SW2021"):
            pd.testing.assert_frame_equal(client.fetch("index_classify", level="L1", src=source).reset_index(drop=True),
                                          expected)
        pd.testing.assert_frame_equal(client.fetch("index_classify", fields="index_code,industry_name").reset_index(drop=True),
                                      expected)
        self.assertEqual(client.fetch("index_classify", fields="index_code").columns.tolist(), ["index_code"])

    def test_builder_writes_catalog_without_changing_historical_membership(self):
        build_industry(CsvClient(self.cache), self.root, self.basic, self.dates)
        provider = self.provider()
        self.assertEqual(provider.industries(), [FARM, CHEM])
        self.assertEqual(provider.industry_names(), {FARM: "农林牧渔", CHEM: "基础化工"})
        catalog = pd.read_csv(self.root / "industry/names.csv", dtype=str)
        pd.testing.assert_frame_equal(catalog, pd.DataFrame({"index_code": [FARM, CHEM],
                                                            "industry_name": ["农林牧渔", "基础化工"]}))
        metadata = json.loads((self.root / "industry_names.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata, {"schema_version": 1,
                                   "source": {"kind": "index_classify", "level": "L1",
                                              "sources": ["SW2014", "SW2021"]},
                                   "names": {FARM: "农林牧渔", CHEM: "基础化工"}})
        farm = provider.universe("industry/" + FARM)
        chemical = provider.universe("industry/" + CHEM)
        self.assertEqual(farm[A].tolist(), [True, True, False, False, False])
        self.assertEqual(chemical[A].tolist(), [False, False, True, True, True])
        self.assertEqual(farm[B].tolist(), [True] * 5)
        # JSON is authoritative; stale legacy labels never override it.
        pd.DataFrame({"index_code": [FARM], "industry_name": ["旧CSV中文标签"]}).to_csv(
            self.root / "industry/names.csv", index=False)
        self.assertEqual(provider.industry_names(), {FARM: "农林牧渔", CHEM: "基础化工"})
        metadata["names"][FARM] = "数据目录JSON标签"
        (self.root / "industry_names.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(provider.industry_names(), {FARM: "数据目录JSON标签", CHEM: "基础化工"})
        pd.testing.assert_frame_equal(provider.universe("industry/" + FARM), farm)
        pd.testing.assert_frame_equal(provider.universe("industry/" + CHEM), chemical)

    def test_missing_catalog_returns_codes_and_json_only_exposes_available_industries(self):
        self.write_membership()
        provider = self.provider()
        self.assertEqual(provider.industries(), [FARM, CHEM, UNKNOWN])
        self.assertEqual(provider.industry_names(), {FARM: FARM, CHEM: CHEM, UNKNOWN: UNKNOWN})
        self.write_json({FARM.lower(): "数据源标签", "800000.SI": "没有成员文件的分类"}, encoding="utf-8-sig")
        self.assertEqual(provider.industry_names(), {FARM: "数据源标签", CHEM: CHEM, UNKNOWN: UNKNOWN})

    def test_builder_code_only_stub_keeps_compatibility(self):
        parent = self

        class CodeOnlyClient:
            def fetch(client, api, **params):
                if api == "index_classify":
                    return pd.DataFrame({"index_code": [FARM, CHEM]})
                return parent.members[(parent.members.l1_code == params["l1_code"]) &
                                      (parent.members.is_new == params["is_new"])].copy()

        build_industry(CodeOnlyClient(), self.root, self.basic, self.dates)
        self.assertFalse((self.root / "industry/names.csv").exists())
        self.assertEqual(json.loads((self.root / "industry_names.json").read_text(encoding="utf-8"))["names"], {})
        self.assertEqual(self.provider().industries(), [FARM, CHEM])
        self.assertEqual(self.provider().industry_names(), {FARM: FARM, CHEM: CHEM})

    def test_distinct_provider_roots_do_not_share_catalogs_or_global_home_labels(self):
        self.write_membership()
        other = self.base / "other"
        (other / "calendars").mkdir(parents=True)
        (other / "calendars/day.txt").write_bytes((self.root / "calendars/day.txt").read_bytes())
        self.write_membership(other)
        self.write_json({FARM: "数据源甲"})
        self.write_json({FARM: "数据源乙", CHEM: "乙分类"}, root=other)
        first, second = self.provider(), LocalProvider(other)
        self.assertEqual(first.industry_names()[FARM], "数据源甲")
        self.assertEqual(second.industry_names()[FARM], "数据源乙")
        self.write_json({FARM: "即时修改甲"})
        self.assertEqual(first.industry_names()[FARM], "即时修改甲")
        self.assertEqual(second.industry_names()[FARM], "数据源乙")
        (self.root / "industry_names.json").unlink()
        global_home = self.base / "global_home"
        global_data = global_home / ".qlib/qlib_data/cn_data"
        global_data.mkdir(parents=True)
        self.write_json({FARM: "其他默认数据源"}, root=global_data)
        with patch("pathlib.Path.home", return_value=global_home):
            self.assertEqual(first.industry_names(), {FARM: FARM, CHEM: CHEM, UNKNOWN: UNKNOWN})

    def test_json_is_authoritative_even_when_partial_empty_or_legacy_is_malformed(self):
        self.write_membership()
        legacy = self.root / "industry/names.csv"
        legacy.write_text(f"index_code,industry_name\n{FARM},旧农林\n{CHEM},旧化工\n", encoding="utf-8")
        self.write_json({FARM: "JSON农林"})
        provider = self.provider()
        self.assertEqual(provider.industry_names(), {FARM: "JSON农林", CHEM: CHEM, UNKNOWN: UNKNOWN})
        legacy.write_text("wrong_column\nbad\n", encoding="utf-8")
        self.assertEqual(provider.industry_names()[FARM], "JSON农林")
        self.write_json({})
        self.assertEqual(provider.industry_names(), {FARM: FARM, CHEM: CHEM, UNKNOWN: UNKNOWN})
        (self.root / "industry_names.json").unlink()
        with self.assertRaisesRegex(ValueError, "Invalid industry name catalog"):
            provider.industry_names()

    def test_malformed_json_schema_and_label_types_never_fall_back_to_legacy(self):
        self.write_membership()
        (self.root / "industry/names.csv").write_text(f"index_code,industry_name\n{FARM},有效旧标签\n", encoding="utf-8")
        path = self.root / "industry_names.json"
        malformed = [[], {}, {"schema_version": 1},
                     *({"schema_version": version, "names": {}} for version in (True, "1", 1., 2)),
                     *({"schema_version": 1, "names": names} for names in (None, [], "names")),
                     *({"schema_version": 1, "names": {FARM: name}} for name in (None, 1, True, [], {}, "", " \t ")),
                     {"schema_version": 1, "names": {"../unsafe": "名称"}},
                     {"schema_version": 1, "names": {"８０１０１０.SI": "名称"}},
                     {"schema_version": 1, "names": {}, "source": "wrong type"}]
        provider = self.provider()
        for catalog in malformed:
            with self.subTest(catalog=catalog):
                path.write_text(json.dumps(catalog), encoding="utf-8")
                with self.assertRaises(ValueError):
                    provider.industry_names()
        for text in ('{"schema_version":1,"names":', '{"schema_version":1,"names":{},"unused":NaN}'):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid industry name catalog"):
                    provider.industry_names()

    def test_normalized_code_conflicts_and_duplicate_json_keys_are_rejected(self):
        self.write_membership()
        provider = self.provider()
        self.write_json({FARM.lower(): "名称甲", FARM: "名称乙"})
        with self.assertRaisesRegex(ValueError, "Conflicting industry names.*801010"):
            provider.industry_names()
        self.write_json({FARM.lower(): "同一名称", FARM: "同一名称"})
        self.assertEqual(provider.industry_names()[FARM], "同一名称")
        path = self.root / "industry_names.json"
        for text in (f'{{"schema_version":1,"names":{{"{FARM}":"甲","{FARM}":"乙"}}}}',
                     '{"schema_version":1,"schema_version":2,"names":{}}'):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Duplicate industry name catalog key"):
                    provider.industry_names()

    def test_legacy_csv_is_reread_and_rejects_empty_names_conflicts_and_bad_rows(self):
        self.write_membership()
        path = self.root / "industry/names.csv"
        provider = self.provider()
        for name in ("旧目录标签", "即时重读标签"):
            path.write_text(f"index_code,industry_name\n{FARM.lower()},{name}\n", encoding="utf-8-sig")
            self.assertEqual(provider.industry_names(), {FARM: name, CHEM: CHEM, UNKNOWN: UNKNOWN})
        malformed = [f"index_code,industry_name\n{FARM},\n",
                     f"index_code,industry_name\n{FARM}, \t \n",
                     f"index_code,industry_name\n{FARM.lower()},甲\n{FARM},乙\n",
                     "index_code,industry_name\n../bad,标签\n",
                     f"index_code,industry_name\n{FARM},标签,多余值\n",
                     f"index_code,industry_name\n{FARM}\n",
                     f'index_code,industry_name\n{FARM},"未闭合引号\n',
                     "index_code,index_code,industry_name\n", ""]
        for content in malformed:
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    provider.industry_names()

    def test_conflicting_names_are_rejected_before_building_membership(self):
        self.members.loc[2, "l1_name"] = "另一个分类名称"
        self.members.to_csv(self.cache / "industry.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Conflicting industry names.*801010"):
            build_industry(CsvClient(self.cache), self.root, self.basic, self.dates)
        self.assertFalse((self.root / "industry").exists())
        industry = self.root / "industry"
        industry.mkdir()
        (industry / (FARM + ".txt")).write_text("")
        pd.DataFrame({"index_code": [FARM, FARM], "industry_name": ["名称甲", "名称乙"]}).to_csv(
            industry / "names.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Conflicting industry names"):
            self.provider().industry_names()
        (industry / "names.csv").write_text("wrong_column\nvalue\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid industry name catalog"):
            self.provider().industry_names()


if __name__ == "__main__":
    unittest.main()
