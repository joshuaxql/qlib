"""Prepare corrected historical industry files for inspection; never publish.

The CSV cache and the existing dataset's calendar/stock_basic are read-only.
Only an explicitly selected, empty output directory receives staged files.
The stage includes memberships and industry_names.json at its data root.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from qlib.data import LocalProvider
from scripts import config as C
from scripts.dump.bin import build_industry
from scripts.tushare.data import CsvClient


def prepare_industry(source_root, output_root, *, cache_root=None):
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    cache = Path(C.CACHE_DIR if cache_root is None else cache_root).expanduser().resolve()
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (source, cache)):
        raise ValueError("Industry stage must be separate from the source dataset and CSV cache")
    if output.is_relative_to((Path.home() / ".qlib").resolve()):
        raise ValueError("Industry preparation must not write into ~/.qlib")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Industry stage must be an empty directory: {output}")
    provider = LocalProvider(source)
    calendar, basic = provider.calendar(), provider.stock_basic()
    source_paths = (source / "calendars/day.txt", source / "stock_basic.csv", cache / "industry.csv")
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    output.mkdir(parents=True, exist_ok=True)
    build_industry(CsvClient(cache), output, basic, calendar)
    files = sorted([path for path in (output / "industry").iterdir() if path.is_file()] +
                   [output / "industry_names.json"])
    receipt = {
        "prepared": True, "source_root": str(source), "cache_root": str(cache), "output_root": str(output),
        "out_date_semantics": "last membership date, inclusive",
        "cross_industry_trading_day_overlaps": 0, "trading_days": len(calendar),
        "industry_codes": sorted(path.stem for path in (output / "industry").glob("*.txt")),
        "source_sha256": source_hashes,
        "files_sha256": {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in files},
    }
    path = output / "rebuild_industry.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    logger.info("行业暂存完成：{}，{} 个行业；请审阅后再发布", output, len(receipt["industry_codes"]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(C.OUTPUT_DIR), help="Existing dataset, read-only")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Empty stage directory for industry memberships and industry_names.json")
    parser.add_argument("--cache-root", type=Path, default=None, help="Read-only CSV cache; defaults to scripts.config.CACHE_DIR")
    args = parser.parse_args()
    prepare_industry(args.source_root, args.output_root, cache_root=args.cache_root)


if __name__ == "__main__":
    main()
