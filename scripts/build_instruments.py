"""Rebuild instrument pools offline, preserving daily/PIT data and a pool backup."""

import argparse
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
from loguru import logger

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlib.log import summarize_warnings
from scripts import config as C
from scripts._staging import staging_directory
from scripts.dump.bin import build_all, build_indices
from scripts.tushare.data import CsvClient


def _staging_directory(root):
    return staging_directory(root, prefix=".instruments-build-")


@summarize_warnings()
def build_instruments(provider_uri=C.OUTPUT_DIR, cache_uri=C.CACHE_DIR):
    """Rebuild all/index pools through the existing calendar's last trading day.

    Other pool files are retained. Both generation steps finish in staging
    before publication; the previous instruments directory remains as a backup.
    Source CSVs, listing metadata, daily data, financial data and calendars are
    never updated by this offline operation. Returns the absolute backup path.
    """
    root = Path(provider_uri).expanduser().resolve()
    cache = Path(cache_uri).expanduser().resolve()
    destination = root / "instruments"
    if (not destination.is_dir() or destination.is_symlink()
            or destination.resolve().parent != root):
        raise ValueError(f"缺少股票池目录或目录是链接：{destination}")
    basic = pd.read_csv(root / "stock_basic.csv", dtype={
        "ts_code": str, "list_date": str, "delist_date": str,
    })
    if (not basic.ts_code.str.fullmatch(r"\d{6}\.(SH|SZ)", na=False).all()
            or basic.ts_code.duplicated().any()):
        raise ValueError("stock_basic 包含无效代码或重复股票")
    for field in ("list_date", "delist_date"):
        if field in basic:
            values = basic[field].astype("string").str.replace(r"\.0$", "", regex=True)
            basic[field] = pd.to_datetime(values, format="mixed")
    calendar = pd.DatetimeIndex(pd.read_csv(root / "calendars/day.txt", header=None)[0])
    if calendar.empty or calendar.hasnans or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("交易日历必须非空、唯一且升序")
    logger.info("股票池重建开始：{}，截止日={:%Y-%m-%d}，离线缓存={}", root, calendar[-1], cache)
    # All publication and cleanup targets are confined to the resolved dataset
    # root. Retain the old pool directory rather than deleting it after publish.
    with _staging_directory(root) as stage:
        shutil.copytree(destination, stage / "instruments", symlinks=True)
        for name in ("all", *C.INDEX_CODES):
            target = stage / "instruments" / f"{name}.txt"
            if target.is_symlink() or target.resolve().parent != stage / "instruments":
                raise ValueError(f"待重建的股票池文件是链接或超出暂存目录：{target}")
        build_all(stage, basic, calendar)
        build_indices(CsvClient(cache), stage, set(basic.ts_code), calendar)
        backup = root / f"instruments.backup-{time.time_ns()}"
        if backup.parent != root or backup.exists():
            raise ValueError(f"无效股票池备份目录：{backup}")
        destination.rename(backup)
        try:
            (stage / "instruments").rename(destination)
        except OSError:
            backup.rename(destination)
            raise
    logger.info("股票池重建完成：{}，原股票池备份={}", destination, backup)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default=C.OUTPUT_DIR)
    parser.add_argument("--cache-uri", default=C.CACHE_DIR)
    args = parser.parse_args()
    build_instruments(args.provider_uri, args.cache_uri)


if __name__ == "__main__":
    main()
