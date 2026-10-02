"""Build PIT v2 independently of daily data; offline CSV mode is the default."""

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
from scripts.dump.pit import build_financial
from scripts.tushare.data import CsvClient, TushareClient, download_financial_cache


@summarize_warnings()
def build_pit(provider_uri=C.OUTPUT_DIR, cache_uri=C.CACHE_DIR, *, today=None, download=False):
    root, cache = Path(provider_uri).expanduser().resolve(), Path(cache_uri).expanduser().resolve()
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    codes = set(pd.read_csv(root / "stock_basic.csv", dtype={"ts_code": str}).ts_code)
    logger.info("PIT 构建开始：{}，截止日={:%Y-%m-%d}，下载={}", root, today, download)
    if download:
        if not C.TOKEN.strip():
            raise ValueError("请设置 TUSHARE_TOKEN")
        import tushare as ts
        cache.mkdir(parents=True, exist_ok=True)
        download_financial_cache(TushareClient(ts.pro_api(C.TOKEN)), cache, today)
    destination = root / "financial"
    with staging_directory(root, prefix=".pit-build-") as stage:
        if (destination / "fields.json").exists():
            (stage / "financial").mkdir()
            shutil.copyfile(destination / "fields.json", stage / "financial" / "fields.json")
        build_financial(CsvClient(cache), stage, codes, today)
        backup = root / f".financial.backup-{time.time_ns()}"
        if destination.exists():
            destination.rename(backup)
        try:
            (stage / "financial").rename(destination)
        except OSError:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    logger.info("PIT v2 构建完成：{}", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default=C.OUTPUT_DIR)
    parser.add_argument("--cache-uri", default=C.CACHE_DIR)
    parser.add_argument("--end", help="Build through YYYY-MM-DD (default: today)")
    parser.add_argument("--download", action="store_true", help="Refresh selected financial CSVs before building")
    args = parser.parse_args()
    build_pit(args.provider_uri, args.cache_uri, today=args.end, download=args.download)


if __name__ == "__main__":
    main()
