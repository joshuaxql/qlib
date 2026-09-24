"""Download or backfill daily up_limit/down_limit binaries in an existing dataset."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import config as C
from scripts.dump.bin import prepare_limits
from scripts.tushare.data import CsvClient, TushareClient, download_limit_cache


def build_limits(provider_uri=C.OUTPUT_DIR, cache_uri=C.CACHE_DIR, *, download=False):
    """Stage both fields, align to existing daily spans, then atomically replace each file.

    The dataset calendar bounds downloads. Missing source rows and source prices
    that are null, nonfinite or nonpositive remain NaN. No forward filling or
    inferred percentage bounds are written. Reruns safely complete publication.
    """
    root = Path(provider_uri).expanduser().resolve()
    cache = Path(cache_uri).expanduser().resolve()
    calendar = pd.DatetimeIndex(pd.read_csv(root / "calendars/day.txt", header=None)[0])
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("日历必须非空、唯一并按日期递增")
    basic = pd.read_csv(root / "stock_basic.csv", dtype={"ts_code": str})
    spans = {}
    for code in sorted(set(basic.ts_code)):
        close = root / "features" / code / "close.day.bin"
        if not close.exists():
            continue
        values = np.fromfile(close, dtype="<f4")
        if len(values) < 2 or close.stat().st_size % 4 or not np.isfinite(values[0]):
            raise ValueError(f"无效日线文件：{close}")
        left = int(values[0])
        right = left + len(values) - 1
        if left != values[0] or left < 0 or right > len(calendar):
            raise ValueError(f"日线日历范围无效：{close}")
        spans[code] = (left, right)
    if not spans:
        raise ValueError("没有可补充涨跌停价格的日线股票")
    if download:
        if not C.TOKEN.strip():
            raise ValueError("请设置 TUSHARE_TOKEN")
        import tushare as ts
        cache.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
        download_limit_cache(TushareClient(ts.pro_api(C.TOKEN)), cache, calendar, today)
    symbols = pd.Index(spans)
    reader = CsvClient(cache)
    total, covered = 0, 0
    with TemporaryDirectory(prefix=".limit-build-", dir=root.parent) as temporary:
        stage = Path(temporary)
        matrix = np.memmap(stage / "limits.f32", mode="w+", dtype="<f4", shape=(len(calendar), len(symbols), 2))
        try:
            for position, date in enumerate(tqdm(calendar, desc="对齐每日涨跌停价格")):
                day = date.strftime("%Y%m%d")
                daily = reader.fetch("daily", trade_date=day, fields="ts_code,trade_date")
                daily = daily[daily.ts_code.isin(symbols)]
                if daily.ts_code.duplicated().any() or not daily.trade_date.eq(day).all():
                    raise ValueError(f"{day} 日线含重复股票或其他日期记录")
                limits = prepare_limits(reader.fetch("stk_limit", trade_date=day), day)
                merged = daily.merge(limits, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
                values = merged[["up_limit", "down_limit"]].to_numpy(dtype="<f4")
                matrix[position] = np.nan
                matrix[position, symbols.get_indexer(merged.ts_code)] = values
                total += len(merged)
                covered += int(np.isfinite(values).all(axis=1).sum())
            for column, code in enumerate(tqdm(symbols, desc="暂存涨跌停二进制文件")):
                left, right = spans[code]
                directory = stage / code
                directory.mkdir()
                for index, field in enumerate(("up_limit", "down_limit")):
                    np.r_[left, matrix[left:right, column, index]].astype("<f4").tofile(directory / f"{field}.day.bin")
        finally:
            del matrix
        for code in symbols:
            for field in ("up_limit", "down_limit"):
                (stage / code / f"{field}.day.bin").replace(root / "features" / code / f"{field}.day.bin")
    result = {"stocks": len(symbols), "trading_days": len(calendar), "daily_rows": total,
              "rows_with_both_limits": covered, "rows_missing_limits": total - covered}
    print(f"涨跌停价格构建完成：{root}，{result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default=C.OUTPUT_DIR)
    parser.add_argument("--cache-uri", default=C.CACHE_DIR)
    parser.add_argument("--download", action="store_true", help="Download/refresh stk_limit before building")
    args = parser.parse_args()
    build_limits(args.provider_uri, args.cache_uri, download=args.download)


if __name__ == "__main__":
    main()
