"""
数据构建入口；配置位于 scripts/config.py。
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

# 兼容直接运行本文件及 python -m scripts.build_data。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlib.log import summarize_warnings
from scripts import config as C
from scripts.dump.bin import (
    DAILY_FIELDS,
    build_daily,
    build_indices,
    build_industry,
    build_stock_basic,
    iso_date,
    write_lines,
)
from scripts.tushare.data import CsvClient, _validate_index_codes, download_data, fetch_calendar
from scripts.dump.pit import build_financial


def _save_state(path, state):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


@summarize_warnings()
def build_data(download=True, resume_dir=None, *, keep_backup=False,
               refresh_index_codes=(), publish=True):
    """Build a complete dataset, optionally preparing it for inspection first.

    ``publish=False`` keeps the completed snapshot and build state in the work
    directory. Resuming that snapshot publishes the same bytes without another
    download or rebuild. ``keep_backup=True`` retains the previous dataset.
    Returns the prepared data root or the published output directory.
    """
    refresh_index_codes = _validate_index_codes(refresh_index_codes)
    output = Path(C.OUTPUT_DIR).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(C.CACHE_DIR).expanduser().resolve()
    work = (
        Path(resume_dir).expanduser().resolve()
        if resume_dir
        else output.with_name(f".{output.name}.build")
    )
    if any(
        work.is_relative_to(path) or path.is_relative_to(work)
        for path in (output, cache_dir)
    ):
        raise ValueError("构建工作目录必须与输出、CSV 缓存目录分开")
    state_path = work / "build.json"
    identity = {
        "cache_dir": str(cache_dir),
        "output_dir": str(output),
        "start_date": C.START_DATE,
        "daily_fields": list(DAILY_FIELDS),
        "pit_fields": {key: list(value) for key, value in C.PIT_FIELDS.items()} if C.BUILD_PIT else None,
        "refresh_index_codes": list(refresh_index_codes),
        "refresh_cache": C.REFRESH_CACHE,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        defaults = {"refresh_index_codes": [], "refresh_cache": False}
        if any(state.get(key, defaults.get(key)) != value for key, value in identity.items()):
            raise ValueError(f"构建目录的配置已改变，请使用新的 resume_dir：{work}")
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError(f"构建目录非空且缺少 build.json：{work}")
        work.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
        state = {
            **identity,
            "today": iso_date(today),
            "cache_ready": False,
            "daily_done": False,
            "prepared": False,
        }
        _save_state(state_path, state)
    today = pd.Timestamp(state["today"])
    logger.info("数据构建开始：{}，截止日={}，断点目录={}", output, iso_date(today), work)
    root = work / "cn_data"
    if root.resolve().parent != work:
        raise ValueError("数据暂存目录超出构建工作目录")
    if state.get("prepared", False):
        calendar = pd.DatetimeIndex(pd.read_csv(root / "calendars/day.txt", header=None)[0])
        logger.info("复用已准备的完整数据快照：{}，{} 个交易日", root, len(calendar))
    else:
        client = CsvClient(cache_dir)
        if download and not state["cache_ready"]:
            client = download_data(cache_dir, today, refresh_index_codes=refresh_index_codes)
        else:
            if refresh_index_codes and not state["cache_ready"]:
                raise ValueError("定向指数刷新需要下载；请先移除 --no-download")
            logger.info("复用本地 CSV 缓存：{}", cache_dir)
        state["cache_ready"] = True
        _save_state(state_path, state)

        root.mkdir(exist_ok=True)
        basic = build_stock_basic(client, root)
        if not state["daily_done"]:
            calendar, future = fetch_calendar(client, today)
            calendar = build_daily(client, root, basic, calendar, today)
            write_lines(root / "calendars" / "day.txt", map(iso_date, calendar))
            write_lines(
                root / "calendars" / "day_future.txt", map(iso_date, future.union(calendar))
            )
            state["daily_done"] = True
            _save_state(state_path, state)
        else:
            calendar = pd.DatetimeIndex(
                pd.read_csv(root / "calendars/day.txt", header=None)[0]
            )
            logger.info("复用已完成日线：{} 个交易日", len(calendar))
        codes = set(basic.ts_code)
        build_indices(client, root, codes, calendar)
        build_industry(client, root, basic, calendar)
        if C.BUILD_PIT:
            # Carry field IDs into staging; the builder retains only selected names.
            dictionary = output / "financial" / "fields.json"
            target = root / "financial" / "fields.json"
            if dictionary.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dictionary, target)
            build_financial(client, root, codes, today)
        state["prepared"] = True
        _save_state(state_path, state)

    if not publish:
        logger.info("完整数据已准备，等待校验后发布：{}，日线截止 {}", root, iso_date(calendar[-1]))
        return root

    backup = None
    if output.exists():
        backup = output.with_name(f"{output.name}.backup-{time.time_ns()}")
        output.rename(backup)
    try:
        root.rename(output)
    except OSError:
        if backup is not None:
            backup.rename(output)
        raise
    if backup is not None and not keep_backup:
        shutil.rmtree(backup)
    shutil.rmtree(work)  # 仅清理本次工作目录；CSV 缓存始终保留。
    logger.info("数据构建完成：{}，日线截止 {}", output, iso_date(calendar[-1]))
    if backup is not None and keep_backup:
        logger.info("原数据备份保留：{}", backup)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-dir", help="指定构建工作目录；恢复时须使用相同下载刷新配置")
    parser.add_argument("--no-download", action="store_true", help="只使用已有 CSV 或已准备的快照")
    parser.add_argument("--keep-backup", action="store_true", help="发布后保留原数据目录备份")
    parser.add_argument("--prepare-only", action="store_true", help="完成暂存构建，校验前不替换目标数据")
    parser.add_argument("--refresh-csi1000", action="store_true", help="定向刷新中证1000全部历史月缓存")
    args = parser.parse_args()
    build_data(download=not args.no_download, resume_dir=args.resume_dir,
               keep_backup=args.keep_backup,
               refresh_index_codes=("000852.SH",) if args.refresh_csi1000 else (),
               publish=not args.prepare_only)


if __name__ == "__main__":
    main()
