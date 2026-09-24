"""
数据构建入口；配置位于 scripts/config.py。
"""

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 兼容直接运行本文件及 python -m scripts.build_data。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from scripts.tushare.data import CsvClient, download_data, fetch_calendar
from scripts.dump.pit import build_financial


def _save_state(path, state):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def build_data(download=True, resume_dir=None):
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
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
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
        }
        _save_state(state_path, state)
    today = pd.Timestamp(state["today"])
    client = CsvClient(cache_dir)
    if download and not state["cache_ready"]:
        client = download_data(cache_dir, today)
    else:
        tqdm.write(f"离线构建：{cache_dir} -> {output}；截止日：{iso_date(today)}")
    state["cache_ready"] = True
    _save_state(state_path, state)
    tqdm.write(f"构建断点：{work}")

    root = work / "cn_data"
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
        tqdm.write(f"复用已完成日线：{len(calendar)} 个交易日")
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
    if backup is not None:
        shutil.rmtree(backup)
    shutil.rmtree(work)  # 仅清理本次工作目录；CSV 缓存始终保留。
    tqdm.write(f"构建完成：{output}，日线截止 {iso_date(calendar[-1])}")


if __name__ == "__main__":
    build_data()
