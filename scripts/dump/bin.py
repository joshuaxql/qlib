"""CSV -> 本地日线、股票池、行业和基础信息。

日线为小端 float32，首项为日历偏移。
价格不复权，成交量为手，市值为万元；股票目录保留大写代码并兼容小写路径。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts import config as C
from scripts.tushare.data import index_month_ranges

DAILY_FIELDS = (
    "open", "high", "low", "close", "vwap", "volume", "total_mv", "circ_mv", "factor",
    "up_limit", "down_limit",
)


def iso_date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for line in lines:
            stream.write(line + "\n")


def stock_dir(root, section, code):
    directory = root / section / code
    directory.mkdir(parents=True, exist_ok=True)
    alias = directory.with_name(code.lower())
    if not alias.exists():
        alias.symlink_to(directory.name, target_is_directory=True)
    return directory


def build_stock_basic(client, root):
    frames = []
    for status in tqdm(("L", "D", "P"), desc="股票基础信息"):
        frame = client.fetch(
            "stock_basic", list_status=status,
            fields="ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,"
                   "exchange,curr_type,list_status,list_date,delist_date,is_hs",
        )
        if status == "L" and frame.empty:
            raise ValueError("stock_basic 上市股票列表为空")
        frames.append(frame)
    basic = pd.concat(frames, ignore_index=True)
    basic = basic[basic.ts_code.str.fullmatch(r"\d{6}\.(SH|SZ)", na=False)]
    basic = basic.drop_duplicates("ts_code").sort_values("ts_code").copy()
    for field in ("list_date", "delist_date"):
        basic[field] = pd.to_datetime(basic[field].replace("", None), format="%Y%m%d")
    if basic.list_date.isna().any():
        raise ValueError("stock_basic 缺少上市日期")
    basic.to_csv(root / "stock_basic.csv", index=False, date_format="%Y-%m-%d", encoding="utf-8")
    return basic


def prepare_limits(frame, day=None):
    """Validate source keys and preserve missing/nonpositive limit prices as NaN."""
    keys = ["ts_code", "trade_date"]
    fields = ["up_limit", "down_limit"]
    if frame.empty:
        return pd.DataFrame(columns=keys + fields)
    if not set(keys + fields).issubset(frame):
        raise ValueError("stk_limit 缺少代码、日期或涨跌停价格字段")
    frame = frame[keys + fields].copy()
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("stk_limit 包含空键或重复股票日期")
    if day is not None and not frame.trade_date.eq(day).all():
        raise ValueError(f"{day} stk_limit 包含其他日期的记录")
    for field in fields:
        values = pd.to_numeric(frame[field], errors="raise")
        frame[field] = values.where(np.isfinite(values) & (values > 0)).astype(float)
    if (frame.up_limit < frame.down_limit).any():
        raise ValueError("stk_limit 涨停价低于跌停价")
    return frame


def merge_daily(daily, factor, basic, limits):
    """以日线为准对齐；历史配套数据缺失保留 NaN，不前填或补成 0/1。"""
    keys = ["ts_code", "trade_date"]
    frame = daily.copy()
    limits = prepare_limits(limits)
    for source, fields in ((factor, ["adj_factor"]), (basic, ["total_mv", "circ_mv"]),
                           (limits, ["up_limit", "down_limit"])):
        if source.empty:
            # Tushare 空响应可能没有列，直接生成数值 NaN 列。
            frame[fields] = np.nan
        else:
            frame = frame.merge(source[keys + fields], on=keys, how="left", validate="one_to_one")
    frame = frame.rename(columns={"vol": "volume", "adj_factor": "factor"})
    for field in [name for name in DAILY_FIELDS if name != "vwap"] + ["amount"]:
        values = pd.to_numeric(frame[field], errors="coerce")
        invalid = frame[field].notna() & values.isna()
        if invalid.any():
            examples = frame.loc[invalid, keys + [field]].head(5).to_dict("records")
            tqdm.write(f"日线字段 {field} 有 {invalid.sum()} 个非数值，保留 NaN，示例：{examples}")
        frame[field] = values.astype("float64")
    frame["vwap"] = frame.amount * 10 / frame.volume.where(frame.volume > 0)
    return frame[keys + list(DAILY_FIELDS)]


def write_intervals(path, rows, calendar):
    """将闭区间对齐交易日并合并；仅明确开放的行业区间保留 2099-12-31。"""
    ranges = []
    for code, start, end in rows:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        left = int(calendar.searchsorted(start))
        right = int(calendar.searchsorted(end, side="right")) - 1
        if left <= right and left < len(calendar):
            ranges.append((code, left, right, end == pd.Timestamp("2099-12-31")))
    merged = []
    for code, left, right, opened in sorted(set(ranges)):
        if merged and merged[-1][0] == code and left <= merged[-1][2] + 1:
            previous = merged[-1]
            merged[-1] = (code, previous[1], max(right, previous[2]), opened or previous[3])
        else:
            merged.append((code, left, right, opened))
    write_lines(path, (
        f"{code}\t{iso_date(calendar[left])}\t"
        f"{'2099-12-31' if opened else iso_date(calendar[right])}"
        for code, left, right, opened in merged
    ))


def build_daily(client, root, basic, calendar, today):
    """逐日读取 CSV，在磁盘映射数组中对齐后直接输出 .bin。"""
    symbols = pd.Index(sorted(set(basic.ts_code)))
    if calendar.empty or symbols.empty:
        raise ValueError("股票列表或交易日历为空")
    codes = set(symbols)
    first = np.full(len(symbols), -1, dtype=int)
    last = first.copy()
    st_rows = []
    completed = []
    all_rows = []
    with TemporaryDirectory(prefix=".daily-", dir=root.parent) as temporary:
        # 按日连续写入，不维护数据库、索引或几万个打开的股票文件。
        matrix = np.memmap(Path(temporary) / "daily.f32", mode="w+", dtype="<f4",
                           shape=(len(calendar), len(symbols), len(DAILY_FIELDS)))
        try:
            for date_index, date in enumerate(tqdm(calendar, desc="逐日日线 / 复权 / 市值 / ST")):
                day = date.strftime("%Y%m%d")
                daily = client.fetch("daily", trade_date=day)
                factor = client.fetch("adj_factor", trade_date=day)
                market = client.fetch("daily_basic", trade_date=day, fields="ts_code,trade_date,total_mv,circ_mv")
                limits = prepare_limits(client.fetch("stk_limit", trade_date=day), day)
                if daily.empty:
                    if date == today:
                        tqdm.write(f"{day} daily 日线尚未发布，历史日历截止上一交易日")
                        break
                    raise ValueError(f"{day} daily 全市场日线为空，无法构建该交易日行情")
                daily = daily[daily.ts_code.isin(codes)]
                if daily.empty:
                    raise ValueError(f"{day} 无沪深股票日线")
                if daily.ts_code.duplicated().any() or not daily.trade_date.eq(day).all():
                    raise ValueError(f"{day} 日线包含重复股票或不属于该日的记录")
                missing = {
                    api: sorted(set(daily.ts_code) - (set() if source.empty else set(source.ts_code)))
                    for api, source in (("adj_factor", factor), ("daily_basic", market))
                }
                if any(missing.values()):
                    if date == today:
                        tqdm.write(f"{day} 配套数据未齐，历史日历截止上一交易日")
                        break
                    for api, absent in missing.items():
                        if absent:
                            tqdm.write(f"{day} {api} 缺少 {len(absent)} 只股票记录，"
                                       f"对应字段保留 NaN，示例：{absent[:10]}")
                frame = merge_daily(daily, factor, market, limits)
                positions = symbols.get_indexer(frame.ts_code)
                matrix[date_index] = np.nan  # 停牌等缺失值保留 NaN。
                matrix[date_index, positions] = frame[list(DAILY_FIELDS)].to_numpy(dtype="<f4")
                first[positions] = np.where(first[positions] < 0, date_index, first[positions])
                last[positions] = date_index
                st = client.fetch("stock_st", trade_date=day)
                if not st.empty:
                    st_rows.extend((code, date, date) for code in set(st.ts_code) & codes)
                completed.append(date)
            if not completed:
                raise ValueError("没有可写入的已发布日线")
            calendar = pd.DatetimeIndex(completed)
            for position in tqdm(np.flatnonzero(first >= 0), desc="写入 Qlib 日线"):
                code = symbols[position]
                left, right = first[position], last[position]
                directory = stock_dir(root, "features", code)
                for field_index, field in enumerate(DAILY_FIELDS):
                    values = np.r_[left, matrix[left:right + 1, position, field_index]].astype("<f4")
                    values.tofile(directory / f"{field}.day.bin")
                all_rows.append((code, calendar[left], calendar[right]))
        finally:
            del matrix  # 先关闭映射，Windows 才能清理临时数组。
    write_intervals(root / "instruments" / "all.txt", all_rows, calendar)
    write_intervals(root / "instruments" / "st.txt", st_rows, calendar)
    return calendar


def snapshot_intervals(snapshots, calendar):
    """快照从其日期起生效，延续至下一快照前一交易日，不向过去回填。"""
    rows = []
    dates = sorted(snapshots)
    for i, date in enumerate(dates):
        left = int(calendar.searchsorted(date))
        right = (int(calendar.searchsorted(dates[i + 1])) - 1
                 if i + 1 < len(dates) else len(calendar) - 1)
        if left <= right and left < len(calendar):
            rows.extend((code, calendar[left], calendar[right]) for code in snapshots[date])
    return rows


def build_indices(client, root, codes, calendar):
    # 没有更早快照时不倒填首个快照。
    for name, index_code in C.INDEX_CODES.items():
        snapshots = {}
        ranges = index_month_ranges(index_code, calendar[-1])
        for start, end in tqdm(ranges, desc=f"{name} 月度成分"):
            frame = client.fetch("index_weight", index_code=index_code,
                                 start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
            if frame.empty:
                continue
            for date, group in frame.groupby("trade_date"):
                snapshots[pd.Timestamp(date)] = set(group.con_code) & codes
        if not snapshots and ranges:
            raise ValueError(f"{name} 没有任何历史成分快照")
        write_intervals(root / "instruments" / f"{name}.txt", snapshot_intervals(snapshots, calendar), calendar)


def build_industry(client, root, basic, calendar):
    classifications = pd.concat([
        client.fetch("index_classify", level="L1", src=source)
        for source in ("SW2014", "SW2021")
    ], ignore_index=True)
    if classifications.empty:
        raise ValueError("申万一级行业列表为空")
    stocks = basic.set_index("ts_code")
    for industry in tqdm(sorted(set(classifications.index_code)), desc="申万一级行业历史成分"):
        rows = []
        for status in ("N", "Y"):
            frame = client.fetch("index_member_all", l1_code=industry, is_new=status)
            if frame.empty:
                continue
            for item in frame[frame.ts_code.isin(stocks.index)].itertuples():
                if pd.isna(item.in_date) or not item.in_date:
                    raise ValueError(f"{industry}/{item.ts_code} 缺少行业纳入日期")
                start = max(pd.Timestamp(item.in_date), stocks.at[item.ts_code, "list_date"])
                if pd.notna(item.out_date) and item.out_date:
                    # 剔除日不再属于该行业，转换为包含两端的区间。
                    end = pd.Timestamp(item.out_date) - pd.Timedelta(days=1)
                elif item.is_new == "Y":
                    end = pd.Timestamp("2099-12-31")
                else:
                    raise ValueError(f"{industry}/{item.ts_code} 历史成分缺少剔除日期")
                delisted = stocks.at[item.ts_code, "delist_date"]
                if pd.notna(delisted):
                    end = min(end, delisted)
                rows.append((item.ts_code, start, end))
        write_intervals(root / "industry" / f"{industry}.txt", rows, calendar)
