"""Quarterly Tushare CSV -> consolidated PIT v2, with in-memory stock grouping."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from qlib.data.pit import register_fields, write_stock
from scripts import config as C
from scripts.tushare.data import FINANCIAL_META, financial_fields


def prepare_financial(frame, table, fields, codes, today):
    """Normalize one report quarter. Values retain their source accounting basis.

    Same-day publication preference: the highest update_flag.
    Distinct publication dates are never collapsed. Null values are revisions,
    not an instruction to carry a previous non-null value forward.
    """
    columns = ["instrument", "date", "period", "field", "value"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if table != "fina_indicator":
        raise ValueError(f"Unsupported financial table: {table}")
    required = {*FINANCIAL_META, *fields}
    if not required.issubset(frame):
        raise ValueError(f"{table} CSV 缺少字段: {sorted(required - set(frame))}; 请刷新该季度缓存")
    frame = frame[frame.ts_code.isin(codes)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    published = pd.to_datetime(frame.ann_date.replace("", None), format="%Y%m%d")
    end = pd.to_datetime(frame.end_date, format="%Y%m%d")
    if published.isna().any() or end.isna().any() or not end.dt.is_quarter_end.all():
        raise ValueError(f"{table} 缺少有效公告日或自然季度报告期")
    premature = published < end
    if premature.any():
        examples = frame.loc[premature, ["ts_code", "ann_date", "end_date"]].head(5).to_dict("records")
        tqdm.write(f"{table} 跳过 {int(premature.sum())} 条公告日早于报告期末的记录：{examples}")
        frame = frame.loc[~premature].copy()
        published, end = published.loc[frame.index], end.loc[frame.index]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["date"] = published.dt.strftime("%Y%m%d").astype(int)
    frame["period"] = end.dt.year * 100 + end.dt.quarter
    frame["_updated"] = pd.to_numeric(frame.get("update_flag", pd.Series(0, index=frame.index)), errors="raise").fillna(0)
    frame = frame[published <= pd.Timestamp(today)]
    keys = ["ts_code", "date", "period"]
    # Equally preferred conflicting source rows have no reliable chronology.
    unique = frame.drop_duplicates(keys + ["_updated"] + fields)
    if unique.duplicated(keys + ["_updated"]).any():
        raise ValueError(f"{table} 同一公告版本存在冲突记录")
    frame = frame.sort_values(keys + ["_updated"], kind="stable").drop_duplicates(keys, keep="last")
    long = frame.melt(id_vars=keys, value_vars=fields, var_name="field", value_name="value")
    long["value"] = pd.to_numeric(long.value, errors="raise")
    if np.isinf(long.value).any():
        raise ValueError(f"{table} 财务指标包含无穷值")
    return long.rename(columns={"ts_code": "instrument"})[columns]


def build_financial(client, root, codes, today, selection=None):
    """Read quarterly CSVs as wide tables in memory, then write each stock directly.

    Only the current stock is expanded to indicator rows. No intermediate
    database or data files are created. Re-running replaces each stock pair.
    """
    root = Path(root)
    financial = root / "financial"
    financial.mkdir(parents=True, exist_ok=True)
    selection = financial_fields(selection)
    definitions = {}
    for table, fields in selection.items():
        for field in fields:
            definitions[field] = {
                "source": f"{table}_vip", "frequency": "quarterly",
                "unit": "source", "basis": "source",
            }
    register_fields(financial, definitions, replace=True)
    quarters = [quarter for quarter in pd.period_range(C.START_DATE, today, freq="Q")
                if quarter.end_time.normalize() <= today]
    for table, fields in selection.items():
        frames = []
        required = [*FINANCIAL_META, *fields]
        for quarter in tqdm(quarters, desc=f"读取 {table} 财务 CSV"):
            frame = client.fetch(f"{table}_vip", period=quarter.end_time.strftime("%Y%m%d"),
                                 fields=",".join(required))
            if frame.empty:
                continue
            if not set(required).issubset(frame):
                raise ValueError(f"{table} CSV 缺少字段: {sorted(set(required) - set(frame))}; 请刷新该季度缓存")
            frame = frame.loc[frame.ts_code.isin(codes), required]
            if not frame.empty:
                frames.append(frame)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        del frames
        groups = combined.groupby("ts_code", sort=True)
        for code, frame in tqdm(groups, total=groups.ngroups, desc="CSV 写入合并 PIT / 每股两文件"):
            rows = prepare_financial(frame, table, fields, {code}, today)
            if not rows.empty:
                write_stock(financial, code, rows.drop(columns="instrument"), update=False)
