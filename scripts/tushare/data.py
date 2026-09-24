"""Tushare 请求、内存分页和永久 CSV 缓存。

日数据按 YYYYMMDD，指数按 YYYYMM 保存。
空 CSV 下次重新补取；当日日数据、当前月指数每次刷新，
基础信息、日历、行业每日刷新。完整 CSV 永久保留，中断请求重取该 CSV。

本目录保持命名空间包（不添加 __init__.py），避免直接运行 scripts 下的
脚本时遮蔽第三方 tushare 包；项目内使用 scripts.tushare.data 导入。
"""

import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts import config as C
from scripts.tushare.fields import FINA_INDICATOR_FIELDS

API_PAGE_SIZES = {"stk_limit": 5800}
LIMIT_FIELDS = ("ts_code", "trade_date", "up_limit", "down_limit")


class TushareClient:
    """串行限速、空值/异常重试；分页在内存合并，完成后交给 CSV 缓存。"""

    def __init__(self, pro):
        self.pro = pro
        self.last_request = 0.0

    def _request_page(self, api, offset, params):
        for attempt in range(C.RETRIES + 1):
            time.sleep(max(0, C.REQUEST_INTERVAL - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                page = self.pro.query(api, limit=API_PAGE_SIZES.get(api, C.PAGE_SIZE), offset=offset, **params)
                if not isinstance(page, pd.DataFrame):
                    raise TypeError(f"{api} 未返回 DataFrame")
            except Exception as exc:
                if attempt == C.RETRIES:
                    raise RuntimeError(f"{api} 请求失败，参数={params}, offset={offset}") from exc
                reason = f"请求失败：{exc}"
            else:
                if not page.empty:
                    return page
                if attempt == C.RETRIES:
                    tqdm.write(f"{api} 参数={params}, offset={offset} 重试 {C.RETRIES} 次后仍为空")
                    return page
                reason = "返回为空"
            tqdm.write(f"{api} 参数={params}, offset={offset} {reason}，第 {attempt + 1}/{C.RETRIES} 次重试")
            time.sleep(3 * (attempt + 1))

    def fetch(self, api, **params):
        pages = []
        offset = 0
        previous = None
        page_size = API_PAGE_SIZES.get(api, C.PAGE_SIZE)
        while True:
            page = self._request_page(api, offset, params)
            if page.empty:
                break
            fingerprint = pd.util.hash_pandas_object(page, index=False).values.tobytes()
            if fingerprint == previous:
                raise RuntimeError(f"{api} 返回重复页，可能未支持 offset，拒绝保存截断数据")
            previous = fingerprint
            pages.append(page)
            offset += len(page)
            # ponytail: 中证1000按单快照取第一页；若需月内多快照，改为按快照日下载。
            if api == "index_weight" and params.get("index_code") == "000852.SH":
                break
            if len(page) < page_size:
                break
        return pd.concat(pages, ignore_index=True).drop_duplicates() if pages else page


CSV_STRING_FIELDS = (
    "ts_code", "symbol", "con_code", "index_code", "l1_code", "l2_code", "l3_code",
    "trade_date", "cal_date", "pretrade_date", "end_date",
    "list_date", "delist_date", "in_date", "out_date", "ann_date", "f_ann_date", "report_type", "update_flag",
)
FINANCIAL_TABLES = ("fina_indicator",)
FINANCIAL_META = ("ts_code", "ann_date", "end_date", "update_flag")


def financial_fields(selection=None):
    selection = C.PIT_FIELDS if selection is None else selection
    result = {}
    for table, fields in selection.items():
        if table not in FINANCIAL_TABLES or isinstance(fields, str):
            raise ValueError(f"Invalid PIT field selection: {table}")
        values = list(dict.fromkeys(fields))
        if any(field not in FINA_INDICATOR_FIELDS for field in values):
            raise ValueError(f"Invalid PIT fields for {table}: {values}")
        if values:
            result[table] = values
    return result


def financial_cache_path(directory, table, period):
    path = Path(directory) / "financial" / table / f"{period[:6]}.csv"
    # Resolve the supported financial cache directory spellings.
    legacy = Path(directory) / "finacial" / table / path.name
    return legacy if not path.exists() and legacy.exists() else path


def read_csv(path, columns=None):
    # 日期/代码不能推断成整数，否则前导零和日期语义会丢失。
    try:
        return pd.read_csv(path, dtype={field: str for field in CSV_STRING_FIELDS}, low_memory=False,
                           usecols=None if columns is None else lambda name: name in columns)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def cache_day(path):
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert("Asia/Shanghai").tz_localize(None).normalize()


def cache_csv(path, load, refresh=False):
    """完整下载后原子替换；空文件不作为下一次下载的完成标记。"""
    existing = read_csv(path) if path.exists() else pd.DataFrame()
    refresh = refresh or C.REFRESH_CACHE
    if not refresh and not existing.empty:
        return existing
    frame = load()
    for column in ("ts_code", "con_code"):
        if column in frame:
            frame = frame[frame[column].astype("string").str.fullmatch(r"\d{6}\.(SH|SZ)", na=False)]
    if frame.empty and not existing.empty:
        tqdm.write(f"{path.name} 刷新为空，保留已有非空缓存")
        return existing
    if not len(frame.columns):
        frame = pd.DataFrame(columns=["ts_code"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)
    return frame


class CsvClient:
    """构建阶段只读 CSV；缺失文件直接报错，不再隐式发起下载。"""

    def __init__(self, directory):
        self.directory = Path(directory)

    def fetch(self, api, **params):
        if api in ("daily", "adj_factor", "daily_basic", "stock_st", "stk_limit"):
            path = Path(api) / f"{params['trade_date']}.csv"
        elif api == "index_weight":
            path = Path(api) / params["index_code"] / f"{params['start_date'][:6]}.csv"
        elif api in tuple(f"{table}_vip" for table in FINANCIAL_TABLES):
            path = financial_cache_path(self.directory, api.removesuffix("_vip"), params["period"]).resolve()
        else:
            path = Path("industry.csv" if api in ("index_classify", "index_member_all") else f"{api}.csv")
        columns = params.get("fields")
        frame = read_csv(self.directory / path, columns.split(",") if columns else None)
        if frame.empty:
            return frame
        if api == "stock_basic":
            frame = frame[frame.list_status == params["list_status"]]
        elif api == "trade_cal":
            frame = frame[(frame.exchange == params["exchange"]) &
                          frame.cal_date.between(params["start_date"], params["end_date"])]
        elif api == "index_classify":
            frame = frame[["l1_code"]].drop_duplicates().rename(columns={"l1_code": "index_code"})
        elif api == "index_member_all":
            frame = frame[(frame.l1_code == params["l1_code"]) & (frame.is_new == params["is_new"])]
        elif api == "index_weight":
            frame = frame[frame.trade_date.between(params["start_date"], params["end_date"])]
        elif api.endswith("_vip") and api.removesuffix("_vip") in FINANCIAL_TABLES:
            frame = frame[frame.end_date == params["period"]]
        return frame.copy()


def index_month_ranges(index_code, end):
    """下载和构建共用范围，首月从接口数据起点开始请求。"""
    start = max(pd.Timestamp(C.START_DATE), pd.Timestamp(C.INDEX_START_DATES.get(index_code, C.START_DATE)))
    end = pd.Timestamp(end)
    if start > end:
        return []
    return [(max(month.start_time, start), min(month.end_time.normalize(), end))
            for month in pd.period_range(start, end, freq="M")]


def fetch_calendar(client, today):
    target = today + pd.DateOffset(years=1)
    dates = set()
    tasks = [(exchange, year) for exchange in ("SSE", "SZSE")
             for year in range(pd.Timestamp(C.START_DATE).year, target.year + 1)]
    for exchange, year in tqdm(tasks, desc="交易日历"):
        frame = client.fetch("trade_cal", exchange=exchange, is_open="1",
                             start_date=max(C.START_DATE, f"{year}0101"),
                             end_date=min(target.strftime("%Y%m%d"), f"{year}1231"))
        if not frame.empty:
            dates.update(frame.loc[pd.to_numeric(frame.is_open) == 1, "cal_date"])
    future = pd.DatetimeIndex(pd.to_datetime(sorted(dates), format="%Y%m%d"))
    future = future[(future >= pd.Timestamp(C.START_DATE)) & (future <= target)]
    history = future[future <= today]
    if history.empty:
        raise ValueError("交易日历为空")
    return history, future


def download_cache(client, directory, today):
    """先完成所有下载，任何后续构建失败都不会删除这里的文件。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def snapshot(name, load):
        path = directory / name
        stale = not path.exists() or cache_day(path) != today
        return cache_csv(path, load, refresh=stale)

    snapshot("stock_basic.csv", lambda: pd.concat([
        client.fetch("stock_basic", list_status=status,
                     fields="ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,"
                            "exchange,curr_type,list_status,list_date,delist_date,is_hs")
        for status in tqdm(("L", "D", "P"), desc="下载股票基础信息")
    ], ignore_index=True))

    def calendar_rows():
        target = today + pd.DateOffset(years=1)
        tasks = [(exchange, year) for exchange in ("SSE", "SZSE")
                 for year in range(pd.Timestamp(C.START_DATE).year, target.year + 1)]
        frames = []
        for exchange, year in tqdm(tasks, desc="下载交易日历"):
            frame = client.fetch("trade_cal", exchange=exchange, is_open="1",
                                 start_date=max(C.START_DATE, f"{year}0101"),
                                 end_date=min(target.strftime("%Y%m%d"), f"{year}1231"))
            frames.append(frame.assign(exchange=exchange))
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    snapshot("trade_cal.csv", calendar_rows)
    reader = CsvClient(directory)
    calendar, _ = fetch_calendar(reader, today)
    for date in tqdm(calendar, desc="下载每日 CSV / 断点复用"):
        day = date.strftime("%Y%m%d")
        for api in ("daily", "adj_factor", "daily_basic", "stock_st"):
            path = directory / api / f"{day}.csv"
            params = {"trade_date": day}
            if api == "daily_basic":
                params["fields"] = "ts_code,trade_date,total_mv,circ_mv"
            cache_csv(path, lambda: client.fetch(api, **params),
                      refresh=date == today or (path.exists() and cache_day(path) <= date))

    download_limit_cache(client, directory, calendar, today)

    for name, index_code in C.INDEX_CODES.items():
        for start, end in tqdm(index_month_ranges(index_code, today), desc=f"下载 {name} 月度 CSV"):
            month = start.to_period("M")
            path = directory / "index_weight" / index_code / f"{month.strftime('%Y%m')}.csv"
            cache_csv(path, lambda: client.fetch(
                "index_weight", index_code=index_code,
                start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")),
                refresh=month == today.to_period("M") or
                (path.exists() and cache_day(path) <= month.end_time.normalize()))

    def industry_rows():
        classes = pd.concat([client.fetch("index_classify", level="L1", src=source)
                             for source in ("SW2014", "SW2021")], ignore_index=True)
        if classes.empty:
            raise ValueError("申万一级行业列表为空")
        frames = []
        for code in tqdm(sorted(set(classes.index_code)), desc="下载行业 CSV"):
            for status in ("N", "Y"):
                frame = client.fetch("index_member_all", l1_code=code, is_new=status)
                frames.append(frame.assign(l1_code=code, is_new=status))
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    snapshot("industry.csv", industry_rows)
    if C.BUILD_PIT:
        download_financial_cache(client, directory, today)
    tqdm.write(f"CSV 下载完成，缓存永久保留：{directory}")
    return reader


def download_limit_cache(client, directory, calendar, today):
    """Cache paginated stk_limit responses by trade date, refreshing today's file."""
    directory = Path(directory)
    for date in tqdm(calendar, desc="下载每日涨跌停价格"):
        day = date.strftime("%Y%m%d")
        path = directory / "stk_limit" / f"{day}.csv"

        def load():
            frame = client.fetch("stk_limit", trade_date=day, fields=",".join(LIMIT_FIELDS))
            if frame.empty:
                return pd.DataFrame(columns=LIMIT_FIELDS)
            if not set(LIMIT_FIELDS).issubset(frame):
                raise ValueError(f"{day} stk_limit 响应缺少字段")
            if not frame.trade_date.eq(day).all() or frame.ts_code.duplicated().any():
                raise ValueError(f"{day} stk_limit 包含重复股票或其他日期的记录")
            return frame[list(LIMIT_FIELDS)]

        cache_csv(path, load, refresh=date == today or (path.exists() and cache_day(path) <= date))


def download_financial_cache(client, directory, today, selection=None):
    """Fetch selected quarterly fields, retaining previously received versions."""
    quarters = [quarter for quarter in pd.period_range(C.START_DATE, today, freq="Q")
                if quarter.end_time.normalize() <= today]
    for table, fields in financial_fields(selection).items():
        required = [*FINANCIAL_META, *fields]
        for position, quarter in enumerate(tqdm(quarters, desc=f"下载 {table} PIT CSV")):
            period = quarter.end_time.strftime("%Y%m%d")
            path = financial_cache_path(directory, table, period)
            existing = read_csv(path) if path.exists() else pd.DataFrame()
            refresh = (C.REFRESH_CACHE or position >= len(quarters) - C.PIT_REFRESH_QUARTERS or
                       not set(required).issubset(existing.columns) or existing.empty)
            if not refresh:
                continue
            incoming = client.fetch(f"{table}_vip", period=period, fields=",".join(required))
            if incoming.empty:
                if not existing.empty:
                    tqdm.write(f"{table} {period} 刷新为空，保留历史版本")
                    continue
                incoming = pd.DataFrame(columns=required)
            elif not set(required).issubset(incoming.columns):
                raise ValueError(f"{table} 响应缺少请求字段: {set(required) - set(incoming.columns)}")
            if not existing.empty:
                # Exact-version replacements use the newest download, while
                # distinct announcement dates and report types remain intact.
                incoming = pd.concat([existing, incoming], ignore_index=True)
                keys = ["ts_code", "end_date", "ann_date", "update_flag"]
                for key in keys:
                    incoming[key] = incoming[key].astype("string")
                incoming = incoming.drop_duplicates(keys, keep="last")
            cache_csv(path, lambda frame=incoming: frame, refresh=True)


def download_data(directory, today):
    """连接第三方 SDK 并下载；离线构建无需导入 SDK 或配置 Token。"""
    if not C.TOKEN.strip():
        raise ValueError("请设置 TUSHARE_TOKEN 环境变量，或填写 scripts/config.py 的 TOKEN")
    import tushare as ts

    tqdm.write(f"CSV 缓存：{directory}；本次构建截止日：{today:%Y-%m-%d}")
    return download_cache(TushareClient(ts.pro_api(C.TOKEN)), directory, today)
