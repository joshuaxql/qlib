"""Local data provider and the shared D facade."""

from copy import copy
import csv
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

PRICE_FIELDS = frozenset({"open", "high", "low", "close", "vwap", "pre_close", "up_limit", "down_limit"})


def component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError(f"Invalid data identifier: {value!r}")
    return value


def normalize_code(code):
    code = component(code).upper()
    if re.fullmatch(r"(?:SH|SZ|BJ)\d{6}", code):
        code = code[2:] + "." + code[:2]
    return code


def date_slice(index, start=None, end=None):
    left = 0 if start is None else index.searchsorted(pd.Timestamp(start))
    right = len(index) if end is None else index.searchsorted(pd.Timestamp(end), side="right")
    if start is not None and end is not None and pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start_time must not be after end_time")
    return index[left:right]


def _industry_name_map(frame):
    """Validate static classification labels without using stock snapshots."""
    if "industry_name" not in frame:
        return {}
    if "index_code" not in frame:
        raise ValueError("Invalid industry name catalog: missing index_code")
    names = {}
    for code, name in frame[["index_code", "industry_name"]].itertuples(index=False, name=None):
        if not isinstance(code, str) or not isinstance(name, str):
            raise ValueError("Invalid industry name catalog: codes and names must be strings")
        code, name = component(code).upper(), name.strip()
        if not name:
            raise ValueError(f"Invalid industry name catalog: empty name for {code}")
        if code in names and names[code] != name:
            raise ValueError(f"Conflicting industry names for {code}: {names[code]!r}, {name!r}")
        names[code] = name
    return names


def _industry_name_json(path):
    """Read the dataset-owned catalog without accepting ambiguous JSON keys."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate industry name catalog key: {key!r}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    try:
        catalog = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object,
                             parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid industry name catalog: {path}: {error}") from error
    if (not isinstance(catalog, dict) or type(catalog.get("schema_version")) is not int
            or catalog["schema_version"] != 1 or not isinstance(catalog.get("names"), dict)
            or ("source" in catalog and not isinstance(catalog["source"], dict))):
        raise ValueError(f"Invalid industry name catalog schema: {path}")
    return _industry_name_map(pd.DataFrame(catalog["names"].items(), columns=["index_code", "industry_name"]))


def _industry_name_csv(path):
    """Validate row widths instead of letting CSV index inference discard data."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream, strict=True))
    except (csv.Error, UnicodeError) as error:
        raise ValueError(f"Invalid industry name catalog: {path}: {error}") from error
    if (not rows or len(rows[0]) != len(set(rows[0]))
            or not {"index_code", "industry_name"}.issubset(rows[0])):
        raise ValueError(f"Invalid industry name catalog: {path}")
    header, records = rows[0], [row for row in rows[1:] if row]
    if any(len(row) != len(header) for row in records):
        raise ValueError(f"Invalid industry name catalog row: {path}")
    return _industry_name_map(pd.DataFrame(records, columns=header))


class LocalProvider:
    def __init__(self, provider_uri="~/.qlib/qlib_data/cn_data", *, cache_uri=None, missing="nan", adjust="hfq",
                 pit_cache_bytes=128 * 1024**2):
        from .pit import PITStore
        self.root = Path(provider_uri).expanduser().resolve()
        self.cache_root = Path(cache_uri).expanduser().resolve() if cache_uri is not None else None
        if missing not in ("nan", "raise"):
            raise ValueError("missing must be 'nan' or 'raise'")
        self.missing = missing
        self.adjust = self._adjust_mode(adjust)
        self._adjustment_end = None
        self._pit = PITStore(self.root / "financial", cache_bytes=pit_cache_bytes, missing=missing)
        # Per-provider bounded caches. clear_cache() is required after rebuilding files.
        self._calendar = lru_cache(maxsize=2)(self._read_calendar)
        self._intervals = lru_cache(maxsize=128)(self._read_intervals)
        self._daily = lru_cache(maxsize=256)(self._read_daily)
        self._basic = lru_cache(maxsize=1)(self._read_basic)
        self._catalog = lru_cache(maxsize=1)(self._read_catalog)
        self._calendar(False)  # Fail early when a path is unbuilt/misconfigured.

    def clear_cache(self):
        self._pit.clear()
        for cache in (self._calendar, self._intervals, self._daily, self._basic, self._catalog):
            cache.cache_clear()

    @staticmethod
    def _adjust_mode(adjust):
        if not isinstance(adjust, str) or adjust not in ("hfq", "qfq", "none"):
            raise ValueError("adjust must be 'hfq', 'qfq' or 'none'")
        return adjust

    def _price_view(self, adjust=None, end_time=None):
        """Query-local settings, sharing only the original raw-data caches.

        Filters and expression engines receive this view so all field reads in
        a query use the same adjustment without mutating the parent provider.
        """
        view = copy(self)
        view.adjust = self._adjust_mode(self.adjust if adjust is None else adjust)
        if end_time is not None:
            view._adjustment_end = pd.Timestamp(end_time)
        return view

    def _signal_view(self):
        """Use the adjustment anchor observable on each strategy signal date.

        Ordinary price queries retain their query-end qfq convention. This
        private view also propagates to expression filters in strategy pools.
        """
        view = copy(self)
        view._signal_causal = True
        return view

    def _expression_frame(self, code, fields, dates, history, allow_future):
        from .base import ExpressionEngine, daily_fields, history_bounds

        bounds = history_bounds(fields, allow_future)

        def evaluate(provider, output_dates, evaluation_history):
            direct = provider._direct_daily_frame(code, fields, output_dates)
            if direct is not None:
                return direct
            if bounds is not None and len(output_dates):
                left, right = bounds
                first = max(0, evaluation_history.searchsorted(output_dates[0]) - left)
                stop = min(len(evaluation_history), evaluation_history.searchsorted(output_dates[-1], side="right") + right)
                evaluation_history = evaluation_history[first:stop]
            engine = ExpressionEngine(provider, code, evaluation_history, allow_future)
            engine.prepare(fields)
            return pd.DataFrame({field: engine.evaluate(field).reindex(output_dates) for field in fields})

        if self.adjust != "qfq" or not getattr(self, "_signal_causal", False) or dates.empty:
            return evaluate(self, dates, history)
        price_dependent = any(daily_fields(field, allow_future).intersection(PRICE_FIELDS) for field in fields)
        if not price_dependent:
            return evaluate(self, dates, history)
        # A fixed qfq anchor rescales the entire expression history, including
        # nested rolling operators and price thresholds. Reusing one evaluation
        # while the visible anchor stays constant avoids a full run per day.
        factors = self._daily(code, "factor")
        anchors = factors.where(np.isfinite(factors) & factors.gt(0)).ffill().reindex(dates)
        previous = anchors.shift()
        same_anchor = anchors.eq(previous) | (anchors.isna() & previous.isna())
        groups = (~same_anchor).cumsum()
        frames = []
        for _, group in anchors.groupby(groups, sort=False):
            end = group.index[-1]
            view = self._price_view(end_time=end)
            frames.append(evaluate(view, group.index, self.calendar(end_time=end)))
        return pd.concat(frames).reindex(dates)

    def _direct_daily_frame(self, code, fields, dates):
        """Project ordinary daily fields with one shared price-factor transform."""
        if (getattr(self._field, "__func__", None) is not LocalProvider._field or
                getattr(self._read_daily, "__func__", None) is not LocalProvider._read_daily):
            # Provider extensions retain the expression engine's field hook.
            return None
        names = []
        for expression in fields:
            if not re.fullmatch(r"\$[A-Za-z_][A-Za-z_0-9]*", expression):
                return None
            name = expression[1:]
            if name in ("is_st", "list_days"):
                return None
            names.append(name)
        calendar = self._calendar(False)
        raw = {name: self._daily(code, name) for name in names}
        if any(value.dtype != np.dtype("float64") or not value.index.equals(calendar) for value in raw.values()):
            return None
        positions = calendar.get_indexer(dates)
        if np.any(positions < 0):
            return None
        values = np.empty((len(dates), len(names)), dtype=float)
        price_columns = [column for column, name in enumerate(names) if name in PRICE_FIELDS]
        factor = None
        if price_columns and self.adjust != "none":
            series = raw.get("factor")
            if series is None:
                series = self._daily(code, "factor")
            if series.dtype != np.dtype("float64") or not series.index.equals(calendar):
                return None
            all_factors = series.to_numpy()
            valid = np.isfinite(all_factors) & (all_factors > 0)
            factor = np.where(valid[positions], all_factors[positions], np.nan)
            if self.adjust == "qfq":
                stop = len(calendar) if self._adjustment_end is None else calendar.searchsorted(
                    self._adjustment_end, side="right")
                known = np.flatnonzero(valid[:stop])
                anchor = all_factors[known[-1]] if len(known) else np.nan
                # Match _field's operation order: normalize factors first.
                factor = factor / anchor
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for column, name in enumerate(names):
                selected = raw[name].to_numpy()[positions]
                values[:, column] = selected * factor if factor is not None and name in PRICE_FIELDS else selected
        values[~np.isfinite(values)] = np.nan
        return pd.DataFrame(values, index=dates, columns=fields)

    @staticmethod
    def _frequency(freq):
        if freq != "day":
            raise ValueError("Only daily frequency ('day') is supported")

    def _read_calendar(self, future):
        path = self.root / "calendars" / ("day_future.txt" if future else "day.txt")
        values = path.read_text(encoding="utf-8").splitlines()
        calendar = pd.DatetimeIndex(pd.to_datetime(values), name="datetime")
        if calendar.empty or calendar.hasnans or calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError(f"Invalid trading calendar: {path}")
        return calendar

    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        self._frequency(freq)
        return date_slice(self._calendar(future), start_time, end_time).copy()

    def _read_intervals(self, market):
        if market.startswith("industry/"):
            path = self.root / "industry" / f"{component(market[9:])}.txt"
        else:
            path = self.root / "instruments" / f"{component(market)}.txt"
        result = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            code, start, end = line.split()
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            if pd.isna(start) or pd.isna(end) or start > end:
                raise ValueError(f"Invalid membership interval: {line}")
            result.setdefault(normalize_code(code), []).append((start, end))
        return result

    def markets(self):
        return sorted(p.stem for p in (self.root / "instruments").glob("*.txt"))

    def industries(self):
        return sorted(p.stem for p in (self.root / "industry").glob("*.txt"))

    def industry_names(self):
        """Static industry-code labels; historical membership stays in .txt files.

        The dataset's industry_names.json is authoritative. Old directories
        may use industry/names.csv; missing labels fall back to their codes.
        Labels are reread on every call and never come from another dataset.
        """
        path = self.root / "industry_names.json"
        if path.exists():
            names = _industry_name_json(path)
        else:
            legacy = self.root / "industry" / "names.csv"
            names = _industry_name_csv(legacy) if legacy.exists() else {}
        return {code: names.get(component(code).upper(), code) for code in self.industries()}

    @staticmethod
    def instruments(market="all", filter_pipe=None):
        return {"market": market, "filter_pipe": list(filter_pipe or [])}

    def universe(self, instruments="all", start_time=None, end_time=None, freq="day", *, adjust=None):
        """A date x instrument boolean matrix, with historical membership and filters."""
        self._frequency(freq)
        provider = self._price_view(adjust, end_time)
        dates = self.calendar(start_time, end_time)
        filters = []
        if isinstance(instruments, dict) and "market" in instruments:
            filters = instruments.get("filter_pipe", [])
            instruments = instruments["market"]
        if isinstance(instruments, str):
            intervals = self._intervals(instruments)
        elif isinstance(instruments, dict):
            intervals = {normalize_code(code): spans for code, spans in instruments.items()}
        else:
            full = self._intervals("all")
            codes = sorted({normalize_code(code) for code in instruments})
            unknown = set(codes) - set(full)
            if unknown:
                raise KeyError(f"Unknown instruments: {sorted(unknown)}")
            intervals = {code: full[code] for code in codes}
        columns = sorted(intervals)
        values = np.zeros((len(dates), len(columns)), dtype=bool)
        for column, code in enumerate(columns):
            spans = intervals[code]
            for start, end in spans:
                start, end = pd.Timestamp(start), pd.Timestamp(end)
                if pd.isna(start) or pd.isna(end):
                    continue
                left = dates.searchsorted(start)
                right = dates.searchsorted(end, side="right")
                values[left:right, column] = True
        mask = pd.DataFrame(values, index=dates, columns=columns)
        if filters:
            from .filter import make_filter
            for item in filters:
                mask &= make_filter(item).apply(provider, mask).reindex_like(mask).fillna(False).astype(bool)
        return mask

    def list_instruments(self, instruments="all", start_time=None, end_time=None, freq="day", as_list=False, *, adjust=None):
        mask = self.universe(instruments, start_time, end_time, freq, adjust=adjust)
        if as_list:
            return mask.columns[mask.any(axis=0)].tolist()
        result = {}
        for code in mask:
            values = mask[code].to_numpy()
            edges = np.diff(np.r_[False, values, False].astype(int))
            spans = [(mask.index[a], mask.index[b - 1]) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
            if spans:
                result[code] = spans
        return result

    def _stock_path(self, section, code, filename):
        code = normalize_code(code)
        for alias in (code, code.lower(), code.split(".")[-1] + code.split(".")[0],
                      (code.split(".")[-1] + code.split(".")[0]).lower()):
            path = self.root / section / alias / filename
            if path.exists():
                return path
        return self.root / section / code / filename

    def _read_catalog(self):
        return sorted({p.name.removesuffix(".day.bin") for p in (self.root / "features").glob("*/*.day.bin")})

    def fields(self, kind="daily", instrument=None):
        if kind == "financial":
            registry = self._pit.registry
            ids = set(registry) if instrument is None else set(map(str, self._pit.stock(instrument).index["field_id"]))
            return sorted(registry[key]["name"] for key in ids)
        if kind == "stock_basic":
            return list(self._basic().columns)
        if kind != "daily":
            raise ValueError("kind must be daily, financial or stock_basic")
        if instrument is None:
            return self._catalog().copy()
        directory = self._stock_path("features", instrument, "")
        return sorted(p.name.removesuffix(".day.bin") for p in directory.glob("*.day.bin"))

    def _read_daily(self, code, field):
        field = component(field)
        if field not in self.fields("daily"):
            raise KeyError(f"Unknown daily field: {field}")
        calendar = self._calendar(False)
        result = pd.Series(np.nan, index=calendar, name=field)
        path = self._stock_path("features", code, f"{field}.day.bin")
        if not path.exists():
            if self.missing == "raise":
                raise FileNotFoundError(path)
            return result
        if path.stat().st_size < 4 or path.stat().st_size % 4:
            raise ValueError(f"Corrupt daily file: {path}")
        data = np.fromfile(path, dtype="<f4")
        offset = data[0]
        if not np.isfinite(offset) or offset < 0 or offset != int(offset) or int(offset) + len(data) - 1 > len(calendar):
            raise ValueError(f"Invalid calendar offset/length: {path}")
        result.iloc[int(offset):int(offset) + len(data) - 1] = data[1:]
        return result

    def _field(self, code, field):
        if field == "is_st":
            dates = self._calendar(False)
            result = pd.Series(False, index=dates)
            for start, end in self._intervals("st").get(code, []):
                result.loc[(dates >= start) & (dates <= end)] = True
            return result
        if field == "list_days":
            basic = self._basic().set_index("ts_code")
            if code not in basic.index or pd.isna(basic.at[code, "list_date"]):
                raise ValueError(f"Missing list_date: {code}")
            return pd.Series((self._calendar(False) - basic.at[code, "list_date"]).days,
                             index=self._calendar(False), dtype=float)
        values = self._daily(code, field)
        if field not in PRICE_FIELDS or self.adjust == "none":
            return values
        factor = self._daily(code, "factor")
        factor = factor.where(np.isfinite(factor) & factor.gt(0))
        if self.adjust == "qfq":
            # The anchor is per instrument and bounded by the query end, even
            # when expression history includes future data for research labels.
            known = factor if self._adjustment_end is None else factor.loc[:self._adjustment_end]
            known = known.dropna()
            factor = factor / (known.iloc[-1] if len(known) else np.nan)
        return (values * factor).rename(field)

    def features(self, instruments, fields, start_time=None, end_time=None, freq="day", *, allow_future=True, adjust=None):
        """Evaluate expressions after adjusting price fields.

        adjust=None inherits the provider default ('hfq' unless configured).
        'hfq': price * factor; 'qfq': price * factor / last valid factor at
        or before end_time; 'none': raw prices. The qfq anchor uses the latest
        local factor when end_time is omitted. Non-price fields stay raw.
        """
        from .base import validate
        provider = self._price_view(adjust, end_time)
        if isinstance(fields, str):
            fields = [fields]
        fields = list(fields)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("fields must be nonempty and unique")
        for expression in fields:
            validate(expression, allow_future)
        mask = provider.universe(instruments, start_time, end_time, freq)
        # Retain complete preceding history for nested/expanding operators.
        # Research labels may explicitly access later data through negative Ref.
        history = self._calendar(False) if allow_future else self.calendar(end_time=end_time)
        frames = {}
        for code in mask:
            dates = mask.index[mask[code]]
            if len(dates):
                frames[code] = provider._expression_frame(code, fields, dates, history, allow_future)
        if not frames:
            index = pd.MultiIndex.from_arrays([[], pd.DatetimeIndex([])], names=["instrument", "datetime"])
            return pd.DataFrame(index=index, columns=fields, dtype=float)
        return pd.concat(frames, names=["instrument", "datetime"]).sort_index()

    def daily(self, instruments="all", fields=None, start_time=None, end_time=None, *, adjust=None):
        """Read daily fields with the same price adjustment as features()."""
        fields = self.fields("daily") if fields is None else ([fields] if isinstance(fields, str) else list(fields))
        return self.features(instruments, [f"${f.lstrip('$')}" for f in fields], start_time, end_time, adjust=adjust).rename(
            columns=lambda c: c.removeprefix("$")
        )

    def _read_basic(self):
        frame = pd.read_csv(self.root / "stock_basic.csv", dtype={"ts_code": str, "symbol": str})
        frame["ts_code"] = frame.ts_code.map(normalize_code)
        if frame.ts_code.duplicated().any():
            raise ValueError("Duplicate stock_basic instruments")
        for field in ("list_date", "delist_date"):
            if field in frame:
                # Both YYYYMMDD and ISO dates are accepted, never epoch integers.
                frame[field] = pd.to_datetime(frame[field].astype("string").str.replace(r"\.0$", "", regex=True), format="mixed")
        return frame

    def stock_basic(self, instruments=None, fields=None):
        """Current snapshot, NOT historical name/industry/list_status data."""
        frame = self._basic()
        if instruments is not None:
            codes = [instruments] if isinstance(instruments, str) else instruments
            frame = frame[frame.ts_code.isin([normalize_code(c) for c in codes])]
        if fields is not None:
            frame = frame[[fields] if isinstance(fields, str) else list(fields)]
        return frame.copy().reset_index(drop=True)

    def financial_records(self, instrument, field=None, asof=None):
        """Publication/revision history from one consolidated stock file."""
        from .pit import date_number
        records = self._pit.stock(instrument).records
        if field is not None:
            records = records[records["field_id"] == self._pit.resolve(field)]
        if asof is not None:
            records = records[records["date"] <= date_number(asof)]
        frame = pd.DataFrame({name: records[name] for name in ("field_id", "date", "period", "value")})
        registry = self._pit.registry
        frame["field"] = [registry[str(key)]["name"] for key in frame.field_id]
        frame["date"] = pd.to_datetime(frame.date.astype(str), format="%Y%m%d")
        return frame.sort_values(["date", "field", "period"], kind="stable").reset_index(drop=True)

    def financial(self, instruments, fields, asof, periods=None):
        """Latest visible version per report period, preserving explicit NaNs."""
        codes = (self.list_instruments(instruments, end_time=asof, as_list=True)
                 if isinstance(instruments, (str, dict)) else [normalize_code(code) for code in instruments])
        fields = [fields] if isinstance(fields, str) else list(fields)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("fields must be nonempty and unique")
        ids = {field: self._pit.resolve(field) for field in fields}
        frames = {}
        for code in codes:
            rows = self._pit.stock(code).snapshot(list(ids.values()), asof)
            columns = {}
            for field, field_id in ids.items():
                selected = rows[rows["field_id"] == field_id]
                columns[field] = pd.Series(selected["value"], index=pd.Index(selected["period"], name="period"), dtype=float)
            frame = pd.DataFrame(columns)
            if periods is not None:
                frame = frame.reindex([periods] if isinstance(periods, (int, np.integer)) else periods)
            frame.index.name = "period"
            frames[code] = frame
        if not frames:
            return pd.DataFrame(columns=fields, index=pd.MultiIndex.from_arrays([[], []], names=["instrument", "period"]))
        return pd.concat(frames, names=["instrument", "period"]).sort_index()

    def read_table(self, relative_path, **equals):
        """Read an existing CSV in cache_uri.

        Filters compare their string forms. Values retain their raw CSV format.
        """
        if self.cache_root is None:
            raise ValueError("Set cache_uri to read original cached CSV tables")
        path = (self.cache_root / relative_path).resolve()
        if not path.is_relative_to(self.cache_root) or path.suffix.lower() != ".csv":
            raise ValueError("Table must be a CSV within cache_uri")
        string_fields = ("ts_code", "symbol", "con_code", "index_code", "l1_code", "l2_code", "l3_code",
                         "trade_date", "cal_date", "pretrade_date", "end_date",
                         "list_date", "delist_date", "in_date", "out_date")
        try:
            frame = pd.read_csv(path, dtype={f: str for f in string_fields}, low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        for field, value in equals.items():
            frame = frame[frame[field].astype(str) == str(value)]
        return frame.reset_index(drop=True)


class _DataFacade:
    def __init__(self):
        self._provider = None

    def register(self, provider):
        self._provider = provider

    def __getattr__(self, name):
        if self._provider is None:
            raise RuntimeError("Call qlib.init(provider_uri=...) before using D")
        return getattr(self._provider, name)


D = _DataFacade()
