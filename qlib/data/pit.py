"""PIT v2: one packed data/index pair per stock and a shared field dictionary.

Records: <IIIdQ (28 bytes). Index entries: <IIQQ (24 bytes).
Both files have a 72-byte header with version, generation and payload checksum.
Records are grouped by (field_id, period), then by publication date, so as-of
queries use binary search rather than following revision pointers repeatedly.
"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
import struct
import uuid

import numpy as np
import pandas as pd

from ._libs.pit import asof_indices

VERSION = 2
HEADER = struct.Struct("<8sII16sQ32s")
DATA_MAGIC, INDEX_MAGIC = b"QLPITD2\0", b"QLPITI2\0"
NO_NEXT = np.iinfo(np.uint64).max
RECORD_DTYPE = np.dtype([("field_id", "<u4"), ("date", "<u4"), ("period", "<u4"),
                         ("value", "<f8"), ("next", "<u8")])
INDEX_DTYPE = np.dtype([("field_id", "<u4"), ("period", "<u4"), ("offset", "<u8"), ("count", "<u8")])
FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?")


def date_number(value):
    date = pd.Timestamp(str(value))
    if pd.isna(date):
        raise ValueError("PIT date must not be missing")
    return int(date.strftime("%Y%m%d"))


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_registry(root):
    path = Path(root) / "fields.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("format_version") != VERSION or catalog.get("byte_order") != "little":
        raise ValueError(f"Unsupported PIT dictionary: {path}")
    fields = catalog["fields"]
    names = set()
    for key, metadata in fields.items():
        name = metadata.get("name", "")
        if not key.isdigit() or not 0 < int(key) <= 0xFFFFFFFF or str(int(key)) != key:
            raise ValueError(f"Invalid PIT field id: {key}")
        if not FIELD_NAME.fullmatch(name) or name in names or metadata.get("frequency") != "quarterly":
            raise ValueError(f"Invalid PIT field metadata: {metadata}")
        names.add(name)
    return catalog


def register_fields(root, definitions, *, replace=False):
    """Add stable IDs; existing metadata cannot be silently redefined.

    definitions maps canonical names to metadata (unit, basis, frequency, ...).
    This is a single-writer operation, normally performed once before building.
    replace=True removes unselected catalog entries; use only when rebuilding
    every stock file in a staging directory. Retained fields keep their IDs.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    catalog = read_registry(root) if (root / "fields.json").exists() else {
        "format_version": VERSION, "byte_order": "little", "fields": {},
    }
    if replace:
        catalog["fields"] = {key: meta for key, meta in catalog["fields"].items()
                             if meta["name"] in definitions}
    fields = catalog["fields"]
    names = {meta["name"]: key for key, meta in fields.items()}
    next_id = max(map(int, fields), default=0) + 1
    for name, attributes in sorted(definitions.items()):
        metadata = {**attributes, "name": name, "frequency": attributes.get("frequency", "quarterly")}
        if not FIELD_NAME.fullmatch(name) or metadata["frequency"] != "quarterly":
            raise ValueError(f"Invalid quarterly PIT field: {name}")
        if name in names:
            if fields[names[name]] != metadata:
                raise ValueError(f"PIT field metadata changed: {name}")
        else:
            if next_id > 0xFFFFFFFF:
                raise OverflowError("Too many PIT fields")
            fields[str(next_id)] = metadata
            names[name] = str(next_id)
            next_id += 1
    temporary = root / "fields.json.tmp"
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(root / "fields.json")
    return {name: int(key) for name, key in names.items()}


def _read_file(path, magic, dtype):
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError(f"Truncated PIT header: {path}")
        actual_magic, version, size, generation, count, digest = HEADER.unpack(header)
        if (actual_magic, version, size) != (magic, VERSION, dtype.itemsize):
            raise ValueError(f"Unsupported PIT layout: {path}")
        if path.stat().st_size != HEADER.size + count * size:
            raise ValueError(f"Invalid PIT file length: {path}")
        records = np.fromfile(stream, dtype=dtype, count=count)
    if hashlib.sha256(records.tobytes()).digest() != digest:
        raise ValueError(f"PIT checksum mismatch: {path}")
    return generation, records


def _make_index(records):
    if not len(records):
        return np.empty(0, dtype=INDEX_DTYPE)
    changes = ((records["field_id"][1:] != records["field_id"][:-1]) |
               (records["period"][1:] != records["period"][:-1]))
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    index = np.empty(len(starts), dtype=INDEX_DTYPE)
    index["field_id"] = records["field_id"][starts]
    index["period"] = records["period"][starts]
    index["offset"] = HEADER.size + starts.astype(np.uint64) * RECORD_DTYPE.itemsize
    index["count"] = np.diff(np.r_[starts, len(records)])
    return index


class StockPIT:
    def __init__(self, records, index):
        self.records, self.index = records, index
        self.dates = np.ascontiguousarray(records["date"])
        self.starts = np.ascontiguousarray((index["offset"] - HEADER.size) // RECORD_DTYPE.itemsize)
        self.counts = np.ascontiguousarray(index["count"])
        self.event_order = np.argsort(self.dates, kind="stable")
        self.nbytes = sum(a.nbytes for a in (records, index, self.dates, self.starts, self.counts, self.event_order))
        for array in (records, index, self.dates, self.starts, self.counts, self.event_order):
            array.flags.writeable = False

    @classmethod
    def read(cls, directory):
        directory = Path(directory)
        generation, records = _read_file(directory / "pit.data", DATA_MAGIC, RECORD_DTYPE)
        index_generation, index = _read_file(directory / "pit.index", INDEX_MAGIC, INDEX_DTYPE)
        if generation != index_generation:
            raise ValueError(f"PIT data/index generation mismatch (incomplete update): {directory}")
        if len(records):
            order = np.lexsort((records["date"], records["period"], records["field_id"]))
            if not np.array_equal(order, np.arange(len(records))):
                raise ValueError(f"Unsorted PIT records: {directory}")
            same = ((records["field_id"][1:] == records["field_id"][:-1]) &
                    (records["period"][1:] == records["period"][:-1]))
            if np.any(same & (records["date"][1:] == records["date"][:-1])):
                raise ValueError(f"Duplicate PIT version: {directory}")
            expected_next = np.full(len(records), NO_NEXT, dtype=np.uint64)
            positions = np.flatnonzero(same)
            expected_next[positions] = HEADER.size + (positions + 1).astype(np.uint64) * RECORD_DTYPE.itemsize
            if not np.array_equal(records["next"], expected_next):
                raise ValueError(f"Invalid PIT revision links: {directory}")
        if not np.array_equal(index, _make_index(records)):
            raise ValueError(f"Invalid PIT period index: {directory}")
        return cls(records, index)

    def snapshot(self, field_ids, asof):
        groups = np.isin(self.index["field_id"], field_ids)
        positions = asof_indices(self.dates, self.starts[groups], self.counts[groups], date_number(asof))
        return self.records[positions[positions >= 0]]

    def events(self, field_ids, end):
        order = self.event_order
        order = order[np.isin(self.records["field_id"][order], field_ids) & (self.dates[order] <= date_number(end))]
        if not len(order):
            return
        dates = self.dates[order]
        boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(order)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            yield int(dates[start]), self.records[order[start:stop]]


def write_stock(root, instrument, frame, *, update=True):
    """Write date/period/field/value rows; explicit NaN revisions are retained.

    update=True merges new versions, replacing only identical field/period/date
    keys. Both files are rebuilt in grouped order and replaced from temporary
    files. Their shared generation detects an interrupted pair replacement.
    """
    from .data import normalize_code
    root = Path(root)
    instrument = normalize_code(instrument)
    catalog = read_registry(root)
    ids = {meta["name"]: int(key) for key, meta in catalog["fields"].items()}
    frame = frame[["date", "period", "field", "value"]].copy()
    unknown = set(frame.field) - set(ids)
    if unknown:
        raise KeyError(f"Unregistered PIT fields: {sorted(unknown)}")
    frame["field_id"] = frame.field.map(ids)
    dates = pd.to_datetime(frame.date.astype(str), format="mixed", errors="raise")
    if dates.isna().any():
        raise ValueError("Missing PIT publication date")
    frame["date"] = dates.dt.strftime("%Y%m%d").astype("uint32")
    periods = pd.to_numeric(frame.period, errors="raise")
    if (periods.isna() | periods.ne(np.floor(periods)) | ~periods.mod(100).between(1, 4) |
            ~periods.floordiv(100).between(1900, 9999)).any():
        raise ValueError("PIT periods must be YYYYQQ (quarters 01..04)")
    frame["period"] = periods.astype("uint32")
    frame["value"] = pd.to_numeric(frame.value, errors="raise").astype(float)
    if np.isinf(frame.value).any():
        raise ValueError("PIT values must be finite or NaN")
    directory = root / instrument
    directory.mkdir(parents=True, exist_ok=True)
    if update and ((directory / "pit.data").exists() or (directory / "pit.index").exists()):
        old = StockPIT.read(directory).records
        previous = pd.DataFrame({name: old[name] for name in ("field_id", "date", "period", "value")})
        frame = pd.concat([previous, frame], ignore_index=True)
    frame = frame.drop_duplicates(["field_id", "period", "date"], keep="last").sort_values(
        ["field_id", "period", "date"], kind="stable")
    records = np.empty(len(frame), dtype=RECORD_DTYPE)
    for name in ("field_id", "date", "period", "value"):
        records[name] = frame[name].to_numpy()
    records["next"] = NO_NEXT
    index = _make_index(records)
    for item in index:
        start = (int(item["offset"]) - HEADER.size) // RECORD_DTYPE.itemsize
        stop = start + int(item["count"])
        records["next"][start:stop - 1] = HEADER.size + np.arange(start + 1, stop, dtype=np.uint64) * RECORD_DTYPE.itemsize
    generation = uuid.uuid4().bytes
    for name, magic, array in (("pit.data", DATA_MAGIC, records), ("pit.index", INDEX_MAGIC, index)):
        payload = array.tobytes()
        header = HEADER.pack(magic, VERSION, array.dtype.itemsize, generation, len(array), hashlib.sha256(payload).digest())
        (directory / f"{name}.tmp").write_bytes(header + payload)
    (directory / "pit.data.tmp").replace(directory / "pit.data")
    (directory / "pit.index.tmp").replace(directory / "pit.index")


class PITStore:
    """Byte-bounded stock cache with automatic file/dictionary invalidation."""
    def __init__(self, root, *, cache_bytes=128 * 1024**2, missing="nan"):
        if not isinstance(cache_bytes, int) or cache_bytes < 0:
            raise ValueError("pit_cache_bytes must be a nonnegative integer")
        self.root, self.cache_bytes, self.missing = Path(root), cache_bytes, missing
        self.clear()

    def clear(self):
        self.cache = OrderedDict()
        self.cached_bytes = 0
        self._registry_signature = None
        self._registry = None

    @property
    def registry(self):
        signature = _signature(self.root / "fields.json")
        if signature != self._registry_signature:
            self._registry = read_registry(self.root)
            self._registry_signature = signature
            self.cache.clear()
            self.cached_bytes = 0
        return self._registry["fields"]

    def resolve(self, name):
        name = name.removeprefix("$$")
        fields = self.registry
        exact = [int(key) for key, meta in fields.items() if meta["name"] == name]
        matches = exact or [int(key) for key, meta in fields.items() if meta["name"].split(".")[-1] == name]
        if len(matches) != 1:
            raise KeyError(f"Unknown or ambiguous PIT field: {name}; use a name from fields('financial')")
        return matches[0]

    def stock(self, instrument):
        from .data import normalize_code
        instrument = normalize_code(instrument)
        registry = self.registry
        directory = self.root / instrument
        for alias in (instrument, instrument.lower(), instrument[-2:] + instrument[:6],
                      (instrument[-2:] + instrument[:6]).lower()):
            if (self.root / alias / "pit.data").exists():
                directory = self.root / alias
                break
        paths = [directory / "pit.data", directory / "pit.index"]
        if not any(path.exists() for path in paths):
            if self.missing == "raise":
                raise FileNotFoundError(paths[0])
            return StockPIT(np.empty(0, RECORD_DTYPE), np.empty(0, INDEX_DTYPE))
        signature = tuple(_signature(path) for path in paths)
        cached = self.cache.pop(instrument, None)
        if cached is not None:
            if cached[0] == signature:
                self.cache[instrument] = cached
                return cached[1]
            self.cached_bytes -= cached[1].nbytes
        stock = StockPIT.read(directory)
        if not set(map(int, stock.index["field_id"])).issubset(map(int, registry)):
            raise ValueError(f"PIT stock contains unknown field IDs: {instrument}")
        if stock.nbytes <= self.cache_bytes:
            while self.cache and self.cached_bytes + stock.nbytes > self.cache_bytes:
                self.cached_bytes -= self.cache.popitem(last=False)[1][1].nbytes
            self.cache[instrument] = signature, stock
            self.cached_bytes += stock.nbytes
        return stock
