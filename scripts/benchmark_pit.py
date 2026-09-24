"""Offline synthetic PIT benchmark; checks event output against a daily oracle."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlib.data import LocalProvider
from qlib.data.pit import register_fields, write_stock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=int, default=4)
    parser.add_argument("--fields", type=int, default=12)
    parser.add_argument("--quarters", type=int, default=32)
    args = parser.parse_args()
    if min(args.stocks, args.fields, args.quarters) < 1:
        parser.error("All sizes must be positive")
    random = np.random.default_rng(2026)
    periods = pd.period_range("2010Q1", periods=args.quarters, freq="Q")
    records = []
    for quarter in periods:
        for revision, delay in enumerate((35, 120)):
            date = int((quarter.end_time.normalize() + pd.Timedelta(days=delay)).strftime("%Y%m%d"))
            for field in range(args.fields):
                records.append((date, quarter.year * 100 + quarter.quarter,
                                f"synthetic.f{field}_q", random.normal(100, 20) + revision))
    rows = pd.DataFrame(records, columns=["date", "period", "field", "value"])
    dates = pd.bdate_range(periods[0].start_time, pd.Timestamp(str(rows.date.max())) + pd.Timedelta(days=5))
    codes = [f"{i + 1:06d}.SZ" for i in range(args.stocks)]
    expressions = ["P($$synthetic.f0_q)", "P(Mean($$synthetic.f0_q, 4))", "PRef($$synthetic.f0_q, -1)"]
    with TemporaryDirectory(prefix="qlib-pit-benchmark-") as temporary:
        root = Path(temporary)
        (root / "calendars").mkdir()
        (root / "instruments").mkdir()
        (root / "calendars/day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")), encoding="utf-8")
        (root / "instruments/all.txt").write_text(
            "\n".join(f"{code} {dates[0]:%Y-%m-%d} {dates[-1]:%Y-%m-%d}" for code in codes), encoding="utf-8")
        start = perf_counter()
        register_fields(root / "financial", {f"synthetic.f{i}_q": {"unit": "ratio"} for i in range(args.fields)})
        for code in codes:
            write_stock(root / "financial", code, rows)
        build = perf_counter() - start
        provider = LocalProvider(root)
        start = perf_counter()
        result = provider.features(codes, expressions, allow_future=False)
        cold = perf_counter() - start
        start = perf_counter()
        warm = provider.features(codes, expressions, allow_future=False)
        warm_time = perf_counter() - start
        pd.testing.assert_frame_equal(result, warm)
        # Deliberately independent, daily reconstruction reference for one stock.
        source = rows[rows.field == "synthetic.f0_q"].sort_values("date")
        expected = []
        start = perf_counter()
        for date in dates:
            visible = source[source.date <= int(date.strftime("%Y%m%d"))]
            snapshot = visible.drop_duplicates("period", keep="last").sort_values("period")
            expected.append(snapshot.value.tail(4).mean())
        reference_time = perf_counter() - start
        np.testing.assert_allclose(result.loc[codes[0], expressions[1]], expected, equal_nan=True)
        files = sum(path.is_file() for path in (root / "financial").rglob("*"))
        print(f"stocks={args.stocks}, fields={args.fields}, records={len(rows) * args.stocks}, trading_days={len(dates)}")
        print(f"final_files={files} (2 per stock + 1 dictionary), build={build:.3f}s")
        print(f"event_query_all_stocks_3_expressions: cold={cold:.3f}s, warm={warm_time:.3f}s")
        print(f"daily_dataframe_oracle_1_stock_1_expression={reference_time:.3f}s")
        print("Event results match the independent daily oracle.")


if __name__ == "__main__":
    main()
