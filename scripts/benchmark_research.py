"""Repeat real-data research workloads and compare complete research outputs.

Use separate processes and the same CPU/thread settings for each source tree.
"""

import argparse
import cProfile
from dataclasses import fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
import gc
import hashlib
import html as html_escape
import json
import os
from pathlib import Path
import pickle
import platform
import pstats
import statistics
import sys
from time import perf_counter, process_time


FACTORS = {
    "momentum20": "$close / Ref($close, 20) - 1",
    "reversal5": "1 - $close / Ref($close, 5)",
    "low_vol20": "-Std($close / Ref($close, 1) - 1, 20)",
    "volume_ratio20": "$volume / Mean($volume, 20)",
    "price_volume_corr20": "Corr($close, $volume, 20)",
    "range_position20": "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20))",
    "illiquidity20": "Mean(Abs($close / Ref($close, 1) - 1) / ($volume * $close), 20)",
    "trend20": "Slope($close, 20) / Mean($close, 20)",
}
DAILY_FIELDS = ["open", "high", "low", "close", "volume", "factor", "total_mv"]
MINING_GRID = {
    **{f"momentum{n}": f"$close / Ref($close, {n}) - 1" for n in (5, 10, 20, 60)},
    **{f"low_vol{n}": f"-Std($close / Ref($close, 1) - 1, {n})" for n in (5, 10, 20, 60)},
    **{f"volume_ratio{n}": f"$volume / Mean($volume, {n})" for n in (5, 10, 20, 60)},
}


def peak_rss_mb():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counter.PeakWorkingSetSize / 1024**2
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def affinity():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        return hex(process_mask.value)
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def tables(value, prefix=""):
    """Keep every research table and configuration, without import-bound objects."""
    if is_dataclass(value):
        return {f.name: tables(getattr(value, f.name), f.name) for f in dataclass_fields(value)}
    if isinstance(value, dict):
        return {key: tables(item, str(key)) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tables(item, prefix) for item in value)
    return value


def run(args):
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    # Set before numerical-library imports. Launch affinity must be set by the
    # caller when required; this records the actual inherited process setting.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import numpy as np
    import pandas as pd
    import scipy
    from loguru import logger
    import qlib
    from qlib.data import LocalProvider
    from qlib.data._libs import expanding, rolling, pit
    from qlib.contrib.report.analysis_model import (
        calculate_factors, calculate_forward_returns, winsorize_factors,
        standardize_factors, neutralize_factors, analyze_factors)
    from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return, calc_long_short_prec, pred_autocorr
    from qlib.contrib.strategy import TopkStrategy, TopkDropoutStrategy
    from qlib.backtest.backtest import backtest

    if not Path(qlib.__file__).resolve().is_relative_to(source):
        raise ValueError("The requested source tree was not imported")
    logger.remove()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    root = args.data_root.expanduser().resolve()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source_root": str(source),
        "data_root": str(root), "market": args.market, "start": args.start, "end": args.end,
        "adjust": args.adjust, "repeats": args.repeats, "profiled": args.profile,
        "threads": args.threads, "affinity": affinity(), "hardware": args.hardware,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "qlib": qlib.__version__, "qlib_import": qlib.__file__,
        "native": {"rolling": rolling.is_available(), "expanding": expanding.is_available(),
                   "pit": pit.LIBRARY_PATH.is_file()},
        "data_fingerprint": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (root / "calendars/day.txt", root / "financial/fields.json", root / "stock_basic.csv")},
        "source_fingerprint": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((source / "qlib").rglob("*"))
                               if p.is_file() and p.suffix in (".py", ".c", ".h", ".dll", ".so")},
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "factors": FACTORS, "mining_grid": MINING_GRID, "horizons": [1, 5, 20], "entry_lag": 1, "quantiles": 5,
        "timing_notes": "perf_counter wall time; CPU time; peak RSS is process lifetime high-water mark. "
                        "Provider-cold clears application caches, not the operating-system file cache. "
                        "Each repeat runs the same pipeline; input preparation for evaluation stages is reported separately.",
    }
    timings = []
    first_results = None
    total_begin = perf_counter()
    for repeat in range(args.repeats):
        provider = LocalProvider(root, adjust=args.adjust)
        dates = provider.calendar(args.start, args.end)
        if dates.empty:
            raise ValueError("The requested date range has no trading days")
        outputs = {}

        def measure(name, function):
            gc.collect()
            profiler = cProfile.Profile() if args.profile else None
            wall_begin, cpu_begin = perf_counter(), process_time()
            if profiler:
                profiler.enable()
            value = function()
            if profiler:
                profiler.disable()
            seconds, cpu_seconds = perf_counter() - wall_begin, process_time() - cpu_begin
            timings.append({"repeat": repeat + 1, "task": name, "seconds": seconds,
                            "cpu_seconds": cpu_seconds, "peak_rss_mb": peak_rss_mb(), "profiled": args.profile})
            print(f"repeat {repeat + 1}/{args.repeats} {name}: {seconds:.4f}s", flush=True)
            if profiler:
                profiler.dump_stats(str(destination / f"{name}.{repeat + 1}.prof"))
                with (destination / f"{name}.{repeat + 1}.profile.txt").open("w", encoding="utf-8") as stream:
                    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(35)
            outputs[name] = tables(value)
            save_json(destination / "timings.partial.json", {"metadata": metadata, "timings": timings})
            return value

        measure("universe", lambda: provider.universe(args.market, args.start, args.end))
        provider.clear_cache()
        raw = measure("daily_provider_cold", lambda: provider.daily(args.market, DAILY_FIELDS, args.start, args.end))
        measure("daily_provider_warm", lambda: provider.daily(args.market, DAILY_FIELDS, args.start, args.end))
        factors = measure("technical_factors", lambda: calculate_factors(args.market, FACTORS, args.start, args.end, provider=provider))
        codes = factors.index.get_level_values("instrument").unique().tolist()
        metadata.update(trading_days=len(dates), stocks=len(codes), observations=len(factors),
                        factor_cells=factors.size, pit_stocks=min(len(codes), args.pit_stocks))
        labels = measure("forward_returns", lambda: calculate_forward_returns(factors, (1, 5, 20), provider=provider))
        clipped = measure("winsorization", lambda: winsorize_factors(factors))
        neutral = measure("neutralization", lambda: neutralize_factors(clipped, provider=provider, min_samples=20))
        processed = measure("standardization", lambda: standardize_factors(neutral))
        pred, label = processed["momentum20"], labels[5]
        measure("ic_rank_ic", lambda: calc_ic(pred, label))
        measure("long_short", lambda: {"returns": calc_long_short_return(pred, label),
                                      "precision": calc_long_short_prec(pred.reorder_levels(["datetime", "instrument"]),
                                                                        label.reorder_levels(["datetime", "instrument"]))})
        measure("autocorrelation", lambda: pred_autocorr(pred))
        analysis = measure("factor_report", lambda: analyze_factors(processed, labels, quantiles=5))

        def mining():
            candidates = calculate_factors(args.market, MINING_GRID, args.start, args.end, provider=provider)
            # Pick the candidate using discovery dates only. Purge six signal
            # dates because entry_lag=1 and the evaluation horizon is five days.
            split = max(7, int(len(dates) * 0.75))
            discovery = dates[:max(0, split - 6)]
            validation = dates[split:]
            rows, daily_ic = [], {}
            for name in candidates:
                ic, rank_ic = calc_ic(candidates[name], labels[5])
                daily_ic[name] = rank_ic
                training = rank_ic.reindex(discovery).dropna()
                testing = rank_ic.reindex(validation).dropna()
                rows.append((name, training.mean(), testing.mean(), len(training), len(testing)))
            summary = pd.DataFrame(rows, columns=["factor", "discovery_rank_ic", "validation_rank_ic",
                                                   "discovery_dates", "validation_dates"]).set_index("factor")
            eligible = summary.discovery_rank_ic.dropna()
            selected = eligible.abs().sort_values(ascending=False, kind="stable").index[:1]
            selection = summary.loc[selected].copy()
            selection["direction"] = np.sign(selection.discovery_rank_ic)
            selection["signed_validation_rank_ic"] = selection.validation_rank_ic * selection.direction
            return {"candidates": candidates, "summary": summary, "daily_rank_ic": pd.DataFrame(daily_ic),
                    "discovery_selection": selection}

        mined = measure("factor_mining_grid", mining)

        def descriptive():
            returns = provider.features(args.market, ["$close / Ref($close, 1) - 1"], args.start, args.end, allow_future=False).iloc[:, 0]
            return {"daily_returns": returns.groupby(level="datetime").agg(["count", "mean", "std", "min", "max"]),
                    "factor_distribution": factors.describe(), "factor_correlation": factors.corr(),
                    "missing_fraction": raw.isna().mean()}

        measure("data_analysis", descriptive)
        exchange = {"delist_policy": "last_close"}
        topk = measure("backtest_topk", lambda: backtest(TopkStrategy(instruments=args.market, topk=20, rebalance=5),
                        args.start, args.end, provider=provider, exchange=exchange))
        external = factors["momentum20"]
        dropout = measure("backtest_dropout", lambda: backtest(TopkDropoutStrategy(topk=50, n_drop=5, instruments=args.market,
                        signal=external, random_seed=2026), args.start, args.end, provider=provider, exchange=exchange))

        def sweep():
            return {f"topk{k}_drop{drop}": backtest(TopkDropoutStrategy(topk=k, n_drop=drop, instruments=args.market,
                        signal=external, random_seed=2026), args.start, args.end, provider=provider, exchange=exchange)
                    for k, drop in ((20, 2), (50, 5), (100, 10))}

        measure("parameter_sweep", sweep)
        pit_codes = codes[:args.pit_stocks]
        measure("pit_snapshot", lambda: provider.financial(pit_codes, ["eps", "roe", "netprofit_yoy", "grossprofit_margin"], asof=args.end))
        pit_defs = {"eps": "P($$eps)", "roe": "P($$roe)", "previous_eps": "PRef($$eps, -1)"}
        measure("pit_factor_projection", lambda: calculate_factors(pit_codes, pit_defs, args.start, args.end, provider=provider))

        if repeat == 0:
            def export():
                # Keep the original CSV export workload comparable across
                # revisions; interactive HTML has a separate export cost.
                table_folder = destination / "factor_research"
                table_folder.mkdir(exist_ok=True)
                for table in ("factors", "forward_returns", "summary", "daily", "quantile_returns",
                              "quantile_membership", "turnover", "autocorrelation"):
                    getattr(analysis, table).to_csv(table_folder / f"{table}.csv")
                save_json(table_folder / "config.json", analysis.config)
                topk.save(destination / "backtest_topk")
                dropout.save(destination / "backtest_dropout")
                (destination / "factor_mining").mkdir(exist_ok=True)
                for name in ("summary", "daily_rank_ic", "discovery_selection"):
                    mined[name].to_csv(destination / "factor_mining" / f"{name}.csv")
                return {"files": len(list((destination / "factor_research").glob("*"))) +
                                 len(list((destination / "backtest_topk").glob("*"))) +
                                 len(list((destination / "backtest_dropout").glob("*"))) + 3}
            measure("report_export", export)
            first_results = outputs
            with (destination / "results.pkl").open("wb") as stream:
                pickle.dump(first_results, stream, protocol=pickle.HIGHEST_PROTOCOL)
            first_results = None
        del outputs, provider, raw, factors, labels, clipped, neutral, processed, analysis, topk, dropout, mined, pred, label, external
    frame = pd.DataFrame(timings)
    frame.to_csv(destination / "timings.csv", index=False)
    summary = frame.groupby("task", sort=False).seconds.agg(["median", "min", "max", "count"])
    summary.to_csv(destination / "timing_summary.csv")
    save_json(destination / "timings.json", {"metadata": metadata, "timings": timings,
                                             "total_elapsed_seconds": perf_counter() - total_begin, "complete": True})
    print(summary.to_string(), flush=True)


def compare(args):
    import numpy as np
    import pandas as pd
    left_root, right_root = args.baseline.resolve(), args.optimized.resolve()
    first = json.loads((left_root / "timings.json").read_text(encoding="utf-8"))
    second = json.loads((right_root / "timings.json").read_text(encoding="utf-8"))
    if not first.get("complete") or not second.get("complete"):
        raise ValueError("Both benchmark runs must finish before comparison")
    for key in ("data_fingerprint", "market", "start", "end", "adjust", "stocks", "observations",
                "trading_days", "pit_stocks", "factors", "mining_grid", "horizons", "entry_lag",
                "quantiles", "affinity", "threads", "native", "repeats", "hardware",
                "python", "numpy", "pandas", "scipy"):
        if first["metadata"][key] != second["metadata"][key]:
            raise ValueError(f"Incomparable benchmark setting: {key}")
    if first["metadata"]["profiled"] or second["metadata"]["profiled"]:
        raise ValueError("Profiled timings cannot be used for the speed comparison")
    with (left_root / "results.pkl").open("rb") as stream:
        left = pickle.load(stream)
    with (right_root / "results.pkl").open("rb") as stream:
        right = pickle.load(stream)
    compared = []

    def check(a, b, name):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b, rtol=1e-11, atol=1e-12,
                                          check_exact=name.endswith((".orders", ".trades", ".quantile_membership",
                                                                     ".technical_factors", ".daily_provider_cold",
                                                                     ".daily_provider_warm", ".factor_mining_grid.candidates")))
            compared.append({"table": name, "rows": len(a), "columns": len(a.columns)})
        elif isinstance(a, pd.Series):
            pd.testing.assert_series_equal(a, b, rtol=1e-11, atol=1e-12)
            compared.append({"table": name, "rows": len(a)})
        elif isinstance(a, dict):
            if list(a) != list(b):
                raise AssertionError(f"Keys differ: {name}")
            for key in a:
                check(a[key], b[key], f"{name}.{key}")
        elif isinstance(a, tuple):
            if len(a) != len(b):
                raise AssertionError(f"Tuple lengths differ: {name}")
            for index, (first, second) in enumerate(zip(a, b)):
                check(first, second, f"{name}.{index}")
        elif a != b:
            raise AssertionError(f"Values differ: {name}: {a!r} != {b!r}")

    check(left, right, "research")
    a = pd.DataFrame(first["timings"]).groupby("task", sort=False).seconds.median()
    b = pd.DataFrame(second["timings"]).groupby("task", sort=False).seconds.median()
    comparison = pd.DataFrame({"baseline_seconds": a, "optimized_seconds": b, "speedup": a / b})
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(destination / "comparison.csv")
    save_json(destination / "equivalence.json", {"passed": True, "tables": compared,
                                                "settings": first["metadata"], "baseline": str(left_root), "optimized": str(right_root)})
    html = "<meta charset='utf-8'><title>Qlib research performance</title><style>body{font:15px system-ui;max-width:1100px;margin:40px auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}h1{font-size:26px}</style>"
    html += "<h1>量化研究速度对比</h1>"
    html += f"<p>{first['metadata']['market']} · {first['metadata']['start']} 至 {first['metadata']['end']} · {first['metadata']['stocks']}只股票 · {first['metadata']['trading_days']}个交易日</p>"
    html += f"<p>首轮完整研究结果对照通过。数据、Python/数值库版本、线程、CPU亲和性与原生内核配置一致；计算任务重复{first['metadata']['repeats']}次，表中为非profile运行中位数，秒；导出一次。provider-cold只清应用缓存，未清操作系统缓存。</p>"
    html += comparison.to_html(float_format=lambda x: f"{x:.3f}")
    html += f"<p>对照 {len(compared)} 张表；日线、原始技术因子/候选、订单、成交和分组成员精确相同，其余数值容差 rtol=1e-11, atol=1e-12。</p>"
    (destination / "comparison.html").write_text(html, encoding="utf-8")
    print(comparison.to_string(), flush=True)
    print(f"Full research equivalence passed: {len(compared)} tables", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    runner = subparsers.add_parser("run")
    runner.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    runner.add_argument("--data-root", type=Path, default=Path("~/.qlib/qlib_data/cn_data"))
    runner.add_argument("--market", default="csi300")
    runner.add_argument("--start", default="2025-01-01")
    runner.add_argument("--end", default="2025-12-31")
    runner.add_argument("--adjust", choices=("hfq", "qfq", "none"), default="hfq")
    runner.add_argument("--pit-stocks", type=int, default=100)
    runner.add_argument("--repeats", type=int, default=3)
    runner.add_argument("--threads", type=int, default=1)
    runner.add_argument("--hardware", default=platform.processor())
    runner.add_argument("--profile", action="store_true")
    runner.add_argument("--output", type=Path, required=True)
    comparator = subparsers.add_parser("compare")
    comparator.add_argument("--baseline", type=Path, required=True)
    comparator.add_argument("--optimized", type=Path, required=True)
    comparator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if min(args.pit_stocks, args.repeats, args.threads) < 1:
            parser.error("pit-stocks, repeats and threads must be positive")
        run(args)
    else:
        destination = args.output.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        save_json(destination / "equivalence.json", {"passed": False, "status": "running"})
        try:
            compare(args)
        except Exception as error:
            save_json(destination / "equivalence.json", {"passed": False, "status": "failed",
                                                        "error": str(error), "error_type": type(error).__name__})
            (destination / "comparison.html").write_text(
                "<meta charset='utf-8'><title>Comparison failed</title>"
                "<h1>研究结果对照失败，未确认加速比</h1><pre>" + html_escape.escape(str(error)) + "</pre>",
                encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
