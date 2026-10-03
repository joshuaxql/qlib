"""Verify a wheel's C libraries against independent numerical references.

Run after installing a platform wheel:
    python -I scripts/verify_native.py --require-installed

Isolated mode prevents accidentally importing an in-tree Python package. The
additional flag rejects source-tree imports explicitly and is used in wheel CI.
"""

import argparse
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

import qlib
from qlib.data import ops
from qlib.data._libs import expanding, pit, rolling


KINDS = ("Mean", "Slope", "Rsquare", "Resi", "Corr", "Cov")


def reference(values, n, kind, left=None):
    """Fit each window independently, without Qlib's C or Python algorithms."""
    result = np.full(len(values), np.nan)
    for end in range(len(values)):
        start = max(0, end + 1 - n) if n else 0
        y = values[start:end + 1]
        x = np.arange(len(y), dtype=float) if left is None else left[start:end + 1]
        valid = np.isfinite(x) & np.isfinite(y)
        if kind == "Mean":
            if valid.any():
                result[end] = np.mean(y[valid])
            continue
        if valid.sum() < 2:
            continue
        xv, yv = x[valid], y[valid]
        if kind == "Cov":
            result[end] = np.cov(xv, yv, ddof=1)[0, 1]
        elif kind == "Corr":
            if np.ptp(xv) and np.ptp(yv):
                result[end] = np.corrcoef(xv, yv)[0, 1]
        else:
            slope, intercept = np.linalg.lstsq(np.column_stack([xv, np.ones(len(xv))]), yv, rcond=None)[0]
            if kind == "Slope":
                result[end] = slope
            elif kind == "Resi":
                result[end] = y[-1] - (slope * x[-1] + intercept)
            elif np.ptp(yv):
                result[end] = 1 - np.square(yv - slope * xv - intercept).sum() / np.square(yv - yv.mean()).sum()
    return result


def verify(require_installed=False):
    package_path = Path(qlib.__file__).resolve()
    source_package = Path(__file__).resolve().parents[1] / "qlib"
    if require_installed and source_package.resolve() in package_path.parents:
        raise RuntimeError(f"Expected an installed wheel, but imported the source tree: {package_path}")
    for module in (rolling, expanding, pit):
        if not module.is_available():
            raise RuntimeError(f"Native library is missing: {module.LIBRARY_PATH}")
        module._library()  # Wrong architecture and missing runtime dependencies fail here.
    for module in (rolling, expanding):
        for kind in KINDS:
            if not module.supports(kind):
                raise RuntimeError(f"Native {module.__name__} {kind} kernel is missing")

    random = np.random.default_rng(20261003)
    values = random.normal(size=80)
    left = random.normal(size=80)
    values[:4] = np.nan
    values[random.random(len(values)) < 0.2] = np.nan
    left[random.random(len(left)) < 0.2] = np.nan
    series = pd.Series(values, name="close", index=pd.bdate_range("2024-01-01", periods=len(values)))
    left_series = pd.Series(left, name="volume", index=series.index)
    checks = 0
    for n in (0, 1, 3, 17, 100):
        module = expanding if n == 0 else rolling
        for kind in KINDS:
            paired = kind in ("Corr", "Cov")
            expected = reference(values, n, kind, left if paired else None)
            args = (left, values) if paired else (values,)
            if n:
                args += (n,)
            function = getattr(module, f"{'expanding' if n == 0 else 'rolling'}_{kind.lower()}")
            np.testing.assert_allclose(function(*args), expected, rtol=1e-8, atol=1e-10, equal_nan=True)
            with patch.object(ops, "_stable_rolling", side_effect=AssertionError("Expression fell back to Python")):
                actual = ops.OPERATORS[kind](left_series, series, n) if paired else ops.OPERATORS[kind](series, n)
            pd.testing.assert_series_equal(actual, pd.Series(expected, name=series.name, index=series.index),
                                           rtol=1e-8, atol=1e-10)
            checks += 2

    dates = [2, 5, 5, 9, 20240103, 20240312, 20240312, 20240401, 0xFFFFFFFF]
    starts, counts = [0, 4, 8, 9], [4, 4, 1, 0]
    for asof in (0, 5, 20240312, 0xFFFFFFFF):
        expected = []
        for start, count in zip(starts, counts):
            matches = [i for i in range(start, start + count) if dates[i] <= asof]
            expected.append(matches[-1] if matches else -1)
        with patch.object(pit.np, "searchsorted", side_effect=AssertionError("PIT lookup fell back to Python")):
            np.testing.assert_array_equal(pit.asof_indices(dates, starts, counts, asof), expected)
        checks += 1
    return {
        "version": qlib.__version__, "python": platform.python_version(), "platform": sys.platform,
        "machine": platform.machine(), "package": str(package_path),
        "libraries": {name: str(module.LIBRARY_PATH.resolve())
                      for name, module in (("rolling", rolling), ("expanding", expanding), ("pit", pit))},
        "native_kernels": 13, "numerical_checks": checks, "python_fallback": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-installed", "--expect-installed", action="store_true",
                        help="Fail if the imported package is the source tree beside this script")
    args = parser.parse_args()
    print(json.dumps(verify(args.require_installed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
