"""Vector operators used by the expression evaluator.

Rolling windows use min_periods=1, N=0 means expanding, Ref(x, 0) means
the first observation. Negative Ref is allowed only in research/label mode.
"""

import operator
import math

import numpy as np
import pandas as pd

from ._libs import expanding as c_expanding, rolling as c_rolling


def integer(value, name="window", minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def window(series, n):
    n = integer(n)
    return series.expanding(min_periods=1) if n == 0 else series.rolling(n, min_periods=1)


def Ref(series, n):
    integer(n, "offset", minimum=-2**31)
    return pd.Series(series.iloc[0], index=series.index) if n == 0 and len(series) else series.shift(n)


def Delta(series, n):
    return series - Ref(series, n)


def EMA(series, n):
    if isinstance(n, float) and 0 < n < 1:
        return series.ewm(alpha=n, min_periods=1).mean()
    return series.ewm(span=integer(n, minimum=1), min_periods=1).mean()


def WMA(series, n):
    def weighted(values):
        weights = np.arange(1, len(values) + 1)
        valid = np.isfinite(values)
        values, weights = values[valid], weights[valid]
        if not len(values):
            return np.nan
        exponent = math.frexp(float(np.max(np.abs(values))))[1] - 1
        scaled = np.ldexp(values, -exponent)
        # Center first as well as scaling: a constant, including float64's
        # largest finite value, remains exactly constant through the average.
        average = scaled[0] + np.dot(scaled - scaled[0], weights) / weights.sum()
        return _ldexp(float(average), exponent)
    return window(series, n).apply(weighted, raw=True)


def _ldexp(value, exponent):
    """Scale without premature overflow; genuine float64 overflow stays Inf."""
    try:
        return math.ldexp(value, exponent)
    except OverflowError:
        return math.copysign(np.inf, value)


def _observation(x, y, time_axis):
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    kx = 0 if time_axis else math.frexp(x)[1] - 1 if x else -1075
    ky = math.frexp(y)[1] - 1 if y else -1075
    return (1, kx, ky, math.ldexp(x, -kx), math.ldexp(y, -ky), 0.0, 0.0, 0.0, 0.0, 0.0)


def _combine(a, b):
    """Merge centered moments after exact binary scaling, without raw sums."""
    if a is None:
        return b
    if b is None:
        return a
    na, axk, ayk, ax, ay, amx, amy, axx, ayy, axy = a
    nb, bxk, byk, bx, by, bmx, bmy, bxx, byy, bxy = b
    n, kx, ky = na + nb, max(axk, bxk), max(ayk, byk)
    ax, ay = math.ldexp(ax, axk - kx), math.ldexp(ay, ayk - ky)
    bx, by = math.ldexp(bx, bxk - kx), math.ldexp(by, byk - ky)
    amx, amy = math.ldexp(amx, axk - kx), math.ldexp(amy, ayk - ky)
    bmx, bmy = math.ldexp(bmx, bxk - kx), math.ldexp(bmy, byk - ky)
    dx, dy = (bx - ax) + bmx - amx, (by - ay) + bmy - amy
    fraction = nb / n
    weight = na * fraction
    xx = math.ldexp(axx, 2 * (axk - kx)) + math.ldexp(bxx, 2 * (bxk - kx)) + dx * dx * weight
    yy = math.ldexp(ayy, 2 * (ayk - ky)) + math.ldexp(byy, 2 * (byk - ky)) + dy * dy * weight
    xy = math.ldexp(axy, axk + ayk - kx - ky) + math.ldexp(bxy, bxk + byk - kx - ky) + dx * dy * weight
    return n, kx, ky, ax, ay, amx + dx * fraction, amy + dy * fraction, xx, yy, xy


def _statistic(moments, kind, x, y):
    if moments is None:
        return np.nan
    n, kx, ky, ax, ay, mx, my, xx, yy, xy = moments
    if kind == "Mean":
        return _ldexp(ay + my, ky)
    if n < 2:
        return np.nan
    if kind == "Cov":
        return _ldexp(xy / (n - 1), kx + ky)
    if xx <= 0:
        return np.nan
    if kind == "Slope":
        return _ldexp(xy / xx, ky - kx)
    if kind == "Resi":
        if not math.isfinite(y):
            return np.nan
        residual = (math.ldexp(y, -ky) - ay - my) - xy / xx * (math.ldexp(x, -kx) - ax - mx)
        return _ldexp(residual, ky)
    if yy <= 0:
        return np.nan
    correlation = xy / math.sqrt(xx) / math.sqrt(yy)
    return correlation if kind == "Corr" else correlation * correlation


def _stable_rolling(values, n, kind, left=None):
    """O(len(values)) aggregate queue; expanding uses O(1) extra space.

    Cached front/back aggregates are built by merging, never subtracting old
    values. Once an outlier leaves, all its rounding effects leave with it.
    """
    output = np.empty(len(values), dtype=float)
    front, back, leaves, aggregate = [], [], [], None
    for i, y in enumerate(values):
        # Keep the scalar math path in Python float64 rather than alternating
        # between Python floats and NumPy scalar subclasses.
        y = float(y)
        x = float(i) if left is None else float(left[i])
        leaf = _observation(x, y, left is None)
        if n == 0:
            aggregate = _combine(aggregate, leaf)
        else:
            if i >= n:
                if not front:
                    for item in reversed(leaves):
                        front.append(_combine(item, front[-1] if front else None))
                    leaves.clear()
                    back.clear()
                front.pop()
            leaves.append(leaf)
            back.append(_combine(back[-1] if back else None, leaf))
            aggregate = _combine(front[-1] if front else None, back[-1])
        output[i] = _statistic(aggregate, kind, x, y)
    return output


def regression(values, kind):
    """One centered fit; constant values have an undefined R squared."""
    if not len(values):
        return np.nan
    return _stable_rolling(np.asarray(values, dtype=float), 0, kind)[-1]


def rolling_method(method):
    return lambda series, n: getattr(window(series, n), method)()


def native_rolling(series, n, kind, left=None):
    n = integer(n)
    backend = c_expanding if n == 0 else c_rolling
    if left is not None:
        left, series = left.align(series)
    values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
    values[~np.isfinite(values)] = np.nan
    if left is not None:
        left_values = left.to_numpy(dtype=float, na_value=np.nan, copy=True)
        left_values[~np.isfinite(left_values)] = np.nan
    if backend.supports(kind):
        # pandas rolling treats infinity as missing; preserve expression semantics.
        args = (values,) if left is None else (left_values, values)
        if n == 0:
            result = getattr(backend, f"expanding_{kind.lower()}")(*args)
        else:
            result = getattr(backend, f"rolling_{kind.lower()}")(*args, n)
    else:
        result = _stable_rolling(values, n, kind, left_values if left is not None else None)
    return pd.Series(result, index=series.index, name=series.name)


def If(condition, yes, no):
    """Preserve date/report indexes so conditional results remain composable."""
    series = [value for value in (condition, yes, no) if isinstance(value, pd.Series)]
    if not series:
        return np.where(condition, yes, no)
    index = series[0].index
    for value in series[1:]:
        index = index.union(value.index)
    values = [value.reindex(index) if isinstance(value, pd.Series) else value
              for value in (condition, yes, no)]
    name = yes.name if isinstance(yes, pd.Series) else no.name if isinstance(no, pd.Series) else None
    return pd.Series(np.where(*values), index=index, name=name)


OPERATORS = {
    name: rolling_method(method) for name, method in {
        "Mean": "mean", "Sum": "sum", "Std": "std", "Var": "var", "Max": "max",
        "Min": "min", "Med": "median", "Skew": "skew", "Kurt": "kurt", "Count": "count",
    }.items()
}
OPERATORS.update({
    "Ref": Ref, "Delta": Delta, "EMA": EMA, "WMA": WMA,
    "Abs": np.abs, "Sign": np.sign, "Log": np.log, "Exp": np.exp, "Sqrt": np.sqrt,
    "Power": operator.pow, "Add": operator.add, "Sub": operator.sub,
    "Mul": operator.mul, "Div": operator.truediv,
    "Greater": np.maximum, "Less": np.minimum,
    "Gt": operator.gt, "Ge": operator.ge, "Lt": operator.lt, "Le": operator.le,
    "Eq": operator.eq, "Ne": operator.ne,
    "And": operator.and_, "Or": operator.or_, "Not": np.logical_not,
    "IsNull": pd.isna, "IsInf": np.isinf,
    "If": If,
    "Clip": lambda series, low, high: series.clip(low, high),
    "Quantile": lambda series, n, q: window(series, n).quantile(q),
    "Corr": lambda left, right, n: native_rolling(right, n, "Corr", left),
    "Cov": lambda left, right, n: native_rolling(right, n, "Cov", left),
    "Rank": lambda series, n: window(series, n).rank(pct=True),
    "IdxMax": lambda series, n: window(series, n).apply(lambda x: np.nanargmax(x) + 1, raw=True),
    "IdxMin": lambda series, n: window(series, n).apply(lambda x: np.nanargmin(x) + 1, raw=True),
})
for _kind in ("Mean", "Slope", "Rsquare", "Resi"):
    OPERATORS[_kind] = lambda series, n, kind=_kind: native_rolling(series, n, kind)
