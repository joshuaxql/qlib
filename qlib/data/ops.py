"""Vector operators used by the expression evaluator.

Rolling windows use min_periods=1, N=0 means expanding, Ref(x, 0) means
the first observation. Negative Ref is allowed only in research/label mode.
"""

import operator

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
        return np.dot(values[valid], weights[valid]) / weights[valid].sum()
    return window(series, n).apply(weighted, raw=True)


def regression(values, kind):
    x = np.arange(len(values), dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    x, y = x[valid], values[valid]
    slope = np.dot(x - x.mean(), y - y.mean()) / np.square(x - x.mean()).sum()
    fitted = y.mean() + slope * (x - x.mean())
    if kind == "Slope":
        return slope
    if kind == "Resi":
        return values[-1] - (y.mean() + slope * (len(values) - 1 - x.mean()))
    variance = np.square(y - y.mean()).sum()
    return 1 - np.square(y - fitted).sum() / variance if variance else np.nan


def rolling_method(method):
    return lambda series, n: getattr(window(series, n), method)()


def native_rolling(series, n, kind):
    n = integer(n)
    backend = c_expanding if n == 0 else c_rolling
    if backend.is_available():
        values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
        # pandas rolling treats infinity as missing; preserve expression semantics.
        values[~np.isfinite(values)] = np.nan
        if n == 0:
            result = getattr(backend, f"expanding_{kind.lower()}")(values)
        else:
            result = getattr(backend, f"rolling_{kind.lower()}")(values, n)
        return pd.Series(result, index=series.index, name=series.name)
    if kind == "Mean":
        return window(series, n).mean()
    return window(series, n).apply(lambda values: regression(values, kind), raw=True)


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
    "And": operator.and_, "Or": operator.or_, "Not": operator.invert,
    "IsNull": pd.isna, "IsInf": np.isinf,
    "If": lambda condition, yes, no: np.where(condition, yes, no),
    "Clip": lambda series, low, high: series.clip(low, high),
    "Quantile": lambda series, n, q: window(series, n).quantile(q),
    "Corr": lambda left, right, n: window(left, n).corr(right),
    "Cov": lambda left, right, n: window(left, n).cov(right),
    "Rank": lambda series, n: window(series, n).rank(pct=True),
    "IdxMax": lambda series, n: window(series, n).apply(lambda x: np.nanargmax(x) + 1, raw=True),
    "IdxMin": lambda series, n: window(series, n).apply(lambda x: np.nanargmin(x) + 1, raw=True),
})
for _kind in ("Mean", "Slope", "Rsquare", "Resi"):
    OPERATORS[_kind] = lambda series, n, kind=_kind: native_rolling(series, n, kind)
