"""Factor performance reports.

The local batch report API builds on contrib.eva.alpha's evaluation functions.
Portfolio returns here are descriptive, not executions.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from numbers import Real

import numpy as np
import pandas as pd
from loguru import logger

from qlib.data import D
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return, pred_autocorr


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _panel(frame, name):
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(frame.name if frame.name is not None else name)
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.MultiIndex):
        raise ValueError(f"{name} must have an (instrument, datetime) MultiIndex")
    if frame.index.nlevels != 2 or set(frame.index.names) != {"instrument", "datetime"}:
        raise ValueError(f"{name} index levels must be named instrument and datetime")
    frame = frame.reorder_levels(["instrument", "datetime"]).copy()
    if frame.index.has_duplicates or frame.index.to_frame(index=False).isna().any().any():
        raise ValueError(f"{name} index must be unique and non-null")
    if not isinstance(frame.index.get_level_values("datetime"), pd.DatetimeIndex):
        raise ValueError(f"{name} datetime level must contain pandas timestamps")
    if not len(frame.columns) or frame.columns.has_duplicates:
        raise ValueError(f"{name} columns must be nonempty and unique")
    if any(not isinstance(code, str) for code in frame.index.get_level_values("instrument")):
        raise ValueError("instrument labels must be strings")
    return frame.astype(float).replace([np.inf, -np.inf], np.nan).sort_index()


def calculate_factors(instruments, factors, start_time=None, end_time=None, *, provider=None, adjust=None):
    """Calculate named expressions with future Ref/Delta disabled.

    factors accepts an expression, a sequence, or {name: expression}. Adjustment
    inherits the provider default unless explicitly supplied. qfq factors use
    the anchor observable on each signal date, keeping historical factors causal.
    """
    provider = D if provider is None else provider
    if isinstance(factors, str):
        factors = [factors]
    definitions = dict(factors) if isinstance(factors, Mapping) else {expr: expr for expr in factors}
    if not definitions or any(not isinstance(name, str) or not name for name in definitions):
        raise ValueError("factors must contain nonempty string names")
    expressions = list(dict.fromkeys(definitions.values()))
    if hasattr(provider, "_signal_view"):
        provider = provider._signal_view()
    data = provider.features(instruments, expressions, start_time, end_time, allow_future=False, adjust=adjust)
    return pd.DataFrame({name: data[expr] for name, expr in definitions.items()}, index=data.index)


def winsorize_factors(factors, *, method="std", n=3.0, mad_scale=1.4826, ddof=0):
    """Clip each date/factor cross-section using std or median absolute deviation.

    Args:
        factors: Series or DataFrame with instrument/datetime MultiIndex.
        method: 'std' (default) clips to mean +/- n * std; 'mad' clips to median +/-
            n * mad_scale * median(abs(x - median(x))). Values are capped,
            not dropped. Statistics use only this date/column's finite inputs.
        n: Positive finite clipping multiplier, default 3.
        mad_scale: Positive MAD multiplier, default 1.4826 (normal-consistent
            scale); set to 1 to use unscaled MAD. Only used by method='mad'.
        ddof: Nonnegative standard-deviation degrees-of-freedom correction,
            default 0 (population). Only used by method='std'.

    Returns:
        DataFrame with original columns and sorted (instrument, datetime) index.
        NaN/inf inputs remain NaN. A zero MAD collapses valid values to the
        median. For std, count <= ddof gives NaN, and constants stay constant.
        The input is not modified. No market data or future labels are read.
    """
    if method not in ("std", "mad"):
        raise ValueError("method must be std or mad")
    n = _positive_number(n, "n")
    mad_scale = _positive_number(mad_scale, "mad_scale")
    ddof = _positive_int(ddof, "ddof", 0)
    values = _panel(factors, "factor")
    data = values.to_numpy()
    output = np.full(data.shape, np.nan)
    for positions in values.groupby(level="datetime", sort=False).indices.values():
        for column in range(data.shape[1]):
            rows = positions[np.isfinite(data[positions, column])]
            if not len(rows) or (method == "std" and len(rows) <= ddof):
                continue
            original = data[rows, column]
            # Binary scaling keeps representable differences. A median anchor
            # preserves ordinary observations even if the first row is an outlier.
            exponent = int(np.frexp(np.max(np.abs(original)))[1])
            sample = np.ldexp(original, -exponent)
            anchor = np.median(sample)
            sample = sample - anchor
            if method == "std":
                center, spread = sample.mean(), sample.std(ddof=ddof)
            else:
                center = np.median(sample)
                spread = np.median(np.abs(sample - center))
            with np.errstate(over="ignore"):
                width = n * spread * (mad_scale if method == "mad" else 1.0)
            lower, upper = center - width, center + width
            clipped = (sample < lower) | (sample > upper)
            output[rows, column] = original
            output[rows[clipped], column] = np.ldexp(np.clip(sample[clipped], lower, upper) + anchor, exponent)
    return pd.DataFrame(output, index=values.index, columns=values.columns)


def standardize_factors(factors, *, ddof=0):
    """Return per-date, per-factor Z-scores: (x - mean) / std.

    Accept a Series or DataFrame with instrument/datetime MultiIndex and return
    a DataFrame with the original columns and sorted canonical index. Finite
    values alone define each cross-section; NaN/inf remain NaN. ddof defaults
    to 0 (population variance), and must be a nonnegative integer. When count
    <= ddof the cross-section is NaN; otherwise constant samples map to zero.
    No filling, clipping, market-data lookup or future-label filtering occurs.
    """
    ddof = _positive_int(ddof, "ddof", 0)
    values = _panel(factors, "factor")
    data = values.to_numpy()
    output = np.full(data.shape, np.nan)
    for positions in values.groupby(level="datetime", sort=False).indices.values():
        for column in range(data.shape[1]):
            rows = positions[np.isfinite(data[positions, column])]
            if len(rows) <= ddof:
                continue
            sample = data[rows, column]
            if np.all(sample == sample[0]):
                output[rows, column] = 0.0
                continue
            exponent = int(np.frexp(np.max(np.abs(sample)))[1])
            sample = np.ldexp(sample, -exponent)
            sample = sample - sample[0]
            centered = sample - sample.mean()
            spread = sample.std(ddof=ddof)
            if spread > 0:
                output[rows, column] = centered / spread
    return pd.DataFrame(output, index=values.index, columns=values.columns)


def preprocess_factors(factors, *, provider=None, winsorize="std", winsorize_n=3.0,
                       mad_scale=1.4826, neutralize=True, market_cap="total_mv",
                       neutralize_min_samples=3, standardize=True, ddof=0):
    """Apply optional winsorization -> industry/size neutralization -> Z-score.

    Defaults apply mean +/- 3 std clipping, industry/log-total-cap neutralization,
    then Z-score standardization. Set winsorize=None, neutralize=False and/or
    standardize=False to disable individual steps. winsorize may also be 'mad'.
    winsorize_n/mad_scale configure winsorize_factors;
    market_cap/neutralize_min_samples configure neutralize_factors; ddof controls
    std-based winsorization and standardization. provider is only needed for
    neutralization. The result is a canonical DataFrame; input is never mutated.
    Only supplied factor rows determine statistics; do not include return labels.
    """
    _preprocessing_config(winsorize, winsorize_n, mad_scale, neutralize, standardize, ddof)
    values = _panel(factors, "factor")
    if winsorize is not None:
        values = winsorize_factors(values, method=winsorize, n=winsorize_n, mad_scale=mad_scale, ddof=ddof)
    if neutralize:
        values = neutralize_factors(values, provider=provider, market_cap=market_cap,
                                    min_samples=neutralize_min_samples)
    if standardize:
        values = standardize_factors(values, ddof=ddof)
    return values


def _preprocessing_config(winsorize, winsorize_n, mad_scale, neutralize, standardize, ddof):
    """Validate processing switches and produce JSON-serializable settings."""
    if winsorize not in (None, "std", "mad"):
        raise ValueError("winsorize must be None, std or mad")
    if not isinstance(neutralize, bool) or not isinstance(standardize, bool):
        raise ValueError("neutralize and standardize must be bools")
    n = _positive_number(winsorize_n, "winsorize_n")
    scale = _positive_number(mad_scale, "mad_scale")
    ddof = _positive_int(ddof, "ddof", 0)
    return {
        "order": [name for name, enabled in (("winsorize", winsorize is not None),
                  ("neutralize", neutralize), ("standardize", standardize)) if enabled],
        "winsorization": ({"method": winsorize, "n": n,
                           **({"mad_scale": scale} if winsorize == "mad" else {"ddof": ddof})}
                          if winsorize is not None else None),
        "standardization": {"method": "zscore", "ddof": ddof} if standardize else None,
    }


def neutralize_factors(factors, *, provider=None, market_cap="total_mv", min_samples=3):
    """Return daily cross-sectional industry + log-market-cap OLS residuals.

    Args:
        factors: Numeric Series or DataFrame indexed by (instrument, datetime),
            using canonical provider stock codes. Each column is fitted separately
            on its own finite observations within the supplied universe.
        provider: Local data provider; None uses the global D provider.
        market_cap: Daily size field, either total_mv (default) or circ_mv.
        min_samples: Minimum valid stocks per date and factor, at least 2.

    Returns:
        A float DataFrame with the factor columns and sorted canonical index.
        Missing factors/industries and nonpositive or nonfinite caps remain NaN.
        A fit needs at least min_samples rows and positive residual degrees of
        freedom (sample count > design rank); otherwise its output is NaN.

    Industry membership comes from historical industry/* intervals on the factor
    date. No current stock_basic snapshot, future return or forward fill is used.
    Multiple simultaneous industry memberships raise ValueError. Missing entire
    industry or market-cap sources raise rather than silently disabling a control.
    The design uses all observed industry dummies (which span an intercept) and
    centered/scaled natural-log cap. SVD least squares handles collinear controls.
    Output is unweighted residuals, without winsorization or standardization.
    """
    min_samples = _positive_int(min_samples, "min_samples", 2)
    if market_cap not in ("total_mv", "circ_mv"):
        raise ValueError("market_cap must be total_mv or circ_mv")
    values = _panel(factors, "factor")
    result = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    if values.empty:
        return result
    provider = D if provider is None else provider
    dates = values.index.get_level_values("datetime")
    if not dates.isin(provider.calendar()).all():
        raise ValueError("Factor dates must belong to the provider trading calendar")
    days = dates.unique().sort_values()
    codes = values.index.get_level_values("instrument").unique()
    date_positions = days.get_indexer(dates)
    code_positions = codes.get_indexer(values.index.get_level_values("instrument"))
    industries = provider.industries()
    if not industries:
        raise ValueError("Industry neutralization requires historical industry/* membership data")
    industry = np.full(len(values), -1, dtype=int)
    for number, name in enumerate(industries):
        mask = provider.universe(f"industry/{name}", days[0], days[-1])
        member = mask.reindex(index=days, columns=codes, fill_value=False).to_numpy(dtype=bool)[
            date_positions, code_positions]
        conflict = member & (industry >= 0)
        if conflict.any():
            example = values.index[np.flatnonzero(conflict)[0]]
            raise ValueError(f"Multiple historical industry memberships for {example}")
        industry[member] = number
    caps = provider.daily(codes.tolist(), [market_cap], days[0], days[-1], adjust="none")
    caps = caps[market_cap].reindex(values.index).to_numpy(dtype=float)
    eligible = (industry >= 0) & np.isfinite(caps) & (caps > 0)
    log_cap = np.full(len(values), np.nan)
    log_cap[eligible] = np.log(caps[eligible])
    data = values.to_numpy()
    output = np.full(data.shape, np.nan)
    for positions in values.groupby(level="datetime", sort=False).indices.values():
        for column in range(data.shape[1]):
            rows = positions[eligible[positions] & np.isfinite(data[positions, column])]
            if len(rows) < min_samples:
                continue
            _, groups = np.unique(industry[rows], return_inverse=True)
            design = np.eye(groups.max() + 1)[groups]
            size = log_cap[rows] - log_cap[rows].mean()
            scale = np.max(np.abs(size))
            if scale > 0:
                design = np.column_stack([design, size / scale])
            # Shift only the binary exponent, then subtract an anchor before
            # centering: this bounds moments while preserving small variation
            # around a large common offset without division-rounding noise.
            exponent = int(np.frexp(np.max(np.abs(data[rows, column])))[1])
            target = np.ldexp(data[rows, column], -exponent)
            target = target - target[0]
            target = target - target.mean()
            coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
            if len(rows) <= rank:
                continue
            residual = target - design @ coefficients
            # An exactly explained factor must not become a numerical-noise signal.
            tolerance = np.finfo(float).eps * max(design.shape) * np.linalg.norm(target)
            if np.linalg.norm(residual) <= tolerance:
                residual[:] = 0.0
            output[rows, column] = np.ldexp(residual, exponent)
    return pd.DataFrame(output, index=values.index, columns=values.columns)


def calculate_forward_returns(factors, horizons=(1, 5, 20), *, provider=None, price="open", entry_lag=1,
                              adjust=None):
    """Label signal t with price[t+entry_lag+h] / price[t+entry_lag] - 1.

    Shifts use the full provider trading calendar, never compressed per-stock
    valid observations. Prices must be positive and finite at both endpoints.
    Future prices beyond the last signal date are read when locally available.
    """
    provider = D if provider is None else provider
    factors = _panel(factors, "factor")
    horizons = list(horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be nonempty and unique")
    horizons = [_positive_int(h, "horizon") for h in horizons]
    entry_lag = _positive_int(entry_lag, "entry_lag", 0)
    if price not in ("open", "close", "vwap"):
        raise ValueError("price must be open, close or vwap")
    result = pd.DataFrame(np.nan, index=factors.index, columns=horizons)
    result.columns.name = "horizon"
    if factors.empty:
        return result
    calendar = provider.calendar()
    signal_dates = factors.index.get_level_values("datetime")
    if not signal_dates.isin(calendar).all():
        raise ValueError("Signal dates must belong to the provider trading calendar")
    left = calendar.searchsorted(signal_dates.min())
    right = min(len(calendar), calendar.searchsorted(signal_dates.max()) + entry_lag + max(horizons) + 1)
    dates = calendar[left:right]
    codes = factors.index.get_level_values("instrument").unique().tolist()
    # Explicit codes retain future prices even after an index/universe exit.
    raw = provider.daily(codes, [price], dates[0], dates[-1], adjust=adjust)
    prices = raw[price].unstack("instrument").reindex(index=dates, columns=codes)
    prices = prices.where(np.isfinite(prices) & prices.gt(0))
    entry = prices.shift(-entry_lag)
    for horizon in horizons:
        returns = prices.shift(-(entry_lag + horizon)).div(entry).sub(1)
        # Construct explicitly to retain missing endpoints and avoid stack's
        # version-dependent dropping of NaN rows.
        series = pd.concat({code: returns[code] for code in codes}, names=["instrument", "datetime"])
        result[horizon] = series.reindex(factors.index)
    return result.replace([np.inf, -np.inf], np.nan)


def _groups(values, quantiles):
    valid = values.dropna()
    result = pd.Series(np.nan, index=values.index)
    if valid.nunique() >= quantiles:
        # Equal values always stay together; groups may be uneven or empty.
        result.loc[valid.index] = np.floor((valid.rank(method="average") - 1) * quantiles / len(valid)) + 1
    return result


@dataclass
class FactorAnalysisResult:
    factors: pd.DataFrame
    forward_returns: pd.DataFrame
    summary: pd.DataFrame
    daily: pd.DataFrame
    quantile_returns: pd.DataFrame
    quantile_membership: pd.DataFrame
    turnover: pd.DataFrame
    autocorrelation: pd.DataFrame
    config: dict

    def save(self, directory):
        """Export all diagnostic tables as CSV and settings as JSON."""
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("factors", "forward_returns", "summary", "daily", "quantile_returns",
                     "quantile_membership", "turnover", "autocorrelation"):
            getattr(self, name).to_csv(directory / f"{name}.csv")
        (directory / "config.json").write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("因子分析报告已保存：{}", directory)


def analyze_factors(factors, forward_returns, *, quantiles=5, min_samples=2, turnover_lag=1):
    """Analyze precomputed panels. Larger factor values always define the top group.

    IC is cross-sectional Pearson; RankIC is Pearson on average ranks. ICIR
    uses sample standard deviation and is NOT annualized. All moments aggregate
    valid dates equally. No forward filling, imputation or automatic sign flip.
    turnover_lag is measured in the ordered dates supplied in factors.
    """
    factors = _panel(factors, "factor")
    labels = _panel(forward_returns, "return").reindex(factors.index)
    quantiles = _positive_int(quantiles, "quantiles", 2)
    min_samples = _positive_int(min_samples, "min_samples", 2)
    turnover_lag = _positive_int(turnover_lag, "turnover_lag")
    if factors.empty:
        raise ValueError("No factor observations to analyze")
    if any(not isinstance(name, str) or not name for name in factors.columns):
        raise ValueError("Factor column names must be nonempty strings")
    for horizon in labels.columns:
        _positive_int(horizon, "return horizon")
    memberships = pd.DataFrame(np.nan, index=factors.index, columns=factors.columns)
    daily_rows, group_rows, turn_rows, auto_rows = [], [], [], []
    dates = factors.index.get_level_values("datetime").unique().sort_values()
    for name in factors:
        official_auto = pred_autocorr(factors[name], lag=turnover_lag)
        evaluations = {}
        for horizon in labels:
            ic, ric = calc_ic(factors[name], labels[horizon])
            spread, average = calc_long_short_return(factors[name], labels[horizon], quantile=1 / quantiles)
            evaluations[horizon] = ic, ric, spread, average
        previous = []
        for date in dates:
            values = factors[name].xs(date, level="datetime")
            groups = _groups(values, quantiles)
            idx = pd.MultiIndex.from_arrays([values.index, [date] * len(values)], names=factors.index.names)
            memberships.loc[idx, name] = groups.to_numpy()
            old_groups = previous[-turnover_lag] if len(previous) >= turnover_lag else None
            auto_rows.append((name, date, official_auto.loc[date]))
            for group in range(1, quantiles + 1):
                current = set(groups.index[groups == group])
                old = set() if old_groups is None else set(old_groups.index[old_groups == group])
                turnover = len(current - old) / len(current) if current and old else np.nan
                turn_rows.append((name, date, group, turnover))
            previous.append(groups)
            day_labels = labels.xs(date, level="datetime")
            for horizon in labels:
                returns = day_labels[horizon]
                paired = values.notna() & returns.notna()
                count = int(paired.sum())
                for group in range(1, quantiles + 1):
                    sample = returns[(groups == group) & paired]
                    group_rows.append((name, horizon, date, group, len(sample), sample.mean(), sample.std(ddof=1)))
                ic, ric, spread, average = evaluations[horizon]
                daily_rows.append((name, horizon, date, len(values), int(values.notna().sum()), count,
                                   values.notna().mean(), count / len(values),
                                   ic.loc[date] if count >= min_samples else np.nan,
                                   ric.loc[date] if count >= min_samples else np.nan,
                                   average.loc[date], spread.loc[date]))
    daily = pd.DataFrame(daily_rows, columns=["factor", "horizon", "datetime", "universe_count", "factor_count",
                                             "pair_count", "coverage", "pair_coverage", "ic", "rank_ic",
                                             "universe_return", "long_short_return"]).set_index(["factor", "horizon", "datetime"])
    grouped = pd.DataFrame(group_rows, columns=["factor", "horizon", "datetime", "quantile", "count", "mean", "std"])
    grouped = grouped.set_index(["factor", "horizon", "datetime", "quantile"])
    turnover = pd.DataFrame(turn_rows, columns=["factor", "datetime", "quantile", "turnover"]).set_index(
        ["factor", "datetime", "quantile"])
    autocorr = pd.DataFrame(auto_rows, columns=["factor", "datetime", "autocorrelation"]).set_index(["factor", "datetime"])
    summaries = []
    for (name, horizon), frame in daily.groupby(level=["factor", "horizon"], sort=False):
        row = {"factor": name, "horizon": horizon, "dates": len(frame), "coverage": frame.coverage.mean(),
               "pair_coverage": frame.pair_coverage.mean(), "mean_pair_count": frame.pair_count.mean()}
        for column in ("ic", "rank_ic"):
            values = frame[column].dropna()
            std = values.std(ddof=1)
            row.update({f"{column}_mean": values.mean(), f"{column}_std": std,
                        f"{column}_ir": np.divide(values.mean(), std),
                        f"{column}_positive_rate": values.gt(0).mean() if len(values) else np.nan,
                        f"{column}_count": len(values)})
        spread = frame.long_short_return.dropna()
        row.update({"long_short_mean": spread.mean(), "long_short_std": spread.std(ddof=1),
                    "long_short_positive_rate": spread.gt(0).mean() if len(spread) else np.nan,
                    "long_short_count": len(spread),
                    "autocorrelation": autocorr.loc[name].autocorrelation.mean(),
                    "top_turnover": turnover.xs((name, quantiles), level=("factor", "quantile")).turnover.mean(),
                    "bottom_turnover": turnover.xs((name, 1), level=("factor", "quantile")).turnover.mean()})
        summaries.append(row)
    summary = pd.DataFrame(summaries).set_index(["factor", "horizon"])
    config = {"quantiles": quantiles, "min_samples": min_samples, "turnover_lag": turnover_lag,
              "horizons": [int(h) for h in labels.columns], "ic_ir_annualized": False,
              "evaluation_api": "qlib.contrib.eva.alpha", "long_short_quantile": 1 / quantiles}
    return FactorAnalysisResult(factors, labels, summary, daily.sort_index(), grouped.sort_index(),
                                memberships, turnover.sort_index(), autocorr.sort_index(), config)


def factor_analysis(instruments, factors, start_time=None, end_time=None, *, provider=None,
                     horizons=(1, 5, 20), price="open", entry_lag=1, adjust=None,
                     quantiles=5, min_samples=2, turnover_lag=1, neutralize=True,
                     market_cap="total_mv", neutralize_min_samples=3, winsorize="std",
                     winsorize_n=3.0, mad_scale=1.4826, standardize=True, ddof=0):
    """Calculate/preprocess factors, build labels and return a complete report.

    neutralize=True applies daily industry + log-market-cap neutralize_factors
    before all diagnostics. market_cap and neutralize_min_samples configure that
    fit independently of min_samples, which controls IC reporting. The returned
    factors contain processed values; forward-return labels are unchanged.
    Optional steps run in order: winsorize ('std'/'mad'), neutralize, standardize
    (Z-score). All are on by default, with std clipping at n=3; see preprocess_factors.
    """
    processing = _preprocessing_config(winsorize, winsorize_n, mad_scale, neutralize, standardize, ddof)
    logger.info("因子分析开始：计算因子、预处理及收益标签")
    values = calculate_factors(instruments, factors, start_time, end_time, provider=provider, adjust=adjust)
    values = preprocess_factors(values, provider=provider, winsorize=winsorize, winsorize_n=winsorize_n,
                                mad_scale=mad_scale, neutralize=neutralize, market_cap=market_cap,
                                neutralize_min_samples=neutralize_min_samples, standardize=standardize, ddof=ddof)
    returns = calculate_forward_returns(values, horizons, provider=provider, price=price,
                                        entry_lag=entry_lag, adjust=adjust)
    result = analyze_factors(values, returns, quantiles=quantiles, min_samples=min_samples, turnover_lag=turnover_lag)
    result.config.update({"price": price, "entry_lag": int(entry_lag),
                           "adjust": (D if provider is None else provider).adjust if adjust is None else adjust})
    result.config["neutralization"] = ({"method": "industry_log_market_cap", "market_cap": market_cap,
                                        "min_samples": int(neutralize_min_samples)} if neutralize else None)
    result.config["preprocessing"] = processing
    logger.info("因子分析完成：{} 个因子，{} 个持有期，{} 条股票日期记录",
                len(result.factors.columns), len(result.forward_returns.columns), len(result.factors))
    return result
