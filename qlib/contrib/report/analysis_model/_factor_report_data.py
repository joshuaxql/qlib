"""Aggregate factor diagnostics for the interactive HTML report.

The payload contains daily diagnostics and distribution summaries, never the
underlying instrument/return panel. Calculations do not alter the result or use
future-return availability to select a stock or set a portfolio weight.
"""

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd
from scipy.special import ndtri, stdtr

def _json_safe(value):
    """Convert scalar/container values to strict JSON, retaining missingness."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return str(value)


def _mean(values):
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        return np.nan
    exponent = int(np.frexp(np.max(np.abs(sample)))[1])
    return float(np.ldexp(np.ldexp(sample, -exponent).mean(), exponent))


def _std(values):
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return np.nan
    if np.all(sample == sample[0]):
        return 0.0
    if np.all(sample == sample[0]):
        return 0.0
    exponent = int(np.frexp(np.max(np.abs(sample)))[1])
    with np.errstate(over="ignore"):
        return float(np.ldexp(np.ldexp(sample, -exponent).std(ddof=1), exponent))


def _cumulative(values):
    """Arithmetic valid-period sum; missing dates remain visible as gaps."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    with np.errstate(over="ignore", invalid="ignore"):
        total = np.cumsum(np.where(valid, values, 0.0))
    return np.where(valid, total, np.nan)


def _net_period_returns(values, commission, stamp_tax):
    """One complete round trip, with purchase commission inside the budget.

    The affine form is equivalent to (1+r)*(1-c-s)/(1+c)-1, retaining
    small returns without the cancellation in (1+r)-1. Unknown returns
    remain unknown; a complete loss remains exactly -1.
    """
    if (not math.isfinite(commission) or not math.isfinite(stamp_tax)
            or commission < 0 or stamp_tax < 0 or commission + stamp_tax >= 1):
        raise ValueError("Fees must be finite, nonnegative and total exit fees below 1")
    values = np.asarray(values, dtype=float)
    if commission == 0 and stamp_tax == 0:
        return values.copy()
    multiplier = (1 - commission - stamp_tax) / (1 + commission)
    cost = (2 * commission + stamp_tax) / (1 + commission)
    net = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    with np.errstate(over="ignore", invalid="ignore"):
        net[valid] = values[valid] * multiplier - cost
    net[values == -1] = -1.0
    # Even if tiny negative wealth rounds to -1, a gross return below -100%
    # must retain the existing invalid-return semantics in risk statistics.
    net[(values < -1) & (net >= -1)] = np.nextafter(-1.0, -np.inf)
    return net


def _net_quantile_payload(returns, group):
    """A fee scenario uses the same report units and missing-date semantics."""
    returns_bps = np.asarray(returns, dtype=float) * 10000
    sample_count = int(np.isfinite(returns_bps).sum())
    box, violin = _return_distribution(returns_bps)
    return {"quantile": group, "mean_return_bps": _mean(returns_bps),
            "standard_error_bps": _std(returns_bps) / np.sqrt(sample_count)
            if sample_count else np.nan,
            "box_bps": box, "violin": violin, "daily_bps": returns_bps,
            "cumulative_bps": _cumulative(returns_bps)}


def _net_sector_returns(sector, commission, stamp_tax):
    """Transform date-equal means without reselecting or recomputing sectors."""
    rows = []
    for item in sector["quantile_returns"]:
        groups = []
        for group in item["groups"]:
            values = _net_period_returns([group["mean_return_bps"] / 10000], commission, stamp_tax)
            groups.append({**group, "mean_return_bps": values[0] * 10000})
        rows.append({**item, "groups": groups})
    return {"quantile_returns": rows}


def _risk_metrics(returns, horizon):
    """Observed nonoverlapping group means; no zero fill or invalid-return clip."""
    sample = np.asarray(returns, dtype=float)
    sample = sample[np.isfinite(sample)]
    undefined = {name: np.nan for name in ("factor_return", "sharpe", "annualized_return", "max_drawdown")}
    if not len(sample) or np.any(sample < -1):
        return undefined
    periods_per_year = 252 / horizon
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        # Log wealth avoids premature product overflow. A -100% return makes
        # log wealth -Inf permanently, retaining zero wealth thereafter.
        log_wealth = np.cumsum(np.log1p(sample))
        log_peak = np.maximum.accumulate(np.r_[0.0, log_wealth])[1:]
        std = _std(sample)
        return {
            "factor_return": np.expm1(log_wealth[-1]),
            "sharpe": _mean(sample) / std * np.sqrt(periods_per_year) if std > 0 else np.nan,
            "annualized_return": np.expm1(log_wealth[-1] * periods_per_year / len(sample)),
            "max_drawdown": float(np.max(-np.expm1(log_wealth - log_peak))),
        }


def _performance_metrics(ic, rank_ic, group_returns, sampled, horizon):
    """Four return diagnostics and ten correlation/group-order statistics."""
    ic, rank_ic = np.asarray(ic, dtype=float), np.asarray(rank_ic, dtype=float)
    ic = ic[np.isfinite(ic)]
    rank_ic = rank_ic[np.isfinite(rank_ic)]
    ic_mean, rank_mean = _mean(ic), _mean(rank_ic)
    ic_std, rank_std = _std(ic), _std(rank_ic)
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    rank_ir = rank_mean / rank_std if rank_std > 0 else np.nan
    t_stat = ic_ir * np.sqrt(len(ic)) if ic_std > 0 else np.nan
    p_value = 2 * stdtr(len(ic) - 1, -abs(t_stat)) if ic_std > 0 else np.nan
    means = np.array([_mean(returns) for returns in group_returns])
    means = means[np.isfinite(means)]
    monotonicity = np.nan
    if len(means) >= 2:
        ranks = pd.Series(means).rank(method="average").to_numpy()
        if np.any(ranks != ranks[0]):
            # Spearman ranks group order too, including when groups are absent.
            monotonicity = float(np.corrcoef(np.arange(len(ranks)), ranks)[0, 1])
    return {
        **_risk_metrics(np.asarray(group_returns[-1])[sampled], horizon),
        "ic_mean": ic_mean, "rank_ic_mean": rank_mean, "ic_std": ic_std,
        "ic_ir": ic_ir, "rank_ic_ir": rank_ir,
        "ic_negative_rate": float(np.mean(ic < -.02)) if len(ic) else np.nan,
        "ic_positive_rate": float(np.mean(ic > .02)) if len(ic) else np.nan,
        "ic_t_stat": t_stat, "ic_p_value": p_value, "monotonicity": monotonicity,
    }


def _rolling(values):
    return pd.Series(values, dtype=float).rolling(20, min_periods=5).mean().to_numpy()


def _distribution(values, dates):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    histogram, qq = [], []
    if len(finite):
        bins = min(30, max(6, int(np.sqrt(len(finite)))))
        low, high = finite.min(), finite.max()
        # Nearly constant ICs may differ by only one or two representable
        # floats. Integer-bin numpy.histogram rejects duplicated edges; retain
        # the observations and use only distinct, representable boundaries.
        edges = np.unique(np.linspace(low, high, bins + 1)) if low < high else bins
        counts, edges = np.histogram(finite, bins=edges)
        histogram = [{"center": (left + right) / 2, "count": int(count), "width": right - left}
                     for count, left, right in zip(counts, edges[:-1], edges[1:])]
        qq = np.column_stack((ndtri((np.arange(len(finite)) + 0.5) / len(finite)), np.sort(finite))).tolist()
    years = sorted(set(int(year) for year in dates.year))
    months = list(range(1, 13))
    year_ids = {year: number for number, year in enumerate(years)}
    monthly = []
    for year in years:
        for month in months:
            monthly.append([month - 1, year_ids[year], _mean(values[(dates.year == year) & (dates.month == month)])])
    return {"histogram": histogram, "qq": qq,
            "monthly": {"years": years, "months": months, "values": monthly}}


def _return_distribution(values):
    """Box and Gaussian KDE of *daily group means*, expressed in basis points."""
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        return None, []
    box = np.quantile(sample, [0, .25, .5, .75, 1]).tolist()
    std = _std(sample)
    if len(sample) < 2 or not np.isfinite(std) or std == 0:
        return box, []
    bandwidth = std * len(sample) ** (-1 / 5)
    grid = np.linspace(sample.min() - 3 * bandwidth, sample.max() + 3 * bandwidth, 64)
    with np.errstate(over="ignore", under="ignore"):
        density = np.exp(-.5 * ((grid[:, None] - sample[None, :]) / bandwidth) ** 2).mean(axis=1)
        density /= bandwidth * np.sqrt(2 * np.pi)
    return box, np.column_stack((grid, density)).tolist()


def _aligned_industries(index, industries):
    if industries is None:
        return None
    if isinstance(industries, pd.Series):
        if not isinstance(industries.index, pd.MultiIndex) or industries.index.nlevels != 2:
            raise ValueError("industries Series must have an (instrument, datetime) MultiIndex")
        if set(industries.index.names) != {"instrument", "datetime"}:
            raise ValueError("industries index levels must be named instrument and datetime")
        industries = industries.reorder_levels(["instrument", "datetime"])
        if industries.index.has_duplicates or industries.index.to_frame(index=False).isna().any().any():
            raise ValueError("industries index must be unique and non-null")
        if not isinstance(industries.index.get_level_values("datetime"), pd.DatetimeIndex):
            raise ValueError("industries datetime level must contain pandas timestamps")
        values = industries.reindex(index).to_numpy(dtype=object)
    elif isinstance(industries, Mapping):
        values = np.array([industries.get(code) for code in index.get_level_values("instrument")], dtype=object)
    else:
        raise TypeError("industries must be an aligned Series or instrument-to-sector mapping")
    return np.array([None if pd.isna(value) else str(value) for value in values], dtype=object)


def _historical_industries(result, provider):
    """Read historical membership once, without consulting stock_basic snapshots."""
    if not hasattr(provider, "industries"):
        return None
    names = provider.industries()
    if not names:
        return None
    index = result.factors.index
    dates = index.get_level_values("datetime")
    days = dates.unique().sort_values()
    codes = index.get_level_values("instrument").unique()
    day_positions = days.get_indexer(dates)
    code_positions = codes.get_indexer(index.get_level_values("instrument"))
    output = np.full(len(index), None, dtype=object)
    for name in names:
        mask = provider.universe(f"industry/{name}", days[0], days[-1])
        member = mask.reindex(index=days, columns=codes, fill_value=False).fillna(False).to_numpy(dtype=bool)[
            day_positions, code_positions]
        conflict = member & pd.notna(output)
        if conflict.any():
            example = index[np.flatnonzero(conflict)[0]]
            raise ValueError(f"Multiple historical industry memberships for {example}")
        output[member] = str(name)
    return pd.Series(output, index=index, name="industry", dtype=object)


def _group_ranks(values, group_ids):
    """Average ties within each group in one lexicographic sort."""
    if not len(values):
        return np.empty(0, dtype=float)
    order = np.lexsort((values, group_ids))
    sorted_ids, sorted_values = group_ids[order], values[order]
    positions = np.arange(len(values))
    group_change = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
    start = np.maximum.accumulate(np.where(group_change, positions, 0))
    ordinal = positions - start + 1
    tie_change = group_change | np.r_[False, sorted_values[1:] != sorted_values[:-1]]
    left = np.flatnonzero(tie_change)
    right = np.r_[left[1:], len(values)]
    average = (ordinal[left] + ordinal[right - 1]) / 2
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat(average, right - left)
    return ranks


def _group_center(values, group_ids, group_count, counts):
    """Bound and anchor each group before computing centered moments."""
    magnitudes = np.zeros(group_count)
    np.maximum.at(magnitudes, group_ids, np.abs(values))
    exponents = np.frexp(magnitudes)[1]
    scaled = np.ldexp(values, -exponents[group_ids])
    # Instrument order is fixed across the original panel. Use the first
    # paired value in each group as an anchor without materializing group lists.
    first = np.full(group_count, len(values), dtype=int)
    np.minimum.at(first, group_ids, np.arange(len(values)))
    anchors = np.zeros(group_count)
    observed = first < len(values)
    anchors[observed] = scaled[first[observed]]
    shifted = scaled - anchors[group_ids]
    means = np.bincount(group_ids, weights=shifted, minlength=group_count) / np.maximum(counts, 1)
    return shifted - means[group_ids]


def _group_correlations(left, right, group_ids, group_count, min_samples):
    counts = np.bincount(group_ids, minlength=group_count)
    output = np.full(group_count, np.nan)
    if not len(left):
        return output
    x = _group_center(left, group_ids, group_count, counts)
    y = _group_center(right, group_ids, group_count, counts)
    x2 = np.bincount(group_ids, weights=x * x, minlength=group_count)
    y2 = np.bincount(group_ids, weights=y * y, minlength=group_count)
    xy = np.bincount(group_ids, weights=x * y, minlength=group_count)
    valid = (counts >= min_samples) & (x2 > 0) & (y2 > 0)
    output[valid] = np.clip(xy[valid] / np.sqrt(x2[valid]) / np.sqrt(y2[valid]), -1, 1)
    return output


def _group_means(values, group_ids, group_count):
    counts = np.bincount(group_ids, minlength=group_count)
    output = np.full(group_count, np.nan)
    if len(values):
        magnitudes = np.zeros(group_count)
        np.maximum.at(magnitudes, group_ids, np.abs(values))
        exponents = np.frexp(magnitudes)[1]
        scaled = np.ldexp(values, -exponents[group_ids])
        sums = np.bincount(group_ids, weights=scaled, minlength=group_count)
        valid = counts > 0
        output[valid] = np.ldexp(sums[valid] / counts[valid], exponents[valid])
    return output


def _sector_payload(values, labels, memberships, sector_ids, date_ids, sector_names,
                    date_count, quantiles, min_samples):
    if not sector_names:
        return {"available": False, "overview": [], "quantile_returns": []}
    sector_count = len(sector_names)
    group_count = date_count * sector_count
    combined_ids = date_ids * sector_count + sector_ids
    paired = (sector_ids >= 0) & np.isfinite(values) & np.isfinite(labels)
    ids = combined_ids[paired]
    scores, returns = values[paired], labels[paired]
    ic = _group_correlations(scores, returns, ids, group_count, min_samples).reshape(date_count, sector_count)
    rank_ic = _group_correlations(_group_ranks(scores, ids), _group_ranks(returns, ids), ids,
                                  group_count, min_samples).reshape(date_count, sector_count)
    eligible = (sector_ids >= 0) & np.isfinite(labels) & np.isfinite(memberships)
    quantile_ids = combined_ids[eligible] * quantiles + memberships[eligible].astype(int) - 1
    daily_groups = _group_means(labels[eligible], quantile_ids, group_count * quantiles).reshape(
        date_count, sector_count, quantiles)
    overview, quantile_returns = [], []
    for sector, name in enumerate(sector_names):
        overview.append({"name": name, "ic_mean": _mean(ic[:, sector]), "rank_ic_mean": _mean(rank_ic[:, sector]),
                         "valid_dates": int(np.isfinite(ic[:, sector]).sum())})
        quantile_returns.append({"name": name, "groups": [
            {"quantile": group, "mean_return_bps": _mean(daily_groups[:, sector, group - 1]) * 10000,
             "count": int(np.isfinite(daily_groups[:, sector, group - 1]).sum())}
            for group in range(1, quantiles + 1)]})
    return {"available": True, "overview": overview, "quantile_returns": quantile_returns}


def _build_report_data(result, *, title=None, industries=None, calendar=None):
    """Build the stable strict-JSON schema consumed by the ECharts document."""
    values = result.factors
    dates = values.index.get_level_values("datetime").unique().sort_values()
    calendar_source = "report_dates" if calendar is None else "trading_calendar"
    calendar = dates if calendar is None else pd.DatetimeIndex(calendar)
    if calendar.hasnans or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Performance calendar must be unique, non-null and sorted")
    calendar_positions = calendar.get_indexer(dates)
    if np.any(calendar_positions < 0):
        raise ValueError("Performance calendar must contain every report date")
    if len(calendar_positions):
        calendar_positions = calendar_positions - calendar_positions[0]
    date_labels = dates.strftime("%Y-%m-%d").tolist()
    positions = [np.asarray(rows) for _, rows in sorted(values.groupby(level="datetime").indices.items())]
    labels = result.forward_returns.reindex(values.index)
    quantiles = int(result.config["quantiles"])
    horizons = list(labels.columns)
    membership = result.quantile_membership.reindex(values.index)
    sectors = _aligned_industries(values.index, industries)
    sector_names = []
    sector_ids = None
    date_ids = dates.get_indexer(values.index.get_level_values("datetime"))
    if sectors is not None:
        sector_names = sorted(set(item for item in sectors if item is not None))
        sector_ids = pd.Categorical(sectors, categories=sector_names).codes
    notes = [
        "收益图采用所选手续费情景下各持有期的收益，单位为基点（1 bp = 0.01%）；未按持有期折算每日收益。",
        "分组收益累计曲线为有效期间收益的算术累计，未知期收益仍为空；多日收益重叠，曲线不是可执行净值。",
        "IC/Rank IC 累计对有效报告日期的相关系数逐期算术相加，缺失日期仍显示为空。",
        "分组由原始全市场因子排名确定，行业图保留全市场分组；未来收益缺失不改变分组。",
        "收益均值按有效日期等权汇总；分组分布展示每日组均值，不是单只股票收益的分布。",
        "绩效使用最高分位组等权收益的现有统计：组内有效未来标签的均值，缺失标签不改变原始分组。",
        "绩效从首报告日锚定，按完整交易日历每隔一个持有期抽样；外部未提供日历时使用报告日期序列。缺失整期跳过，不当作零，也不重选抽样日期。",
        "因子收益为有效非重叠采样期间收益的复利总收益；年化收益和 Sharpe 按每年 252/持有期个有效期间折算。它们是已观测期间诊断，不是可执行净值。",
        "最大回撤包含起始净值 1，并以正数幅度表示；期间收益小于 -100% 时四项收益绩效为空，-100% 使净值归零且之后保持为零。",
        "IC/Rank IC 均值、样本标准差和 IR 使用有效日期，IR 不年化；IC 的双侧单样本 t 检验未做自相关/HAC 调整，重叠持有期可能使显著性偏乐观。",
        "IC 小于 -0.02、大于 0.02 的比例均使用严格不等号；单调性为分组序号与各组有效日期等权平均期间收益的 Spearman 相关，保留正负方向。",
        "IC/Rank IC 的滚动均值使用 20 个报告日期，最少 5 个有效观测；IR 未年化。",
        "所有非有限数值（NaN、正负无穷）导出为 null，图中显示为空；非有限 IR 不会显示成零。",
        "此报告是因子诊断，收益按所选费率扣费，未包含成交约束或资金管理。",
    ]
    meta = {"title": "因子分析报告" if title is None else str(title),
            "start": date_labels[0] if date_labels else None, "end": date_labels[-1] if date_labels else None,
            "trading_days": len(dates), "instruments": values.index.get_level_values("instrument").nunique(),
            "observations": len(values), "quantiles": quantiles, "horizons": [int(h) for h in horizons],
            "performance_quantile": quantiles, "performance_calendar": calendar_source,
            "performance_sampling_anchor": date_labels[0] if date_labels else None,
            "performance_sampling": "every_h_trading_days_from_first_report_date",
            "performance_annualization_trading_days": 252,
            "config": result.config.copy(), "notes": notes}
    meta["fee_options"] = [{"value": "none", "label": "无", "commission": 0, "stamp_tax": 0},
                           {"value": "commission_stamp", "label": "3‱佣金 + 1‰印花税",
                            "commission": .0003, "stamp_tax": .001}]
    meta["fee_model"] = ("每个持有期模拟完整买卖，买入预算包含佣金，卖出按成交金额扣佣金和印花税；"
                         "净期收益为 (1+r)*(1-c-s)/(1+c)-1。费率不按持有期天数折算，"
                         "缺失标签和整期收益保留缺失，IC、分组、覆盖率及原始分析表保持不变。")
    payload = {"meta": meta, "factors": []}
    label_data = labels.to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for name in values.columns:
            factor_values = values[name].to_numpy(dtype=float)
            groups = membership[name].to_numpy(dtype=float)
            summaries = []
            for horizon in horizons:
                summary = {field: result.summary.loc[(name, horizon), field] for field in
                           ("top_turnover", "bottom_turnover", "autocorrelation", "coverage", "pair_coverage", "dates")}
                summaries.append({"horizon": int(horizon), **summary})
            turn = result.turnover.loc[name]
            factor = {"name": name, "summary": summaries,
                      "turnover": {"dates": date_labels,
                          "top": turn.xs(quantiles, level="quantile").turnover.reindex(dates).to_numpy(),
                          "bottom": turn.xs(1, level="quantile").turnover.reindex(dates).to_numpy()},
                      "autocorrelation": {"dates": date_labels,
                          "values": result.autocorrelation.loc[name].autocorrelation.reindex(dates).to_numpy()},
                      "horizons": {}}
            for horizon_column, horizon in enumerate(horizons):
                daily = result.daily.loc[(name, horizon)].reindex(dates)
                grouped = result.quantile_returns.loc[(name, horizon)]
                quantile_payload = []
                group_period_returns = []
                for group in range(1, quantiles + 1):
                    group_frame = grouped.xs(group, level="quantile").reindex(dates)
                    period_returns = group_frame["mean"].to_numpy(dtype=float)
                    group_period_returns.append(period_returns)
                    returns = period_returns * 10000
                    sample_count = int(np.isfinite(returns).sum())
                    box, violin = _return_distribution(returns)
                    quantile_payload.append({"quantile": group, "mean_return_bps": _mean(returns),
                                             "standard_error_bps": _std(returns) / np.sqrt(sample_count)
                                             if sample_count else np.nan,
                                             "box_bps": box, "violin": violin, "daily_bps": returns,
                                             "cumulative_bps": _cumulative(returns)})
                horizon_payload = {"dates": date_labels, "daily": {
                    field: daily[field].to_numpy() for field in
                    ("ic", "rank_ic", "coverage", "pair_coverage", "universe_count", "factor_count", "pair_count")},
                    "quantiles": quantile_payload,
                    "performance": _performance_metrics(daily.ic.to_numpy(), daily.rank_ic.to_numpy(),
                                                         group_period_returns, calendar_positions % int(horizon) == 0,
                                                         int(horizon)),
                    "ic_distribution": _distribution(daily.ic.to_numpy(), dates),
                    "rank_ic_distribution": _distribution(daily.rank_ic.to_numpy(), dates),
                    "sector": _sector_payload(factor_values, label_data[:, horizon_column], groups, sector_ids,
                                              date_ids, sector_names, len(dates), quantiles,
                                              int(result.config["min_samples"]))}
                horizon_payload["daily"].update({"ic_rolling": _rolling(daily.ic.to_numpy()),
                                                 "rank_ic_rolling": _rolling(daily.rank_ic.to_numpy()),
                                                 "ic_cumulative": _cumulative(daily.ic.to_numpy()),
                                                 "rank_ic_cumulative": _cumulative(daily.rank_ic.to_numpy())})
                horizon_payload["daily"]["label_count"] = [int(np.isfinite(label_data[rows, horizon_column]).sum()) for rows in positions]
                horizon_payload["daily"]["missing_label_count"] = (daily.factor_count - daily.pair_count).to_numpy()
                commission, stamp_tax = .0003, .001
                net_returns = [_net_period_returns(period, commission, stamp_tax) for period in group_period_returns]
                net_performance = {**horizon_payload["performance"],
                                   **_risk_metrics(net_returns[-1][calendar_positions % int(horizon) == 0],
                                                   int(horizon))}
                horizon_payload["fee_scenarios"] = {"commission_stamp": {
                    "quantiles": [_net_quantile_payload(period, group)
                                  for group, period in enumerate(net_returns, 1)],
                    "performance": net_performance,
                    "sector": _net_sector_returns(horizon_payload["sector"], commission, stamp_tax)}}
                factor["horizons"][str(int(horizon))] = horizon_payload
            payload["factors"].append(factor)
    return _json_safe(payload)
