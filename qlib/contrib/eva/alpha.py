# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License (see LICENSE in this directory).
"""Cross-sectional alpha evaluation API.

Scalar evaluators align with pandas and evaluate date groups as NumPy arrays.
Batch evaluation uses joblib workers and returns dictionaries of result Series.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger


def _selected_labels(frame, quantile, largest):
    """Rank finite predictions only; missing labels do not alter the selection."""
    eligible = frame[np.isfinite(frame["pred"])]
    count = int(len(eligible) * quantile)
    selected = eligible.nlargest(count, "pred") if largest else eligible.nsmallest(count, "pred")
    return selected["label"]


def _pearson_corr(left, right):
    """Compute pairwise Pearson correlation without squaring the original scale."""
    left, right = left.align(right, join="inner")
    paired = left.notna() & right.notna()
    if paired.sum() < 2:
        return np.nan
    return _pearson_arrays(left[paired].to_numpy(dtype=float), right[paired].to_numpy(dtype=float))


def _pearson_arrays(left, right):
    """Pearson for already paired float arrays, preserving small differences."""
    if len(left) < 2:
        return np.nan
    vectors = []
    for values in (left, right):
        if not np.isfinite(values).all():
            return np.nan
        magnitude = np.max(np.abs(values))
        if magnitude == 0:
            return np.nan
        # Binary exponent shifts preserve the significand. Subtract an anchor
        # before taking the mean so a large common offset cannot erase variation.
        exponent = int(np.frexp(magnitude)[1])
        values = np.ldexp(values, -exponent)
        values = values - values[0]
        values = values - values.mean()
        norm = np.linalg.norm(values)
        if norm == 0:
            return np.nan
        vectors.append(values / norm)
    return float(np.clip(np.dot(*vectors), -1.0, 1.0))


def _array_correlations(pred, label):
    """Pair before ranking: each horizon can have a different valid subset.

    Infinity invalidates Pearson but is an ordered observation for Spearman,
    matching pandas Series.corr. Only missing values remove pairs here.
    """
    paired = ~np.isnan(pred) & ~np.isnan(label)
    pred, label = pred[paired], label[paired]
    if len(pred) < 2:
        return np.nan, np.nan
    ic = _pearson_arrays(pred, label)
    pred_rank, label_rank = _rank_values(pred), _rank_values(label)
    if np.all(pred_rank == pred_rank[0]) or np.all(label_rank == label_rank[0]):
        ric = np.nan
    else:
        # This is SciPy spearmanr's coefficient path, without computing its
        # unused p-value. Matching corrcoef also preserves perfect-rank ICIR.
        ric = float(np.corrcoef(pred_rank, label_rank)[0, 1])
    return ic, ric


def _rank_values(values):
    # Import only when ranks are requested, not for expression/label research.
    from scipy.stats import rankdata
    return rankdata(values)


def _float_values(frame):
    """View ordinary float blocks; nullable extension blocks need NA conversion."""
    if any(isinstance(dtype, pd.api.extensions.ExtensionDtype) for dtype in frame.dtypes):
        return frame.to_numpy(dtype=float, na_value=np.nan)
    return frame.to_numpy(dtype=float, copy=False)


def _selected_positions(values, quantile, largest):
    """Return positions with nlargest/nsmallest's stable boundary tie rule."""
    eligible = np.flatnonzero(np.isfinite(values))
    count = int(len(eligible) * quantile)
    if count <= 0:
        return np.empty(0, dtype=np.intp)
    scores = values[eligible]
    # Keep pandas' integer ordering (float conversion could collapse large
    # integers) and its special all-selected sorting path for compatibility.
    if values.dtype.kind != "f" or count >= len(eligible):
        series = pd.Series(scores, index=eligible)
        selected = series.nlargest(count) if largest else series.nsmallest(count)
        return selected.index.to_numpy(dtype=np.intp)
    key = -scores if largest else scores
    threshold = np.partition(key, count - 1)[count - 1]
    candidates = np.flatnonzero(key <= threshold)
    order = np.argsort(key[candidates], kind="stable")[:count]
    return eligible[candidates[order]]


def _nan_mean(values):
    """The same NaN-skipping sum/count used by Series.mean, without a Series."""
    valid = ~np.isnan(values)
    if not valid.any():
        return np.nan
    with np.errstate(invalid="ignore", over="ignore"):
        return np.where(valid, values, 0).sum() / valid.sum()


def _long_short_arrays(df, group, quantile, precision=False):
    """Numeric fast path; None retains pandas for unsupported legacy dtypes."""
    pred = df["pred"].to_numpy()
    label = df["label"].to_numpy()
    if pred.dtype.kind not in "fiub" or label.dtype.kind not in "fiub":
        return None
    label = label.astype(float, copy=False)
    result = []
    for positions in group.indices.values():
        scores, returns = pred[positions], label[positions]
        row = []
        for largest in (True, False):
            selected = returns[_selected_positions(scores, quantile, largest)]
            if precision:
                selected = selected[~np.isnan(selected)]
                row.append(np.mean(selected > 0 if largest else selected < 0) if len(selected) else np.nan)
            else:
                row.append(_nan_mean(selected))
        result.append(row)
    index = group.size().index
    return tuple(pd.Series([r[column] for r in result], index=index, dtype=float) for column in (0, 1))


def calc_long_short_prec(
    pred: pd.Series, label: pd.Series, date_col="datetime", quantile: float = 0.2, dropna=False, is_alpha=False
) -> Tuple[pd.Series, pd.Series]:
    """Daily positive-return precision of top scores and negative-return precision
    of bottom scores. Inputs are Series indexed by (datetime, instrument).

    Select floor(N * quantile) stocks on each side, where N counts finite
    predictions, using pandas' default nlargest/nsmallest tie handling. Missing
    labels do not alter selection and are excluded from precision denominators.
    is_alpha demeans labels by date before selection. dropna explicitly removes
    missing pred/label pairs before counting N. No selected labels gives NaN.
    The instrument-count guard uses the second index level.
    """
    if is_alpha:
        label = label - label.groupby(level=date_col, group_keys=False).mean()
    if int(1 / quantile) >= len(label.index.get_level_values(1).unique()):
        raise ValueError("Need more instruments to calculate precision")

    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)
    group = df.groupby(level=date_col, group_keys=False)
    if df.empty:
        empty = pd.Series(index=group.size().index, dtype=float)
        return empty.copy(), empty.copy()
    fast = _long_short_arrays(df, group, quantile, precision=True)
    if fast is not None:
        return fast
    long = group.apply(lambda x: _selected_labels(x, quantile, True).dropna().gt(0).mean())
    short = group.apply(lambda x: _selected_labels(x, quantile, False).dropna().lt(0).mean())
    return long, short


def calc_long_short_return(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    quantile: float = 0.2,
    dropna: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """Return ((top mean - bottom mean) / 2, all-stock label mean) by date.

    Labels must be raw stock returns, not cross-sectionally normalized labels.
    The second output is long_avg_r: the all-stock label mean.
    N counts finite predictions. Missing labels do not alter selection by default;
    dropna=True explicitly filters missing pairs first. Empty sides give NaN.
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)
    group = df.groupby(level=date_col, group_keys=False)
    if df.empty:
        empty = pd.Series(index=group.size().index, dtype=float)
        return empty.copy(), empty.copy()
    fast = _long_short_arrays(df, group, quantile)
    if fast is None:
        r_long = group.apply(lambda x: _selected_labels(x, quantile, True).mean())
        r_short = group.apply(lambda x: _selected_labels(x, quantile, False).mean())
    else:
        r_long, r_short = fast
    r_avg = group.label.mean()
    return (r_long - r_short) / 2, r_avg


def pred_autocorr(pred: pd.Series, lag=1, inst_col="instrument", date_col="datetime"):
    """Pearson self-correlation across stocks, shifted by lag observed dates.

    For sparse dates, lag counts adjacent rows after unstacking.
    A DataFrame uses its first column. date_col does not alter this calculation.
    """
    if isinstance(pred, pd.DataFrame):
        if len(pred.columns) > 1:
            logger.warning("pred_autocorr uses only the first column; {} extra columns ignored", len(pred.columns) - 1)
        pred = pred.iloc[:, 0]
    pred_ustk = pred.sort_index().unstack(inst_col)
    if not pd.api.types.is_numeric_dtype(pred.dtype) or not isinstance(lag, (int, np.integer)):
        corr_s = {idx: _pearson_corr(cur, prev) for (idx, cur), (_, prev) in
                  zip(pred_ustk.iterrows(), pred_ustk.shift(lag).iterrows())}
        return pd.Series(corr_s).sort_index()
    data = _float_values(pred_ustk)
    # Reuse pandas' shift argument validation without copying the wide panel.
    previous = pd.Series(np.arange(len(data)), dtype=float).shift(lag).to_numpy()
    corr_s = {}
    for i, idx in enumerate(pred_ustk.index):
        if np.isnan(previous[i]):
            corr_s[idx] = np.nan
        else:
            other = data[int(previous[i])]
            paired = ~np.isnan(data[i]) & ~np.isnan(other)
            corr_s[idx] = _pearson_arrays(data[i, paired], other[paired])
    return pd.Series(corr_s).sort_index()


def pred_autocorr_all(pred_dict, n_jobs=-1, **kwargs):
    """Return {method: autocorrelation Series}; n_jobs follows joblib."""
    keys = list(pred_dict)
    results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(pred_autocorr)(pred_dict[key], **kwargs) for key in keys)
    return dict(zip(keys, results))


def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False) -> (pd.Series, pd.Series):
    """Return daily (Pearson IC, Spearman RankIC).

    Index alignment and pairwise missing-value handling follow pandas. dropna
    drops undefined correlations from each output, not rows before grouping.
    Two valid nonconstant pairs suffice; no extra min_samples restriction.
    Pearson inputs are scaled before centering to avoid moment overflow/underflow.
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    group = df.groupby(date_col, group_keys=False)
    if df.empty or not all(pd.api.types.is_numeric_dtype(df[col]) for col in ("pred", "label")):
        # Retain pandas' behavior for unusual non-numeric/empty legacy inputs.
        ic = group.apply(lambda frame: _pearson_corr(frame["pred"], frame["label"]))
        ric = group.apply(lambda frame: frame["pred"].corr(frame["label"], method="spearman"))
    else:
        data = _float_values(df)
        result = [_array_correlations(data[positions, 0], data[positions, 1])
                  for positions in group.indices.values()]
        index = group.size().index
        ic = pd.Series([r[0] for r in result], index=index, dtype=float)
        ric = pd.Series([r[1] for r in result], index=index, dtype=float)
    if dropna:
        return ic.dropna(), ric.dropna()
    return ic, ric


def calc_all_ic(pred_dict_all, label, date_col="datetime", dropna=False, n_jobs=-1):
    """Return {method: {'ic': Series, 'ric': Series}} using joblib workers."""
    keys = list(pred_dict_all)
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(calc_ic)(pred_dict_all[key], label, date_col=date_col, dropna=dropna) for key in keys
    )
    return {key: {"ic": ic, "ric": ric} for key, (ic, ric) in zip(keys, results)}
