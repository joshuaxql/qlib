# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License (see LICENSE in this directory).
"""Cross-sectional alpha evaluation API.

Scalar evaluators use pandas correlation, selection and grouping operations.
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
    vectors = []
    for series in (left, right):
        values = series[paired].to_numpy(dtype=float)
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
    r_long = group.apply(lambda x: _selected_labels(x, quantile, True).mean())
    r_short = group.apply(lambda x: _selected_labels(x, quantile, False).mean())
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
    corr_s = {}
    for (idx, cur), (_, prev) in zip(pred_ustk.iterrows(), pred_ustk.shift(lag).iterrows()):
        corr_s[idx] = _pearson_corr(cur, prev)
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
    ic = df.groupby(date_col, group_keys=False).apply(lambda df: _pearson_corr(df["pred"], df["label"]))
    ric = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
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
