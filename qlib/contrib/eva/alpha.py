# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License (see LICENSE in this directory).
"""Cross-sectional alpha evaluation API.

Scalar evaluators use pandas correlation, selection and grouping operations.
Batch evaluation uses joblib workers and returns dictionaries of result Series.
"""

import logging
from typing import Tuple

import pandas as pd
from joblib import Parallel, delayed


def calc_long_short_prec(
    pred: pd.Series, label: pd.Series, date_col="datetime", quantile: float = 0.2, dropna=False, is_alpha=False
) -> Tuple[pd.Series, pd.Series]:
    """Daily positive-return precision of top scores and negative-return precision
    of bottom scores. Inputs are Series indexed by (datetime, instrument).

    Select floor(N * quantile) stocks on each side, using pandas' default
    nlargest/nsmallest tie handling. is_alpha demeans labels by date before
    selection. dropna removes missing pred/label pairs before counting N.
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

    def N(x):
        return int(len(x) * quantile)

    long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label)
    short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label)
    groupll = long.groupby(date_col, group_keys=False)
    l_dom = groupll.apply(lambda x: x > 0)
    l_c = groupll.count()
    groups = short.groupby(date_col, group_keys=False)
    s_dom = groups.apply(lambda x: x < 0)
    s_c = groups.count()
    return (l_dom.groupby(date_col, group_keys=False).sum() / l_c), (
        s_dom.groupby(date_col, group_keys=False).sum() / s_c
    )


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
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)
    group = df.groupby(level=date_col, group_keys=False)

    def N(x):
        return int(len(x) * quantile)

    r_long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label.mean())
    r_short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label.mean())
    r_avg = group.label.mean()
    return (r_long - r_short) / 2, r_avg


def pred_autocorr(pred: pd.Series, lag=1, inst_col="instrument", date_col="datetime"):
    """Pearson self-correlation across stocks, shifted by lag observed dates.

    For sparse dates, lag counts adjacent rows after unstacking.
    A DataFrame uses its first column. date_col does not alter this calculation.
    """
    if isinstance(pred, pd.DataFrame):
        logging.getLogger("pred_autocorr").warning("Only the first column in %s of `pred` is kept", pred.columns)
        pred = pred.iloc[:, 0]
    pred_ustk = pred.sort_index().unstack(inst_col)
    corr_s = {}
    for (idx, cur), (_, prev) in zip(pred_ustk.iterrows(), pred_ustk.shift(lag).iterrows()):
        corr_s[idx] = cur.corr(prev)
    return pd.Series(corr_s).sort_index()


def pred_autocorr_all(pred_dict, n_jobs=-1, **kwargs):
    """Return {method: autocorrelation Series}; n_jobs follows joblib."""
    keys = list(pred_dict)
    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(pred_autocorr)(pred_dict[key], **kwargs) for key in keys)
    return dict(zip(keys, results))


def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False) -> (pd.Series, pd.Series):
    """Return daily (Pearson IC, Spearman RankIC).

    Index alignment and pairwise missing-value handling follow pandas. dropna
    drops undefined correlations from each output, not rows before grouping.
    Two valid nonconstant pairs suffice; no extra min_samples restriction.
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    ic = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"]))
    ric = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
    if dropna:
        return ic.dropna(), ric.dropna()
    return ic, ric


def calc_all_ic(pred_dict_all, label, date_col="datetime", dropna=False, n_jobs=-1):
    """Return {method: {'ic': Series, 'ric': Series}} using joblib workers."""
    keys = list(pred_dict_all)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(calc_ic)(pred_dict_all[key], label, date_col=date_col, dropna=dropna) for key in keys
    )
    return {key: {"ic": ic, "ric": ric} for key, (ic, ric) in zip(keys, results)}
