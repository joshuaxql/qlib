"""Signals are dated when observed; the engine executes them next session."""

# TopkDropoutStrategy selection is adapted from Microsoft Qlib (MIT).
# Copyright (c) Microsoft Corporation. See LICENSE in this directory.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlib.data.base import validate
from qlib.data.ops import integer
from qlib.data.data import normalize_code


@dataclass
class WeightStrategy:
    """User-supplied signal-date target weights. Absent dates mean no rebalance.

    Each supplied row is a complete portfolio: omitted/NaN instruments have zero
    target weight. Callers are responsible for causality of external signals.
    """
    weights: pd.DataFrame

    def target_weights(self, provider, start_time, end_time):
        frame = self.weights.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="datetime")
        frame.columns = [normalize_code(c) for c in frame.columns]
        if frame.index.has_duplicates or frame.columns.has_duplicates or frame.index.hasnans:
            raise ValueError("Weight dates/instruments must be unique and dates non-null")
        if len(frame.index.difference(provider.calendar())):
            raise ValueError("Weight signals must be on historical trading dates")
        frame = frame.astype(float).fillna(0)
        if not np.isfinite(frame.to_numpy()).all() or (frame < 0).any().any() or (frame.sum(axis=1) > 1 + 1e-10).any():
            raise ValueError("Weights must be finite, nonnegative and sum to <= 1")
        return frame.sort_index().loc[pd.Timestamp(start_time):pd.Timestamp(end_time)]


@dataclass
class TopkStrategy:
    score: str = "$close / Ref($close, 20) - 1"
    topk: int = 20
    instruments: object = "all"
    rebalance: int = 1
    risk_degree: float = 0.95
    ascending: bool = False

    def __post_init__(self):
        integer(self.topk, "topk", 1)
        integer(self.rebalance, "rebalance", 1)
        if not np.isfinite(self.risk_degree) or not 0 <= self.risk_degree <= 1:
            raise ValueError("risk_degree must be between 0 and 1")
        validate(self.score, allow_future=False)

    def target_weights(self, provider, start_time, end_time):
        dates = provider.calendar(start_time, end_time)
        features = provider.features(self.instruments, [self.score], start_time, end_time, allow_future=False)
        if features.empty:
            return pd.DataFrame(index=dates[::self.rebalance], dtype=float)
        scores = features[self.score].unstack("instrument").reindex(dates)
        weights = pd.DataFrame(0.0, index=dates[::self.rebalance], columns=sorted(scores.columns))
        for date in weights.index:
            # Stable code ordering gives reproducible tie-breaking.
            ranked = scores.loc[date].reindex(weights.columns).replace([np.inf, -np.inf], np.nan).dropna()
            selected = ranked.sort_values(ascending=self.ascending, kind="stable").head(self.topk).index
            if len(selected):
                weights.loc[date, selected] = self.risk_degree / len(selected)
        return weights


@dataclass(kw_only=True)
class TopkDropoutStrategy:
    """Hold top-scoring stocks and replace up to n_drop ranked holdings per day.

    Parameters
    ----------
    topk, n_drop : int
        Target stock count and daily replacement count; n_drop=0 only fills vacancies.
    signal : pandas.Series, pandas.DataFrame, str or None
        Scores indexed by (datetime, instrument), a date-by-stock score matrix,
        or an expression. A MultiIndex DataFrame uses its first column.
        None evaluates score. External scores must be observable on their dates.
    score : str
        Expression evaluated with allow_future=False when signal is None.
    instruments : object
        Historical universe/filter configuration, also applied to external scores.
    method_sell, method_buy : str
        Sell by bottom combined rank or random holding; buy top new scores or
        randomly sample non-held stocks from the topk ranked universe.
    hold_thresh : int
        Minimum completed holding sessions before a sale, including suspended days.
    only_tradable : bool
        Skip untradable stocks during candidate ranking, testing both limit sides.
    forbid_all_trade_at_limit : bool
        Block both directions at either price limit; False uses directional limits.
    risk_degree : float
        Fraction of cash available after sales allocated equally to buy candidates.
        Retained positions are not resized.
    random_seed : int or None
        Per-run random seed. None uses numpy's global random state.

    Selection precedes holding-period and execution checks: blocked or partial
    sales can leave more than topk stocks. Actual fills drive the next decision.
    Missing/nonfinite new scores are excluded; an entirely missing score row
    produces no orders. Ties use stable instrument-code ordering.
    """

    topk: int
    n_drop: int
    method_sell: str = "bottom"
    method_buy: str = "top"
    hold_thresh: int = 1
    only_tradable: bool = False
    forbid_all_trade_at_limit: bool = True
    signal: object = None
    score: str = "$close / Ref($close, 20) - 1"
    instruments: object = "all"
    risk_degree: float = 0.95
    random_seed: int | None = None

    def __post_init__(self):
        integer(self.topk, "topk", 1)
        integer(self.n_drop, "n_drop", 0)
        integer(self.hold_thresh, "hold_thresh", 0)
        if self.method_sell not in ("bottom", "random") or self.method_buy not in ("top", "random"):
            raise ValueError("method_sell must be bottom/random and method_buy must be top/random")
        if not np.isfinite(self.risk_degree) or not 0 <= self.risk_degree <= 1:
            raise ValueError("risk_degree must be between 0 and 1")
        for name in ("only_tradable", "forbid_all_trade_at_limit"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if self.random_seed is not None:
            integer(self.random_seed, "random_seed", 0)
            np.random.RandomState(self.random_seed)  # Validate the generator's seed range.
        if self.signal is None or isinstance(self.signal, str):
            validate(self.score if self.signal is None else self.signal, allow_future=False)
        elif not isinstance(self.signal, (pd.Series, pd.DataFrame)):
            raise TypeError("signal must be an expression, Series or DataFrame")

    def signal_scores(self, provider, start_time, end_time):
        """Return signal-date scores, respecting historical membership and filters."""
        if self.signal is None or isinstance(self.signal, str):
            expression = self.score if self.signal is None else self.signal
            features = provider.features(self.instruments, [expression], start_time, end_time, allow_future=False)
            if features.empty:
                return pd.DataFrame(index=provider.calendar(start_time, end_time), dtype=float)
            frame = features[expression].unstack("instrument")
        else:
            frame = self.signal.copy()
            if isinstance(frame.index, pd.MultiIndex):
                if frame.index.nlevels != 2 or set(frame.index.names) != {"datetime", "instrument"}:
                    raise ValueError("Signal index levels must be datetime and instrument")
                if isinstance(frame, pd.DataFrame):
                    if not len(frame.columns):
                        raise ValueError("Signal DataFrame must contain a score column")
                    frame = frame.iloc[:, 0]
                frame = frame.unstack("instrument")
            elif isinstance(frame, pd.Series):
                raise ValueError("Signal Series must have a datetime/instrument MultiIndex")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="datetime")
        frame.columns = [normalize_code(code) for code in frame.columns]
        if frame.index.has_duplicates or frame.columns.has_duplicates or frame.index.hasnans:
            raise ValueError("Signal dates/instruments must be unique and dates non-null")
        if len(frame.index.difference(provider.calendar())):
            raise ValueError("Signals must be on historical trading dates")
        frame = frame.astype(float).replace([np.inf, -np.inf], np.nan).sort_index().loc[start_time:end_time]
        universe = provider.universe(self.instruments, start_time, end_time)
        return frame.where(universe.reindex(index=frame.index, columns=frame.columns, fill_value=False))

    def select_stocks(self, scores, holdings, *, is_tradable, rng):
        """Return ranked sell/buy candidates before holding-age and fill checks.

        is_tradable(code) tests current-session tradability on both limit sides.
        rng supplies numpy-compatible choice(..., replace=False).
        """
        scores = scores.reindex(sorted(scores.index)).replace([np.inf, -np.inf], np.nan).dropna()
        if scores.empty:
            return [], []
        last = scores.reindex(sorted(holdings)).sort_values(ascending=False, kind="stable").index

        def eligible(codes):
            return [code for code in codes if not self.only_tradable or is_tradable(code)]

        count = max(0, self.n_drop + self.topk - len(last))
        ranked = scores.sort_values(ascending=False, kind="stable").index
        if self.method_buy == "top":
            today = eligible(ranked[~ranked.isin(last)])[:count]
        else:
            candidates = [code for code in eligible(ranked)[:self.topk] if code not in last]
            today = list(rng.choice(candidates, count, replace=False)) if len(candidates) >= count else candidates
        combined = scores.reindex(last.union(pd.Index(today))).sort_values(ascending=False, kind="stable").index
        if self.n_drop == 0:
            sell = []
        elif self.method_sell == "bottom":
            bottom = eligible(combined)[-self.n_drop:]
            sell = [code for code in last if code in bottom]
        else:
            candidates = eligible(last)
            sell = list(rng.choice(candidates, self.n_drop, replace=False)) if len(candidates) >= self.n_drop else candidates
        return sell, today[:max(0, len(sell) + self.topk - len(last))]

    def _order_requests(self, scores, holdings, holding_days, *, is_tradable, rng, get_cash, get_price, lot_size):
        """Yield sales first, then size purchases using cash after actual sale fills."""
        sell, buy = self.select_stocks(scores, holdings, is_tradable=is_tradable, rng=rng)
        for code in list(holdings):
            if code in sell and holding_days[code] >= self.hold_thresh:
                yield False, code, holdings[code], True
        value = get_cash() * self.risk_degree / len(buy) if buy else 0
        if value <= 0:
            return
        for code in buy:
            price = get_price(code)
            quantity = np.floor(value / price / lot_size) * lot_size if np.isfinite(price) and price > 0 else np.nan
            yield True, code, quantity, False
