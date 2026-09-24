"""Composable date-wise filters; True always means retain the instrument."""

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


class Filter:
    def apply(self, provider, universe):
        raise NotImplementedError

    def __and__(self, other):
        return CompositeFilter("and", (self, other))

    def __or__(self, other):
        return CompositeFilter("or", (self, other))

    def __invert__(self):
        return CompositeFilter("not", (self,))


@dataclass
class CompositeFilter(Filter):
    operation: str
    filters: tuple

    def apply(self, provider, universe):
        values = [make_filter(f).apply(provider, universe) for f in self.filters]
        result = values[0]
        if self.operation == "not":
            return ~result
        if self.operation not in ("and", "or"):
            raise ValueError("Unknown filter composition")
        for value in values[1:]:
            result = result & value if self.operation == "and" else result | value
        return result


@dataclass
class ExpressionFilter(Filter):
    expression: str
    filter_start_time: object = None
    filter_end_time: object = None

    def apply(self, provider, universe):
        from .base import ExpressionEngine
        end = universe.index[-1] if len(universe.index) else None
        provider = provider._price_view(end_time=end)
        history = provider.calendar(end_time=end)
        result = pd.DataFrame(False, index=universe.index, columns=universe.columns)
        for code in universe:
            values = ExpressionEngine(provider, code, history, allow_future=False).evaluate(self.expression)
            # NaN/inf never passes an expression condition.
            values = values.reindex(universe.index)
            result[code] = values.notna() & values.ne(0)
        if self.filter_start_time is not None:
            result.loc[result.index < pd.Timestamp(self.filter_start_time)] = True
        if self.filter_end_time is not None:
            result.loc[result.index > pd.Timestamp(self.filter_end_time)] = True
        return result


class ExpressionDFilter(ExpressionFilter):
    """A keep-when-true filter configured with rule_expression."""
    def __init__(self, rule_expression, filter_start_time=None, filter_end_time=None):
        super().__init__(rule_expression, filter_start_time, filter_end_time)


@dataclass
class STFilter(Filter):
    exclude: bool = True

    def apply(self, provider, universe):
        st = provider.universe("st").reindex(index=universe.index, columns=universe.columns, fill_value=False)
        return ~st if self.exclude else st


@dataclass
class ListingDaysFilter(Filter):
    min_days: int = 0
    max_days: int | None = None
    trading_days: bool = False

    def __post_init__(self):
        from .ops import integer
        integer(self.min_days, "min_days")
        if self.max_days is not None:
            integer(self.max_days, "max_days", self.min_days)

    def apply(self, provider, universe):
        basic = provider.stock_basic().set_index("ts_code")
        result = pd.DataFrame(False, index=universe.index, columns=universe.columns)
        calendar = provider.calendar()
        for code in universe:
            if code not in basic.index or pd.isna(basic.at[code, "list_date"]):
                raise ValueError(f"Missing list_date: {code}")
            listed = basic.at[code, "list_date"]
            if self.trading_days:
                if listed < calendar[0]:
                    raise ValueError(f"Calendar starts after {code} listing; trading age is unknown")
                age = calendar.searchsorted(universe.index) - calendar.searchsorted(listed)
            else:
                age = (universe.index - listed).days
            valid = (age >= self.min_days) & (universe.index >= listed)
            if self.max_days is not None:
                valid &= age <= self.max_days
            if "delist_date" in basic and pd.notna(basic.at[code, "delist_date"]):
                valid &= universe.index < basic.at[code, "delist_date"]
            result[code] = valid
        return result


@dataclass
class MembershipFilter(Filter):
    market: str

    def apply(self, provider, universe):
        return provider.universe(self.market).reindex(index=universe.index, columns=universe.columns, fill_value=False)


class IndustryFilter(MembershipFilter):
    def __init__(self, industry):
        super().__init__(f"industry/{industry}")


@dataclass
class NameDFilter(Filter):
    """Regex on instrument codes, never on today's stock name."""
    name_rule_re: str

    def apply(self, provider, universe):
        pattern = re.compile(self.name_rule_re)
        values = [bool(pattern.match(code)) for code in universe]
        return pd.DataFrame(np.tile(values, (len(universe), 1)), index=universe.index, columns=universe.columns)


class TradableFilter(ExpressionFilter):
    def __init__(self):
        super().__init__("($open > 0) & ($close > 0) & ($volume > 0)")


FILTERS = {cls.__name__: cls for cls in (ExpressionFilter, ExpressionDFilter, STFilter,
           ListingDaysFilter, MembershipFilter, IndustryFilter, NameDFilter, TradableFilter)}


def make_filter(config):
    if isinstance(config, Filter):
        return config
    if not isinstance(config, dict):
        raise TypeError("Filter must be a Filter instance or a configuration dict")
    config = dict(config)
    name = config.pop("filter_type", config.pop("class", None))
    kwargs = config.pop("kwargs", {})
    if name not in FILTERS:
        raise ValueError(f"Unknown filter type: {name}")
    return FILTERS[name](**config, **kwargs)
