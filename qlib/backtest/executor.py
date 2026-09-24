"""Daily backtest execution with next-session signals."""

import math

import numpy as np
import pandas as pd

from .exchange import ExchangeConfig
from .report import BacktestResult
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategy


class BacktestEngine:
    def __init__(self, provider=None, *, initial_cash=1_000_000, exchange=None, periods_per_year=252):
        if provider is None:
            from qlib.data import D
            provider = D
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self.provider, self.initial_cash = provider, float(initial_cash)
        self.exchange = ExchangeConfig(**exchange) if isinstance(exchange, dict) else (exchange or ExchangeConfig())
        self.periods_per_year = periods_per_year

    def run(self, strategy, start_time, end_time, *, benchmark=None):
        provider, exchange = self.provider, self.exchange
        dates = provider.calendar(start_time, end_time)
        if dates.empty:
            raise ValueError("Backtest range contains no historical trading days")
        calendar = provider.calendar(end_time=end_time)
        first = calendar.get_loc(dates[0])
        signal_start = calendar[max(0, first - 1)]
        dropout = isinstance(strategy, TopkDropoutStrategy)
        if dropout:
            signals = strategy.signal_scores(provider, signal_start, dates[-1])
            rng = np.random if strategy.random_seed is None else np.random.RandomState(strategy.random_seed)
        else:
            weights = strategy.target_weights(provider, signal_start, dates[-1])
            # Validate custom strategies just as strictly as externally supplied weights.
            signals = WeightStrategy(weights).target_weights(provider, signal_start, dates[-1])
        known = set(provider.list_instruments("all", as_list=True))
        if set(signals.columns) - known:
            raise ValueError(f"Unknown signal instruments: {sorted(set(signals.columns) - known)}")
        if not dropout:
            # Keep zero rows: they are explicit liquidation instructions.
            signals = signals.loc[:, signals.gt(0).any(axis=0)]
        codes = list(signals.columns)
        fields = {"close", "volume", exchange.deal_price}
        if exchange.adjust_positions:
            fields.add("factor")
        fields.update(set(provider.fields("daily")) & {"up_limit", "down_limit"})
        # Execution quotes must be raw: corporate actions are accounted for
        # through equivalent shares below, independently of signal adjustment.
        quotes = {code: pd.DataFrame({f: provider._daily(code, f).reindex(calendar) for f in fields}) for code in codes}
        basic = provider.stock_basic().set_index("ts_code")
        membership = provider.universe("all", dates[0], dates[-1])
        if benchmark is not None:
            benchmark = pd.Series(benchmark, dtype=float).copy()
            benchmark.index = pd.DatetimeIndex(pd.to_datetime(benchmark.index))
            if benchmark.index.has_duplicates:
                raise ValueError("Benchmark dates must be unique")
            benchmark = benchmark.reindex(dates)
            if not np.isfinite(benchmark.to_numpy()).all() or (benchmark < -1).any():
                raise ValueError("Benchmark must contain a valid daily return for every backtest date")
        cash, previous_equity = self.initial_cash, self.initial_cash
        holdings, marks, factors, holding_days = {}, {}, {}, {}
        reports, positions, trades, orders = [], [], [], []
        for date in dates:
            bars = {code: frame.loc[date] for code, frame in quotes.items()}
            # Adjustment-factor changes are modelled as equivalent shares, not
            # exact cash dividends/rights issues. No return jump at a pure split.
            for code in list(holdings):
                if code in basic.index and "delist_date" in basic and pd.notna(basic.at[code, "delist_date"]):
                    if date >= basic.at[code, "delist_date"]:
                        if exchange.delist_policy == "raise":
                            delisted = basic.at[code, "delist_date"].date()
                            raise ValueError(
                                f"Held instrument {code} delisted on {delisted}; "
                                "set delist_policy='last_close' for cash settlement at the last valuation"
                            )
                        # Explicit valuation assumption, not a market sale. Use
                        # the last mark already observed, never a later quote.
                        # No trading fees, slippage, volume limits or turnover.
                        quantity = holdings.pop(code)
                        price = marks.pop(code)
                        factors.pop(code, None)
                        holding_days.pop(code, None)
                        notional = quantity * price
                        cash += notional
                        orders.append((date, pd.NaT, code, "settle", quantity, quantity,
                                       "filled", "delisted_last_close"))
                        trades.append((date, pd.NaT, code, "settle", quantity, price, notional, 0.0))
                        continue
                factor = bars[code].get("factor", np.nan)
                has_quote = np.isfinite(bars[code].get("close", np.nan))
                if exchange.adjust_positions and has_quote:
                    if not np.isfinite(factor) or factor <= 0:
                        raise ValueError(f"Missing/invalid adjustment factor for held {code} on {date.date()}")
                    ratio = factor / factors[code]
                    holdings[code] *= ratio
                    marks[code] /= ratio
                    factors[code] = factor
            deal_marks = {}
            for code, quantity in holdings.items():
                price = bars[code].get(exchange.deal_price, np.nan)
                deal_marks[code] = price if np.isfinite(price) and price > 0 else marks[code]
            equity_at_deal = cash + sum(holdings[c] * p for c, p in deal_marks.items())
            index = calendar.get_loc(date)
            signal_date = calendar[index - 1] if index > 0 else None
            costs, traded = 0.0, 0.0
            reason_cache = {}

            def order_reason(code, side, both_limits=False):
                key = (code, side, both_limits)
                if key in reason_cache:
                    return reason_cache[key]
                bar = bars[code]
                previous = quotes[code].iloc[:index]
                valid_previous = previous.close.dropna()
                previous_close = valid_previous.iloc[-1] if len(valid_previous) else np.nan
                factor = bar.get("factor", np.nan)
                if exchange.adjust_positions and len(valid_previous) and np.isfinite(factor) and factor > 0:
                    previous_close *= previous.loc[valid_previous.index[-1], "factor"] / factor
                reason = exchange.block_reason(side, bar, previous_close)
                if both_limits and not reason:
                    reason = exchange.block_reason("sell" if side == "buy" else "buy", bar, previous_close)
                if side == "buy":
                    if not membership.at[date, code]:
                        reason = "outside_listing_interval"
                    if code in basic.index:
                        if ("list_date" in basic and date < basic.at[code, "list_date"]) or (
                            "delist_date" in basic and pd.notna(basic.at[code, "delist_date"])
                            and date >= basic.at[code, "delist_date"]
                        ):
                            reason = "outside_listing_interval"
                    if exchange.adjust_positions and (not np.isfinite(factor) or factor <= 0):
                        reason = "missing_adjustment_factor"
                reason_cache[key] = reason
                return reason

            if signal_date is not None and signal_date in signals.index:
                if dropout:
                    # Consume lazily: sale fills change cash before buys are sized.
                    requests = strategy._order_requests(
                        signals.loc[signal_date], holdings, holding_days,
                        is_tradable=lambda code: order_reason(code, "sell" if code in holdings else "buy", True) is None,
                        rng=rng, get_cash=lambda: cash,
                        get_price=lambda code: bars[code].get(exchange.deal_price, np.nan), lot_size=exchange.lot_size,
                    )
                else:
                    target = signals.loc[signal_date]
                    requests = []
                    for code in sorted(set(holdings) | set(target.index)):
                        bar = bars[code]
                        price = bar.get(exchange.deal_price, np.nan)
                        current = holdings.get(code, 0.0)
                        weight = target.get(code, 0.0)
                        # Missing quotes still produce an auditable rejected order.
                        if weight == 0:
                            desired = 0.0
                        elif np.isfinite(price) and price > 0:
                            desired = math.floor(weight * equity_at_deal / price / exchange.lot_size) * exchange.lot_size
                        else:
                            if not current:
                                orders.append((date, signal_date, code, "buy", np.nan, 0.0, "rejected", "suspended_or_missing_quote"))
                            continue
                        delta = desired - current
                        if abs(delta) > 1e-8:
                            requests.append((delta > 0, code, abs(delta), desired == 0))
                    requests = sorted(requests)
                # Sell before buy; this also ensures bought shares cannot be sold
                # in the same session (one complete target portfolio per session).
                for buy, code, requested, liquidate in requests:
                    side = "buy" if buy else "sell"
                    bar = bars[code]
                    factor = bar.get("factor", np.nan)
                    reason = order_reason(code, side, dropout and strategy.forbid_all_trade_at_limit)
                    if reason:
                        orders.append((date, signal_date, code, side, requested, 0.0, "rejected", reason))
                        continue
                    price = bar[exchange.deal_price] * (1 + exchange.slippage * (1 if buy else -1))
                    quantity = requested
                    if exchange.volume_limit is not None:
                        quantity = min(quantity, bar.volume * exchange.volume_unit * exchange.volume_limit)
                    if buy or not liquidate or quantity < requested:
                        quantity = math.floor((quantity + 1e-9) / exchange.lot_size) * exchange.lot_size
                    if buy:
                        # Both proportional fee and minimum fee must fit in cash.
                        affordable = min(cash / (1 + exchange.buy_cost), max(0, cash - exchange.min_cost)) / price
                        quantity = min(quantity, math.floor((affordable + 1e-9) / exchange.lot_size) * exchange.lot_size)
                    if quantity <= 1e-8:
                        orders.append((date, signal_date, code, side, requested, 0.0, "rejected", "cash_volume_or_lot"))
                        continue
                    notional = quantity * price
                    fee = exchange.fee(side, notional)
                    if not buy and cash + notional < fee:
                        orders.append((date, signal_date, code, side, requested, 0.0, "rejected", "insufficient_fee_cash"))
                        continue
                    cash += -notional - fee if buy else notional - fee
                    holdings[code] = holdings.get(code, 0.0) + (quantity if buy else -quantity)
                    if holdings[code] <= 1e-8:
                        holdings.pop(code, None)
                        marks.pop(code, None)
                        factors.pop(code, None)
                        holding_days.pop(code, None)
                    else:
                        holding_days.setdefault(code, 0)
                        marks[code] = price
                        if exchange.adjust_positions:
                            factors[code] = factor
                    status = "filled" if quantity >= requested - 1e-8 else "partial"
                    orders.append((date, signal_date, code, side, requested, quantity, status, "" if status == "filled" else "cash_volume_or_lot"))
                    trades.append((date, signal_date, code, side, quantity, price, notional, fee))
                    costs += fee
                    traded += notional
            stale = 0
            market_value = 0.0
            for code, quantity in holdings.items():
                holding_days[code] += 1
                close = bars[code].get("close", np.nan)
                if np.isfinite(close) and close > 0:
                    marks[code] = close
                else:
                    stale += 1
                value = quantity * marks[code]
                market_value += value
                positions.append((date, code, quantity, marks[code], value))
            equity = cash + market_value
            if cash < -1e-6 or equity <= 0:
                raise RuntimeError("Account invariant violated: negative cash or nonpositive equity")
            reports.append((date, cash, market_value, equity, equity / self.initial_cash,
                            equity / previous_equity - 1, costs, traded / previous_equity, stale))
            previous_equity = equity
        report = pd.DataFrame(reports, columns=["datetime", "cash", "market_value", "equity", "net_value",
                                               "return", "cost", "turnover", "stale_positions"]).set_index("datetime")
        report["drawdown"] = report.net_value / report.net_value.cummax().clip(lower=1) - 1
        metrics = risk_analysis(report["return"], self.periods_per_year)
        metrics["total_cost"] = report.cost.sum()
        metrics["average_turnover"] = report.turnover.mean()
        if benchmark is not None:
            report["benchmark_return"] = benchmark
            report["benchmark_net_value"] = (1 + benchmark).cumprod()
            report["excess_return"] = report["return"] - benchmark
            metrics["benchmark_total_return"] = report.benchmark_net_value.iloc[-1] - 1
            metrics["excess_total_return"] = metrics.total_return - metrics.benchmark_total_return
        return BacktestResult(
            report,
            pd.DataFrame(positions, columns=["datetime", "instrument", "quantity", "price", "market_value"]).set_index(["datetime", "instrument"]),
            pd.DataFrame(trades, columns=["datetime", "signal_date", "instrument", "side", "quantity", "price", "notional", "cost"]),
            pd.DataFrame(orders, columns=["datetime", "signal_date", "instrument", "side", "requested", "filled", "status", "reason"]),
            metrics,
        )
