"""Backtest entry point; execution lives in executor.py."""

from .executor import BacktestEngine


def backtest(strategy, start_time, end_time, *, provider=None, initial_cash=1_000_000, exchange=None, benchmark=None):
    return BacktestEngine(provider, initial_cash=initial_cash, exchange=exchange).run(
        strategy, start_time, end_time, benchmark=benchmark
    )
