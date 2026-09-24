"""Portfolio reports derived from net-of-cost daily account values."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BacktestResult:
    report: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: pd.Series

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("report", "positions", "trades", "orders", "metrics"):
            getattr(self, name).to_csv(directory / f"{name}.csv", encoding="utf-8-sig")
