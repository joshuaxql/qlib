"""Portfolio evaluation helpers.

The risk_analysis API returns a Series using geometric annualization,
as used by the daily backtest engine.
"""

import numpy as np
import pandas as pd


def risk_analysis(returns, periods_per_year=252):
    returns = pd.Series(returns, dtype=float)
    if periods_per_year <= 0 or not np.isfinite(returns.to_numpy()).all() or (returns < -1).any():
        raise ValueError("Returns must be finite and >= -1; annualization must be positive")
    if returns.empty:
        return pd.Series(dtype=float)
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1)
    volatility = returns.std(ddof=1) if len(returns) > 1 else 0.0
    annualized = wealth.iloc[-1] ** (periods_per_year / len(returns)) - 1
    return pd.Series({
        "total_return": wealth.iloc[-1] - 1,
        "annualized_return": annualized,
        "annualized_volatility": volatility * np.sqrt(periods_per_year),
        "sharpe": returns.mean() / volatility * np.sqrt(periods_per_year) if volatility > 0 else np.nan,
        "max_drawdown": (wealth / peak - 1).min(),
        "win_rate": returns.gt(0).mean(),
    })
