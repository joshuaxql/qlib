"""Explicit daily-bar execution assumptions, independent of current market rules."""

from dataclasses import dataclass

import numpy as np

from qlib.data.ops import integer


@dataclass(frozen=True)
class ExchangeConfig:
    deal_price: str = "open"
    lot_size: int = 100
    buy_cost: float = 0.0003
    sell_cost: float = 0.0003
    min_cost: float = 5.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    limit_threshold: float | None = None
    volume_limit: float | None = None
    volume_unit: int = 100
    adjust_positions: bool = True
    delist_policy: str = "raise"

    def __post_init__(self):
        if self.deal_price not in ("open", "close", "vwap"):
            raise ValueError("deal_price must be open, close or vwap")
        if self.delist_policy not in ("raise", "last_close"):
            raise ValueError("delist_policy must be 'raise' or 'last_close'")
        integer(self.lot_size, "lot_size", 1)
        integer(self.volume_unit, "volume_unit", 1)
        for name in ("buy_cost", "sell_cost", "sell_tax", "slippage"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not np.isfinite(self.min_cost) or self.min_cost < 0:
            raise ValueError("min_cost must be finite and nonnegative")
        for name in ("limit_threshold", "volume_limit"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f"{name} must be in (0, 1]")

    def fee(self, side, notional):
        if notional <= 0:
            return 0.0
        rate = self.buy_cost if side == "buy" else self.sell_cost
        return max(self.min_cost, notional * rate) + (notional * self.sell_tax if side == "sell" else 0)

    def block_reason(self, side, bar, previous_close):
        price = bar.get(self.deal_price, np.nan)
        if not np.isfinite(price) or price <= 0 or not np.isfinite(bar.get("volume", np.nan)) or bar["volume"] <= 0:
            return "suspended_or_missing_quote"
        if not np.isfinite(bar.get("close", np.nan)) or bar["close"] <= 0:
            return "missing_close"
        band = "up_limit" if side == "buy" else "down_limit"
        bound = bar.get(band, np.nan)
        if not np.isfinite(bound) and self.limit_threshold is not None:
            if not np.isfinite(previous_close) or previous_close <= 0:
                return "missing_limit_reference"
            bound = previous_close * (1 + self.limit_threshold * (1 if side == "buy" else -1))
        if np.isfinite(bound) and ((side == "buy" and price >= bound - 1e-8) or (side == "sell" and price <= bound + 1e-8)):
            return "limit_up" if side == "buy" else "limit_down"
        return None
