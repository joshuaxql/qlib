"""Local market data, factor research and daily backtesting API."""

from .data import D, LocalProvider

__version__ = "0.1.1"


def init(provider_uri="~/.qlib/qlib_data/cn_data", **kwargs):
    """Configure the provider; adjust='hfq' (default), 'qfq', or 'none' selects prices."""
    provider = LocalProvider(provider_uri, **kwargs)
    D.register(provider)
    return provider


__all__ = ["init", "D", "LocalProvider"]
