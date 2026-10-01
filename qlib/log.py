"""Key-event logging without configuring the application's Loguru handlers."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

from loguru import logger

_WARNING_COUNTS = ContextVar("qlib_warning_counts", default=None)


def log_warning(message, count=1):
    """Count a fixed warning category, or emit it immediately outside a batch.

    Use a category describing the counting unit, not per-row values, credentials
    or request payloads. Batch memory and output then depend on categories only.
    """
    counts = _WARNING_COUNTS.get()
    if counts is None:
        logger.warning("{}；数量={}", message, count)
    else:
        counts[message] += count


@contextmanager
def summarize_warnings():
    """Emit one warning per category on leaving the outermost batch, even on error.

    Nested downloads/builds share the current operation's counters. This helper
    does not catch exceptions or add, remove or reconfigure Loguru sinks.
    """
    if _WARNING_COUNTS.get() is not None:
        yield
        return
    counts = Counter()
    token = _WARNING_COUNTS.set(counts)
    try:
        yield
    finally:
        _WARNING_COUNTS.reset(token)
        for message, count in counts.items():
            logger.warning("{}；数量={}", message, count)
