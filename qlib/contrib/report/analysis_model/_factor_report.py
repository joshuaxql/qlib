"""Offline ECharts rendering for an existing factor analysis result."""

from functools import lru_cache
from importlib.resources import files
from hashlib import sha256
import json
from pathlib import Path
import re

from ._factor_report_data import _build_report_data, _historical_industries


def _factor_result(result, factor=None):
    """Select one factor without altering panels, metrics or configuration."""
    names = list(result.factors.columns)
    if factor is None:
        if len(names) != 1:
            raise ValueError("Choose factor=... for a single HTML, or use save(directory) to export every factor")
        return result
    if not isinstance(factor, str) or factor not in names:
        raise ValueError(f"Unknown factor: {factor!r}")
    from .analysis_model_performance import FactorAnalysisResult
    selected = FactorAnalysisResult(
        result.factors[[factor]], result.forward_returns,
        result.summary.loc[[factor]], result.daily.loc[[factor]],
        result.quantile_returns.loc[[factor]], result.quantile_membership[[factor]],
        result.turnover.loc[[factor]], result.autocorrelation.loc[[factor]], result.config.copy(),
    )
    if hasattr(result, "_report_provider_uri"):
        selected._report_provider_uri = result._report_provider_uri
    return selected


def _factor_directories(names):
    """Keep ordinary names, while making arbitrary labels safe on Windows."""
    used = set()
    output = []
    for name in names:
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(" .") or "factor"
        if safe in (".", ".."):
            safe = "factor"
        if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", safe):
            safe = "_" + safe
        safe = safe[:100]
        if safe != name or safe.casefold() in used:
            safe += "-" + sha256(name.encode("utf-8")).hexdigest()[:10]
        candidate, counter = safe, 1
        while candidate.casefold() in used:
            candidate = safe + "-" + str(counter)
            counter += 1
        used.add(candidate.casefold())
        output.append((name, candidate))
    return output


@lru_cache(maxsize=1)
def _assets():
    """Read the packaged template and pinned, licensed library once."""
    directory = files("qlib.contrib.report.analysis_model").joinpath("_assets")
    return {name: directory.joinpath(name).read_text(encoding="utf-8") for name in (
        "factor_report.html", "echarts.min.js", "ECHARTS-LICENSE.txt", "ECHARTS-NOTICE.txt",
        "ECHARTS-D3-LICENSE.txt")}


def _write_html(result, path, *, factor=None, title=None, industries=None, provider=None):
    """Write a self-contained report; optional sectors use signal-date membership."""
    result = _factor_result(result, factor)
    if industries is not None and provider is not None:
        raise ValueError("Supply industries or provider, not both")
    if industries is None:
        uri = getattr(result, "_report_provider_uri", None)
        # Serialized results remain exportable when their original local data
        # directory was moved. Explicit providers still report invalid data.
        if provider is None and uri is not None and (Path(uri) / "calendars/day.txt").is_file():
            from qlib.data import LocalProvider
            provider = LocalProvider(uri)
        if provider is not None:
            industries = _historical_industries(result, provider)
    calendar = provider.calendar() if provider is not None and hasattr(provider, "calendar") else None
    performance_calendar = calendar
    # A provider supplied only for sector labels may not cover external signals.
    # Keep those reports exportable using their date sequence for performance.
    if calendar is not None and not result.factors.index.get_level_values("datetime").isin(calendar).all():
        performance_calendar = None
    payload = _build_report_data(result, title=title, industries=industries, calendar=performance_calendar)
    if calendar is not None and "entry_lag" in result.config:
        import pandas as pd
        if len(calendar):
            payload["meta"]["price_data_end"] = calendar[-1].strftime("%Y-%m-%d")
            for item in payload["factors"]:
                for horizon, values in item["horizons"].items():
                    positions = calendar.get_indexer(pd.to_datetime(values["dates"]))
                    offset = int(result.config["entry_lag"]) + int(horizon)
                    values["daily"]["pending_forward_return"] = [
                        bool(position + offset >= len(calendar) and label_count == 0)
                        if position >= 0 else None
                        for position, label_count in zip(positions, values["daily"]["label_count"])]
    if provider is not None and hasattr(provider, "industry_names"):
        names = provider.industry_names()
        for item in payload["factors"]:
            for horizon in item["horizons"].values():
                sector_rows = [horizon["sector"]["overview"], horizon["sector"]["quantile_returns"]]
                sector_rows.extend(scenario["sector"]["quantile_returns"]
                                   for scenario in horizon.get("fee_scenarios", {}).values())
                for rows in sector_rows:
                    for row in rows:
                        row["code"] = row["name"]
                        row["name"] = names.get(row["code"], row["code"])
    from ._factor_report_charts import _add_chart_options
    _add_chart_options(payload)
    from pyecharts import __version__ as pyecharts_version
    payload["meta"].update(renderer="pyecharts", pyecharts_version=pyecharts_version)
    assets = _assets()
    payload["meta"].update(echarts_version="6.1.0", licenses=[{
        "name": "Apache ECharts 6.1.0",
        "source": "https://github.com/apache/echarts/tree/6.1.0",
        "license": assets["ECHARTS-LICENSE.txt"], "notice": assets["ECHARTS-NOTICE.txt"],
    }, {
        "name": "ECharts d3.js subcomponents (BSD-3-Clause)",
        "source": "https://github.com/apache/echarts/blob/6.1.0/licenses/LICENSE-d3",
        "license": assets["ECHARTS-D3-LICENSE.txt"], "notice": "",
    }])
    # HTML raw-text parsing recognizes a closing script even inside a JSON
    # string. Escape every '<', plus line separators, without altering data.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    library = assets["echarts.min.js"]
    if "</script" in library.lower():
        raise ValueError("The embedded ECharts bundle contains a closing script tag")
    template = assets["factor_report.html"]
    if template.count("__REPORT_DATA__") != 1 or template.count("__ECHARTS_LIBRARY__") != 1:
        raise ValueError("Invalid factor-report template placeholders")
    # Insert payload last; factor names containing a placeholder must remain text.
    document = template.replace("__ECHARTS_LIBRARY__", library).replace("__REPORT_DATA__", encoded)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
