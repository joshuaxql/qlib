# Qlib

[中文](https://github.com/joshuaxql/qlib/blob/main/README.md) | **English**

Qlib is a Python library for local stock-market quantitative research. It covers market and financial data management, expression-based factors, historical universe filtering, factor evaluation, and daily backtesting. A unified data interface connects local datasets to analysis and backtest results.

[PyPI](https://pypi.org/project/qlib-joshuaxql/) · [Documentation](https://qlib-joshuaxql.readthedocs.io/) · [Overview](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/overview.html) · [Quick start](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/quickstart.html) · [API reference](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/api/index.html) · [Dataset](https://huggingface.co/datasets/joshuaxql/qlib_data)

## Features

| Module | Capabilities |
|---|---|
| Data management | Daily prices, trading calendars, stock metadata, historical index membership, industries, and ST intervals; Tushare downloads, persistent CSV caches, and direct local data builds |
| Price adjustment | Backward-adjusted prices by default, with forward-adjusted and unadjusted modes; adjustment takes place before expression evaluation |
| Factor expressions | Arithmetic, conditional, rolling, expanding, ranking, correlation, and regression operators; automatic warm-up history and optional rejection of future references |
| Historical filters | ST status, listing age, industry, index membership, tradability, and expression filters with composable conditions |
| Point-in-time financials | All 163 numeric fields from Tushare `fina_indicator`; announcement and revision history stored in one `pit.data` / `pit.index` pair per stock |
| Factor evaluation | IC, RankIC, ICIR, long-short returns, autocorrelation, quantile returns, and turnover; multi-factor, multi-horizon analysis and report export |
| Factor neutralization | Daily joint regression on historical industry membership and log market cap; evaluate residuals using total or circulating market capitalization |
| Strategies and backtesting | `TopkStrategy`, `TopkDropoutStrategy`, and `WeightStrategy`; daily execution, fees, slippage, price limits, volume constraints, and position reports |
| Native acceleration | Pure C rolling, expanding, and PIT query kernels, built with MinGW-w64 and accessed through NumPy / ctypes |

## Usage

### 1. Install

Requires **Python 3.10+**. Install from PyPI:

```bash
python -m pip install qlib-joshuaxql
```

The distribution name is **`qlib-joshuaxql`**, while the Python import remains **`qlib`**:

```python
import qlib

print(qlib.__version__)
```

Upgrade with `python -m pip install --upgrade qlib-joshuaxql`. A dedicated virtual environment is recommended.
The PyPI package supports data reading, factor computation, and backtesting; market and financial datasets are obtained separately.

To run the `scripts/` data maintenance, native compilation, or local documentation commands below, first obtain and install the source:

```powershell
git clone https://github.com/joshuaxql/qlib.git
cd qlib
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[download,docs]"
```

For data reading, factor analysis, and backtesting only, install with `-e .`. On Linux/macOS, use `.venv/bin/python` as the Python executable.

On Windows, optionally build the C kernels using **MinGW-w64 GCC** with the same architecture as Python:

```powershell
# When GCC is on PATH
.venv\Scripts\python.exe scripts/build_rolling.py

# Or specify the compiler path
.venv\Scripts\python.exe scripts/build_rolling.py --cc D:/software/mingw64/bin/gcc.exe
```

The PyPI wheel does not include precompiled DLLs. Expression evaluation and PIT queries use pandas/NumPy fallbacks when the DLLs are unavailable. See the [installation guide](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/installation.html) for environment and build details.

### 2. Prepare data

Download `cn_data.zip` from the [Hugging Face dataset](https://huggingface.co/datasets/joshuaxql/qlib_data) and extract it into `~/.qlib/qlib_data/`. The resulting layout should be:

```text
~/.qlib/qlib_data/cn_data/
├─ calendars/
├─ features/
├─ instruments/
├─ industry/
├─ financial/
└─ stock_basic.csv
```

Alternatively, configure the data directories and start date in `scripts/config.py`, then build from Tushare:

```powershell
$env:TUSHARE_TOKEN = "your token"
.venv\Scripts\python.exe scripts/build_data.py
```

Downloads are cached as complete CSV files, which are read directly to build Qlib data. Financials and daily price limits can also be maintained independently:

```powershell
# Build financial data from existing quarterly CSVs; add --download to fetch/refresh first
.venv\Scripts\python.exe scripts/build_pit.py

# Download and add daily price limits to an existing daily dataset
.venv\Scripts\python.exe scripts/build_limits.py --download
```

Reading and backtesting work offline without a Tushare token. See [data management](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/data.html) for details.

### 3. Read prices and compute factors

```python
import qlib
from qlib.data import D
from qlib.data.filter import ListingDaysFilter, STFilter

provider = qlib.init("~/.qlib/qlib_data/cn_data", adjust="hfq")

prices = D.daily(
    ["000001.SZ", "600000.SH"], ["open", "close", "volume"],
    "2025-01-01", "2025-12-31",
)

pool = D.instruments("csi300", [STFilter(), ListingDaysFilter(min_days=180)])
features = D.features(
    pool,
    ["$close", "Mean($close, 20)", "$close / Ref($close, 20) - 1", "P($$eps)"],
    "2025-01-01", "2025-12-31",
    allow_future=False,
)
print(features.head())
```

- `adjust="hfq"` is the default backward-adjusted mode; `"qfq"` is forward-adjusted; `"none"` returns raw prices.
- `daily()` and `features()` return DataFrames indexed by `(instrument, datetime)`.
- `P($$eps)` reads EPS for the latest reporting period announced as of the observation date; `PRef($$eps, -1)` reads the preceding calendar quarter.
- Call `D.clear_cache()` or initialize again after updating local data.

See [expressions](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/expressions.html), [historical filters](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/filters.html), and [PIT financials](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/pit.html) for operator and timing semantics.

### 4. Evaluate factors

Using `provider` and `pool` from the example above:

```python
from qlib.contrib.report.analysis_model import factor_analysis

analysis = factor_analysis(
    pool,
    {"momentum20": "$close / Ref($close, 20) - 1"},
    "2025-01-01", "2025-12-31",
    provider=provider, horizons=(1, 5, 20), quantiles=5,
    price="open", entry_lag=1,
)
print(analysis.summary)
analysis.save("outputs/factor_analysis")
```

Default labels enter at the next trading session's open and measure the open-to-open return over the specified holding period. Metric definitions and lower-level evaluation APIs are documented in [factor analysis](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/factor.html).

Set `neutralize=True, market_cap="total_mv", neutralize_min_samples=20` in `factor_analysis()`
to jointly neutralize industry and size exposures before evaluating residual factors. You can also process an existing factor panel:

```python
from qlib.contrib.report.analysis_model import neutralize_factors

neutral = neutralize_factors(
    features[["$close / Ref($close, 20) - 1"]],
    provider=provider, market_cap="total_mv", min_samples=20,
)
```

Industry membership and market cap are taken on the factor date; missing or invalid observations remain NaN.
Use `market_cap="circ_mv"` for circulating market capitalization.

### 5. Run a backtest

```python
from qlib.backtest import ExchangeConfig, backtest
from qlib.contrib.strategy import TopkDropoutStrategy

strategy = TopkDropoutStrategy(
    topk=50, n_drop=5,
    score="$close / Ref($close, 20) - 1",
    instruments=pool, risk_degree=0.95,
)
result = backtest(
    strategy, "2025-01-01", "2025-12-31",
    provider=provider, initial_cash=1_000_000,
    exchange=ExchangeConfig(
        deal_price="open", lot_size=100,
        buy_cost=0.0003, sell_cost=0.0003, min_cost=5,
    ),
)
print(result.metrics)
result.save("outputs/backtest")
```

Signals execute on the next trading session using raw prices and automatically read `up_limit` / `down_limit` bounds. `TopkDropoutStrategy` selects replacements from actual holdings, sells before buying, and leaves retained positions at their existing quantities. Results include portfolio value, positions, trades, orders, and performance metrics. See [strategies and backtesting](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/backtest.html) for execution assumptions and configuration.

### 6. Browse the documentation

Read online: **[Qlib documentation](https://qlib-joshuaxql.readthedocs.io/)**.

The documentation uses Sphinx and `sphinx_rtd_theme`, covering user guides, Python APIs, expressions, and C interfaces. The detailed guides are currently in Chinese.

```powershell
.venv\Scripts\python.exe -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` after building. Further references: [package structure](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/structure.html), [API reference](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/api/index.html), and [documentation maintenance](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/documentation.html).

## License

This project is licensed under the [MIT License](https://github.com/joshuaxql/qlib/blob/main/LICENSE). Copyright and license notices for incorporated code are retained in the corresponding modules:
[native kernels](https://github.com/joshuaxql/qlib/blob/main/qlib/data/_libs/LICENSE), [factor evaluation](https://github.com/joshuaxql/qlib/blob/main/qlib/contrib/eva/LICENSE), and [strategies](https://github.com/joshuaxql/qlib/blob/main/qlib/contrib/strategy/LICENSE).

## Acknowledgments

- [Microsoft Qlib](https://github.com/microsoft/qlib) for its open-source implementations and quantitative research work. The relationship and scope of references are described in the [overview](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/overview.html).
- [Tushare](https://tushare.pro/) for market data, financial indicators, daily price limits, and other data APIs.
- [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/), and [joblib](https://joblib.readthedocs.io/) for numerical computing, data processing, and parallel execution.
- [MinGW-w64](https://www.mingw-w64.org/), [Sphinx](https://www.sphinx-doc.org/), and [Read the Docs](https://about.readthedocs.com/) for native compilation and documentation tools; [Hugging Face](https://huggingface.co/) for dataset hosting.
