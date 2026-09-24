# Qlib

**中文** | [English](https://github.com/joshuaxql/qlib/blob/main/README.en.md)

Qlib 是面向本地股票量化研究的 Python 工具库，覆盖行情与财务数据管理、表达式因子计算、历史股票池筛选、因子评估和日频回测。通过统一的数据接口，将本地数据转化为可分析、可回测的研究结果。

[PyPI](https://pypi.org/project/qlib-joshuaxql/) · [在线文档](https://qlib-joshuaxql.readthedocs.io/) · [功能介绍](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/overview.html) · [快速上手](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/quickstart.html) · [API 参考](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/api/index.html) · [数据集](https://huggingface.co/datasets/joshuaxql/qlib_data)

## 功能

| 模块 | 主要能力 |
|---|---|
| 数据管理 | 日线、交易日历、基础信息、历史指数成分、行业与 ST 区间；Tushare 下载与永久 CSV 缓存，直接构建本地数据 |
| 价格复权 | 默认后复权，支持前复权与不复权；价格先复权，再进行表达式计算 |
| 因子表达式 | 算术、条件、滚动、累计、排名、相关与回归算子；自动加载预热历史，可禁止未来引用 |
| 历史过滤 | ST、上市天数、行业、指数成分、可交易性与表达式过滤；支持组合条件 |
| PIT 财务 | Tushare `fina_indicator` 全部 163 个数值指标；保留公告与修订历史，每股一组 `pit.data` / `pit.index` |
| 因子评估 | IC、RankIC、ICIR、多空收益、自相关、分组收益与换手率；多因子、多持有期分析及报告导出 |
| 策略与回测 | `TopkStrategy`、`TopkDropoutStrategy`、`WeightStrategy`；日频撮合、费用、滑点、涨跌停、成交量约束及持仓报告 |
| 数值加速 | 纯 C 实现 rolling、expanding 和 PIT 查询核心，使用 MinGW-w64 构建，通过 NumPy / ctypes 调用 |

## 用法

### 1. 安装

需要 **Python 3.10+**。从 PyPI 安装：

```bash
python -m pip install qlib-joshuaxql
```

发行包名称为 **`qlib-joshuaxql`**，Python 导入名仍是 **`qlib`**：

```python
import qlib

print(qlib.__version__)
```

升级时使用 `python -m pip install --upgrade qlib-joshuaxql`。建议使用独立虚拟环境。
PyPI 安装即可读取数据、计算因子和运行回测；行情与财务数据需单独准备。

运行下文的 `scripts/` 数据维护、原生编译或本地文档构建命令时，请先获取源码并安装：

```powershell
git clone https://github.com/joshuaxql/qlib.git
cd qlib
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[download,docs]"
```

仅使用数据读取、因子分析和回测时，可安装 `-e .`。Linux/macOS 对应的 Python 路径为 `.venv/bin/python`。

Windows 下可选编译 C 核心，需要与 Python 位数匹配的 **MinGW-w64 GCC**：

```powershell
# GCC 已在 PATH 中时
.venv\Scripts\python.exe scripts/build_rolling.py

# 也可指定编译器路径
.venv\Scripts\python.exe scripts/build_rolling.py --cc D:/software/mingw64/bin/gcc.exe
```

PyPI wheel 不包含预编译 DLL。未构建 DLL 时，表达式计算和 PIT 查询使用 pandas/NumPy 回退。环境与编译说明见[安装指南](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/installation.html)。

### 2. 准备数据

从 [Hugging Face 数据集](https://huggingface.co/datasets/joshuaxql/qlib_data) 下载 `cn_data.zip`，解压到 `~/.qlib/qlib_data/`。解压后应存在：

```text
~/.qlib/qlib_data/cn_data/
├─ calendars/
├─ features/
├─ instruments/
├─ industry/
├─ financial/
└─ stock_basic.csv
```

也可配置 `scripts/config.py` 中的数据目录和起始日期，再从 Tushare 构建：

```powershell
$env:TUSHARE_TOKEN = "你的 Token"
.venv\Scripts\python.exe scripts/build_data.py
```

下载以完整 CSV 为缓存单位；构建直接读取 CSV 写出 Qlib 数据。财务和涨跌停字段也可独立维护：

```powershell
# 从已有季度 CSV 构建财务数据；添加 --download 可先下载/刷新
.venv\Scripts\python.exe scripts/build_pit.py

# 为已有日线数据下载并补充每日涨跌停价
.venv\Scripts\python.exe scripts/build_limits.py --download
```

读取和回测可离线运行，无需 Tushare Token。详细说明见[数据维护](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/data.html)。

### 3. 读取行情与计算因子

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

- `adjust="hfq"` 为默认后复权；`"qfq"` 为前复权；`"none"` 为原始价格。
- `daily()` 和 `features()` 返回以 `(instrument, datetime)` 为索引的 DataFrame。
- `P($$eps)` 读取当时已公告的最新报告期 EPS；`PRef($$eps, -1)` 读取前一自然季度。
- 更新本地数据后调用 `D.clear_cache()` 或重新初始化。

算子与时点语义见[表达式](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/expressions.html)、[历史过滤](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/filters.html)和[PIT 财务](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/pit.html)。

### 4. 评估因子

沿用上面的 `provider` 和 `pool`：

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

默认标签为下一交易日开盘进入、持有指定交易日数后的开盘价收益。指标口径与底层评估接口见[因子分析](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/factor.html)。

### 5. 运行回测

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

信号在下一交易日执行，撮合使用原始价格，并自动读取 `up_limit` / `down_limit` 边界。`TopkDropoutStrategy` 根据实际持仓换出股票，先卖后买，保留股票不重新调权。结果包含净值、持仓、成交、订单和绩效指标。执行假设与配置见[策略与回测](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/backtest.html)。

### 6. 浏览文档

在线阅读：**[Qlib 文档](https://qlib-joshuaxql.readthedocs.io/)**。

文档使用 Sphinx 和 `sphinx_rtd_theme`，涵盖使用指南、Python API、表达式与 C 接口：

```powershell
.venv\Scripts\python.exe -m sphinx -b html -W --keep-going docs docs/_build/html
```

构建后打开 `docs/_build/html/index.html`。更多入口：[包结构](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/structure.html)、[API 参考](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/api/index.html)、[文档维护](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/documentation.html)。

## 开源协议

本项目采用 [MIT License](https://github.com/joshuaxql/qlib/blob/main/LICENSE)。引用代码的版权与许可声明保留在相应模块中：
[数值核心](https://github.com/joshuaxql/qlib/blob/main/qlib/data/_libs/LICENSE)、[因子评估](https://github.com/joshuaxql/qlib/blob/main/qlib/contrib/eva/LICENSE)、[策略](https://github.com/joshuaxql/qlib/blob/main/qlib/contrib/strategy/LICENSE)。

## 鸣谢

- [Microsoft Qlib](https://github.com/microsoft/qlib)：感谢其开源实现与量化研究工作；项目关系和参考范围见[功能介绍](https://qlib-joshuaxql.readthedocs.io/zh-cn/latest/overview.html)。
- [Tushare](https://tushare.pro/)：提供行情、财务指标和每日涨跌停价格等数据接口。
- [NumPy](https://numpy.org/)、[pandas](https://pandas.pydata.org/)、[SciPy](https://scipy.org/) 和 [joblib](https://joblib.readthedocs.io/)：提供数值计算、数据处理与并行计算支持。
- [MinGW-w64](https://www.mingw-w64.org/)、[Sphinx](https://www.sphinx-doc.org/) 和 [Read the Docs](https://about.readthedocs.com/)：提供原生编译与文档工具；[Hugging Face](https://huggingface.co/) 提供数据集托管。
