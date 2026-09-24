# 快速开始

以下示例使用已经构建的本地日线和财务数据，不在查询时发起下载。

## 初始化与读取

```python
import qlib
from qlib.data import D

provider = qlib.init("~/.qlib/qlib_data/cn_data", adjust="hfq")
prices = D.daily(["000001.SZ"], ["open", "close", "volume"],
                 "2025-01-01", "2025-12-31")
print(prices.head())
```

## 计算因子

```python
factors = D.features(
    "csi300",
    ["$close / Ref($close, 20) - 1", "P($$eps)"],
    "2025-01-01", "2025-12-31",
    allow_future=False,
)
```

## 因子分析

```python
from qlib.contrib.report.analysis_model import factor_analysis

analysis = factor_analysis(
    "csi300", {"momentum20": "$close / Ref($close, 20) - 1"},
    "2025-01-01", "2025-12-31",
    provider=provider, horizons=(1, 5, 20), quantiles=5,
)
print(analysis.summary)
analysis.save("outputs/factor_analysis")
```

默认收益标签从信号后的下一交易日开盘起算，详见[因子分析](factor.md)。

## 日频回测

```python
from qlib.contrib.strategy import TopkStrategy
from qlib.backtest import backtest, ExchangeConfig

result = backtest(
    TopkStrategy(score="$close / Ref($close, 20) - 1", instruments="csi300", topk=20),
    "2025-01-01", "2025-12-31", provider=provider,
    initial_cash=1_000_000,
    exchange=ExchangeConfig(delist_policy="last_close"),
)
print(result.metrics)
result.save("outputs/backtest")
```

此例显式采用退市前最后估值现金结算假设。完整成交、费用及持仓处理规则见[回测](backtest.md)。
