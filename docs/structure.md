# 包目录与接口位置

```text
qlib/
├─ __init__.py                         # init()
├─ data/
│  ├─ __init__.py                     # D、LocalProvider 导出
│  ├─ data.py                         # 本地数据提供器、D 实例
│  ├─ base.py                         # AST 表达式解析和求值
│  ├─ ops.py                          # 数值算子
│  ├─ filter.py                       # 历史过滤器
│  ├─ pit.py                          # 合并财务数据
│  └─ _libs/                          # C 源码、头文件、ctypes、DLL
├─ backtest/
│  ├─ backtest.py                     # 回测入口
│  ├─ executor.py                     # 执行逻辑
│  ├─ exchange.py                     # 交易配置
│  └─ report.py                       # 结果对象
└─ contrib/
   ├─ evaluate.py                     # 组合评估
   ├─ eva/alpha.py                    # 因子评估
   ├─ strategy/signal_strategy.py     # 信号策略
   └─ report/analysis_model/
      └─ analysis_model_performance.py
```

业务模块按数据、回测、策略和评估职责组织；原生模块采用纯 C 实现。
每个包的 `__init__.py` 提供文档所列的便捷导出。

## 公共导入

```python
from qlib.data import D, LocalProvider
from qlib.backtest import backtest, BacktestEngine, ExchangeConfig
from qlib.contrib.strategy import TopkDropoutStrategy, TopkStrategy, WeightStrategy
from qlib.contrib.eva.alpha import calc_ic
from qlib.contrib.report.analysis_model import factor_analysis
```

`qlib.data.D` 与 `qlib.data.data.D` 是同一实例。
