# 功能介绍

本项目使用本地 Qlib 风格数据，提供可组合的数据查询、因子研究与股票日频回测。

| 模块 | 功能 | 入口 |
|---|---|---|
| 数据 | 交易日历、历史股票池、日线、基础信息、复权、CSV 查询 | `qlib.init()`、`D`、`LocalProvider` |
| 表达式 | 滚动、累计、回归、条件与算术运算 | `D.features()` |
| 过滤器 | ST、上市天数、行业、成分、可交易性、表达式组合 | `qlib.data.filter` |
| PIT 财务 | 公告时点、报告期、修订版本、合并存储 | `D.financial()`、`P()`、`PRef()` |
| 因子评估 | IC、RankIC、多空收益、准确率、自相关及批量评估 | `qlib.contrib.eva.alpha` |
| 因子中性化 | 按日控制历史行业与对数市值，输出联合回归残差 | `neutralize_factors()` |
| 分析报告 | 多因子、多持有期、覆盖率、分组收益与换手、CSV 导出 | `factor_analysis()` |
| 回测 | 次日执行、整手、费用、成交限制、退市结算、报告 | `qlib.backtest` |
| 策略 | 定期 Topk、TopkDropout 换仓、外部目标权重 | `qlib.contrib.strategy` |
| 数据维护 | Tushare 下载、CSV 缓存、断点续传、离线构建 | `scripts/` |
| C 核心 | rolling、expanding、PIT 批量版本定位 | `qlib.data._libs` |

## 数据处理流程

```text
Tushare / 已有 CSV → 数据构建 → 本地行情与 PIT
                                 ↓
                        历史股票池 + 复权读取
                                 ↓
                     表达式因子 → 因子评估 / 回测
                                 ↓
                          DataFrame / CSV 报告
```

## 关键约定

- 默认读取后复权价格；前复权和不复权可配置，复权发生在表达式求值之前。
- 日期查询通常为两端包含；股票池使用历史生效区间。
- 因子与交易信号应禁用未来引用，研究标签可显式使用未来价格。
- PIT 使用截至观察日已公告的版本，旧季度修订不会变成新的报告期。
- 回测执行使用原始价格，信号可使用复权价格；成交假设详见[回测](backtest.md)。

## 与官方 Qlib 的关系

本项目的目录组织和因子评估接口参考 [Microsoft Qlib](https://github.com/microsoft/qlib)。
`TopkDropoutStrategy` 的选股和现金分配逻辑参考其
[signal_strategy.py](https://github.com/microsoft/qlib/blob/be725493eb1a6bbb42bf11b37aa7669f59610ff1/qlib/contrib/strategy/signal_strategy.py)，
并接入本项目的日频撮合、费用和持仓管理接口。
数据提供器、合并 PIT 财务存储、批量报告和日频回测使用本站列出的实现与接口。
