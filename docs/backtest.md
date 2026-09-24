# 策略与回测

## 策略

```python
from qlib.contrib.strategy import TopkStrategy, WeightStrategy

strategy = TopkStrategy(
    score="$close / Ref($close, 20) - 1",
    instruments="csi300", topk=20, rebalance=5, risk_degree=0.95, ascending=False,
)
```

{py:class}`qlib.contrib.strategy.signal_strategy.TopkStrategy` 每 N 个信号交易日选取 topk，等权分配 risk_degree 仓位。
排序按股票代码稳定处理同值，score 禁止未来引用；ascending=True 选较小值。

{py:class}`qlib.contrib.strategy.signal_strategy.WeightStrategy` 接受日期 × 股票权重表。
每个已提供日期行表示完整目标组合，NaN/未指定股票为 0，零行清仓；未提供的日期不调仓。
权重非负且合计不超过 1。自定义策略可实现 `target_weights(provider,start_time,end_time)` 返回同类表，外部信号因果性由调用者保证。

### TopkDropoutStrategy

{py:class}`qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy` 根据实际持仓进行每日换仓。
将持仓与候选新股合并排序，选择尾部股票卖出，再按排名买入候选股票。
保留股票的持股数量不变；买入预算为卖出成交后的可用现金乘以 `risk_degree`，在待买候选之间均分。

```python
from qlib.contrib.strategy import TopkDropoutStrategy

strategy = TopkDropoutStrategy(
    topk=50,
    n_drop=5,
    score="$close / Ref($close, 20) - 1",
    instruments="csi300",
    risk_degree=0.95,
    hold_thresh=1,
    only_tradable=False,
    forbid_all_trade_at_limit=True,
)
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `topk` | 必填 | 目标持股数量，正整数 |
| `n_drop` | 必填 | 每日换出候选数量，非负整数；0 表示仅补足空位 |
| `method_sell` | `"bottom"` | `bottom` 按合并排名末尾选卖出股票；`random` 从持仓随机选择 |
| `method_buy` | `"top"` | `top` 选择排名靠前的未持有股票；`random` 从排名前 topk 中的未持有股票随机选择 |
| `hold_thresh` | `1` | 卖出前至少完成的持有交易日数，停牌交易日也计数 |
| `only_tradable` | `False` | 排名筛选时是否跳过不可交易股票，筛选同时检查涨停和跌停 |
| `forbid_all_trade_at_limit` | `True` | 涨停或跌停时禁止两个方向交易；False 时按买入/卖出方向限制 |
| `risk_degree` | `0.95` | 卖出后现金用于新买入的比例 |
| `score` | `"$close / Ref($close, 20) - 1"` | 因子表达式，禁止未来引用 |
| `signal` | `None` | 外部预测分数或表达式；提供时优先于 score |
| `instruments` | `"all"` | 历史股票池与过滤配置，同时作用于外部信号 |
| `random_seed` | `None` | 随机选股种子；设置整数可复现重复回测，None 使用 NumPy 全局随机状态 |

外部预测信号可以是 `(datetime, instrument)` 索引的 Series、同索引的 DataFrame，
或日期 × 股票的分数矩阵。MultiIndex 的两个层级也可交换顺序；DataFrame 有多列预测时使用第一列。

```python
# pred 是日期与股票双层索引的预测分数 Series
strategy = TopkDropoutStrategy(topk=50, n_drop=5, signal=pred)
```

执行约定：

- 使用前一交易日分数，实际买卖在下一交易日进行；无信号日期或当日全部分数缺失时不产生新订单。
- 新买入排除 NaN/无穷分数，持仓缺少分数时在合并排名中置后；同分按股票代码稳定排序。
- 先确定候选名单，再检查最短持有期和执行条件。卖出失败或部分成交后，买入按实际现金计算；
  因此持股数可能暂时超过 topk。下一次决策使用真实成交后的持仓。
- `only_tradable=False` 时不可交易候选仍占候选名额，拒绝订单记录在 `orders`；
  `True` 时会在排名筛选阶段寻找可交易候选。实际成交仍遵守费用、整手和成交量限制。
- 候选不足时持股数可能小于 topk；排序末尾选中的持仓即使没有对应新股也可能被卖出。
- 状态由每次回测独立维护；复权因子改变等价股数时不会重置持有天数。

## 运行

```python
from qlib.backtest import backtest, ExchangeConfig

result = backtest(
    strategy, "2025-01-01", "2025-12-31", provider=provider,
    initial_cash=1_000_000,
    exchange=ExchangeConfig(deal_price="open", delist_policy="last_close"),
)
result.save("outputs/backtest")
```

{py:func}`qlib.backtest.backtest.backtest` 返回 `BacktestResult`；provider=None 使用 D。
也可创建 {py:class}`qlib.backtest.executor.BacktestEngine` 后调用 `run()`，构造参数 periods_per_year 控制年化，默认 252。
benchmark 可传覆盖回测日期的日收益 Series，用于基准与超额收益报告。

## 交易配置

{py:class}`qlib.backtest.exchange.ExchangeConfig` 的全部构造参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| deal_price | open | open/close/vwap |
| lot_size | 100 | 每手股数 |
| buy_cost / sell_cost | 0.0003 | 买入/卖出佣金率 |
| min_cost | 5.0 | 最低佣金 |
| sell_tax | 0.0 | 卖出税率 |
| slippage | 0.0 | 成交价偏移比例 |
| limit_threshold | None | 可选统一涨跌幅边界 |
| volume_limit | None | 可选当日成交量占比限制 |
| volume_unit | 100 | 数据成交量到股数的换算 |
| adjust_positions | True | 按复权因子变化调整等价持股数量 |
| delist_policy | raise | raise 或 last_close |

`fee(side,notional)` 计算费用；`block_reason(side,bar,previous_close)` 返回拒绝成交原因或 None。

## 执行规则

- 前一个交易日信号在下一交易日撮合，回测起点前的一个信号日也可用于首日。
- 先卖后买；买入按整手，清仓可卖零股；不借款、不做空。
- 执行使用原始价格；复权因子用于等价股份调整，不是现金分红、配股与税费的逐笔公司行动账本。
- 缺失价格或无有效成交量时拒绝成交；已有持仓按最后估值保留，并计入 stale_positions。
- 有 up_limit/down_limit 时优先使用精确边界，否则使用配置的统一 limit_threshold；默认不开启统一幅度限制。
- 数据构建从 Tushare `stk_limit` 写入 up_limit/down_limit。判断使用未加滑点的原始撮合价格：
  买价达到涨停价时拒绝买入，卖价达到跌停价时拒绝卖出；TopkDropout 可进一步禁止触及任一边界时的双向交易。
- `last_close` 在退市日（休市则下一交易日）以最后估值现金结算，记录 side='settle'，不计费用、滑点和换手。这是显式估值假设。
- 末日不强制清仓，持仓计入账户净值。成交量约束基于日线总量，是日频近似。

## 结果与指标

{py:class}`qlib.backtest.report.BacktestResult` 包含：

| 属性 | 内容 |
|---|---|
| report | cash、market_value、equity、net_value、return、cost、turnover、drawdown、stale_positions |
| positions | 每日持股 quantity、估值 price、market_value |
| trades | 成交/结算日期、信号日期、方向、数量、价格、金额、费用 |
| orders | requested、filled、状态和原因 |
| metrics | 收益、风险、费用和换手指标 |

`save(directory)` 将上述 5 张表导出 CSV。
{py:func}`qlib.contrib.evaluate.risk_analysis` 接受收益序列与 periods_per_year，返回 Series：
total_return、annualized_return（几何年化）、annualized_volatility、sharpe（零无风险利率）、max_drawdown、win_rate。
最大回撤包含初始本金；回测额外返回 total_cost、average_turnover，双边换手为当日买卖金额 / 前日净资产。
因子评估接口位于 `contrib.eva.alpha`，用于横截面信号分析。
