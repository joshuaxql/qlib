# 因子计算与分析

## 因子评估函数

{py:mod}`qlib.contrib.eva.alpha` 提供 IC、RankIC、多空收益、准确率和自相关计算。
pred/label 是按日期和股票索引的 pandas Series；函数按索引对齐数据。

```python
from qlib.data import D
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return, pred_autocorr

data = D.features("csi300", [
    "$close / Ref($close, 20) - 1",
    "Ref($close, -2) / Ref($close, -1) - 1",
], "2025-01-01", "2025-12-31")
pred, label = data.iloc[:, 0], data.iloc[:, 1]
ic, ric = calc_ic(pred, label)
print({"IC": ic.mean(), "ICIR": ic.mean() / ic.std(),
       "Rank IC": ric.mean(), "Rank ICIR": ric.mean() / ric.std()})
long_short_r, long_avg_r = calc_long_short_return(pred, label)
autocorr = pred_autocorr(pred, lag=1)
```

| 函数 | 返回值与语义 |
|---|---|
| `calc_ic(pred, label, date_col='datetime', dropna=False)` | `(ic, ric)`：逐日 Pearson 和 Spearman |
| `calc_all_ic(pred_dict_all, label, ..., n_jobs=-1)` | `{名称: {'ic': Series, 'ric': Series}}` |
| `calc_long_short_return(pred, label, date_col='datetime', quantile=0.2, dropna=False)` | `(多空收益, 全体平均收益)` |
| `calc_long_short_prec(pred, label, ..., quantile=0.2, dropna=False, is_alpha=False)` | `(多头准确率, 空头准确率)` |
| `pred_autocorr(pred, lag=1, inst_col='instrument', date_col='datetime')` | 原始预测值的逐日 Pearson 自相关 |
| `pred_autocorr_all(pred_dict, n_jobs=-1, **kwargs)` | `{名称: 自相关 Series}` |

### 统计口径

- IC 按成对有效值计算，两对非恒定样本即可计算。RankIC 使用平均秩处理并列值。
- `calc_ic(dropna=True)` 分别删除两个输出中的 NaN 日期。
- 多空选每日 `int(N * quantile)` 个最大/最小预测值，收益为 **(多头均值 − 空头均值) / 2**。
- 多空函数的 `dropna=True` 在选股前删除缺失 pred/label 行；默认 False，N 包含缺失行，均值忽略缺失 label。
- `long_avg_r` 是全部输入标签均值，不是多头超额收益。
- 多头准确率是选中标签中严格大于 0 的比例，空头为严格小于 0；`is_alpha=True` 先按日对标签去均值。
- precision 的股票数检查使用第二层索引，调用时应使用 `(datetime, instrument)` 顺序。
- ICIR 使用均值 / 样本标准差，不年化；标准差为 0 时可产生 NaN/inf。
- 自相关是 Pearson，非 Rank 自相关；稀疏日期中 lag 按实际输入日期行计数。
- 批量函数采用 joblib，n_jobs=1 串行，-1 使用全部可用 CPU。DataFrame 自相关输入取第一列并记录日志。

## 批量因子分析

```python
from qlib.contrib.report.analysis_model import factor_analysis

result = factor_analysis(
    "csi300",
    {"momentum20": "$close / Ref($close, 20) - 1", "eps": "P($$eps)"},
    "2025-01-01", "2025-12-31",
    horizons=(1, 5, 20), price="open", entry_lag=1,
    quantiles=5, min_samples=2, turnover_lag=1,
)
print(result.summary)
result.save("outputs/factor_analysis")
```

此模块调用因子评估函数，并提供覆盖率、分位组、换手率与导出功能。
完整 API 位于 {py:mod}`qlib.contrib.report.analysis_model.analysis_model_performance`。

| 函数 / 类 | 输入与输出 |
|---|---|
| `calculate_factors(instruments, factors, start_time=None, end_time=None, *, provider=None, adjust=None)` | factors 为表达式、列表或名称到表达式的字典；返回因子表，固定禁用未来引用 |
| `calculate_forward_returns(factors, horizons=(1,5,20), *, provider=None, price='open', entry_lag=1, adjust=None)` | 输入因子表索引；返回以正整数持有期为列的收益标签表 |
| `analyze_factors(factors, forward_returns, *, quantiles=5, min_samples=2, turnover_lag=1)` | 分析已有因子与标签，返回 FactorAnalysisResult |
| `factor_analysis(...)` | 连续完成上述三步，返回 FactorAnalysisResult |
| `FactorAnalysisResult.save(directory)` | 导出 8 张 CSV 和 config.json |

`provider=None` 使用全局 D，`adjust=None` 继承 provider 复权配置。因子表索引为 `(instrument, datetime)`，因子列名为非空字符串；标签列为正整数持有期。
Series 也可输入，name 分别为因子名或持有期。索引必须唯一，datetime 为时间戳；分析以因子表索引为准，缺少标签时保留 NaN，无穷值转换为 NaN。

### 标签时点

```text
return(t,h) = price(t + entry_lag + h) / price(t + entry_lag) - 1
```

默认从下一交易日开盘进入，h 个交易日之后的开盘退出。end_time 限定信号日期，标签可读取其后本地已有报价。
偏移按完整交易日历计算，缺失报价不跳过、不前向填充。股票离开研究池后仍可用有效报价计算之前信号的标签。
设 `price='close', entry_lag=0` 可分析当日收盘到未来收盘收益，调用者须依据因子可用时点解释结果。

### 分组、覆盖率与换手

因子覆盖率 = 有效因子数 / 当日输入行数；配对覆盖率 = 因子和标签同时有效数量 / 当日输入行数。
外部因子表应保留缺失股票的 NaN 行，否则分母会缩小。

扩展分位组按当日有效因子的平均秩分组，相同值留在同一组，可能产生不等大组或空组；不同值数量不足 quantiles 时不分组。
分组不依赖未来标签是否缺失。多空收益函数使用 top/bottom N 选择，与此分位组在并列值或缺失时可能不同。

组换手率 = 当前组中新进入的股票数 / 当前组股票数。当前或 lag 期前组为空时为 NaN。
turnover_lag 按输入日期序列计数，自相关采用相同 lag。
多日标签存在重叠，分组收益样本不直接连乘为可交易组合净值。

### 结果对象

| 属性 / CSV | 索引与内容 |
|---|---|
| `factors` | 股票、日期 × 因子值 |
| `forward_returns` | 股票、日期 × 各持有期标签 |
| `summary` | `(factor,horizon)`：IC 均值/标准差/IR/正值比例/有效日期数，覆盖率、多空均值/标准差/正值比例、上下组换手、自相关 |
| `daily` | `(factor,horizon,datetime)`：样本数、coverage、pair_coverage、ic、rank_ic、universe_return、long_short_return |
| `quantile_returns` | `(factor,horizon,datetime,quantile)`：count、mean、std |
| `quantile_membership` | 股票、日期 × 各因子组号 |
| `turnover` | `(factor,datetime,quantile)`：turnover |
| `autocorrelation` | `(factor,datetime)`：autocorrelation |

汇总 IC 列名为 `ic_mean/ic_std/ic_ir/ic_positive_rate/ic_count`，RankIC 对应 `rank_ic_*`。
其余包括 `coverage/pair_coverage/mean_pair_count`、`long_short_mean/long_short_std/long_short_positive_rate/long_short_count`、`top_turnover/bottom_turnover/autocorrelation/dates`。
