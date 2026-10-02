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
- 多空选每日 `int(N * quantile)` 个最大/最小预测值，N 为当日有限预测值数量；NaN/inf 预测不贡献 N，也不会被选中。收益为 **(多头均值 − 空头均值) / 2**，选股数量为 0 时返回 NaN。
- 多空函数默认 `dropna=False`，未来 label 缺失不改变选股，收益均值和准确率分母忽略缺失 label；`dropna=True` 明确要求在选股前删除缺失 pred/label 行。
- `long_avg_r` 是全部输入标签均值，不是多头超额收益。
- 多头准确率是选中有效标签中严格大于 0 的比例，空头为严格小于 0；无有效选中标签时为 NaN，单日期输入也返回按日期索引的 Series。`is_alpha=True` 先按日对标签去均值。
- precision 的股票数检查使用第二层索引，调用时应使用 `(datetime, instrument)` 顺序。
- ICIR 使用均值 / 样本标准差，不年化；标准差为 0 时可产生 NaN/inf。
- 自相关是 Pearson，非 Rank 自相关；稀疏日期中 lag 按实际输入日期行计数。
- Pearson IC、自相关和中性化使用二进制指数缩放，先减首值再中心化；中性化在有界尺度拟合后还原残差。这避免有限大数/小数的矩运算溢出或下溢，并保留大基数上的小差值。
- 批量函数采用 joblib，n_jobs=1 串行，-1 使用全部可用 CPU，不打印 joblib 任务进度。
  DataFrame 自相关输入取第一列；仅在存在额外列被忽略时发出 Loguru 警告。

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
`factor_analysis()` 仅输出开始和完成摘要，`save()` 输出导出路径，不逐日逐因子刷屏；见[日志配置](logging.md)。
完整 API 位于 {py:mod}`qlib.contrib.report.analysis_model.analysis_model_performance`。

| 函数 / 类 | 输入与输出 |
|---|---|
| `calculate_factors(instruments, factors, start_time=None, end_time=None, *, provider=None, adjust=None)` | factors 为表达式、列表或名称到表达式的字典；返回因子表，固定禁用未来引用 |
| `winsorize_factors(factors, *, method='std', n=3.0, mad_scale=1.4826, ddof=0)` | 按日横截面进行均值标准差或中位数 MAD 去极值，截断到上下界 |
| `standardize_factors(factors, *, ddof=0)` | 按日、按因子进行 Z-score 标准化 |
| `preprocess_factors(...)` | 对已有因子表依次去极值、中性化、标准化；各步骤默认开启 |
| `neutralize_factors(factors, *, provider=None, market_cap='total_mv', min_samples=3)` | 按交易日进行行业＋对数市值联合回归，返回残差因子表 |
| `calculate_forward_returns(factors, horizons=(1,5,20), *, provider=None, price='open', entry_lag=1, adjust=None)` | 输入因子表索引；返回以正整数持有期为列的收益标签表 |
| `analyze_factors(factors, forward_returns, *, quantiles=5, min_samples=2, turnover_lag=1)` | 分析已有因子与标签，返回 FactorAnalysisResult |
| `factor_analysis(..., winsorize='std', neutralize=True, standardize=True)` | 因子计算后默认依次去极值、中性化、标准化，再生成标签并评估，返回 FactorAnalysisResult |
| `FactorAnalysisResult.save(directory)` | 导出 8 张 CSV 和 config.json |

`provider=None` 使用全局 D，`adjust=None` 继承 provider 复权配置。因子表索引为 `(instrument, datetime)`，因子列名为非空字符串；标签列为正整数持有期。
`calculate_factors` 和 `factor_analysis` 的 qfq 因子按各信号日已知的复权因子锚计算；延长查询终点或加入未来拆股，不改变早期因子及排名。普通 `provider.features` 查询继续使用查询终点的前复权锚，收益标签继续按两个端点的调整价格比计算。
Series 也可输入，name 分别为因子名或持有期。索引必须唯一，datetime 为时间戳；分析以因子表索引为准，缺少标签时保留 NaN，无穷值转换为 NaN。

### 去极值与标准化

这些操作按**每个交易日的股票横截面、每个因子列**独立计算，仅使用输入表中的有限值。
标准化先做二进制指数缩放，再减首值中心化；去极值以中位数为锚在缩放后的差值空间计算边界，仅还原实际截断的值，未截断值保持原值。这样可保留大基数上的小变化，并避免有限极大值的矩运算溢出。
与表达式 `Mean($close, 20)` 的时间窗口统计不同，它们不跨日期拟合，不依赖未来收益标签。
输入支持 Series / DataFrame，返回保留原列名、按 `(instrument, datetime)` 排序的 DataFrame；不会修改输入。

#### 两种去极值方法

{py:func}`qlib.contrib.report.analysis_model.analysis_model_performance.winsorize_factors`
将超界值截断到边界，保留这些股票行：

| `method` | 计算方法 | 默认边界 |
|---|---|---|
| `"std"`（默认） | 均值标准差法（均值方差法）：μ 为均值，σ 为标准差 | μ ± 3σ |
| `"mad"` | 中位数法：m = median(x)，MAD = median(abs(x − m)) | m ± 3 × 1.4826 × MAD |

`n` 控制倍数，默认 3；`mad_scale` 默认 1.4826，为正态一致性缩放，可设为 1 使用原始 MAD。
`ddof` 默认 0（总体标准差），可设为 1 使用样本标准差，仅影响 `std` 方法。

```python
from qlib.contrib.report.analysis_model import winsorize_factors, standardize_factors

# pred 为已有因子 Series 或 DataFrame
clipped_std = winsorize_factors(pred, method="std", n=3, ddof=0)
clipped_mad = winsorize_factors(pred, method="mad", n=3, mad_scale=1.4826)
scaled = standardize_factors(clipped_mad)
```

#### Z-score 标准化

{py:func}`qlib.contrib.report.analysis_model.analysis_model_performance.standardize_factors`
使用 `(x − mean(x)) / std(x)`，默认 `ddof=0`。非恒定的有效横截面得到均值约为 0、
按同一 `ddof` 计算的方差约为 1。该函数本身不去极值，也不进行中性化。

退化与缺失值规则：

- NaN、正负无穷不参与统计，输出保持 NaN；不同因子可以使用不同的有效股票集合。
- 标准差法去极值和标准化中，有效数 `N <= ddof` 时该日该列输出 NaN。
- 有效数足够时，恒定横截面去极值后保持原值，标准化后为 0；单个有效值在 `ddof=0` 时也标准化为 0。
- 中位数法的 MAD 为 0 时，上下界都等于中位数，所有有效值截断到中位数。
  因此高度集中的离散因子可能被压为常数，随后标准化为 0，IC 可能为 NaN。

#### 组合处理与批量报告

{py:func}`qlib.contrib.report.analysis_model.analysis_model_performance.preprocess_factors`
提供固定处理顺序：**去极值 → 行业＋市值中性化 → Z-score 标准化**。
先去极值可减少极端因子值对回归的影响；最后只做横截面线性缩放，可保留中性化残差的正交性。

```python
from qlib.contrib.report.analysis_model import preprocess_factors, factor_analysis

processed = preprocess_factors(
    pred,
    winsorize="std", winsorize_n=3,
    neutralize=True, market_cap="total_mv", neutralize_min_samples=20,
    standardize=True, ddof=0,
)

result = factor_analysis(
    "csi300", {"momentum20": "$close / Ref($close, 20) - 1"},
    "2025-01-01", "2025-12-31",
    winsorize="std", winsorize_n=3,
    neutralize=True, neutralize_min_samples=20,
    standardize=True,
    horizons=(1, 5, 20), quantiles=5,
)
```

两接口默认开启全部步骤：`winsorize="std"`、`winsorize_n=3`、`neutralize=True`、`standardize=True`，
中性化默认使用 `market_cap="total_mv"`，标准差口径默认 `ddof=0`。
因此默认调用需要历史行业归属和总市值数据。可用 `winsorize=None`、`neutralize=False`、
`standardize=False` 分别关闭对应步骤；三项同时关闭即可分析原始因子。
改用中位数法时设置 `winsorize="mad"`。关闭中性化后，单独去极值、标准化无需行业或市值数据。
组合接口中的 `ddof` 同时控制标准差法去极值和 Z-score，
如需两步使用不同口径，可分别调用独立函数。

`result.factors` 和所有报告指标使用处理后的因子；`forward_returns` 保持原有收益标签定义。
`config.json` 的 `preprocessing` 保存实际启用的处理顺序、去极值方法与参数、标准化参数；
`neutralization` 保存中性化参数。只向预处理函数传入因子列，不要传入未来收益标签。

### 行业＋市值联合中性化

{py:func}`qlib.contrib.report.analysis_model.analysis_model_performance.neutralize_factors`
在每个交易日、每个因子的有效股票横截面上，进行等权普通最小二乘回归：

```text
factor_i = intercept + beta × ln(market_cap_i) + 行业哑变量效应 + residual_i
```

返回的 `residual_i` 即中性化因子。行业与市值在同一次回归中控制。
默认使用总市值 `total_mv`，也可以指定流通市值 `circ_mv`。

#### 已有因子中性化后计算 IC

```python
from qlib.data import D
from qlib.contrib.eva.alpha import calc_ic
from qlib.contrib.report.analysis_model import neutralize_factors

data = D.features("csi300", [
    "-Delta((2 * $close - $low - $high) / ($high - $low), 1)",
    "Ref($close, -6) / Ref($close, -1) - 1",
], "2025-01-01", "2025-12-31")

pred = data.iloc[:, 0].rename("alpha")
label = data.iloc[:, 1]
neutral = neutralize_factors(pred, market_cap="total_mv", min_samples=20)
ic, ric = calc_ic(neutral["alpha"], label)
```

输入可为 Series 或多因子 DataFrame；输出总是 DataFrame，列名保留，
索引规范为排序后的 `(instrument, datetime)`。股票代码使用数据提供器的标准格式，如 `000001.SZ`。
只传入需要中性化的因子列；未来收益标签单独用于评估。

#### 批量分析中启用

```python
result = factor_analysis(
    "csi300", {"momentum20": "$close / Ref($close, 20) - 1"},
    "2025-01-01", "2025-12-31",
    winsorize=None, neutralize=True, standardize=False,
    market_cap="total_mv", neutralize_min_samples=20,
    horizons=(1, 5, 20), quantiles=5,
)
```

`neutralize` 默认 True。上例关闭去极值和标准化以单独中性化，此时 `result.factors` 保存残差，IC、分组收益、换手率和自相关
都使用残差因子；收益标签保持原有定义。`config.json` 的 `neutralization` 记录方法、市值字段及最小样本数。
`neutralize_min_samples` 控制回归样本数，与 IC 报告的 `min_samples` 独立。

#### 时点、样本与退化情况

- 行业使用 `industry/*.txt` 在**因子当日**有效的历史区间，不使用 `stock_basic` 的当前行业快照。
  市值使用同日原始字段；只根据传入因子行拟合，不自动扩展到全市场，不使用未来收益筛选样本。
- 行业缺失、因子非有限或市值缺失、非有限、非正时，该行该因子输出 NaN；不前填或插补。
  不同行业同时覆盖同一股票日期时明确报错；缺少整个行业或市值数据源时也报错。
- 每个因子按自身有效样本独立回归。至少满足 `min_samples`（默认 3，可设为不小于 2 的整数），
  且有效股票数必须大于设计矩阵秩，保留正的残差自由度，否则该日该因子输出 NaN。
- 设计矩阵使用全部有效行业哑变量（已包含截距空间）与居中、缩放后的自然对数市值；
  通过 SVD 最小二乘处理共线控制变量。只有一个行业、恒定市值或行业内恒定市值时仍可在自由度足够时计算。
- 输出未经去极值或标准化的等权残差。完全被控制变量解释的因子残差置零，避免浮点误差形成虚假信号。
  全零残差没有横截面区分度，对应 IC 可为 NaN。

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
| `factors` | 股票、日期 × 最终因子值，包含已启用的去极值、中性化和标准化处理 |
| `forward_returns` | 股票、日期 × 各持有期标签 |
| `summary` | `(factor,horizon)`：IC 均值/标准差/IR/正值比例/有效日期数，覆盖率、多空均值/标准差/正值比例、上下组换手、自相关 |
| `daily` | `(factor,horizon,datetime)`：样本数、coverage、pair_coverage、ic、rank_ic、universe_return、long_short_return |
| `quantile_returns` | `(factor,horizon,datetime,quantile)`：count、mean、std |
| `quantile_membership` | 股票、日期 × 各因子组号 |
| `turnover` | `(factor,datetime,quantile)`：turnover |
| `autocorrelation` | `(factor,datetime)`：autocorrelation |

汇总 IC 列名为 `ic_mean/ic_std/ic_ir/ic_positive_rate/ic_count`，RankIC 对应 `rank_ic_*`。
其余包括 `coverage/pair_coverage/mean_pair_count`、`long_short_mean/long_short_std/long_short_positive_rate/long_short_count`、`top_turnover/bottom_turnover/autocorrelation/dates`。
