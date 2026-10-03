# 研究运行速度与复现实验

`scripts/benchmark_research.py` 使用本地真实数据运行常见研究，并同时保存耗时与完整研究结果。
不同源码版本的结果可以逐表对照，速度比较只有在结果对照通过后生成。

## 覆盖的研究

- 历史股票池、7 个日线字段的首次读取和重复读取。
- 8 类技术因子：动量、反转、波动率、相对成交量、价量相关性、区间位置、非流动性和趋势。
- 下一日开盘进入的 1、5、20 交易日收益标签。
- 去极值、行业及市值中性化、标准化、IC、RankIC、多空收益、准确率和自相关。
- 五分组收益、覆盖率、换手率及完整因子报告。
- 12 个参数化因子候选的搜索；前 75% 日期用于发现，后 25% 日期留出检验。
  使用 5 日标签时从发现期末剔除 6 个信号日，避免标签跨入检验期；检验期不参与候选选择。
- 日收益分布、缺失率和因子相关性。
- 按五日调仓的 Topk、逐日 TopkDropout，以及三组持仓数和换仓数参数回测。
- 四字段 PIT 财务快照、EPS/ROE/前一报告期 EPS 的因子投影和 CSV 导出。

这些是用于检查运行速度的历史研究。每个因子的收益方向和选股规模在实验配置中明确记录；
回测固定佣金与印花税假设，采用 `delist_policy='last_close'`，保留停牌、涨跌停和整手规则。

## 计时

```bash
python scripts/benchmark_research.py run --market csi300 \
  --start 2025-01-01 --end 2025-12-31 --repeats 3 --pit-stocks 100 \
  --output outputs/research_performance/current
```

默认使用 `~/.qlib/qlib_data/cn_data`，可用 `--data-root` 指定其他路径。
`--source-root` 指定待测源码目录，默认项目根目录；`--adjust` 选择复权方式。
`--threads` 设置数值库线程数，默认 1；结果记录进程实际 CPU 亲和性。
CPU 亲和性由调用环境设置，脚本不会修改系统配置。
本次本机对照使用逻辑 CPU 23（亲和性 `0x800000`）和数值库单线程。
Windows 可在进程启动时固定相同设置，例如在项目根目录的 PowerShell 中运行：

```powershell
cmd /c 'start "" /b /wait /affinity 800000 ".venv\Scripts\python.exe" -u scripts/benchmark_research.py run --threads 1 --output outputs/research_performance/current'
```

解释器路径按实际环境替换；两版本必须使用相同解释器、可用的 CPU 和原生内核配置。

每项任务记录 `perf_counter` 墙钟耗时、进程 CPU 时间和进程至当时的峰值工作集。
峰值工作集是整个进程的高水位，不能解释为单项任务的额外内存。
`daily_provider_cold` 清理提供器应用缓存，**不清理操作系统文件缓存**；warm 为随后同参数读取。
预处理、评估、回测等任务各自计时，输入准备的耗时另列。
CSV 导出只在第一次重复时计时，避免把重复覆盖同一文件的速度混入计算速度。

输出包括 `timings.csv`、`timing_summary.csv`、`timings.json`、完整结果 `results.pkl`，
以及因子挖掘、因子报告和回测 CSV。JSON 同时记录数据规模、数据指纹、库版本、
原生内核可用状态、线程和 CPU 配置。新的运行同时记录待测 Python/C/原生库文件的
SHA-256 和基准脚本指纹，便于确认速度记录对应的具体源码。

## 对比与定位瓶颈

先保留一个固定源码快照，再在独立进程中运行两个版本，使用相同数据、日期和配置。
准备固定基线，例如：

```bash
git archive --format=zip --output=outputs/baseline.zip v0.2.0
# 解压至 outputs/baseline_source 后运行：
python scripts/benchmark_research.py run --source-root outputs/baseline_source \
  --output outputs/research_performance/baseline
python scripts/benchmark_research.py run --output outputs/research_performance/optimized
python scripts/benchmark_research.py compare \
  --baseline outputs/research_performance/baseline \
  --optimized outputs/research_performance/optimized \
  --output outputs/research_performance/comparison
```

比较生成 `comparison.csv`、`comparison.html` 和 `equivalence.json`。
日线、原始技术因子/挖掘候选、订单、成交、分组成员精确对照；其他数值采用 `rtol=1e-11, atol=1e-12`，
索引、列、类型及 NaN 位置也必须一致。数据指纹、规模及运行配置不一致时拒绝生成速度结论。
对照失败时 `equivalence.json` 记录 `passed=false` 及原因，HTML 不展示已确认的加速比。
`results.pkl` 只用于同一实验的本地产物对照，应仅加载自己生成且信任的结果文件。

可在小规模单次实验添加 `--profile`，保存每项任务的 `.prof` 和累计耗时排名。
分析器本身会增加耗时；脚本拒绝将 profile 运行作为正式加速比。
每项任务的中位数、最小值、最大值和重复次数均保留，便于辨认首次导入和正常波动。

## 已采用的优化

股票池按日期边界一次构造数组；表达式复用有界语法计划。
普通日线字段按股票批量取值，共享一次复权因子及锚点处理；自定义字段钩子保留原路径。
仅需股票代码列表时直接检查日期掩码，省去重建每股有效区间。
纯逐点表达式、固定偏移和已验证的极值/计数算子只读取所需历史。
Mean、Std、Corr、Slope 等统计、EMA、扩展窗口、`Ref(x, 0)` 和不能确定依赖的表达式
保留完整历史，以保持浮点累积顺序。滚动窗口具有有限的数学依赖，并不保证裁去前缀后
浮点结果相同；微小差异经中性化可能改变并列排名，因此这类裁剪不能仅凭误差小而启用。
PIT 报告历史仍完整保留，简单字段投影减少逐公告构造 Series。

原生统计合并中，二进制缩放指数为零时直接保留数值，省去重复的 `scalbn(x, 0)` 调用；
非零指数的缩放和统计合并运算顺序继续保留。

因子分析复用每个日期的行位置、有效样本和选股结果，跨持有期共享排序；
RankIC 仅对各持有期的成对有效样本排名。自相关直接操作宽表数组。
所有极值缩放、并列选择、缺失标签和因子预处理规则继续保留。

回测只保留执行窗口行情，当天按持仓与实际订单读取股票行。
前一有效收盘价及该日复权因子一次预计算，成交限制检查使用常数时间查询。
策略排序保留稳定股票代码顺序，订单、费用、股份调整和现金流水的执行顺序不变。
默认提供器缓存容量未扩大。
