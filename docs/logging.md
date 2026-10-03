# 日志配置

Qlib 使用 **Loguru** 记录关键事件，作为核心依赖随安装提供，无需另外安装 tqdm。
库和脚本不在导入时添加、移除或重新配置应用的 Loguru 处理器，也不自动创建日志文件。
没有应用自定义配置时，使用 Loguru 自带的 stderr 处理器。

## 输出范围

项目主动输出的日志仅使用 INFO / WARNING：

| 场景 | 输出 |
|---|---|
| 数据下载与构建 | 阶段开始、目标/断点目录、股票/交易日/指标数量、完成摘要 |
| 因子分析 | 开始、完成后的因子/持有期/股票日期记录数；保存成功后的目录 |
| 回测 | 开始日期范围与资金；完成后的权益、收益、费用、成交/结算和拒绝/部分成交数量 |
| C 编译 / PIT 基准 | 编译器与目标、原生库完成路径；基准结果核对及耗时 |
| 数据质量或恢复情况 | 缺失配套记录、非数值值、跳过异常公告、空刷新保留缓存、异常请求重试恢复等警告 |

- 不记录逐条股票/交易日/订单、缓存命中、每次重试或每一页请求；不显示 tqdm 或 joblib 进度。
- 普通数据读取、表达式求值和底层数值计算不输出成功日志。`pred_autocorr()` 只在多列输入被截取时警告。
- 批量下载/构建中，相同类别警告累加计数，在最外层操作退出时统一输出；中途失败也输出已有计数。
  独立调用底层清洗函数时则直接报告该次计数。警告不包含逐行样本和完整请求参数。
- 无 ST 股票、无财务记录的单次响应或分页尾部可以正常为空，不逐次记录；
  请求异常重试耗尽仍抛出异常，数据损坏/非法配置等错误也不会被吞掉或改成成功日志。
- Python/NumPy/pandas 自身的 warnings、编译器诊断和异常 traceback 不属于这些日志，项目不全局屏蔽它们。
  Sphinx 继续使用其原生日志，以保留 `-W` 和文档覆盖检查的行为。

## 在应用入口配置

通常只需在自己的主程序入口设置一次。以下由应用主动替换全部已有 Loguru 处理器；
如果应用已经有日志配置，请复用它，不必再次 `remove()`。

```python
import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
    backtrace=False,
    diagnose=False,
)
```

`INFO` 包含阶段摘要及警告；只看需要处理的问题时设 `level="WARNING"`。
Loguru 自带处理器的默认阈值是 DEBUG，但 Qlib 不发出逐条 DEBUG 明细。
`diagnose=False` 避免应用记录异常时附带局部变量值；不要在自己的日志中加入 Token 等凭据。

运行源码脚本时，也可在启动 Python **之前**设置 Loguru 环境变量，控制默认处理器：

```powershell
$env:LOGURU_LEVEL = "WARNING"
$env:LOGURU_FORMAT = "{time:HH:mm:ss} | {level} | {message}"
$env:LOGURU_DIAGNOSE = "NO"
.venv\Scripts\python.exe scripts/benchmark_pit.py
```

该示例的基准成功信息属于 INFO，因此在 WARNING 级别不会显示。改回 `INFO` 即显示关键摘要。
这些环境变量作用于进程内使用默认处理器的 Loguru 日志，不仅限于 Qlib；已经导入 Loguru 后修改环境变量不会重配现有处理器。
joblib 子进程有独立的日志配置，启动前设置环境变量也便于子进程继承。

## 可选文件日志与关闭日志

要在保留控制台输出的同时增加日志文件，可在应用入口显式添加文件处理器：

```python
from loguru import logger

file_handler = logger.add(
    "outputs/qlib.log",
    level="INFO",
    encoding="utf-8",
    rotation="10 MB",
    retention="7 days",
    backtrace=False,
    diagnose=False,
)
# 不再需要文件输出时：logger.remove(file_handler)
```

如果只想关闭导入模块产生的日志，而保留其他库的 Loguru 输出：

```python
logger.disable("qlib")
logger.disable("scripts")
# 恢复：logger.enable("qlib"); logger.enable("scripts")
```

直接执行脚本的入口模块名为 `__main__`，以上模块开关不覆盖入口自身的消息；
这类场景优先使用处理器级别或启动前的环境变量。批量警告汇总的辅助接口见 {py:mod}`qlib.log`。
