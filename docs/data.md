# 数据读取与维护

## 下载数据包

构建好的数据发布在 [Hugging Face 数据集](https://huggingface.co/datasets/joshuaxql/qlib_data)。
安装 `huggingface_hub` 后可下载并解压：

```python
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

archive = hf_hub_download("joshuaxql/qlib_data", "cn_data.zip", repo_type="dataset")
destination = Path("~/.qlib/qlib_data").expanduser()
destination.mkdir(parents=True, exist_ok=True)
with ZipFile(archive) as stream:
    stream.extractall(destination)
```

解压后的数据目录为 `~/.qlib/qlib_data/cn_data`。
数据集说明页列出数据截止日和字段范围，`manifest.json` 提供文件清单与 SHA-256 校验值。

## 初始化、参数与返回值

```python
import qlib
from qlib.data import D, LocalProvider

provider = qlib.init(
    "~/.qlib/qlib_data/cn_data",
    cache_uri="E:/data/data_cache",
    adjust="hfq", missing="nan", pit_cache_bytes=128 * 1024**2,
)
```

`qlib.init()` 返回 `LocalProvider` 并注册到全局 `D`。也可以直接创建多个 provider，分别读取不同目录。

| 参数 | 说明 |
|---|---|
| `provider_uri` | 本地数据根目录，支持 `~` |
| `cache_uri` | 可选 CSV 缓存目录，供 `read_table()` 使用 |
| `adjust` | `hfq` 后复权（默认）、`qfq` 前复权、`none` 不复权 |
| `missing` | 缺少行情文件或股票 PIT 文件时返回 NaN，或设 `raise` 抛错；损坏文件始终报错 |
| `pit_cache_bytes` | PIT 股票数组 LRU 缓存字节限额，0 表示不保留股票缓存 |

## 数据接口速查

各方法完整签名见 {py:class}`qlib.data.data.LocalProvider`。

| 方法 | 返回值 / 用途 |
|---|---|
| `calendar(start_time, end_time, freq='day', future=False)` | `DatetimeIndex`；`future=True` 读取已公布的未来交易日历 |
| `markets()`、`industries()` | 可查询的市场 / 行业名称列表 |
| `instruments(market='all', filter_pipe=None)` | 股票池配置字典 |
| `universe(instruments, start_time, end_time, freq='day', adjust=None)` | 日期 × 股票布尔表，True 为当日有效成员 |
| `list_instruments(..., as_list=False)` | 股票到有效日期区间的字典；`as_list=True` 返回股票列表 |
| `fields(kind='daily', instrument=None)` | 字段列表；kind 支持 daily、financial、stock_basic |
| `features(instruments, fields, start_time, end_time, ..., allow_future=True, adjust=None)` | `(instrument, datetime)` 索引、表达式为列的 DataFrame |
| `daily(instruments='all', fields=None, ...)` | 同上，列名为原始字段名 |
| `stock_basic(instruments=None, fields=None)` | 当前基础信息快照 DataFrame |
| `financial(instruments, fields, asof, periods=None)` | `(instrument, period)` 索引的 PIT 截面 |
| `financial_records(instrument, field=None, asof=None)` | 公告与修订长表 |
| `read_table(relative_path, **equals)` | 缓存目录内 CSV，可按列值筛选 |
| `clear_cache()` | 清理行情、日历、股票池及 PIT 缓存 |

日期区间两端包含。股票代码兼容 `000001.SZ`、`SZ000001`，返回时使用前一种形式。
`instruments` 可为市场名称、股票列表、区间字典或带过滤器的股票池配置。

```python
pool = D.instruments("csi300")
prices = D.daily(pool, ["close", "factor", "volume"], "2025-01-01", "2025-12-31")
features = D.features(pool, ["Mean($close, 20)"], "2025-01-01", "2025-12-31",
                      allow_future=False)
```

## 复权语义

- **后复权**：原始价格 × 当日复权因子。
- **前复权**：后复权价格 / 该股票截至查询 `end_time` 的最后一个有效复权因子；未给 end_time 时使用本地最新有效因子。
- **不复权**：原始价格。

逐次查询的 `adjust=None` 继承 provider 设置。复权先作用于 open、high、low、close、vwap、pre_close、up_limit、down_limit，再计算表达式和过滤条件；成交量、财务值等不复权。复权因子须有限且为正，缺失因子不会默认为 1。
原始行情缓存在不同复权查询之间共享，计算视图独立。重建日线或股票池后调用 `clear_cache()`。

## 本地文件格式

```text
cn_data/
├─ calendars/day.txt
├─ calendars/day_future.txt
├─ instruments/all.txt
├─ instruments/csi300.txt
├─ instruments/st.txt
├─ industry/801780.SI.txt
├─ features/000001.SZ/close.day.bin
├─ features/000001.SZ/factor.day.bin
├─ features/000001.SZ/up_limit.day.bin
├─ features/000001.SZ/down_limit.day.bin
├─ stock_basic.csv
└─ financial/
   ├─ fields.json
   └─ 000001.SZ/{pit.data,pit.index}
```

日线为小端 `float32`：第一项为在交易日历中的起始偏移，其余为连续交易日值，缺失位置保留 NaN。存储价格为未复权原价，volume 为手，total_mv/circ_mv 为万元。VWAP 为 `amount × 10 / volume`，单位元/股。
股票池、ST 和行业文件每行是 `股票代码 起始日期 结束日期`；同一股票可以有多段有效区间。

### 每日涨跌停价格

`up_limit`、`down_limit` 来自 Tushare `stk_limit`，以元/股为单位存储原始价格，
按 `(ts_code, trade_date)` 与实际日线记录对齐。来源缺失、空值、非有限值或非正价格写为 NaN，
停牌日不前向填充。相互矛盾的上下界、重复键或错误日期会中止构建。

```python
limits = D.daily(["000001.SZ"], ["up_limit", "down_limit"],
                 "2025-01-01", "2025-12-31", adjust="none")
```

默认复权读取时这两个字段也会复权；回测始终使用存储的原始价格边界。
`ExchangeConfig(limit_threshold=None)` 会自动使用已提供的边界，缺失的方向不推算；
显式指定 `limit_threshold` 才会在该方向价格缺失时按统一涨跌幅近似回退。

已有日线数据可独立补充：

```powershell
# 下载本地日历覆盖的全部交易日，并写入涨跌停字段
.venv\Scripts\python.exe scripts/build_limits.py --download
# 仅使用已有 stk_limit/YYYYMMDD.csv 缓存
.venv\Scripts\python.exe scripts/build_limits.py
```

{py:func}`scripts.build_limits.build_limits` 先暂存并校验全部新字段，再逐文件原子替换；
重跑可补全中断的发布。返回股票数、日线记录数和双侧边界覆盖数，完成后调用 `D.clear_cache()`。

## 下载与构建

配置位于 `scripts/config.py`：

| 配置 | 含义 |
|---|---|
| `TOKEN` | 从环境变量 `TUSHARE_TOKEN` 读取 |
| `OUTPUT_DIR` | 构建输出，默认 `~/.qlib/qlib_data/cn_data` |
| `CACHE_DIR` | 永久 CSV 缓存目录 |
| `START_DATE` | 下载/构建起点 |
| `REFRESH_CACHE` | 是否刷新已有历史缓存 |
| `REQUEST_INTERVAL`、`RETRIES`、`PAGE_SIZE` | 请求间隔、额外重试次数、通用分页大小；stk_limit 使用接口上限 5800 条并继续翻页 |
| `INDEX_CODES`、`INDEX_START_DATES` | 指数代码与数据起点 |
| `BUILD_PIT`、`PIT_FIELDS`、`PIT_REFRESH_QUARTERS` | 财务构建开关、fina_indicator 字段与刷新范围；默认全部 163 个数值指标 |

```powershell
$env:TUSHARE_TOKEN = "你的 Token"
.venv\Scripts\python.exe scripts/build_data.py
```

离线构建或恢复指定工作目录：

```python
from scripts.build_data import build_data

build_data(download=False)
# build_data(download=False, resume_dir="D:/data/.cn_data.build")
```

构建先写入工作目录，成功后替换目标。`build.json` 保存配置和完成状态，CSV 缓存保留。
日线按日期缓存，指数按月缓存，财务按季度缓存。分页结果在内存合并，完整下载后原子写入 CSV；
中断时重取尚未完成的 CSV，已完成的历史 CSV 可复用。构建直接读取 CSV 写出 Qlib 数据。
日线构建包含 `stk_limit/YYYYMMDD.csv`；当日涨跌停价格每次刷新，历史非空缓存可复用。
相关函数完整签名见[脚本 API](_generated/scripts_index.rst)。
