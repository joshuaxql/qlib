# PIT 财务数据

PIT（Point in Time）保留报告期、公告日期和修订历史，使每个观察日期只能看到当时已公告的信息。
本项目使用合并 PIT v2：每只股票所有指标共用两个文件，减少逐指标文件数量。

## 下载与构建

财务数据仅来自 Tushare `fina_indicator`。默认包含其全部 **163 个数值指标**，
包括接口中默认不显示的字段。配置位于 `scripts/config.py`：

```python
from scripts.tushare.fields import FINA_INDICATOR_FIELDS

BUILD_PIT = True
PIT_FIELDS = {"fina_indicator": FINA_INDICATOR_FIELDS}
PIT_REFRESH_QUARTERS = 8
```

下载使用支持按季度获取全市场数据的 `fina_indicator_vip`，逐页显式请求全部字段。
季度 CSV 缓存使用 `financial/fina_indicator/YYYYMM.csv`。刷新保留不同公告日期的财务记录。
`ts_code`、`ann_date`、`end_date`、`update_flag` 用于股票标识、公告日、报告期和同日修订选择，
不作为数值指标存储。完整数值字段清单如下：

```{literalinclude} ../scripts/tushare/fields.py
:language: python
:start-at: FINA_INDICATOR_FIELDS
```

```powershell
# 使用已有缓存，独立构建财务数据
.venv\Scripts\python.exe scripts/build_pit.py --provider-uri "~/.qlib/qlib_data/cn_data"
# 先刷新缓存，再构建
.venv\Scripts\python.exe scripts/build_pit.py --download
```

独立构建在临时目录完成后替换 `financial/`，股票列表来自 `stock_basic.csv`。
季度 CSV 以宽表读入内存并按股票分组，仅将当前股票展开为指标记录，直接写出 `pit.data` / `pit.index`。
内存占用随选中的 CSV 数据量增长；构建过程中不生成中间数据库。
最近若干季度每次刷新，历史季度可通过 `REFRESH_CACHE=True` 全量刷新。来源未提供的历史版本不会由本模块推断。

## 读取

```python
from qlib.data import D

fields = D.fields("financial")
history = D.financial_records("000001.SZ", "eps")
snapshot = D.financial(["000001.SZ"], ["eps", "roe"],
                       asof="2025-05-01", periods=[202404, 202501])
features = D.features(["000001.SZ"], [
    "P($$eps)",
    "PRef($$eps, -1)",
    "P(Mean($$eps, 4))",
], "2025-01-01", "2025-12-31", allow_future=False)
```

`financial_records()` 返回 field_id/date/period/value/field 长表；`financial()` 返回截止 asof 每个报告期的最后版本。
`P()` 返回最新可见报告期位置，直接使用 `$$field` 等价于 `P($$field)`；`PRef(...,-1)` 返回上一自然季度位置，缺失季度保留 NaN。
指标名直接使用 Tushare 原名，如 `eps`、`roe`、`grossprofit_margin`、`q_eps`，
不添加表名前缀或额外后缀。`D.fields("financial")` 可列出全部已构建指标。

## 时点与口径

- 报告期使用 `YYYYQQ`，季度为 01～04，年报为 Q4。
- 可见日期为 `ann_date`，当日可见；同日相关字段更新完成后计算表达式。
- 同一报告期/公告日优先选择更大的 `update_flag`；同优先级冲突记录报错。
- 公告日早于报告期末的来源异常记录会跳过并打印明细，避免提前暴露报告期数据。
- 数值、单位和计算口径保持 Tushare 定义，包括累计值、期末值、单季度值和百分比。
  百分比字段不自动除以 100；`q_eps` 等单季度指标直接使用来源值，四季度均值不等同于 TTM。
- 旧季度修订只更新该季度。显式 NaN 修订会使旧值失效，不做数值层的前向填充。
- P 内不能混入日线字段、嵌套 P/PRef 或引用未来季度。日线与投影结果可以在 P 外组合。

## 存储格式

```text
financial/
├─ fields.json
└─ 000001.SZ/
   ├─ pit.data
   └─ pit.index
```

全局字典保存 field_id、原始名称、来源和频率，单位及口径标记为 `source`。
重建保留仍在配置中的字段编号，仅发布当前选择的指标和股票文件。

两文件头部均为 72 字节，struct 为 `<8sII16sQ32s`：magic、版本、记录大小、generation、记录数、SHA-256。

| 文件 | 记录布局（小端、无填充） | 字节数 |
|---|---|---:|
| `pit.data` | field_id:uint32、date:uint32、period:uint32、value:float64、next:uint64 | 28 |
| `pit.index` | field_id:uint32、period:uint32、offset:uint64、count:uint64 | 24 |

记录按 `(field_id, period, date)` 排列。offset/next 为包含头部的绝对字节偏移，链尾 next 为 `0xFFFFFFFFFFFFFFFF`。
加载时验证文件长度、校验和、generation、排序、索引和修订链。

## 增量写入与缓存

```python
from qlib.data.pit import register_fields, write_stock

register_fields("D:/data/cn_data/financial", {
    "custom_metric": {"frequency": "quarterly", "unit": "ratio", "basis": "period_end"},
})
# rows 是 date/period/field/value DataFrame
write_stock("D:/data/cn_data/financial", "000001.SZ", rows, update=True)
```

相同 field/period/date 由本次输入替换，其他历史版本保留。每次重写该股文件对，保持连续版本布局。
文件对以相同 generation 检测中断更新；按单写者使用，更新完成后查询。
股票级数组缓存受 `pit_cache_bytes` 限制，文件或字典变化会自动失效。多表达式共用一次公告事件扫描，再映射到交易日。
API 见 {py:mod}`qlib.data.pit`，C 批量版本定位见[原生 API](api/native.rst)。
