# 数据维护

## 数据来源与权限

| 数据 | 主要 Tushare 接口 |
|---|---|
| 交易日历 | `trade_cal` |
| 日线与成交 | `daily` |
| 复权因子 | `adj_factor` |
| 总市值与流通市值 | `daily_basic` |
| 每日涨跌停价 | `stk_limit` |
| 历史 ST | `stock_st` |
| 上市 / 退市基础信息 | `stock_basic`（L、D、P） |
| 指数成分 | `index_weight` |
| 申万一级行业 | `index_classify`、`index_member_all` |
| 财务指标（全部 163 个数值字段） | `fina_indicator_vip` |

## 数据存储与字段

```text
cn_data/
├─ calendars/
│  ├─ day.txt
│  └─ day_future.txt
├─ features/000001.SZ/
│  ├─ open.day.bin
│  ├─ high.day.bin
│  ├─ low.day.bin
│  ├─ close.day.bin
│  ├─ vwap.day.bin
│  ├─ volume.day.bin
│  ├─ total_mv.day.bin
│  ├─ circ_mv.day.bin
│  ├─ factor.day.bin
│  ├─ up_limit.day.bin
│  └─ down_limit.day.bin
├─ instruments/
│  ├─ all.txt
│  ├─ csi300.txt
│  ├─ csi500.txt
│  ├─ csi800.txt
│  ├─ csi1000.txt 
│  └─ st.txt
├─ industry/801050.SI.txt
├─ stock_basic.csv
└─ financial/
   ├─ fields.json
   └─ 000001.SZ/
      ├─ pit.data
      └─ pit.index
```

### 字段字典

| 表达式字段 | 存储 | 单位 / 含义 |
|---|---|---|
| `$open`、`$high`、`$low`、`$close` | 同名 `.day.bin` | 不复权价格，元 / 股 |
| `$volume` | `volume.day.bin` | 成交量，手；1 手 = 100 股 |
| `$factor` | `factor.day.bin` | 复权因子 |
| `$total_mv`、`$circ_mv` | 同名 `.day.bin` | 总 / 流通市值，万元 |
| `$vwap` | 从 amount 与 volume 推导 | 原价 amount × 10 / volume，元 / 股，可复权 |
| `$up_limit`、`$down_limit` | 同名 `.day.bin` | stk_limit 每日涨跌停原价，元 / 股；缺失保留 NaN，可复权读取 |

### 日期闭区间

股票池、行业和 ST 都使用以下三列，制表符分隔、UTF-8、两端包含：

```text
000001.SZ	2024-01-02	2024-01-05
000001.SZ	2024-01-10	2024-01-12
```

- 指数池：来自月度快照
- `industry/*.SI.txt`：申万一级行业；已知当前有效区间可用 `2099-12-31` 表示开放末端。
- `instruments/st.txt`：记录 ST 区间，按连续交易日合并。
- `stock_basic.csv`：当前基础信息快照；`list_date`、`delist_date` 为日期，`list_status` 为 L / D / P。

### 已公布未来日历

`day_future.txt` 包含 `day.txt` 完整历史，并请求延伸至**北京时间下一年的今天**。2 月 29 日顺延一年时按 pandas 年偏移落到次年 2 月 28 日。只写已公布的开市日期：目标日休市时，文件最后一行可以早于目标日。

### 构建数据

只要上海和深圳交易所的股票的数据，日期为20100101到现在，通过循环日期来以这个格式：先按照这个格式下载所有日线数据：
data_cache/
├─ trade_cal.csv
├─ daily/YYYYMMDD.csv
├─ adj_factor/YYYYMMDD.csv
├─ daily_basic/YYYYMMDD.csv
├─ stk_limit/YYYYMMDD.csv
├─ stock_st/YYYYMMDD.csv
├─ index_weight/000300.SH/YYYYMM.csv
├─ industry.csv
├─ financial/fina_indicator/YYYYMM.csv
└─ stock_basic.csv
缓存所有日线数据，使用 tqdm 进度条反馈下载与构建进度。

涨跌停数据按交易日请求全部分页，每页最多 5800 条。当日缓存每次刷新。
已有数据可通过 `python scripts/build_limits.py --download` 独立补充两个价格字段；
来源缺失或非正边界保留 NaN，不前填或写入固定比例估算值。

财务下载和构建由 `scripts/config.py` 中的 `BUILD_PIT`、`PIT_FIELDS` 控制，默认包含 `fina_indicator` 全部 163 个数值指标。字段直接使用 Tushare 原名，无表名前缀或额外后缀。每股全部指标合并为两个文件，保留各报告期公告与修订版本；可通过 `scripts/build_pit.py` 从现有 CSV 独立构建。详细格式和使用方式见 [合并 PIT v2](docs/pit.md)。

## 数据读取与研究

数据读取、表达式算子、历史过滤和日频回测已实现于 `qlib/`，接口及完整示例见 [README.md](README.md)。
