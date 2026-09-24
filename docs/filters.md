# 历史过滤器

过滤器输入是日期 × 股票的布尔股票池，输出相同形状，True 始终表示保留。
多个过滤器顺序取交集，并受原始股票池历史区间限制。

```python
from qlib.data import D
from qlib.data.filter import STFilter, ListingDaysFilter, ExpressionFilter

pool = D.instruments("csi300", [
    STFilter(),
    ListingDaysFilter(min_days=180),
    ExpressionFilter("($close > 5) & ($volume > Mean($volume, 20))"),
])
data = D.features(pool, ["$close"], "2025-01-01", "2025-12-31", allow_future=False)
```

| 类 / 函数 | 参数与语义 |
|---|---|
| `Filter` | 扩展基类，实现 `apply(provider, universe)` |
| `CompositeFilter(operation, filters)` | and/or/not 组合；通常通过 `&`、`|`、`~` 创建 |
| `ExpressionFilter(expression, filter_start_time=None, filter_end_time=None)` | 条件为真时保留；指定区间外不施加此条件 |
| `ExpressionDFilter(rule_expression, ...)` | 使用 rule_expression 参数指定保留条件 |
| `STFilter(exclude=True)` | 排除历史 ST；False 仅保留 ST |
| `ListingDaysFilter(min_days=0, max_days=None, trading_days=False)` | 上市天数范围；默认自然日，上市日为 0 |
| `MembershipFilter(market)` | 指定历史成分池 |
| `IndustryFilter(industry)` | 指定行业，如 `801780.SI` |
| `NameDFilter(name_rule_re)` | 对股票代码做正则匹配，不读取当前股票名称 |
| `TradableFilter()` | open、close、volume 均为正 |
| `make_filter(config)` | 接受 Filter 对象或配置字典，返回 Filter |

完整类、构造参数和方法签名见 {py:mod}`qlib.data.filter`。

```python
from qlib.data.filter import IndustryFilter, MembershipFilter

combined = (IndustryFilter("801780.SI") | MembershipFilter("csi300")) & STFilter()
configured = D.instruments("all", [
    {"filter_type": "STFilter"},
    {"filter_type": "ListingDaysFilter", "kwargs": {"min_days": 180}},
])
```

表达式过滤器禁用未来引用，NaN/inf 不通过，价格复权与同次查询一致。
按交易日计上市天数时，若本地日历开始时间晚于股票上市日会报错，避免把数据起点误当成上市起点。
