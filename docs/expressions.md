# 表达式与算子

表达式使用安全 AST 解释器，支持算术、比较、逻辑和具名函数调用，不允许属性访问、导入或任意 Python 调用。
日线字段为 `$close`，财务字段为 `$$eps`、`$$roe` 等 Tushare 原名。

```python
values = D.features(
    "csi300",
    ["Mean($close, 20)", "$close / Ref($close, 20) - 1", "P($$eps)"],
    "2025-01-01", "2025-12-31", allow_future=False,
)
```

## 通用参数和返回值

`x/y/a/b` 表示按日期对齐的序列或可广播数值，`n` 为窗口长度，`q` 为分位水平。
算子在每只股票上计算；`D.features()` 将输出统一为指定日期的 Series，并把最终无穷值转换为 NaN。
滚动窗口从第一条数据开始计算（`min_periods=1`）；`n=0` 使用起点至当前的全部历史，Ref 的 0 另有定义。
查询起点前的完整历史会用于预热，避免滚动值因查询切片而改变。

## 全部表达式算子

以下覆盖当前 `OPERATORS` 注册表和 PIT 投影语法。部分函数仅注册在表达式环境中，应通过 `D.features()` 或 `OPERATORS[name]` 调用。

| 算子 | 含义 |
|---|---|
| `Ref(x,n)` | n>0 引用过去 n 个交易日；n=0 引用序列第一条；n<0 引用未来 |
| `Delta(x,n)` | x − Ref(x,n) |
| `Mean(x,n)` | 窗口均值 |
| `Sum(x,n)` | 窗口和 |
| `Std(x,n)` | 样本标准差，ddof=1 |
| `Var(x,n)` | 样本方差，ddof=1 |
| `Max(x,n)`、`Min(x,n)` | 窗口极值 |
| `Med(x,n)` | 中位数 |
| `Skew(x,n)`、`Kurt(x,n)` | pandas 窗口偏度、峰度 |
| `Count(x,n)` | 非缺失值计数 |
| `EMA(x,n)` | 指数加权均值；正整数为 span，0<n<1 的 float 为 alpha |
| `WMA(x,n)` | 时间位置权重从 1 递增的加权均值 |
| `Rank(x,n)` | 窗口内百分位排名，非横截面排名 |
| `IdxMax(x,n)`、`IdxMin(x,n)` | 极值在窗口中的位置，从 1 开始 |
| `Quantile(x,n,q)` | 窗口 q 分位数 |
| `Corr(x,y,n)`、`Cov(x,y,n)` | 窗口相关系数、样本协方差 |
| `Slope(x,n)` | 对时间位置做 OLS 的斜率 |
| `Rsquare(x,n)` | OLS 的 R² |
| `Resi(x,n)` | 当前末值减其 OLS 拟合值 |
| `Abs(x)`、`Sign(x)` | 绝对值、符号 |
| `Log(x)`、`Exp(x)`、`Sqrt(x)` | 自然对数、指数、平方根 |
| `Power(x,y)` | 幂 |
| `Add(x,y)`、`Sub(x,y)`、`Mul(x,y)`、`Div(x,y)` | 加、减、乘、除 |
| `Greater(x,y)`、`Less(x,y)` | 逐元素最大值、最小值 |
| `Gt(x,y)`、`Ge(x,y)`、`Lt(x,y)`、`Le(x,y)` | 大于、大于等于、小于、小于等于 |
| `Eq(x,y)`、`Ne(x,y)` | 等于、不等于 |
| `And(x,y)`、`Or(x,y)`、`Not(x)` | 布尔序列的与、或、取反 |
| `IsNull(x)`、`IsInf(x)` | 缺失值、无穷值检测 |
| `If(condition,a,b)` | 条件成立取 a，否则取 b |
| `Clip(x,low,high)` | 上下界截断 |
| `P(expr)` | 在可见的季度序列上计算，返回最新报告期位置 |
| `PRef(expr,offset)` | 返回相对于最新报告期的季度位置，offset 为非正整数 |

支持中缀 `+ - * / ** %`、`& |`、比较运算、一元正负号与取反。布尔条件建议使用带括号的比较和 `& |`。
底层辅助函数签名见 {py:mod}`qlib.data.ops`；解析和验证接口见 {py:mod}`qlib.data.base`。

## allow_future

- `False`：禁止负偏移的 Ref/Delta，用于因子、过滤和策略信号。
- `True`：允许使用本地已有的未来价格，用于收益标签；这是 `D.features()` 的默认值。

```python
# 因子：过去收益
factor = D.features("csi300", ["$close / Ref($close, 20) - 1"], allow_future=False)
# 标签：未来收益
label = D.features("csi300", ["Ref($close, -5) / $close - 1"], allow_future=True)
```

财务 PIT 始终按公告日期可见。`PRef(expr,-1)` 是上一自然季度，与日线 `Ref(x,-1)` 的偏移符号约定不同；P 内禁止未来季度引用。
