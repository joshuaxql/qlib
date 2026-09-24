C 与 ctypes API
===============

Python 包装函数
----------------

* :mod:`qlib.data._libs.rolling`：``rolling_mean/rolling_slope/rolling_rsquare/rolling_resi``。
* :mod:`qlib.data._libs.expanding`：对应的 ``expanding_*`` 四个函数。
* :mod:`qlib.data._libs.pit`：``asof_indices(dates, starts, counts, asof)``。

Python 数值包装接受一维数据，转为连续且对齐的 float64 数组，返回等长新数组。
rolling 的 window 必须是正整数；expanding 没有窗口参数。
NaN 忽略但回归时间位置保留，有效值不足两个时回归结果为 NaN，常数窗口 R² 为 NaN，末值缺失时残差为 NaN。
底层数值 API 只将 NaN 视为缺失，表达式层会将无穷值转换为缺失。

PIT 输入日期为 uint32，starts/counts 为 uint64；每组日期有序。
返回 int64 记录位置数组，取公告日期不晚于 asof 的最后版本，不存在则为 -1。

原生构建方式见 :doc:`../installation`。头文件不依赖 Python 或 NumPy C ABI。

Rolling C 接口
--------------

.. c:function:: int qlib_rolling_mean(const double *input, size_t length, size_t window, double *output)

   对每个末端位置计算窗口均值。

.. c:function:: int qlib_rolling_slope(const double *input, size_t length, size_t window, double *output)

   对保留时间位置的有效样本计算 OLS 斜率。

.. c:function:: int qlib_rolling_rsquare(const double *input, size_t length, size_t window, double *output)

   返回 OLS 的 R²。

.. c:function:: int qlib_rolling_resi(const double *input, size_t length, size_t window, double *output)

   返回窗口末值减对应拟合值。

输入和输出是 length 个 double 的非重叠数组，window > 0；从第一个位置开始计算部分窗口。
返回 0 表示成功、1 表示参数错误。空数组指针可为 NULL。函数不分配内存。

.. literalinclude:: ../../qlib/data/_libs/rolling.h
   :language: c
   :start-at: #ifndef

Expanding C 接口
----------------

.. c:function:: int qlib_expanding_mean(const double *input, size_t length, double *output)

   每个位置使用从起点至当前的全部历史计算均值。

.. c:function:: int qlib_expanding_slope(const double *input, size_t length, double *output)

   累计 OLS 斜率。

.. c:function:: int qlib_expanding_rsquare(const double *input, size_t length, double *output)

   累计 OLS 的 R²。

.. c:function:: int qlib_expanding_resi(const double *input, size_t length, double *output)

   当前值减累计窗口的末端拟合值。

指针、长度及状态码约定与 rolling 相同。

.. literalinclude:: ../../qlib/data/_libs/expanding.h
   :language: c
   :start-at: #ifndef

PIT C 接口
----------

.. c:function:: int qlib_pit_asof(const uint32_t *dates, size_t length, const uint64_t *starts, const uint64_t *counts, size_t groups, uint32_t asof, int64_t *output)

   对 groups 个连续版本区间分别二分查找最后一个 date <= asof 的位置。
   starts/counts 和 output 长度均为 groups；返回位置是 dates 数组下标，不是文件字节偏移。
   找不到则输出 -1。返回 0 表示成功、1 表示参数错误。调用者保证各组日期升序。

.. literalinclude:: ../../qlib/data/_libs/pit.h
   :language: c
