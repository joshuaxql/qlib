API 导航与约定
==============

所有 Python 函数、类及其方法均从当前源码生成签名与 docstring，并提供源码链接。
没有 docstring 的定义仍会列出签名；下划线开头的辅助函数位于各模块的“内部实现参考”。
动态表达式算子完整列于 :doc:`../expressions`，C 导出函数列于 :doc:`native`。

常用入口
--------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 用途
     - 入口
   * - 初始化
     - :func:`qlib.init`
   * - 数据读取
     - :class:`qlib.data.data.LocalProvider`
   * - 表达式
     - :class:`qlib.data.base.ExpressionEngine`
   * - 过滤器
     - :class:`qlib.data.filter.Filter`、:func:`qlib.data.filter.make_filter`
   * - 财务存储
     - :class:`qlib.data.pit.PITStore`、:func:`qlib.data.pit.write_stock`
   * - IC / RankIC
     - :func:`qlib.contrib.eva.alpha.calc_ic`
   * - 因子报告
     - :func:`qlib.contrib.report.analysis_model.analysis_model_performance.factor_analysis`
   * - 行业＋市值中性化
     - :func:`qlib.contrib.report.analysis_model.analysis_model_performance.neutralize_factors`
   * - 策略
     - :class:`qlib.contrib.strategy.signal_strategy.TopkStrategy`
   * - 回测
     - :func:`qlib.backtest.backtest.backtest`、:class:`qlib.backtest.executor.BacktestEngine`
   * - 交易配置
     - :class:`qlib.backtest.exchange.ExchangeConfig`
   * - 组合指标
     - :func:`qlib.contrib.evaluate.risk_analysis`

公共导出
--------

以下路径是便捷导出，函数文档仅在其定义模块登记一次：

.. code-block:: python

   from qlib.data import D, LocalProvider
   from qlib.contrib.strategy import TopkStrategy, WeightStrategy
   from qlib.contrib.report.analysis_model import (
       FactorAnalysisResult, calculate_factors, calculate_forward_returns,
       analyze_factors, factor_analysis, neutralize_factors,
   )
   from qlib.backtest import backtest, BacktestEngine, BacktestResult, ExchangeConfig

.. py:data:: qlib.data.D

   全局数据访问代理。``qlib.init()`` 将 LocalProvider 注册到此对象。
   ``D.calendar()``、``D.features()`` 等方法的参数与返回值见 LocalProvider。
   ``qlib.data.data.D`` 指向同一个实例。

返回值与参数指南
----------------

* 数据表、复权参数和 missing 策略见 :doc:`../data`。
* 表达式算子的参数、窗口和返回值见 :doc:`../expressions`。
* 因子指标输入索引、统计口径及各结果表见 :doc:`../factor`。
* 回测、交易配置和报告字段见 :doc:`../backtest`。
* 下载/构建函数需要源码目录中的 scripts 命名空间；配置项见 :doc:`../data`。

覆盖范围
--------

构建时自动扫描 ``qlib/**/*.py`` 与 ``scripts/**/*.py`` 中的模块级函数、类及直接定义的方法。
重导出的同一函数不会重复登记，函数体内的闭包不是独立 API。
生成后会检查每个发现的对象是否进入 Sphinx 索引；遗漏将导致严格构建失败。
覆盖清单输出到 HTML 构建目录中的 ``api-coverage.json``。
