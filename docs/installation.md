# 安装

## 环境要求

- Python **3.10 或更新版本**；文档构建推荐 Python **3.12**。
- 核心依赖：NumPy、pandas、SciPy、joblib。
- 下载数据：额外安装 Tushare、tqdm，并配置账户 Token。
- 原生 DLL：Windows、与 Python 位数匹配的 MinGW-w64 GCC。

本项目的 Python 包名为 `qlib`，建议使用独立虚拟环境安装。

## 从项目源码安装

在项目根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

安装下载和文档依赖：

```powershell
.venv\Scripts\python.exe -m pip install -e ".[download,docs]"
```

Linux/macOS 的纯 Python 使用及文档构建：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[docs]'
```

## 编译 C 核心

```powershell
# GCC 已在 PATH 中；默认构建 rolling、expanding、PIT
.venv\Scripts\python.exe scripts/build_rolling.py

# 或指定 MinGW-w64 GCC
.venv\Scripts\python.exe scripts/build_rolling.py --cc D:/software/mingw64/bin/gcc.exe

# 只编译一个模块
.venv\Scripts\python.exe scripts/build_rolling.py --only pit
```

编译器必须是 MinGW-w64 GCC，不使用 MSVC。脚本会检查目标架构和 Python 位数，输出位于
`qlib/data/_libs/`。DLL 是本机生成文件；更新前应退出已加载它的 Python 进程。

未构建 DLL 时，表达式层的数值运算使用 pandas/NumPy 回退，PIT 查询也有回退。
直接调用 `rolling_*`、`expanding_*` 底层 Python 包装函数则需要对应 DLL。
文档构建只导入模块，不调用本地数据初始化或 DLL，因此 Read the Docs 的 Linux 环境可直接构建。

## 准备数据

已有数据时直接使用：

```python
import qlib

provider = qlib.init("~/.qlib/qlib_data/cn_data", adjust="hfq")
```

路径至少需要有效的 `calendars/day.txt`，完整使用还需要行情和股票池文件。
从 CSV 或 Tushare 构建的步骤见[数据维护](data.md)及[PIT 财务](pit.md)。

## 验证安装

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m sphinx -b html -W --keep-going docs docs/_build/html
```

原生数值测试需要先构建 DLL。文档生成后打开 `docs/_build/html/index.html`。
