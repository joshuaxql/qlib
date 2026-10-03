# 安装

## 环境要求

- Python **3.10 或更新版本**；文档构建推荐 Python **3.12**。
- 核心依赖：NumPy、pandas、SciPy、joblib、Loguru、pyecharts。
- 下载数据：额外安装 Tushare，并配置账户 Token。
- 平台 wheel 安装不需要编译器。源码构建需要 Windows MinGW-w64 GCC、Linux GCC 或 macOS Clang，编译器目标必须与 Python 架构匹配。

PyPI 发行包名为 **`qlib-joshuaxql`**，Python 导入名为 **`qlib`**，建议使用独立虚拟环境安装。

## 从 PyPI 安装

```bash
python -m pip install qlib-joshuaxql
```

[PyPI 项目页面](https://pypi.org/project/qlib-joshuaxql/) 提供版本信息、wheel 和源码压缩包。
当前版本 **0.3.0** 提供五个平台预编译 wheel，以及由同一版本源码生成的新源码包（sdist）。
平台 wheel 内置 rolling、expanding 和 PIT 三个预编译库，提供数据读取、因子分析和日频回测。
覆盖 Windows x86_64、Linux x86_64/aarch64 和 macOS arm64/x86_64；pip 自动选择兼容的平台文件。
Linux wheel 采用 `manylinux_2_17` / `manylinux2014` 标签；macOS 的最低系统版本以最终 wheel 的 `macosx_*` 标签为准。
NumPy、pandas 和 SciPy 仍各自遵循其版本的系统要求。
数据集需单独下载或构建，见下方“准备数据”。

```python
import qlib

print(qlib.__version__)
provider = qlib.init("~/.qlib/qlib_data/cn_data")
```

升级及可选依赖：

```bash
python -m pip install --upgrade qlib-joshuaxql
python -m pip install "qlib-joshuaxql[download,docs]"
```

`download` 安装 Tushare；`docs` 安装文档构建依赖。
Loguru 随核心包自动安装；日志仅保留关键事件，不再依赖 tqdm 进度条，配置见[日志](logging.md)。
wheel 提供 `qlib` 库与 C 源文件；运行本页及其他指南中的 `scripts/`、`tests/`、`docs/` 命令，
请使用 Git 仓库或 PyPI 源码压缩包的根目录。

从 0.2.0 升级到 0.3.0 时，使用上方普通升级命令即可；pip 自动选择兼容的平台 wheel。

## 从项目源码安装

获取源码并在项目根目录执行：
先准备目标平台的 C 编译器；Windows 使用 MinGW-w64 GCC，不使用 MSVC。

```powershell
git clone https://github.com/joshuaxql/qlib.git
cd qlib
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

安装下载和文档依赖：

```powershell
.venv\Scripts\python.exe -m pip install -e ".[download,docs]"
```

Linux/macOS 的源码安装及文档构建：

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

Linux/macOS 使用对应的解释器和 GCC/Clang：

```bash
.venv/bin/python scripts/build_rolling.py
# 显式选择编译器
.venv/bin/python scripts/build_rolling.py --cc clang
```

脚本检查目标架构，输出位于 `qlib/data/_libs/`：Windows 为 `.dll`，Linux 为 `.so`，macOS 为 `.dylib`。
可使用 `QLIB_CC` 或 `CC` 指定源码包构建的编译器，优先级为 `QLIB_CC`、`CC`、平台默认编译器。
macOS 构建还支持 `ARCHFLAGS` 与 `MACOSX_DEPLOYMENT_TARGET`。
修改或替换已加载的原生库前，退出使用它的 Python 进程。

未找到原生库时，表达式计算和 PIT 查询使用 pandas/NumPy 回退；直接调用底层
`rolling_*`、`expanding_*` 包装函数需要对应原生库。
既有 0.2.0 通用 wheel 没有这些库，仍可使用回退；其源码包也保留原发布内容。
新源码构建和多平台 wheel 验证见[原生 wheel 构建](native-wheels.md)。

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

源码中的原生数值测试需要先构建本机库。已安装平台 wheel 时，可在新源码目录运行：

```bash
python -I scripts/verify_native.py --require-installed
```

此命令拒绝导入旁边的源码包，强制校验三个库及表达式原生路径，任何 Python 回退均导致失败。
文档生成后打开 `docs/_build/html/index.html`。
