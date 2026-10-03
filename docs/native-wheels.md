# 原生 wheel 构建

平台 wheel 内置三个 ctypes 数值库：rolling、expanding、PIT。
C 头文件不使用 Python 或 NumPy C API，因此 wheel 标签为 `py3-none-<平台>`；
Python 最低版本由包元数据中的 `Requires-Python >=3.10` 限制。
每个目标平台只构建一个 wheel，并用多个 Python 版本验证同一文件。

## 构建目标

| 平台 | 架构 | 编译与检查 | CI 安装验证 |
|---|---|---|---|
| Windows | x86_64 | MinGW-w64 GCC / UCRT64 | Python 3.10、3.14 |
| Linux | x86_64、aarch64 | manylinux2014 容器中的 GCC；auditwheel 检查和修复 | Python 3.10、3.14 |
| macOS | arm64、x86_64 | Clang；delocate 检查依赖与架构 | Python 3.11、3.14 |

Linux 原生库目标为 glibc 2.17，修复后的文件可包含等价的
`manylinux_2_17` 与 `manylinux2014` 两个标签。
macOS 构建请求 `MACOSX_DEPLOYMENT_TARGET=11.0`；最终最低版本以 delocate 验证后文件的
`macosx_*` 标签为准。当前未构建 Windows ARM64、32 位 Windows 或 Alpine/musl wheel。

从源码构建 wheel 时，每次重新编译 C 库，不复制源码目录里残留的 DLL/SO；
编译器缺失、目标架构不匹配或编译失败会直接使构建失败。
`QLIB_CC` 优先于 `CC`；macOS 支持 `ARCHFLAGS` 和 `MACOSX_DEPLOYMENT_TARGET`。
本地构建命令：

```bash
python -m pip install build
python -m build
```

`dist/` 中的本地 Linux wheel 仍是 `linux_*` 标签，公开分发前需在相应 manylinux 环境构建并执行 auditwheel。
完整的库调用语义见 [C 与 ctypes API](api/native.rst)。

## 手动运行 GitHub Actions

工作流为 `.github/workflows/native-wheels.yml`，名称为 **Native wheels**。
仅由 `workflow_dispatch` 触发，无自定义输入，所选 Git ref 决定构建源码。
工作流文件需先位于仓库默认分支；也可用 GitHub CLI 指定待构建分支：

```bash
gh workflow run native-wheels.yml --ref main
gh run list --workflow native-wheels.yml
gh run watch RUN_ID
gh run download RUN_ID --dir native-dist
```

构建权限为 `contents: read`。执行者需要仓库的 Actions 调度权限；
工作流不需要 PyPI Token、Tushare Token、发布 secret 或 `id-token: write`。
构建和数值验证全部成功后，各平台上传以下独立 artifacts：

- `native-wheel-windows-x86_64`
- `native-wheel-linux-x86_64`
- `native-wheel-linux-aarch64`
- `native-wheel-macos-arm64`
- `native-wheel-macos-x86_64`

每份 artifact 包含一个 wheel、`SHA256SUMS`、`source-commit.txt` 和两个 Python 版本的验证 JSON。
安装验证使用 `python -I scripts/verify_native.py --require-installed`，拒绝导入源码目录中的 `qlib`。
它强制加载三个库，以独立 NumPy/最小二乘/PIT 参照检查 13 个原生函数与表达式路径；
Python 回退会使验证失败。Windows 验证时移除编译器目录的 PATH，防止编译环境掩盖 DLL 依赖。

## 上传经过验证的文件

工作流只上传 Actions artifacts。维护者检查源码提交、SHA256 和验证 JSON 后，
在自己的发布环境使用 PyPI 凭据上传下载的 wheel：

```bash
python -m twine check "native-dist/*/*.whl"
python -m twine upload "native-dist/*/*.whl"
```

0.2.0 已发布的 `py3-none-any` wheel 与源码包保留；平台 wheel 使用不同文件名追加至同一版本。
PyPI 已存在的文件名不能覆盖。已安装旧通用 wheel 的用户按[安装指南](installation.md)重新安装同版本。

## 官方工具资料

- [GitHub hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)：当前 runner 标签与架构。
- [MSYS2 setup action](https://github.com/msys2/setup-msys2)：UCRT64 安装与 `msys2-location` 输出。
- [manylinux](https://github.com/pypa/manylinux)：构建镜像和 glibc 基线。
- [auditwheel](https://github.com/pypa/auditwheel)：Linux 依赖检查与 wheel 修复。
- [delocate](https://github.com/matthew-brett/delocate)：macOS 动态库依赖与架构检查。
- [Python Packaging compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)：Python、ABI 和平台标签语义。
