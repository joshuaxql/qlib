# 文档构建与 Read the Docs

本站使用 Sphinx、MyST Markdown 和 **sphinx_rtd_theme**。Sphinx 负责 API 签名、索引与源码链接，MyST 支持 Markdown 使用指南。

## 本地构建

从项目根目录执行：

```powershell
.venv\Scripts\python.exe -m pip install -e ".[docs]"
.venv\Scripts\python.exe -m sphinx -b html -W --keep-going docs docs/_build/html
```

打开 `docs/_build/html/index.html`，或通过本地 HTTP 服务预览：

```powershell
.venv\Scripts\python.exe -m http.server 8000 --directory docs/_build/html
```

浏览器访问 `http://localhost:8000`。服务保持运行期间可浏览所有页面，Ctrl+C 停止。
已激活虚拟环境时，也可使用 Windows 的 `docs\make.bat html`，或在 docs 内运行 `make html`。

`-W --keep-going` 会将警告作为构建失败处理，并收集完整诊断。
文档仅导入 Python 定义，不需要 Tushare Token、本地行情或已编译 DLL。

## Read the Docs 配置

仓库根目录提供 `.readthedocs.yaml`，配置如下：

- Ubuntu 24.04、Python 3.12；
- 通过 `pip install .[docs]` 安装项目与文档依赖；
- 使用 `docs/conf.py`；
- 开启 `fail_on_warning`；
- 生成网站及可下载的 HTML ZIP。

将仓库推送到 GitHub 后，在 Read the Docs 导入该仓库，配置文件路径使用 `.readthedocs.yaml`，启动首次构建。
连接 GitHub Webhook 后，后续推送会自动触发构建；分支、版本和域名在 Read the Docs 项目设置中管理。
本仓库配置可以用于托管构建；在线站点地址以实际导入的 Read the Docs 项目为准。

## 编辑文档

- `docs/index.rst`：站点目录；新增使用指南后加入 toctree。
- `docs/*.md`：功能、安装与使用说明。
- `docs/api/native.rst`：C 导出 API 与头文件。
- `docs/_ext/api_reference.py`：自动发现 Python 函数、类、方法并生成 API 页。
- `docs/_generated/`：构建时生成，不手动维护、不提交。
- `docs/_build/`：构建输出，不提交。

API 页面签名和已有 docstring 来自代码。类的构造方法、私有方法和没有 docstring 的函数也会生成条目。
源码新增函数后，下次构建会自动纳入；覆盖检查确保发现的对象进入 Sphinx 索引，并写入 `api-coverage.json`。
源码内新增丰富的参数/返回值说明会自动出现在站点中。
