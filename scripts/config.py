"""数据构建配置；目录相对于项目根目录，无命令行参数。"""

import os

from scripts.tushare.fields import FINA_INDICATOR_FIELDS

TOKEN = os.environ.get("TUSHARE_TOKEN", "")
OUTPUT_DIR = "~/.qlib/qlib_data/cn_data"
CACHE_DIR = "E:/data/data_cache"
START_DATE = "20100101"
REFRESH_CACHE = False
REQUEST_INTERVAL = 0.12  # 秒；根据账户接口频率权限调整
RETRIES = 3  # 首次请求之外的重试次数，即最多请求4次
PAGE_SIZE = 1000
INDEX_CODES = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
    "csi800": "000906.SH",
    "csi1000": "000852.SH",
}
# 接口实际数据起点，不是指数基日或发布日期。
INDEX_START_DATES = {"000852.SH": "20141031"}

# 每股仅两个文件；下载 fina_indicator 全部数值字段。设 False 可只维护日线。
BUILD_PIT = True
PIT_FIELDS = {"fina_indicator": FINA_INDICATOR_FIELDS}
PIT_REFRESH_QUARTERS = 8  # 更新近两年报告期；REFRESH_CACHE=True 则刷新所有历史期
