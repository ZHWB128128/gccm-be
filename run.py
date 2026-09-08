#!/usr/bin/env python3
"""GCCM-BE 一键启动程序。

启动 Web 控制台 + REST API（零外部依赖，纯标准库 HTTP），可选自动打开浏览器。

用法
----
    python run.py                      # 默认引擎，127.0.0.1:8080，自动开浏览器
    python run.py --port 9000          # 指定端口
    python run.py --config examples/config.json   # 从配置文件构建引擎
    python run.py --host 0.0.0.0       # 允许局域网访问（注意：服务无鉴权）
    python run.py --no-browser         # 不自动打开浏览器

等价入口：`python -m gccm_be`，或 `pip install -e .` 后的 `gccm-be` 命令。
"""
from __future__ import annotations

import sys

from gccm_be.app.launcher import main

if __name__ == "__main__":
    sys.exit(main())
