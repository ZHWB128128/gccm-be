"""GCCM-BE launcher: start the Web dashboard + REST API.

Shared by both entry points (`python run.py` and `python -m gccm_be`) so they
behave identically and the logic stays importable after `pip install`.
"""
from __future__ import annotations

import argparse
import threading
import webbrowser


def build_engine(config_path: str | None):
    if config_path:
        from .config import engine_from_config
        return engine_from_config(config_path)
    from ..engine import GCCMEngine
    return GCCMEngine()


def _try_open(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass  # 无 GUI/无浏览器环境下静默跳过，服务照常运行


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gccm-be",
        description="GCCM-BE 控制引擎：启动 Web 控制台 + REST API",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1，仅本机；0.0.0.0 开放局域网）")
    parser.add_argument("--port", type=int, default=8080, help="端口（默认 8080）")
    parser.add_argument("--config", default=None,
                        help="引擎配置 JSON 路径（见 examples/config.json）")
    parser.add_argument("--no-browser", action="store_true",
                        help="不自动打开浏览器")
    args = parser.parse_args(argv)

    from .api import start_api

    engine = build_engine(args.config)
    server = start_api(engine, host=args.host, port=args.port)

    browse_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    url = f"http://{browse_host}:{args.port}/"

    banner = (
        "\n"
        "  +-----------------------------------------------+\n"
        "  |   GCCM-BE 控制台已启动                          |\n"
        "  +-----------------------------------------------+\n"
        f"    控制台:  {url}\n"
        f"    API:     http://{browse_host}:{args.port}/introspection\n"
        f"    配置:    {args.config or '默认引擎（GCCMEngine 默认参数）'}\n"
    )
    if args.host == "0.0.0.0":
        banner += "    警告:   绑定 0.0.0.0 且服务无鉴权，勿暴露到公网。\n"
    banner += "    按 Ctrl+C 停止。\n"
    print(banner, flush=True)

    if not args.no_browser:
        threading.Timer(0.6, lambda: _try_open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止…", flush=True)
    finally:
        server.server_close()
    return 0
