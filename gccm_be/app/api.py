"""应用层 API 服务：使用标准库提供轻量 REST 接口。"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from ..engine import GCCMEngine
from ..types import SystemState

MAX_BODY_BYTES = 64 * 1024
ALLOWED_MODES = ["comfort", "balanced", "energy", "demand_response"]


class SimpleAPI:
    """极简 API 封装，适用于边缘演示环境（线程安全）。"""

    def __init__(self, engine: GCCMEngine) -> None:
        self.engine = engine
        self._lock = threading.Lock()  # 引擎共享可变状态，串行化 optimize

    def handle_control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload.get("state"), (list, tuple)) or len(payload["state"]) == 0:
            raise ValueError("payload['state'] 必须是非空数值列表")
        state = SystemState(payload["state"], payload.get("labels"))
        time_h = float(payload.get("time_h", 8.0))
        prev = payload.get("prev_control")
        prev_control = None
        if prev is not None:
            if not isinstance(prev, (list, tuple)) or len(prev) == 0:
                raise ValueError("prev_control 必须是非空数值列表")
            from ..types import ControlInput
            prev_control = ControlInput(prev, payload.get("control_labels"))
        forced_mode = payload.get("forced_mode")
        if forced_mode is not None and forced_mode not in ALLOWED_MODES:
            raise ValueError(f"forced_mode 必须是 {ALLOWED_MODES} 之一，收到 {forced_mode!r}")
        # 引擎 optimize 会写共享可变状态（warm start/模式/降级标志），必须加锁
        with self._lock:
            decision = self.engine.optimize(state, time_h, prev_control=prev_control, forced_mode=forced_mode)
        return decision.as_dict()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            mode = self.engine.mode_manager.current_mode
        return {
            "mode": mode,
            "horizon": self.engine.horizon,
            "version": _version(),
        }


def _version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "0.1.0"


def start_api(engine: GCCMEngine, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    api = SimpleAPI(engine)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"status": "ok"})
            elif self.path == "/status":
                self._send_json(api.status())
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/control":
                self._send_json({"error": "not found"}, 404)
                return
            try:
                length_str = self.headers.get("Content-Length", "0")
                length = int(length_str)
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError(f"Content-Length 非法（1~{MAX_BODY_BYTES} 字节）")
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                result = api.handle_control(payload)
                self._send_json(result)
            except (ValueError, json.JSONDecodeError) as exc:
                # 客户端错误：请求格式/参数问题
                self._send_json({"error": f"bad request: {exc}"}, 400)
            except Exception as exc:  # noqa: BLE001
                # 服务端错误：引擎/求解器内部故障
                self._send_json({"error": f"internal error: {exc}"}, 500)

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GCCM-BE REST API")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径（见 examples/config.json）")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.config:
        from .config import engine_from_config
        engine = engine_from_config(args.config)
        cfg_host, cfg_port = args.host, args.port
    else:
        from ..engine import GCCMEngine
        engine = GCCMEngine()
        cfg_host, cfg_port = args.host, args.port
    server = start_api(engine, host=cfg_host, port=cfg_port)
    print(f"GCCM-BE API listening on http://{cfg_host}:{cfg_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
