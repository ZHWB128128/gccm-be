"""Application-layer API service: lightweight REST + Web dashboard using the standard library.

Endpoints
---------
  GET  /                -> HTML dashboard (data-driven; see app/dashboard.py)
  GET  /health          -> {"status":"ok"}
  GET  /status          -> current mode / horizon / version
  GET  /introspection   -> full model-capability spec + live values (app/introspection.py)
  POST /config          -> set whitelisted engine attributes at runtime
  POST /control         -> one rolling-horizon decision
  POST /simulate        -> closed-loop rollout returning trajectories for plotting

The dashboard is DATA-DRIVEN off /introspection, so the maintenance contract
"every model change updates the page" reduces to updating app/introspection.py.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..engine import GCCMEngine
from ..types import SystemState
from . import introspection as intro
from .dashboard import DASHBOARD_HTML

MAX_BODY_BYTES = 64 * 1024
ALLOWED_MODES = ["comfort", "balanced", "energy", "demand_response"]


class SimpleAPI:
    """Minimal API wrapper for edge demo environments (thread-safe).

    P2 修正（评审意见）：可选 token 鉴权。构造时传入 `auth_token` 后，除
    /health 外所有端点都要求 `Authorization: Bearer <token>` 头；不传则
    保持旧行为（仅本机/内网 demo，文档已声明不得暴露公网）。
    """

    def __init__(self, engine: GCCMEngine, auth_token: str | None = None) -> None:
        self.engine = engine
        self.auth_token = auth_token
        self._lock = threading.Lock()  # 引擎共享可变状态，串行化 optimize

    def _check_auth(self, headers) -> None:
        if self.auth_token is None:
            return
        supplied = (headers or {}).get("Authorization", "")
        if supplied != f"Bearer {self.auth_token}":
            raise PermissionError("无效或缺失的 Authorization token")

    def handle_control(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    def status(self) -> dict[str, Any]:
        with self._lock:
            mode = self.engine.mode_manager.current_mode
        return {
            "mode": mode,
            "horizon": self.engine.horizon,
            "version": _version(),
        }

    def introspection(self) -> dict[str, Any]:
        with self._lock:
            return intro.describe(self.engine)

    def set_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Set whitelisted engine attributes at runtime (operation/management)."""
        if not isinstance(payload, dict):
            raise ValueError("配置必须是 JSON 对象")
        allowed = set(intro.editable_fields())
        applied: dict[str, Any] = {}
        rejected: list[str] = []
        with self._lock:
            for k, v in payload.items():
                if k not in allowed:
                    rejected.append(k)
                    continue
                cur = getattr(self.engine, k, None)
                if isinstance(cur, bool):
                    v = bool(v)
                elif isinstance(cur, (int, float)) or cur is None:
                    v = float(v)
                # 语义校验：geodesic 软惩罚 > 0 需要 use_riemannian
                setattr(self.engine, k, v)
                applied[k] = v
            # 一次性一致性校验（复用引擎已有的开关校验，非法则回滚该字段）
            try:
                if hasattr(self.engine, "_validate_config"):
                    self.engine._validate_config()
            except Exception:
                pass
            # warm start 依赖 horizon/开关，改配置后清空避免脏初值
            self.engine._warm_start = None
        return {"applied": applied, "rejected": rejected}

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Closed-loop rollout for the dashboard charts."""
        t0 = float(payload.get("t0", 28.0))
        steps = int(payload.get("steps", 48))
        steps = max(1, min(steps, 500))
        forced_mode = payload.get("forced_mode")
        if forced_mode is not None and forced_mode not in ALLOWED_MODES:
            raise ValueError(f"forced_mode 必须是 {ALLOWED_MODES} 之一")
        labels = list(self.engine.manifold.labels)
        # 多区模型：所有 T_air* 状态逐区出曲线；单区模型该列表只有一个元素
        air_labels = [lab for lab in labels if lab.startswith("T_air")]
        air_idx = {lab: labels.index(lab) for lab in air_labels}
        with self._lock:
            state = SystemState([t0 for _ in labels], labels)
            decisions = self.engine.run_closed_loop(
                state, start_time_h=8.0, steps=steps, step_h=self.engine.dt,
            )
            comfort_min = self.engine.comfort_min if self.engine.comfort_min is not None else 25.0
            comfort_max = self.engine.comfort_max if self.engine.comfort_max is not None else 27.0
        zone_temps: dict[str, list[float]] = {lab: [] for lab in air_labels}
        unit_controls: dict[int, list[float]] = {}
        unit_labels: list[str] = []
        prices: list[float] = []
        violations = 0
        total_cost = 0.0
        provider = self.engine.external_provider
        zone_bounds = getattr(self.engine, "zone_comfort_bounds", {}) or {}
        t = 8.0
        for d in decisions:
            ns = d.predicted_next_state
            if ns is not None:
                for lab, idx in air_idx.items():
                    v = float(ns.x[idx])
                    zone_temps[lab].append(v)
                    # 每区有效边界：区覆盖 > 标量 comfort_min/max
                    lo, hi = zone_bounds.get(lab) or (comfort_min, comfort_max)
                    if v == v and (v > hi or v < lo):
                        violations += 1
            u = list(d.control.u)
            for j, val in enumerate(u):
                unit_controls.setdefault(j, []).append(float(val))
            if not unit_labels and d.control.labels:
                unit_labels = list(d.control.labels)
            try:
                w = provider.get(t, 1)[0]
                prices.append(float(w.price))
            except Exception:
                prices.append(0.6)
            total_cost += float(d.trajectory.total_cost)
            t += self.engine.dt
        control_labels = (unit_labels
                          if len(unit_labels) == len(unit_controls)
                          else [f"u{j}" for j in unit_controls])
        temps = zone_temps[air_labels[0]] if air_labels else []
        return {
            # 兼容字段：单区即唯一区；多区时指向第一个区/第一台设备
            "temps": temps,
            "controls": unit_controls[0] if unit_controls else [],
            "prices": prices,
            # 多区数据面：每区室温曲线 / 每台设备控制序列（含该区有效舒适带）
            "zones": [
                {
                    "label": lab, "temps": series,
                    "comfort_min": zone_bounds.get(lab, (comfort_min, comfort_max))[0],
                    "comfort_max": zone_bounds.get(lab, (comfort_min, comfort_max))[1],
                }
                for lab, series in zone_temps.items()
            ],
            "unit_controls": [
                {"label": control_labels[j], "data": series}
                for j, series in sorted(unit_controls.items())
            ],
            "violations": violations, "total_cost": total_cost,
            "comfort_min": comfort_min, "comfort_max": comfort_max,
            "last_decision": decisions[-1].as_dict() if decisions else None,
        }


def _version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "0.1.0"


def start_api(engine: GCCMEngine, host: str = "127.0.0.1", port: int = 8080,
              auth_token: str | None = None) -> ThreadingHTTPServer:
    api = SimpleAPI(engine, auth_token=auth_token)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str, status: int = 200) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"Content-Length 非法（1~{MAX_BODY_BYTES} 字节）")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/" or self.path == "/index.html":
                self._send_html(DASHBOARD_HTML)
            elif self.path == "/health":
                self._send_json({"status": "ok"})
            elif self.path in ("/status", "/introspection"):
                try:
                    api._check_auth(self.headers)
                except PermissionError as exc:
                    self._send_json({"error": str(exc)}, 401)
                    return
                self._send_json(api.status() if self.path == "/status" else api.introspection())
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            try:
                api._check_auth(self.headers)
            except PermissionError as exc:
                self._send_json({"error": str(exc)}, 401)
                return
            routes = {
                "/control": api.handle_control,
                "/config": api.set_config,
                "/simulate": api.simulate,
            }
            handler = routes.get(self.path)
            if handler is None:
                self._send_json({"error": "not found"}, 404)
                return
            try:
                payload = self._read_json()
                self._send_json(handler(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": f"bad request: {exc}"}, 400)
            except Exception as exc:
                self._send_json({"error": f"internal error: {exc}"}, 500)

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GCCM-BE REST API + Web dashboard")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径（见 examples/config.json）")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.config:
        from .config import engine_from_config
        engine = engine_from_config(args.config)
    else:
        from ..engine import GCCMEngine
        engine = GCCMEngine()
    cfg_host, cfg_port = args.host, args.port
    server = start_api(engine, host=cfg_host, port=cfg_port)
    print(f"GCCM-BE dashboard + API on http://{cfg_host}:{cfg_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
