#!/usr/bin/env python3
"""Unified health check (10H).

Checks: backend, inference service, PostgreSQL, frontend, PLC, OPC UA, MES,
deployment manifest and model artifacts. Outputs a status per check plus an
overall verdict:

  READY            - core stack + industrial integration + manifest all good
  NOT_INTEGRATED   - core + manifest good, industrial simulators absent
                     (the system runs in its honest NOT_INTEGRATED mode)
  DEGRADED         - core good, but some optional component is down
  FAILED           - a core component or the deployment boundary failed

Usage:  bash scripts/run_clean.sh python scripts/health_check.py
"""

from __future__ import annotations

import json
import socket
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BACKEND = "http://127.0.0.1:8000"
INFERENCE = "http://127.0.0.1:8100"
FRONTEND = "http://127.0.0.1:5173"
PLC = "http://127.0.0.1:8501"
MES = "http://127.0.0.1:8502"
OPCUA = ("127.0.0.1", 8503)
PG_PORT = 5433
MANIFEST = ROOT / "backend" / "config" / "deployment_manifest.yaml"


def _http(url: str, timeout: float = 3.0) -> tuple[int, dict | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body)
            except Exception:  # noqa: BLE001
                return resp.status, None
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)}


def _tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest() -> tuple[str, str]:
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "backend"))
    try:
        from app.mlops.manifest import load_manifest, validate_artifacts

        m = load_manifest(MANIFEST)
        problems = validate_artifacts(m, ROOT)
        if problems:
            return "FAILED", "; ".join(problems)
        return "READY", f"vision_stack {m['vision_stack_version']} (yolo {m['yolo']['version']}, patchcore {m['patchcore']['version']})"
    except Exception as exc:  # noqa: BLE001
        return "FAILED", f"manifest load failed: {exc}"


def main() -> int:
    results: dict[str, tuple[str, str]] = {}

    # ---- core ----
    st, body = _http(f"{BACKEND}/ready")
    results["backend"] = ("READY" if st == 200 and body.get("status") == "ready" else "FAILED",
                          f"HTTP {st} {json.dumps(body or {}, ensure_ascii=False)[:120]}")
    st, body = _http(f"{INFERENCE}/ready")
    ok = st == 200 and body.get("status") == "ready"
    results["inference"] = ("READY" if ok else "FAILED",
                            f"HTTP {st} {json.dumps(body or {}, ensure_ascii=False)[:120]}")
    results["postgres"] = ("READY" if _tcp("127.0.0.1", PG_PORT) else "FAILED", f"tcp 127.0.0.1:{PG_PORT}")
    st, _ = _http(f"{FRONTEND}", timeout=2.0)
    results["frontend"] = ("READY" if st == 200 else "DEGRADED", f"HTTP {st}")

    # ---- deployment boundary ----
    results["manifest"] = check_manifest()
    yolo_art = ROOT / "inference-service" / "models" / "best.pt"
    pc_art = ROOT / "inference-service" / "models" / "patchcore-bottle" / "bank.npz"
    missing = [p.name for p in (yolo_art, pc_art) if not p.exists()]
    results["model_artifacts"] = ("READY" if not missing else "FAILED", "present" if not missing else f"missing {missing}")

    # ---- industrial (optional -> NOT_INTEGRATED when absent) ----
    st, body = _http(f"{PLC}/v1/state")
    results["plc_http"] = ("READY" if st == 200 else "NOT_INTEGRATED", f"HTTP {st} {json.dumps(body or {}, ensure_ascii=False)[:80]}")
    results["opcua"] = ("READY" if _tcp(*OPCUA) else "NOT_INTEGRATED", f"tcp {OPCUA[0]}:{OPCUA[1]}")
    st, _ = _http(f"{MES}/v1/products/x")
    results["mes"] = ("READY" if st == 200 else "NOT_INTEGRATED", f"HTTP {st}")

    # ---- verdict ----
    core_failed = any(results[k][0] == "FAILED" for k in ("backend", "inference", "postgres", "manifest", "model_artifacts"))
    industrial_absent = all(results[k][0] == "NOT_INTEGRATED" for k in ("plc_http", "opcua", "mes"))
    degraded = any(results[k][0] == "DEGRADED" for k in results)

    if core_failed:
        overall = "FAILED"
    elif industrial_absent and not degraded:
        overall = "NOT_INTEGRATED"
    elif degraded:
        overall = "DEGRADED"
    else:
        overall = "READY"

    print(f"{'component':<16} {'status':<15} detail")
    print("-" * 90)
    for k, (status, detail) in results.items():
        print(f"{k:<16} {status:<15} {detail}")
    print("-" * 90)
    print(f"OVERALL: {overall}")
    return 0 if overall in ("READY", "NOT_INTEGRATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
