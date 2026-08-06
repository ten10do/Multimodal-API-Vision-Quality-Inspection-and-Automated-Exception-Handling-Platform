"""Phase 8 real E2E (8O): registry -> promote -> inspection traceability ->
metrics -> feedback -> drift -> candidate -> promote v2 -> rollback.

Requires the live stack (backend 8000 with the new /api/v1/models routes,
inference service 8100). No retraining: existing artifacts are registered
with explicit versions.

Run:  bash scripts/run_clean.sh python scripts/mlops_e2e.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BACKEND = "http://127.0.0.1:8000"
OUT = ROOT / "docs" / "phase8-e2e.json"

YOLO_V1 = {
    "model_name": "neu-yolov8s",
    "model_version": "1.0.0",
    "model_type": "yolo",
    "artifact_uri": "model-training/runs/neu-det-yolov8s-baseline-2/weights/best.pt",
    "artifact_sha256": "9c9409aae38b18dbb3fb0bd12fb7cdccd64032347b2ea07b9a2656d7134b48a1",
    "dataset_version": "neu-det-yolo-v1",
    "training_run_id": "e2e-run-1",
    "metrics": {"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5},
    "domain_validated": True,
    "notes": "E2E v1",
}
YOLO_V2 = {**YOLO_V1, "model_version": "2.0.0", "metrics": {"mAP50": 0.9, "recall": 0.85, "latency_p95_ms": 18.0},
           "training_run_id": "e2e-run-2", "notes": "E2E v2"}


def _post_image() -> dict:
    img = next(iter((ROOT / "model-training/datasets/neu-det-yolo/test/images").glob("*.jpg")))
    name = f"e2e8-{id(img) % 1000000}.jpg"
    r = httpx.post(
        f"{BACKEND}/api/v1/inspections",
        files={"file": (name, img.read_bytes(), "image/jpeg")},
        data={"product_id": "P-MLOPS-E2E", "production_line": "line-a", "station": "qc"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def _cleanup_models() -> None:
    """Delete any previously registered neu-yolov8s rows so the E2E is
    idempotent."""
    rows = httpx.get(f"{BACKEND}/api/v1/models", timeout=10).json()
    for m in rows:
        if m["model_name"] == "neu-yolov8s":
            httpx.delete(f"{BACKEND}/api/v1/models/{m['id']}", timeout=10)


def main() -> int:
    checks: dict = {}

    # 0. readiness of the new endpoints
    r = httpx.get(f"{BACKEND}/ready", timeout=5)
    r.raise_for_status()
    _cleanup_models()

    # 1. register v1
    r = httpx.post(f"{BACKEND}/api/v1/models", json=YOLO_V1, timeout=10)
    r.raise_for_status()
    v1 = r.json()
    assert v1["status"] == "CANDIDATE"
    checks["register_v1"] = {"id": v1["id"], "status": v1["status"]}

    # 2. promote v1 (gate passes)
    r = httpx.post(f"{BACKEND}/api/v1/models/{v1['id']}/promote", json={"required_domain": "steel"}, timeout=10)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "PRODUCTION"
    checks["promote_v1"] = {"version": "1.0.0", "gate_passed": r.json()["gate"]["passed"]}

    # 3. inspection -> deployment_version stamped + production pointer
    insp = _post_image()
    assert insp.get("deployment_version") == "2026.08.1", insp
    prod = httpx.get(f"{BACKEND}/api/v1/models/production/neu-yolov8s", timeout=10).json()
    assert prod["model_version"] == "1.0.0"
    checks["inspection_traceability"] = {"inspection_id": insp["inspection_id"], "deployment_version": insp["deployment_version"], "production": prod["model_version"]}

    # 4. model metrics
    m = httpx.get(f"{BACKEND}/api/v1/model-metrics", timeout=10).json()
    assert m["inference_count"] >= 1
    checks["model_metrics"] = {"inference_count": m["inference_count"], "error_rate": m["error_rate"], "p95": m["inference_latency_p95_ms"]}

    # 5. register v2, promote (v1 archived automatically)
    r = httpx.post(f"{BACKEND}/api/v1/models", json=YOLO_V2, timeout=10)
    r.raise_for_status()
    v2 = r.json()
    r = httpx.post(f"{BACKEND}/api/v1/models/{v2['id']}/promote", json={"required_domain": "steel"}, timeout=10)
    assert r.status_code == 200, r.text
    prod2 = httpx.get(f"{BACKEND}/api/v1/models/production/neu-yolov8s", timeout=10).json()
    assert prod2["model_version"] == "2.0.0"
    checks["promote_v2"] = {"version": "2.0.0"}

    # 6. rollback to v1 -> production pointer switches, no rebuild
    r = httpx.post(f"{BACKEND}/api/v1/models/rollback", json={"model_name": "neu-yolov8s", "model_version": "1.0.0"}, timeout=10)
    assert r.status_code == 200, r.text
    prod3 = httpx.get(f"{BACKEND}/api/v1/models/production/neu-yolov8s", timeout=10).json()
    assert prod3["model_version"] == "1.0.0"
    checks["rollback"] = {"production_after_rollback": prod3["model_version"]}

    # 7. drift + feedback endpoints respond
    d = httpx.get(f"{BACKEND}/api/v1/drift", timeout=10).json()
    assert d["overall"] in ("NORMAL", "WARNING", "CRITICAL")
    f = httpx.get(f"{BACKEND}/api/v1/human-feedback", timeout=10).json()
    assert "defect_confirmation_rate" in f
    checks["drift"] = {"overall": d["overall"]}
    checks["feedback"] = {"resolved": f["resolved"]}

    OUT.write_text(__import__("json").dumps(checks, indent=2, ensure_ascii=False))
    print(__import__("json").dumps(checks, indent=2, ensure_ascii=False))
    print("PHASE 8 E2E: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
