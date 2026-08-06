"""Phase 8: MLOps API integration (registry, monitoring, feedback, drift)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


YOLO = {
    "model_name": "neu-yolov8s-e2e",
    "model_version": "1.0.0",
    "model_type": "yolo",
    "artifact_uri": "inference-service/models/best.pt",
    "artifact_sha256": "abc123",
    "dataset_version": "neu-det-yolo-v1",
    "training_run_id": "run-1",
    "metrics": {"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5},
    "domain_validated": True,
}


async def test_register_and_promote(client, db_session):
    r = await client.post("/api/v1/models", json=YOLO)
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    assert r.json()["status"] == "CANDIDATE"

    # gate dry-run
    g = await client.post(f"/api/v1/models/{mid}/gate", json={"required_domain": "steel"})
    assert g.status_code == 200
    assert g.json()["gate"]["passed"] is True

    # promote (gate passes: metrics ok + domain validated)
    p = await client.post(f"/api/v1/models/{mid}/promote", json={"required_domain": "steel"})
    assert p.status_code == 200, p.text
    assert p.json()["status"] == "PRODUCTION"

    prod = await client.get("/api/v1/models/production/neu-yolov8s-e2e")
    assert prod.status_code == 200
    assert prod.json()["model_version"] == "1.0.0"


async def test_duplicate_version_rejected(client, db_session):
    await client.post("/api/v1/models", json=YOLO)
    r = await client.post("/api/v1/models", json=YOLO)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "duplicate_version"


async def test_promote_rejected_when_domain_not_validated(client, db_session):
    body = {**YOLO, "model_name": "mvtec-patchcore-e2e", "model_type": "patchcore",
            "metrics": {"image_auroc": 1.0, "pixel_auroc": 0.986, "latency_ms": 755.0},
            "domain_validated": False}
    r = await client.post("/api/v1/models", json=body)
    mid = r.json()["id"]
    # perfect AUROC but steel domain NOT validated -> gate rejects (8F)
    g = await client.post(f"/api/v1/models/{mid}/gate", json={"required_domain": "steel"})
    assert g.json()["gate"]["passed"] is False
    assert any("domain" in b for b in g.json()["gate"]["blocked"])
    p = await client.post(f"/api/v1/models/{mid}/promote", json={"required_domain": "steel"})
    assert p.status_code == 422
    assert p.json()["error"]["code"] == "promotion_gate_failed"


async def test_rollback_switches_production(client, db_session):
    await client.post("/api/v1/models", json=YOLO)
    r2 = await client.post("/api/v1/models", json={**YOLO, "model_version": "2.0.0"})
    m2 = r2.json()["id"]
    await client.post(f"/api/v1/models/{m2}/promote", json={"required_domain": "steel"})

    # v2 is production; rollback to v1
    rb = await client.post("/api/v1/models/rollback", json={"model_name": "neu-yolov8s-e2e", "model_version": "1.0.0"})
    assert rb.status_code == 200
    prod = await client.get("/api/v1/models/production/neu-yolov8s-e2e")
    assert prod.json()["model_version"] == "1.0.0"


async def test_model_metrics_and_human_feedback_endpoints(client, db_session):
    m = await client.get("/api/v1/model-metrics")
    assert m.status_code == 200
    assert "inference_count" in m.json()

    f = await client.get("/api/v1/human-feedback")
    assert f.status_code == 200
    assert "defect_confirmation_rate" in f.json()

    d = await client.get("/api/v1/drift")
    assert d.status_code == 200
    assert d.json()["overall"] in ("NORMAL", "WARNING", "CRITICAL")
    assert "data drift only" in d.json()["note"]


async def test_training_candidate_source_identity_fields(client, db_session):
    """Semantic fix (Phase 9): dataset / model / deployment identities are
    distinct fields on the retraining candidate manifest."""
    cands = (await client.get("/api/v1/training-candidates", params={"kind": "all"})).json()
    for c in cands:
        assert "source_dataset_version" in c
        assert "source_model_version" in c
        assert "source_deployment_version" in c
        assert "dataset_version" not in c
        assert "model_version" not in c
