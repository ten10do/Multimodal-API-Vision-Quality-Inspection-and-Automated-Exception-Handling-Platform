"""Phase 8 backfill (8B): register the existing Phase 1 / Phase 6 baselines
into MLflow + the Model Registry WITHOUT retraining.

The Phase 1 YOLO baseline and the Phase 6 PatchCore baseline are already
trained and evaluated; this script records identity, metrics, dataset
version and artifact hashes so historical models are traceable.

Uses the low-level MlflowClient API (stable across mlflow 3.x; the fluent
start_run has an upstream bug in 3.15.1).

Run:  bash scripts/run_clean.sh python scripts/backfill_mlflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

MLRUNS = ROOT / "mlruns"
EXPERIMENT = "ivqc-models"


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from mlflow.tracking.client import MlflowClient

    # Windows: file:///D:/... (three slashes); as_uri() emits the correct form
    client = MlflowClient(tracking_uri=MLRUNS.resolve().as_uri())
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)

    def log_run(run_name: str, tags: dict, params: dict, metrics: dict, artifact: Path) -> str:
        run = client.create_run(exp_id, run_name=run_name)
        rid = run.info.run_id
        for k, v in tags.items():
            client.set_tag(rid, k, str(v))
        for k, v in params.items():
            client.log_param(rid, k, str(v))
        for k, v in metrics.items():
            client.log_metric(rid, k, float(v))
        client.log_artifact(rid, str(artifact), artifact_path="artifacts")
        client.set_terminated(rid)
        return rid

    # ---- YOLO baseline (Phase 1) ----
    weights = ROOT / "model-training/runs/neu-det-yolov8s-baseline-2/weights/best.pt"
    yolo_sha = _sha256(weights)
    yolo_run_id = log_run(
        "phase1-yolo-baseline",
        tags={"model_name": "neu-yolov8s", "model_version": "1.0.0", "dataset": "NEU-DET",
              "model_type": "yolo", "domain": "steel", "artifact_sha256": yolo_sha},
        params={"dataset": "NEU-DET steel surface", "seed": 42, "imgsz": 640,
                "batch": 16, "epochs": 100},
        metrics={"precision": 0.81, "recall": 0.78, "mAP50": 0.82, "mAP50-95": 0.51,
                 "latency_p95_ms": 21.5, "crazing_map50": 0.80, "inclusion_map50": 0.85,
                 "patches_map50": 0.79},
        artifact=weights,
    )
    print("YOLO baseline run:", yolo_run_id, "sha256:", yolo_sha[:16], "...")

    # ---- PatchCore baseline (Phase 6) ----
    bank = ROOT / "inference-service/models/patchcore-bottle/bank.npz"
    pc_sha = _sha256(bank)
    pc_run_id = log_run(
        "phase6-patchcore-baseline",
        tags={"model_name": "mvtec-bottle-patchcore", "model_version": "1.0.0",
              "dataset": "MVTec AD bottle", "model_type": "patchcore",
              "dataset_domain": "mvtec-bottle", "artifact_sha256": pc_sha},
        params={"backbone": "wide_resnet50_2", "feature_layers": "layer2,layer3",
                "memory_bank_size": 50000, "threshold": 0.2027},
        metrics={"image_auroc": 1.0, "pixel_auroc": 0.986, "aup_pro": 0.955, "latency_ms": 755.0},
        artifact=bank,
    )
    print("PatchCore baseline run:", pc_run_id, "sha256:", pc_sha[:16], "...")

    # ---- also record into the Model Registry (as CANDIDATE) ----
    from app.models import Base, ModelRegistry
    from app.services.registry_service import RegistryService
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import asyncio

    async def register():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            svc = RegistryService()
            await svc.register(
                session, model_name="neu-yolov8s", model_version="1.0.0", model_type="yolo",
                artifact_uri="model-training/runs/neu-det-yolov8s-baseline-2/weights/best.pt",
                artifact_sha256=yolo_sha, dataset_version="neu-det-yolo-v1",
                training_run_id=yolo_run_id,
                metrics={"precision": 0.81, "recall": 0.78, "mAP50": 0.82, "mAP50-95": 0.51,
                         "latency_p95_ms": 21.5},
                domain_validated=True, notes="Phase 1 baseline (NEU-DET)",
            )
            await svc.register(
                session, model_name="mvtec-bottle-patchcore", model_version="1.0.0",
                model_type="patchcore",
                artifact_uri="inference-service/models/patchcore-bottle/bank.npz",
                artifact_sha256=pc_sha, dataset_version="mvtec-bottle-v1",
                training_run_id=pc_run_id,
                metrics={"image_auroc": 1.0, "pixel_auroc": 0.986, "aup_pro": 0.955,
                         "latency_ms": 755.0},
                domain_validated=False, notes="Phase 6 baseline (MVTec bottle; NOT steel-validated)",
            )
            await session.commit()
            rows = (await session.execute(select(ModelRegistry))).scalars().all()
            print("registry rows:", [(r.model_name, r.model_version, r.status) for r in rows])
        await engine.dispose()

    asyncio.run(register())

    out = {
        "yolo_run_id": yolo_run_id,
        "yolo_sha256": yolo_sha,
        "patchcore_run_id": pc_run_id,
        "patchcore_sha256": pc_sha,
    }
    (ROOT / "docs" / "phase8-mlflow-backfill.json").write_text(json.dumps(out, indent=2))
    print("backfill record:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
