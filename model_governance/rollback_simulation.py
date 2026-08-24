"""Runnable lifecycle rollback drill using disposable simulated artifacts."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .model_lifecycle import ModelLifecycleManager, sha256_file


METRICS = {"image_auroc": 0.82, "pixel_auroc": 0.92, "aupro": 0.80}


def run_simulation(workdir: str | Path) -> dict:
    root = Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = root / "model-1.2.0.simulated"
    candidate = root / "model-1.3.0.simulated"
    previous.write_bytes(b"simulated-previous-production-artifact")
    candidate.write_bytes(b"simulated-failing-candidate-artifact")

    manager = ModelLifecycleManager(root / "model_history.json", project_root=root)
    previous_hash = sha256_file(previous)
    candidate_hash = sha256_file(candidate)
    manager.register("1.2.0", previous, previous_hash, operator="simulation-validator")
    manager.validate("1.2.0", metrics=METRICS, operator="simulation-validator")
    manager.promote("1.2.0", operator="simulation-approver")
    manager.promote("1.2.0", operator="simulation-approver")
    manager.register("1.3.0", candidate, candidate_hash, operator="simulation-validator")
    manager.validate("1.3.0", metrics=METRICS, operator="simulation-validator")
    manager.promote("1.3.0", operator="simulation-approver")

    rollback = manager.rollback(
        "1.3.0",
        "1.2.0",
        operator="simulation-operator",
        reason="simulated inference failure",
    )
    return {
        "failed_version": "1.3.0",
        "inference_outcome": "FAILURE",
        "restored_version": rollback["restored_version"],
        "rollback_status": rollback["status"],
        "expected_artifact_hash": previous_hash,
        "restored_artifact_hash": sha256_file(previous),
        "artifact_hash_consistent": rollback["artifact_hash"] == previous_hash == sha256_file(previous),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="industrial-model-rollback-") as temp_dir:
        print(json.dumps(run_simulation(temp_dir), indent=2))
