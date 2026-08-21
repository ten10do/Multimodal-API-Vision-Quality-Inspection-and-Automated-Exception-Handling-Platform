"""Container-only D3 release smoke inference and health endpoint."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATUS: dict = {"status": "starting"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_smoke() -> dict:
    root = Path(os.environ.get("D3_PROJECT_ROOT", "/workspace")).resolve()
    sys.path.insert(0, str(root / "model-training"))
    sys.path.insert(0, str(root / "inference-service"))

    import numpy as np
    import torch
    from PIL import Image

    from inference_app.d3_dual_branch_predictor import D3DualBranchPredictor
    from steel_patchcore.d3_release_package import ReleasePackageRegistry
    from steel_patchcore.dual_candidate_registry import DualCandidateRegistry

    release_manifest = root / "model-training/registry/steel-patchcore-d3-release/1.3.0/manifest.json"
    candidate_manifest = root / "model-training/registry/steel-patchcore-d3-candidate/1.3.0-candidate.1/manifest.json"
    image_path = root / "model-training/datasets/severstal-steel/raw/train_images/005da33cf.jpg"
    release = ReleasePackageRegistry(root).load(release_manifest)
    registry = DualCandidateRegistry(root)
    candidate = registry.load_manifest(candidate_manifest)
    before = registry.verify_artifacts(candidate)[1]
    predictor = D3DualBranchPredictor.from_manifest(candidate_manifest, project_root=root, device="cuda:0")
    with Image.open(image_path) as opened:
        output = predictor.infer(opened.convert("RGB"))
    after = registry.verify_artifacts(candidate)[1]
    if before != after:
        raise RuntimeError("artifact hashes changed during container inference")
    if output.threshold != release.manifest["threshold"] or output.heatmap.shape != (256, 1600):
        raise RuntimeError("container inference contract mismatch")
    return {
        "status": "ready",
        "environment_fingerprint": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "architecture": platform.machine(),
        },
        "runtime_fingerprint": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0),
            "numpy": np.__version__,
        },
        "release_manifest_sha256": _sha256(release_manifest),
        "release_manifest_payload_sha256": release.manifest["manifest_payload_sha256"],
        "artifact_hashes": before,
        "inference": {
            "image_id": image_path.stem,
            "image_score": output.image_score,
            "threshold": output.threshold,
            "anomaly_label": output.anomaly_label,
            "heatmap_shape": list(output.heatmap.shape),
            "model_version": output.model_version,
            "artifact_version": output.artifact_version,
            "latency_ms": output.latency_ms,
        },
        "production_promotion": False,
    }


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps(STATUS, separators=(",", ":")).encode()
        code = 200 if self.path == "/health" and STATUS.get("status") == "ready" else 503
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def serve() -> int:
    global STATUS
    try:
        STATUS = run_smoke()
    except Exception as exc:  # fail closed while retaining a health endpoint
        STATUS = {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
    port = int(os.environ.get("D3_REVIEW_PORT", "8080"))
    ThreadingHTTPServer(("0.0.0.0", port), _Handler).serve_forever()
    return 0


def healthcheck() -> int:
    port = int(os.environ.get("D3_REVIEW_PORT", "8080"))
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            payload = json.load(response)
        return 0 if response.status == 200 and payload.get("status") == "ready" else 1
    except Exception:
        return 1


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if command == "serve":
        return serve()
    if command == "smoke":
        print(json.dumps(run_smoke(), indent=2))
        return 0
    if command == "healthcheck":
        return healthcheck()
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
