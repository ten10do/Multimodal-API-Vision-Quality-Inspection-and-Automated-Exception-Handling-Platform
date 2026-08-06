"""Deployment manifest (8D / 8E).

A fixed manifest pins the whole AI stack that judges every inspection:

    vision_stack_version: "2026.08.1"
    yolo:        {model, version, sha256}
    patchcore:   {model, version, sha256}
    fusion:      {version}
    quality_rules: {version}

The backend stamps inspections.deployment_version from it, and the
inference service resolves + SHA256-validates artifacts against it before
declaring READY. No version-less `best.pt` startup (8C boundary).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "deployment_manifest.yaml"

_MANIFEST: dict[str, Any] | None = None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the deployment manifest YAML."""
    import yaml

    p = path or DEFAULT_MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(f"deployment manifest not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "vision_stack_version" not in data:
        raise ValueError("manifest must contain vision_stack_version")
    for section in ("yolo", "patchcore"):
        if section not in data:
            raise ValueError(f"manifest missing section {section}")
        for key in ("model", "version"):
            if key not in data[section]:
                raise ValueError(f"manifest {section}.{key} missing")
    return data


def get_manifest(path: Path | None = None) -> dict[str, Any]:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = load_manifest(path)
    return _MANIFEST


def reset_manifest_cache() -> None:
    global _MANIFEST
    _MANIFEST = None


def current_deployment_version(path: Path | None = None) -> str:
    return str(get_manifest(path)["vision_stack_version"])


def validate_artifacts(manifest: dict[str, Any], root: Path) -> list[str]:
    """Return a list of problems. Empty list means all artifacts pass.

    artifact_uri is interpreted relative to the project root (`root`)."""
    problems: list[str] = []
    for section in ("yolo", "patchcore"):
        m = manifest[section]
        uri = m.get("artifact_uri") or m.get("path") or m.get("model")
        if not uri:
            problems.append(f"{section}: no artifact_uri")
            continue
        p = Path(uri)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            problems.append(f"{section}: artifact missing {p}")
            continue
        expected = m.get("sha256")
        if expected:
            actual = sha256_of(p)
            if actual != expected:
                problems.append(f"{section}: sha256 mismatch for {p} (expected {expected}, got {actual[:16]}...)")
    return problems
