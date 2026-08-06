"""Phase 8 fault injection (8N): invalid models must never silently become
production; readiness must fail on bad artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mlops.manifest import validate_artifacts


def test_wrong_sha256_detected(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"x" * 1024)
    m = {
        "vision_stack_version": "v",
        "yolo": {"model": "m", "version": "1", "artifact_uri": str(f), "sha256": "00000000"},
        "patchcore": {"model": "p", "version": "1", "artifact_uri": str(f), "sha256": "00000000"},
    }
    problems = validate_artifacts(m, tmp_path)
    assert len(problems) == 2
    assert all("sha256 mismatch" in p for p in problems)


def test_missing_artifact_detected(tmp_path):
    m = {
        "vision_stack_version": "v",
        "yolo": {"model": "m", "version": "1", "artifact_uri": "missing.pt", "sha256": None},
        "patchcore": {"model": "p", "version": "1", "artifact_uri": "missing.npz", "sha256": None},
    }
    problems = validate_artifacts(m, tmp_path)
    assert len(problems) == 2
    assert all("missing" in p for p in problems)


def test_manifest_missing_section_raises(tmp_path):
    import pytest

    from app.mlops.manifest import load_manifest

    bad = tmp_path / "manifest.yaml"
    bad.write_text("vision_stack_version: x\nyolo: {model: m, version: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(bad)
