"""Optimization 2: D3 candidate artifact, registry, inference, heatmap, and evaluation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from inference_app.d3_candidate_predictor import D3CandidatePredictor  # noqa: E402
from steel_patchcore.candidate_registry import (  # noqa: E402
    ALLOWED_STATUS,
    HASH_FIELDS,
    MODEL_NAME,
    ArtifactVerification,
    CandidateRegistry,
    CandidateRegistryError,
    LoadedCandidate,
    canonical_sha256,
    evaluate_candidate_gate,
    sha256_file,
    validate_manifest,
)
from steel_patchcore.d3_candidate_evaluation import (  # noqa: E402
    CandidateEvaluationError,
    generate_evaluation_report,
)

MANIFEST_PATH = ROOT / "model-training/registry/steel-patchcore-d3-candidate/manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _rehash_manifest(manifest: dict) -> dict:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = canonical_sha256(payload)
    return manifest


def _temp_manifest(tmp_path: Path) -> dict:
    manifest = _manifest()
    for index, hash_field in enumerate(HASH_FIELDS):
        uri_field = hash_field.replace("_sha256", "_uri")
        artifact = tmp_path / f"artifact-{index}.bin"
        artifact.write_bytes(f"artifact-{index}".encode())
        manifest[uri_field] = artifact.name
        manifest[hash_field] = sha256_file(artifact)
    return _rehash_manifest(manifest)


def _loaded_candidate() -> LoadedCandidate:
    manifest = {
        "model_name": MODEL_NAME,
        "model_version": "test-candidate",
        "artifact_version": "test-artifact",
        "threshold": 0.5,
    }
    bank = np.zeros((2, 768), dtype=np.float32)
    bank[0, 0] = 1.0
    bank[1, 1] = 1.0
    mean = np.zeros(768, dtype=np.float32)
    whitening = np.eye(768, dtype=np.float32)
    return LoadedCandidate(manifest, {}, {}, bank, mean, whitening)


class _FakeDino(torch.nn.Module):
    embed_dim = 768
    patch_size = 14

    def forward_features(self, tensor):
        tokens = torch.zeros((1, 324, 768), dtype=torch.float32, device=tensor.device)
        alpha = torch.linspace(0.1, 1.0, 324, device=tensor.device)
        tokens[0, :, 0] = alpha
        tokens[0, :, 1] = torch.flip(alpha, dims=(0,))
        return {"x_norm_patchtokens": tokens}


def test_candidate_manifest_schema_and_identity():
    manifest = _manifest()
    validate_manifest(manifest)
    assert manifest["status"] == ALLOWED_STATUS == "CANDIDATE"
    assert manifest["model_name"] == MODEL_NAME
    assert manifest["backbone"] == "dinov2_vitb14"
    assert manifest["aggregation"] == "A0"
    assert manifest["distance"] == "cosine-1NN"
    assert manifest["embedding_dim"] == 768
    assert manifest["production_promotion"] is False


def test_manifest_rejects_production_and_schema_drift():
    manifest = _manifest()
    manifest["status"] = "PRODUCTION"
    _rehash_manifest(manifest)
    with pytest.raises(CandidateRegistryError, match="CANDIDATE_ONLY"):
        validate_manifest(manifest)
    manifest = _manifest()
    manifest["unexpected"] = True
    with pytest.raises(CandidateRegistryError, match="FIELD_SET"):
        validate_manifest(manifest)


def test_registry_register_verify_and_load_manifest(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    manifest = _temp_manifest(project)
    registry = CandidateRegistry(tmp_path / "registry", project)
    monkeypatch.setattr(registry, "_lineage_errors", lambda candidate: [])
    verification = registry.verify_artifact(manifest)
    assert verification.passed and verification.errors == ()
    path = registry.register(manifest, tests_passed=True)
    assert path == tmp_path / "registry" / MODEL_NAME / "manifest.json"
    assert registry.load_manifest() == manifest


def test_registry_fails_closed_on_hash_mismatch(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    manifest = _temp_manifest(project)
    Path(project / manifest["bank_uri"]).write_bytes(b"mutated")
    registry = CandidateRegistry(tmp_path / "registry", project)
    verification = registry.verify_artifact(manifest)
    assert not verification.passed
    assert any("ARTIFACT_SHA_MISMATCH:bank_sha256" in error for error in verification.errors)
    with pytest.raises(CandidateRegistryError, match="CANDIDATE_GATE_BLOCKED"):
        registry.register(manifest, tests_passed=True)


def test_candidate_gate_requires_all_five_conditions():
    manifest = _manifest()
    verified = ArtifactVerification(True, {field: manifest[field] for field in HASH_FIELDS}, ())
    gate = evaluate_candidate_gate(manifest, verified, tests_passed=True)
    assert gate.passed and gate.blocked == ()
    assert {row["check"] for row in gate.checks} >= {
        "artifact_integrity", "full_development_auroc", "recovery_holdout", "lineage", "tests"
    }
    blocked = evaluate_candidate_gate(manifest, verified, tests_passed=False)
    assert not blocked.passed and "tests" in blocked.blocked


def test_registry_lineage_check_rejects_evidence_drift():
    manifest = _manifest()
    manifest["evaluation_evidence"]["full_development"]["image_auroc"] = 0.99
    registry = CandidateRegistry(ROOT / "model-training/registry", ROOT)
    assert "LINEAGE_MISMATCH:full_auroc" in registry._lineage_errors(manifest)


def test_threshold_and_bank_semantics_are_immutable_in_manifest():
    manifest = _manifest()
    assert manifest["threshold"] == 0.8471092581748962
    assert manifest["bank_sha256"] == "40fe43331885422c8a32364a48fc403b766f807f69faafee775a2eb2403cbbda"
    changed = dict(manifest)
    changed["threshold"] += 1e-12
    _rehash_manifest(changed)
    with pytest.raises(CandidateRegistryError, match="THRESHOLD_MISMATCH"):
        validate_manifest(changed)


@pytest.mark.artifact
def test_actual_artifact_loading_is_verified_and_read_only():
    registry = CandidateRegistry(ROOT / "model-training/registry", ROOT)
    manifest = registry.load_manifest()
    before = registry.verify_artifact(manifest)
    assert before.passed, before.errors
    loaded = registry.load_artifact()
    assert loaded.bank.shape == (50000, 768)
    assert loaded.whitening_mean.shape == (768,)
    assert loaded.whitening_matrix.shape == (768, 768)
    assert not loaded.bank.flags.writeable
    assert not loaded.whitening_mean.flags.writeable
    assert not loaded.whitening_matrix.flags.writeable
    with pytest.raises(ValueError):
        loaded.bank[0, 0] = 0.0
    assert registry.verify_artifact(manifest).hashes == before.hashes


def test_d3_inference_is_deterministic_and_emits_required_payload():
    predictor = D3CandidatePredictor(_loaded_candidate(), device="cpu", model=_FakeDino())
    image = Image.fromarray(np.full((256, 1600, 3), 127, dtype=np.uint8))
    first = predictor.infer(image)
    second = predictor.infer(image)
    assert first.summary() == second.summary()
    assert set(first.summary()) == {"anomaly_score", "threshold", "is_anomaly", "model_version", "artifact_version"}
    assert first.is_anomaly == (first.anomaly_score >= first.threshold)
    assert first.model_version == "test-candidate"
    assert first.artifact_version == "test-artifact"


def test_heatmap_generation_preserves_a0_image_score(monkeypatch):
    predictor = D3CandidatePredictor(_loaded_candidate(), device="cpu", model=_FakeDino())
    grids = []
    for index in range(7):
        grid = np.linspace(0.1, 0.2 + index * 0.1, 324, dtype=np.float32).reshape(18, 18)
        grids.append(grid)
    generated = iter(grids)
    monkeypatch.setattr(predictor, "_raw_patch_grid", lambda model, tile: next(generated))
    image = Image.fromarray(np.zeros((256, 1600, 3), dtype=np.uint8))
    output = predictor.infer(image)
    assert output.anomaly_score == pytest.approx(float(np.stack(grids).max()))
    assert output.raw_anomaly_map.shape == (256, 1600)
    assert output.normalized_heatmap.shape == (256, 1600)
    assert 0.0 <= float(output.normalized_heatmap.min()) <= float(output.normalized_heatmap.max()) <= 1.0
    assert not output.raw_anomaly_map.flags.writeable
    assert not output.normalized_heatmap.flags.writeable


def test_shared_contract_carries_artifact_version():
    predictor = D3CandidatePredictor(_loaded_candidate(), device="cpu", model=_FakeDino())
    grids = iter([np.full((18, 18), 0.6, dtype=np.float32) for _ in range(7)])
    predictor._raw_patch_grid = lambda model, tile: next(grids)
    result = predictor.predict(Image.fromarray(np.zeros((256, 1600, 3), dtype=np.uint8)), include_map_png=False)
    assert result.artifact_version == "test-artifact"
    assert result.anomaly_score == pytest.approx(0.6)
    assert result.is_anomalous is True


def test_evaluation_pipeline_is_reproducible_and_lineage_bound():
    manifest = _manifest()
    dataset = {"test_normal": ["n1", "n2"], "recovery_holdout_anomaly": ["a1", "a2", "a3", "a4"]}
    records = [
        {"image_id": "n1", "split_role": "test_normal", "score": 0.1},
        {"image_id": "n2", "split_role": "test_normal", "score": 0.2},
        {"image_id": "a1", "split_role": "recovery_holdout_anomaly", "score": 0.6, "quartile": 1},
        {"image_id": "a2", "split_role": "recovery_holdout_anomaly", "score": 0.7, "quartile": 2},
        {"image_id": "a3", "split_role": "recovery_holdout_anomaly", "score": 0.8, "quartile": 3},
        {"image_id": "a4", "split_role": "recovery_holdout_anomaly", "score": 0.9, "quartile": 4},
    ]
    hashes = {field: manifest[field] for field in HASH_FIELDS}
    kwargs = dict(
        dataset_manifest=dataset,
        score_records=records,
        candidate_manifest=manifest,
        artifact_hashes=hashes,
        timestamp="2026-08-21T00:00:00Z",
        bootstrap_iterations=50,
    )
    first = generate_evaluation_report(**kwargs)
    second = generate_evaluation_report(**kwargs)
    assert first == second
    assert first["metrics"]["image_auroc"] == 1.0
    assert first["verdict"] == "RECOVERY_HOLDOUT_PASS"
    assert first["lineage"]["source_split_sha256"] == manifest["source_split_sha256"]
    assert first["artifact_hashes"] == hashes
    assert first["timestamp"] == "2026-08-21T00:00:00Z"


def test_evaluation_pipeline_rejects_incomplete_dataset():
    manifest = _manifest()
    with pytest.raises(CandidateEvaluationError, match="INCOMPLETE"):
        generate_evaluation_report(
            dataset_manifest={"test_normal": ["n1"], "recovery_holdout_anomaly": ["a1"]},
            score_records=[{"image_id": "n1", "split_role": "test_normal", "score": 0.1}],
            candidate_manifest=manifest,
            artifact_hashes={field: manifest[field] for field in HASH_FIELDS},
            timestamp="2026-08-21T00:00:00Z",
            bootstrap_iterations=10,
        )


def test_inference_service_candidate_selection_is_explicit():
    source = (ROOT / "inference-service/inference_app/api.py").read_text(encoding="utf-8")
    assert 'os.environ.get("IVQC_D3_CANDIDATE_MANIFEST")' in source
    assert "if D3_CANDIDATE_MANIFEST:" in source
    assert "D3CandidatePredictor.from_registry" in source
    assert "PATCHCORE_BANK_DEFAULT" in source


def test_inference_service_loads_candidate_only_when_opted_in(monkeypatch):
    from types import SimpleNamespace
    from inference_app import api

    fake = SimpleNamespace(device="cpu")
    captured = {}

    def load(registry_root, *, project_root, device):
        captured.update(registry_root=registry_root, project_root=project_root, device=device)
        return fake

    monkeypatch.setattr(api, "_anomaly", None)
    monkeypatch.setattr(api, "D3_CANDIDATE_MANIFEST", str(MANIFEST_PATH))
    monkeypatch.setattr(api.D3CandidatePredictor, "from_registry", load)
    assert api.get_anomaly_predictor() is fake
    assert Path(captured["registry_root"]) == ROOT / "model-training/registry"
    assert Path(captured["project_root"]) == ROOT
