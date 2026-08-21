"""D3 1.3 dual-branch candidate compatibility and isolation tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from inference_app.d3_dual_branch_predictor import D3DualBranchPredictor  # noqa: E402
from steel_patchcore.candidate_registry import (  # noqa: E402
    CandidateRegistryError,
    LoadedCandidate,
    canonical_sha256,
    sha256_file,
)
from steel_patchcore.dual_candidate_registry import (  # noqa: E402
    MODEL_VERSION,
    DualCandidateRegistry,
    LoadedDualCandidate,
    localization_bank_bundle_sha256,
    localization_feature_payload,
    validate_dual_manifest,
)
from steel_patchcore.d3_recovery_holdout import FROZEN_THRESHOLD  # noqa: E402

MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/1.3.0-candidate.1/manifest.json"
LEGACY_MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/manifest.json"
EVALUATION_REPORT = ROOT / "docs/dual-branch-evaluation-report.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _loaded() -> LoadedDualCandidate:
    legacy_manifest = {
        "model_name": "steel-patchcore-d3-candidate",
        "model_version": "1.2.0-candidate.1",
        "artifact_version": "d3-full-development-9b1ea19",
        "threshold": FROZEN_THRESHOLD,
    }
    image_bank = np.zeros((2, 768), dtype=np.float32)
    image_bank[:, :2] = np.eye(2, dtype=np.float32)
    image = LoadedCandidate(
        legacy_manifest,
        {},
        {},
        image_bank,
        np.zeros(768, dtype=np.float32),
        np.eye(768, dtype=np.float32),
    )
    banks = {}
    for name in ("R-L1", "R-L2"):
        bank = image_bank.copy()
        bank.flags.writeable = False
        banks[name] = bank
    return LoadedDualCandidate(_manifest(), image, banks, {}, {})


class _FakeDino(torch.nn.Module):
    embed_dim = 768
    patch_size = 14

    @staticmethod
    def _tokens(batch: int, patches: int, device: torch.device) -> torch.Tensor:
        tokens = torch.zeros((batch, patches, 768), dtype=torch.float32, device=device)
        alpha = torch.linspace(0.1, 1.0, patches, device=device)
        tokens[:, :, 0] = alpha
        tokens[:, :, 1] = torch.flip(alpha, dims=(0,))
        return tokens

    def forward_features(self, tensor):
        patches = 1024 if tensor.shape[-1] == 448 else 324
        return {"x_norm_patchtokens": self._tokens(tensor.shape[0], patches, tensor.device)}

    def get_intermediate_layers(self, tensor, n, norm):
        assert n == [7] and norm is True
        return [self._tokens(tensor.shape[0], 324, tensor.device)]


def test_dual_manifest_is_candidate_only_and_lineage_complete():
    manifest = _manifest()
    validate_dual_manifest(manifest)
    assert manifest["status"] == "CANDIDATE"
    assert manifest["model_version"] == MODEL_VERSION
    assert manifest["production_promotion"] is False
    assert manifest["image_branch"]["threshold"] == FROZEN_THRESHOLD
    assert manifest["image_branch"]["aggregation"] == "A0"
    assert set(manifest["hashes"]) == {
        "model_sha256", "feature_sha256", "image_bank_sha256", "localization_bank_sha256",
        "whitening_sha256", "protocol_sha256",
    }


def test_legacy_manifest_is_byte_bound_and_image_constants_are_unchanged():
    manifest = _manifest()
    legacy = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))
    assert sha256_file(LEGACY_MANIFEST) == manifest["image_branch"]["manifest_sha256"]
    assert legacy["model_version"] == "1.2.0-candidate.1"
    assert legacy["threshold"] == manifest["image_branch"]["threshold"] == FROZEN_THRESHOLD
    assert legacy["bank_sha256"] == manifest["hashes"]["image_bank_sha256"]
    assert legacy["whitening_sha256"] == manifest["hashes"]["whitening_sha256"]


def test_feature_and_localization_bank_bundle_hashes_are_canonical():
    manifest = _manifest()
    assert canonical_sha256(localization_feature_payload()) == manifest["hashes"]["feature_sha256"]
    banks = manifest["localization_branch"]["banks"]
    assert localization_bank_bundle_sha256(banks["R-L1"]["sha256"], banks["R-L2"]["sha256"]) == (
        manifest["hashes"]["localization_bank_sha256"]
    )


def test_dual_registry_verifies_all_actual_references():
    paths, hashes = DualCandidateRegistry(ROOT).verify_artifacts(_manifest())
    assert set(paths) == {
        "legacy_manifest", "weights", "whitening", "image_bank", "R-L1", "R-L2",
        "protocol", "investigation_results",
    }
    assert all(sha256_file(paths[name]) == digest for name, digest in hashes.items())


def test_missing_localization_artifact_fails_closed(tmp_path):
    manifest = _manifest()
    manifest["localization_branch"]["banks"]["R-L2"]["uri"] = "missing/r-l2-bank.npz"
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_sha256(payload)
    registry = DualCandidateRegistry(ROOT)
    with pytest.raises(CandidateRegistryError, match="DUAL_ARTIFACT_MISSING:R-L2"):
        registry.verify_artifacts(manifest)


@pytest.mark.artifact
def test_actual_dual_artifact_loading_is_verified_and_read_only():
    loaded = DualCandidateRegistry(ROOT).load_artifact(MANIFEST)
    assert loaded.threshold == FROZEN_THRESHOLD
    assert loaded.image_candidate.manifest["model_version"] == "1.2.0-candidate.1"
    for bank in loaded.localization_banks.values():
        assert bank.shape == (50000, 768)
        assert not bank.flags.writeable


def test_dual_inference_keeps_image_score_and_threshold_immutable(monkeypatch):
    predictor = D3DualBranchPredictor(_loaded(), device="cpu", model=_FakeDino())
    frozen = SimpleNamespace(anomaly_score=0.817907, threshold=FROZEN_THRESHOLD)
    monkeypatch.setattr(predictor.image_predictor, "infer", lambda image: frozen)
    localization = np.linspace(0.0, 1.0, 256 * 1600, dtype=np.float32).reshape(256, 1600)
    monkeypatch.setattr(predictor, "_localization_heatmap", lambda image: localization)
    output = predictor.infer(Image.fromarray(np.zeros((256, 1600, 3), dtype=np.uint8)))
    assert output.image_score == frozen.anomaly_score
    assert output.threshold == FROZEN_THRESHOLD
    assert output.anomaly_label == "NORMAL"
    assert set(output.summary()) == {
        "image_score", "anomaly_label", "heatmap", "confidence", "artifact_version", "localization_metadata"
    }
    assert output.confidence["calibrated_probability"] is False
    assert output.artifact_version == "d3-dual-rl3-0b148a6"
    assert output.localization_metadata["branch"] == "R-L3"


def test_r_l3_heatmap_is_reproducible_and_does_not_replace_image_score():
    predictor = D3DualBranchPredictor(_loaded(), device="cpu", model=_FakeDino())
    grids = [np.full((18, 18), 0.3 + index * 0.01, dtype=np.float32) for index in range(7)]
    predictor.image_predictor._raw_patch_grid = lambda model, tile: grids.pop(0)
    image = Image.fromarray(np.full((256, 1600, 3), 127, dtype=np.uint8))
    first = predictor.infer(image)
    grids.extend([np.full((18, 18), 0.3 + index * 0.01, dtype=np.float32) for index in range(7)])
    second = predictor.infer(image)
    assert first.image_score == second.image_score == pytest.approx(0.36)
    assert np.array_equal(first.heatmap, second.heatmap)
    assert not first.heatmap.flags.writeable


def test_api_routes_explicit_dual_manifest_without_production_promotion(monkeypatch):
    from inference_app import api

    fake = SimpleNamespace(device="cpu")
    captured = {}

    def load(manifest_path, *, project_root, device):
        captured.update(manifest_path=manifest_path, project_root=project_root, device=device)
        return fake

    monkeypatch.setattr(api, "_anomaly", None)
    monkeypatch.setattr(api, "D3_CANDIDATE_MANIFEST", str(MANIFEST))
    monkeypatch.setattr(api.D3DualBranchPredictor, "from_manifest", load)
    assert api.get_anomaly_predictor() is fake
    assert Path(captured["manifest_path"]) == MANIFEST
    assert Path(captured["project_root"]) == ROOT


def test_sealed_dual_branch_evaluation_report_passes_all_gates():
    report = json.loads(EVALUATION_REPORT.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    assert report["schema_version"] == "steel_patchcore_d3_dual_branch_evaluation_v1"
    assert report["model_version"] == MODEL_VERSION
    assert report["candidate_status"] == "CANDIDATE"
    assert report["production_promotion"] is False
    assert report["verdict"] == "PASS"
    assert report["artifact_unchanged"] is True
    assert report["image_score_immutable"] is True
    assert report["threshold_changed"] is False
    assert report["failures"] == []
    assert report["dataset"] == {"test_normal": 591, "recovery_holdout_anomaly": 3333, "total": 3924}
    assert metrics["image_auroc"] == 0.8179071714278028
    assert metrics["threshold"] == FROZEN_THRESHOLD
    assert metrics["image_score_mismatch_count"] == 0
    assert metrics["pixel_auroc"] >= 0.75
    assert metrics["aupro"] >= 0.50
