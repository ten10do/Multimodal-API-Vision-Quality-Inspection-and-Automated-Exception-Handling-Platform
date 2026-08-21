"""Dual-branch D3 inference: frozen D3 A0 score plus R-L3 heatmap."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vision_contract import AnomalyResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "model-training"))

from steel_patchcore.d3_localization_representation import (  # noqa: E402
    HIGH_RESOLUTION_SIDE,
    INTERMEDIATE_BLOCK_INDEX,
    fuse_dense_maps,
)
from steel_patchcore.d3_production_readiness import threshold_margin_confidence  # noqa: E402
from steel_patchcore.dual_candidate_registry import (  # noqa: E402
    DualCandidateRegistry,
    LoadedDualCandidate,
)
from steel_patchcore.tile import TILE_X0, stitch_scores  # noqa: E402

from .d3_candidate_predictor import (  # noqa: E402
    EXPECTED_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    D3CandidatePredictor,
    _normalize_heatmap,
    _resize_raw_patch_grid,
)
from .patchcore_predictor import PatchCoreError, _map_png  # noqa: E402


@dataclass(frozen=True)
class D3DualInferenceOutput:
    image_score: float
    anomaly_label: str
    heatmap: np.ndarray
    confidence: dict
    localization_metadata: dict
    threshold: float
    model_version: str
    artifact_version: str
    latency_ms: float

    def summary(self) -> dict:
        return {
            "image_score": self.image_score,
            "anomaly_label": self.anomaly_label,
            "heatmap": self.heatmap,
            "confidence": self.confidence,
            "artifact_version": self.artifact_version,
            "localization_metadata": self.localization_metadata,
        }


class D3DualBranchPredictor:
    """Keeps image inference inside the unchanged D3 predictor and isolates R-L3."""

    def __init__(
        self,
        candidate: LoadedDualCandidate,
        *,
        device: str | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.candidate = candidate
        self.image_predictor = D3CandidatePredictor(candidate.image_candidate, device=device, model=model)
        self.device = self.image_predictor.device
        self.model_name = str(candidate.manifest["model_name"])
        self.model_version = str(candidate.manifest["model_version"])
        self.artifact_version = str(candidate.manifest["artifact_version"])
        self.threshold = float(candidate.threshold)
        self._localization_bank_tensors: dict[str, torch.Tensor] = {}

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        project_root: str | Path = PROJECT_ROOT,
        device: str | None = None,
    ) -> "D3DualBranchPredictor":
        loaded = DualCandidateRegistry(Path(project_root)).load_artifact(manifest_path)
        return cls(loaded, device=device)

    def _bank_tensor(self, name: str) -> torch.Tensor:
        if name not in self._localization_bank_tensors:
            self._localization_bank_tensors[name] = torch.tensor(
                self.candidate.localization_banks[name], device=self.device
            )
        return self._localization_bank_tensors[name]

    def _tile_batch(self, image: Image.Image, side: int) -> torch.Tensor:
        tensors = []
        for x0 in TILE_X0:
            tile = image.crop((x0, 0, x0 + 256, 256)).convert("RGB")
            array = np.asarray(tile, dtype=np.float32) / 255.0
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        batch = torch.stack(tensors).to(self.device)
        if side != 256:
            batch = F.interpolate(batch, size=(side, side), mode="bilinear", align_corners=False)
        return batch

    @staticmethod
    def _distance_grids(tokens: torch.Tensor, bank: torch.Tensor, grid_side: int) -> np.ndarray:
        grids = []
        for tile_tokens in tokens:
            distance = 1.0 - (tile_tokens @ bank.T).max(dim=1).values
            grids.append(distance.reshape(grid_side, grid_side).detach().cpu().numpy().astype(np.float32))
        output = np.stack(grids)
        if not np.isfinite(output).all():
            raise PatchCoreError("R-L3 localization map contains non-finite values")
        return output

    def _localization_heatmap(self, image: Image.Image) -> np.ndarray:
        model = self.image_predictor._ensure_model()
        with torch.no_grad():
            l1 = model.get_intermediate_layers(
                self._tile_batch(image, 252), n=[INTERMEDIATE_BLOCK_INDEX], norm=True
            )[0]
            l2 = model.forward_features(self._tile_batch(image, HIGH_RESOLUTION_SIDE))["x_norm_patchtokens"]
        if tuple(l1.shape) != (7, 324, 768) or tuple(l2.shape) != (7, 1024, 768):
            raise PatchCoreError(f"R-L3 feature shape mismatch: {tuple(l1.shape)}:{tuple(l2.shape)}")
        l1_grids = self._distance_grids(F.normalize(l1, p=2, dim=2), self._bank_tensor("R-L1"), 18)
        l2_grids = self._distance_grids(F.normalize(l2, p=2, dim=2), self._bank_tensor("R-L2"), 32)
        tile_maps = [
            fuse_dense_maps(_resize_raw_patch_grid(first), _resize_raw_patch_grid(second))
            for first, second in zip(l1_grids, l2_grids)
        ]
        heatmap = stitch_scores(tile_maps)
        heatmap.flags.writeable = False
        return heatmap

    def _metadata(self) -> dict:
        manifest = self.candidate.manifest
        localization = manifest["localization_branch"]
        return {
            "branch": "R-L3",
            "model_version": self.model_version,
            "artifact_version": self.artifact_version,
            "feature_sha256": manifest["hashes"]["feature_sha256"],
            "bank_sha256": manifest["hashes"]["localization_bank_sha256"],
            "protocol_sha256": manifest["hashes"]["protocol_sha256"],
            "component_bank_sha256": {
                name: localization["banks"][name]["sha256"] for name in ("R-L1", "R-L2")
            },
        }

    def infer(self, image: Image.Image) -> D3DualInferenceOutput:
        if image.size != EXPECTED_SIZE:
            raise PatchCoreError(f"D3 candidate expects 1600x256 steel originals, got {image.size[0]}x{image.size[1]}")
        started = time.perf_counter()
        # This call is the complete, unchanged 1.2 D3 A0 image branch.
        image_output = self.image_predictor.infer(image)
        frozen_score = image_output.anomaly_score
        if image_output.threshold != self.threshold:
            raise PatchCoreError("D3 dual candidate threshold lineage mismatch")
        raw_heatmap = self._localization_heatmap(image)
        normalized = _normalize_heatmap(raw_heatmap)
        normalized.flags.writeable = False
        if image_output.anomaly_score != frozen_score:
            raise PatchCoreError("D3 image score changed during localization")
        return D3DualInferenceOutput(
            image_score=frozen_score,
            anomaly_label="ANOMALY" if frozen_score >= self.threshold else "NORMAL",
            heatmap=normalized,
            confidence=threshold_margin_confidence(frozen_score, self.threshold),
            localization_metadata=self._metadata(),
            threshold=self.threshold,
            model_version=self.model_version,
            artifact_version=self.artifact_version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def predict(
        self,
        image: Image.Image,
        *,
        image_w: int | None = None,
        image_h: int | None = None,
        include_map_png: bool = True,
    ) -> AnomalyResult:
        output = self.infer(image)
        return AnomalyResult(
            model_name=self.model_name,
            model_version=output.model_version,
            artifact_version=output.artifact_version,
            anomaly_score=output.image_score,
            threshold=output.threshold,
            is_anomalous=output.anomaly_label == "ANOMALY",
            regions=[],
            latency_ms=output.latency_ms,
            anomaly_map_png=_map_png(output.heatmap) if include_map_png else None,
        )


__all__ = ["D3DualBranchPredictor", "D3DualInferenceOutput"]
