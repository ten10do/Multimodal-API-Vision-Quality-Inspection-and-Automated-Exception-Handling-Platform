"""Inference adapter for the frozen, candidate-only steel D3 artifact."""
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

from steel_patchcore.candidate_registry import (  # noqa: E402
    MODEL_NAME,
    CandidateRegistry,
    CandidateRegistryError,
    LoadedCandidate,
)
from steel_patchcore.domain_representation import adapted_input_side  # noqa: E402
from steel_patchcore.tile import TILE_X0, stitch_scores  # noqa: E402

from .patchcore_predictor import PatchCoreError, _map_png  # noqa: E402

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
EXPECTED_SIZE = (1600, 256)


@dataclass(frozen=True)
class D3InferenceOutput:
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    model_version: str
    artifact_version: str
    raw_anomaly_map: np.ndarray
    normalized_heatmap: np.ndarray
    latency_ms: float

    def summary(self) -> dict:
        """Stable service payload; heatmaps remain separately available arrays."""
        return {
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "is_anomaly": self.is_anomaly,
            "model_version": self.model_version,
            "artifact_version": self.artifact_version,
        }


def _tile_tensor(tile: Image.Image, device: torch.device) -> torch.Tensor:
    rgb = tile.convert("RGB") if tile.mode != "RGB" else tile
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _resize_raw_patch_grid(grid: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(grid, dtype=np.float32), mode="F")
    return np.asarray(image.resize((256, 256), Image.BILINEAR), dtype=np.float32)


def _normalize_heatmap(raw_map: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_map, dtype=np.float32)
    minimum, maximum = float(raw.min()), float(raw.max())
    if maximum <= minimum:
        return np.zeros_like(raw, dtype=np.float32)
    return ((raw - minimum) / (maximum - minimum)).astype(np.float32, copy=False)


class D3CandidatePredictor:
    """Read-only DINOv2-B + ZCA + cosine-1NN candidate inference."""

    def __init__(
        self,
        candidate: LoadedCandidate,
        *,
        device: str | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.candidate = candidate
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._model = model
        self._bank_tensor: torch.Tensor | None = None
        self._mean_tensor: torch.Tensor | None = None
        self._whitening_tensor: torch.Tensor | None = None
        self.model_name = str(candidate.manifest["model_name"])
        self.model_version = str(candidate.manifest["model_version"])
        self.artifact_version = str(candidate.manifest["artifact_version"])
        self.threshold = float(candidate.manifest["threshold"])

    @classmethod
    def from_registry(
        cls,
        registry_root: str | Path,
        *,
        project_root: str | Path = PROJECT_ROOT,
        device: str | None = None,
    ) -> "D3CandidatePredictor":
        registry = CandidateRegistry(Path(registry_root), Path(project_root))
        return cls(registry.load_artifact(MODEL_NAME), device=device)

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)
            state = torch.load(self.candidate.paths["weights_uri"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            self._model = model
        model = self._model.eval().to(self.device)
        if int(getattr(model, "embed_dim", -1)) != 768 or int(getattr(model, "patch_size", -1)) != 14:
            raise PatchCoreError("D3 candidate backbone identity mismatch")
        return model

    def _ensure_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._bank_tensor is None:
            self._bank_tensor = torch.tensor(self.candidate.bank, device=self.device)
            self._mean_tensor = torch.tensor(self.candidate.whitening_mean, device=self.device)
            self._whitening_tensor = torch.tensor(self.candidate.whitening_matrix, device=self.device)
        assert self._mean_tensor is not None and self._whitening_tensor is not None
        return self._bank_tensor, self._mean_tensor, self._whitening_tensor

    def _raw_patch_grid(self, model: torch.nn.Module, tile: Image.Image) -> np.ndarray:
        tensor = _tile_tensor(tile, self.device)
        side = adapted_input_side(tensor.shape[-1], 14)
        if tensor.shape[-1] != side:
            tensor = F.interpolate(tensor, size=(side, side), mode="bilinear", align_corners=False)
        with torch.no_grad():
            tokens = model.forward_features(tensor)["x_norm_patchtokens"][0]
        if tuple(tokens.shape) != (324, 768):
            raise PatchCoreError(f"D3 patch token shape mismatch: {tuple(tokens.shape)}")
        bank, mean, whitening = self._ensure_tensors()
        embedding = F.normalize((tokens - mean) @ whitening, p=2, dim=1)
        distance = 1.0 - (embedding @ bank.T).max(dim=1).values
        grid = distance.reshape(18, 18).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(grid).all():
            raise PatchCoreError("D3 raw anomaly map contains non-finite values")
        return grid

    def infer(self, image: Image.Image) -> D3InferenceOutput:
        if image.size != EXPECTED_SIZE:
            raise PatchCoreError(f"D3 candidate expects 1600x256 steel originals, got {image.size[0]}x{image.size[1]}")
        started = time.perf_counter()
        model = self._ensure_model()
        grids = []
        for x0 in TILE_X0:
            tile = image.crop((x0, 0, x0 + 256, 256))
            grids.append(self._raw_patch_grid(model, tile))
        raw_tile_grids = np.stack(grids).astype(np.float32, copy=False)
        # Frozen image score is calculated before and independently of heatmap processing.
        image_score = float(raw_tile_grids.max())
        raw_map = stitch_scores([_resize_raw_patch_grid(grid) for grid in raw_tile_grids])
        normalized = _normalize_heatmap(raw_map)
        raw_map.flags.writeable = False
        normalized.flags.writeable = False
        return D3InferenceOutput(
            anomaly_score=image_score,
            threshold=self.threshold,
            is_anomaly=image_score >= self.threshold,
            model_version=self.model_version,
            artifact_version=self.artifact_version,
            raw_anomaly_map=raw_map,
            normalized_heatmap=normalized,
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
        """Adapt the D3 payload to the existing shared inference contract."""
        output = self.infer(image)
        return AnomalyResult(
            model_name=self.model_name,
            model_version=output.model_version,
            artifact_version=output.artifact_version,
            anomaly_score=output.anomaly_score,
            threshold=output.threshold,
            is_anomalous=output.is_anomaly,
            regions=[],
            latency_ms=output.latency_ms,
            anomaly_map_png=_map_png(output.normalized_heatmap) if include_map_png else None,
        )


__all__ = [
    "CandidateRegistryError",
    "D3CandidatePredictor",
    "D3InferenceOutput",
    "_normalize_heatmap",
    "_resize_raw_patch_grid",
]
