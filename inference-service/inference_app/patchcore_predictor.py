"""PatchCore-style anomaly detection (Phase 6).

Independent anomaly pipeline: a pretrained WideResNet-50-2 extracts multi-scale
patch features; a memory bank of normal patch features (built in a single pass
over normal images, no backprop) is scored by nearest-neighbour distance.

Outputs are objective (score, map, regions, threshold) and deliberately free
of any quality judgement (6A). The quality judgement is made downstream by
the Quality Rule Engine.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from vision_contract import AnomalyRegion, AnomalyResult, utc_now_iso

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LAYERS = ("layer2", "layer3")


class PatchCoreError(Exception):
    """Raised when the anomaly pipeline cannot run (model/bank unavailable)."""


def _to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


class PatchCorePredictor:
    """Nearest-neighbour anomaly detector over a frozen normal feature bank."""

    def __init__(
        self,
        bank_path: str | Path | None = None,
        device: str | None = None,
        image_size: int = 224,
        num_neighbors: int = 1,
    ) -> None:
        self.image_size = image_size
        self.num_neighbors = num_neighbors
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._bank: np.ndarray | None = None
        self._threshold: float | None = None
        self.model_name = "patchcore-wrn50-2"
        self.model_version = "mvtec-bottle-baseline"
        self._model: torch.nn.Module | None = None
        if bank_path is not None:
            self.load_bank(bank_path)

    # ---- model + bank ----
    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            model = wide_resnet50_2(weights=weights)
            model.eval()
            model.to(self.device)
            self._model = model
        return self._model

    def load_bank(self, bank_path: str | Path) -> None:
        data = np.load(bank_path)
        self._bank = data["features"].astype(np.float32)
        self._threshold = float(data["threshold"])
        self.model_name = str(data.get("model_name", self.model_name))
        self.model_version = str(data.get("model_version", self.model_version))

    def _embed(self, image: Image.Image) -> np.ndarray:
        model = self._ensure_model()
        x = _to_tensor(image, self.image_size).to(self.device)
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)
            h = model.layer1(h)
            h2 = model.layer2(h)  # [B, 512, H/8, W/8]
            h3 = model.layer3(h2)  # [B, 1024, H/16, W/16]
            # upsample layer3 to the layer2 grid, then concat channels
            h3_up = torch.nn.functional.interpolate(h3, size=h2.shape[-2:], mode="bilinear", align_corners=False)
            feat_map = torch.cat([h2, h3_up], dim=1)  # [B, 1536, H/8, W/8]
            b, c, hh, ww = feat_map.shape
            flat = feat_map.permute(0, 2, 3, 1).reshape(b, hh * ww, c)
            feat = flat[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat, axis=1, keepdims=True)
        norm[norm == 0] = 1e-8
        return feat / norm

    def score(self, image: Image.Image) -> tuple[np.ndarray, float]:
        """Return (anomaly_map [H,W] in image_size, image-level score)."""
        if self._bank is None:
            raise PatchCoreError("memory bank not loaded")
        feats = self._embed(image)  # [P, D]
        # nearest neighbour distance to the bank (cosine on unit vectors)
        sim = feats @ self._bank.T  # [P, N]
        nn_sim = np.max(sim, axis=1)
        dist = 1.0 - nn_sim  # [P]
        patches_h = self.image_size // 8
        patches_w = self.image_size // 8
        map_p = dist.reshape(patches_h, patches_w)
        # upsample to image resolution
        from PIL import Image as _Img

        def _up(m: np.ndarray, size: int) -> np.ndarray:
            arr = (m - m.min()) / (m.max() - m.min() + 1e-8) * 255.0
            return np.asarray(_Img.fromarray(arr.astype(np.uint8)).resize((size, size), _Img.BILINEAR), dtype=np.float32) / 255.0

        anomaly_map = _up(map_p, self.image_size)
        # normalize map to [0, 1]
        mmin, mmax = float(anomaly_map.min()), float(anomaly_map.max())
        if mmax > mmin:
            anomaly_map = (anomaly_map - mmin) / (mmax - mmin)
        image_score = float(dist.max())
        return anomaly_map, image_score

    def _regions(self, anomaly_map: np.ndarray, threshold: float, image_w: int, image_h: int) -> list[AnomalyRegion]:
        """Connected components above the map threshold -> objective regions."""
        from skimage import measure

        binary = anomaly_map >= min(threshold, 1.0)
        labels = measure.label(binary, connectivity=2)
        regions: list[AnomalyRegion] = []
        if labels.max() == 0:
            return regions
        scale_x = image_w / self.image_size
        scale_y = image_h / self.image_size
        for region in measure.regionprops(labels, intensity_image=anomaly_map):
            if region.area < 4:  # noise floor
                continue
            y1, x1, y2, x2 = region.bbox
            bbox_xyxy = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            regions.append(
                AnomalyRegion(
                    bbox_xyxy=bbox_xyxy,
                    bbox_normalized=(x1 / self.image_size, y1 / self.image_size, x2 / self.image_size, y2 / self.image_size),
                    area_ratio=round(float(region.area) / (self.image_size * self.image_size), 4),
                    region_score=round(float(region.intensity_mean), 4),
                )
            )
        return regions

    def predict(
        self,
        image: Image.Image,
        *,
        image_w: int | None = None,
        image_h: int | None = None,
        include_map_png: bool = True,
    ) -> AnomalyResult:
        if self._threshold is None:
            raise PatchCoreError("threshold not loaded")
        started = time.perf_counter()
        anomaly_map, image_score = self.score(image)
        is_anomalous = image_score >= self._threshold
        w = image_w or image.width
        h = image_h or image.height
        regions = self._regions(anomaly_map, self._threshold, w, h)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return AnomalyResult(
            model_name=self.model_name,
            model_version=self.model_version,
            anomaly_score=round(image_score, 5),
            threshold=round(self._threshold, 5),
            is_anomalous=is_anomalous,
            regions=regions,
            latency_ms=round(latency_ms, 2),
            anomaly_map_png=_map_png(anomaly_map) if include_map_png else None,
        )


def _map_png(anomaly_map: np.ndarray) -> str:
    """Encode a colormapped heatmap as a small base64 PNG (for the review UI)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.imsave(buf, anomaly_map, cmap="jet", vmin=0.0, vmax=1.0, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


__all__ = ["PatchCoreError", "PatchCorePredictor", "utc_now_iso"]
