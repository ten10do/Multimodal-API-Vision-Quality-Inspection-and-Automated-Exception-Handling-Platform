from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from vision_contract import VisionResult

from .fusion import fuse
from .patchcore_predictor import PatchCoreError, PatchCorePredictor
from .yolo_predictor import ModelLoadError, VisionError, VisionInputError, YoloPredictor

logger = logging.getLogger(__name__)

WEIGHTS_DEFAULT = Path(__file__).resolve().parents[1] / "models" / "best.pt"
WEIGHTS = Path(os.environ.get("IVQC_WEIGHTS", WEIGHTS_DEFAULT))
PATCHCORE_BANK_DEFAULT = Path(__file__).resolve().parents[1] / "models" / "patchcore-bottle" / "bank.npz"
PATCHCORE_BANK = Path(os.environ.get("IVQC_PATCHCORE_BANK", str(PATCHCORE_BANK_DEFAULT)))

_predictor: YoloPredictor | None = None
_anomaly: PatchCorePredictor | None = None
_load_error: str | None = None


def get_predictor() -> YoloPredictor:
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor
    if _load_error is not None:
        raise ModelLoadError(_load_error)
    try:
        _predictor = YoloPredictor(
            WEIGHTS,
            model_name="yolov8s",
            model_version=os.environ.get("IVQC_MODEL_VERSION", "phase1-baseline"),
            device=os.environ.get("IVQC_DEVICE") or None,
        )
        _predictor.predict(np.zeros((64, 64, 3), dtype=np.uint8), inspection_id="warmup")  # triggers CUDA warmup
        logger.info("predictor loaded from %s on %s", WEIGHTS, _predictor.device)
    except Exception as exc:  # pragma: no cover - depends on runtime assets
        _load_error = f"model load failed: {exc}"
        logger.exception("model load failed")
        raise ModelLoadError(_load_error) from exc
    return _predictor


def get_anomaly_predictor() -> PatchCorePredictor | None:
    """Load PatchCore lazily; return None when the bank is missing or the
    model cannot load. The YOLO path must never be blocked by it."""
    global _anomaly
    if _anomaly is not None:
        return _anomaly
    if not PATCHCORE_BANK.exists():
        logger.warning("patchcore bank missing: %s (anomaly channel disabled)", PATCHCORE_BANK)
        return None
    try:
        _anomaly = PatchCorePredictor(PATCHCORE_BANK, device=os.environ.get("IVQC_DEVICE") or None)
        logger.info("patchcore loaded from %s on %s", PATCHCORE_BANK, _anomaly.device)
        return _anomaly
    except Exception as exc:  # pragma: no cover
        logger.exception("patchcore load failed; anomaly channel disabled")
        return None


def create_app() -> FastAPI:
    app = FastAPI(title="IndustrialVision-QC Inference Service", version="0.2.0")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": _predictor is not None,
            "anomaly_loaded": _anomaly is not None,
        }

    @app.get("/ready")
    async def ready() -> dict:
        try:
            get_predictor()
            return {"status": "ready", "model_loaded": True, "anomaly_loaded": _anomaly is not None}
        except ModelLoadError:
            return {"status": "not_ready", "model_loaded": False, "anomaly_loaded": _anomaly is not None}

    @app.post("/v1/infer", response_model=VisionResult)
    async def infer(
        request: Request,
        file: UploadFile = File(...),
        inspection_id: str | None = Form(default=None),
    ) -> VisionResult:
        rid = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        data = await file.read()
        start = time.perf_counter()
        try:
            predictor = get_predictor()
            image = predictor._read_image(data)
            height, width = image.shape[:2]
            pil_image = _to_pil(image)

            # YOLO
            t0 = time.perf_counter()
            yolo = predictor.predict(image, inspection_id=inspection_id)
            latency_yolo = (time.perf_counter() - t0) * 1000.0

            # PatchCore (independent channel, never blocks YOLO)
            latency_anomaly = 0.0
            anomaly_result = None
            anomaly_predictor = get_anomaly_predictor()
            if anomaly_predictor is not None:
                t1 = time.perf_counter()
                try:
                    anomaly_result = anomaly_predictor.predict(
                        pil_image, image_w=width, image_h=height, include_map_png=True
                    )
                except Exception as exc:  # noqa: BLE001 - the anomaly channel is best-effort
                    logger.warning("patchcore inference failed request_id=%s: %s", rid, exc)
                    anomaly_result = None
                latency_anomaly = (time.perf_counter() - t1) * 1000.0

            # Fusion
            t2 = time.perf_counter()
            fusion_class = fuse(len(yolo.detections), anomaly_result.is_anomalous if anomaly_result else None)
            latency_fusion = (time.perf_counter() - t2) * 1000.0
            latency_total = (time.perf_counter() - start) * 1000.0

            result = VisionResult(
                inspection_id=yolo.inspection_id,
                model_name=yolo.model_name,
                model_version=yolo.model_version,
                image_width=width,
                image_height=height,
                detections=yolo.detections,
                anomaly=anomaly_result,
                fusion_class=fusion_class,
                latency_yolo_ms=round(latency_yolo, 2),
                latency_anomaly_ms=round(latency_anomaly, 2),
                latency_fusion_ms=round(latency_fusion, 2),
                inference_latency_ms=round(latency_total, 2),
                device=yolo.device,
                timestamp=yolo.timestamp,
            )
        except VisionInputError as exc:
            logger.warning("vision input error request_id=%s: %s", rid, exc)
            raise HTTPException(status_code=422, detail={"error": {"code": "invalid_image", "message": str(exc), "request_id": rid}}) from exc
        except ModelLoadError as exc:
            logger.error("model load error request_id=%s: %s", rid, exc)
            raise HTTPException(status_code=503, detail={"error": {"code": "model_unavailable", "message": str(exc), "request_id": rid}}) from exc
        except VisionError as exc:
            logger.error("vision error request_id=%s: %s", rid, exc)
            raise HTTPException(status_code=500, detail={"error": {"code": "vision_error", "message": str(exc), "request_id": rid}}) from exc

        logger.info(
            "infer ok request_id=%s inspection=%s detections=%d fusion=%s yolo=%.1fms anomaly=%.1fms total=%.1fms",
            rid,
            result.inspection_id,
            len(result.detections),
            result.fusion_class,
            result.latency_yolo_ms,
            result.latency_anomaly_ms,
            result.inference_latency_ms,
        )
        return result

    return app


def _to_pil(image: np.ndarray):
    from PIL import Image

    if image.ndim == 3:
        return Image.fromarray(image[..., ::-1])  # BGR -> RGB
    return Image.fromarray(image)


app = create_app()
