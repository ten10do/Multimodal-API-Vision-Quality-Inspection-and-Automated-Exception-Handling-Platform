from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from vision_contract import InferenceResult

from .yolo_predictor import ModelLoadError, VisionInputError, YoloPredictor

logger = logging.getLogger(__name__)

WEIGHTS_DEFAULT = Path(__file__).resolve().parents[1] / "models" / "best.pt"
WEIGHTS = Path(os.environ.get("IVQC_WEIGHTS", WEIGHTS_DEFAULT))

_predictor: YoloPredictor | None = None
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


def create_app() -> FastAPI:
    app = FastAPI(title="IndustrialVision-QC Inference Service", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model_loaded": _predictor is not None}

    @app.get("/ready")
    async def ready() -> dict:
        try:
            get_predictor()
            return {"status": "ready", "model_loaded": True}
        except ModelLoadError:
            return {"status": "not_ready", "model_loaded": False}

    @app.post("/v1/infer", response_model=InferenceResult)
    async def infer(
        request: Request,
        file: UploadFile = File(...),
        inspection_id: str | None = Form(default=None),
    ) -> InferenceResult:
        rid = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        data = await file.read()
        start = time.perf_counter()
        try:
            predictor = get_predictor()
            result = predictor.predict(data, inspection_id=inspection_id)
        except VisionInputError as exc:
            logger.warning("vision input error request_id=%s: %s", rid, exc)
            raise HTTPException(status_code=422, detail={"error": {"code": "invalid_image", "message": str(exc), "request_id": rid}}) from exc
        except ModelLoadError as exc:
            logger.error("model load error request_id=%s: %s", rid, exc)
            raise HTTPException(status_code=503, detail={"error": {"code": "model_unavailable", "message": str(exc), "request_id": rid}}) from exc

        logger.info(
            "infer ok request_id=%s inspection=%s detections=%d latency=%.1fms total=%.1fms",
            rid,
            result.inspection_id,
            len(result.detections),
            result.inference_latency_ms,
            (time.perf_counter() - start) * 1000,
        )
        return result

    return app


app = create_app()
