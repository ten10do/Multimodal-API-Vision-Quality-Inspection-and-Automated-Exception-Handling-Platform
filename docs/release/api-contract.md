# D3 Release API Contract

Contract status: frozen for `steel-patchcore-d3-release@1.3.0`. This document describes the existing HTTP boundary; it does not promote the release.

## Endpoints

- `GET /health`: process liveness. HTTP 200 with `status`, `model_loaded`, and `anomaly_loaded`.
- `GET /ready`: verifies deployment artifacts and model smoke state. `status=ready` is required before traffic.
- `POST /v1/infer`: multipart image inference.

## Input schema

`POST /v1/infer` uses `multipart/form-data`:

| Field | Type | Required | Constraint |
|---|---|---|---|
| `file` | binary image | yes | Decodable image; D3 anomaly branch requires 1600×256 RGB input |
| `inspection_id` | string | no | Caller traceable inspection identifier |

The caller may send `X-Request-ID`. If absent, the service creates `req-` followed by 12 lowercase hexadecimal characters.

## Output schema

HTTP 200 returns the frozen `VisionResult` JSON contract:

- `inspection_id`, `model_name`, `model_version`
- `image_width`, `image_height`
- `detections[]`: class ID/name, confidence, pixel and normalized boxes, area
- `anomaly`: nullable object containing `model_name`, `model_version`, `artifact_version`, `anomaly_score`, frozen `threshold`, `is_anomalous`, regions, latency and optional base64 PNG heatmap
- `fusion_class`: `NORMAL_CANDIDATE`, `KNOWN_DEFECT`, `UNKNOWN_ANOMALY`, or `KNOWN_DEFECT_WITH_ANOMALY`
- `latency_yolo_ms`, `latency_anomaly_ms`, `latency_fusion_ms`, `inference_latency_ms`
- `device`, `timestamp`

Unknown fields are forbidden by the shared Pydantic contract.

## Error codes

| HTTP | Code/state | Meaning | Required handling |
|---:|---|---|---|
| 422 | `invalid_image` | Image cannot be decoded or violates input handling | Reject request; do not infer |
| 422 | FastAPI validation | Missing/invalid multipart field | Reject request |
| 503 | `model_unavailable` | Model load or warm-up failed | Fail closed; hold production decision |
| 503 | `d3_inference_failed` | Required D3 branch timed out, failed artifact loading, or raised at runtime | Fail closed to `HOLD`; never run YOLO-only fusion |
| 500 | `vision_error` | Inference pipeline failure | Fail closed; record trace |
| 200 | `status=not_ready` on `/ready` | Artifact/model verification failed | Do not route traffic |

Errors include `detail.error.code`, `message`, and `request_id` when emitted by the inference handler.

A D3 failure response additionally includes these frozen fields under `detail.error`:

- `decision`: always `HOLD`
- `reason_code`: `d3_inference_timeout`, `d3_artifact_load_failure`, or `d3_runtime_failure`
- `trace_id`: the supplied or generated request trace
- `d3_status`: always `FAILED`
- `error_category`: `timeout`, `artifact_load_failure`, or `runtime_exception`

## Trace format

- Request trace: supplied `X-Request-ID` or `req-[0-9a-f]{12}`.
- Factory trace: `fat-` plus 24 lowercase hexadecimal characters in FAT evidence.
- Industrial command idempotency: `cmd-{inspection_id}-{RELEASE|REJECT|HOLD}`.
- Timestamps use ISO 8601 UTC with `Z` or an explicit UTC offset.

## Decision workflow

1. Vision output is objective evidence; it does not directly actuate equipment.
2. Normal candidate evidence may become business `PASS` and industrial `RELEASE` only after the full decision pipeline succeeds.
3. Known or human-confirmed defect becomes `FAIL` and `REJECT`.
4. Unknown anomaly becomes `REVIEW_REQUIRED` and `HOLD`.
5. Missing artifact, hash mismatch, invalid input, timeout, model failure, unknown state or unavailable downstream integration must fail closed to `HOLD`.
6. Human feedback is recorded separately and never modifies the prediction snapshot, model, artifact or threshold.

The required D3 branch fails closed before fusion. Legacy deployments without a configured D3 candidate retain their existing optional PatchCore behavior. Successful `VisionResult` output and normal fusion semantics are unchanged.
