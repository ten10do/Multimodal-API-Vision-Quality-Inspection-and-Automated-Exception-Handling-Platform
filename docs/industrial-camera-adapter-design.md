# Industrial Camera Adapter Design (`industrial_loop/camera/`)

Status: **Virtual Camera Adapter** — an architecture extension adding the factory
image-acquisition layer around the frozen D3 release. The D3 model, weights, bank,
threshold, feature extractor, inference results and all existing evaluation metrics are
untouched: the camera layer only produces frames that feed the existing pipeline.

## 1. 工业视觉采集架构 (Acquisition architecture)

```
                Dataset / Replay
                       |
                       v
        +----------------------------+
        |       Camera Adapter       |   industrial_loop/camera/camera_base.py
        |  connect / trigger /       |   VirtualFileCamera (today)
        |  capture / disconnect      |   GigE / USB3 / vendor SDK (future)
        +----------------------------+
                       |  CameraFrame (frame_id, camera_id, timestamp,
                       |               image_path, w x h, trigger_id,
                       |               capture_status)
                       v
                 D3 Inference            (unchanged)
                       |
                 Decision Engine         (unchanged; fail-close)
                       |
              ---------+---------
              v                  v
             PLC                MES
```

Package layout:

| File | Phase | Responsibility |
|---|---|---|
| `camera/camera_base.py` | 1, 5 | `CameraAdapter` ABC, `TriggerInfo`, `CaptureStatus`, `CameraHealthState`, `CameraHealthMonitor`, error taxonomy |
| `camera/frames.py` | 2 | `CameraFrame` schema + `load_image()` into the inference pipeline |
| `camera/virtual_file_camera.py` | 3 | `VirtualFileCamera`: dataset replay as a camera |
| `camera/camera_trigger.py` | 4, 5 | `CameraTriggerService` (PLC interlock + trigger ids), fail-close bridge |

## 2. Camera Adapter 设计

The interface mirrors a machine-vision SDK without binding any vendor:

```python
class CameraAdapter(ABC):
    def connect(self) -> None          # open session (idempotent)
    def disconnect(self) -> None       # close session (idempotent)
    def trigger(self, trigger_id=None) -> TriggerInfo   # arm/fire one shot
    def capture(self) -> CameraFrame   # acquire one frame (on-demand mode)
    def health_check(self) -> dict     # device health probe
    def get_status(self) -> dict       # connection + counters + health
```

* On-demand acquisition: `capture()` requires an armed trigger (`require_trigger=True`);
  the frame is stamped with the pending `trigger_id`.
* Adapters are context managers (`with camera:` maps to connect/disconnect).
* `VirtualFileCamera` replays a dataset folder (read-only) with sequential playback,
  batch replay (`capture_batch(n)`), `reset()`, looping wrap-around and seeded failure
  injection (`failure_rate`, deterministic per seed). Frame ids follow
  `CAM01_000001...`; image dimensions are probed from the file header.
* The factory simulator auto-provisions a small placeholder-image pool under
  `runs/industrial-loop/camera-feed/` when no dataset directory is given, so the loop
  runs hermetically; pass `--dataset-dir` to replay real steel images instead.

### Frame event model

Required fields: `frame_id`, `camera_id`, `timestamp`, `image_path`, `width`, `height`,
`trigger_id`, `capture_status` (+ `sequence_number`, `capture_latency_ms`,
`error_detail`). Validation: FAILED frames must carry `error_detail`; SUCCESS frames
cannot. `load_image()` decodes the frame to a PIL RGB image ready for the unchanged D3
predictor (or the file bytes can be uploaded to `/v1/infer`).

## 3. Trigger 流程

```
PLC READY/RUNNING -> Camera Trigger -> Capture Image -> Inference -> Decision -> PLC Action
```

`CameraTriggerService` owns the PLC-side trigger namespace
(`PLC_TRIGGER_000001`, ...), stamps each frame, and enforces the safety interlock:
a STOPped line refuses triggers (`CameraInterlockError`). In the factory simulation
every product is one full cycle; the PLC action of the previous decision gates the
next trigger exactly as on a real line.

## 4. 异常处理 (fail-close)

Health states `ONLINE / OFFLINE / ERROR` with counters `last_capture_time`,
`frame_count`, `failure_count`. Semantics: OFFLINE until connect / after disconnect;
ERROR after a failed capture (recovers on the next success).

Every acquisition anomaly is converted by `safe_inference_result(...)` into a
`D3InferenceResult.failure(...)`, which the existing decision engine routes to
**HOLD · AI_SYSTEM_FAILURE** — never PASS:

| condition | bridge result |
|---|---|
| transport/lifecycle exception (e.g. disconnected camera) | `camera_error:...` |
| camera health != ONLINE (dominates everything else) | `camera_health_offline/error` |
| failed capture (`capture_status=FAILED`) | `camera_capture_failed:...` |
| healthy frame + healthy camera | `None` → normal inference proceeds |

A STOPped line additionally refuses triggers outright (interlock). No threshold,
score or model path is involved in any of these decisions.

## 5. Factory Simulator 接入结果

`FactorySimulator` now sources every product from the virtual camera
(camera -> frame -> D3). Business results are unchanged versus the pre-camera flow
(same seed): **1000 products -> 913 PASS / 72 REJECT / 15 HOLD**, PLC 72 reject_signals /
15 stop_signals / 0 NACKs, MES 72 orders closed. New report field:

```json
"camera_stats": {
  "total_frames": 1000,
  "success_frames": 1000,
  "failed_frames": 0,
  "average_capture_latency": 0.0009,
  "triggers_issued": 1000,
  "camera_id": "steel-camera-01",
  "final_health": {"state": "ONLINE", "frame_count": 1000, "failure_count": 0}
}
```

With `camera_failure_rate=1.0` the run fails closed end-to-end: every product becomes
HOLD (AI_SYSTEM_FAILURE), the line stops and restarts via supervisor reset, and the
report shows `failed_frames = total_frames`, `final_health.state = ERROR`.

## 6. Tests

`inference-service/tests/test_camera_adapter.py` — 21 tests covering the five required
areas: lifecycle (connect/trigger/capture/disconnect, determinism, loop/reset/batch),
trigger flow (sequential PLC trigger ids, frame stamping, STOP interlock), frame schema
(field completeness, validation, PIL loading), camera failure (offline/ERROR/failed
capture -> HOLD, never PASS), and factory e2e (camera-driven run consistency, full
fail-closed run, tiny-dataset loop replay, cross-instance determinism). Full repo suite:
469 passed.

## 7. 未来真实工业相机扩展路径

The closed loop depends only on `CameraAdapter`, so a physical camera is a new adapter
implementation plus configuration — no changes to inference, decision engine, PLC/MES
or dashboard:

1. **GigE Vision / GenICam**: implement `GigECameraAdapter` over an SDK such as
   Harvesters+GenTL, Aravis, or a vendor GenICam stack; map `connect()` to device
   discovery + stream setup, `trigger()` to hardware/software line trigger,
   `capture()` to buffer fetch, and expose width/height/pixel-format in `get_status()`.
2. **USB3 Vision**: same pattern over a USB3 Vision host library (e.g. Aravis, vendor
   runtime); watch USB bandwidth for high-rate lines.
3. **Vendor SDKs** (Basler pypylon, FLIR Spinnaker, HIK MVS, IDS peak): thin wrappers
   translating callback-based APIs into the synchronous trigger/capture contract;
   keep frame metadata (exposure, gain, timestamp) by extending `CameraFrame` with
   optional fields rather than changing the schema.
4. **Hard-real-time trigger path**: replace `CameraTriggerService` polling with the
   PLC's actual trigger event (OPC UA subscription or hardware line) while keeping the
   same interlock semantics.

Deployment note: adapters are injected into `FactorySimulator(backend=..., ...)` /
the trigger service, so production swaps the virtual camera for a physical one purely
through configuration and dependency injection.
