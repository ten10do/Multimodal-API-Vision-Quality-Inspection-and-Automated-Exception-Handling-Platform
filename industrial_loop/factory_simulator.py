"""Phase 7 — end-to-end factory simulation (1000 products).

Pipeline per product:

    camera trigger+capture -> CameraFrame -> D3 inference -> decision engine
        -> PLC -> MES -> human review

The acquisition layer is the virtual industrial camera
(``industrial_loop.camera``): every product enters the loop as a triggered
``CameraFrame``; a failed capture or an unhealthy camera is bridged into the
decision engine as an inference failure (fail-close HOLD), never a PASS.

Inference backends:
  * ``SyntheticD3Backend`` (default): deterministic, seeded score generator
    calibrated around the FROZEN release threshold. Simulation only — it does
    not touch the model; it exists so the loop is testable and repeatable.
  * ``LiveHttpBackend``: posts the captured frame to the unchanged inference
    service (``POST /v1/infer``) when a live stack is explicitly requested.

Run:
    python -m industrial_loop.factory_simulator --products 1000
Report: runs/industrial-loop/factory_simulation_report.json (gitignored).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .camera.camera_base import CaptureStatus
from .camera.camera_trigger import CameraTriggerService, safe_inference_result
from .camera.frames import CameraFrame
from .camera.virtual_file_camera import VirtualFileCamera, write_placeholder_png
from .config import (
    FROZEN_THRESHOLD,
    MODEL_VERSION,
    PROJECT_ROOT,
    RELEASE_ID,
    RUNTIME_ROOT,
    DecisionPolicy,
)
from .dashboard import LoopStore, create_app
from .decision_service import D3InferenceResult, DecisionEngine
from .events import Decision, OperatorStatus, PlcStatus, utc_now_iso
from .human_review import HumanReviewWorkflow, ReviewOutcome
from .mes_service import MesService
from .plc_adapter import InMemoryPlc, PlcCommand, PlcTraceLog, plc_status_for
from model_governance import ModelLifecycleManager

DEFAULT_PRODUCTS = 1000
DEFECT_RATE = 0.08
AI_FAILURE_RATE = 0.01
DEFAULT_CAMERA_ID = "steel-camera-01"
CAMERA_FEED_POOL = 24


# --- inference backends -------------------------------------------------------


def _confidence(score: float, threshold: float) -> dict:
    return {
        "kind": "absolute_threshold_margin_ratio",
        "value": min(abs(score - threshold) / threshold, 1.0),
        "calibrated_probability": False,
    }


def _heatmap_preview(rng: np.random.Generator, hot: bool, amplitude: float) -> list:
    """12x12 normalized grid the dashboard renders as a heatmap canvas."""
    grid = rng.uniform(0.02, 0.10, size=(12, 12))
    if hot:
        cy, cx = rng.integers(3, 9), rng.integers(3, 9)
        yy, xx = np.mgrid[0:12, 0:12]
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)) * amplitude
        grid = np.clip(grid + blob, 0.0, 1.0)
    return [[round(float(v), 4) for v in row] for row in grid]


class SyntheticD3Backend:
    """Deterministic D3 stand-in anchored to the frozen threshold (simulation only)."""

    def __init__(self, *, seed: int = 42, threshold: float = FROZEN_THRESHOLD,
                 ai_failure_rate: float = AI_FAILURE_RATE) -> None:
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.ai_failure_rate = ai_failure_rate

    def infer(self, frame: CameraFrame, *, defect_injected: bool, product_id: str) -> tuple[D3InferenceResult, list]:
        if self.rng.random() < self.ai_failure_rate:
            latency = float(self.rng.normal(900, 200))
            return D3InferenceResult.failure("simulated_inference_stream_error", latency_ms=max(latency, 1.0)), []
        thr = self.threshold
        if defect_injected:
            score = thr * (1.0 + abs(self.rng.normal(0.018, 0.012)))
            score = float(np.clip(score, thr * 1.0001, thr * 1.25))
            pixel = float(self.rng.uniform(0.55, 0.95))
        else:
            score = thr * (1.0 - abs(self.rng.normal(0.025, 0.015)))
            score = float(np.clip(score, thr * 0.55, thr * 0.999))
            pixel = float(self.rng.uniform(0.02, 0.30))
        preview = _heatmap_preview(self.rng, hot=defect_injected, amplitude=pixel)
        result = D3InferenceResult(
            ok=True,
            model_version=MODEL_VERSION,
            artifact_version=RELEASE_ID,
            image_score=score,
            pixel_score=pixel,
            threshold=self.threshold,
            confidence=_confidence(score, self.threshold),
            heatmap_reference=f"sim://heatmap/{product_id}.png",
            latency_ms=float(max(self.rng.normal(180, 40), 1.0)),
        )
        return result, preview


class LiveHttpBackend:
    """Optional adapter to the REAL unchanged inference service (/v1/infer)."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        import httpx

        self._client = httpx.Client(timeout=timeout_s)
        self.base_url = base_url.rstrip("/")

    def infer(self, frame: CameraFrame, *, defect_injected: bool, product_id: str) -> tuple[D3InferenceResult, list]:
        started = time.perf_counter()
        image_file = frame.image_file()
        payload = image_file.read_bytes() if image_file.exists() else _tiny_png()
        try:
            response = self._client.post(
                f"{self.base_url}/v1/infer",
                files={"file": (f"{product_id}.png", payload, "image/png")},
                data={"inspection_id": product_id},
            )
        except Exception as exc:  # noqa: BLE001 - fail closed on transport errors
            return D3InferenceResult.failure(f"transport_error:{exc}"), []
        latency = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            detail = str(response.json().get("detail", ""))[:160] if response.headers.get("content-type", "").startswith("application/json") else response.text[:160]
            return D3InferenceResult.failure(f"http_{response.status_code}:{detail}", latency_ms=latency), []
        body = response.json()
        anomaly = body.get("anomaly") or {}
        score, threshold = anomaly.get("anomaly_score"), anomaly.get("threshold")
        if score is None or threshold is None:
            return D3InferenceResult.failure("missing_anomaly_payload", latency_ms=latency), []
        return (
            D3InferenceResult(
                ok=True,
                model_version=anomaly.get("model_version"),
                artifact_version=anomaly.get("artifact_version"),
                image_score=float(score),
                pixel_score=None,
                threshold=float(threshold),
                confidence=None,
                heatmap_reference=f"http://{self.base_url}/artifacts/{product_id}/map.png",
                latency_ms=latency,
            ),
            [],
        )


def _tiny_png() -> bytes:  # pragma: no cover - fallback upload payload
    """A minimal valid 8x8 grayscale PNG so /v1/infer accepts the upload."""
    import struct
    import zlib

    width = height = 8
    raw = b"".join(b"\x00" + bytes([128] * width) for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


# --- camera feed provisioning -------------------------------------------------


def ensure_camera_feed(dataset_dir: str | Path | None, products: int) -> Path:
    """Resolve the replay source: a real dataset dir or a generated pool."""
    if dataset_dir is not None:
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"camera dataset not found: {root}")
        return root
    pool_dir = RUNTIME_ROOT / "camera-feed"
    existing = sorted(pool_dir.glob("feed_*.png"))
    needed = min(max(products, 1), CAMERA_FEED_POOL)
    if len(existing) < needed:
        for index in range(len(existing), needed):
            write_placeholder_png(pool_dir / f"feed_{index:04d}.png", gray=100 + index * 6)
    return pool_dir


# --- factory simulation -------------------------------------------------------


class FactorySimulator:
    def __init__(
        self,
        *,
        products: int = DEFAULT_PRODUCTS,
        seed: int = 42,
        backend=None,  # noqa: ANN001
        policy: DecisionPolicy | None = None,
        dataset_dir: str | Path | None = None,
        camera_failure_rate: float = 0.0,
        camera_id: str = DEFAULT_CAMERA_ID,
    ) -> None:
        self.products = products
        self.seed = seed
        self.policy = policy or DecisionPolicy()
        self.backend = backend or SyntheticD3Backend(seed=seed)
        self.engine = DecisionEngine(self.policy)
        self.plc = InMemoryPlc()
        self.mes = MesService()
        self.review = HumanReviewWorkflow(mes=self.mes)
        self.store = LoopStore(plc=self.plc, mes=self.mes, review=self.review)
        self.lifecycle = ModelLifecycleManager(
            PROJECT_ROOT / "model_governance/model_history.json",
            project_root=PROJECT_ROOT,
        )
        self.app = create_app(self.store, lifecycle_manager=self.lifecycle)
        self.operator_rng = np.random.default_rng(seed + 1)
        # acquisition layer (Phase 6): camera -> frame -> D3
        self.camera = VirtualFileCamera(
            ensure_camera_feed(dataset_dir, products),
            camera_id=camera_id,
            seed=seed,
            failure_rate=camera_failure_rate,
        )
        self.trigger_service = CameraTriggerService(plc=self.plc)

    # -- loop -----------------------------------------------------------------

    def run(self) -> dict:
        started_wall = utc_now_iso()
        t0 = time.perf_counter()
        self.plc.start()
        latencies: list[float] = []
        inference_latencies: list[float] = []
        ground_truth = {"true_defects": 0, "detected": 0, "missed": 0, "false_rejects": 0}
        supervisor_resumes = 0
        camera_stats = {"total_frames": 0, "success_frames": 0, "failed_frames": 0}
        capture_latencies: list[float] = []
        triggers_used = 0

        # deterministic defect schedule (identical draw pattern as the legacy
        # camera simulation, so business results stay consistent end-to-end)
        defects = np.random.default_rng(self.seed).random(self.products) < DEFECT_RATE

        with self.camera:
            for index in range(self.products):
                step_started = time.perf_counter()
                product_id = f"P{index + 1:04d}"
                batch_id = f"B-{2026}-{index // 250 + 1:03d}"

                # --- acquisition: PLC-gated trigger + capture ---------------
                frame: CameraFrame | None = None
                camera_error: str | None = None
                try:
                    trigger = self.trigger_service.request_trigger(self.camera)
                    triggers_used += 1
                    frame = self.camera.capture()
                except Exception as exc:  # noqa: BLE001 - any acquisition error fails closed
                    camera_error = str(exc)
                if frame is not None:
                    camera_stats["total_frames"] += 1
                    if frame.capture_status is CaptureStatus.SUCCESS:
                        camera_stats["success_frames"] += 1
                        if frame.capture_latency_ms is not None:
                            capture_latencies.append(frame.capture_latency_ms)
                    else:
                        camera_stats["failed_frames"] += 1

                # --- inference (unchanged D3 contract) ----------------------
                preview: list = []
                bridge = safe_inference_result(
                    frame, self.camera.health.snapshot(), error=camera_error
                )
                if bridge is not None:
                    result = bridge
                else:
                    result, preview = self.backend.infer(
                        frame, defect_injected=bool(defects[index]), product_id=product_id
                    )
                if result.latency_ms is not None:
                    inference_latencies.append(result.latency_ms)

                event = self.engine.decide(
                    result,
                    product_id=product_id,
                    batch_id=batch_id,
                    camera_id=frame.camera_id if frame is not None else self.camera.camera_id,
                    trace_id=f"trc-{index:06d}",
                )

                plc_result = self.plc.apply(
                    PlcCommand(
                        command_id=f"cmd-{event.id}",
                        event_id=event.id,
                        product_id=event.product_id,
                        decision=event.decision,
                    )
                )
                status = plc_status_for(plc_result)
                if not plc_result.ack and event.decision is not Decision.PASS:
                    status = PlcStatus.NACK
                event = event.with_updates(plc_status=status)
                if plc_result.state_after.value == "STOP":
                    self.plc.reset()
                    supervisor_resumes += 1

                if event.decision is Decision.REJECT:
                    order = self.mes.create_from_event(event)
                    event = event.with_updates(mes_status=order.status.value)
                if event.decision in (Decision.REJECT, Decision.HOLD):
                    self.review.enqueue(event)
                    event = event.with_updates(operator_status=OperatorStatus.PENDING)

                # ground-truth accounting (simulation only)
                if bool(defects[index]):
                    ground_truth["true_defects"] += 1
                    if event.decision is Decision.REJECT:
                        ground_truth["detected"] += 1
                    else:
                        ground_truth["missed"] += 1
                elif event.decision is Decision.REJECT:
                    ground_truth["false_rejects"] += 1

                self.store.add_event(event, heatmap_preview=preview or None)
                latencies.append((time.perf_counter() - step_started) * 1000.0)

            # end-of-production health (inside the session, before disconnect)
            final_camera_health = self.camera.health.snapshot()
        self._resolve_reviews()
        wall = time.perf_counter() - t0
        return self._report(
            started_wall, utc_now_iso(), wall, latencies, inference_latencies,
            ground_truth, supervisor_resumes, camera_stats, capture_latencies,
            triggers_used, final_camera_health,
        )

    # -- operators ------------------------------------------------------------

    def _resolve_reviews(self) -> None:
        for _ in range(4):  # bounded passes; REQUEST_RECHECK re-enters the queue
            for event in list(self.review.pending()):
                roll = self.operator_rng.random()
                if event.decision is Decision.HOLD:
                    outcome = ReviewOutcome.REQUEST_RECHECK if roll < 0.5 else ReviewOutcome.FALSE_ALARM
                elif roll < 0.70:
                    outcome = ReviewOutcome.CONFIRM_DEFECT
                elif roll < 0.90:
                    outcome = ReviewOutcome.FALSE_ALARM
                else:
                    outcome = ReviewOutcome.REQUEST_RECHECK
                record, updated = self.review.submit(
                    event.id,
                    reviewer=f"operator-{self.operator_rng.integers(1, 5)}",
                    outcome=outcome,
                    comment="factory simulation review",
                )
                if updated is None:
                    continue
                order = self.mes.find_by_event(event.id)
                mes_status = order.status.value if order is not None else updated.mes_status.value
                self.store.update_event(updated.with_updates(mes_status=mes_status))
            if self.review.pending_count() == 0:
                break

    # -- report ---------------------------------------------------------------

    def _report(
        self,
        started_at: str,
        finished_at: str,
        wall_s: float,
        latencies: list[float],
        inference_latencies: list[float],
        truth: dict,
        supervisor_resumes: int,
        camera_stats: dict,
        capture_latencies: list[float],
        triggers_used: int,
        final_camera_health: dict,
    ) -> dict:
        summary = self.store.summary()
        stats = self.engine.stats.snapshot()
        plc_trace = self.plc.trace_log.read_all()
        report = {
            "schema_version": "industrial_factory_simulation_report_v1",
            "release": RELEASE_ID,
            "lineage": {
                "model_version": self.policy.expected_model_version,
                "reject_threshold": self.policy.reject_threshold,
                "threshold_modified": False,
                "d3_model_modified": False,
                "backend": type(self.backend).__name__,
            },
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_clock_seconds": round(wall_s, 3),
            "total_count": summary["total"],
            "pass_count": summary["pass"],
            "reject_count": summary["reject"],
            "hold_count": summary["hold"],
            "decisions": stats,
            "plc_actions": {
                "total": len(plc_trace),
                "pass_kept_running": self.plc.counters["pass_kept_running"],
                "reject_signals": self.plc.counters["reject_signals"],
                "stop_signals": self.plc.counters["stop_signals"],
                "nacks": sum(1 for row in plc_trace if not row["ack"]),
                "supervisor_resumes": supervisor_resumes,
                "final_state": self.plc.state.value,
            },
            "mes_orders": self.mes.counts(),
            "reviews": self.review.counts(),
            "camera_stats": {
                **camera_stats,
                "average_capture_latency": round(sum(capture_latencies) / len(capture_latencies), 4)
                if capture_latencies
                else None,
                "triggers_issued": triggers_used,
                "camera_id": self.camera.camera_id,
                "final_health": final_camera_health,
            },
            "latency": {
                "loop_avg_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
                "loop_p95_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
                "loop_max_ms": round(max(latencies), 3) if latencies else None,
                "inference_avg_ms": round(sum(inference_latencies) / len(inference_latencies), 3)
                if inference_latencies
                else None,
            },
            "ground_truth_check": {
                **truth,
                "detection_rate": round(truth["detected"] / truth["true_defects"], 4) if truth["true_defects"] else None,
                "false_reject_rate": round(
                    truth["false_rejects"] / (summary["total"] - truth["true_defects"]), 4
                )
                if (summary["total"] - truth["true_defects"]) > 0
                else None,
                "note": "synthetic-backend metric only; validates loop wiring, not D3 quality",
            },
        }
        return report

    def write_report(self, report: dict, out_path: Path | None = None) -> Path:
        path = out_path or (RUNTIME_ROOT / "factory_simulation_report.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def main() -> int:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Industrial closed-loop factory simulation")
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["synthetic", "live"], default="synthetic")
    parser.add_argument("--live-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset-dir", default=None,
                        help="optional image folder for the virtual camera (default: generated feed pool)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    backend = None
    if args.backend == "live":
        backend = LiveHttpBackend(args.live_url)
    simulator = FactorySimulator(
        products=args.products, seed=args.seed, backend=backend, dataset_dir=args.dataset_dir
    )
    report = simulator.run()
    path = simulator.write_report(report, Path(args.out) if args.out else None)
    print(
        f"total={report['total_count']} pass={report['pass_count']} "
        f"reject={report['reject_count']} hold={report['hold_count']}"
    )
    print(f"plc={report['plc_actions']} mes={report['mes_orders']}")
    print(f"camera={ {k: report['camera_stats'][k] for k in ('total_frames', 'success_frames', 'failed_frames')} }")
    print(f"report written: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
