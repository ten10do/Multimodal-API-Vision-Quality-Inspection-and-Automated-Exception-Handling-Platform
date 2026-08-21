"""Phase 7 — end-to-end factory simulation (1000 products).

Pipeline per product:

    camera frame -> D3 inference -> decision engine -> PLC -> MES -> human review

Two inference backends:
  * ``SyntheticD3Backend`` (default): deterministic, seeded score generator
    calibrated around the FROZEN release threshold. Simulation only — it does
    not touch the model; it exists so the loop is testable and repeatable.
  * ``LiveHttpBackend``: posts real frames to the unchanged inference service
    (``POST /v1/infer``) when a live stack is explicitly requested.

Run:
    python -m industrial_loop.factory_simulator --products 1000
Report: runs/industrial-loop/factory_simulation_report.json (gitignored).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    FROZEN_THRESHOLD,
    MODEL_VERSION,
    RELEASE_ID,
    RUNTIME_ROOT,
    DecisionPolicy,
)
from .dashboard import LoopStore, create_app
from .decision_service import D3InferenceResult, DecisionEngine
from .events import Decision, InspectionEvent, OperatorStatus, PlcStatus, utc_now_iso
from .human_review import HumanReviewWorkflow, ReviewOutcome
from .mes_service import MesService
from .plc_adapter import InMemoryPlc, PlcCommand, PlcTraceLog, plc_status_for

DEFAULT_PRODUCTS = 1000
DEFECT_RATE = 0.08
AI_FAILURE_RATE = 0.01


# --- camera simulation --------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    index: int
    product_id: str
    batch_id: str
    camera_id: str
    defect_injected: bool
    defect_type: str | None


class CameraSimulation:
    """Seeded production line: 3 cameras, one batch per 250 products."""

    def __init__(self, products: int, *, seed: int = 42, defect_rate: float = DEFECT_RATE) -> None:
        self.products = products
        self.rng = np.random.default_rng(seed)
        self.defect_rate = defect_rate

    def frames(self) -> list[Frame]:
        frames = []
        defects = self.rng.random(self.products) < self.defect_rate
        for index in range(self.products):
            number = f"{index + 1:04d}"
            frames.append(
                Frame(
                    index=index,
                    product_id=f"P{number}",
                    batch_id=f"B-{2026}-{index // 250 + 1:03d}",
                    camera_id=f"CAM-{index % 3 + 1:02d}",
                    defect_injected=bool(defects[index]),
                    defect_type="inclusion" if defects[index] and index % 2 else "scratch" if defects[index] else None,
                )
            )
        return frames


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

    def infer(self, frame: Frame) -> tuple[D3InferenceResult, list]:
        if self.rng.random() < self.ai_failure_rate:
            latency = float(self.rng.normal(900, 200))
            return D3InferenceResult.failure("simulated_inference_stream_error", latency_ms=max(latency, 1.0)), []
        thr = self.threshold
        if frame.defect_injected:
            score = thr * (1.0 + abs(self.rng.normal(0.018, 0.012)))
            score = float(np.clip(score, thr * 1.0001, thr * 1.25))
            pixel = float(self.rng.uniform(0.55, 0.95))
        else:
            score = thr * (1.0 - abs(self.rng.normal(0.025, 0.015)))
            score = float(np.clip(score, thr * 0.55, thr * 0.999))
            pixel = float(self.rng.uniform(0.02, 0.30))
        preview = _heatmap_preview(self.rng, hot=frame.defect_injected, amplitude=pixel)
        result = D3InferenceResult(
            ok=True,
            model_version=MODEL_VERSION,
            artifact_version=RELEASE_ID,
            image_score=score,
            pixel_score=pixel,
            threshold=self.threshold,
            confidence=_confidence(score, self.threshold),
            heatmap_reference=f"sim://heatmap/{frame.product_id}.png",
            latency_ms=float(max(self.rng.normal(180, 40), 1.0)),
        )
        return result, preview


class LiveHttpBackend:
    """Optional adapter to the REAL unchanged inference service (/v1/infer)."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        import httpx

        self._client = httpx.Client(timeout=timeout_s)
        self.base_url = base_url.rstrip("/")

    def infer(self, frame: Frame) -> tuple[D3InferenceResult, list]:
        started = time.perf_counter()
        png = _tiny_png(frame)
        try:
            response = self._client.post(
                f"{self.base_url}/v1/infer",
                files={"file": (f"{frame.product_id}.png", png, "image/png")},
                data={"inspection_id": frame.product_id},
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
                heatmap_reference=f"http://{self.base_url}/artifacts/{frame.product_id}/map.png",
                latency_ms=latency,
            ),
            [],
        )


def _tiny_png(frame: Frame) -> bytes:  # pragma: no cover - only used by live backend
    """A minimal valid 8x8 grayscale PNG so /v1/infer accepts the upload."""
    width = height = 8
    raw = b"".join(b"\x00" + bytes([128] * width) for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        import struct
        import zlib

        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


# --- factory simulation -------------------------------------------------------


class FactorySimulator:
    def __init__(
        self,
        *,
        products: int = DEFAULT_PRODUCTS,
        seed: int = 42,
        backend=None,  # noqa: ANN001
        policy: DecisionPolicy | None = None,
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
        self.app = create_app(self.store)
        self.operator_rng = np.random.default_rng(seed + 1)

    # -- loop -----------------------------------------------------------------

    def run(self) -> dict:
        started_wall = utc_now_iso()
        t0 = time.perf_counter()
        self.plc.start()
        latencies: list[float] = []
        inference_latencies: list[float] = []
        ground_truth = {"true_defects": 0, "detected": 0, "missed": 0, "false_rejects": 0}
        supervisor_resumes = 0

        for frame in CameraSimulation(self.products, seed=self.seed).frames():
            step_started = time.perf_counter()
            result, preview = self.backend.infer(frame)
            if result.latency_ms is not None:
                inference_latencies.append(result.latency_ms)
            event = self.engine.decide(
                result,
                product_id=frame.product_id,
                batch_id=frame.batch_id,
                camera_id=frame.camera_id,
                trace_id=f"trc-{frame.index:06d}",
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
            if frame.defect_injected:
                ground_truth["true_defects"] += 1
                if event.decision is Decision.REJECT:
                    ground_truth["detected"] += 1
                else:
                    ground_truth["missed"] += 1
            elif event.decision is Decision.REJECT:
                ground_truth["false_rejects"] += 1

            self.store.add_event(event, heatmap_preview=preview or None)
            latencies.append((time.perf_counter() - step_started) * 1000.0)

        self._resolve_reviews()
        wall = time.perf_counter() - t0
        return self._report(started_wall, utc_now_iso(), wall, latencies, inference_latencies, ground_truth, supervisor_resumes)

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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    backend = None
    if args.backend == "live":
        backend = LiveHttpBackend(args.live_url)
    simulator = FactorySimulator(products=args.products, seed=args.seed, backend=backend)
    report = simulator.run()
    path = simulator.write_report(report, Path(args.out) if args.out else None)
    print(
        f"total={report['total_count']} pass={report['pass_count']} "
        f"reject={report['reject_count']} hold={report['hold_count']}"
    )
    print(f"plc={report['plc_actions']} mes={report['mes_orders']}")
    print(f"report written: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
