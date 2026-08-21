"""Industrial camera adapter tests (lifecycle / trigger / schema / failure / e2e)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_loop.camera.camera_base import (
    CameraConnectionError,
    CameraError,
    CameraHealthState,
    CameraInterlockError,
    CameraNotTriggeredError,
    CaptureStatus,
)
from industrial_loop.camera.camera_trigger import (
    CameraTriggerService,
    safe_inference_result,
)
from industrial_loop.camera.frames import CameraFrame
from industrial_loop.camera.virtual_file_camera import (
    VirtualFileCamera,
    write_placeholder_png,
)
from industrial_loop.decision_service import DecisionEngine
from industrial_loop.events import Decision
from industrial_loop.factory_simulator import FactorySimulator
from industrial_loop.plc_adapter import InMemoryPlc, PlcCommand


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    for index in range(3):
        write_placeholder_png(tmp_path / f"strip_{index}.png", width=160, height=40, gray=90 + index * 30)
    return tmp_path


def _stop_plc(plc: InMemoryPlc) -> None:
    plc.apply(PlcCommand(command_id="stop-1", event_id="e", product_id="P", decision=Decision.HOLD))


# --- 1. camera lifecycle -------------------------------------------------------

class TestCameraLifecycle:
    def test_capture_requires_connect_then_trigger(self, dataset):
        camera = VirtualFileCamera(dataset)
        with pytest.raises(CameraConnectionError):
            camera.capture()
        camera.connect()
        with pytest.raises(CameraNotTriggeredError):
            camera.capture()
        camera.trigger("PLC_TRIGGER_000001")
        frame = camera.capture()
        assert frame.capture_status is CaptureStatus.SUCCESS
        camera.disconnect()
        with pytest.raises(CameraConnectionError):
            camera.capture()

    def test_sequential_playback_is_deterministic(self, dataset):
        first = VirtualFileCamera(dataset)
        second = VirtualFileCamera(dataset)
        with first, second:
            batch_a = first.capture_batch(6)
            batch_b = second.capture_batch(6)
        assert [f.image_path for f in batch_a] == [f.image_path for f in batch_b]
        assert [f.frame_id for f in batch_a] == [f.frame_id for f in batch_b]
        # 3 images replayed in order, wrapping
        assert batch_a[0].sequence_number == 1 and batch_a[0].frame_id.endswith("000001")
        assert Path(batch_a[0].image_path).name == "strip_0.png"
        assert Path(batch_a[3].image_path).name == "strip_0.png"  # wrapped

    def test_loop_disabled_exhausts_dataset(self, dataset):
        camera = VirtualFileCamera(dataset, loop=False)
        with camera:
            camera.capture_batch(3)
            camera.trigger("PLC_TRIGGER_000004")
            with pytest.raises(CameraConnectionError):
                camera.capture()

    def test_reset_restarts_playback(self, dataset):
        camera = VirtualFileCamera(dataset)
        with camera:
            first = camera.capture_batch(3)
            camera.reset()
            again = camera.capture_batch(3)
        assert [f.image_path for f in first] == [f.image_path for f in again]

    def test_health_and_status_tracking(self, dataset):
        camera = VirtualFileCamera(dataset)
        assert camera.health.state is CameraHealthState.OFFLINE
        camera.connect()
        camera.capture_batch(2)
        snapshot = camera.health.snapshot()
        assert snapshot["state"] == "ONLINE"
        assert snapshot["frame_count"] == 2 and snapshot["failure_count"] == 0
        assert snapshot["last_capture_time"] is not None
        status = camera.get_status()
        assert status["connected"] is True and status["triggers_issued"] == 2
        camera.disconnect()
        assert camera.health.state is CameraHealthState.OFFLINE

    def test_seeded_failure_injection_is_deterministic(self, dataset):
        cam_a = VirtualFileCamera(dataset, failure_rate=0.5, seed=11)
        cam_b = VirtualFileCamera(dataset, failure_rate=0.5, seed=11)
        with cam_a, cam_b:
            outcomes_a = [f.capture_status for f in cam_a.capture_batch(12)]
            outcomes_b = [f.capture_status for f in cam_b.capture_batch(12)]
        assert outcomes_a == outcomes_b
        assert CaptureStatus.FAILED in outcomes_a


# --- 2. trigger flow (PLC -> camera) ------------------------------------------

class TestTriggerFlow:
    def test_plc_trigger_ids_are_sequential_and_stamped_on_frames(self, dataset):
        plc = InMemoryPlc()
        plc.start()
        service = CameraTriggerService(plc=plc)
        camera = VirtualFileCamera(dataset)
        with camera:
            record, frame = service.capture_cycle(camera)
            assert record.trigger_id == "PLC_TRIGGER_000001"
            assert record.source == "PLC" and record.plc_state_before == "RUNNING"
            assert record.timestamp.endswith("Z")
            assert frame.trigger_id == record.trigger_id
            assert frame.camera_id == camera.camera_id
            _, frame2 = service.capture_cycle(camera)
            assert frame2.trigger_id == "PLC_TRIGGER_000002"

    def test_plc_interlock_refuses_trigger_when_line_stopped(self, dataset):
        plc = InMemoryPlc()
        plc.start()
        service = CameraTriggerService(plc=plc)
        camera = VirtualFileCamera(dataset)
        _stop_plc(plc)
        with camera:
            with pytest.raises(CameraInterlockError):
                service.request_trigger(camera)

    def test_trigger_requires_connection(self, dataset):
        service = CameraTriggerService()
        with pytest.raises(CameraError):
            service.request_trigger(VirtualFileCamera(dataset))


# --- 3. frame schema ----------------------------------------------------------

class TestCameraFrameSchema:
    def test_required_fields_complete(self):
        frame = CameraFrame(
            frame_id="CAM01_000001",
            camera_id="steel-camera-01",
            timestamp="2026-01-01T00:00:00.000Z",
            image_path="feed/strip_0.png",
            width=1600,
            height=256,
            trigger_id="PLC_TRIGGER_000001",
            capture_status="SUCCESS",
        )
        payload = frame.model_dump()
        for field in (
            "frame_id", "camera_id", "timestamp", "image_path", "width",
            "height", "trigger_id", "capture_status",
        ):
            assert field in payload

    def test_failed_frames_require_error_detail(self):
        with pytest.raises(Exception):
            CameraFrame(
                frame_id="F", camera_id="c", image_path="x", width=1, height=1,
                trigger_id="t", capture_status="FAILED",
            )
        with pytest.raises(Exception):
            CameraFrame(
                frame_id="F", camera_id="c", image_path="x", width=1, height=1,
                trigger_id="t", capture_status="SUCCESS", error_detail="oops",
            )

    def test_dimensions_must_be_positive(self):
        with pytest.raises(Exception):
            CameraFrame(
                frame_id="F", camera_id="c", image_path="x", width=0, height=1,
                trigger_id="t", capture_status="SUCCESS",
            )

    def test_frame_loads_into_inference_pipeline(self, dataset):
        camera = VirtualFileCamera(dataset)
        with camera:
            camera.trigger("PLC_TRIGGER_000001")
            frame = camera.capture()
        image = frame.load_image()  # PIL RGB, ready for the D3 predictor
        assert image.size == (160, 40)
        assert frame.width == 160 and frame.height == 40  # probed from the file


# --- 4. camera failure -> HOLD (fail-close) ------------------------------------

class TestCameraFailureHold:
    def _decide(self, result):
        return DecisionEngine().decide(result, product_id="P1", batch_id="B", camera_id="C")

    def test_failed_capture_bridges_to_hold_never_pass(self, dataset):
        camera = VirtualFileCamera(dataset, failure_rate=1.0)
        with camera:
            camera.trigger("PLC_TRIGGER_000001")
            frame = camera.capture()
            snapshot = camera.health.snapshot()
        assert frame.capture_status is CaptureStatus.FAILED
        result = safe_inference_result(frame, snapshot)
        assert result is not None and result.ok is False
        event = self._decide(result)
        assert event.decision is Decision.HOLD
        assert event.reason_code.value == "AI_SYSTEM_FAILURE"
        assert event.error_detail.startswith("camera_")

    def test_unhealthy_camera_bridges_to_hold(self, dataset):
        camera = VirtualFileCamera(dataset, failure_rate=1.0)
        with camera:
            camera.trigger("t1")
            camera.capture()  # health -> ERROR
            snapshot = camera.health.snapshot()
        assert snapshot["state"] == "ERROR"
        result = safe_inference_result(None, snapshot)
        event = self._decide(result)
        assert event.decision is Decision.HOLD
        assert "camera_health" in event.error_detail

    def test_disconnected_camera_bridges_to_hold(self, dataset):
        camera = VirtualFileCamera(dataset)
        error = None
        try:
            camera.capture()
        except CameraConnectionError as exc:
            error = str(exc)
        assert error is not None
        event = self._decide(safe_inference_result(None, None, error=error))
        assert event.decision is Decision.HOLD
        assert event.reason_code.value == "AI_SYSTEM_FAILURE"

    def test_healthy_frame_returns_none_bridge(self, dataset):
        camera = VirtualFileCamera(dataset)
        with camera:
            camera.trigger("t1")
            frame = camera.capture()
            snapshot = camera.health.snapshot()
        assert snapshot["state"] == "ONLINE"
        assert safe_inference_result(frame, snapshot) is None


# --- 5. factory e2e: camera -> D3 -> decision -> PLC -> MES ---------------------

class TestFactoryCameraE2E:
    def test_camera_driven_run_is_consistent(self, dataset):
        simulator = FactorySimulator(products=40, seed=7, dataset_dir=dataset)
        report = simulator.run()
        assert report["total_count"] == 40
        assert report["pass_count"] + report["reject_count"] + report["hold_count"] == 40
        stats = report["camera_stats"]
        assert stats["total_frames"] == 40
        assert stats["success_frames"] == 40 and stats["failed_frames"] == 0
        assert stats["triggers_issued"] == 40
        assert stats["average_capture_latency"] is not None
        assert stats["final_health"]["state"] == "ONLINE"
        # every event entered through the virtual camera
        events = simulator.store.events(limit=100)
        assert {row["camera_id"] for row in events} == {"steel-camera-01"}
        assert report["mes_orders"]["total"] == report["reject_count"]

    def test_camera_failure_run_fails_closed(self, dataset):
        simulator = FactorySimulator(products=20, seed=3, dataset_dir=dataset, camera_failure_rate=1.0)
        report = simulator.run()
        assert report["hold_count"] == 20
        assert report["pass_count"] == 0 and report["reject_count"] == 0
        stats = report["camera_stats"]
        assert stats["failed_frames"] == 20 and stats["success_frames"] == 0
        assert stats["final_health"]["state"] == "ERROR"
        events = simulator.store.events(limit=50)
        assert all(row["reason_code"] == "AI_SYSTEM_FAILURE" for row in events)
        assert all(row["plc_status"] == "ACK_STOP_SIGNAL" for row in events)

    def test_looping_replay_over_tiny_dataset(self, tmp_path):
        write_placeholder_png(tmp_path / "only.png", width=64, height=32)
        simulator = FactorySimulator(products=25, seed=5, dataset_dir=tmp_path)
        report = simulator.run()
        assert report["camera_stats"]["total_frames"] == 25

    def test_deterministic_across_instances(self, dataset):
        first = FactorySimulator(products=25, seed=11, dataset_dir=dataset).run()
        second = FactorySimulator(products=25, seed=11, dataset_dir=dataset).run()
        assert (first["pass_count"], first["reject_count"], first["hold_count"]) == (
            second["pass_count"], second["reject_count"], second["hold_count"],
        )
