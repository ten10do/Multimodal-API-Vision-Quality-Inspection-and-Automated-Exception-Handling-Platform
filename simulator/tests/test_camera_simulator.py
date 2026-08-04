from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from simulator.camera_simulator import CameraSimulator, SimulatorState  # noqa: E402
from simulator.config import SimulatorConfig  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def make_images(tmp_path: Path, count: int = 5) -> Path:
    for i in range(count):
        (tmp_path / f"img_{i:02d}.jpg").write_bytes(b"\xff\xd8\xff" + bytes([i % 256]))
    return tmp_path


@pytest.mark.asyncio
async def test_start_produces_unique_captures(tmp_path):
    src = make_images(tmp_path, 5)
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), q)
    sim.start()
    captures = []
    for _ in range(5):
        try:
            captures.append(await asyncio.wait_for(q.get(), timeout=3))
        except asyncio.TimeoutError:
            break
    await sim.stop()

    assert len(captures) == 5
    product_ids = [c.product_id for c in captures]
    capture_ids = [c.capture_id for c in captures]
    assert len(set(product_ids)) == 5, "product_id must be unique"
    assert len(set(capture_ids)) == 5, "capture_id must be unique"
    assert all(c.timestamp for c in captures)
    assert all(c.image_bytes for c in captures)


@pytest.mark.asyncio
async def test_pause_resume(tmp_path):
    src = make_images(tmp_path, 100)
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), q)
    sim.start()
    await asyncio.wait_for(q.get(), timeout=3)
    sim.pause()
    assert sim.state == SimulatorState.PAUSED
    before = sim.captured_count
    await asyncio.sleep(0.15)
    assert sim.captured_count == before, "no captures while paused"
    sim.resume()
    assert sim.state == SimulatorState.RUNNING
    await asyncio.wait_for(q.get(), timeout=3)
    assert sim.captured_count > before
    await sim.stop()


@pytest.mark.asyncio
async def test_stop_halts(tmp_path):
    src = make_images(tmp_path, 1000)
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=True), q)
    sim.start()
    await asyncio.wait_for(q.get(), timeout=3)
    await sim.stop()
    assert sim.state == SimulatorState.STOPPED
    before = sim.captured_count
    await asyncio.sleep(0.1)
    assert sim.captured_count == before


@pytest.mark.asyncio
async def test_loop_false_exhausts_source(tmp_path):
    src = make_images(tmp_path, 3)
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), q)
    sim.start()
    for _ in range(3):
        await asyncio.wait_for(q.get(), timeout=3)
    await asyncio.sleep(0.2)
    assert sim.state == SimulatorState.STOPPED


@pytest.mark.asyncio
async def test_product_id_unique_across_loop_cycles(tmp_path):
    src = make_images(tmp_path, 3)
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=True), q)
    sim.start()
    ids = []
    for _ in range(12):
        try:
            c = await asyncio.wait_for(q.get(), timeout=3)
            ids.append(c.product_id)
        except asyncio.TimeoutError:
            break
    await sim.stop()
    assert len(ids) >= 6, "should have cycled at least twice"
    assert len(set(ids)) == len(ids), "product_id must stay unique across loop cycles"


def test_missing_source_dir_raises():
    q: asyncio.Queue = asyncio.Queue(maxsize=4)
    sim = CameraSimulator(SimulatorConfig(source_directory="/no/such/dir"), q)
    with pytest.raises(FileNotFoundError):
        sim._collect_images()


@pytest.mark.asyncio
async def test_backpressure_blocks_producer(tmp_path):
    """queue full (block policy): producer blocks at capacity, depth never
    exceeds maxsize, and no capture is dropped (captured == source count)."""
    src = make_images(tmp_path, 3)
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), q)
    sim.start()
    await asyncio.sleep(0.25)
    # blocked at capacity: queue holds maxsize; the 3rd capture's seq is
    # assigned but its put() is blocked (backpressure), nothing dropped
    assert q.qsize() == 2
    assert sim.captured_count == 3
    # drain one -> blocked put completes, queue back to capacity
    await asyncio.wait_for(q.get(), timeout=2)
    await asyncio.sleep(0.25)
    assert q.qsize() == 2
    assert sim.captured_count == 3
    # drain the rest -> source exhausted, stops cleanly, nothing dropped
    await asyncio.wait_for(q.get(), timeout=2)
    await asyncio.wait_for(q.get(), timeout=2)
    await asyncio.sleep(0.25)
    assert sim.state == SimulatorState.STOPPED
    assert sim.captured_count == 3
    assert q.empty()
