"""Camera simulator: emulates a line camera continuously capturing images.

The simulator only produces Capture objects (image + identity metadata). It
never calls the model, never touches PostgreSQL and never evaluates quality.
It is a pure capture source and can be replaced by a real camera / edge
capture service without touching backend business logic.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .config import SimulatorConfig


class SimulatorState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass(frozen=True)
class Capture:
    product_id: str
    capture_id: str
    timestamp: str
    image_bytes: bytes
    filename: str
    production_line: str
    station: str
    batch_id: str


class CameraSimulator:
    def __init__(self, config: SimulatorConfig, outbound_queue: asyncio.Queue) -> None:
        self.config = config
        self._queue = outbound_queue
        self._state = SimulatorState.STOPPED
        self._task: asyncio.Task | None = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._seq = 0
        self._stop_requested = False
        self.max_captures: int | None = None

    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    @property
    def state(self) -> SimulatorState:
        return self._state

    @property
    def captured_count(self) -> int:
        return self._seq

    def _collect_images(self) -> list[Path]:
        root = Path(self.config.source_directory)
        if not root.is_dir():
            raise FileNotFoundError(f"source directory not found: {root}")
        images = sorted(
            p for p in root.iterdir()
            if p.suffix.lower() in self.config.image_extensions and p.is_file()
        )
        if not images:
            raise FileNotFoundError(f"no images with {self.config.image_extensions} in {root}")
        if self.config.shuffle:
            rng = random.Random(self.config.random_seed)
            rng.shuffle(images)
        return images

    def start(self) -> None:
        if self._state in (SimulatorState.RUNNING, SimulatorState.PAUSED):
            return
        self._stop_requested = False
        self._state = SimulatorState.RUNNING
        self._task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        self._stop_requested = True
        if self._state in (SimulatorState.RUNNING, SimulatorState.PAUSED):
            self._pause_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._state = SimulatorState.STOPPED

    def pause(self) -> None:
        if self._state == SimulatorState.RUNNING:
            self._state = SimulatorState.PAUSED
            self._pause_event.clear()

    def resume(self) -> None:
        if self._state == SimulatorState.PAUSED:
            self._state = SimulatorState.RUNNING
            self._pause_event.set()

    async def _capture_loop(self) -> None:
        images = self._collect_images()
        interval = self.config.interval_ms / 1000.0
        try:
            while not self._stop_requested:
                for image in images:
                    if self.max_captures is not None and self._seq >= self.max_captures:
                        return
                    await self._pause_event.wait()
                    if self._stop_requested:
                        return
                    self._seq += 1
                    # product_id stays unique across loops because _seq never resets
                    product_id = f"{self.config.production_line}-{self.config.batch_id}-{self._seq:06d}"
                    capture = Capture(
                        product_id=product_id,
                        capture_id=f"cap-{uuid.uuid4().hex[:12]}",
                        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                        image_bytes=image.read_bytes(),
                        filename=image.name,
                        production_line=self.config.production_line,
                        station=self.config.station,
                        batch_id=self.config.batch_id,
                    )
                    await self._queue.put(capture)  # block when the queue is full (backpressure)
                    await asyncio.sleep(interval)
                if not self.config.loop or self._stop_requested:
                    return
        finally:
            # give consumers a chance to drain before flipping to stopped,
            # but never block inside stop()/cancellation
            if not self._stop_requested:
                while not self._queue.empty():
                    await asyncio.sleep(0.05)
            self._state = SimulatorState.STOPPED
