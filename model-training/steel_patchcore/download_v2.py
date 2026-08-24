"""Adaptive Downloader v2 for Severstal train_images.

Reliability-first downloader with a GLOBAL rate limiter shared by all workers:

  * pacing: shared minimum request interval + jitter (no tight bursts)
  * 429 handling: prefer Retry-After header; else exponential backoff + full
    jitter; workers NEVER retry immediately on their own
  * circuit breaker: consecutive 429s -> concurrency 2 -> 1, global cooldown;
    only after a stable run does concurrency return to 2 (never 3/4/8)
  * persistent runtime state (download_state.json, gitignored): per-image
    status/attempts/final_size/last_http_status/last_error/next_retry_at
  * COMPLETED requires: file exists, size>0, JPEG decodes, dimensions
    (1600,256) match. An HTTP 200 is NOT completion evidence.
  * periodic checkpoint summaries only (no per-file spam, no secrets)
  * if 429 persists even at concurrency=1 + cooldown + pacing: save state,
    report KAGGLE_RATE_LIMIT_PAUSED and exit (never hammer the API)

Usage:
  python model-training/steel_patchcore/download_v2.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets/severstal-steel/raw"
IMG_DIR = RAW / "train_images"
LISTING = Path(tempfile.gettempdir()) / "kaggle_full.json"
STATE = RAW / "download_state.json"
QUARANTINE = RAW / "quarantine"

COMPETITION = "severstal-steel-defect-detection"
EXPECTED_DIM = (1600, 256)

CONFIG = {
    "workers_max": 2,                # never raised above 2 this round
    "workers_after_429": 1,
    "min_request_interval_s": 0.4,   # relaxed after observing only 5x 429 in ~280 downloads
    "jitter_s": 0.1,
    "base_wait_s": 2.0,
    "max_wait_s": 240.0,
    "max_attempts_per_file": 10,
    "cooldown_after_consecutive_429": 3,
    "global_cooldown_s": 120.0,
    "stable_requests_to_restore": 20,
    "checkpoint_every": 50,
    "rate_limit_pause_after_429s": 12,
}


class RateLimitPaused(Exception):
    pass


class RateLimiter:
    """Global controller: pacing, concurrency, cooldown, circuit breaker."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._max_active = CONFIG["workers_max"]
        self._active = 0
        self.consecutive_429 = 0
        self._stable = 0
        self._cooldown_until = 0.0
        self._last_request = 0.0

    def acquire(self) -> None:
        with self._cv:
            while self._active >= self._max_active:
                self._cv.wait()
            self._active += 1

    def release(self) -> None:
        with self._cv:
            self._active -= 1
            self._cv.notify_all()

    def pace(self) -> None:
        """Block until the shared pacing window is satisfied."""
        with self._cv:
            now = time.time()
            if now < self._cooldown_until:
                time.sleep(self._cooldown_until - now)
            elapsed = time.time() - self._last_request
            interval = CONFIG["min_request_interval_s"] + random.uniform(0, CONFIG["jitter_s"])
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_request = time.time()

    def on_success(self) -> None:
        with self._cv:
            self.consecutive_429 = 0
            self._stable += 1
            if self._max_active == CONFIG["workers_after_429"] and self._stable >= CONFIG["stable_requests_to_restore"]:
                self._max_active = CONFIG["workers_max"]
                self._cv.notify_all()

    def on_429(self, retry_after: float | None) -> float:
        with self._cv:
            self.consecutive_429 += 1
            self._stable = 0
            if self._max_active > CONFIG["workers_after_429"]:
                self._max_active = CONFIG["workers_after_429"]
            if self.consecutive_429 >= CONFIG["cooldown_after_consecutive_429"]:
                self._cooldown_until = time.time() + CONFIG["global_cooldown_s"]
            if retry_after:
                wait = retry_after + random.uniform(0, 1.0)
            else:
                wait = min(CONFIG["base_wait_s"] * (2 ** min(self.consecutive_429 - 1, 5)),
                           CONFIG["max_wait_s"])
                wait += random.uniform(0, wait)  # full jitter
            if self.consecutive_429 >= CONFIG["rate_limit_pause_after_429s"]:
                raise RateLimitPaused()
            return wait

    def concurrency(self) -> int:
        with self._cv:
            return self._max_active


def jpeg_valid(target: Path) -> bool:
    try:
        with Image.open(target) as im:
            im.load()
            return im.size == EXPECTED_DIM
    except Exception:  # noqa: BLE001
        return False


def load_state() -> dict:
    if STATE.exists():
        return json.load(open(STATE, encoding="utf-8"))
    return {}


def main() -> int:
    if not LISTING.exists():
        print("listing missing:", LISTING)
        return 2
    listing = json.load(open(LISTING, encoding="utf-8"))
    names = sorted(k for k in listing if k.startswith("train_images/"))
    print(f"train_images in listing: {len(names)}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)

    # rebuild state from disk: existing valid JPEGs are COMPLETED
    state = load_state()
    existing = list(IMG_DIR.glob("*.jpg"))
    for p in existing:
        if p.name not in state or state[p.name].get("status") != "completed":
            if jpeg_valid(p):
                state[p.name] = {"status": "completed", "attempts": 0, "final_size": p.stat().st_size}
            else:
                # invalid -> quarantine (never delete)
                try:
                    p.rename(QUARANTINE / p.name)
                    state[p.name] = {"status": "quarantined", "attempts": 0, "last_error": "invalid jpeg"}
                except OSError:
                    state[p.name] = {"status": "pending"}

    completed = {k for k, v in state.items() if v.get("status") == "completed"}
    todo = [n for n in names if n.split("/")[-1] not in completed]
    print(f"completed: {len(completed)}  to download: {len(todo)}")

    api = KaggleApi()
    api.authenticate()
    limiter = RateLimiter()
    counts = {"completed": len(completed), "pending": len(todo), "failed": 0,
              "retry_wait": 0, "http_200": 0, "http_429": 0}
    t0 = time.time()
    state_lock = threading.Lock()

    def update(name: str, **fields) -> None:
        with state_lock:
            entry = state.setdefault(name, {"attempts": 0})
            entry.update(fields)

    def checkpoint() -> None:
        elapsed = time.time() - t0
        req_min = (counts["http_200"] + counts["http_429"]) / (elapsed / 60.0) if elapsed > 0 else 0.0
        done_now = counts["completed"] - len(completed)
        eta = (len(todo) - done_now) / (done_now / (elapsed / 3600.0)) if done_now > 0 and elapsed > 0 else None
        summary = {
            "completed": counts["completed"], "pending": counts["pending"],
            "retry_wait": counts["retry_wait"], "failed": counts["failed"],
            "http_200": counts["http_200"], "http_429": counts["http_429"],
            "effective_req_per_min": round(req_min, 1),
            "elapsed_s": round(elapsed, 1),
            "estimated_remaining_h": round(eta, 2) if eta else None,
        }
        print(json.dumps(summary), flush=True)
        with state_lock:
            json.dump(state, open(STATE, "w"))

    def fetch(item: str) -> str:
        name = item
        fname = name.split("/")[-1]
        target = IMG_DIR / fname
        # fast path: an already-valid local copy never needs an API request
        if target.exists() and jpeg_valid(target):
            counts["completed"] += 1
            counts["pending"] -= 1
            update(fname, status="completed", attempts=0, final_size=target.stat().st_size,
                   last_http_status=None, last_error=None, next_retry_at=None)
            return "ok"
        for attempt in range(CONFIG["max_attempts_per_file"]):
            limiter.acquire()
            last_exc = None
            status = 0
            try:
                limiter.pace()
                try:
                    api.competition_download_file(COMPETITION, name, path=str(IMG_DIR), quiet=True)
                    status = 200
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    resp = getattr(e, "response", None)
                    status = resp.status_code if resp is not None else 0
            finally:
                limiter.release()

            if status == 200:
                # completion requires full validation, not just HTTP 200
                if jpeg_valid(target):
                    counts["http_200"] += 1
                    counts["completed"] += 1
                    counts["pending"] -= 1
                    update(fname, status="completed", attempts=attempt + 1, final_size=target.stat().st_size,
                           last_http_status=200, last_error=None, next_retry_at=None)
                    limiter.on_success()
                    return "ok"
                # invalid download: quarantine and retry later (not completed)
                try:
                    target.rename(QUARANTINE / fname)
                except OSError:
                    pass
                update(fname, status="retry_wait", attempts=attempt + 1,
                       last_http_status=200, last_error="invalid jpeg after 200")
            else:
                counts["http_429"] += 1
                update(fname, status="retry_wait", attempts=attempt + 1,
                       last_http_status=status, last_error=f"HTTP {status}")
                if status == 429:
                    counts["retry_wait"] += 1
                    retry_after = None
                    resp = getattr(last_exc, "response", None) if last_exc else None
                    if resp is not None:
                        h = resp.headers.get("Retry-After")
                        if h:
                            try:
                                retry_after = float(h)
                            except ValueError:
                                retry_after = None
                    wait = limiter.on_429(retry_after)  # may raise RateLimitPaused
                    update(fname, next_retry_at=time.time() + wait)
                    time.sleep(wait)
                    continue
                # non-429 error: short wait then retry
                time.sleep(min(2.0 * (attempt + 1), 30.0))
        counts["failed"] += 1
        counts["pending"] -= 1
        update(fname, status="failed", attempts=CONFIG["max_attempts_per_file"], last_error="max attempts")
        return "failed"

    from concurrent.futures import ThreadPoolExecutor, as_completed

    paused = False
    with ThreadPoolExecutor(max_workers=CONFIG["workers_max"]) as ex:
        futures = {ex.submit(fetch, n): n for n in todo}
        done_any = 0
        try:
            for fut in as_completed(futures):
                fut.result()
                done_any += 1
                if done_any % CONFIG["checkpoint_every"] == 0:
                    checkpoint()
        except RateLimitPaused:
            paused = True

    checkpoint()
    if paused:
        print(f"KAGGLE_RATE_LIMIT_PAUSED completed={counts['completed']}/12568 "
              f"remaining={counts['pending']} resume state preserved")
        return 3
    if counts["failed"]:
        print(f"DOWNLOAD_PARTIAL completed={counts['completed']} failed={counts['failed']}")
        return 1
    print(f"DOWNLOAD_COMPLETE completed={counts['completed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
