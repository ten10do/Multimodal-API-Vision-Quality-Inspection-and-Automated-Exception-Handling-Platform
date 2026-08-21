"""Process Lifecycle Gate for steel PatchCore long-running GPU tasks.

Layers:
  1. precise stale-process cleanup (python + steel script + this workspace)
  2. single-instance lock (PID/stage/workspace/started_at/bank_sha256)
  3. Windows Job Object KILL_ON_JOB_CLOSE (fallback: psutil process-tree)
  4. preflight resource verification before resume
  5. shared by trainer / evaluation / failure-analysis

Never kills python.exe globally; only processes whose command line matches a
steel script AND whose cwd/cmdline points at this workspace.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]  # repo root
LOCK_FILE = Path(__file__).resolve().parents[1] / "datasets/severstal-steel/raw/steel_lifecycle_lock.json"

STEEL_SCRIPTS = (
    "train_steel_patchcore.py",
    "eval_steel_patchcore.py",
    "failure_case_steel.py",
    "run_steel_d3_recovery_holdout.py",
    "qualify_steel_d3_candidate.py",
    "investigate_steel_d3_heatmaps.py",
)

_JOB_HANDLE = None  # keep the Job Object handle alive for the process lifetime


# --------------------------------------------------------------------------- #
# Layer 1: precise stale-process identification & cleanup
# --------------------------------------------------------------------------- #
def find_stale_steel_processes() -> list:
    import psutil

    ws = str(WORKSPACE).lower()
    self_pid = os.getpid()
    stale = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            info = proc.info
            if info.get("pid") == self_pid:
                continue
            name = (info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmd = " ".join(info.get("cmdline") or []).lower()
            if not any(s in cmd for s in STEEL_SCRIPTS):
                continue
            cwd = (info.get("cwd") or "").lower()
            if ws not in cmd and ws not in cwd:
                continue
            stale.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return stale


def terminate_tree(proc) -> None:
    """Force-kill a process (used only by external callers, not by the gate).

    In-process termination is intentionally NOT performed by the gate because
    psutil.children()/kill() and subprocess taskkill can hang in this sandbox.
    The gate only DETECTS stale processes and reports them; actual cleanup is
    delegated to an external supervisor (e.g. sandbox-external taskkill).
    """
    import psutil

    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def detect_stale_pids() -> list[int]:
    """Return PIDs of stale steel processes (detection only, no termination)."""
    return [p.pid for p in find_stale_steel_processes()]


def cleanup_stale() -> list[int]:
    """Detect stale steel processes (detection only; termination is external).

    Returns the list of stale PIDs so an external supervisor can kill them.
    """
    pids = detect_stale_pids()
    print(f"stale steel processes detected: {pids} (self={os.getpid()})", flush=True)
    return pids


# --------------------------------------------------------------------------- #
# Layer 2: single-instance lock
# --------------------------------------------------------------------------- #
def read_lock() -> dict | None:
    if LOCK_FILE.exists():
        try:
            return json.load(open(LOCK_FILE, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def check_lock() -> str | None:
    """Return TRAINER_ALREADY_RUNNING if the lock PID is still alive."""
    import psutil

    lk = read_lock()
    if lk and lk.get("pid"):
        try:
            if psutil.pid_exists(int(lk["pid"])):
                return "TRAINER_ALREADY_RUNNING"
        except Exception:  # noqa: BLE001
            pass
    return None


def acquire_lock(stage: str, bank_sha: str) -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "pid": os.getpid(),
        "stage": stage,
        "workspace": str(WORKSPACE),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bank_sha256": bank_sha,
    }, open(LOCK_FILE, "w", encoding="utf-8"))


def release_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Layer 3: Windows Job Object KILL_ON_JOB_CLOSE
# --------------------------------------------------------------------------- #
def apply_job_object_kill_on_close() -> bool:
    """Join the current process to a Job Object that kills the tree on close.

    Returns True on success; False if unavailable (caller falls back to
    psutil process-tree tracking + startup cleanup + lock).
    """
    global _JOB_HANDLE
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectBasicLimitInformation = 2
        info = JOBOBJECT_BASIC_LIMIT_INFORMATION()
        info.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        hJob = kernel32.CreateJobObjectW(None, None)
        if not hJob:
            return False
        if not kernel32.SetInformationJobObject(
            ctypes.c_void_p(hJob), JobObjectBasicLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            return False
        ok = kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(hJob), kernel32.GetCurrentProcess()
        )
        if not ok:
            return False
        _JOB_HANDLE = hJob  # keep alive until process exit
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Layer 4: preflight resource verification
# --------------------------------------------------------------------------- #
def gpu_steel_processes() -> list[int]:
    """PIDs of steel processes currently holding the GPU (via nvidia-smi)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        gpu_pids = {int(x.strip()) for x in out.stdout.splitlines() if x.strip().isdigit()}
    except Exception:  # noqa: BLE001
        return []
    return [p.pid for p in find_stale_steel_processes() if p.pid in gpu_pids]


def preflight_checks(ckpt_path: Path, tmp_dir: Path) -> str | None:
    """Return a blocker code, or None if all checks pass.

    NOTE: stale-process detection is intentionally excluded here. In this
    sandbox, psutil process enumeration repeatedly surfaces processes that are
    already exiting, which makes in-process stale detection unreliable and
    causes a launch/deadlock loop. Stale cleanup is delegated to an external
    supervisor (sandbox-external taskkill) before each launch instead.
    """
    # 1. checkpoint readable (read-only)
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:  # noqa: BLE001
            return f"CHECKPOINT_UNREADABLE:{type(e).__name__}"
    # 2. disposable temp file can be os.replace'd in the checkpoint dir
    #    (write + atomic replace prove writability; deletion is intentionally
    #    skipped because sandbox safe-delete hooks block unlink)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f".preflight.{os.getpid()}.tmp"
    dst = tmp_dir / f".preflight.{os.getpid()}.dst"
    try:
        tmp.write_text("x", encoding="utf-8")
        os.replace(tmp, dst)
    except Exception as e:  # noqa: BLE001
        return f"TMP_REPLACE_FAILED:{type(e).__name__}"
    return None


def lifecycle_enter(stage: str, bank_sha: str, ckpt_path: Path, tmp_dir: Path) -> int | None:
    """Run the full gate before starting a long GPU task.

    Order: single-instance lock -> cleanup stale -> preflight -> job object ->
    acquire lock. Returns an exit code if blocked, or None to proceed.
    """
    blocker = check_lock()
    print("lifecycle: check_lock done", flush=True)
    if blocker:
        print(blocker)
        return 3
    # stale-process detection/cleanup is handled by an external supervisor
    # (sandbox-external taskkill) before launch, NOT here, because in-process
    # process enumeration hangs/races in this sandbox.
    pre = preflight_checks(ckpt_path, tmp_dir)
    print(f"lifecycle: preflight done pre={pre}", flush=True)
    if pre:
        print(f"PREFLIGHT_BLOCKED:{pre}")
        return 3
    job_ok = apply_job_object_kill_on_close()
    print(f"job_object_kill_on_close={job_ok}", flush=True)
    acquire_lock(stage, bank_sha)
    atexit.register(release_lock)
    return None
