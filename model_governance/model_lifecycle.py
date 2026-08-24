"""Fail-closed model lifecycle journal.

The manager governs references to immutable artifacts.  It never writes model
artifacts, changes thresholds, or performs training.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping


class ModelState(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATED = "VALIDATED"
    CANDIDATE = "CANDIDATE"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


class LifecycleError(RuntimeError):
    """Raised when a lifecycle operation cannot be proven safe."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelLifecycleManager:
    """Persist lifecycle transitions to an append-only JSON journal."""

    SCHEMA_VERSION = "industrial_model_history_v1"

    def __init__(
        self,
        history_path: str | Path,
        *,
        project_root: str | Path | None = None,
        required_metrics: tuple[str, ...] = ("image_auroc", "pixel_auroc", "aupro"),
    ) -> None:
        self.history_path = Path(history_path).resolve()
        self.project_root = Path(project_root).resolve() if project_root else self.history_path.parent
        self.required_metrics = required_metrics
        self._lock = threading.RLock()
        if not self.history_path.exists():
            self._write({"schema_version": self.SCHEMA_VERSION, "records": [], "rollback_status": None})
        self._load()

    def register(
        self,
        model_version: str,
        artifact_path: str | Path,
        artifact_hash: str,
        *,
        operator: str,
        metrics: Mapping[str, float] | None = None,
    ) -> dict:
        with self._lock:
            data = self._load()
            if self._latest(data, model_version) is not None:
                raise LifecycleError(f"MODEL_ALREADY_REGISTERED:{model_version}")
            resolved = self._resolve_artifact(artifact_path)
            self._verify_artifact(resolved, artifact_hash)
            return self._append(
                data,
                model_version=model_version,
                artifact_path=self._portable_path(resolved),
                artifact_hash=artifact_hash,
                metrics=dict(metrics or {}),
                state=ModelState.DEVELOPMENT,
                transition=f"NONE -> {ModelState.DEVELOPMENT.value}",
                operator=operator,
                approval_status="NOT_REVIEWED",
            )

    def validate(
        self,
        model_version: str,
        *,
        metrics: Mapping[str, float],
        operator: str,
    ) -> dict:
        with self._lock:
            data = self._load()
            current = self._require_state(data, model_version, {ModelState.DEVELOPMENT})
            checked_metrics = self._verify_metrics(metrics)
            self._verify_record_artifact(current)
            return self._append_from(
                data,
                current,
                state=ModelState.VALIDATED,
                transition=f"{ModelState.DEVELOPMENT.value} -> {ModelState.VALIDATED.value}",
                operator=operator,
                metrics=checked_metrics,
                approval_status="VALIDATION_PASSED",
            )

    def promote(self, model_version: str, *, operator: str) -> dict:
        with self._lock:
            data = self._load()
            current = self._require_state(data, model_version, {ModelState.VALIDATED, ModelState.CANDIDATE})
            self._verify_record_artifact(current)
            self._verify_metrics(current.get("metrics", {}))
            source = ModelState(current["state"])
            target = ModelState.CANDIDATE if source is ModelState.VALIDATED else ModelState.PRODUCTION
            if target is ModelState.PRODUCTION:
                for record in self._current_records(data).values():
                    if record["state"] == ModelState.PRODUCTION.value and record["model_version"] != model_version:
                        self._append_from(
                            data,
                            record,
                            state=ModelState.RETIRED,
                            transition=f"{ModelState.PRODUCTION.value} -> {ModelState.RETIRED.value}",
                            operator=operator,
                            approval_status=f"SUPERSEDED_BY:{model_version}",
                        )
            return self._append_from(
                data,
                current,
                state=target,
                transition=f"{source.value} -> {target.value}",
                operator=operator,
                approval_status="APPROVED" if target is ModelState.PRODUCTION else "CANDIDATE_APPROVED",
            )

    def rollback(
        self,
        failed_version: str,
        previous_version: str,
        *,
        operator: str,
        reason: str,
    ) -> dict:
        with self._lock:
            data = self._load()
            failed = self._require_state(data, failed_version, {ModelState.CANDIDATE, ModelState.PRODUCTION})
            previous = self._require_state(data, previous_version, {ModelState.PRODUCTION, ModelState.RETIRED})
            if failed_version == previous_version:
                raise LifecycleError("ROLLBACK_TARGET_EQUALS_FAILED_VERSION")
            self._verify_record_artifact(failed)
            self._verify_record_artifact(previous)
            self._verify_metrics(previous.get("metrics", {}))
            if failed["state"] != ModelState.RETIRED.value:
                self._append_from(
                    data,
                    failed,
                    state=ModelState.RETIRED,
                    transition=f"{failed['state']} -> {ModelState.RETIRED.value}",
                    operator=operator,
                    approval_status="ROLLBACK_SOURCE_FAILED",
                )
            if previous["state"] != ModelState.PRODUCTION.value:
                previous = self._append_from(
                    data,
                    previous,
                    state=ModelState.PRODUCTION,
                    transition=f"{previous['state']} -> {ModelState.PRODUCTION.value}",
                    operator=operator,
                    approval_status="ROLLBACK_APPROVED",
                )
            data["rollback_status"] = {
                "status": "COMPLETED",
                "failed_version": failed_version,
                "restored_version": previous_version,
                "artifact_hash": previous["artifact_hash"],
                "reason": reason,
                "operator": operator,
                "timestamp": _utc_now(),
            }
            self._write(data)
            return dict(data["rollback_status"])

    def retire(self, model_version: str, *, operator: str) -> dict:
        with self._lock:
            data = self._load()
            current = self._require_state(
                data,
                model_version,
                {ModelState.DEVELOPMENT, ModelState.VALIDATED, ModelState.CANDIDATE, ModelState.PRODUCTION},
            )
            return self._append_from(
                data,
                current,
                state=ModelState.RETIRED,
                transition=f"{current['state']} -> {ModelState.RETIRED.value}",
                operator=operator,
                approval_status="RETIRED",
            )

    def operations_snapshot(self) -> dict:
        with self._lock:
            data = self._load()
            current = self._current_records(data)
            production = next((r for r in current.values() if r["state"] == ModelState.PRODUCTION.value), None)
            candidate = next((r for r in current.values() if r["state"] == ModelState.CANDIDATE.value), None)
            selected = production or candidate
            return {
                "available": selected is not None,
                "current_model_version": selected["model_version"] if selected else None,
                "lifecycle_state": selected["state"] if selected else None,
                "artifact_hash": selected["artifact_hash"] if selected else None,
                "rollback_status": data.get("rollback_status"),
            }

    def model_snapshot(self) -> dict:
        with self._lock:
            data = self._load()
            rows = sorted(self._current_records(data).values(), key=lambda row: row["timestamp"], reverse=True)
            return {
                "available": bool(rows),
                "versions": [
                    {
                        "model_version": row["model_version"],
                        "state": row["state"],
                        "metrics": row.get("metrics", {}),
                        "approval_status": row["approval_status"],
                        "artifact_hash": row["artifact_hash"],
                        "timestamp": row["timestamp"],
                    }
                    for row in rows
                ],
                "history": list(data["records"]),
            }

    def _load(self) -> dict:
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleError("MODEL_HISTORY_UNREADABLE") from exc
        if data.get("schema_version") != self.SCHEMA_VERSION or not isinstance(data.get("records"), list):
            raise LifecycleError("MODEL_HISTORY_SCHEMA_MISMATCH")
        return data

    def _write(self, data: dict) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="model-history-", suffix=".json", dir=self.history_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            for attempt in range(3):
                try:
                    os.replace(temp_name, self.history_path)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _append(self, data: dict, **record: object) -> dict:
        row = {"timestamp": _utc_now(), **record}
        data["records"].append(row)
        self._write(data)
        return dict(row)

    def _append_from(self, data: dict, current: dict, **updates: object) -> dict:
        base = {
            "model_version": current["model_version"],
            "artifact_path": current["artifact_path"],
            "artifact_hash": current["artifact_hash"],
            "metrics": dict(current.get("metrics", {})),
        }
        base.update(updates)
        return self._append(data, **base)

    @staticmethod
    def _latest(data: dict, model_version: str) -> dict | None:
        return next((row for row in reversed(data["records"]) if row["model_version"] == model_version), None)

    def _current_records(self, data: dict) -> dict[str, dict]:
        current: dict[str, dict] = {}
        for row in data["records"]:
            current[row["model_version"]] = row
        return current

    def _require_state(self, data: dict, model_version: str, allowed: set[ModelState]) -> dict:
        row = self._latest(data, model_version)
        if row is None:
            raise LifecycleError(f"MODEL_NOT_REGISTERED:{model_version}")
        if ModelState(row["state"]) not in allowed:
            raise LifecycleError(f"INVALID_TRANSITION:{model_version}:{row['state']}")
        return row

    def _resolve_artifact(self, artifact_path: str | Path) -> Path:
        path = Path(artifact_path)
        return (self.project_root / path).resolve() if not path.is_absolute() else path.resolve()

    def _portable_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return str(path)

    def _verify_record_artifact(self, record: Mapping[str, object]) -> None:
        self._verify_artifact(self._resolve_artifact(str(record["artifact_path"])), str(record["artifact_hash"]))

    @staticmethod
    def _verify_artifact(path: Path, expected_hash: str) -> None:
        if not path.is_file():
            raise LifecycleError(f"ARTIFACT_MISSING:{path}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise LifecycleError(f"ARTIFACT_HASH_MISMATCH:{path}")

    def _verify_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        missing = [name for name in self.required_metrics if name not in metrics or metrics[name] is None]
        if missing:
            raise LifecycleError(f"METRIC_MISSING:{','.join(missing)}")
        return {str(name): float(value) for name, value in metrics.items()}
