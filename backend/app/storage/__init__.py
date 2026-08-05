"""Storage abstraction for inspection images (Phase 4D).

The backend saves the raw uploaded image when the inspection is created and
serves it back to the browser so the dashboard can draw Bounding Boxes from the
Vision Contract. The local filesystem provider is the default; the interface
is kept small so a MinIO (S3) provider can be dropped in later without touching
the inspection flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..config import get_settings


class StorageService(Protocol):
    def save(self, inspection_id: str, data: bytes) -> Path:
        """Persist raw image bytes for an inspection. Returns the stored path."""

    def save_anomaly_map(self, inspection_id: str, data: bytes) -> Path:
        """Persist the anomaly heatmap PNG for an inspection (Phase 6)."""

    def load(self, inspection_id: str) -> bytes | None:
        """Return stored bytes or None when the image is missing."""

    def load_anomaly_map(self, inspection_id: str) -> bytes | None:
        """Return the anomaly heatmap PNG bytes or None."""

    def path_for(self, inspection_id: str) -> Path | None:
        """Return the stored path or None when missing."""


class LocalStorageProvider:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, inspection_id: str) -> Path:
        return self.root / f"{inspection_id}.jpg"

    def _anomaly_path(self, inspection_id: str) -> Path:
        return self.root / f"{inspection_id}-anomaly.png"

    def save(self, inspection_id: str, data: bytes) -> Path:
        path = self._path(inspection_id)
        path.write_bytes(data)
        return path

    def save_anomaly_map(self, inspection_id: str, data: bytes) -> Path:
        path = self._anomaly_path(inspection_id)
        path.write_bytes(data)
        return path

    def load(self, inspection_id: str) -> bytes | None:
        path = self._path(inspection_id)
        if path.exists():
            return path.read_bytes()
        return None

    def load_anomaly_map(self, inspection_id: str) -> bytes | None:
        path = self._anomaly_path(inspection_id)
        if path.exists():
            return path.read_bytes()
        return None

    def path_for(self, inspection_id: str) -> Path | None:
        path = self._path(inspection_id)
        return path if path.exists() else None


_storage: StorageService | None = None


def get_storage() -> StorageService:
    global _storage
    if _storage is None:
        root = Path(get_settings().storage_dir)
        _storage = LocalStorageProvider(root)
    return _storage


def reset_storage() -> None:
    global _storage
    _storage = None
