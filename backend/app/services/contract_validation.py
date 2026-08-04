from __future__ import annotations

from ..config import get_settings

_IMAGE_MAGIC: dict[str, bytes] = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
    "bmp": b"BM",
}


class InvalidImageError(Exception):
    pass


def validate_image_bytes(data: bytes) -> None:
    if not data:
        raise InvalidImageError("empty image payload")
    if len(data) > get_settings().max_upload_bytes:
        raise InvalidImageError("image exceeds max upload size")
    if not any(data.startswith(magic) for magic in _IMAGE_MAGIC.values()):
        raise InvalidImageError("unsupported or corrupted image (magic bytes mismatch)")
