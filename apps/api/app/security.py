import hashlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.config import Settings

ALLOWED_TYPES = {
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "image/webp": {".webp"},
}
PIL_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


async def validate_and_store_image(
    upload: UploadFile, settings: Settings
) -> tuple[bytes, str, str]:
    original_name = Path(upload.filename or "").name
    suffix = Path(original_name).suffix.lower()
    declared_type = (upload.content_type or "").lower()
    if declared_type not in ALLOWED_TYPES or suffix not in ALLOWED_TYPES[declared_type]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "unsupported_image_type", "message": "仅支持 JPG、PNG、WEBP 图片"},
        )

    content = await upload.read(settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_file", "message": "上传文件为空"},
        )
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail={"code": "file_too_large", "message": "图片超过允许的大小限制"},
        )
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
            detected_type = PIL_FORMATS.get(image.format or "")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_image", "message": "图片内容损坏或无法解码"},
        ) from exc
    if detected_type != declared_type:
        raise HTTPException(
            status_code=422,
            detail={"code": "mime_mismatch", "message": "文件内容与声明类型不一致"},
        )

    stored_name = f"{uuid4().hex}{suffix}"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target = settings.upload_dir / stored_name
    target.write_bytes(content)
    return content, stored_name, hashlib.sha256(content).hexdigest()
