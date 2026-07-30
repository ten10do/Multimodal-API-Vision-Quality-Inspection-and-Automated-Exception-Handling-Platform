from httpx import AsyncClient

from tests.factories import png_for_bucket


async def post_file(
    client: AsyncClient,
    *,
    filename: str,
    content_type: str,
    content: bytes,
    key: str,
) -> object:
    return await client.post(
        "/api/v1/inspections",
        headers={"Idempotency-Key": key},
        data={"product_code": "AX-240", "batch_code": "SECURITY"},
        files={"image": (filename, content, content_type)},
    )


async def test_rejects_unsupported_extension(client: AsyncClient) -> None:
    response = await post_file(
        client,
        filename="part.gif",
        content_type="image/png",
        content=png_for_bucket(0),
        key="extension-check",
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_image_type"


async def test_rejects_unsupported_declared_mime(client: AsyncClient) -> None:
    response = await post_file(
        client,
        filename="part.png",
        content_type="image/gif",
        content=png_for_bucket(0),
        key="mime-type-check",
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_image_type"


async def test_rejects_mime_that_does_not_match_real_content(client: AsyncClient) -> None:
    response = await post_file(
        client,
        filename="part.jpg",
        content_type="image/jpeg",
        content=png_for_bucket(0),
        key="mime-content-mismatch",
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "mime_mismatch"


async def test_rejects_damaged_image(client: AsyncClient) -> None:
    response = await post_file(
        client,
        filename="part.png",
        content_type="image/png",
        content=b"\x89PNG\r\n\x1a\ncorrupted",
        key="damaged-image-check",
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


async def test_rejects_oversized_image_before_decoding(client: AsyncClient) -> None:
    response = await post_file(
        client,
        filename="part.png",
        content_type="image/png",
        content=b"x" * (10 * 1024 * 1024 + 1),
        key="oversized-image-check",
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


async def test_rejects_empty_upload(client: AsyncClient) -> None:
    response = await post_file(
        client,
        filename="part.png",
        content_type="image/png",
        content=b"",
        key="empty-image-check",
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "empty_file"


async def test_path_like_filename_is_sanitized(client: AsyncClient) -> None:
    response = await post_file(
        client,
        filename="../../part.png",
        content_type="image/png",
        content=png_for_bucket(0),
        key="path-sanitize-check",
    )
    assert response.status_code == 201
    assert response.json()["original_filename"] == "part.png"
