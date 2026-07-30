import hashlib
from io import BytesIO

from httpx import AsyncClient
from PIL import Image

from app.enums import RiskLevel
from app.schemas import Defect, InspectionContext, VisionInspectionResult


def png_for_bucket(target: int) -> bytes:
    for red in range(256):
        buffer = BytesIO()
        Image.new("RGB", (16, 16), (red, 80, 120)).save(buffer, format="PNG")
        content = buffer.getvalue()
        if hashlib.sha256(content).digest()[0] % 4 == target:
            return content
    raise AssertionError("Unable to produce deterministic mock image")


def inspection_context() -> InspectionContext:
    return InspectionContext(
        product_code="AX-240",
        batch_code="B20260730",
        image_mime_type="image/png",
        quality_rules=["critical requires approval"],
    )


def vision_result(risk: RiskLevel | None) -> VisionInspectionResult:
    if risk is None:
        return VisionInspectionResult(
            is_defective=False,
            overall_confidence=0.96,
            defects=[],
            summary="No defect",
        )
    return VisionInspectionResult(
        is_defective=True,
        overall_confidence=0.9,
        defects=[
            Defect(
                defect_type=f"{risk.value}_defect",
                confidence=0.9,
                severity=risk,
                description="Synthetic test defect",
            )
        ],
        summary=f"{risk.value} defect",
    )


async def upload_image(
    client: AsyncClient,
    bucket: int,
    *,
    key: str,
    product_code: str = "AX-240",
    batch_code: str = "B20260730",
) -> object:
    return await client.post(
        "/api/v1/inspections",
        headers={"Idempotency-Key": key},
        data={"product_code": product_code, "batch_code": batch_code},
        files={"image": ("part.png", png_for_bucket(bucket), "image/png")},
    )
