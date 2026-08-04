"""Create deterministic PNG fixtures for every Mock Provider outcome."""

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT = Path(__file__).resolve().parents[1] / "sample-data"
LABELS = {0: "pass", 1: "medium", 2: "high", 3: "critical"}


def image_for_bucket(bucket: int) -> bytes:
    for red in range(256):
        image = Image.new("RGB", (640, 420), (red, 84, 108))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((120, 90, 520, 330), radius=28, fill=(225, 232, 226))
        draw.line((210, 175, 440, 250), fill=(120 + red // 3, 40, 45), width=8)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        content = buffer.getvalue()
        if hashlib.sha256(content).digest()[0] % 4 == bucket:
            return content
    raise RuntimeError(f"Could not generate bucket {bucket}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for bucket, label in LABELS.items():
        target = OUTPUT / f"mock-{label}.png"
        target.write_bytes(image_for_bucket(bucket))
        print(f"created {target.name}")


if __name__ == "__main__":
    main()
