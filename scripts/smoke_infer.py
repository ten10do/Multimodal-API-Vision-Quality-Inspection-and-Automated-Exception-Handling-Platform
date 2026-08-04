import json
import urllib.request

boundary = "----ivqc"
img = open(
    "model-training/datasets/neu-det-yolo/test/images/crazing_101.jpg", "rb"
).read()
head = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="file"; filename="crazing_101.jpg"\r\n'
    "Content-Type: image/jpeg\r\n\r\n"
).encode()
body = head + img + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(
    "http://127.0.0.1:8100/v1/infer",
    data=body,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "X-Request-ID": "test-req-1",
    },
)
r = urllib.request.urlopen(req, timeout=60)
out = json.loads(r.read().decode())
print("status:", r.status)
print(
    {
        k: out[k]
        for k in [
            "inspection_id",
            "model_name",
            "model_version",
            "device",
            "image_width",
            "image_height",
            "inference_latency_ms",
        ]
    }
)
print("detections:", len(out["detections"]))
print("first detection keys:", list(out["detections"][0].keys()) if out["detections"] else "none")
