"""Chunked bundle downloader v2 with strict resume semantics.

Strict Range semantics (user requirement):
  * Range: bytes=<local_size>- must return 206 with Content-Range start ==
    local_size to append. A 200 (Range ignored) means TRUNCATE and restart.
  * 401/403 -> refresh the signed URL via the official API, then resume from
    the current offset (never hammer the same dead URL 30x).
  * 429 -> Retry-After / backoff. 5xx / 10054 / timeout -> retry + resume.
    404 -> hard failure. 416 -> check local_size vs expected size.
  * COMPLETED requires ALL of: actual bytes == manifest zip_bytes,
    SHA256 == manifest sha256, ZIP integrity, expected image entries,
    duplicate entries == 0, JPEG sample decode OK.

Expected sizes / SHA256 / per-part image counts come from bundle_manifest.json
(the kernel-side manifest). State is persisted to chunk_download_state.json.

Usage:
  python model-training/steel_patchcore/download_chunks.py [--smoke] [--resume-test]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "datasets/severstal-steel/kernel-pack/chunk_out"
STATE_PATH = ROOT / "datasets/severstal-steel/kernel-pack/chunk_download_state.json"
MANIFEST = CHUNK_DIR / "bundle_manifest.json"

OWNER = "ten10do"
KERNEL = "ivqc-severstal-pack"
MAX_WORKERS = 2
MAX_ATTEMPTS_PER_URL = 3

state_lock = threading.Lock()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refresh_urls(api: KaggleApi, max_retries: int = 5) -> dict:
    """Fetch fresh signed URLs via the official API (with retries)."""
    last = None
    for attempt in range(max_retries):
        try:
            client = api.build_kaggle_client()
            req = ApiListKernelSessionOutputRequest()
            req.user_name = OWNER
            req.kernel_slug = KERNEL
            resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
            return {f.file_name: f.url for f in resp.files}
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"refresh_urls failed: {last}")


URLS_CACHE = CHUNK_DIR.parent / "chunk_urls.json"


def get_urls(api: KaggleApi, force: bool = False) -> dict:
    """Cached signed URLs; refresh on demand (401/403 or first run)."""
    if not force and URLS_CACHE.exists():
        cached = json.load(open(URLS_CACHE, encoding="utf-8"))
        if cached:
            return cached
    urls = refresh_urls(api)
    json.dump(urls, open(URLS_CACHE, "w"))
    return urls


def download_one(api: KaggleApi, name: str, target: Path,
                 expected_size: int, expected_sha: str, expected_entries: int) -> tuple[bool, str]:
    """Download one part with strict resume semantics."""
    urls = get_urls(api)
    if name not in urls:
        return False, "no signed url"
    url = urls[name]
    for cycle in range(MAX_ATTEMPTS_PER_URL):
        # refresh URL on 401/403; treat each cycle as a fresh signed URL
        local = target.stat().st_size if target.exists() else 0
        if expected_size and local >= expected_size:
            break
        headers = {"Range": f"bytes={local}-"} if local > 0 else {}
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=(60, 240))
            code = r.status_code
            if code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) + 1 if ra else 10.0 * (cycle + 1)
                time.sleep(min(wait, 120))
                r.close()
                continue
            if code in (401, 403):
                # signed URL expired: refresh and restart this cycle
                urls = get_urls(api, force=True)
                if name not in urls:
                    return False, "refresh failed"
                url = urls[name]
                r.close()
                continue
            if code == 404:
                r.close()
                return False, "404 hard failure"
            if code == 416:
                r.close()
                # range not satisfiable: either complete or corrupt
                if expected_size and target.stat().st_size == expected_size:
                    break
                return False, f"416 with local={target.stat().st_size} expected={expected_size}"
            if code == 200:
                # server ignored Range: MUST truncate and restart
                with open(target, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
                r.close()
                continue  # re-evaluate size at loop top
            if code == 206:
                cr = r.headers.get("Content-Range", "")
                m = re.match(r"bytes (\d+)-", cr)
                start = int(m.group(1)) if m else -1
                if start != local:
                    # unexpected offset: truncate and restart to be safe
                    with open(target, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            if chunk:
                                fh.write(chunk)
                    r.close()
                    continue
                with open(target, "ab") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
                r.close()
                continue
            # other 5xx etc.
            r.close()
            time.sleep(2 * (cycle + 1))
        except (requests.ConnectionError, requests.Timeout, OSError) as e:
            # resume from current offset on the next iteration
            time.sleep(2 * (cycle + 1))
            continue

    # ---- full validation (COMPLETED criteria) ----
    ok, msg = validate_chunk(target, expected_size, expected_sha, expected_entries)
    return ok, msg


def validate_chunk(target: Path, expected_size: int, expected_sha: str, expected_entries: int) -> tuple[bool, str]:
    if not target.exists():
        return False, "missing"
    if target.stat().st_size != expected_size:
        return False, f"size={target.stat().st_size} expected={expected_size}"
    sha = sha256_file(target)
    if sha != expected_sha:
        return False, f"sha mismatch"
    try:
        import zipfile

        with zipfile.ZipFile(target) as z:
            names = z.namelist()
            if len(names) != expected_entries:
                return False, f"entries={len(names)} expected={expected_entries}"
            if len(names) != len(set(names)):
                return False, "duplicate entries"
            # sample JPEG decode (first 3)
            for n in names[:3]:
                with Image.open(io.BytesIO(z.read(n))) as im:
                    im.load()
                    if im.size != (1600, 256):
                        return False, f"bad dims {im.size}"
    except Exception as e:  # noqa: BLE001
        return False, f"zip error: {type(e).__name__}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="only part-000.zip")
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    # fetch manifest first (small file) if not yet local
    if not MANIFEST.exists():
        urls = get_urls(api)
        if "bundle_manifest.json" in urls:
            r = requests.get(urls["bundle_manifest.json"], timeout=60)
            if r.status_code == 200 and len(r.content) > 0:
                CHUNK_DIR.mkdir(parents=True, exist_ok=True)
                MANIFEST.write_bytes(r.content)
                print("manifest fetched", len(r.content), "bytes", flush=True)
    if not MANIFEST.exists():
        print("MANIFEST_MISSING")
        return 2
    m = json.load(open(MANIFEST, encoding="utf-8"))
    parts = {p["filename"]: p for p in m.get("parts", [])}
    print(f"manifest: parts={len(parts)} total={m.get('total_images')}", flush=True)

    state = {}
    if STATE_PATH.exists():
        state = json.load(open(STATE_PATH, encoding="utf-8"))

    targets = sorted(parts.keys())
    if args.smoke:
        targets = [t for t in targets if t == "part-000.zip"]

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> tuple[str, str]:
        info = parts[name]
        target = CHUNK_DIR / name
        ok, msg = download_one(api, name, target, info["zip_bytes"], info["sha256"], info["image_count"])
        with state_lock:
            state[name] = {"status": "ok" if ok else "failed", "size": target.stat().st_size if target.exists() else 0,
                           "msg": msg}
            json.dump(state, open(STATE_PATH, "w"))
        print(f"{name}: {'OK' if ok else 'FAIL'} size={state[name]['size']} msg={msg}", flush=True)
        return name, "ok" if ok else "failed"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch, n): n for n in targets}
        for fut in as_completed(futs):
            fut.result()

    failed = [k for k, v in state.items() if v.get("status") != "ok"]
    print("summary:", json.dumps({k: v["status"] for k, v in state.items()}), flush=True)
    if failed:
        print("CHUNK_DOWNLOAD_FAILED:", failed, flush=True)
        return 1
    print("CHUNK_DOWNLOAD_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
