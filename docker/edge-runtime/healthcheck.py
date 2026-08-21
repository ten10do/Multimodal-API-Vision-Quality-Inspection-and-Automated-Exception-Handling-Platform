"""Docker HEALTHCHECK probe for the edge runtime container.

Talks to the local edge service over HTTP (stdlib only, no dependencies):
exit 0 when /health reports an overall healthy/degraded runtime, exit 1
otherwise (Docker restart policy takes over). Override the URL with
EDGE_HEALTHCHECK_URL for non-default ports.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/health"
ACCEPTED_STATES = {"RUNNING", "DEGRADED"}


def probe(url: str, timeout_s: float = 3.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any probe failure fails the check
        return False, f"probe_failed:{type(exc).__name__}"
    overall = str(body.get("overall", ""))
    state = str(body.get("state", ""))
    if overall in ("healthy", "degraded") and state in ACCEPTED_STATES:
        return True, f"ok:{state}:{overall}"
    return False, f"unhealthy:{state}:{overall}"


def main() -> int:
    url = os.environ.get("EDGE_HEALTHCHECK_URL", DEFAULT_URL)
    ok, detail = probe(url)
    print(detail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
