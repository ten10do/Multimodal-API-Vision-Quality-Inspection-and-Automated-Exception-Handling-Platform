"""Threshold verification for steel PatchCore (no torch dependency).

Binds threshold.json to the immutable bank artifact. Any mismatch in
model/version/bank_sha256/calibration coverage is rejected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_NAME = "steel-patchcore"
MODEL_VERSION = "1.0.0"
EXPECTED_ORIGINALS = 4721


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_verify_threshold(tpath: Path, bank_sha: str) -> tuple[float, dict]:
    """Load threshold.json and verify it binds to the given immutable bank.

    Raises AssertionError on any mismatch. Never writes to bank.npz.
    """
    tdata = json.load(open(tpath, encoding="utf-8"))
    assert tdata.get("model") == MODEL_NAME, "threshold model mismatch"
    assert tdata.get("version") == MODEL_VERSION, "threshold version mismatch"
    assert tdata.get("bank_sha256") == bank_sha, "threshold bank_sha256 mismatch with current bank"
    assert tdata.get("scored_originals") == tdata.get("expected_originals") == EXPECTED_ORIGINALS, \
        "threshold calibration must cover all 4721 train_normal originals"
    return float(tdata["threshold"]), tdata
