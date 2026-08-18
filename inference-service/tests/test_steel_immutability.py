"""Artifact immutability regression tests for steel PatchCore.

1. threshold generation must NOT change bank.npz SHA256
2. evaluator must reject a threshold whose bank_sha256 mismatches the bank
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service/scripts"))
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.threshold_verify import load_and_verify_threshold  # noqa: E402

BANK = ROOT / "inference-service/models/steel-patchcore/bank.npz"
META = ROOT / "inference-service/models/steel-patchcore/bank_meta.json"
THRESH = ROOT / "model-training/datasets/severstal-steel/threshold.json"
SEALED_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"

pytestmark = pytest.mark.unit


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_bank_sha256_unchanged_since_seal():
    """The sealed bank artifact must still hash to the recorded SEALED value."""
    if not BANK.exists():
        pytest.skip("steel bank not present locally")
    assert _sha(BANK) == SEALED_SHA


def test_threshold_generation_does_not_change_bank_sha256():
    """threshold.json must reference the bank SHA256; a rewritten bank would break it."""
    if not (BANK.exists() and THRESH.exists()):
        pytest.skip("threshold.json / bank not present locally")
    cur = _sha(BANK)
    tdata = json.load(open(THRESH, encoding="utf-8"))
    assert tdata["bank_sha256"] == cur, "threshold references a different bank than on disk"
    assert tdata["scored_originals"] == tdata["expected_originals"] == 4721


def test_bank_meta_matches_threshold_reference():
    if not (BANK.exists() and META.exists() and THRESH.exists()):
        pytest.skip("bank artifacts not present locally")
    meta = json.load(open(META, encoding="utf-8"))
    tdata = json.load(open(THRESH, encoding="utf-8"))
    assert meta["bank_sha256"] == tdata["bank_sha256"] == _sha(BANK)


def test_evaluator_rejects_mismatched_bank_sha():
    """Evaluator must reject a threshold bound to a different bank."""
    bank_sha = _sha(BANK) if BANK.exists() else SEALED_SHA
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "threshold.json"
        bad.write_text(json.dumps({
            "model": "steel-patchcore", "version": "1.0.0",
            "threshold": 0.5, "bank_sha256": "0" * 64,
            "scored_originals": 4721, "expected_originals": 4721,
        }))
        with pytest.raises(AssertionError):
            load_and_verify_threshold(bad, bank_sha)


def test_evaluator_rejects_incomplete_calibration():
    """A threshold covering fewer than 4721 originals must be rejected."""
    bank_sha = _sha(BANK) if BANK.exists() else SEALED_SHA
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "threshold.json"
        bad.write_text(json.dumps({
            "model": "steel-patchcore", "version": "1.0.0",
            "threshold": 0.5, "bank_sha256": bank_sha,
            "scored_originals": 100, "expected_originals": 4721,
        }))
        with pytest.raises(AssertionError):
            load_and_verify_threshold(bad, bank_sha)
