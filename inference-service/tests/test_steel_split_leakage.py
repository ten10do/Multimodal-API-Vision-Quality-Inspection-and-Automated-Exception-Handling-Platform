"""Split / leakage assertions for the steel dataset (needs local data files)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "model-training/datasets/severstal-steel"
SPLIT = DS / "split_manifest.json"
LEAK = DS / "leakage_report.json"
CSV = DS / "raw/train.csv"

pytestmark = pytest.mark.unit


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not present (dataset not prepared locally)")
    return json.load(open(path, encoding="utf-8"))


def test_split_all_disjoint():
    sm = _load(SPLIT)
    sets = {k: set(v) for k, v in sm["splits"].items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (sets[names[i]] & sets[names[j]]), f"{names[i]} overlaps {names[j]}"


def test_split_expected_four_groups():
    sm = _load(SPLIT)
    for k in ("train_normal", "validation_normal", "test_normal", "test_anomaly"):
        assert k in sm["splits"]
        assert len(sm["splits"][k]) > 0


def test_train_normal_has_no_anomaly_csv_rows():
    sm = _load(SPLIT)
    if not CSV.exists():
        pytest.skip("train.csv not present")
    import pandas as pd

    df = pd.read_csv(CSV)
    anomaly = set(df["ImageId"].astype(str).unique())
    train = set(sm["splits"]["train_normal"])
    assert not (train & anomaly), "train_normal contains anomaly images"


def test_leakage_gate_passed():
    rep = _load(LEAK)
    assert rep.get("gate_passed") is True


def test_bank_not_containing_test():
    sm = _load(SPLIT)
    bank = DS / "bank_source.json"
    if not bank.exists():
        pytest.skip("bank_source.json not present")
    bank_ids = set(json.load(open(bank, encoding="utf-8")))
    test_ids = set(sm["splits"]["test_normal"]) | set(sm["splits"]["test_anomaly"])
    assert not (bank_ids & test_ids), "memory bank sources leak into test"
