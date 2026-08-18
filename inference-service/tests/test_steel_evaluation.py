"""Steel PatchCore formal-test aggregation and checkpoint-resume regressions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "inference-service/scripts/eval_steel_patchcore.py"
SPEC = importlib.util.spec_from_file_location("eval_steel_patchcore", EVAL_SCRIPT)
assert SPEC and SPEC.loader
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def _results(validation_score: float = 0.9) -> dict[str, dict]:
    return {
        "validation_normal": {
            "scores": [validation_score],
            "tn": 0,
            "fp": 1,
            "tp": 0,
            "fn": 0,
            "pixel_aucs": [1.0],
            "aupros": [1.0],
        },
        "test_normal": {
            "scores": [0.1],
            "tn": 1,
            "fp": 0,
            "tp": 0,
            "fn": 0,
            "pixel_aucs": [0.2],
            "aupros": [0.4],
        },
        "test_anomaly": {
            "scores": [0.8],
            "tn": 0,
            "fp": 0,
            "tp": 1,
            "fn": 0,
            "pixel_aucs": [0.8],
            "aupros": [0.6],
        },
    }


def test_formal_confusion_excludes_validation_false_positive():
    aggregate = EVAL.aggregate_evaluation(
        _results(), {"test_normal": 1, "test_anomaly": 1}
    )

    assert aggregate["operating_point"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert aggregate["validation_diagnostic"] == {"n": 1, "tn": 0, "fp": 1, "fpr": 1.0}


def test_formal_image_auroc_uses_test_splits_only():
    aggregate = EVAL.aggregate_evaluation(
        _results(validation_score=0.95), {"test_normal": 1, "test_anomaly": 1}
    )

    assert aggregate["image_auroc"] == 1.0


def test_f1_is_zero_when_no_anomaly_is_predicted():
    results = _results()
    results["test_anomaly"].update(tp=0, fn=1)

    aggregate = EVAL.aggregate_evaluation(
        results, {"test_normal": 1, "test_anomaly": 1}
    )

    assert aggregate["operating_point"]["precision"] != aggregate["operating_point"]["precision"]
    assert aggregate["operating_point"]["recall"] == 0.0
    assert aggregate["operating_point"]["f1"] == 0.0


def test_formal_sample_count_gate_requires_7257_samples():
    results = _results()
    results["test_normal"].update(scores=[0.1] * 591, tn=591)
    results["test_anomaly"].update(scores=[0.8] * 6666, tp=6666)

    aggregate = EVAL.aggregate_evaluation(results)
    assert aggregate["formal_counts"] == {
        "test_normal": 591,
        "test_anomaly": 6666,
        "total": 7257,
    }

    results["test_anomaly"].update(scores=[0.8] * 6665, tp=6665)
    with pytest.raises(RuntimeError, match="FORMAL_SAMPLE_COUNT_MISMATCH"):
        EVAL.aggregate_evaluation(results)


def test_formal_pixel_aggregation_uses_test_allowlist_only():
    aggregate = EVAL.aggregate_evaluation(
        _results(), {"test_normal": 1, "test_anomaly": 1}
    )

    assert aggregate["pixel_auroc_mean"] == pytest.approx(0.5)
    assert aggregate["aup_pro_mean"] == pytest.approx(0.5)


def test_checkpoint_resume_skips_completed_originals():
    assert EVAL.pending_original_ids(
        ["already_done", "remaining"], {"already_done": {"score": 0.1, "pred": 0}}
    ) == ["remaining"]


def test_csv_mask_id_is_normalized_to_split_id():
    assert EVAL.normalize_image_id("0002cc93b.jpg") == "0002cc93b"
    assert EVAL.normalize_image_id("0002cc93b") == "0002cc93b"


def test_pixel_backfill_targets_only_entries_missing_evidence():
    completed = {
        "missing": {"score": 0.8, "pred": 1},
        "complete": {"score": 0.9, "pred": 1, "pixel_auc": 0.7, "aupro": 0.6},
    }

    assert EVAL.pending_pixel_evidence_ids(["missing", "complete", "new"], completed) == ["missing"]
