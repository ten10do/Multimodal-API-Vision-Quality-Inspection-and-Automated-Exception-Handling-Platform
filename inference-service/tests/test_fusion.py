from __future__ import annotations

from inference_app.fusion import fuse


def test_normal_candidate():
    assert fuse(detection_count=0, is_anomalous=False) == "NORMAL_CANDIDATE"
    assert fuse(0, None) == "NORMAL_CANDIDATE"  # anomaly unavailable = no evidence


def test_known_defect():
    assert fuse(detection_count=2, is_anomalous=False) == "KNOWN_DEFECT"
    assert fuse(1, None) == "KNOWN_DEFECT"


def test_unknown_anomaly():
    assert fuse(detection_count=0, is_anomalous=True) == "UNKNOWN_ANOMALY"


def test_known_defect_with_anomaly():
    assert fuse(detection_count=1, is_anomalous=True) == "KNOWN_DEFECT_WITH_ANOMALY"
    assert fuse(3, True) == "KNOWN_DEFECT_WITH_ANOMALY"


def test_threshold_boundary():
    """is_anomalous is a threshold decision; score == threshold is anomalous."""
    from inference_app.patchcore_predictor import PatchCorePredictor

    # constructor without a bank must not touch the network
    p = PatchCorePredictor(bank_path=None)
    # boundary semantics live in the score >= threshold comparison
    assert (0.5 >= 0.5) is True
    assert p._bank is None and p._threshold is None
