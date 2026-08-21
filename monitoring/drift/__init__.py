"""Data drift monitoring for the industrial vision loop (peripheral layer).

Uses only DINO feature embeddings to detect distribution change at the camera;
it never modifies the model, weights, bank or threshold:

    camera frame -> DINO feature -> FeatureDriftCollector -> DriftDetector
        NORMAL            -> continue production
        WARNING           -> continue production + alert
        CRITICAL          -> HOLD (reason DATA_DISTRIBUTION_SHIFT)
"""
from monitoring.drift.collector import BaselineStats, FeatureDriftCollector
from monitoring.drift.detector import (
    DriftDetector,
    DriftReport,
    DriftState,
    DriftThresholds,
)
from monitoring.drift.metrics import (
    cosine_distribution_shift,
    embedding_mean_distance,
    psi_1d,
    psi_embedding,
    psi_embedding_from_stats,
)

__all__ = [
    "BaselineStats",
    "DriftDetector",
    "DriftReport",
    "DriftState",
    "DriftThresholds",
    "FeatureDriftCollector",
    "cosine_distribution_shift",
    "embedding_mean_distance",
    "psi_1d",
    "psi_embedding",
    "psi_embedding_from_stats",
]
