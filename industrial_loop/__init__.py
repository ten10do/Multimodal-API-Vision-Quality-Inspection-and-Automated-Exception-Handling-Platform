"""Industrial closed-loop simulation layer (peripheral to the frozen D3 stack).

This package adds the factory control loop around D3 inference:

    Camera Simulation -> Inference (D3, unchanged) -> Decision Engine
        -> OPC UA PLC simulator / MES work orders / Human review -> Dashboard

Hard constraint: it never modifies the D3 model, weights, bank, threshold,
feature extractor or any existing evaluation result. The decision engine only
READS the frozen release lineage and fails closed on any anomaly.
"""
from industrial_loop.config import (
    FROZEN_THRESHOLD,
    MODEL_VERSION,
    RELEASE_ID,
    RUNTIME_ROOT,
    DecisionPolicy,
)

__all__ = [
    "DecisionPolicy",
    "FROZEN_THRESHOLD",
    "MODEL_VERSION",
    "RELEASE_ID",
    "RUNTIME_ROOT",
]
