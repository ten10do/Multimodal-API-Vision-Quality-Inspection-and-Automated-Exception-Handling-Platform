# Rollback Guide

Rollback selects a previously verified model identity. It does not edit or recreate artifacts.

## Preconditions

- identify failed and previous versions;
- record operator, reason, incident/change ID, and timestamp;
- stop new intake and hold queued products;
- verify both manifests and every target artifact hash;
- confirm required metrics and approval history are present.

## Procedure

1. Load governance history and select the previous approved version.
2. Recompute the target artifact SHA-256.
3. Abort on missing artifact, hash mismatch, missing metric, malformed history, or invalid state transition.
4. Mark the failed Candidate/Production version RETIRED.
5. Restore the verified previous version to PRODUCTION state.
6. Run smoke inference and the test-product industrial chain.
7. Resume only after runtime, decision, PLC/MES, monitoring, and review paths report ready.
8. Retain the failed package and evidence for investigation.

## Runnable drill

```powershell
.\.venv\Scripts\python.exe -m model_governance.rollback_simulation
```

The drill creates disposable simulated artifacts, promotes version 1.2.0, registers 1.3.0 as Candidate, injects an inference failure, restores 1.2.0, and verifies that the restored SHA-256 is identical. It does not access D3 artifacts.

## Abort state

Any verification or state-write failure leaves the line in HOLD. Rollback must never fall through to an automatic upgrade, retraining action, threshold edit, or manifest rewrite.

## Evidence

- [Release rollback procedure](../release/rollback-procedure.md)
- [Change management](../change-management.md)
- [`model_governance/model_lifecycle.py`](../../model_governance/model_lifecycle.py)
