# Technical Deep Dive

This document is a trace map for discussing the system from problem statement to operational evidence. Each claim points to implementation or qualification artifacts.

## 1. The engineering problem

The platform is not only an image classifier. It must acquire a traceable frame, produce model evidence, make a fail-closed decision, coordinate PLC/MES behavior, allow human resolution, monitor runtime and data drift, and preserve model identity through rollback.

Trace: [system architecture](../architecture/system-architecture.md) · [`industrial_loop/events.py`](../../industrial_loop/events.py)

## 2. Why the first anomaly model failed

The WideResNet PatchCore baseline achieved image AUROC `0.4817` on the formal steel split, with anomaly median below normal median and zero anomaly recall at the frozen threshold. Pixel metrics retained signal, proving that local evidence and image ranking were different problems.

Trace: [failure analysis](../steel-patchcore-failure-analysis.md) · [`test_steel_evaluation.py`](../../inference-service/tests/test_steel_evaluation.py)

## 3. How D3 recovered validity

DINOv2-B/14 improved frozen feature quality; train-normal ZCA then aligned the metric geometry with steel texture. The controlled adaptation ladder moved image AUROC from `0.6029` (D0 WRN) to `0.8208` (D3), followed by `0.8362` full-development confirmation and `0.8179071714` on the sealed recovery holdout.

Trace: [D3 adaptation](../ai/d3-domain-adaptation.md) · [D3 system design](../steel-patchcore-d3-system-design.md)

## 4. Why localization became a second branch

Post-processing H0–H5 could not recover D3 localization. R-L3 reached pixel AUROC `0.924139` and AUPRO `0.799398`, but its standalone image AUROC was insufficient. The design therefore preserves D3-ZCA A0 for image ranking and uses R-L3 only for spatial evidence. The integration gate requires zero D3 score mismatches.

Trace: [dual-branch decision](../decisions/why-dual-branch.md) · [`d3_dual_branch_predictor.py`](../../inference-service/inference_app/d3_dual_branch_predictor.py)

## 5. How model evidence becomes a safe industrial action

The decision engine validates finite score/threshold fields, model and artifact versions, frozen threshold identity, inference status, and optional guard-band state. A malformed or unavailable result becomes HOLD. PASS/REJECT/HOLD then map to explicit, idempotent PLC and MES behavior.

Trace: [`decision_service.py`](../../industrial_loop/decision_service.py) · [PLC/MES loop](../industrial/plc-mes-loop.md)

## 6. How failures remain traceable

Every product has one immutable event. PLC, MES, and operator states enrich copies rather than overwrite the original observation. Human review retains AI decision, evidence, reviewer, outcome, comment, and timestamp.

Trace: [`events.py`](../../industrial_loop/events.py) · [`human_review.py`](../../industrial_loop/human_review.py)

## 7. How edge and drift behavior avoid hidden model changes

The edge runtime manages services and telemetry around the frozen inference service. Drift monitoring consumes feature embeddings and emits NORMAL/WARNING/CRITICAL. WARNING is visibility only; CRITICAL routes the line to HOLD. Neither path can retrain, recalculate ZCA, rebuild the bank, or tune threshold.

Trace: [edge runtime](../industrial/edge-runtime.md) · [drift monitoring](../industrial/drift-monitoring.md)

## 8. How releases fail closed

The release loader verifies dependency lock, candidate manifest, qualification reports, and every artifact SHA-256. Lifecycle promotion rechecks artifact and required metrics. Rollback re-verifies the previous version and records operator, reason, timestamp, target, and restored hash.

Trace: [`d3_release_package.py`](../../model-training/steel_patchcore/d3_release_package.py) · [`model_lifecycle.py`](../../model_governance/model_lifecycle.py)

## 9. What is proven and what is not

Proven in the repository: frozen-dataset/holdout model gates, dual-branch isolation, simulator-backed industrial semantics, resource/drift behavior, rollback, production-readiness qualification, FAT, and default automated tests.

Not proven: physical camera/PLC/MES site acceptance, continuous production SLA, cross-site model generalization, real financial ROI, or production deployment authorization.

Trace: [maturity report](../industrial-platform-maturity-report.md) · [factory acceptance](../d3-factory-acceptance-report.md)

## Evidence index

| Claim | Code | Test / report |
|---|---|---|
| Artifact verification | `model-training/steel_patchcore/d3_release_package.py` | `test_steel_d3_release_package.py` |
| D3 image immutability | `inference_app/d3_dual_branch_predictor.py` | `dual-branch-evaluation-report.json` |
| Camera fail-close | `industrial_loop/camera/camera_trigger.py` | `test_camera_adapter.py` |
| PLC/MES semantics | `industrial_loop/plc_adapter.py`, `mes_service.py` | `test_industrial_loop_plc.py`, `test_industrial_loop_mes.py` |
| Drift CRITICAL → HOLD | `monitoring/drift/` | `test_edge_drift_e2e.py` |
| Governance rollback | `model_governance/model_lifecycle.py` | `test_industrial_delivery_governance.py` |
