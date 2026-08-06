# Phase 8 Report: MLOps & Model Governance

Date: 2026-08-06
Commit: (filled at finalization)

## Scope

Model Registry, Versioning, Evaluation, Deployment Governance, Production
Monitoring, Drift Detection, Human Feedback Metrics, Retraining Candidate,
Rollback. No retraining was performed; existing Phase 1 / Phase 6 baselines
were registered with explicit versions.

## 8A. Model Identity

Every inference now answers: which model, which version, which weights,
which dataset, what metrics, what config.

- `model_registry` table: `model_name`, `model_version`, `model_type`
  (yolo | patchcore), `artifact_uri`, `artifact_sha256`, `dataset_version`,
  `training_run_id`, `created_at`, `status`, `metadata_json` (metrics),
  `domain_validated`, `promoted_at`.
- YOLO and PatchCore are versioned independently under their own model_name.

## 8B. MLflow

`mlflow==2.20.0` (local file store under `mlruns/`). `scripts/backfill_mlflow.py`
registers the historical baselines without retraining:

| run | params | metrics |
|---|---|---|
| phase1-yolo-baseline | dataset=NEU-DET, seed 42, imgsz 640, batch 16, epochs 100 | precision 0.81, recall 0.78, mAP50 0.82, mAP50-95 0.51, per-class mAP50, latency p95 21.5 ms, artifact sha256 |
| phase6-patchcore-baseline | backbone wide_resnet50_2, layers layer2,layer3, bank 50000, threshold 0.2027 | image AUROC 1.0, pixel AUROC 0.986, AUPRO 0.955, latency 755 ms, artifact sha256 |

Run ids recorded in `docs/phase8-mlflow-backfill.json`. Note: mlflow 3.15.1
had an upstream bug (`start_run` referencing an undefined `MlflowClient` and a
broken package install on this machine); 2.20.0 is used and verified.

## 8C. Model Registry

Statuses: CANDIDATE -> STAGING -> PRODUCTION -> ARCHIVED. At most one
PRODUCTION row per model_name is enforced by a partial unique index
(`uq_model_registry_active_production`). A deployment must come from the
registry; version-less `best.pt` startup is forbidden (8E).

## 8D. Deployment Manifest

`backend/config/deployment_manifest.yaml` pins the whole AI stack
(`vision_stack_version: 2026.08.1`, yolo + patchcore + fusion +
quality_rules with artifact sha256). Every inspection stamps
`inspections.deployment_version` at creation, so "which AI stack judged
this batch" is always answerable (verified in the E2E).

## 8E. Safe Model Loading

The inference service `/ready` now:
1. reads the deployment manifest
2. resolves artifacts and SHA256-validates them
3. loads YOLO + PatchCore and runs a smoke inference
4. returns `ready` only when all four steps pass

Any artifact missing / hash mismatch / load failure / smoke failure makes
`/ready` return `not_ready` with the problems list. Verified live:
`{"status":"ready","model_loaded":true,"anomaly_loaded":true,"deployment_version":"2026.08.1"}`.

## 8F. Promotion Gate

`backend/app/mlops/promotion_gate.py` (pure functions):

- YOLO: mAP50 >= 0.6, recall >= 0.6, latency_p95 <= 120 ms (configurable).
- PatchCore: image_auroc >= 0.9, pixel_auroc >= 0.9, latency <= 2000 ms.
- Domain: `domain_validated` must be true for the required domain.
  The MVTec bottle PatchCore has `steel_domain_validated=false`, so even
  with image_auroc=1.0 it can never be promoted to a steel production
  model (`test_patchcore_domain_mismatch_rejects_perfect_auroc`).

Promotion is NOT a manual click: the API runs the gate and refuses when it
fails (422 `promotion_gate_failed`).

## 8G. Production Monitoring

`GET /api/v1/model-metrics` aggregates over the inspection window:
inference_count, error_count/rate, latency avg + p95, confidence
distribution (10 bins), defect distribution, anomaly score distribution,
review_rate. Aggregated from PostgreSQL (documented trade-off: PG is not
used as a time-series DB; Prometheus/Grafana is the scale-up path).

## 8H. Human Feedback Metrics

`GET /api/v1/human-feedback` returns Phase 5 ground truth sliced by
model_version / defect_type / line / station / time window:
defect_confirmation_rate, ai_human_label_agreement_rate, pass_override_rate,
corrected_label_rate, plus a per-defect breakdown. This is closer to
production quality than training-set mAP (the E2E sees 18 resolved reviews).

## 8I. Drift Detection

`backend/app/mlops/drift.py` + `GET /api/v1/drift`: PSI on confidence and
anomaly score, KS on distributions, max-delta on defect distribution and
review rate, with a baseline vs current window, each signal classified
NORMAL / WARNING / CRITICAL. The API explicitly notes: data drift, not
quality degradation; quality degradation requires human-review ground
truth.

## 8J. Retraining Candidate

`GET /api/v1/training-candidates?kind=all|corrected|disagreed|low_confidence`
now emits a manifest with `dataset_version` (from the deployment stamp) on
top of image / ai label / human label / confidence / model version /
reason / anomaly score. No automatic retraining is triggered; every future
training run must reference an exact dataset_version.

## 8K. Dataset Versioning

`dataset_versions` table (manifest_uri + sha256) plus `dataset_version`
strings on registry rows (`neu-det-yolo-v1`, `mvtec-bottle-v1`). A model
can always be traced to an exact dataset version. DVC was not introduced:
manifest + SHA256 is sufficient at this scale (tool-for-tool's-sake
avoided).

## 8L. Rollback

`POST /api/v1/models/rollback` switches the production pointer to a prior
version and archives the current one; no rebuild. Verified in the real E2E:
promote v1 -> promote v2 (v1 archived) -> rollback v1 -> production is v1
again, and inspections before/after remain traceable to their own
deployment_version.

## 8M. Model Operations Dashboard

New "Model Operations" tab: Current Production Models, registry table with
status badges, Promote (gate-gated) and Rollback buttons, Production
Metrics, Drift signals with overall badge, Human Feedback with per-defect
breakdown. Screenshot: `docs/screenshots/12-phase8-modelops.png`.

## 8N. Fault Injection

- wrong sha256 -> `verify_deployment` / validate_artifacts reports mismatch
  (`test_manifest_artifact_wrong_hash_detected`)
- missing artifact -> reported missing (`test_missing_artifact_detected`)
- manifest missing section -> load raises (`test_manifest_missing_section_raises`)
- bad candidate (domain not validated) -> gate blocks + promote 422
- rollback of a non-registered version -> 404
- invalid models can never silently become production: every promotion
  path runs the gate; `/ready` fails on bad artifacts

## 8O. Tests & Real E2E

| Suite | result |
|---|---|
| backend pytest (unit) | 135 passed, 22 deselected |
| test_model_registry (gate/drift/manifest/registry) | 13 passed |
| test_mlops_api (registry API integration) | 6 passed |
| test_mlops_faults (fault injection) | 3 passed |
| frontend vitest | 33 passed |
| Playwright Browser E2E | modelops 1 + industrial 2 + review 8 + review-anomaly 2 = 13 passed |
| Phase 8 real E2E (scripts/mlops_e2e.py) | all green |

Real E2E chain (existing artifacts, no retraining): register baseline v1
-> gate -> promote PRODUCTION -> inspection with deployment_version
stamped -> model-metrics -> promote v2 (v1 archived) -> rollback v1 ->
production pointer switches -> drift + feedback respond. Recorded in
`docs/phase8-e2e.json`.

## Known issues

1. This session still runs CPU torch (PyPI default wheel; the cu128 wheel
   download is impractical here and the Bash environment requires the
   `env -i` wrapper `scripts/run_clean.sh` for torch commands). GPU
   coexistence numbers remain the Phase 6 canonical record.
2. mlflow 3.15.1 has an upstream `start_run` bug on this environment;
   2.20.0 is pinned and verified.
3. NEU domain has no natural AI-PASS samples (cross-domain PatchCore);
   the real RELEASE path is exercised via human PASS (Phase 7 scenario 3).
4. Monitoring is a PostgreSQL aggregation (not a time-series DB);
   Prometheus/Grafana is the documented scale-up path.
5. `deployment_version` stamps come from the pinned manifest; when the
   manifest changes, only new inspections get the new stamp (old rows keep
   their original stamp, which is the intended audit behaviour).

## Phase 9 recommendation

- Prometheus/Grafana production monitoring + alerting
- Real steel-domain PatchCore bank with steel_domain_validated=true
  (moves the MVTec baseline from benchmark to production claim)
- Automated retraining pipeline that consumes the retraining candidate
  manifest and emits a new registry CANDIDATE
- Registry promotion approval workflow (multi-role approval)
