# D3 Human Review Workflow

Scope: `steel-patchcore-d3-candidate@1.3.0-candidate.1`, candidate-only.

## Prediction snapshot

Every review item freezes the following values at inference time:

- `image_id`
- `image_score` from the immutable D3-ZCA A0 image branch
- `anomaly_label` derived from the frozen threshold `0.8471092581748962`
- `heatmap` from the independent R-L3 localization branch; persisted review systems store a content-addressed reference
- `confidence`, explicitly identified as an uncalibrated absolute threshold-margin ratio, never a probability
- `model_version` and `artifact_version`

The prediction snapshot is immutable. A human decision is appended as a separate audit record and does not overwrite the AI output.

## Review actions

- `human_confirmation`: the operator confirms the model-level decision.
- `false_positive`: valid only when the frozen prediction label is `ANOMALY` and the operator confirms clean steel.
- `false_negative`: valid only when the frozen prediction label is `NORMAL` and the operator identifies a missed anomaly.

Every feedback record includes reviewer, reason, timestamp, the complete prediction snapshot, and explicit `automatic_retraining=false` and `automatic_threshold_change=false` flags.

## Existing backend integration

The existing review queue provides claim ownership, immutable resolution, audit corrections, and human-feedback metrics. D3 mappings are:

- Confirmed anomaly → `CONFIRM_DEFECT` with a human label.
- False positive → `PASS` with a reason.
- False negative → create a manual review incident, then resolve as `OTHER_DEFECT` or `CORRECT_DEFECT` with a human label. It must not be silently injected into a completed PASS inspection.

The existing `/api/v1/human-feedback` endpoint can aggregate resolved records by model version and operational slice. Training-candidate export remains manual evidence only; this workflow never triggers training, threshold changes, candidate promotion, or deployment.
