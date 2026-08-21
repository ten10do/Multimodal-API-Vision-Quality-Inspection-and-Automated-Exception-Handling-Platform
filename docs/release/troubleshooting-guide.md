# D3 Release Troubleshooting Guide

| Symptom | Required action | Forbidden shortcut |
|---|---|---|
| Artifact hash mismatch | Block load; restore the exact approved file or roll back | Editing the manifest hash |
| Artifact missing | Block load; verify mounted artifact paths | Generating a replacement bank |
| Model load failure | Mark runtime unavailable and hold requests | Falling back to an unverified model |
| Inference timeout | Record failure and issue `HOLD` | Defaulting to `PASS/RELEASE` |
| Unexpected image dimensions | Reject at gateway | Implicit resizing outside protocol |
| Non-finite score/heatmap | Fail the request closed | Clamping silently |
| GPU memory growth | Stop intake, collect monitoring evidence, restart or roll back | Tuning the model online |
| Score distribution shift | Raise warning and increase human review | Automatic threshold tuning |
| Heatmap looks inconsistent | Confirm R-L3 hashes and input protocol; compare reproducible smoke output | Replacing the image score |
| PLC/MES unavailable | Hold and retry with the same idempotency key | Issuing duplicate physical actions |

## Triage order

1. Preserve the trace and failure reason.
2. Verify release, candidate and artifact hashes.
3. Verify input dimensions and file readability.
4. Verify GPU/device state and model-load logs.
5. Reproduce with the sealed smoke request.
6. If integrity or reproducibility cannot be restored, execute rollback.

Escalate any secret exposure, unexpected writable artifact, hash mismatch or release-state mutation as a blocking security incident.
