# D3 Release Rollback Procedure

Rollback is a controlled candidate selection change. It is not production promotion and must not modify artifacts.

## Preconditions

- Identify the active candidate and the previous approved candidate.
- Resolve both manifest paths explicitly inside the project registry.
- Verify each manifest SHA-256 and every referenced artifact before activation.
- Stop new intake and drain or hold queued work.

## Procedure

1. Record operator, reason, timestamp, active version and target previous version.
2. Verify the target manifest and artifact hashes using the fail-closed registry loader.
3. If verification fails, do not change active state; keep the line in safe hold.
4. Atomically switch the candidate-selection state to the verified previous candidate.
5. Load the previous candidate and run its smoke test.
6. Resume only after monitoring and the decision layer report ready.
7. Retain the failed candidate package and evidence for investigation; do not overwrite it.

## Verified pair

- Active release candidate: `steel-patchcore-d3-candidate@1.3.0-candidate.1`.
- Previous candidate supported by the qualification drill: `steel-patchcore-d3-candidate@1.2.0-candidate.1`.

## Abort conditions

Abort and remain in safe hold on missing file, hash mismatch, model-load failure, smoke-score mismatch or state-write failure. Rollback must never fall through to an automatic upgrade or production promotion.
