# Simulated Site Acceptance Test Plan

This is a site-acceptance template backed by current repository evidence. It has not been run at a physical factory and cannot issue a production acceptance verdict. Site owners must approve their own criteria, execute the plan on qualified equipment, and sign the result.

## Evidence Boundary

### Implemented capability

Offline metrics, dual-branch invariance, runtime health, PLC/MES idempotency, fail-closed paths, monitoring, human feedback, and rollback are implemented and covered by committed tests/reports.

### Simulation

Virtual camera, HTTP/OPC UA simulators, measured-latency replay, accelerated stability/FAT workloads, fault injection, and disposable rollback drills provide pre-SAT evidence. They do not validate physical devices, plant networks, or real production timing.

### Future deployment

A real SAT must use the approved camera, lighting, edge computer, PLC, MES, network, product mix, operators, safety controls, and site acceptance dataset. Every criterion below requires a named owner and signed threshold before execution.

## SAT Entry Criteria

- Approved installation drawings, optical setup, trigger/I/O map, network flow matrix, protocol schema, and safety review.
- Frozen source revision, deployment configuration, model/artifact identities, threshold identity, and rollback target.
- Sealed site acceptance dataset and annotation/review procedure with no overlap into candidate development.
- Calibrated time sources and a trace plan from product/trigger through camera, inference, PLC/MES, review, and audit store.
- Incident, containment, rollback, and manual-inspection procedures rehearsed by the site team.

## AI Metric Tests

| Test | Implemented capability | Simulation/current evidence | Future deployment acceptance |
|---|---|---|---|
| Image AUROC | Sealed image-ranking evaluation with lineage checks | D3 image AUROC `0.817907171428` on the committed sealed recovery protocol | Recalculate on the sealed site dataset; compare with a pre-approved site gate and confidence interval |
| Pixel AUROC | Independent R-L3 localization evaluation | Pixel AUROC `0.924139` | Recalculate only where site pixel masks are reliable; document excluded/unlabeled regions |
| AUPRO | Region-overlap evaluation for localization | AUPRO `0.799398` | Recalculate with the approved region/mask protocol and site defect-size distribution |
| Operating point | Frozen threshold identity and confusion evidence | Current threshold is `0.8471092581748962`; existing reports explicitly separate AUROC from conservative threshold recall | Approve site false-positive, false-negative, HOLD, and review-load criteria without editing the released threshold |
| Branch isolation | Exact image-score invariance test | `0` D3 score mismatches after dual-branch integration | Reconfirm score/threshold identity on the deployed package |

Repository metrics are reference evidence, not transferable site SAT targets.

## System Metric Tests

### Latency and Throughput

| Check | Simulation/current evidence | Future deployment execution |
|---|---|---|
| Candidate measured profile | Dual-branch measured profile p50/p95/p99 `533.438/546.674/562.130 ms` over 240 samples | Measure capture-to-decision and capture-to-actuation distributions at site line rate |
| Accelerated workload | 4,800/4,800 completed, `0` failed; virtual 8-hour replay; E2E p95 `553.711 ms` | Run production-equivalent sustained load and wall-clock soak; include burst, queue, storage, and network contention |
| Timeout recovery | 8 injected and 8 recovered in FAT replay | Validate the approved timeout/retry/HOLD policy with real gateways and product containment |

Source: [FAT throughput evidence](../d3-fat-throughput-report.json).

### Stability

- **Implemented capability:** bounded runtime telemetry, readiness, error/latency/resource histories, drift state, and artifact identity.
- **Simulation:** accelerated 24-hour virtual-clock run completed 240 requests with `0` errors and `0` score-drift events; it is not a wall-clock production soak.
- **Future deployment:** execute a site-approved wall-clock endurance test covering thermal cycles, shifts, reconnects, storage pressure, log rotation, camera/PLC/MES restarts, clock behavior, and recovery.

Source: [24-hour stability simulation](../d3-24h-stability-report.json).

### PLC/MES Interaction

- **Implemented capability:** deterministic command ID, PASS/REVIEW_REQUIRED/FAIL mapping, PLC and MES acknowledgement, duplicate suppression, and safe HOLD for inference timeout.
- **Simulation:** four FAT records cover normal, unknown anomaly, confirmed defect, and inference timeout; `idempotency_verified=true`, `failure_safe_hold=true`, and `production_connection_used=false`.
- **Future deployment:** validate physical PLC nodes/registers, scan/timing, interlocks, reject distance, acknowledgement, reconnect, MES schema/authentication, offline queue, replay, and state reconciliation.

Source: [PLC/MES FAT evidence](../d3-fat-plc-mes-report.json).

## Safety Tests

### Fail Closed

Inject camera disconnect/invalid frame, inference timeout/error/non-finite output, missing or mismatched identity, PLC timeout/NACK/offline, MES failure, drift CRITICAL, and dependent-service outage.

- **Implemented capability:** uncertain AI or field state becomes HOLD/pending/error; only explicit valid evidence can release a product.
- **Simulation:** unit, integration, fault-injection, FAT, and drift reports cover these software behaviors.
- **Future deployment:** confirm actual PLC safe state, physical product containment, alarm/escalation, manual mode, recovery ownership, and safety-system independence.

### Rollback

- **Implemented capability:** select a previously verified identity, verify hashes, switch lifecycle state, run smoke/industrial checks, and retain the failed package.
- **Simulation:** the candidate rollback drill returned from 1.3.0-candidate.1 to the previous 1.2.0-candidate.1 package; an injected previous-hash mismatch was blocked with state unchanged. This was candidate-only and did not promote production.
- **Future deployment:** rehearse rollback on production-equivalent hardware with PLC/MES reconciliation, operator communication, held-product disposition, configuration/database compatibility, and signed recovery evidence.

Source: [rollback drill evidence](../d3-rollback-drill-report.json) and [rollback guide](../operations/rollback-guide.md).

## SAT Result Record

For every case, record test ID, requirement, environment, product/batch, source revision, model/artifact/threshold identity, device/protocol versions, steps, expected/actual result, raw evidence URI/hash, operator, timestamp, deviation, containment, and approval. The final verdict can only be issued by the designated site authority after all mandatory cases pass or have formally accepted deviations.
