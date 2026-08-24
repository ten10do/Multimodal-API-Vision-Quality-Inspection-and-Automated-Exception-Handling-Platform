# Production Maintenance Plan

This cadence is a deployment template. The repository implements health, monitoring, traceability, artifact verification, review, and rollback capabilities, but it does not prove that daily, weekly, monthly, or quarterly maintenance has been performed at a factory.

## Maintenance Boundary

### Implemented capability

- Runtime health/readiness/status, resource and latency telemetry, bounded histories, reason-coded errors, and drift state.
- Traceable camera/inference/decision/PLC/MES/review evidence.
- SHA-256 artifact verification, read-only artifact loading, lifecycle history, rollback validation, and regression tests.

### Simulation

Virtual camera, PLC/MES simulators, accelerated stability/FAT workloads, fault injection, drift scenarios, and disposable rollback drills demonstrate the maintenance interfaces. They do not exercise physical cleaning, calibration, cabling, environmental degradation, vendor firmware, or plant support processes.

### Future deployment

The plant must assign owners, shifts, alarm routes, maintenance windows, service-level objectives, retention, spare parts, calibration tools, evidence storage, and escalation. The proposed cadence must be adapted to process risk and vendor requirements.

## Daily / Start of Shift

| Activity | Repository capability | Site action required |
|---|---|---|
| Runtime readiness | `/health`, `/ready`, `/status`, service state and last error | Confirm required services are healthy; stop at the first failed gate |
| Camera path | camera health, trigger/frame identity and error semantics | Inspect lens/window/lighting, power/link, focus/exposure, trigger and a test product |
| Model identity | model/artifact version and expected hashes are observable | Reconcile release ticket, manifest, deployment configuration, and approved package |
| Resource/latency | CPU/GPU/memory/request/error/latency metrics | Compare with site-approved envelopes; investigate trends and capacity pressure |
| Drift and review | NORMAL/WARNING/CRITICAL plus review queue evidence | Review alerts, HOLD reasons, open work, false-alarm/missed-defect reports, and ownership |
| PLC/MES trace | desired command, acknowledgement and sync state | Run or review a site-approved test chain; reconcile pending/unacknowledged events |
| Shift handoff | trace IDs and append-only states | Record open incidents, held product, manual mode, deviations, and next owner |

## Weekly

| Activity | Repository capability | Site action required |
|---|---|---|
| Trend review | bounded resource, latency, error and drift evidence | Review shifts by product/material/camera; distinguish data drift from quality degradation |
| Alert path | warning/critical states and reason codes | Test notifications, on-call routing, acknowledgement, and escalation |
| Data reconciliation | inspection, PLC/MES and review identities | Reconcile missing/duplicate/pending events and sample trace completeness end to end |
| Backup sample | manifests, histories and database-backed records have defined roles | Restore a controlled backup sample and record integrity; do not overwrite active data |
| Log lifecycle | structured trace keys | Verify rotation, retention, access, time sync and secret/PII redaction |
| Physical inspection | not implemented | Inspect mounts, cables, enclosure, cooling, lighting, optics and contamination per vendor/site SOP |

## Monthly

| Activity | Repository capability | Site action required |
|---|---|---|
| Artifact audit | exact SHA-256 verification and fail-closed load | Recompute approved package hashes and verify read-only storage/backup copies |
| Rollback drill | disposable candidate-only rollback and hash-mismatch blocking | Rehearse the site procedure on production-equivalent equipment without unsafe actuation |
| Access review | lifecycle actions are explicit and auditable | Review service accounts, roles, certificates, credentials, remote access and leavers |
| Review-quality audit | original AI and human disposition remain separate | Sample reviewed and PASS items; assess annotation consistency and unresolved categories |
| Capacity review | queue, throughput, latency, memory and storage evidence | Forecast image/database/log growth and verify spare capacity and backup windows |
| Change reconciliation | model/dataset/deployment identities are separate | Confirm no unapproved firmware, optics, lighting, protocol, model, threshold or configuration drift |

## Quarterly

| Activity | Repository capability | Site action required |
|---|---|---|
| Validation review | offline metrics, FAT/readiness evidence and test matrix | Re-run the approved site regression/SAT subset across representative products and conditions |
| Failure exercise | camera, inference, PLC/MES, drift and rollback simulations exist | Conduct a controlled cross-team incident exercise including containment and communications |
| Security/recovery review | manifests, hashes, audit history and recovery procedures | Review network rules, patching, vulnerability findings, backups, restore, certificates and incident plan |
| Model/data review | drift and feedback are evidence only | Decide whether evidence justifies a separately governed new candidate; never train online |
| Lifecycle audit | candidate/promotion/retirement/rollback state is recorded | Verify approvals, superseded packages, rollback availability and evidence retention |
| Hardware/optical qualification | not implemented | Perform vendor/site calibration, preventive replacement and environmental checks |

## Maintenance Abort Rules

- Hash or identity mismatch: keep inference not-ready/HOLD and restore approved bytes; do not edit the manifest.
- Failed camera/test frame, PLC acknowledgement, MES reconciliation, or rollback gate: do not resume automatic operation.
- Drift CRITICAL or unexplained quality shift: HOLD and investigate; do not tune the threshold.
- Unapproved model, configuration, firmware, optics, lighting, or protocol change: open change control and requalify the affected scope.

Every maintenance record should include time, system/site identity, operator, observed state, evidence URI/hash, action, result, deviation, product containment, approval, and follow-up owner.

Related documents: [operation manual](../operations/operation-manual.md), [existing maintenance guide](../industrial-deployment/maintenance-guide.md), [incident response](incident-response.md), and [rollback guide](../operations/rollback-guide.md).
