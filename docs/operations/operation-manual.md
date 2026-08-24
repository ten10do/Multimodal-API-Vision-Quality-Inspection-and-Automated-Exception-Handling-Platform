# Operation Manual

## Start-of-shift checks

1. Runtime state is RUNNING/HEALTHY and required services have no unresolved error.
2. Camera is online; trigger and test-frame capture succeed.
3. GPU/CPU/memory/disk readings are within the site-approved envelope.
4. Model version, lifecycle state, artifact hash, and release ticket agree.
5. Drift is NORMAL, or an approved WARNING observation plan is active.
6. PASS + REJECT + HOLD totals reconcile with the prior shift and open work orders.
7. A test product completes Camera → inference → decision → PLC/MES → review traceability.

## During production

- Monitor request volume, P95 latency, error rate, queue depth, resources, and drift.
- Investigate a HOLD increase by reason code; never compensate by tuning threshold.
- Keep AI evidence immutable and record human decisions separately.
- Reconcile unresolved HOLD events and MES work orders at shift handoff.

## Fault response

| Fault | Immediate action | Recovery gate |
|---|---|---|
| Camera failure | HOLD current product; check trigger, link, power, frame validity | Camera health + test frame |
| Inference failure | HOLD; save request ID and error; verify runtime and hashes | Readiness + smoke inference |
| PLC failure | HOLD; isolate current product; reconcile command state | Communication + idempotent test command |
| MES failure | Preserve event for replay; do not discard trace | Idempotent synchronization |
| Drift WARNING | Continue + alert + increase observation | Stable windows or completed investigation |
| Drift CRITICAL | HOLD and notify quality/release owners | Approved recovery or rollback |

## Shutdown and recovery

Stop new triggers, drain or persist the queue, confirm a safe PLC state, and retain logs. Recover in the order Camera → runtime → inference → decision → PLC → MES → review. Stop at the first failed gate.

## Detailed SOP

[Industrial deployment operation manual](../industrial-deployment/operation-manual.md) · [Maintenance guide](../industrial-deployment/maintenance-guide.md)
