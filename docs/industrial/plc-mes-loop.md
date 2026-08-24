# PLC and MES Closed Loop

## Decision mapping

| Decision | PLC behavior | MES / review behavior |
|---|---|---|
| PASS | Keep line running | Record inspection |
| REJECT | Pulse reject signal | Create idempotent work order and review evidence |
| HOLD | Assert stop signal | Queue operator review; prohibit automatic release |

The decision engine validates score, threshold, model identity, artifact identity, and failure state before mapping an event. Unknown or invalid input is HOLD, never PASS.

## Idempotency

Every command ID is derived from inspection identity and command type. Both HTTP and OPC UA simulator paths suppress duplicates, allowing bounded retries without double actuation. MES work-order creation is similarly idempotent per event ID.

## State machines

- PLC: `READY → RUNNING ↔ STOP`.
- MES work order: `OPEN → PROCESSING → CLOSED`.
- Review: pending evidence → confirm defect / false alarm / request recheck.

Illegal state transitions raise errors and remain observable rather than being coerced into success.

## Traceability

AI decision, final human disposition, PLC terminal state, and MES state are separate fields. This preserves the answers to four different questions: what the model observed, what policy decided, what the operator concluded, and what the field layer executed.

## Verification

- deterministic 1,000-product simulation;
- command duplicate suppression;
- PLC NACK and unavailable-path tests;
- MES lifecycle and review reconciliation;
- OPC UA namespace resolution and real server/adapter integration gate;
- camera/inference/drift failures proven to reach HOLD.

## Trace

[Closed-loop design](../industrial-closed-loop-design.md) · [FAT report](../d3-factory-acceptance-report.md) · [Engineering decisions](../engineering-decisions.md)
