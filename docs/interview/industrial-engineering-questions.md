# Industrial Engineering Questions

## Why does every error become HOLD rather than REJECT?

REJECT asserts that a defect was detected. A camera, model, contract, artifact, or communication error is uncertainty, so HOLD is the honest safe state. It stops automatic release and requests investigation without fabricating defect evidence.

Trace: [`decision_service.py`](../../industrial_loop/decision_service.py)

## How are duplicate PLC actions prevented?

Commands carry deterministic IDs derived from inspection identity and command type. Adapters and simulators suppress replays, so bounded retries do not cause a second reject or stop action.

Trace: [Engineering decision 9](../engineering-decisions.md#9-plc-command-idempotency-by-command_id)

## What is the difference between NOT_INTEGRATED and SAFE_HOLD?

NOT_INTEGRATED means no field adapter was enabled and no physical state is claimed. SAFE_HOLD represents a real integration failure after the field path was enabled. Conflating them would make telemetry dishonest.

Trace: [Engineering decision 8](../engineering-decisions.md#8-not_integrated--safe_hold)

## How would a real camera be integrated?

Implement the existing adapter contract with a GigE/GenICam, USB3 Vision, or vendor SDK transport. Preserve trigger ID, frame metadata, health, failure taxonomy, and interlock semantics. Inference and decision code remain unchanged.

Trace: [Camera integration](../industrial/camera-integration.md)

## What happens if MES is unavailable after PLC execution?

The field action and business synchronization remain separate states. The event must be retained in a durable outbox and replayed idempotently when MES returns; PLC execution is not repeated merely because MES synchronization failed.

Trace: [PLC/MES loop](../industrial/plc-mes-loop.md)

## How does drift affect production?

WARNING continues production with an alert and higher observation. CRITICAL maps subsequent work to HOLD. Drift cannot tune, retrain, or rebuild artifacts; recovery requires an approved investigation or rollback.

Trace: [Drift monitoring](../industrial/drift-monitoring.md)

## What makes the FAT evidence honest?

The report distinguishes accelerated discrete-event replay from an eight-hour wall-clock soak, identifies simulator-backed field devices, verifies failure semantics, and explicitly states that no production promotion or artifact change occurred.

Trace: [Factory acceptance report](../d3-factory-acceptance-report.md)

## Which gates cannot run on hosted CI?

GPU inference, real OPC UA server/adapter, live industrial simulators, and other environment-bound gates are documented local gates. Required simulator gates fail rather than silently skip when dependencies are absent.

Trace: [Test matrix](../test-matrix.md)

## What remains before site acceptance?

Vendor SDK integration, electrical and network safety, PLC/MES namespace and timing qualification, watchdog/redundancy, target throughput, fault recovery, cybersecurity review, SAT, shift SOP validation, and continuous SLA evidence.

Trace: [Industrial requirement specification](../industrial-deployment/industrial-requirement-spec.md)
