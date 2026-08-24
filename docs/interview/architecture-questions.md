# Architecture Questions

## Why separate inference from the decision engine?

Inference produces model evidence; the decision engine owns business and safety policy. This keeps threshold/lineage validation and PASS/REJECT/HOLD semantics reviewable without embedding PLC behavior in the model service. It also allows a malformed inference response to fail closed.

Trace: [`vision_contract.py`](../../inference-service/inference_app/vision_contract.py) · [`decision_service.py`](../../industrial_loop/decision_service.py)

## Why keep AI, human, and field states separate?

They answer different audit questions. A model observation cannot prove that a PLC acted, and a human correction must not erase the original model output. Separate fields preserve causality and accountability.

Trace: [Engineering decision 2](../engineering-decisions.md#2-ai-result--human-result--final-result-are-three-separate-fields)

## Why is PostgreSQL the source of truth rather than WebSocket?

WebSocket delivery is transient. Persisting first and using WebSocket as notification allows reconnect/replay and avoids losing decisions when a browser disconnects.

Trace: [Engineering decision 3](../engineering-decisions.md#3-websocket-is-a-notification-channel-postgresql-is-the-source-of-truth)

## Where are the trust boundaries?

Camera input is validated before inference; manifests and hashes are validated before model load; the decision engine validates the response before PLC mapping; Dashboard and monitoring are read-only observers; lifecycle writes require valid transitions and verified artifacts.

Trace: [System architecture](../architecture/system-architecture.md)

## Why is the runtime manager transport-agnostic?

Services are injected as start/stop/health callables. That permits simulator, local process, container, or vendor gateway implementations without coupling lifecycle logic to one transport.

Trace: [`runtime_manager.py`](../../industrial_runtime/runtime_manager.py)

## How would this scale beyond one edge node?

Move the local JSON governance journal to an RBAC-controlled signed registry, use durable event/outbox storage for industrial messages, centralize artifact storage, add fleet identity and remote attestation, and aggregate metrics in a time-series platform. The current local contracts remain the edge-facing boundaries.

Trace: [Maturity gaps](../industrial-platform-maturity-report.md)

## Why not place artifacts inside the Docker image?

Separating code image and model identity enables read-only mounting, independent hash verification, rollback without rebuilding runtime code, and clearer supply-chain audit. It also prevents large/private artifacts entering Git.

Trace: [Deployment architecture](../architecture/deployment-architecture.md)

## How is partial failure handled?

Acquisition or inference failure produces HOLD; PLC failure does not assume execution; MES failure preserves replayable state; degraded runtime remains observable; CRITICAL drift holds the line. Recovery has explicit gates rather than optimistic continuation.

Trace: [Data-flow failure table](../architecture/data-flow.md#failure-propagation)
