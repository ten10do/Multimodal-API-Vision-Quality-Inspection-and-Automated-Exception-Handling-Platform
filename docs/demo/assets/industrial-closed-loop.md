# Industrial Closed-Loop Diagram

This diagram presents the simulator-backed industrial workflow. PLC, MES, and review integrations are separate idempotent adapters in the implementation; the arrows show the inspection narrative and evidence reconciliation, not a claim that MES is physically downstream of the PLC network.

```mermaid
flowchart TD
    CAM["Camera trigger and capture"] --> HEALTH{"Frame healthy?"}
    HEALTH -->|"no"| HOLD["HOLD / safe state"]
    HEALTH -->|"yes"| AI["AI Inspection"]
    AI --> VALID{"Evidence valid?"}
    VALID -->|"no"| HOLD
    VALID -->|"yes"| DEC{"PASS / REJECT / HOLD"}

    DEC -->|"PASS"| RUN["PLC: continue line"]
    DEC -->|"REJECT"| REJECT["PLC: reject command"]
    DEC -->|"HOLD"| STOP["PLC: stop / hold command"]

    REJECT --> MES["MES work order"]
    STOP --> MES
    MES --> REVIEW["Human Review"]
    REVIEW --> FINAL["Final disposition and audit evidence"]
    RUN --> AUDIT["Inspection record"]
    REJECT --> AUDIT
    STOP --> AUDIT
    FINAL --> AUDIT
    AUDIT --> DRIFT["Runtime / Drift Monitoring"]

    PLCFAIL["PLC timeout / NACK"] -.-> HOLD
    MESFAIL["MES unavailable"] -.-> PENDING["Observable pending/error state"]
```

## Safety and Traceability

- A deterministic command ID prevents retries from duplicating PLC actions.
- MES work orders are idempotent by inspection/event identity.
- Communication uncertainty remains HOLD or an explicit pending/error state.
- Human review appends a disposition without overwriting the AI observation.
- Monitoring produces alerts and lifecycle evidence; it does not retrain or tune the model.

See the [PLC/MES loop](../../industrial/plc-mes-loop.md) and [protocol adaptation](../../industrial/protocol-adaptation.md) for implementation boundaries.
