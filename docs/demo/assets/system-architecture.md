# System Architecture Diagram

This portfolio view condenses the canonical [system architecture](../../architecture/system-architecture.md). It preserves the separation between OT devices, edge inference, quality services, monitoring, and model governance.

```mermaid
flowchart LR
    subgraph OT["OT device zone"]
        CAM["Industrial Camera"]
        PLC["PLC / Line Control"]
        LINE["Production Process"]
    end

    subgraph EDGE["Edge compute zone"]
        ACQ["Camera Adapter"]
        RT["Edge Runtime"]
        AI["AI Inspection"]
        DEC["Fail-Closed Decision"]
    end

    subgraph QUALITY["Quality service zone"]
        MES["MES Workflow"]
        REVIEW["Human Review"]
        AUDIT["Inspection / Audit Data"]
        DASH["Operations Dashboard"]
    end

    subgraph OPS["Operations and governance"]
        MON["Runtime / Drift Monitoring"]
        GOV["Model Lifecycle Governance"]
    end

    CAM --> ACQ --> AI --> DEC
    RT -. "lifecycle and health" .-> ACQ
    RT -. "lifecycle and health" .-> AI
    RT -. "lifecycle and health" .-> DEC
    DEC --> PLC --> LINE
    DEC --> MES
    DEC --> REVIEW
    MES --> AUDIT
    REVIEW --> AUDIT
    DEC --> AUDIT --> DASH
    AI --> MON --> DASH
    GOV -. "verified artifact identity" .-> AI
```

## Reading Guide

- The camera adapter owns acquisition identity and health, not quality decisions.
- AI inspection produces evidence; the decision engine owns PASS, REJECT, and HOLD policy.
- PLC, MES, and human review preserve distinct execution and disposition states.
- Monitoring observes runtime and feature distributions but cannot mutate or promote a model.
- Governance supplies verified model identity and lifecycle evidence; artifacts remain externally managed and read-only.

Detailed deployment boundaries are available in the [deployment architecture](../../architecture/deployment-architecture.md).
