# Data Flow

## AI inference pipeline

```mermaid
flowchart LR
    FRAME["Validated 256×1600 frame"] --> TILES["7 × 256×256 tiles"]
    TILES --> DINO["Frozen DINOv2-B/14"]
    DINO --> FINAL["Final normalized patch tokens"]
    FINAL --> ZCA["Frozen train-normal ZCA"]
    ZCA --> L2["Per-patch L2"]
    L2 --> NN["Cosine 1-NN / D3 bank"]
    NN --> A0["A0 global-max image score"]

    DINO --> MID["Block-7 patch tokens / 252"]
    DINO --> HIGH["Final patch tokens / 448"]
    MID --> RL1["R-L1 bank distance map"]
    HIGH --> RL2["R-L2 bank distance map"]
    RL1 --> FUSE["Equal-mean R-L3 fusion"]
    RL2 --> FUSE
    FUSE --> STITCH["Overlap-mean 256×1600 heatmap"]

    A0 --> RESULT["Inference evidence"]
    STITCH --> RESULT
```

The image branch is evaluated before localization. It uses frozen ZCA, the frozen D3 bank, cosine 1-NN, A0 aggregation, and threshold `0.8471092581748962`. The localization branch is independent and cannot supply an image score.

## Industrial control flow

```mermaid
sequenceDiagram
    participant PLC as PLC / Trigger
    participant Camera as Camera Adapter
    participant AI as Inference Service
    participant Decision as Decision Engine
    participant MES as MES
    participant Review as Human Review

    PLC->>Camera: READY + trigger_id
    Camera->>Camera: capture and validate frame
    alt camera or frame failure
        Camera->>Decision: failure evidence
        Decision->>PLC: HOLD / stop signal
        Decision->>Review: create review record
    else valid frame
        Camera->>AI: image + frame lineage
        AI->>Decision: score + threshold + heatmap + model identity
        alt PASS
            Decision->>PLC: continue / release
        else REJECT
            Decision->>PLC: idempotent reject signal
            Decision->>MES: create work order
            Decision->>Review: queue evidence
        else invalid, uncertain, or critical drift
            Decision->>PLC: HOLD / stop signal
            Decision->>Review: queue evidence
        end
    end
    Review->>MES: confirm / false alarm / recheck
```

## Persistence and observability

- The closed-loop simulator uses a thread-safe event store; the full backend uses PostgreSQL as source of truth.
- WebSocket is a notification mechanism, not persistence.
- Runtime metrics and drift reports are bounded histories exposed by read-only APIs.
- Model transitions are appended to `model_governance/model_history.json` with version, hash, metrics, operator, timestamp, transition, and approval state.

## Failure propagation

| Origin | Evidence | Downstream behavior |
|---|---|---|
| Camera | offline, capture error, invalid frame | HOLD and operator review |
| Inference | timeout, exception, non-finite/missing fields | HOLD; no reuse of a previous score |
| Lineage | missing artifact, hash or version mismatch | load/promotion blocked; line remains safe |
| PLC | NACK or unacknowledged command | HOLD/review; no assumed execution |
| MES | unavailable workflow endpoint | preserve event for idempotent replay |
| Drift | WARNING | continue with alert and observation |
| Drift | CRITICAL | HOLD; no automatic model change |

## Evidence

- [D3 dual-branch protocol](../d3-dual-branch-protocol.md)
- [Industrial camera design](../industrial-camera-adapter-design.md)
- [Industrial closed-loop design](../industrial-closed-loop-design.md)
- [Human-review workflow](../d3-human-review-workflow.md)
