# AI Inspection Flow Diagram

The AI path uses a frozen image-scoring branch and an independent localization branch. This diagram is a presentation view of the documented [D3 adaptation](../../ai/d3-domain-adaptation.md) and [anomaly-detection pipeline](../../ai/anomaly-detection.md); it does not define a new model or operating point.

```mermaid
flowchart LR
    FRAME["Validated CameraFrame"] --> CONTRACT{"Input and lineage valid?"}
    CONTRACT -->|"no"| HOLD["HOLD / AI_SYSTEM_FAILURE"]
    CONTRACT -->|"yes"| TILE["Seven overlapping tiles"]

    subgraph IMAGE["Frozen D3 image branch"]
        TILE --> DINO["Frozen DINOv2 ViT-B/14"]
        DINO --> ZCA["Frozen train-normal ZCA"]
        ZCA --> BANK["Cosine 1-NN D3 bank"]
        BANK --> SCORE["A0 image score"]
    end

    subgraph LOCAL["Independent R-L3 localization branch"]
        DINO --> RL1["R-L1 feature map"]
        DINO --> RL2["R-L2 feature map"]
        RL1 --> HEAT["Fused defect heatmap"]
        RL2 --> HEAT
    end

    SCORE --> VALID{"Finite score and identity match?"}
    VALID -->|"no"| HOLD
    VALID -->|"yes"| POLICY["Decision Engine"]
    HEAT -. "operator evidence only" .-> EVIDENCE["Inspection Evidence"]
    POLICY --> EVIDENCE
```

## Invariants

1. Input or identity failure cannot become PASS.
2. The D3 score and frozen threshold determine image-level policy evidence.
3. R-L3 localization provides a heatmap but cannot change the D3 image score or threshold.
4. The output retains model version, artifact version, score, reason, latency, and trace identity.

Measured capability and limitations remain in the [v1.0.0 release notes](../../release/v1.0.0-release-notes.md).
