# Architecture Decisions

This record captures the architectural choices that turn model evidence into an auditable industrial candidate. Each section separates committed observations from the decision and from possible future work.

## 1. Candidate Registry Before Production Promotion

### Observed

The industrialization report verified all seven referenced files and cross-checked manifest lineage against artifact hashes, threshold, splits, metrics, and verdicts. The registry accepts only `CANDIDATE`; payload drift, path traversal, missing files, SHA mismatch, evidence mismatch, threshold drift, or conflicting registration fails closed. Candidate inference is opt-in, and `production_promotion=false` remains explicit.

### Decision

Use a candidate registry as a quarantine and verification boundary. Registration proves identity and evidence consistency; it does not imply deployment or promotion. Promotion remains a separate governed action with readiness, factory, site, and human approval gates.

### Future

A multi-site deployment may replace the file registry with a signed remote registry and approval service, but it must preserve immutable identity, evidence lineage, role separation, and rollback compatibility.

Source: [D3 industrialization report](../steel-patchcore-d3-industrialization-report.md).

## 2. Fail Closed on Uncertainty

### Observed

The platform distinguishes valid PASS/REJECT decisions from camera failure, invalid/non-finite inference, identity mismatch, drift criticality, PLC timeout/NACK, and unavailable dependencies. Production-readiness qualification reports PASS for stability, robustness, monitoring, human review, rollback, and tests while leaving model, artifact, threshold, configuration, and deployment unchanged.

### Decision

Unknown or unverified state maps to HOLD or an explicit pending/error state, never optimistic PASS. Field execution and MES synchronization retain their own terminal states so a model result cannot fabricate a successful physical action.

### Future

Physical-site work must bind each failure class to plant alarms, interlocks, product containment, escalation, and recovery ownership. A new adapter cannot weaken the HOLD invariant.

Sources: [system architecture](../architecture/system-architecture.md) and [production-readiness qualification](../d3-production-readiness-report.md).

## 3. Separate Image and Pixel Branches

### Observed

The frozen D3 branch achieved image AUROC `0.817907171428`. R-L3 achieved Pixel AUROC `0.924139` and AUPRO `0.799398`, but its standalone image AUROC was only `0.661694`. Dual integration produced `0` D3 score mismatches.

### Decision

Assign one objective to each branch: D3-ZCA A0 is the only image-ranking/decision score, while R-L3 produces localization evidence for review. The heatmap cannot modify the image score or threshold.

### Future

Evaluate new localization representations independently and require score invariance before integration. A joint model is not excluded in principle, but it would be a new candidate with new evidence rather than an in-place change.

Sources: [localization investigation](../d3-localization-representation-investigation.md) and [dual-branch evaluation](../dual-branch-evaluation-report.md).

## 4. No Online Training

### Observed

The drift layer observes feature distributions without retraining, recalibrating, rebuilding a bank, or tuning a threshold. Its documented scenarios include 10,000 normal frames remaining NORMAL, brightness shift becoming WARNING, and material change becoming CRITICAL with eight PLC stop signals and zero PASS after critical. Factory-acceptance feedback records operator review, false positive, and false negative examples with `automatic_retraining=false`.

### Decision

Production feedback and drift are evidence, not training commands. Recovery and candidate creation are explicit offline governance activities with frozen data roles, evaluation gates, review, and rollback. Runtime processes have no authority to mutate the model.

### Future

An offline retraining or adaptation pipeline may consume approved, versioned feedback after label-quality review. It must create a new dataset version and candidate, pass sealed evaluation, and never overwrite the running artifact.

Sources: [drift monitoring](../industrial/drift-monitoring.md) and [FAT human-feedback report](../d3-fat-human-feedback-report.json).

## 5. Artifact Hash Governance

### Observed

Candidate registration resolves controlled URIs, hashes every referenced artifact, checks result lineage, loads NPZ with pickle disabled, validates exact shapes and finiteness, marks arrays read-only, and re-hashes after load. The performance report recorded identical before/after hashes and `artifact_unchanged=true` across its candidate benchmark.

### Decision

Bind runtime identity to artifact bytes, not filenames or mutable version labels. Hash mismatch blocks startup, registration, promotion, or rollback; the system restores approved bytes rather than editing a manifest to match an unexpected file.

### Future

Site deployments can add signed manifests, hardware-backed keys, transparency logs, replicated registries, and software-bill-of-materials attestations. SHA-256 identity and immutable rollback targets remain the minimum contract.

Sources: [D3 industrialization report](../steel-patchcore-d3-industrialization-report.md), [performance report](../d3-performance-report.json), and [deployment architecture](../architecture/deployment-architecture.md).
