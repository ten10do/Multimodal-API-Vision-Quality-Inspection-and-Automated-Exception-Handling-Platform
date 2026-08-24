# Factory Integration Guide

This guide describes the work required to move the simulator-backed reference platform toward a real factory installation. It is a planning and acceptance document, not production authorization.

## Capability Boundary

### Implemented

- Vendor-neutral camera trigger/capture/health contract with deterministic virtual-camera replay and failure injection.
- Frozen inference evidence contract, fail-closed PASS/REJECT/HOLD decision semantics, and model/artifact identity checks.
- Idempotent PLC command and MES work-order behavior through HTTP/OPC UA simulator-backed paths.
- Human review, traceability, dashboard, runtime health/resource monitoring, drift evidence, and lifecycle governance.
- Read-only artifact mounting, SHA-256 verification, startup gates, rollback procedures, and simulator-backed factory acceptance evidence.

### Future Integration

- Physical camera, lens, enclosure, trigger, lighting, and mechanical installation.
- Vendor SDK or transport adapter, physical PLC/MES mapping, plant network and cybersecurity configuration.
- Site image collection, labeling, acceptance datasets, operating-point approval, and cross-shift/product validation.
- Machine-safety integration, electrical certification, production change control, operator training, maintenance ownership, and site acceptance testing.

## 1. Camera Installation

**Implemented:** `CameraAdapter` defines connect, trigger, capture, health, status, and disconnect behavior. `CameraFrame` preserves camera, frame, sequence, trigger, timestamp, dimensions, latency, and error identity. Acquisition failures reach HOLD.

**Future Integration:** Select camera, lens, working distance, field of view, resolution, enclosure, cooling, cable path, mount rigidity, and maintenance access. Define hardware trigger wiring and encoder/product association. Implement and qualify the vendor/GigE Vision/USB3 Vision adapter under line speed, vibration, contamination, reconnect, and multi-camera load.

Acceptance evidence includes calibration records, focus/distortion checks, full-surface coverage, trigger-to-product matching, missed/duplicate frame rates, timestamp accuracy, and recovery drills.

## 2. Lighting Configuration

**Implemented:** The platform validates image/frame contracts and can expose acquisition failures; it does not control a physical lighting system.

**Future Integration:** Select illumination geometry, spectrum, polarization, diffusion, strobe mode, controller, exposure window, shielding, cooling, and mounting. Lock approved camera exposure/gain/white-balance settings where applicable. Test reflectivity, oil, scale, vibration, ambient light, material variation, lamp aging, and cleaning intervals.

Capture reference images and measurable illumination uniformity/exposure limits for startup and maintenance checks. Lighting changes require controlled requalification because they can shift input distributions.

## 3. PLC Signal Design

**Implemented:** The decision layer maps valid evidence to PASS, REJECT, or HOLD; simulator-backed adapters provide deterministic command IDs, bounded retries, acknowledgement, duplicate suppression, and visible NACK/timeout states.

**Future Integration:** Produce an approved I/O or node/register specification covering ready, trigger, product identity, PASS/REJECT/HOLD, command ID/sequence, acknowledgement, busy, heartbeat, reset, bypass, manual mode, and safe-state ownership. Validate pulse timing, reject distance compensation, interlocks, watchdogs, restart behavior, and PLC scan-cycle constraints on the physical controller.

Machine safety, emergency stop, guards, and safety PLC functions remain independent of AI software and must follow plant safety engineering and applicable regulations.

## 4. Network Isolation

**Implemented:** Architecture documents separate OT, edge, service, and management responsibilities. Runtime health and dashboard paths are distinct from model and PLC control authority.

**Future Integration:** Create plant-approved VLAN/firewall rules, allow-listed flows, fixed addressing/DNS, time synchronization, certificate and credential ownership, remote-access controls, logging, backup links, and incident response. Qualify camera bandwidth, latency, jitter, packet loss, reconnect, and degraded/offline operation.

Use the [network topology](industrial-network-topology.md) as a starting point, then replace every placeholder interface with the site's reviewed source, destination, protocol, port, owner, and availability requirement.

## 5. Data Collection

**Implemented:** The event model preserves product, batch, camera, model/artifact, score, decision, PLC/MES, review, and timestamp evidence. Review and drift components can consume traceable observations.

**Future Integration:** Define consent/ownership, retention, access, storage capacity, image sampling, defect taxonomy, annotation procedure, reviewer agreement, privacy rules, and data export. Collect representative normal and defect data across products, suppliers, shifts, speeds, lighting states, maintenance conditions, and seasonal/environmental variation.

Keep site acceptance data sealed from development decisions. Do not alter frozen artifacts or operating thresholds to force a gate to pass; a new model candidate requires a separate governed lifecycle.

## 6. Model Deployment

**Implemented:** The release path verifies dependency lock, release/candidate manifests, qualification evidence, artifact hashes, array validity, and smoke readiness before manual release review. Artifacts are externally managed and mounted read-only.

**Future Integration:** Qualify the industrial computer, GPU, drivers, CUDA/runtime compatibility, storage, power, thermal envelope, watchdog, startup order, service account, artifact registry, network access, backup, and observability. Execute site-specific accuracy, latency, throughput, endurance, failover, and security gates before production authorization.

Deployment approval must bind source revision, deployment configuration, model/artifact identities, site dataset/evidence, hardware identity, approvers, and rollback target.

## 7. Exception Handling

**Implemented:** Missing frames, inference errors, identity mismatch, invalid/non-finite results, PLC NACK/unavailability, and other uncertain states are observable and use fail-closed HOLD or explicit pending/error semantics. Human review appends disposition without overwriting AI evidence.

**Future Integration:** Assign alarms, severity, escalation, response time, operator actions, bypass authority, manual inspection, product quarantine, MES reconciliation, network outage behavior, and maintenance ownership. Test camera disconnect, lighting failure, GPU/service loss, PLC restart, MES outage, clock drift, storage pressure, and partial recovery with actual plant systems.

Document who may release held material. AI or dashboard software must not replace plant safety systems or unilaterally override physical interlocks.

## 8. Rollback

**Implemented:** Governance records candidate identity and supports fail-closed rollback to a previously verified package without editing artifact bytes. Rollback procedures and drills are documented.

**Future Integration:** Define the approved previous version, artifact/config backups, database compatibility, PLC/MES coordination, operator communication, maintenance window, product containment, rollback trigger, authority, recovery time objective, and post-rollback verification. Rehearse rollback on production-equivalent hardware and gateways before launch.

Rollback is complete only when the prior verified inference package is healthy, PLC/MES state is reconciled, held products are controlled, monitoring is normal or understood, and the change record is closed.

## Site Acceptance Exit Criteria

- Mechanical, optical, electrical, network, cybersecurity, and safety reviews are approved.
- Camera/lighting repeatability and product-to-frame traceability meet signed site limits.
- Physical PLC and MES integrations pass idempotency, timeout, disconnect, restart, and reconciliation tests.
- Site data and model evidence meet a separately approved acceptance protocol without mutating the frozen release.
- End-to-end latency, throughput, soak, recovery, monitoring, review, and rollback gates pass on qualified hardware.
- Operators and maintainers complete training; support, escalation, backups, and change control have named owners.
- A designated plant authority grants production release. Repository qualification alone is insufficient.

Related documents: [protocol adaptation guide](protocol-adaptation-guide.md), [deployment guide](../../operations/deployment-guide.md), [operation manual](../../operations/operation-manual.md), and [rollback guide](../../operations/rollback-guide.md).
