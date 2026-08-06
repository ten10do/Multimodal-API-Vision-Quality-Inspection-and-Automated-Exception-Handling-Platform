# Interview Guide

## 30-second intro

> "I built an end-to-end industrial AI quality inspection platform. A camera
> simulator feeds frames through YOLO for known defects and PatchCore for
> unknown anomalies; a versioned rule engine decides PASS/FAIL/REVIEW; humans
> review the uncertain cases; the final decision is executed against
> simulated PLC (HTTP and OPC UA) and MES with idempotent, fail-safe
> commands. On top of that, an MLOps layer governs model promotion,
> monitoring, drift and rollback, and a read-only Quality Copilot answers
> natural-language questions with evidence-grounded numbers. Everything is
> tested — 156 unit tests, industrial integration gates with zero skips, and
> a deterministic 46-case Copilot evaluation where unsupported numeric
> claims are zero."

## 2-minute intro

Structure: problem → pipeline → industrial closed loop → governance →
Copilot → engineering highlights → honest limits.

1. **Problem**: factories need end-to-end quality control: detect known
   defects, flag unknown anomalies, let humans arbitrate, and execute the
   decision on the field layer — all traceable and governable.
2. **Pipeline**: simulator → queue → inference (YOLO bbox + PatchCore
   anomaly score) → fusion → rule engine → PostgreSQL → WebSocket dashboard.
3. **Human review**: uncertain (REVIEW) cases create audited review tasks;
   AI evidence is never overwritten.
4. **Industrial**: PASS→RELEASE, FAIL→REJECT, REVIEW→HOLD via HTTP or OPC UA
   PLC adapter + MES sync; `command_id` idempotency makes retries safe;
   unknown state → SAFE_HOLD (never RELEASE); disabled PLC → honest
   NOT_INTEGRATED.
5. **MLOps**: registry with single-PRODUCTION, deployment manifest + SHA-256
   safe loading, promotion gate with domain validation, drift (PSI/KS),
   rollback via registry pointer.
6. **Copilot**: read-only LLM tool-calling over a fixed allowlist,
   evidence-first answers, deterministic numeric grounding.
7. **Highlights**: 8.8× E2E fix from httpx client reuse; OPC UA namespace
   hard-coding bug caught by a fail-fast gate; CUDA crash traced to process
   environment.
8. **Honest limits**: PatchCore steel-domain accuracy not validated;
   real-LLM provider smoke pending an external endpoint; PLC/MES are
   simulated.

## Architecture deep-dive

- Four responsibility boundaries: AI decision (models), Human decision
  (review), Final quality result (rules + review resolution), Industrial
  execution (PLC/MES). See `docs/architecture.md`.
- DB is the source of truth; WebSocket is a notification channel.
- `AI result / Human result / Final result` are separate fields.

## Why YOLO does not make the quality decision

YOLO emits raw detections (class, confidence, bbox). Whether a detection
means reject depends on rules, severity, and human context — business logic
that must be versioned, auditable and reviewable independently of the model.
Keeping models evidence-only means you can swap models without changing
business semantics, and you can add review/override without touching the
detector.

## Why PatchCore needs domain matching

PatchCore's memory bank is domain-specific. The MVTec-bottle baseline scores
Image AUROC 1.000 on bottles but is cross-domain on steel and flags almost
everything anomalous. A perfect benchmark number that cannot generalize is
worth zero in production — hence `steel_domain_validated=false` and a domain
gate that blocks promotion. (Great follow-up discussion: few-shot adaptation,
per-class normal banks, domain validation procedure.)

## Why human-in-the-loop

Long-tail and novel defects are exactly where fixed rules fail; humans
provide ground truth that (a) resolves REVIEW cases, (b) feeds retraining
candidate manifests, and (c) enables quality-degradation detection (drift
alone cannot claim accuracy loss without human ground truth).

## PLC fail-safe

Unknown field state → SAFE_HOLD; only explicit PASS → RELEASE; NACK →
COMMAND_FAILED; retries are bounded and idempotent by command_id. A
communication failure never produces a release.

## OPC UA bug story

The server allocates the Plc object's namespace index at startup (2); the
adapter hard-coded "0:Plc" → BadNoMatch, and the test silently skipped
because the error string contained "unreachable". Fixed with URI→index
resolution + browse fallback and made the gate fail-fast. Two lessons:
never hard-code deployment-specific namespace indices, and a skipped gate is
not a pass.

## MLOps rollback

Rollback switches the registry production pointer (archive current → promote
target). No rebuild, no copying best.pt. Inspections before and after the
rollback trace to their own deployment_version, so you can always answer
"which AI stack judged this batch".

## Copilot safety

No write tools exist in the registry — safety is structural. The system
prompt marks tool output as untrusted data (prompt injection in DB fields is
treated as data). A user asking to "release this product" gets analysis and a
read-only declaration, never an action. Numeric grounding strips unsupported
numbers (target 0).

## Performance optimization case

Phase 3 benchmark: E2E avg 561.7 ms. Profiling showed a new
`httpx.AsyncClient` per request (~150 ms construction, no connection reuse).
Caching one client per event loop → 63.6 ms (~8.8×), throughput 4.85/s,
with no business change. Lesson: measure, find the systemic cause, fix it,
re-benchmark.

## Likely follow-up questions (prepare answers)

- *Why PostgreSQL and not a TSDB for monitoring?* — Volume is modest; PG is
  transactional truth; metrics are computed by aggregation queries, not
  stored as a time series. Prometheus was deliberately not added.
- *Why not Redis?* — Conversation context and stats caching are in-memory
  with TTL; acceptable at this scale; noted as a roadmap item.
- *How do you prevent double PLC execution on retry?* — command_id
  idempotency at the simulator/gateway level (verified 50/50).
- *What does NOT_INTEGRATED mean?* — PLC disabled: desired command recorded,
  adapter never called, no fabricated field state.
- *How is the Copilot evaluated?* — 46 fixed cases with expected tools /
  required facts / forbidden claims; metrics: tool selection, grounding,
  fact coverage, unsupported-claim rate (0), latency; offline deterministic
  provider keeps it CI-safe; real-provider smoke is a separate pending gate.
- *What would you do next?* — steel-domain PatchCore + domain validation,
  real-LLM gate, Redis/multi-worker, real PLC/MES gateway pilots.
