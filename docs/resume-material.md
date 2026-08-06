# Resume Material

## Project name

**IndustrialVision-QC — End-to-End Industrial AI Quality Inspection Platform**

## One-line description

A complete industrial machine-vision quality platform: YOLO known-defect
detection + PatchCore anomaly detection → quality rule engine → human
review → PLC/MES field execution → model governance (MLOps) → read-only
evidence-grounded Quality Copilot — all fully tested end-to-end with
simulated factory equipment.

## Resume bullets (3)

1. **Built a full industrial AI quality platform** (FastAPI + PyTorch +
   React): realtime pipeline from camera simulator to YOLO/PatchCore fusion,
   versioned quality rules, human-in-the-loop review, and HTTP/OPC UA PLC +
   MES integration with idempotent commands and fail-safe states
   (`SAFE_HOLD`, never an unverified `RELEASE`). 156 backend unit tests,
   18/18 industrial integration (0 skipped), fault-injection E2E 6/6.

2. **Designed MLOps model governance end-to-end**: Model Registry
   (CANDIDATE→PRODUCTION, single-PRODUCTION enforced), deployment manifest
   with artifact SHA-256 and safe loading (manifest→checksum→smoke→READY),
   promotion gate with **domain validation** (a cross-domain PatchCore with
   AUROC 1.0 is blocked from steel production), drift detection (PSI/KS,
   drift ≠ degradation), registry-driven rollback — every inspection traces
   to deployment + model + dataset versions.

3. **Engineered a read-only Quality Copilot** (LLM tool-calling): fixed
   15-tool allowlist, bounded loop (≤6 calls), deterministic numeric
   grounding (unsupported critical numeric claims = **0** on a 46-case eval),
   causality-safe wording, prompt-injection boundaries, offline
   `FakeLlmProvider` for deterministic CI/eval — a natural-language quality
   analytics assistant that structurally cannot perform write operations.

## Tech stack

Python 3.11 · FastAPI · SQLAlchemy 2 (async) · Alembic · PyTorch · YOLOv8 ·
PatchCore · asyncua (OPC UA) · MLflow · React 18 + TypeScript + Vite ·
WebSocket · PostgreSQL 16 · Docker Compose · pytest · Playwright · Vitest ·
GitHub Actions

## Core metrics

| Metric | Value |
|---|---|
| E2E pipeline (GPU, after client-reuse fix) | ~**8.8×** faster: 561.7 → 63.6 ms avg; 4.85 inspections/s |
| YOLO NEU-DET | mAP50 0.82, recall 0.78 |
| PatchCore MVTec bottle | Image AUROC 1.000 (benchmark domain only; steel NOT validated) |
| Industrial command success / duplicate suppression | 1.0 / 50-50 (idempotency) |
| Copilot deterministic eval (46 cases) | tool selection 1.0, grounding 1.0, **unsupported claims 0.0** |
| Test totals | backend 156 unit + 18 industrial integration (0 skipped) + 33 vitest + 18 Playwright |

## Most valuable engineering problems (interview anchors)

1. **httpx client reuse** — profiled and removed a ~150 ms/client-construction
   bottleneck → 8.8× E2E improvement with zero semantic change.
2. **CUDA/pytest native crash** — environment (DLL) bisection, not package
   versions; clean-env wrapper.
3. **OPC UA namespace hard-coding bug + silent-skip** — a real bug that was
   being reported as a pass; fixed with dynamic URI→index resolution and
   fail-fast gates.
4. **Fail-safe industrial semantics** — `NOT_INTEGRATED` vs `SAFE_HOLD`,
   idempotent commands, "unknown → SAFE_HOLD, never RELEASE".
5. **MLOps domain honesty** — a perfect-AUROC cross-domain model is blocked
   from production by the domain gate.
6. **Copilot safety by construction** — zero write tools + deterministic
   numeric grounding (unsupported claims = 0).

> Positioning: this is **industrial AI systems engineering** — the value is
> in correctness boundaries, fail-safety, traceability and honest gates, not
> in the framework list.
