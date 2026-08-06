# IndustrialVision-QC

> An end-to-end industrial AI quality inspection platform combining
> **known-defect detection** (YOLO), **unknown-anomaly detection** (PatchCore),
> **human review**, **PLC/MES industrial integration**, **model governance**
> and **evidence-grounded quality analytics** (read-only Quality Copilot).

A portfolio-grade, fully testable implementation of a real factory quality
control software chain — from camera frame to field-layer execution — with
every industrial device simulated in software (see `docs/`).

![Overview](docs/screenshots/01-overview.png)
*Real system screenshots are in [docs/screenshots/](docs/screenshots/).*

## Core capabilities

- 🔵 **Known-defect detection** — YOLOv8s on NEU-DET steel surface (6 classes,
  bbox + confidence + severity).
- 🔵 **Unknown-anomaly detection** — PatchCore memory-bank anomaly scoring +
  heatmap; fused with YOLO (`UNKNOWN_ANOMALY` when defect + anomaly disagree).
- 🟠 **Quality rule engine** — maps vision results to `PASS / FAIL / REVIEW`
  with explicit, versioned rules.
- 🟢 **Human-in-the-loop review** — claim/resolve review tasks; AI evidence is
  never overwritten; every decision is audited (who/when/why, corrections).
- 🔴 **Industrial integration** — HTTP **and** OPC UA PLC adapters + MES sync;
  idempotent commands by `command_id`; fail-safe (`SAFE_HOLD`, never an
  unverified `RELEASE`); `NOT_INTEGRATED` is the honest default when the field
  layer is absent.
- 🧠 **MLOps & model governance** — Model Registry (CANDIDATE → STAGING →
  PRODUCTION → ARCHIVED, single-PRODUCTION enforced), deployment manifest with
  artifact SHA-256, safe model loading (manifest → checksum → smoke → READY),
  promotion gate with **domain validation**, production monitoring, drift
  detection (PSI/KS), retraining candidate manifests, registry-driven rollback.
- 💬 **Quality Copilot** — a **read-only** natural-language quality analysis
  assistant: bounded tool-calling over a fixed allowlist, evidence-first
  answers, deterministic numeric grounding (unsupported claims = 0), strict
  causality wording, prompt-injection boundaries.
- 🖥️ **Dashboard** — React/TypeScript: Production Overview, Live Inspection
  (WebSocket), Quality Traceability, Review Queue, Model Operations, Quality
  Copilot.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full diagram with the
four responsibility boundaries (AI decision / Human decision / Final quality
result / Industrial execution).

```
Camera Simulator
   │  frame
   ▼
Realtime Queue / Orchestrator
   │
   ▼
Inference Service ── YOLO (known defects) + PatchCore (anomaly) ──► Vision Fusion
   │
   ▼
Quality Rule Engine ── PASS / FAIL / REVIEW (final quality result)
   │
   ▼
PostgreSQL ◄── WebSocket ──► Dashboard   (DB is the source of truth)
   │
   ├──► Human Review (claim / resolve, audited)
   ├──► Industrial Integration: HTTP PLC · OPC UA · MES (idempotent, fail-safe)
   ├──► MLOps: Registry · Deployment Manifest · Monitoring · Drift · Rollback
   └──► Quality Copilot (read-only, evidence-grounded, no write tools)
```

## End-to-end workflow

1. Simulator frame → inference service: YOLO detections + PatchCore anomaly.
2. Fusion + rule engine → final quality result (PASS / FAIL / REVIEW).
3. Persisted to PostgreSQL; dashboard updates via WebSocket.
4. REVIEW → human review task → human confirm/correct/pass (audited).
5. Final result → PLC command (RELEASE / REJECT / HOLD) → ACK → MES sync;
   unknown state → `SAFE_HOLD` (never RELEASE).
6. MLOps stamps model + deployment version on every inspection; drift and
   monitoring aggregates are served from the same DB.
7. Copilot answers questions with tool-grounded evidence only.

## Tech stack

| Layer | Technologies |
|---|---|
| Models | PyTorch, YOLOv8 (ultralytics), PatchCore (wide_resnet50_2) |
| Backend | Python 3.11, FastAPI, SQLAlchemy 2 (async), Alembic, Pydantic v2, uvicorn |
| Realtime | WebSocket (async), httpx pooled client |
| Industrial | asyncua 2.x (OPC UA), HTTP adapters, bounded retries |
| MLOps | MLflow (file store), custom registry/drift/manifest, deterministic grounding |
| Frontend | React 18, TypeScript, Vite, @tanstack/react-query, ECharts |
| Infra | PostgreSQL 16 (Docker, host 5433), Docker Compose, GitHub Actions |
| QA | pytest (+asyncio), Playwright, Vitest, alembic clean-DB migration |

> Honest note: Redis / MinIO / Prometheus / Grafana / OCR were **not** used in
> the final system. Monitoring is served via API aggregation from PostgreSQL
> (not a TSDB), storage is the local filesystem provider, and there is no OCR.
> See the Capability Matrix below.

## Key engineering decisions

The 13 most valuable decisions (each with rationale and measured impact) are
documented in [docs/engineering-decisions.md](docs/engineering-decisions.md).
Highlights:

- AI result / Human result / Final result are **three separate fields**.
- **DB is the source of truth**; WebSocket is only a notification channel.
- Reusing one `httpx.AsyncClient` cut E2E latency ~**8.8×** (561.7 ms → 63.6 ms).
- `NOT_INTEGRATED` ≠ `SAFE_HOLD` — never fabricate a field state.
- OPC UA **namespace-index hard-coding bug** caught by a real integration gate.
- **Silent skip → fail-fast** gate policy (a skipped gate is not a pass).
- Deployment / model / dataset **versions kept distinct**.
- Copilot is **evidence-first and read-only** by construction.

## Benchmark

Unified summary: [docs/benchmark-summary.md](docs/benchmark-summary.md).
Selected numbers (RTX 5060 8 GB / CPU fallback noted where relevant):

| Benchmark | Result |
|---|---|
| YOLO inference (RTX 5060) | mean 15.4 ms · p50 13.3 ms · p95 21.5 ms |
| YOLO NEU-DET | mAP50 0.82 · recall 0.78 (steel domain) |
| PatchCore MVTec bottle | Image AUROC 1.000 · Pixel 0.986 · AUPRO 0.955 (benchmark domain only) |
| E2E pipeline (GPU, pooled client) | throughput 4.85/s · P50 47.4 ms · P95 73.8 ms |
| PLC ACK / MES sync | mean 240 ms / 257 ms; command success 1.0; duplicate suppression 50/50 |
| Copilot deterministic eval (46 cases) | tool selection 1.0 · grounding 1.0 · **unsupported claims 0.0** · P95 70 ms |

## Quick Start

```bash
# 1. infrastructure (PostgreSQL 16 on host port 5433)
docker compose up -d postgres

# 2. one-command demo (simulators + inference + backend + frontend + seed)
bash scripts/demo_up.sh          # infrastructure in Docker, GPU inference on host

# 3. health check
bash scripts/run_clean.sh python scripts/health_check.py
```

Open http://127.0.0.1:5173 → Overview. The demo seed creates deterministic
scenarios (known defect / unknown anomaly / human review / PLC hold-release /
monitoring / copilot-ready data) — it is a **demo fixture**, not production
data ([docs/10-phase7-report.md](docs/10-phase7-report.md) documents the
honest limits, e.g. PatchCore steel-domain accuracy is not validated).

GPU inference stays **on the host** (Windows + RTX); Docker runs PostgreSQL.
See [docs/10-phase7-report.md](docs/10-phase7-report.md) for the environment
notes (Bash-session DLL isolation via `scripts/run_clean.sh`).

## Tests

| Suite | Command | Count |
|---|---|---|
| Backend unit (incl. MLOps, Copilot, semantics) | `bash scripts/run_clean.sh python -m pytest backend/tests/ simulator/tests/ inference-service/tests/ -q` | 156 passed |
| Industrial integration (PG + simulators, fail-fast) | `IVQC_REQUIRE_SIMULATORS=1 ... pytest backend/tests/test_industrial_integration.py backend/tests/test_review_concurrency_pg.py -m "integration or opcua or industrial-e2e"` | 18/18, 0 skipped |
| Fault-injection E2E | `bash scripts/run_clean.sh python scripts/fault_injection_e2e.py` | 6/6 |
| Copilot deterministic eval (46 fixed cases) | `bash scripts/run_clean.sh python scripts/copilot_eval.py` | targets met (unsupported=0) |
| Copilot real E2E (7 scenarios) | `bash scripts/run_clean.sh python scripts/copilot_e2e.py` | 7/7 |
| Frontend vitest | `cd frontend && npm test` | 33/33 |
| Browser E2E (Playwright) | `cd frontend && npx playwright test e2e/` | 18/18 |

Full matrix with commands / counts / environment: [docs/test-matrix.md](docs/test-matrix.md).
GPU, OPC UA and industrial-simulator gates run **locally** (documented gates,
never silently skipped in CI — see [.github/workflows/](.github/workflows/)).

## Project structure

```
backend/            FastAPI: rules, review, industrial, MLOps, Copilot, alembic
inference-service/  YOLO + PatchCore + fusion; manifest-verified /ready
simulator/          camera pipeline + HTTP PLC / MES / OPC UA simulators
frontend/           React + Vite dashboard + Playwright e2e
packages/           shared vision-contract
scripts/            run_clean.sh, fault_injection_e2e, mlops_e2e, copilot_eval,
                    copilot_e2e, health_check, demo_up, demo_seed, backfill_mlflow
model-training/     training + datasets (ignored artifacts)
docs/               phase reports 0-12 + benchmarks + engineering guides
copilot-eval/       fixed 46-case evaluation dataset
```

## Honest Capability Matrix

| Capability | Status |
|---|---|
| YOLO steel defect detection (NEU-DET) | ✅ Verified (mAP50 0.82) |
| PatchCore MVTec-bottle benchmark | ✅ Verified (Image AUROC 1.000) |
| PatchCore steel-domain accuracy | ⚠️ **Not validated** — cross-domain baseline only |
| Human-in-the-loop review | ✅ Verified |
| HTTP PLC integration | ✅ Verified |
| OPC UA integration | ✅ Verified (real server → adapter → persistence gate) |
| MES integration | ✅ Verified |
| Model Registry / single-PRODUCTION / Rollback | ✅ Verified |
| Deployment manifest + SHA-256 safe loading | ✅ Verified |
| Promotion gate + domain validation | ✅ Verified (MVTec AUROC 1.0 blocked for steel) |
| Drift detection (PSI/KS, drift ≠ degradation) | ✅ Verified |
| Copilot deterministic eval (46 cases) | ✅ Verified (unsupported claims = 0) |
| Copilot real-LLM provider smoke | ⏳ **Pending external endpoint** (`REAL_LLM_GATE_NOT_RUN`) |
| Redis / MinIO / Prometheus / Grafana / OCR | ❌ Not used (by design) |

Two items are **explicitly not hidden**:
- **PatchCore domain mismatch**: the MVTec-bottle baseline is a benchmark only;
  `steel_domain_validated=false` blocks promotion to a steel production model.
- **Real LLM gate not run**: `OpenAiLlmProvider` is implemented and the 5-case
  gate is defined, but no local/cloud OpenAI-compatible endpoint or API key
  exists on this machine → `REAL_LLM_GATE_NOT_RUN` (external integration
  pending). `FakeLlmProvider` remains the deterministic CI/eval provider.

## Limitations

1. PatchCore is a cross-domain MVTec baseline; steel-domain accuracy is **not
   validated** (honest boundary, see [docs/09-phase6-report.md](docs/09-phase6-report.md)).
2. Real-LLM Copilot smoke is pending an external endpoint (`REAL_LLM_GATE_NOT_RUN`).
3. PLC / MES / OPC UA are **simulated**; field deployment requires real
   gateways (the adapters and idempotency protocol are transport-level).
4. Conversation context and Copilot stats caching are in-memory (single
   worker; TTL-based). No Redis.
5. GPU inference runs on the host (Windows + RTX), not inside Docker.
6. The environment requires `scripts/run_clean.sh` for torch processes
   (Bash-session DLL isolation), see [docs/10-phase7-report.md](docs/10-phase7-report.md).

## Roadmap

- Real-LLM Copilot gate once an endpoint/API key is available (5-case gate
  already defined in [docs/12-phase9-report.md](docs/12-phase9-report.md)).
- Steel-domain PatchCore training + domain validation (replaces the MVTec
  baseline).
- Redis-backed cache / multi-worker conversation store; optional Prometheus
  exposition for large-scale monitoring.
- Real PLC / MES gateway adapters for field pilots.

---

Phase docs: [docs/](docs/) — every phase has a report with decisions,
benchmarks and honest known-issues. Interview preparation:
[docs/interview-guide.md](docs/interview-guide.md) ·
[docs/resume-material.md](docs/resume-material.md) ·
[docs/demo-script.md](docs/demo-script.md).
