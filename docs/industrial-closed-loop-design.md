# Industrial Closed-Loop Design (`industrial_loop/`)

Status: **simulation layer around the frozen D3 release** (`steel-patchcore-d3-release@1.3.0`).
This layer is strictly peripheral: it does not modify the D3 model, weights, memory bank,
threshold, feature extractor, or any existing evaluation result. The decision engine only
READS the frozen lineage and fails closed on any anomaly.

## 1. System architecture

```
+-------------------+     +--------------------+     +---------------------------+
| Camera Simulation | --> | Inference Service  | --> | Decision Engine           |
| (CameraSimulation)|     | (D3, UNCHANGED)    |     | (industrial_loop/         |
| frames + GT flags |     | /v1/infer          |     |  decision_service.py)     |
+-------------------+     +--------------------+     +------------+--------------+
                                                                  |
                     +--------------------------------------------+-------------+
                     |                        |                   |            |
                     v                        v                   v            v
          +---------------------+   +-----------------+   +-----------+   +-----------+
          | OPC UA PLC Simulator|   | MES Work Orders |   | Human     |   | Production|
          | READY/RUNNING/STOP  |   | OPEN/PROCESSING |   | Review    |   | Dashboard |
          | reject/stop signal  |   | -> CLOSED       |   | Workflow  |   | (FastAPI) |
          +---------------------+   +-----------------+   +-----------+   +-----------+
```

Package layout (all new code; nothing existing was modified):

| File | Phase | Responsibility |
|---|---|---|
| `industrial_loop/events.py` | 1 | `InspectionEvent` schema + status enums |
| `industrial_loop/decision_service.py` | 2 | fail-close PASS/REJECT/HOLD engine |
| `industrial_loop/plc_adapter.py` | 3 | PLC state machine, signals, JSONL trace, asyncua server/client |
| `industrial_loop/mes_service.py` | 4 | work-order mock (idempotent per event) |
| `industrial_loop/human_review.py` | 5 | review queue + CONFIRM/FALSE_ALARM/RECHECK |
| `industrial_loop/dashboard.py` | 6 | FastAPI APIs + embedded zero-dependency SPA |
| `industrial_loop/factory_simulator.py` | 7 | end-to-end simulation + report |

## 2. Inspection event model (Phase 1)

One traceable event per product. Required fields: `id`, `timestamp`, `product_id`,
`batch_id`, `camera_id`, `model_version`, `artifact_version`, `image_score`, `pixel_score`,
`decision`, `reason_code`, `heatmap_reference`, `operator_status`, `plc_status`,
`mes_status` (+ traceability extras: `schema_version`, `trace_id`, `threshold`,
`confidence`, `error_detail`, `latency_ms`).

Decision/reason pairing is validated at construction:

| decision | allowed reason_code |
|---|---|
| PASS | NORMAL |
| REJECT | DEFECT_DETECTED |
| HOLD | AI_SYSTEM_FAILURE, LOW_CONFIDENCE, PLC_UNACKNOWLEDGED |

Events are immutable; enrichment returns copies (`with_updates`), so the camera→inference→
PLC→MES→review lifecycle of a product is reconstructable from `trace_id` alone.

## 3. Data flow

1. **Camera simulation** emits seeded frames (`product_id`, `batch_id`, `camera_id`,
   ground-truth defect flag for wiring validation only).
2. **Inference**: the synthetic backend emulates D3 scores anchored to the frozen threshold
   (deterministic, seed 42); the optional live backend posts real frames to the unchanged
   service (`POST /v1/infer`) and maps `anomaly.anomaly_score/threshold/model_version/
   artifact_version` into the same adapter view.
3. **Decision engine** produces the event (see §4).
4. **PLC** receives one command per event; the resulting ACK/NACK is written back into
   `plc_status`.
5. **MES**: every REJECT auto-creates one work order (idempotent per event id).
6. **Human review**: every REJECT/HOLD is enqueued; operator answers update
   `operator_status` and close the MES order.
7. **Dashboard** reads the shared in-memory store over REST.

## 4. Decision engine & exception-handling strategy (Phase 2)

Fail-close rules, evaluated in order — any anomaly yields HOLD, never PASS:

| condition | outcome |
|---|---|
| backend error / transport failure | HOLD · AI_SYSTEM_FAILURE |
| missing or non-finite image score / threshold | HOLD · AI_SYSTEM_FAILURE |
| missing model/artifact version, version mismatch | HOLD · AI_SYSTEM_FAILURE |
| threshold ≠ frozen release threshold | HOLD · AI_SYSTEM_FAILURE (lineage guard) |
| `image_score >= 0.8471092581748962` | REJECT · DEFECT_DETECTED |
| inside optional guard band (default off) | HOLD · LOW_CONFIDENCE → human review |
| otherwise | PASS · NORMAL |

The reject threshold is the frozen D3 release value; this layer never tunes it. The optional
guard band only routes borderline sub-threshold products to human review (strictly more
conservative than auto-PASS) and is disabled by default so decisions mirror the frozen
threshold exactly. A PLC NACK on a REJECT/HOLD is surfaced as `plc_status=NACK` and the
product still goes to human review.

## 5. PLC flow (Phase 3)

States `READY → RUNNING ⇄ STOP`. Mapping: PASS keeps the line running; REJECT pulses
`reject_signal` (line continues); HOLD asserts `stop_signal` (line STOPs; a supervisor reset
returns it to RUNNING — counted as `supervisor_resumes`). Commands are idempotent by
`command_id`; every command is appended to `runs/industrial-loop/plc_trace.jsonl`
(gitignored). `InMemoryPlc` implements the semantics deterministically for tests/simulation;
`OpcUaPlcServer` + `OpcUaPlcAdapter` provide the real asyncua address space
(`State`, `RejectSignal`, `StopSignal`, `LastCommandId`, `ExecutedCount`, method `execute`)
for the marked integration test and standalone runs
(`python -m industrial_loop.plc_adapter`). No real PLC is ever contacted.

## 6. MES flow (Phase 4)

```
AI REJECT -> create work order (OPEN) -> operator PROCESSING -> CLOSED
```

Work-order fields: `work_order_id`, `event_id`, `batch_id`, `defect_type`,
`image_id`, `severity` (HIGH/MEDIUM/LOW from the score margin), `status`, timestamps,
`closed_reason`, `reviewed_by`. Creation is idempotent per inspection event; illegal
transitions raise.

## 7. Human review workflow (Phase 5)

REJECT/HOLD events are enqueued (`operator_status=PENDING`). Operators submit
`CONFIRM_DEFECT` / `FALSE_ALARM` / `REQUEST_RECHECK` with reviewer name, time and comment.
CONFIRM/FALSE_ALARM close the linked MES order (`confirmed_defect` / `false_alarm`);
REQUEST_RECHECK increments the recheck counter and re-queues the item. Records keep the full
audit tuple `(review_id, event_id, trace_id, ai_decision, image_score, reviewer, outcome,
comment, reviewed_at)`.

## 8. Dashboard (Phase 6)

FastAPI app (`create_app(store)`) serving:
`/api/summary`, `/api/events`, `/api/anomalies/recent`, `/api/trend`, `/api/work-orders`,
`/api/reviews`, `/api/plc/state`, plus `/` — an embedded single-page dashboard (counters,
decision-trend chart, recent anomalies with heatmap previews rendered client-side, latest
events table). Framework note: React/Streamlit were evaluated; a self-contained vanilla-JS
page was chosen so the dashboard runs fully offline with no build step and no CDN, while the
REST API stays framework-agnostic for a future React client.

## 9. End-to-end factory simulation (Phase 7)

```
python -m industrial_loop.factory_simulator --products 1000 --seed 42 [--backend live --live-url ...]
```

Latest committed-config run (synthetic backend, seed 42): **1000 products → 913 PASS /
72 REJECT / 15 HOLD**, PLC 1000 actions (72 reject_signals, 15 stop_signals, 0 NACKs,
15 supervisor resumes, final state RUNNING), MES 72 orders (all CLOSED after review),
112 review submissions (55 confirm / 32 false-alarm / 25 recheck), loop overhead ≈0.35 ms
per product. Ground-truth wiring check: 73 injected defects → 72 auto-rejected, 1 captured
as an AI-failure HOLD that went to human review, 0 false rejects. Report:
`runs/industrial-loop/factory_simulation_report.json` (gitignored runtime artifact).

## 10. Tests

`inference-service/tests/test_industrial_loop_*.py` — 51 default-suite tests covering the
five required areas (decision engine incl. fail-close matrix, PLC mapping/idempotency/NACK/
trace, MES lifecycle, human review, dashboard API + end-to-end factory simulation with
conservation + determinism checks), plus one `opcua`-marked live server/client roundtrip
(excluded from the default suite by repo convention; verified passing).

## 11. Deployment

* **Simulation (default)**: pure Python, no external services;
  `python -m industrial_loop.factory_simulator`.
* **With live inference**: start the unchanged inference service (Docker GPU runtime,
  `IVQC_D3_CANDIDATE_MANIFEST` pinned) and run the simulator with `--backend live`.
* **Standalone OPC UA simulator**: `python -m industrial_loop.plc_adapter`
  (endpoint `opc.tcp://127.0.0.1:4840/freeopcua/server/`).
* Runtime artifacts (PLC trace, simulation report) live under `runs/industrial-loop/`
  and are gitignored; only source, tests and docs are committed.
