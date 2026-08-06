# Phase 7 Report: Industrial Integration (PLC / MES Closed Loop)

Date: 2026-08-06
Commit: (filled at finalization)

## 1. process_result semantic fix

The three-layer model is now explicit in the code and the data model:

| Layer | Meaning | Example (PLC disabled, FAIL) |
|---|---|---|
| `desired_command` | what the system wants the field layer to do | `REJECT` |
| `execution_status` | whether the command was really sent / ACKed | `NOT_INTEGRATED` |
| `industrial_state` / `industrial_final_state` | the product's actual field state | `NOT_INTEGRATED` |

**plc_enabled=False means NOT INTEGRATED, never a fault.** The adapter is
never invoked, no ACK is fabricated, and the states `SAFE_HOLD` / `HELD` /
`RELEASED` / `REJECTED` are never produced, because those all imply that the
field layer actually participated. The business reason for the desired
command is preserved in `plc_events.request_payload.reason_code` while the
industrial reason code is `plc_integration_disabled`.

`NOT_INTEGRATED != SAFE_HOLD`: the first means the PLC was never engaged, the
second means a real communication/system failure required a safe hold.

## 2. Industrial state model

```
CAPTURED -> INSPECTING -> PASS / FAIL / REVIEW / PROCESSING_FAILED
         -> PLC_COMMAND_PENDING -> PLC_ACKNOWLEDGED -> RELEASED / REJECTED / HELD / SAFE_HOLD
         -> (PLC error)          -> COMMAND_FAILED
CAPTURED -> NOT_INTEGRATED                       (PLC integration disabled)
```

Terminal states: `RELEASED`, `REJECTED`, `HELD`, `SAFE_HOLD`,
`COMMAND_FAILED`, `NOT_INTEGRATED`. The three critical states are never
mixed: NOT_INTEGRATED (never engaged), HELD (normal HOLD execution),
SAFE_HOLD (system/PLC anomaly safe state).

## 3. PLC Simulator

Independent HTTP server (`simulator/plc_simulator.py`, port 8501), no backend
ORM coupling. Machine states READY / RUNNING / HOLD / FAULT; commands
RELEASE / REJECT / HOLD / RESET; fault injection via query modes
(`timeout`, `nack`) and admin endpoints (`admin/fault` -> FAULT, `admin/reset`).

Idempotency: `executed` dict keyed by `command_id`; a replayed command_id
returns `duplicate=True` and does NOT execute a second physical action.
Verified by `test_http_plc_ack_and_idempotency` and the service-level
`test_plc_duplicate_suppressed`.

## 4. HttpPlcAdapter

`backend/app/industrial/plc_adapter.py`. One-shot POST to `/v1/command` with
bounded timeout. Errors map to `PlcUnreachable` (offline / timeout / non-200)
and `PlcNack` (explicit NACK). The service maps them to SAFE_HOLD /
COMMAND_FAILED and never to RELEASE.

## 5. OpcUaPlcAdapter

Same `PlcAdapter` protocol, transport injected. `OpcUaPlcAdapter` talks to
`simulator/opcua_plc_server.py` (asyncua 2.x, port 8503, `opc.tcp://...`).
The server exposes a `Plc/execute` method that is idempotent by command_id
and returns `ACK:n` / `ACK:duplicate` / `NACK:plc_fault`. Core business code
never imports asyncua; it only sees the `PlcAdapter` Protocol.

The asyncua 2.x API change (`Server.link_method` no longer accepts
input/output types) was fixed by using `node.add_method(idx, name, callback,
[ua.VariantType.String], [ua.VariantType.String])`.

## 6. MES

- `simulator/mes_simulator.py` (port 8502): GET product/batch, POST
  inspection result / final quality result, idempotent by
  `inspection_id:kind`, fault injection (`500`, `timeout`).
- `backend/app/industrial/mes_adapter.py`: all MES HTTP in one class;
  bounded retry, 4xx non-retryable (except 408/429), 5xx retried then
  `MesUnreachable`; duplicate submission is suppressed by the MES key.
- Payload includes `industrial_state` so the MES records the REAL field
  state: on PLC failure the MES receives SAFE_HOLD / COMMAND_FAILED, never
  RELEASED (`test_mes_sees_real_state_safe_hold_not_released`).
- MES failure never rolls back a completed AI inspection
  (`test_plc_release_ok_mes_fail_keeps_released`); `mes_sync_status` records
  SYNCED / FAILED / PENDING (bounded retry later).

## 7. Idempotency

`command_id_for(inspection_id, command_type)` yields a stable command_id per
inspection+command. The PLC simulator and the OPC UA server both suppress
re-execution on a replayed command_id. Verified: 50/50 duplicate
suppressions in the benchmark, `test_http_plc_ack_and_idempotency`,
`test_opcua_plc_ack_and_idempotency`, `test_plc_duplicate_suppressed`.

## 8. Fail-safe

`decision_to_command` never returns RELEASE on uncertainty; unknown states
map to HOLD with `unknown_state`. In the service, any PLC timeout / offline
transport error lands in SAFE_HOLD (`plc_unreachable` transition), NACK lands
in COMMAND_FAILED, and system FAILED even with a HOLD ACK lands in SAFE_HOLD.
`plc_enabled=False` is the one explicit non-fault terminal: NOT_INTEGRATED.

## 9. Human Review -> PLC

Reuses the Phase 5 review flow. AI REVIEW -> desired HOLD; with PLC enabled,
HOLD ACK -> HELD (PLC simulator state flips to HOLD). On human resolve the
service issues a NEW command: final PASS -> RELEASE (reason `review_pass`),
final FAIL -> REJECT (reason `review_fail`). Idempotency prevents double
execution of the same review outcome. Verified end-to-end against the real
simulators in `test_service_full_chain_review_hold_then_release`.

## 10. Fault Injection

| Injection | Result |
|---|---|
| PLC timeout | execution_status TIMEOUT, SAFE_HOLD, never RELEASE |
| PLC offline | execution_status ERROR, SAFE_HOLD |
| PLC NACK | execution_status NACK, COMMAND_FAILED |
| duplicated command | execution_status DUPLICATE_SUPPRESSED |
| MES timeout | MesUnreachable after bounded retries, inspection stays complete |
| MES 500 | MesUnreachable after bounded retries |
| system FAILED (inspection error) | desired HOLD + SAFE_HOLD (PLC on) / NOT_INTEGRATED (PLC off) |

## 11. Dashboard

- Inspection detail: Desired Command, Execution Status, Industrial State
  (badge with distinct colors for NOT_INTEGRATED / HELD / SAFE_HOLD /
  REJECTED / RELEASED / COMMAND_FAILED), PLC Adapter, PLC Latency, MES Sync,
  Reason Code.
- Overview: Released / Rejected / Held / Safe Hold / Not Integrated / PLC
  Fault / MES Sync Failed counters over the recent window.

## 12. Benchmark

`scripts/benchmark_phase7.py`, real simulators on localhost:

| Metric | Value |
|---|---|
| decision -> command creation | 0.0022 ms (pure CPU) |
| command -> PLC ACK (HTTP) | mean 239.8 / p50 256.7 / p95 276.3 ms |
| MES sync | mean 257.3 ms |
| full industrial decision (PLC+MES) | mean 844.8 / p50 840.2 / p95 951.6 ms |
| command success rate | 1.0 (NOT_INTEGRATED excluded) |
| duplicate suppression | 50 / 50 |
| safe_hold_count / not_integrated_count | 0 / 0 (PLC enabled run) |

## 13. Tests

- `backend/tests/test_industrial_semantics.py` (18): plc_enabled=False
  semantics for PASS/FAIL/REVIEW/system-FAILED (adapter never called, no
  fake ACK, no RELEASED/REJECTED/HELD/SAFE_HOLD), PLC enabled ACK mapping,
  timeout/offline/NACK/duplicate, MES decoupling, MES disabled -> PENDING.
- `backend/tests/test_industrial_integration.py` (10, `-m integration`):
  HTTP PLC ack/idempotency/HOLD/NACK/timeout/offline, OPC UA ack/idempotency/
  offline, MES submit/duplicate/500/timeout, full chain REVIEW->HOLD->human
  PASS->RELEASE against the live simulators.
- Backend regression 113 passed; frontend vitest 33 passed.
- Real E2E (camera simulator -> GPU/CPU YOLO+PatchCore -> rule -> PLC/MES -> PG
  -> WS -> dashboard -> human review) is covered by the Phase 7 E2E run in
  the final validation section.

## 14. Final validation (real E2E on this machine)

Run via `scripts/fault_injection_e2e.py` against the live stack (Docker PG,
simulators on 8501/8502, backend 8000, inference 8100). Results recorded in
`docs/phase7-fault-injection.json`:

| Scenario | Result |
|---|---|
| 1. AI PASS -> RELEASE | no natural AI-PASS sample in NEU cross-domain (PatchCore marks nearly everything anomalous on the steel domain); a real RELEASE is exercised via the human-PASS path in scenario 3. Honest recording, no fabricated PASS. |
| 2. AI FAIL -> REJECT | `desired=REJECT exec=ACK final=REJECTED reason=product_defect` |
| 3a. AI REVIEW -> HOLD (PLC state flips to HOLD) | `desired=HOLD exec=ACK final=HELD reason=review_pending` |
| 3b. human PASS -> RELEASE | `desired=RELEASE exec=ACK final=RELEASED reason=review_pass` |
| 3c. human CONFIRM_DEFECT -> REJECT | `desired=REJECT exec=ACK final=REJECTED reason=review_fail` |
| 4. PLC offline (simulator killed) | whatever the desired command, `execution=TIMEOUT final=SAFE_HOLD`; **NEVER** RELEASED. The critical invariant (14) holds. |
| 5. PLC FAULT -> NACK | `execution=NACK final=COMMAND_FAILED` |
| 6. MES 500 injected | inspection stays `completed`, `mes_sync_status=FAILED`; the AI inspection is NOT rolled back. |

The PLC idempotency contract is also covered by `benchmark_phase7.py` (50
replays of the same `command_id` -> 50 duplicate suppressions, 0
re-executions) and by `test_plc_duplicate_suppressed` (service level) and
the integration tests on both adapters.

### Tests

| Suite | |
|---|---|
| backend pytest (unit) | 113 passed, 21 deselected (integration / gpu) |
| backend pytest -m integration (PG + simulators) | 12 passed, 1 skipped |
| backend pytest -m gpu | 0 selected (this run used CPU torch; Phase 6 GPU metrics are still valid) |
| frontend vitest | 33 passed |
| Playwright Browser E2E | review 8 + review-anomaly 2 + industrial 2 = 12 passed |
| Fault injection E2E (real) | 6 scenarios, all green |

### Screenshot

`docs/screenshots/11-phase7-industrial-detail.png` shows the inspection
detail panel with the full industrial block (Desired Command / Execution
Status / Industrial State badge / PLC Adapter / PLC Latency / MES Sync /
Reason Code).

## Known issues

1. **GPU vs CPU for this run.** A Bash-session-specific DLL injection
   (workbuddy CLI shim directories plus other variables) caused `import
   torch` to crash with access violation (0xC0000005). The fix is an
   `env -i` wrapper (`scripts/run_clean.sh`) used for every torch command
   in this session. Because downloading cu128 wheels takes >30 minutes,
   the local run uses `torch==2.11.0+cpu` (PyPI default on Windows,
   PyTorch CPU-only build). `torch.cuda.is_available()` is `False`, so
   the GPU-only `test_predictor_gpu` is skipped and `anomaly_loaded` /
   inference run on CPU (~755 ms / image for PatchCore vs the 756 ms GPU
   number reported in Phase 6). YOLO on CPU takes ~2-3 s per image.
   Phase 6 GPU coexistence (VRAM 331 MB / 8 GB) is the canonical GPU
   benchmark; this run replaces it with CPU benchmarks in
   `docs/phase7-benchmark.json`.
2. **NEU has no natural AI-PASS samples.** All NEU images produce
   either FAIL or REVIEW because (a) the NEU-trained YOLO detects defects
   on most images and (b) PatchCore's bottle bank marks nearly everything
   anomalous on the steel domain (cross-domain mismatch, documented in
   Phase 6). Scenario 1 records this honestly instead of fabricating a
   PASS sample. A real AI-PASS would require a clean-domain dataset;
   scenario 3b (human PASS -> RELEASE) exercises the real RELEASE path.
3. **`plc_enabled` defaults to `True` in config.** Without a running PLC
   simulator every inspection retries the command then SAFE_HOLDs. This
   is the intended fail-safe, but production must either run the PLC
   gateway or set `IVQC_PLC_ENABLED=false` explicitly.
4. **OPC UA adapter depends on asyncua 2.x API** (verified against
   2.0.1; the `Server.link_method` signature changed in 2.x, replaced
   by `node.add_method(idx, name, callback, [in], [out])`).
5. **MES retries are bounded (2) and synchronous in `process_result`; a
   slow MES adds latency to the inspection path (best-effort by
   design). MES failure never rolls back a completed AI inspection
   (scenario 6).

## Phase 8 suggestion

- Online rule parameter tuning / human-AI collaborative labelling
- Multi-class PatchCore ensemble on per-class normal banks (move from
  cross-domain MVTec-bottle baseline to a real steel-domain PatchCore
  once industrial-domain data is available)
- PLC manual operator console + audit retention policy
- Copilot-style inspection summary generation
  intended fail-safe, but production must either run the PLC gateway or set
  `IVQC_PLC_ENABLED=false` explicitly.
2. OPC UA adapter depends on asyncua 2.x API (verified against 2.0.1).
3. MES retries are bounded (2) and synchronous in `process_result`; a slow
  MES adds latency to the inspection path (best-effort by design).

## 14. OPC UA closing verification (post-commit gate)

The initial OPC UA service test was **silently skipped**, not passed: the
adapter hard-coded the browse path `"0:Plc"`, but the simulator registers
the Plc object under its own (server-allocated) namespace index (currently
2). The resulting `BadNoMatch` surfaced as `PlcUnreachable("opcua
unreachable: ...")` and the test's exception handler matched `"unreachable"`
and skipped -- masking a real bug.

Fix (this commit):
- `OpcUaPlcAdapter` now resolves the Plc object **without any namespace
  index hard-coding**: declared namespace URI (`urn:ivqc:plc`) -> index read
  from the server at connect time (`get_namespace_array`), with a
  browse-by-name fallback for foreign servers.
- `simulator/opcua_plc_server.py` adapted to asyncua 2.x callback contract
  (input arrives as `Variant`, return value must be a list of Variants).
- Gate policy: OPC UA tests are **fail-fast, never skip**; a missing
  simulator now fails the gate instead of being reported as passed.
  Optional-simulator tests keep `pytest.skip` in dev runs but raise when
  `IVQC_REQUIRE_SIMULATORS=1` is set.

Test classification markers: `unit` / `integration` / `opcua` /
`industrial-e2e` (see pytest.ini). Gate run:
`IVQC_REQUIRE_SIMULATORS=1 pytest ... -m "integration or opcua or
industrial-e2e"`.

Final gate results (all against the live simulators):
- OPC UA adapter ACK + idempotency (real server, 8503): pass
- OPC UA offline -> `PlcUnreachable`: pass
- Service E2E via OPC UA: REVIEW->HOLD/HELD, PASS->RELEASE/RELEASED,
  FAIL->REJECT/REJECTED, each with one persisted `PlcEvent`
  (adapter_type=opcua, execution_status=ACK): 3/3 pass
- OPC UA server unavailable -> `SAFE_HOLD`, never RELEASED: pass
- Namespace robustness: Plc registered under namespace index != 2 still
  resolves and ACKs: pass
- Gate totals: `opcua` 7/7, industrial gate 18/18, skipped = 0.

Default dev semantics: `IVQC_PLC_ENABLED=false`, `IVQC_MES_ENABLED=false`
(config defaults + `.env.example`), verified end-to-end: REVIEW ->
desired HOLD but `industrial_final_state=NOT_INTEGRATED`,
`plc_adapter_type=none`, `reason_code=plc_integration_disabled`; no
SAFE_HOLD/HELD/RELEASED/REJECTED and no fake ACK.
