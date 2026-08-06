# Failure Scenarios & Recovery

Each row: **failure → expected behavior → recovery → data consistency**.
These behaviors are covered by tests / fault-injection suites, not just
documented.

| Failure | Expected behavior | Recovery | Data consistency |
|---|---|---|---|
| Inference service unavailable | backend returns 503 `model_unavailable`/`inference_unreachable`; inspection marked FAILED | retry via API; service restarts idempotently | failed inspection persisted with error_message; no partial row |
| PostgreSQL unavailable | API 5xx; `/ready` reports `database: error` | Docker restart; app reconnects (async engine pool) | DB is source of truth; no writes lost after recovery |
| WebSocket disconnected | dashboard freezes live feed, keeps polling API | frontend auto-reconnects, re-syncs from DB | all state re-queryable from PG (WS is notify-only) |
| PLC timeout | bounded retries → `TIMEOUT` → **SAFE_HOLD** (never RELEASE) | operator investigates; safe-state holds product | desired_command recorded; no false RELEASE |
| PLC NACK | definitive refusal → `COMMAND_FAILED` (no retry) | operator action required | command + NACK persisted; state remains fail-safe |
| PLC offline | adapter `PlcUnreachable` → retries → **SAFE_HOLD** | restart gateway; product stays held | `plc_events` row with real (un-acked) state |
| MES 500 / timeout | bounded retries → `MesUnreachable` → `mes_sync_status=FAILED` | MES back online → re-sync (idempotent by inspection_id) | inspection stays completed; **never rolled back** |
| Model artifact checksum mismatch | inference `/ready` → `not_ready` with problem list | re-provision artifact or correct manifest | service refuses to run unverified weights (8E) |
| Model load / smoke failure | `/ready` not-ready; 503 on infer | fix artifact; restart | no inference served with broken model |
| Model registry unavailable | registry API 5xx; promote/rollback fail closed | restore backend; DB has the state | production pointer unchanged (registry is DB-backed) |
| Bad candidate model | promotion gate rejects (metric/domain) | register a passing candidate | invalid model never becomes PRODUCTION (8N) |
| Rollback after bad v2 | registry switches production pointer back to v1 | manifest + registry flip; no rebuild | inspections before/after rollback trace to their own deployment_version |
| LLM provider unavailable | Copilot returns controlled message + limitation, **no traceback, no hang, no fabricated answer** | provider back → retry | conversation + evidence preserved; safety read_only intact |
| Copilot tool timeout / 500 | tool error recorded in evidence; loop continues | next turn re-plans | answer still grounded; error surfaced in limitations |
| Empty database / unknown entity | tools return honest errors; Copilot recovers with a message | seed demo data (scripts/demo_seed.py) | no crash, no invented facts |

## Principles

1. **Fail-safe over convenience**: unknown field state → `SAFE_HOLD`; only an
   explicit PASS produces `RELEASE`.
2. **Never fabricate**: disabled PLC → `NOT_INTEGRATED`, not a fake ACK or a
   fake hold.
3. **DB is the truth**: every recovery path reconciles against PostgreSQL.
4. **Idempotency everywhere**: PLC commands by `command_id`, MES submissions
   by `inspection_id:kind`, review resolution by task version — retries are
   safe by design.
5. **Fail-fast gates**: a missing simulator/key/artifact fails the gate; it is
   never silently skipped.
