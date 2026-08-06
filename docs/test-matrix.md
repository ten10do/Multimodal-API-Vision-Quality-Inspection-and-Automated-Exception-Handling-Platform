# Test Matrix

Every row is a real, executed gate. **A gate that cannot run on hosted CI is a
documented local gate — never a silent skip.** Status values are from the last
full run on this machine (Windows 11 + RTX 5060, Python 3.11 venv).

| Group | Command | passed | failed | skipped | Environment |
|---|---|---|---|---|---|
| Backend unit (incl. MLOps, Copilot, semantics) | `bash scripts/run_clean.sh python -m pytest backend/tests/ simulator/tests/ inference-service/tests/ -q` | 156 | 0 | 26 deselected* | host, CPU |
| PostgreSQL integration | `IVQC_DATABASE_URL=... pytest backend/tests/test_review_concurrency_pg.py -m integration` | 12 | 0 | 0 | PG 5433 + host |
| Industrial integration (HTTP PLC + OPC UA + MES + PG) | `IVQC_REQUIRE_SIMULATORS=1 ... pytest backend/tests/test_industrial_integration.py backend/tests/test_review_concurrency_pg.py -m "integration or opcua or industrial-e2e" -q` | 18 | 0 | **0** | simulators 8501/8502/8503 + PG |
| OPC UA real gate (fail-fast, never skip) | `IVQC_REQUIRE_SIMULATORS=1 ... pytest backend/tests/test_industrial_integration.py -m opcua -q` | 7 | 0 | **0** | OPC UA server 8503 |
| GPU inference (YOLO + PatchCore) | `pytest -m gpu` (local) + `docs/phase6-benchmark.json` | pass | 0 | — | RTX 5060, **local gate** |
| Fault Injection E2E (6 scenarios) | `bash scripts/run_clean.sh python scripts/fault_injection_e2e.py` | 6/6 | 0 | 0 | live stack |
| MLOps unit (registry/gate/drift/manifest) | `pytest backend/tests/test_model_registry.py` | 14 | 0 | 0 | host（含 `test_manifest_artifact_sha256_matches_files`：需本地模型 artifact，**local gate**，CI 显式排除 `-m "not ... and not artifact"`，见 backend-ci.yml） |
| MLOps API + faults | `pytest backend/tests/test_mlops_api.py backend/tests/test_mlops_faults.py` | 9 | 0 | 0 | host |
| MLOps real E2E (register→promote→rollback) | `bash scripts/run_clean.sh python scripts/mlops_e2e.py` | pass | 0 | 0 | live stack |
| Copilot unit/adversarial | `pytest backend/tests/test_copilot.py` | 20 | 0 | 0 | host |
| Copilot deterministic eval (46 fixed cases) | `bash scripts/run_clean.sh python scripts/copilot_eval.py` | targets met | 0 | 0 | offline fake provider |
| Copilot real E2E (7 scenarios) | `bash scripts/run_clean.sh python scripts/copilot_e2e.py` | 7/7 | 0 | 0 | live stack |
| Frontend vitest | `cd frontend && npm test` | 33 | 0 | 0 | node 22 |
| Browser E2E (Playwright, all specs) | `cd frontend && npx playwright test e2e/` | 18 | 0 | 0 | live stack + Vite |
| Clean-DB migration | `cd backend && python -m alembic upgrade head` (fresh PG) | pass | 0 | 0 | PG, also in CI |

\* `deselected` are the `integration / gpu / opcua / industrial-e2e` marked
tests excluded from the plain unit run — they are executed by the dedicated
rows above, never skipped silently.

## CI (GitHub Actions)

- `backend-ci.yml` (ubuntu): clean-DB migration on a fresh PostgreSQL service
  container → backend unit tests → contract tests → python syntax check.
- `frontend-ci.yml` (ubuntu + windows): `npm ci` → `tsc -b && vite build`
  (typecheck + build) → vitest.
- **Local integration gates** (documented, not in hosted CI): GPU inference,
  OPC UA real server, industrial simulators, fault-injection E2E, MLOps E2E,
  Copilot live E2E. Exact commands are in this table.

## Honesty rules

- A gate reported "passed" ran and asserted real behavior.
- `skipped` is only used for optional-simulator dev runs; the gate runs set
  `IVQC_REQUIRE_SIMULATORS=1` so a missing simulator **fails** the gate.
- `REAL_LLM_GATE_NOT_RUN`: the Copilot real-provider smoke cannot run on this
  machine (no external endpoint/API key); it is external integration pending,
  not a pass.
