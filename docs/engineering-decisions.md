# Engineering Decisions

The 13 decisions below are the highest-value interview material. Each is a
**real problem encountered and solved in this project**, with rationale and
measured impact where available.

## 1. `severity` does not enter the Vision Contract

The shared `vision-contract` returns detections (class, confidence, bbox).
Severity is derived **downstream** by the quality rule engine from
confidence + area + rules — not hard-coded in the model output. Why: model
output should be raw evidence; business meaning (severity, pass/fail) belongs
to a versioned, reviewable rule layer. This keeps the model swappable without
changing business semantics.

## 2. AI result / Human result / Final result are three separate fields

An inspection stores `quality_result` (AI), `final_quality_result`
(human-resolved) and the industrial terminal state separately. A REVIEW with
`final_quality_result=None` is a **pending** state — and this caught a real
bug: passing `None` into the industrial mapper produced `SAFE_HOLD` for what
should have been `HOLD`. The three-layer split makes "who decided what" always
answerable.

## 3. WebSocket is a notification channel; PostgreSQL is the source of truth

The dashboard updates via WebSocket push, but every screen re-fetches from
the DB for truth. A WS disconnect never loses data — replay from PG on
reconnect (the frontend reconnects and re-syncs). This decoupling also made
the realtime path easy to test independently of persistence.

## 4. httpx client reuse → ~8.8× E2E improvement

Phase 3 benchmark showed E2E avg 561.7 ms / P95 822 ms. Root cause: the
inference client created a new `httpx.AsyncClient` per request (≈150 ms
construction on Windows, no connection reuse). Fix: cache one client per
event loop. E2E avg → 63.6 ms (~**8.8×**), throughput 3.5/s → 4.85/s, with
**zero** business-semantics change. Classic "profile before optimizing".

## 5. pytest/CUDA native crash isolation (Bash-session DLL pollution)

`import torch` crashed with `0xC0000005` inside pytest on this machine. Root
cause was **not** torch: the Bash session injected environment (WorkBuddy
shim PYTHONPATH / MCP JSON) that loaded conflicting DLLs. A clean
`env -i` wrapper (`scripts/run_clean.sh`) runs torch processes reliably.
Lesson: when a native stack crashes, bisect the **process environment**, not
just the package versions.

## 6. PostgreSQL on 5433 coexists with the native instance

The machine already ran Windows PostgreSQL on 5432. The project's container
uses host 5433 → container 5432, with `industrialvision_dev` /
`industrialvision_test` databases kept separate. Tests never touch the dev DB.

## 7. PatchCore domain mismatch is a feature, not a bug

The MVTec-bottle PatchCore baseline scores Image AUROC 1.000 — but on NEU
steel it is cross-domain and marks nearly everything anomalous. The system
**honestly reports** `steel_domain_validated=false` and blocks promotion to a
steel production model via the domain gate, even though the benchmark number
is perfect. A 1.0 AUROC that cannot generalize is worth zero in production.

## 8. `NOT_INTEGRATED` ≠ `SAFE_HOLD`

When the PLC is disabled, the system records `NOT_INTEGRATED`: the desired
command is computed and logged, the adapter is **never** called, and no field
state is claimed. `SAFE_HOLD` is reserved for a real communication failure
when integration **is** enabled. Fabricating `SAFE_HOLD` (or worse,
`RELEASED`) for a disabled PLC would lie about the field layer. Default dev
mode is `NOT_INTEGRATED`; industrial mode is explicit opt-in.

## 9. PLC command idempotency by `command_id`

Every industrial command carries a `command_id` derived from
`inspection_id + command_type`. Both the HTTP PLC simulator and the OPC UA
server suppress replays (duplicate → `DUPLICATE_SUPPRESSED`, no second
physical action). Verified 50/50 in the benchmark and end-to-end in the
fault-injection suite. This is what makes retries safe: bounded retries can
re-send the same command without double-execution.

## 10. OPC UA namespace-index hard-coding bug

The OPC UA server registers the `Plc` object under a **server-allocated**
namespace index (2). The adapter originally used the hard-coded browse path
`"0:Plc"` → `BadNoMatch`. Worse, the test silently **skipped** on that
failure (the exception string contained "unreachable"), so a broken adapter
was reported as green. Fix: resolve `Plc` via the declared namespace
URI → index read at connect time, with browse-name fallback, and make the
OPC UA gate **fail-fast** (a skipped gate is not a pass).

## 11. Silent skip → fail-fast gate policy

Industrial gates declare their dependencies (`IVQC_REQUIRE_SIMULATORS=1`): a
missing simulator now **fails** the gate instead of skipping. Test
classification is explicit: `unit / integration / opcua / industrial-e2e`.
CI never fabricates success by skipping what it cannot run; GPU/OPC UA/
industrial gates are documented **local** gates.

## 12. Deployment / model / dataset versions are distinct

`deployment_version` = the online AI stack (vision stack 2026.08.1);
`model_version` = the model that judged a sample; `dataset_version` = the
training data identity (manifest `dataset_version_for_model`). A training
candidate manifest carries all three (`source_deployment_version`,
`source_model_version`, `source_dataset_version`) — a regression fixed an
earlier design that conflated dataset identity with deployment identity.

## 13. Copilot is evidence-first and read-only by construction

The Quality Copilot has **zero write tools** (a fixed 15-tool read-only
allowlist, no `execute_sql`), a bounded tool loop (≤6 calls, ≤3 turns,
deadline), and a deterministic numeric grounding validator that strips any
number not present in the tool evidence (`unsupported claims = 0`). Prompt
injection in DB fields is treated as data; "release this product" produces
analysis, never an action. Safety is structural (no write capability), not
just prompt-level.
