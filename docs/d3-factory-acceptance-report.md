# D3 Factory Acceptance Test

Verdict: **`FACTORY_ACCEPTANCE_PASS`**

- Candidate: `steel-patchcore-d3-candidate@1.3.0-candidate.1`
- Artifact: `d3-dual-rl3-0b148a6`
- Frozen threshold: `0.8471092581748962`

| Phase | Verdict |
|---|---|
| industrial_pipeline | PASS |
| throughput | PASS |
| plc_mes | PASS |
| drift | PASS |
| human_feedback | PASS |
| tests | PASS |

The 8-hour workload is an accelerated discrete-event simulation replaying measured candidate latencies; it is not an eight-hour wall-clock soak.

No model, artifact, threshold, feature extractor, production configuration, or deployment was changed. Drift only raises a warning and feedback does not trigger retraining.
