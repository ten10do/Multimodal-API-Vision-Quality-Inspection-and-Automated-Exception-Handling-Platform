# Steel PatchCore — Representation Diagnostics (offline)

- Schema: `steel_patchcore_representation_diagnostics_v1`
- Commit: `f2a1de8481c3a8ee6f958300507676c94dfed415`
- Holdout access: `0`

## 1. Raw nearest-bank distance distributions

| Role | n | A0 median | A0 p99 | A0 max | patch-mean median | patch-mean p99 | patch-p99 median |
|---|---|---|---|---|---|---|---|
| train_normal | 4721 | 0.364893 | 0.453317 | 0.490039 | 0.175226 | 0.270296 | 0.292084 |
| validation_normal | 590 | 0.365912 | 0.451142 | 0.462758 | 0.176571 | 0.275379 | 0.295035 |
| recovery_dev_anomaly | 3333 | 0.359619 | 0.440421 | 0.477310 | 0.181639 | 0.249691 | 0.283445 |

## 2. Median ordering

- anomaly − validation-normal A0 median = -0.006293

## 3. Memory-bank coverage audit

- train-normal median patch-mean distance = 0.175226
- validation-normal median patch-mean distance = 0.176571
- validation − train gap = 0.001344
- validation p95 patch-mean = 0.231858

Per-tile median raw max (tiles 0..6):

- train: 0.274362, 0.282600, 0.294981, 0.305393, 0.293123, 0.287247, 0.287783
- validation: 0.272070, 0.281934, 0.300695, 0.313080, 0.291500, 0.281358, 0.285127

## 4. Defect-size correlates

- area ratio vs A0 Spearman rho = 0.3760814971416684
- max component area vs A0 Spearman rho = 0.3874366959325038
- quartile A0 medians: {Q1: 0.345825, Q2: 0.350853, Q3: 0.358869, Q4: 0.394201}

