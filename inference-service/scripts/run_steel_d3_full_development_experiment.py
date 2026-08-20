"""D3 full-development confirmation experiment (frozen D3, full dev split).

Reproduces the frozen D3 method (DINOv2 ViT-B/14 + train-normal ZCA whitening)
on the full authorized development split. Re-fits whitening on ALL 4721 train
normals (NOT the diagnostic 1000), builds a new 50k reservoir bank, calibrates a
train-only threshold, and evaluates 590 validation normal + 3333 recovery dev
anomaly. Holdout (test_normal + recovery_holdout_anomaly) is fail-closed sealed.

Stages (resumable, per-original checkpoints): stats -> sanity -> bank -> score -> eval.
  .venv-steel/Scripts/python.exe inference-service/scripts/run_steel_d3_full_development_experiment.py
  options: --stage stats|sanity|bank|score|all (default all)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from steel_patchcore.aggregation import (  # noqa: E402
    auroc,
    distribution,
    normal_vs_quartile_auroc,
    operating_point,
)
from steel_patchcore.d3_full_development import (  # noqa: E402
    ADAPTATION_BANK_BUDGET,
    ADAPTATION_SEED,
    D3_DIAGNOSTIC_AUROC,
    D3_DIAGNOSTIC_QUARTILES,
    D3_FULL_DEV_GATE,
    D3_METHOD,
    DINO_B_WEIGHTS_SHA256,
    DIAGNOSTIC_MANIFEST_SHA256,
    RECOVERY_SPLIT_SHA256,
    SOURCE_SPLIT_SHA256,
    d3_full_development_gate_passed,
    fail_closed_membership,
    small_defect_full_dev_signal,
)
from steel_patchcore.domain_adaptation import (  # noqa: E402
    chan_update_batch,
    covariance_from_stats,
    epsilon_rule,
    whiten,
    whitening_numerical_healthy,
    whitening_sanity,
    zca_whitening_matrix,
)
from steel_patchcore.domain_representation import adapted_input_side  # noqa: E402
from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.tile import TILE_X0  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
IMG_DIR = DS / "raw/train_images"
SOURCE_SPLIT = DS / "split_manifest.json"
RECOVERY_SPLIT = DS / "recovery_split_manifest.json"
SUBSET_MANIFEST = DS / "representation_diagnostic_manifest.json"
FRZ = ROOT / "inference-service/models/steel-patchcore/bank.npz"
EXPECTED_FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"
WEIGHTS_PATH = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"

RUN_ROOT = ROOT / "model-training/runs/steel-d3-full-development"
D3F_DIR = RUN_ROOT / "D3-full-development"
STATS_PATH = D3F_DIR / "stats.npz"
WHITENING_PATH = D3F_DIR / "whitening.npz"
WHITENING_MANIFEST_PATH = D3F_DIR / "whitening_manifest.json"
SANITY_PATH = D3F_DIR / "sanity.json"
BANK_PATH = D3F_DIR / "bank.npz"
BANK_PROGRESS_PATH = D3F_DIR / "bank_progress.npz"
BANK_RNG_PATH = D3F_DIR / "bank_progress.rng"
BANK_MANIFEST_PATH = D3F_DIR / "bank_manifest.json"
SCORES_PATH = D3F_DIR / "scores.json"
RESULTS_JSON = RUN_ROOT / "results.json"
RESULTS_MD = RUN_ROOT / "results.md"

STATS_CHECKPOINT_EVERY = 50
BANK_CHECKPOINT_EVERY = 100
SCORE_CHECKPOINT_EVERY = 100
SANITY_STRIDE = 118  # deterministic uniform sample across the 4721 train IDs (~41 images)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload) -> None:
    def sanitize(v):
        if isinstance(v, dict):
            return {k: sanitize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [sanitize(val) for val in v]
        if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
            return None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_rep_module():
    script = ROOT / "inference-service/scripts/run_steel_representation_experiment.py"
    spec = importlib.util.spec_from_file_location("_repr_experiment_d3f", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dino_model(device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    return model


def extract_dino_patch_tokens_raw(model, tile):
    side = adapted_input_side(tile.shape[-1], 14)
    if tile.shape[-1] != side:
        tile = F.interpolate(tile, size=(side, side), mode="bilinear", align_corners=False)
    with torch.no_grad():
        out = model.forward_features(tile)
    return out["x_norm_patchtokens"][0]  # (N, 768)


def whiten_normalize(tokens, mean_t, w_t):
    return F.normalize((tokens - mean_t) @ w_t, p=2, dim=1)


def model_identity_report(model, device) -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    pe = getattr(model, "patch_embed", None)
    return {
        "module": type(model).__module__ + "." + type(model).__name__,
        "embed_dim": int(getattr(model, "embed_dim", -1)),
        "patch_size": int(getattr(model, "patch_size", -1)),
        "img_size": list(getattr(pe, "img_size", [])) if pe is not None else [],
        "depth": len(getattr(model, "blocks", ())),
        "num_register_tokens": int(getattr(model, "num_register_tokens", -1)),
        "gpu": gpu_name,
        "environment": {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                        "torch": torch.__version__},
    }


# --- Stage A: streaming whitening statistics (resumable) ---------------------

def accumulate_stats(model, device, train_ids) -> tuple[int, np.ndarray, np.ndarray]:
    D3F_DIR.mkdir(parents=True, exist_ok=True)
    count, mean, m2, processed = 0, np.zeros(768, dtype=np.float64), np.zeros((768, 768), dtype=np.float64), 0
    if STATS_PATH.exists():
        ck = np.load(STATS_PATH)
        count, mean, m2, processed = int(ck["count"]), ck["mean"].astype(np.float64), ck["m2"].astype(np.float64), int(ck["processed_images"])
        print(f"stats resume: processed={processed} count={count}", flush=True)
    rep = load_rep_module()
    for i in range(processed, len(train_ids)):
        for tile in rep.load_normalized_tiles(IMG_DIR / f"{train_ids[i]}.jpg", device):
            toks = extract_dino_patch_tokens_raw(model, tile).cpu().numpy().astype(np.float64)
            count, mean, m2 = chan_update_batch(count, mean, m2, toks)
        processed = i + 1
        if processed % STATS_CHECKPOINT_EVERY == 0 or processed == len(train_ids):
            np.savez(STATS_PATH, count=np.asarray(count, np.int64), mean=mean, m2=m2,
                     processed_images=np.asarray(processed, np.int64))
    return count, mean, m2


# --- Stage B: resumable reservoir bank ---------------------------------------

def _new_rand(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _reservoir_put(rng, reservoir, seen, feature, budget):
    seen += 1
    if seen <= budget:
        reservoir[seen - 1] = feature
    else:
        j = int(rng.integers(0, seen))
        if j < budget:
            reservoir[j] = feature
    return seen


def build_bank(model, device, train_ids, mean_t, w_t) -> tuple[np.ndarray, int]:
    D3F_DIR.mkdir(parents=True, exist_ok=True)
    budget = ADAPTATION_BANK_BUDGET
    reservoir = np.zeros((budget, 768), dtype=np.float32)
    seen = 0
    rng = _new_rand(ADAPTATION_SEED)
    processed = 0
    if BANK_PROGRESS_PATH.exists() and BANK_RNG_PATH.exists():
        d = np.load(BANK_PROGRESS_PATH)
        reservoir = d["reservoir"].astype(np.float32)
        seen = int(d["seen"])
        processed = int(d["processed_images"])
        rng = _new_rand(ADAPTATION_SEED)
        rng.bit_generator.state = pickle.loads(BANK_RNG_PATH.read_bytes())
        print(f"bank resume: processed={processed} seen={seen}", flush=True)

    rep = load_rep_module()
    for i in range(processed, len(train_ids)):
        for tile in rep.load_normalized_tiles(IMG_DIR / f"{train_ids[i]}.jpg", device):
            toks = whiten_normalize(extract_dino_patch_tokens_raw(model, tile), mean_t, w_t).cpu().numpy().astype(np.float32)
            for f in toks:
                seen = _reservoir_put(rng, reservoir, seen, f, budget)
        processed = i + 1
        if processed % BANK_CHECKPOINT_EVERY == 0 or processed == len(train_ids):
            np.savez(BANK_PROGRESS_PATH, reservoir=reservoir, seen=np.asarray(seen, np.int64),
                     processed_images=np.asarray(processed, np.int64))
            BANK_RNG_PATH.write_bytes(pickle.dumps(rng.bit_generator.state))
    return reservoir, seen


# --- Stage C/D/E: scores (resumable) -----------------------------------------

def score_role(model, device, bank_t, mean_t, w_t, ids, role_key, existing) -> dict:
    rep = load_rep_module()
    scores = dict(existing)
    for idx, image_id in enumerate(ids):
        if image_id in scores:
            continue
        tile_scores = []
        for tile in rep.load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device):
            emb = whiten_normalize(extract_dino_patch_tokens_raw(model, tile), mean_t, w_t)
            sim = emb @ bank_t.T
            dist = 1.0 - sim.max(dim=1).values
            tile_scores.append(float(dist.max()))
        scores[image_id] = max(tile_scores)
        if (idx + 1) % SCORE_CHECKPOINT_EVERY == 0:
            yield scores, False
    yield scores, True


def evaluate(train_scores_arr, val_scores_arr, dev_scores_arr, quartiles) -> dict:
    normal = np.asarray(val_scores_arr, dtype=np.float64)
    anomaly = np.asarray(dev_scores_arr, dtype=np.float64)
    threshold = float(np.max(train_scores_arr))
    image_auroc = auroc(
        np.concatenate([normal, anomaly]),
        np.concatenate([np.zeros(normal.size, dtype=int), np.ones(anomaly.size, dtype=int)]),
    )
    op = operating_point(normal, anomaly, threshold)
    q_arr = np.asarray(quartiles, dtype=int)
    q_rows = []
    for q in (1, 2, 3, 4):
        q_scores = anomaly[q_arr == q]
        q_rows.append({"quartile": q, "count": int(q_scores.size),
                       "normal_vs_quartile_auroc": normal_vs_quartile_auroc(normal, q_scores)})
    return {
        "threshold": threshold,
        "image_auroc": image_auroc,
        "normal_distribution": distribution(normal),
        "anomaly_distribution": distribution(anomaly),
        "anomaly_minus_normal_median": float(np.median(anomaly) - np.median(normal)),
        "operating_point": op,
        "quartiles": q_rows,
    }


def render_markdown(payload: dict, path: Path) -> None:
    L = []
    A = L.append
    A("# Steel PatchCore — D3 Full-Development Confirmation Results")
    A("")
    A(f"Verdict: `{payload['verdict']}`")
    A(f"- Small-defect full-dev signal: `{payload['small_defect_full_dev_signal']}`")
    A(f"- Canonical PatchCore status: `{payload['canonical_patchcore_status']}`")
    A(f"- Holdout access count: `{payload['holdout_access_count']}`")
    A("")
    A("## Diagnostic vs Full comparison")
    A("")
    m = payload["full"]["metrics"]
    A(f"- Diagnostic D3 AUROC = {payload['diagnostic']['auroc']:.4f}")
    A(f"- Full D3 AUROC = {m['image_auroc']:.4f}  (Δ diagnostic−full {payload['full']['diagnostic_minus_full']:+.4f})")
    A(f"- Gate: `{payload['gate']}`")
    A("")
    A("| | AUROC | Q1 | Q2 | Q3 | Q4 |")
    A("|---|---|---|---|---|---|")
    q = m["quartiles"]
    A(f"| Diagnostic D3 | {payload['diagnostic']['auroc']:.4f} | {payload['diagnostic']['quartiles']['Q1']:.4f} | {payload['diagnostic']['quartiles']['Q2']:.4f} | {payload['diagnostic']['quartiles']['Q3']:.4f} | {payload['diagnostic']['quartiles']['Q4']:.4f} |")
    A(f"| Full D3 | {m['image_auroc']:.4f} | {q[0]['normal_vs_quartile_auroc']:.4f} | {q[1]['normal_vs_quartile_auroc']:.4f} | {q[2]['normal_vs_quartile_auroc']:.4f} | {q[3]['normal_vs_quartile_auroc']:.4f} |")
    A("")
    A("## Train calibration + diagnostic operating point")
    A("")
    td = payload["train_distribution"]
    A(f"- threshold = {m['threshold']:.6f} (max of {td['n']} train-normal scores)")
    A(f"- train score distribution: min {td['min']:.5f} / p50 {td['p50']:.5f} / p95 {td['p95']:.5f} / p99 {td['p99']:.5f} / max {td['max']:.5f}")
    op = m["operating_point"]
    A(f"- TP={op['tp']} TN={op['tn']} FP={op['fp']} FN={op['fn']}")
    A(f"- Precision={op['precision']:.4f} Recall={op['recall']:.4f} F1={op['f1']:.4f}")
    A(f"- Normal FPR={op['normal_fpr']:.4f} Anomaly Recall={op['anomaly_recall']:.4f}")
    A(f"- anomaly median − normal median = {m['anomaly_minus_normal_median']:.6f}")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stats", "sanity", "bank", "score", "all"], default="all")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("D3_FULL_DEV_REQUIRES_GPU")
    device = torch.device("cuda:0")

    from steel_patchcore.lifecycle import lifecycle_enter

    blocked = lifecycle_enter("d3_full_development", "experimental_banks", RUN_ROOT / "checkpoint.json", RUN_ROOT)
    if blocked is not None:
        return int(blocked)

    if sha256_file(FRZ) != EXPECTED_FROZEN_BANK_SHA:
        raise RuntimeError("FROZEN_BANK_SHA_MISMATCH")
    if sha256_file(SOURCE_SPLIT) != SOURCE_SPLIT_SHA256:
        raise RuntimeError("SOURCE_SPLIT_SHA_MISMATCH")
    if sha256_file(RECOVERY_SPLIT) != RECOVERY_SPLIT_SHA256:
        raise RuntimeError("RECOVERY_SPLIT_SHA_MISMATCH")
    if sha256_file(SUBSET_MANIFEST) != DIAGNOSTIC_MANIFEST_SHA256:
        raise RuntimeError("DIAGNOSTIC_MANIFEST_SHA_MISMATCH")
    if sha256_file(WEIGHTS_PATH) != DINO_B_WEIGHTS_SHA256:
        raise RuntimeError("DINO_WEIGHTS_SHA_MISMATCH")

    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))["splits"]
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    train_ids = list(source["train_normal"])
    val_ids = list(source["validation_normal"])
    test_normal_ids = list(source["test_normal"])
    dev_ids = list(recovery["recovery_dev_anomaly"])
    holdout_ids = list(recovery["recovery_holdout_anomaly"])

    ok, reason = fail_closed_membership(train_ids, val_ids, dev_ids, test_normal_ids, holdout_ids)
    if not ok:
        raise RuntimeError(f"HOLDOUT_FAIL_CLOSED: {reason}")

    model = build_dino_model(device)
    identity = model_identity_report(model, device)

    run_stages = {args.stage} if args.stage != "all" else {"stats", "sanity", "bank", "score"}

    # ---- stats ----
    if "stats" in run_stages and not WHITENING_PATH.exists():
        t0 = time.time()
        count, mean, m2 = accumulate_stats(model, device, train_ids)
        cov = covariance_from_stats(m2, count)
        eps = epsilon_rule(cov)
        cov_reg = cov + eps * np.eye(768, dtype=np.float64)
        w_64, eigenvalues = zca_whitening_matrix(cov_reg)
        D3F_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(WHITENING_PATH, mean=mean, covariance=cov, eigenvalues=eigenvalues,
                 epsilon=np.asarray(eps, np.float64), whitening_matrix=w_64, count=np.asarray(count, np.int64))
        atomic_write_json(WHITENING_MANIFEST_PATH, {
            "candidate_id": "D3-full",
            "adaptation": "train-normal ZCA covariance whitening",
            "protocol_version": "d3_full_development_protocol_v1",
            "count": int(count),
            "dim": int(mean.shape[0]),
            "epsilon": eps,
            "epsilon_rule": "1e-6 * trace(cov) / d",
            "eigenvalue_min": float(eigenvalues.min()),
            "eigenvalue_max": float(eigenvalues.max()),
            "condition_number": float(eigenvalues.max() / eigenvalues.min()),
            "whitening_sha256": sha256_file(WHITENING_PATH),
            "dino_weights_sha256": DINO_B_WEIGHTS_SHA256,
            "source_split_sha256": SOURCE_SPLIT_SHA256,
            "recovery_split_sha256": RECOVERY_SPLIT_SHA256,
            "source": "train_normal_subset only (full 4721)",
            "source_count": len(train_ids),
            "created_at": utc_now(),
        })
        print(f"stats done: count={count} eps={eps:.3e} cond={eigenvalues.max()/eigenvalues.min():.3e} "
              f"whitening_sha={sha256_file(WHITENING_PATH)[:12]} in {time.time()-t0:.0f}s", flush=True)

    if not WHITENING_PATH.exists():
        raise RuntimeError("whitening artifact missing; run --stage stats first")
    wz = np.load(WHITENING_PATH)
    mean = wz["mean"].astype(np.float64)
    w_64 = wz["whitening_matrix"].astype(np.float64)
    eigenvalues = wz["eigenvalues"].astype(np.float64)
    mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
    w_t = torch.from_numpy(w_64.astype(np.float32)).to(device)

    # ---- sanity (pre-L2) ----
    if "sanity" in run_stages and not SANITY_PATH.exists():
        sample_ids = train_ids[::SANITY_STRIDE]
        rep = load_rep_module()
        samples = []
        for sid in sample_ids:
            for tile in rep.load_normalized_tiles(IMG_DIR / f"{sid}.jpg", device):
                toks = extract_dino_patch_tokens_raw(model, tile).cpu().numpy().astype(np.float64)
                samples.append(whiten(toks, mean, w_64))
        whitened = np.concatenate(samples, axis=0)
        sanity = whitening_sanity(whitened)
        sanity["sample_strategy"] = f"deterministic uniform stride {SANITY_STRIDE} over {len(train_ids)} train ids"
        sanity["sample_original_ids"] = int(len(sample_ids))
        sanity["whitened_sample_shape"] = list(whitened.shape)
        healthy, hreason = whitening_numerical_healthy(sanity, eigenvalues)
        sanity["numerical_healthy"] = bool(healthy)
        sanity["numerical_health_reason"] = hreason
        atomic_write_json(SANITY_PATH, sanity)
        print("sanity done:", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in sanity.items()}), flush=True)
        if not healthy:
            verdict = "D3_FULL_DEV_NUMERICAL_BLOCKED"
            atomic_write_json(RESULTS_JSON, {
                "schema_version": "steel_patchcore_d3_full_development_results_v1",
                "verdict": verdict, "numerical_blocked_reason": hreason,
                "whitening_manifest": json.loads(WHITENING_MANIFEST_PATH.read_text(encoding="utf-8")),
                "sanity": sanity, "holdout_access_count": 0, "generated_at": utc_now(),
            })
            RESULTS_MD.write_text(f"# D3 Full-Development\n\nVerdict: `{verdict}`\n\nReason: `{hreason}`\n",
                                  encoding="utf-8")
            print(verdict)
            return 0

    # ---- bank (resumable) ----
    if "bank" in run_stages and not BANK_PATH.exists():
        t0 = time.time()
        bank, seen = build_bank(model, device, train_ids, mean_t, w_t)
        np.savez_compressed(BANK_PATH, features=bank.astype(np.float32))
        bank_sha = sha256_file(BANK_PATH)
        atomic_write_json(BANK_MANIFEST_PATH, {
            "candidate_id": "D3-full",
            "representation": "DINOv2 ViT-B/14 patch tokens, full train-normal ZCA whitened + per-patch L2",
            "dim": int(bank.shape[1]), "rows": int(bank.shape[0]), "dtype": "float32",
            "sampling": "reservoir Algorithm R", "seed": ADAPTATION_SEED, "budget": ADAPTATION_BANK_BUDGET,
            "candidate_patches": seen, "source": "train_normal (full 4721)", "source_count": len(train_ids),
            "bank_sha256": bank_sha, "whitening_sha256": sha256_file(WHITENING_PATH),
            "dino_weights_sha256": DINO_B_WEIGHTS_SHA256, "source_split_sha256": SOURCE_SPLIT_SHA256,
            "recovery_split_sha256": RECOVERY_SPLIT_SHA256, "created_at": utc_now(),
        })
        print(f"bank built: rows={bank.shape[0]} dim={bank.shape[1]} candidate_patches={seen} "
              f"sha={bank_sha[:12]} in {time.time()-t0:.0f}s", flush=True)

    if args.stage in ("stats", "sanity"):
        return 0
    if not BANK_PATH.exists():
        raise RuntimeError("missing bank; run --stage bank first")
    bank = np.load(BANK_PATH)["features"].astype(np.float32)
    bank_t = torch.from_numpy(bank).to(device)

    # ---- scores (resumable) ----
    if "score" in run_stages:
        cached = json.loads(SCORES_PATH.read_text(encoding="utf-8")) if SCORES_PATH.exists() else {"train_normal": {}, "validation_normal": {}, "dev_anomaly": {}}
        if not SCORES_PATH.exists():
            atomic_write_json(SCORES_PATH, cached)

        def run_role(role_key, ids):
            sc = dict(cached.get(role_key, {}))
            for snapshot, done in score_role(model, device, bank_t, mean_t, w_t, ids, role_key, sc):
                cached[role_key] = snapshot
                atomic_write_json(SCORES_PATH, cached)
                if done:
                    break

        run_role("train_normal", train_ids)
        run_role("validation_normal", val_ids)
        run_role("dev_anomaly", dev_ids)

    if "score" not in run_stages and not SCORES_PATH.exists():
        raise RuntimeError("missing scores; run --stage score first")
    cached = json.loads(SCORES_PATH.read_text(encoding="utf-8"))

    if args.stage == "bank":
        print("bank stage complete (no evaluation yet)")
        return 0

    # ---- evaluate + gate ----
    train_scores = np.asarray([cached["train_normal"][i] for i in train_ids], dtype=np.float64)
    val_scores = np.asarray([cached["validation_normal"][i] for i in val_ids], dtype=np.float64)
    dev_scores = np.asarray([cached["dev_anomaly"][i] for i in dev_ids], dtype=np.float64)
    ratios, quartiles = load_rep_module().load_area_ratios(dev_ids)
    ev = evaluate(train_scores, val_scores, dev_scores, quartiles)
    q = {row["quartile"]: row["normal_vs_quartile_auroc"] for row in ev["quartiles"]}
    passed = d3_full_development_gate_passed(ev["image_auroc"], np.median(dev_scores), np.median(val_scores))
    small = small_defect_full_dev_signal(q[1])
    verdict = "D3_FULL_DEVELOPMENT_CONFIRMED" if passed else "D3_FULL_DEVELOPMENT_FAILED"
    regression = D3_DIAGNOSTIC_AUROC - ev["image_auroc"]

    payload = {
        "schema_version": "steel_patchcore_d3_full_development_results_v1",
        "protocol_version": "d3_full_development_protocol_v1",
        "verdict": verdict,
        "gate": D3_FULL_DEV_GATE,
        "small_defect_full_dev_signal": bool(small),
        "development_scale_regression": regression,
        "regression_flagged": regression > 0.05,
        "diagnostic": {"candidate": "D3", "auroc": D3_DIAGNOSTIC_AUROC, "quartiles": D3_DIAGNOSTIC_QUARTILES,
                       "source": "frozen diagnostic results (1000/300/1000), not re-run"},
        "full": {
            "method": D3_METHOD,
            "runtime_identity": identity,
            "whitening_manifest": json.loads(WHITENING_MANIFEST_PATH.read_text(encoding="utf-8")),
            "sanity": json.loads(SANITY_PATH.read_text(encoding="utf-8")) if SANITY_PATH.exists() else None,
            "bank_sha256": sha256_file(BANK_PATH),
            "metrics": ev,
            "diagnostic_minus_full": regression,
            "gate_passed": bool(passed),
        },
        "train_distribution": distribution(train_scores),
        "data_counts": {"train_normal": len(train_ids), "validation_normal": len(val_ids), "dev_anomaly": len(dev_ids),
                        "test_normal_forbidden": len(test_normal_ids), "recovery_holdout_forbidden": len(holdout_ids)},
        "canonical_patchcore_status": "CANONICAL_PATCHCORE_REFERENCE_BLOCKED",
        "holdout_access_count": 0,
        "generated_at": utc_now(),
    }
    atomic_write_json(RESULTS_JSON, payload)
    render_markdown(payload, RESULTS_MD)
    print(json.dumps({
        "verdict": verdict,
        "full_auroc": round(ev["image_auroc"], 4),
        "diagnostic_auroc": D3_DIAGNOSTIC_AUROC,
        "diagnostic_minus_full": round(regression, 4),
        "anomaly_median": round(float(np.median(dev_scores)), 5),
        "normal_median": round(float(np.median(val_scores)), 5),
        "Q1": round(q[1], 4), "Q2": round(q[2], 4), "Q3": round(q[3], 4), "Q4": round(q[4], 4),
        "small_defect_full_dev_signal": bool(small),
    }, default=str))
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())