"""Steel-domain adaptation experiment (D3 = DINOv2 ViT-B/14 + train-normal ZCA).

Changes ONLY the representation via train-normal covariance whitening on top of
frozen DINOv2 ViT-B/14. Everything else frozen: diagnostic subset, reservoir
50k/seed 42, 7 tiles + A0 max, train-only threshold, per-patch L2 + cosine 1-NN.
No holdout, no baseline/D1/D2 bank write, no fine-tuning, no bank-strategy change.

Stages (resumable): stats -> sanity -> bank -> score -> evaluate.
  .venv-steel/Scripts/python.exe inference-service/scripts/run_steel_domain_adaptation_experiment.py
  options: --stage stats|sanity|bank|score|all (default all)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
from steel_patchcore.domain_adaptation import (  # noqa: E402
    ADAPTATION_BANK_BUDGET,
    ADAPTATION_GATE,
    ADAPTATION_SEED,
    D0_AUROC,
    D0_QUARTILES,
    D1_AUROC,
    D1_QUARTILES,
    D2_AUROC,
    D2_QUARTILES,
    D2_REFERENCE,
    adaptation_gate_passed,
    adaptation_strong_signal,
    chan_update_batch,
    covariance_from_stats,
    epsilon_rule,
    small_defect_adaptation_signal,
    whiten,
    whitening_numerical_healthy,
    whitening_sanity,
    zca_whitening_matrix,
)
from steel_patchcore.domain_representation import adapted_input_side  # noqa: E402
from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.representation import reservoir_from_stream  # noqa: E402
from steel_patchcore.tile import TILE_X0  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
IMG_DIR = DS / "raw/train_images"
SUBSET_MANIFEST = DS / "representation_diagnostic_manifest.json"
SUBSET_MANIFEST_SHA = DS / "representation_diagnostic_manifest.sha256"
FRZ = ROOT / "inference-service/models/steel-patchcore/bank.npz"
EXPECTED_FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"

RUN_ROOT = ROOT / "model-training/runs/steel-domain-adaptation"
D3_DIR = RUN_ROOT / "D3-dinov2-b14-zca"
STATS_PATH = D3_DIR / "stats.npz"
WHITENING_PATH = D3_DIR / "whitening.npz"
WHITENING_MANIFEST_PATH = D3_DIR / "whitening_manifest.json"
SANITY_PATH = D3_DIR / "sanity.json"
BANK_PATH = D3_DIR / "bank.npz"
BANK_MANIFEST_PATH = D3_DIR / "bank_manifest.json"
SCORES_PATH = D3_DIR / "scores.json"
RESULTS_JSON = RUN_ROOT / "results.json"
RESULTS_MD = RUN_ROOT / "results.md"

SANITY_IMAGES = 20
STATS_CHECKPOINT_EVERY = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict) -> None:
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
    spec = importlib.util.spec_from_file_location("_repr_experiment_d3", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dino_model(device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    return model


def extract_dino_patch_tokens_raw(model: torch.nn.Module, tile: torch.Tensor) -> torch.Tensor:
    """(1,3,256,256) tile -> (N, 768) raw patch tokens (no L2; whitening is applied first)."""
    side = adapted_input_side(tile.shape[-1], 14)
    if tile.shape[-1] != side:
        tile = F.interpolate(tile, size=(side, side), mode="bilinear", align_corners=False)
    with torch.no_grad():
        out = model.forward_features(tile)
    return out["x_norm_patchtokens"][0]  # (N, 768)


def whiten_normalize(tokens: torch.Tensor, mean_t: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
    """train-normal centering + ZCA whitening + per-patch L2 (all on GPU)."""
    return F.normalize((tokens - mean_t) @ w_t, p=2, dim=1)


def model_identity_report(model, device) -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    patch_embed = getattr(model, "patch_embed", None)
    num_heads = -1
    if hasattr(model, "blocks") and len(model.blocks):
        attn = getattr(model.blocks[0], "attn", None)
        if attn is not None:
            num_heads = int(getattr(attn, "num_heads", -1))
    return {
        "module": type(model).__module__ + "." + type(model).__name__,
        "embed_dim": int(getattr(model, "embed_dim", -1)),
        "patch_size": int(getattr(model, "patch_size", -1)),
        "img_size": list(getattr(patch_embed, "img_size", [])) if patch_embed is not None else [],
        "depth": len(getattr(model, "blocks", ())),
        "num_heads": num_heads,
        "num_register_tokens": int(getattr(model, "num_register_tokens", -1)),
        "num_cls_tokens": 1,
        "gpu": gpu_name,
        "environment": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch": torch.__version__,
        },
    }


# --- streaming statistics (resumable) ----------------------------------------

def accumulate_stats(model, device, train_ids) -> tuple[int, np.ndarray, np.ndarray]:
    D3_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    mean = np.zeros(768, dtype=np.float64)
    m2 = np.zeros((768, 768), dtype=np.float64)
    processed = 0
    if STATS_PATH.exists():
        ck = np.load(STATS_PATH)
        count = int(ck["count"])
        mean = ck["mean"].astype(np.float64)
        m2 = ck["m2"].astype(np.float64)
        processed = int(ck["processed_images"])
        print(f"stats resume: processed={processed} count={count}", flush=True)

    rep = load_rep_module()
    for i in range(processed, len(train_ids)):
        image_id = train_ids[i]
        for tile in rep.load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device):
            toks = extract_dino_patch_tokens_raw(model, tile).cpu().numpy().astype(np.float64)
            count, mean, m2 = chan_update_batch(count, mean, m2, toks)
        processed = i + 1
        if processed % STATS_CHECKPOINT_EVERY == 0 or processed == len(train_ids):
            np.savez(STATS_PATH, count=np.asarray(count, dtype=np.int64), mean=mean, m2=m2,
                     processed_images=np.asarray(processed, dtype=np.int64))
    return count, mean, m2


# --- sanity (resumable) ------------------------------------------------------

def run_sanity(model, device, train_ids, mean, w_64) -> dict:
    if SANITY_PATH.exists():
        return json.loads(SANITY_PATH.read_text(encoding="utf-8"))

    rep = load_rep_module()
    samples = []
    for image_id in train_ids[:SANITY_IMAGES]:
        for tile in rep.load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device):
            toks = extract_dino_patch_tokens_raw(model, tile).cpu().numpy().astype(np.float64)
            samples.append(whiten(toks, mean, w_64))
    whitened = np.concatenate(samples, axis=0)
    sanity = whitening_sanity(whitened)
    sanity["whitened_sample_shape"] = list(whitened.shape)
    atomic_write_json(SANITY_PATH, sanity)
    return sanity


# --- bank (resumable) --------------------------------------------------------

def build_bank(model, device, train_ids, mean_t, w_t) -> tuple[np.ndarray, int]:
    rep = load_rep_module()

    def stream():
        for image_id in train_ids:
            for tile in rep.load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device):
                yield whiten_normalize(extract_dino_patch_tokens_raw(model, tile), mean_t, w_t).cpu().numpy().astype(np.float32)

    bank, seen = reservoir_from_stream(stream(), ADAPTATION_BANK_BUDGET, ADAPTATION_SEED)
    return bank, seen


def score_images(model, device, bank_t, mean_t, w_t, image_ids) -> list[float]:
    rep = load_rep_module()
    scores: list[float] = []
    for image_id in image_ids:
        tile_scores: list[float] = []
        for tile in rep.load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device):
            emb = whiten_normalize(extract_dino_patch_tokens_raw(model, tile), mean_t, w_t)  # (N,768)
            sim = emb @ bank_t.T
            dist = 1.0 - sim.max(dim=1).values
            tile_scores.append(float(dist.max()))
        scores.append(max(tile_scores))
    return scores


def evaluate(train_scores, val_scores, dev_scores, quartiles) -> dict:
    normal = np.asarray(val_scores, dtype=np.float64)
    anomaly = np.asarray(dev_scores, dtype=np.float64)
    threshold = float(np.max(train_scores))
    image_auroc = auroc(
        np.concatenate([normal, anomaly]),
        np.concatenate([np.zeros(normal.size, dtype=int), np.ones(anomaly.size, dtype=int)]),
    )
    op = operating_point(normal, anomaly, threshold)
    q_arr = np.asarray(quartiles, dtype=int)
    q_rows = []
    for q in (1, 2, 3, 4):
        q_scores = anomaly[q_arr == q]
        q_rows.append({
            "quartile": q,
            "count": int(q_scores.size),
            "normal_vs_quartile_auroc": normal_vs_quartile_auroc(normal, q_scores),
        })
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
    L: list[str] = []
    A = L.append
    A("# Steel PatchCore — Steel-Domain Adaptation Results (D0 WRN / D1 S / D2 B / D3 B+ZCA)")
    A("")
    A(f"Verdict: `{payload['verdict']}`")
    A(f"- Strong signal: `{payload['strong_signal']}`")
    A(f"- Small-defect adaptation signal: `{payload['small_defect_adaptation_signal']}`")
    A(f"- Canonical PatchCore status: `{payload['canonical_patchcore_status']}`")
    A(f"- Holdout access count: `{payload['holdout_access_count']}`")
    A("")
    A("## D0 / D1 / D2 / D3 comparison")
    A("")
    m = payload["d3"]["metrics"]
    A(f"- D0 WRN S2 AUROC = {payload['d0']['image_auroc']:.4f}")
    A(f"- D1 DINOv2-S/14 AUROC = {payload['d1']['image_auroc']:.4f}")
    A(f"- D2 DINOv2-B/14 AUROC = {payload['d2']['image_auroc']:.4f}")
    A(f"- D3 DINOv2-B/14 + ZCA AUROC = {m['image_auroc']:.4f}  (Δ vs D2 {payload['d3']['delta_vs_d2']:+.4f}, Δ vs D0 {payload['d3']['delta_vs_d0']:+.4f})")
    A(f"- Gate: `{payload['gate']}`")
    A("")
    A("| | AUROC | Q1 | Q2 | Q3 | Q4 |")
    A("|---|---|---|---|---|---|")
    q = m["quartiles"]
    A(f"| D0 | {payload['d0']['image_auroc']:.4f} | {payload['d0']['quartiles']['Q1']:.4f} | {payload['d0']['quartiles']['Q2']:.4f} | {payload['d0']['quartiles']['Q3']:.4f} | {payload['d0']['quartiles']['Q4']:.4f} |")
    A(f"| D1 | {payload['d1']['image_auroc']:.4f} | {payload['d1']['quartiles']['Q1']:.4f} | {payload['d1']['quartiles']['Q2']:.4f} | {payload['d1']['quartiles']['Q3']:.4f} | {payload['d1']['quartiles']['Q4']:.4f} |")
    A(f"| D2 | {payload['d2']['image_auroc']:.4f} | {payload['d2']['quartiles']['Q1']:.4f} | {payload['d2']['quartiles']['Q2']:.4f} | {payload['d2']['quartiles']['Q3']:.4f} | {payload['d2']['quartiles']['Q4']:.4f} |")
    A(f"| D3 | {m['image_auroc']:.4f} | {q[0]['normal_vs_quartile_auroc']:.4f} | {q[1]['normal_vs_quartile_auroc']:.4f} | {q[2]['normal_vs_quartile_auroc']:.4f} | {q[3]['normal_vs_quartile_auroc']:.4f} |")
    A("")
    A("## Diagnostic operating point (train-only threshold)")
    A("")
    op = m["operating_point"]
    A(f"- threshold = {m['threshold']:.6f}")
    A(f"- TP={op['tp']} TN={op['tn']} FP={op['fp']} FN={op['fn']}")
    A(f"- Precision={op['precision']:.4f} Recall={op['recall']:.4f} F1={op['f1']:.4f}")
    A(f"- Normal FPR={op['normal_fpr']:.4f} Anomaly Recall={op['anomaly_recall']:.4f}")
    A(f"- anomaly median - normal median = {m['anomaly_minus_normal_median']:.6f}")
    A("")
    A("## Score distributions")
    A("")
    A("| role | n | min | p50 | p95 | p99 | max |")
    A("|---|---|---|---|---|---|---|")
    for role, key in (("normal", "normal_distribution"), ("anomaly", "anomaly_distribution")):
        d = m[key]
        A(f"| {role} | {d['n']} | {d['min']:.5f} | {d['p50']:.5f} | {d['p95']:.5f} | {d['p99']:.5f} | {d['max']:.5f} |")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stats", "sanity", "bank", "score", "all"], default="all")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DOMAIN_ADAPTATION_EXPERIMENT_REQUIRES_GPU")
    device = torch.device("cuda:0")

    from steel_patchcore.lifecycle import lifecycle_enter

    blocked = lifecycle_enter("domain_adaptation_d3", "experimental_banks",
                              RUN_ROOT / "checkpoint.json", RUN_ROOT)
    if blocked is not None:
        return int(blocked)

    if sha256_file(FRZ) != EXPECTED_FROZEN_BANK_SHA:
        raise RuntimeError("FROZEN_BANK_SHA_MISMATCH")
    if sha256_file(SUBSET_MANIFEST) != SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip():
        raise RuntimeError("SUBSET_MANIFEST_SHA_MISMATCH")

    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    train_ids = list(manifest["train_normal_subset"])
    val_ids = list(manifest["validation_normal_subset"])
    dev_ids = list(manifest["recovery_dev_anomaly_subset"])
    assert manifest["holdout_access_count"] == 0

    model = build_dino_model(device)
    identity = model_identity_report(model, device)

    run_stages = {args.stage} if args.stage != "all" else {"stats", "sanity", "bank", "score"}

    # ---- stats ----
    if "stats" in run_stages and not WHITENING_PATH.exists():
        t0 = time.time()
        count, mean, m2 = accumulate_stats(model, device, train_ids)
        cov = covariance_from_stats(m2, count)
        eps = epsilon_rule(cov)
        cov_reg = cov + eps * np.eye(cov.shape[0], dtype=np.float64)
        w_64, eigenvalues = zca_whitening_matrix(cov_reg)
        D3_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(WHITENING_PATH, mean=mean, covariance=cov, covariance_reg=cov_reg,
                 eigenvalues=eigenvalues, epsilon=np.asarray(eps, dtype=np.float64),
                 whitening_matrix=w_64, count=np.asarray(count, dtype=np.int64))
        whitening_sha = sha256_file(WHITENING_PATH)
        atomic_write_json(WHITENING_MANIFEST_PATH, {
            "candidate_id": "D3",
            "adaptation": "train-normal ZCA covariance whitening",
            "protocol_version": "domain_adaptation_protocol_v1",
            "mean_sha": None,
            "count": int(count),
            "dim": int(mean.shape[0]),
            "epsilon": eps,
            "epsilon_rule": "1e-6 * trace(cov) / d",
            "eigenvalue_min": float(eigenvalues.min()),
            "eigenvalue_max": float(eigenvalues.max()),
            "condition_number": float(eigenvalues.max() / eigenvalues.min()),
            "whitening_sha256": whitening_sha,
            "dino_weights_sha256": D2_REFERENCE["weights_sha256"],
            "diagnostic_manifest_sha256": SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip(),
            "source": "train_normal_subset only",
            "source_count": len(train_ids),
            "created_at": utc_now(),
        },)
        print(f"stats done: count={count} eps={eps:.3e} cond={eigenvalues.max()/eigenvalues.min():.3e} "
              f"whitening_sha={whitening_sha[:12]} in {time.time()-t0:.0f}s", flush=True)

    if not WHITENING_PATH.exists():
        raise RuntimeError("whitening artifact missing; run --stage stats first")
    wz = np.load(WHITENING_PATH)
    mean = wz["mean"].astype(np.float64)
    w_64 = wz["whitening_matrix"].astype(np.float64)
    eigenvalues = wz["eigenvalues"].astype(np.float64)
    mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
    w_t = torch.from_numpy(w_64.astype(np.float32)).to(device)

    # ---- sanity + numerical gate ----
    if "sanity" in run_stages:
        if not SANITY_PATH.exists():
            sanity = run_sanity(model, device, train_ids, mean, w_64)
            print("sanity done:", json.dumps(sanity), flush=True)
        sanity = json.loads(SANITY_PATH.read_text(encoding="utf-8"))
        healthy, reason = whitening_numerical_healthy(sanity, eigenvalues)
        if not healthy:
            verdict = "ADAPTATION_NUMERICAL_BLOCKED"
            payload = {
                "schema_version": "steel_patchcore_domain_adaptation_results_v1",
                "protocol_version": "domain_adaptation_protocol_v1",
                "verdict": verdict,
                "numerical_blocked_reason": reason,
                "whitening_manifest": json.loads(WHITENING_MANIFEST_PATH.read_text(encoding="utf-8")),
                "sanity": sanity,
                "canonical_patchcore_status": "CANONICAL_PATCHCORE_REFERENCE_BLOCKED",
                "holdout_access_count": 0,
                "generated_at": utc_now(),
            }
            atomic_write_json(RESULTS_JSON, payload)
            RESULTS_MD.write_text(
                "# Steel PatchCore — Steel-Domain Adaptation Results\n\n"
                f"Verdict: `{verdict}`\n\n"
                f"Numerical blocker reason: `{reason}`\n\n"
                "Canonical PatchCore status: `CANONICAL_PATCHCORE_REFERENCE_BLOCKED`\n"
                "Holdout access count: `0`\n",
                encoding="utf-8",
            )
            print(verdict)
            print(reason)
            return 0

    sanity = json.loads(SANITY_PATH.read_text(encoding="utf-8")) if SANITY_PATH.exists() else None

    # ---- bank ----
    if "bank" in run_stages and not BANK_PATH.exists():
        t0 = time.time()
        bank, seen = build_bank(model, device, train_ids, mean_t, w_t)
        D3_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BANK_PATH, features=bank.astype(np.float32))
        bank_sha = sha256_file(BANK_PATH)
        atomic_write_json(BANK_MANIFEST_PATH, {
            "candidate_id": "D3",
            "representation": "DINOv2 ViT-B/14 patch tokens, train-normal ZCA whitened + per-patch L2",
            "model_identifier": D2_REFERENCE["model_identifier"],
            "dim": int(bank.shape[1]),
            "rows": int(bank.shape[0]),
            "dtype": "float32",
            "sampling": "reservoir Algorithm R",
            "seed": ADAPTATION_SEED,
            "budget": ADAPTATION_BANK_BUDGET,
            "candidate_patches": seen,
            "source": "train_normal_subset",
            "source_count": len(train_ids),
            "bank_sha256": bank_sha,
            "whitening_sha256": sha256_file(WHITENING_PATH),
            "dino_weights_sha256": D2_REFERENCE["weights_sha256"],
            "subset_manifest_sha256": SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip(),
            "created_at": utc_now(),
        })
        print(f"bank built: dim={bank.shape[1]} rows={bank.shape[0]} candidate_patches={seen} "
              f"sha={bank_sha[:12]} in {time.time()-t0:.0f}s", flush=True)

    # ---- score ----
    if "score" in run_stages and not SCORES_PATH.exists():
        if not BANK_PATH.exists():
            raise RuntimeError("missing bank; run --stage bank first")
        bank = np.load(BANK_PATH)["features"].astype(np.float32)
        bank_t = torch.from_numpy(bank).to(device)
        train_scores = score_images(model, device, bank_t, mean_t, w_t, train_ids)
        atomic_write_json(SCORES_PATH, {"train": train_scores, "validation_normal": None, "dev_anomaly": None,
                                        "checkpoint": "train_only"})
        val_scores = score_images(model, device, bank_t, mean_t, w_t, val_ids)
        atomic_write_json(SCORES_PATH, {"train": train_scores, "validation_normal": val_scores, "dev_anomaly": None,
                                        "checkpoint": "train_val"})
        dev_scores = score_images(model, device, bank_t, mean_t, w_t, dev_ids)
        atomic_write_json(SCORES_PATH, {"train": train_scores, "validation_normal": val_scores,
                                        "dev_anomaly": dev_scores, "checkpoint": "complete"})

    if args.stage in ("stats", "sanity", "bank"):
        return 0  # partial stage only; no evaluation yet

    # ---- evaluate + gate ----
    cached = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
    ratios, quartiles = load_rep_module().load_area_ratios(dev_ids)
    ev = evaluate(cached["train"], cached["validation_normal"], cached["dev_anomaly"], quartiles)
    q = {row["quartile"]: row["normal_vs_quartile_auroc"] for row in ev["quartiles"]}
    passed = adaptation_gate_passed(D2_AUROC, ev["image_auroc"])
    strong = adaptation_strong_signal(ev["image_auroc"])
    small = small_defect_adaptation_signal(q[1], D2_QUARTILES["Q1"])

    verdict = "STEEL_DOMAIN_ADAPTATION_SIGNAL_FOUND" if passed else "STEEL_DOMAIN_WHITENING_GATE_FAILED"

    payload = {
        "schema_version": "steel_patchcore_domain_adaptation_results_v1",
        "protocol_version": "domain_adaptation_protocol_v1",
        "verdict": verdict,
        "gate": ADAPTATION_GATE,
        "strong_signal": strong,
        "small_defect_adaptation_signal": small,
        "small_defect_q1_delta_vs_d2": q[1] - D2_QUARTILES["Q1"],
        "small_defect_q1_delta_vs_d0": q[1] - D0_QUARTILES["Q1"],
        "d0": {"candidate": "S2", "image_auroc": D0_AUROC, "quartiles": D0_QUARTILES,
               "source": "frozen spatial-context results, not re-run"},
        "d1": {"candidate": "D1", "representation": "DINOv2 ViT-S/14 raw patch tokens",
               "image_auroc": D1_AUROC, "quartiles": D1_QUARTILES,
               "status": "DOMAIN_REPRESENTATION_GATE_FAILED", "small_defect_signal": True,
               "source": "frozen domain-representation results, not re-run"},
        "d2": {"candidate": "D2", "representation": "DINOv2 ViT-B/14 raw patch tokens",
               "image_auroc": D2_AUROC, "quartiles": D2_QUARTILES,
               "status": "DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED",
               "source": "frozen capacity results, not re-run"},
        "d3": {
            "reference": D2_REFERENCE,
            "runtime_identity": identity,
            "whitening_manifest": json.loads(WHITENING_MANIFEST_PATH.read_text(encoding="utf-8")),
            "sanity": sanity,
            "bank_sha256": sha256_file(BANK_PATH) if BANK_PATH.exists() else None,
            "metrics": ev,
            "delta_vs_d2": ev["image_auroc"] - D2_AUROC,
            "delta_vs_d0": ev["image_auroc"] - D0_AUROC,
            "gate_passed": passed,
        },
        "canonical_patchcore_status": "CANONICAL_PATCHCORE_REFERENCE_BLOCKED",
        "holdout_access_count": 0,
        "generated_at": utc_now(),
    }
    atomic_write_json(RESULTS_JSON, payload)
    render_markdown(payload, RESULTS_MD)
    print(json.dumps({
        "verdict": verdict,
        "D3_image_auroc": round(ev["image_auroc"], 4),
        "D2_image_auroc": D2_AUROC,
        "delta_vs_d2": round(ev["image_auroc"] - D2_AUROC, 4),
        "Q1": round(q[1], 4),
        "Q2": round(q[2], 4),
        "Q3": round(q[3], 4),
        "Q4": round(q[4], 4),
        "strong_signal": strong,
        "small_defect_adaptation_signal": small,
    }, default=str))
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())