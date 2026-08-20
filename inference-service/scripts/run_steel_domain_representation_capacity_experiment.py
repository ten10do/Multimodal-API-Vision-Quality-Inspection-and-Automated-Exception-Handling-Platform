"""DINOv2 capacity cross-check (D2 = ViT-B/14) GPU experiment.

Changes ONLY the DINOv2 model capacity vs D1 (ViT-S/14 -> ViT-B/14). Everything
else is frozen: diagnostic subset, reservoir 50k/seed 42, frozen 7 tiles + A0 max,
train-only threshold, per-patch L2 + cosine 1-NN (k=1, no reweighting). No holdout,
no baseline/D1-bank write, no fine-tuning, no bank-strategy change.

Run with the CUDA env (GPU required):
  .venv-steel/Scripts/python.exe inference-service/scripts/run_steel_domain_representation_capacity_experiment.py
  options: --stage bank|score|all  (resumable checkpoints: bank.npz, scores.json)
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from steel_patchcore.aggregation import (  # noqa: E402
    auroc,
    distribution,
    normal_vs_quartile_auroc,
    operating_point,
)
from steel_patchcore.domain_representation import (  # noqa: E402
    D0_AUROC,
    D0_QUARTILES,
    adapted_input_side,
)
from steel_patchcore.domain_representation_capacity import (  # noqa: E402
    CAPACITY_BANK_BUDGET,
    CAPACITY_GATE,
    CAPACITY_SEED,
    D1_AUROC,
    D1_QUARTILES,
    D1_SMALL_DEFECT_SIGNAL,
    D2_REFERENCE,
    capacity_gain,
    capacity_gate_passed,
    capacity_strong_signal,
    d2_reference_sha256,
)
from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.representation import reservoir_from_stream  # noqa: E402
from steel_patchcore.tile import TILE_X0  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
IMG_DIR = DS / "raw/train_images"
CSV = DS / "raw/train.csv"
SUBSET_MANIFEST = DS / "representation_diagnostic_manifest.json"
SUBSET_MANIFEST_SHA = DS / "representation_diagnostic_manifest.sha256"
FRZ = ROOT / "inference-service/models/steel-patchcore/bank.npz"
EXPECTED_FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"

RUN_ROOT = ROOT / "model-training/runs/steel-domain-representation-capacity"
D2_DIR = RUN_ROOT / "D2-dinov2-b14"
BANK_PATH = D2_DIR / "bank.npz"
BANK_MANIFEST_PATH = D2_DIR / "bank_manifest.json"
SCORES_PATH = D2_DIR / "scores.json"
RESULTS_JSON = RUN_ROOT / "results.json"
RESULTS_MD = RUN_ROOT / "results.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict) -> None:
    def sanitize(v):
        if isinstance(v, dict):
            return {k: sanitize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [sanitize(val) for val in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_rep_module():
    script = ROOT / "inference-service/scripts/run_steel_representation_experiment.py"
    spec = importlib.util.spec_from_file_location("_repr_experiment_d2", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dino_model(device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    return model


def extract_dino_patch_tokens(model: torch.nn.Module, tile: torch.Tensor) -> torch.Tensor:
    """(1,3,256,256) tile -> (N_patch, 768) per-patch-L2-normalized patch tokens."""
    side = adapted_input_side(tile.shape[-1], 14)
    if tile.shape[-1] != side:
        tile = torch.nn.functional.interpolate(tile, size=(side, side), mode="bilinear", align_corners=False)
    with torch.no_grad():
        out = model.forward_features(tile)
    patch_tokens = out["x_norm_patchtokens"]  # (1, N, 768)
    return torch.nn.functional.normalize(patch_tokens[0], p=2, dim=1)


def build_bank(model, device, train_ids) -> tuple[np.ndarray, int]:
    def stream():
        for image_id in train_ids:
            for tile in load_rep_module().load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device):
                yield extract_dino_patch_tokens(model, tile).cpu().numpy().astype(np.float32)

    bank, seen = reservoir_from_stream(stream(), CAPACITY_BANK_BUDGET, CAPACITY_SEED)
    return bank, seen


def score_images(model, device, bank_t: torch.Tensor, image_ids) -> list[float]:
    scores: list[float] = []
    for image_id in image_ids:
        tile_scores: list[float] = []
        for tile in load_rep_module().load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device):
            emb = extract_dino_patch_tokens(model, tile)  # (N,768) L2-normalized
            sim = emb @ bank_t.T
            dist = 1.0 - sim.max(dim=1).values
            tile_scores.append(float(dist.max()))
        scores.append(max(tile_scores))  # A0: original score = max over 7 tiles
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
        "feedforward_output_keys": ["x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm", "masks"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["bank", "score", "all"], default="all")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DOMAIN_REPRESENTATION_CAPACITY_EXPERIMENT_REQUIRES_GPU")
    device = torch.device("cuda:0")

    from steel_patchcore.lifecycle import lifecycle_enter

    blocked = lifecycle_enter("domain_representation_capacity_d2", "experimental_banks",
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
    ref_identity = {**D2_REFERENCE, "reference_sha256": d2_reference_sha256()}

    # ---- bank (resumable) ----
    if args.stage in ("bank", "all") and BANK_PATH.exists():
        print("bank already built; loading (resume)", flush=True)
        bank = np.load(BANK_PATH)["features"].astype(np.float32)
        seen = int(bank.shape[0])
    elif args.stage in ("bank", "all"):
        t0 = time.time()
        bank, seen = build_bank(model, device, train_ids)
        D2_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BANK_PATH, features=bank.astype(np.float32))
        bank_sha = sha256_file(BANK_PATH)
        atomic_write_json(BANK_MANIFEST_PATH, {
            "candidate_id": "D2",
            "representation": "DINOv2 ViT-B/14 spatial patch tokens (raw, per-patch L2)",
            "model_identifier": D2_REFERENCE["model_identifier"],
            "dim": int(bank.shape[1]),
            "rows": int(bank.shape[0]),
            "dtype": "float32",
            "sampling": "reservoir Algorithm R",
            "seed": CAPACITY_SEED,
            "budget": CAPACITY_BANK_BUDGET,
            "candidate_patches": seen,
            "source": "train_normal_subset",
            "source_count": len(train_ids),
            "bank_sha256": bank_sha,
            "subset_manifest_sha256": SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip(),
            "reference_sha256": d2_reference_sha256(),
            "created_at": utc_now(),
        })
        print(f"bank built: dim={bank.shape[1]} rows={bank.shape[0]} candidate_patches={seen} sha={bank_sha[:12]} "
              f"in {time.time() - t0:.0f}s", flush=True)
    else:  # score-only: bank must exist
        if not BANK_PATH.exists():
            raise RuntimeError("missing bank; run with --stage bank first")
        bank = np.load(BANK_PATH)["features"].astype(np.float32)

    bank_t = torch.from_numpy(bank).to(device)

    # ---- score (resumable) ----
    if args.stage in ("score", "all") and SCORES_PATH.exists():
        cached = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        train_scores = cached["train"]
        val_scores = cached["validation_normal"]
        dev_scores = cached["dev_anomaly"]
        print("scores already computed; loading (resume)", flush=True)
    elif args.stage in ("score", "all"):
        train_scores = score_images(model, device, bank_t, train_ids)
        atomic_write_json(SCORES_PATH, {"train": train_scores, "validation_normal": None, "dev_anomaly": None,
                                        "checkpoint": "train_only"})
        val_scores = score_images(model, device, bank_t, val_ids)
        atomic_write_json(SCORES_PATH, {"train": train_scores, "validation_normal": val_scores, "dev_anomaly": None,
                                        "checkpoint": "train_val"})
        dev_scores = score_images(model, device, bank_t, dev_ids)
        atomic_write_json(SCORES_PATH, {"train": train_scores, "validation_normal": val_scores,
                                        "dev_anomaly": dev_scores, "checkpoint": "complete"})
    else:
        cached = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        train_scores = cached["train"]
        val_scores = cached["validation_normal"]
        dev_scores = cached["dev_anomaly"]

    # ---- evaluate + gate ----
    ratios, quartiles = load_rep_module().load_area_ratios(dev_ids)
    ev = evaluate(train_scores, val_scores, dev_scores, quartiles)
    q = {row["quartile"]: row["normal_vs_quartile_auroc"] for row in ev["quartiles"]}
    q1, q2 = q[1], q[2]
    passed = capacity_gate_passed(D0_AUROC, ev["image_auroc"])
    strong = capacity_strong_signal(ev["image_auroc"])
    gain = capacity_gain(ev["image_auroc"], D1_AUROC)

    verdict = (
        "DOMAIN_REPRESENTATION_CAPACITY_SIGNAL_FOUND" if passed
        else "DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED"
    )

    payload = {
        "schema_version": "steel_patchcore_domain_representation_capacity_results_v1",
        "protocol_version": "domain_representation_capacity_protocol_v1",
        "verdict": verdict,
        "gate": CAPACITY_GATE,
        "strong_signal": strong,
        "capacity_gain": gain,
        "small_defect_q1_delta_vs_d1": q1 - D1_QUARTILES["Q1"],
        "small_defect_q2_delta_vs_d1": q2 - D1_QUARTILES["Q2"],
        "small_defect_q1_delta_vs_d0": q1 - D0_QUARTILES["Q1"],
        "d0": {"candidate": "S2", "image_auroc": D0_AUROC, "quartiles": D0_QUARTILES,
               "source": "frozen spatial-context results, not re-run"},
        "d1": {"candidate": "D1", "representation": "DINOv2 ViT-S/14 raw patch tokens",
               "image_auroc": D1_AUROC, "quartiles": D1_QUARTILES,
               "status": "DOMAIN_REPRESENTATION_GATE_FAILED",
               "small_defect_signal": D1_SMALL_DEFECT_SIGNAL,
               "source": "frozen domain-representation results, not re-run"},
        "d2": {
            "reference": ref_identity,
            "runtime_identity": identity,
            "bank_sha256": sha256_file(BANK_PATH) if BANK_PATH.exists() else None,
            "bank_manifest": json.loads(BANK_MANIFEST_PATH.read_text(encoding="utf-8")) if BANK_MANIFEST_PATH.exists() else None,
            "metrics": ev,
            "delta_vs_d0": ev["image_auroc"] - D0_AUROC,
            "delta_vs_d1": ev["image_auroc"] - D1_AUROC,
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
        "D2_image_auroc": round(ev["image_auroc"], 4),
        "D1_image_auroc": D1_AUROC,
        "D0_image_auroc": D0_AUROC,
        "delta_vs_d0": round(ev["image_auroc"] - D0_AUROC, 4),
        "delta_vs_d1": round(ev["image_auroc"] - D1_AUROC, 4),
        "capacity_gain": gain,
        "Q1": round(q1, 4),
        "Q2": round(q2, 4),
        "Q3": round(q[3], 4),
        "Q4": round(q[4], 4),
        "strong_signal": strong,
    }, default=str))
    print(verdict)
    return 0


def render_markdown(payload: dict, path: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# Steel PatchCore — DINOv2 Capacity Cross-Check Results (D0 WRN vs D1 S/14 vs D2 B/14)")
    A("")
    A(f"Verdict: `{payload['verdict']}`")
    A(f"- Strong signal: `{payload['strong_signal']}`")
    A(f"- Capacity gain (D2-D1>=+0.03): `{payload['capacity_gain']}`")
    A(f"- Canonical PatchCore status: `{payload['canonical_patchcore_status']}`")
    A(f"- Holdout access count: `{payload['holdout_access_count']}`")
    A("")
    A("## D0 / D1 / D2 comparison")
    A("")
    d0 = payload["d0"]
    d1 = payload["d1"]
    d2 = payload["d2"]
    m = d2["metrics"]
    A(f"- D0 (WRN layer3 + 5x5) Image AUROC = {d0['image_auroc']:.4f}")
    A(f"- D1 (DINOv2 ViT-S/14) Image AUROC = {d1['image_auroc']:.4f}")
    A(f"- D2 (DINOv2 ViT-B/14) Image AUROC = {m['image_auroc']:.4f}  (Δ vs D0 {d2['delta_vs_d0']:+.4f}, Δ vs D1 {d2['delta_vs_d1']:+.4f})")
    A(f"- Gate: `{payload['gate']}`")
    A("")
    A("| | AUROC | Q1 | Q2 | Q3 | Q4 |")
    A("|---|---|---|---|---|---|")
    q = m["quartiles"]
    A(f"| D0 | {d0['image_auroc']:.4f} | {d0['quartiles']['Q1']:.4f} | {d0['quartiles']['Q2']:.4f} | {d0['quartiles']['Q3']:.4f} | {d0['quartiles']['Q4']:.4f} |")
    A(f"| D1 | {d1['image_auroc']:.4f} | {d1['quartiles']['Q1']:.4f} | {d1['quartiles']['Q2']:.4f} | {d1['quartiles']['Q3']:.4f} | {d1['quartiles']['Q4']:.4f} |")
    A(f"| D2 | {m['image_auroc']:.4f} | {q[0]['normal_vs_quartile_auroc']:.4f} | {q[1]['normal_vs_quartile_auroc']:.4f} | {q[2]['normal_vs_quartile_auroc']:.4f} | {q[3]['normal_vs_quartile_auroc']:.4f} |")
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


if __name__ == "__main__":
    raise SystemExit(main())