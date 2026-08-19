"""Representation investigation GPU experiment for Steel PatchCore (Stage R).

Builds isolated experimental banks for R0/R1/R2 from the frozen diagnostic
subset and scores it with the frozen representation semantics (reservoir 50k,
seed 42, cosine 1-NN, A0 global max, frozen 7 tiles). No holdout, no registry,
no mutation of the frozen 1.0.0 bank.

Run with the CUDA python env (GPU required):
  .venv-steel/Scripts/python.exe inference-service/scripts/run_steel_representation_experiment.py
  (options: --only-banks | --only-score)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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
from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.representation import (  # noqa: E402
    FEATURE_LAYER_CANDIDATES,
    FEATURE_LAYER_GATE,
    feature_layer_gate_passed,
)
from steel_patchcore.tile import TILE_X0, tile_coords  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BANK_PATCHES = 50_000
SEED = 42
IMAGE_SIZE = 256

DS = ROOT / "model-training/datasets/severstal-steel"
IMG_DIR = DS / "raw/train_images"
CSV = DS / "raw/train.csv"
SOURCE_SPLIT = DS / "split_manifest.json"
SUBSET_MANIFEST = DS / "representation_diagnostic_manifest.json"
SUBSET_MANIFEST_SHA = DS / "representation_diagnostic_manifest.sha256"

FRZ = ROOT / "inference-service/models/steel-patchcore/bank.npz"
EXPECTED_FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"

RUN_ROOT = ROOT / "model-training/runs/steel-representation"
RUN_MANIFEST = RUN_ROOT / "experiment_manifest.json"
RESULTS_JSON = RUN_ROOT / "results.json"
RESULTS_MD = RUN_ROOT / "results.md"

FEATURES = {
    "R0": {"layers": ("layer2", "layer3")},
    "R1": {"layers": ("layer2",)},
    "R2": {"layers": ("layer3",)},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize(value):
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_normalized_tiles(image_path: Path, device: torch.device) -> list[torch.Tensor]:
    """Crop the frozen 7 tiles from the 256x1600 original and ImageNet-normalize.

    The original is 256x1600; tiles are 256x256 crops at the frozen offsets
    (no resize — a tile is already 256x256, matching the 1.0.0 path).
    """
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tiles: list[torch.Tensor] = []
    for tile_id in range(len(TILE_X0)):
        x0, y0, w, h = tile_coords(tile_id)
        tile_img = image.crop((x0, y0, x0 + w, y0 + h))
        arr = np.asarray(tile_img, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
        tiles.append(torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device))
    return tiles


def build_model(device: torch.device):
    from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    return model


def extract_tile_spatial(model: torch.nn.Module, x: torch.Tensor):
    """Return (layer2 (1,512,32,32), layer3_up (1,1024,32,32)) on device."""
    with torch.no_grad():
        h = model.conv1(x)
        h = model.bn1(h)
        h = model.relu(h)
        h = model.maxpool(h)
        h = model.layer1(h)
        h2 = model.layer2(h)
        h3 = model.layer3(h2)
        h3_up = torch.nn.functional.interpolate(
            h3, size=h2.shape[-2:], mode="bilinear", align_corners=False
        )
    return h2, h3_up


def feature_views(h2: torch.Tensor, h3_up: torch.Tensor) -> dict[str, torch.Tensor]:
    """Row-major (H*W, D) L2-normalized patch embeddings per candidate."""
    b = h2.shape[0]
    hh, ww = h2.shape[-2], h2.shape[-1]
    h2f = h2.permute(0, 2, 3, 1).reshape(b, hh * ww, -1)
    h3f = h3_up.permute(0, 2, 3, 1).reshape(b, hh * ww, -1)
    r0 = torch.cat([h2f, h3f], dim=2)
    return {
        "R0": torch.nn.functional.normalize(r0, p=2, dim=2)[0],
        "R1": torch.nn.functional.normalize(h2f, p=2, dim=2)[0],
        "R2": torch.nn.functional.normalize(h3f, p=2, dim=2)[0],
    }


def reservoir_from_tiles(
    model: torch.nn.Module,
    device: torch.device,
    image_ids: list[str],
    budgets: dict[str, int],
) -> dict[str, np.ndarray]:
    """Build one reservoir bank per candidate over the given images."""
    rng = np.random.default_rng(SEED)
    reservoirs: dict[str, np.ndarray] = {}
    seen = 0
    for image_id in image_ids:
        tiles = load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device)
        for tile in tiles:
            h2, h3_up = extract_tile_spatial(model, tile)
            views = feature_views(h2, h3_up)
            arrays = {cid: v.cpu().numpy().astype(np.float32) for cid, v in views.items()}
            if not reservoirs:
                reservoirs = {
                    cid: np.zeros((budgets[cid], arrays[cid].shape[1]), dtype=np.float32)
                    for cid in arrays
                }
            for index in range(arrays["R0"].shape[0]):
                seen += 1
                if seen <= BANK_PATCHES:
                    for cid, arr in arrays.items():
                        reservoirs[cid][seen - 1] = arr[index]
                else:
                    j = int(rng.integers(0, seen))
                    if j < BANK_PATCHES:
                        for cid, arr in arrays.items():
                            reservoirs[cid][j] = arr[index]
    return reservoirs


def score_candidates(
    model: torch.nn.Module,
    device: torch.device,
    banks: dict[str, torch.Tensor],
    image_ids: list[str],
) -> dict[str, list[float]]:
    """A0 (global max raw cosine 1-NN distance) per image for each candidate."""
    scores: dict[str, list[float]] = {}
    for image_id in image_ids:
        tiles = load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device)
        tile_max: dict[str, list[float]] = {}
        for tile in tiles:
            h2, h3_up = extract_tile_spatial(model, tile)
            views = feature_views(h2, h3_up)
            for cid, emb in views.items():
                sim = emb @ banks[cid].T  # (P, N)
                dist = 1.0 - sim.max(dim=1).values  # (P,)
                tile_max.setdefault(cid, []).append(float(dist.max()))
        for cid, tm in tile_max.items():
            scores.setdefault(cid, []).append(max(tm))
    return scores


def load_area_ratios(image_ids: list[str]) -> tuple[np.ndarray, list[int]]:
    wanted = set(image_ids)
    df = pd.read_csv(CSV, keep_default_na=True)
    info: dict[str, float] = {}
    for img_id, group in df.groupby(df["ImageId"].astype(str)):
        norm = Path(str(img_id)).stem
        if norm not in wanted:
            continue
        mask = np.zeros((256, 1600), dtype=np.uint8)
        for value in group["EncodedPixels"]:
            if pd.isna(value):
                continue
            from steel_patchcore.rle import rle_decode

            mask = np.maximum(mask, rle_decode(str(value)))
        info[norm] = float(mask.sum()) / (256 * 1600)
    missing = wanted - set(info)
    if missing:
        raise RuntimeError(f"MISSING_MASK:{sorted(missing)[:5]}")
    ratios = np.asarray([info[i] for i in image_ids], dtype=np.float64)
    q1, q2, q3 = (float(v) for v in np.quantile(ratios, [0.25, 0.5, 0.75], method="linear"))
    quartiles = np.empty(len(ratios), dtype=np.int8)
    quartiles[ratios < q1] = 1
    quartiles[(ratios >= q1) & (ratios < q2)] = 2
    quartiles[(ratios >= q2) & (ratios < q3)] = 3
    quartiles[ratios >= q3] = 4
    return ratios, [int(v) for v in quartiles]


def evaluate(scores_train, scores_val, scores_dev, quartiles):
    normal = np.asarray(scores_val, dtype=np.float64)
    anomaly = np.asarray(scores_dev, dtype=np.float64)
    threshold = float(np.max(scores_train))
    image_auroc = auroc(np.concatenate([normal, anomaly]),
                        np.concatenate([np.zeros(normal.size, dtype=int), np.ones(anomaly.size, dtype=int)]))
    op = operating_point(normal, anomaly, threshold)
    quartiles_arr = np.asarray(quartiles, dtype=int)
    q_rows = []
    for q in (1, 2, 3, 4):
        q_scores = anomaly[quartiles_arr == q]
        q_rows.append({
            "quartile": q,
            "count": int(q_scores.size),
            "median_score": float(np.median(q_scores)) if q_scores.size else None,
            "recall": float((q_scores >= threshold).mean()) if q_scores.size else None,
            "normal_vs_quartile_auroc": normal_vs_quartile_auroc(normal, q_scores),
        })
    return {
        "threshold": threshold,
        "image_auroc": image_auroc,
        "normal_median": float(np.median(normal)),
        "anomaly_median": float(np.median(anomaly)),
        "anomaly_minus_normal_median": float(np.median(anomaly) - np.median(normal)),
        "operating_point": op,
        "quartiles": q_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-banks", action="store_true")
    parser.add_argument("--only-score", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("REPRESENTATION_EXPERIMENT_REQUIRES_GPU")
    device = torch.device("cuda:0")

    from steel_patchcore.lifecycle import lifecycle_enter

    blocked = lifecycle_enter(
        "representation_r0_r1_r2", "experimental_banks", RUN_ROOT / "checkpoint.json", RUN_ROOT
    )
    if blocked is not None:
        return int(blocked)

    if sha256_file(FRZ) != EXPECTED_FROZEN_BANK_SHA:
        raise RuntimeError("FROZEN_BANK_SHA_MISMATCH")
    if sha256_file(SUBSET_MANIFEST) != SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip():
        raise RuntimeError("SUBSET_MANIFEST_SHA_MISMATCH")

    subset = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    train_ids = list(subset["train_normal_subset"])
    val_ids = list(subset["validation_normal_subset"])
    dev_ids = list(subset["recovery_dev_anomaly_subset"])
    assert subset["holdout_access_count"] == 0

    budgets = {c["id"]: BANK_PATCHES for c in FEATURE_LAYER_CANDIDATES}
    model = build_model(device)

    # ---- banks ----
    bank_paths: dict[str, Path] = {}
    if not args.only_score:
        print("building R0/R1/R2 banks from train_normal_subset", flush=True)
        t0 = time.time()
        reservoirs = reservoir_from_tiles(model, device, train_ids, budgets)
        for cid, arr in reservoirs.items():
            out_dir = RUN_ROOT / cid
            out_dir.mkdir(parents=True, exist_ok=True)
            bank_path = out_dir / "bank.npz"
            np.savez_compressed(bank_path, features=arr.astype(np.float32))
            bank_sha = sha256_file(bank_path)
            atomic_write_json(out_dir / "bank_manifest.json", {
                "candidate_id": cid,
                "feature_layers": list(FEATURES[cid]["layers"]),
                "dim": int(arr.shape[1]),
                "rows": int(arr.shape[0]),
                "sampling": "reservoir Algorithm R",
                "seed": SEED,
                "budget": BANK_PATCHES,
                "source": "train_normal_subset",
                "source_count": len(train_ids),
                "bank_sha256": bank_sha,
                "subset_manifest_sha256": SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip(),
                "created_at": utc_now(),
            })
            bank_paths[cid] = bank_path
            print(f"{cid}: bank dim={arr.shape[1]} rows={arr.shape[0]} sha={bank_sha[:12]}", flush=True)
        print(f"banks built in {time.time() - t0:.0f}s", flush=True)
    else:
        for cid in FEATURES:
            p = RUN_ROOT / cid / "bank.npz"
            if not p.exists():
                raise RuntimeError(f"missing bank for {cid}; run without --only-score first")
            bank_paths[cid] = p

    # ---- score ----
    if not args.only_banks:
        banks = {cid: torch.from_numpy(np.load(p)["features"].astype(np.float32)).to(device)
                 for cid, p in bank_paths.items()}
        train_scores = score_candidates(model, device, banks, train_ids)
        val_scores = score_candidates(model, device, banks, val_ids)
        dev_scores = score_candidates(model, device, banks, dev_ids)
        ratios, quartiles = load_area_ratios(dev_ids)

        results = {"verdict": None, "candidates": {}, "gate": FEATURE_LAYER_GATE}
        for cid in FEATURES:
            ev = evaluate(train_scores[cid], val_scores[cid], dev_scores[cid], quartiles)
            results["candidates"][cid] = ev
        r0_auroc = results["candidates"]["R0"]["image_auroc"]
        signal = []
        for cid in ("R1", "R2"):
            cand_auroc = results["candidates"][cid]["image_auroc"]
            passed = feature_layer_gate_passed(r0_auroc, cand_auroc)
            results["candidates"][cid]["gate_passed"] = passed
            if passed:
                signal.append(cid)
        results["r0_frozen_bank_auroc_reference"] = None  # filled below
        results["verdict"] = "FEATURE_LAYER_SIGNAL_FOUND" if signal else "FEATURE_LAYER_GATE_FAILED"
        results["signal_candidates"] = signal
        results["holdout_access_count"] = 0
        results["generated_at"] = utc_now()

        # frozen-bank R0 reference on the SAME subset, from existing raw evidence (no inference)
        try:
            ref = frozen_bank_reference_auroc(val_ids, dev_ids)
            results["r0_frozen_bank_auroc_reference"] = ref
        except Exception as exc:  # noqa: BLE001
            results["r0_frozen_bank_auroc_reference"] = {"error": str(exc)}

        atomic_write_json(RESULTS_JSON, results)
        render_markdown(results, RESULTS_MD)
        print(json.dumps({
            "verdict": results["verdict"],
            "R0": round(r0_auroc, 4),
            "R1": round(results["candidates"]["R1"]["image_auroc"], 4),
            "R2": round(results["candidates"]["R2"]["image_auroc"], 4),
            "frozen_ref": results["r0_frozen_bank_auroc_reference"],
        }))
        print(results["verdict"])
    return 0


def frozen_bank_reference_auroc(val_ids: list[str], dev_ids: list[str]) -> dict:
    """Frozen 1.0.0 bank A0 AUROC on the exact subset, from existing evidence."""
    EVIDENCE_ROOT = DS / "raw/recovery-evidence"
    scores: dict[str, dict[str, float]] = {"validation_normal": {}, "recovery_dev_anomaly": {}}
    want = {"validation_normal": set(val_ids), "recovery_dev_anomaly": set(dev_ids)}
    for role in ("validation_normal", "recovery_dev_anomaly"):
        for path in sorted((EVIDENCE_ROOT / role).glob("shard-*.npz")):
            with np.load(path, allow_pickle=False) as data:
                ids = [str(v) for v in data["original_ids"].tolist()]
                base = data["baseline_scores"]
            for image_id, score in zip(ids, base):
                if image_id in want[role]:
                    scores[role][image_id] = float(score)
    normal = np.asarray([scores["validation_normal"][i] for i in val_ids])
    anomaly = np.asarray([scores["recovery_dev_anomaly"][i] for i in dev_ids])
    au = auroc(np.concatenate([normal, anomaly]),
               np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)]))
    return {
        "image_auroc": au,
        "normal_median": float(np.median(normal)),
        "anomaly_median": float(np.median(anomaly)),
    }


def render_markdown(results: dict, path: Path) -> None:
    lines: list[str] = []

    def add(s=""):
        lines.append(s)

    def fmt(v, d=4):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:.{d}f}"

    add("# Steel PatchCore — Representation Feature-layer Results (R0/R1/R2)")
    add("")
    add(f"Verdict: `{results['verdict']}`")
    add(f"- Generated at: `{results['generated_at']}`")
    add(f"- Holdout access count: `{results['holdout_access_count']}`")
    if results["r0_frozen_bank_auroc_reference"] and "image_auroc" in results["r0_frozen_bank_auroc_reference"]:
        add(f"- R0 (frozen 1.0.0 bank) reference AUROC on subset: {fmt(results['r0_frozen_bank_auroc_reference']['image_auroc'])}")
    add("")
    add("| Candidate | AUROC | normal median | anomaly median | Δmedian | TP | TN | FP | FN | FPR | Recall | Gate |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in ("R0", "R1", "R2"):
        c = results["candidates"][cid]
        op = c["operating_point"]
        gate = c.get("gate_passed")
        gate_txt = "PASS" if gate is True else ("FAIL" if gate is False else "-")
        add(f"| {cid} | {fmt(c['image_auroc'])} | {fmt(c['normal_median'])} | {fmt(c['anomaly_median'])} | {fmt(c['anomaly_minus_normal_median'])} | {op['tp']} | {op['tn']} | {op['fp']} | {op['fn']} | {fmt(op['normal_fpr'])} | {fmt(op['anomaly_recall'])} | {gate_txt} |")
    add("")
    add("## Normal-vs-quartile AUROC")
    add("")
    add("| Candidate | Q1 | Q2 | Q3 | Q4 |")
    add("|---|---|---|---|---|")
    for cid in ("R0", "R1", "R2"):
        qs = results["candidates"][cid]["quartiles"]
        add(f"| {cid} | {fmt(qs[0]['normal_vs_quartile_auroc'])} | {fmt(qs[1]['normal_vs_quartile_auroc'])} | {fmt(qs[2]['normal_vs_quartile_auroc'])} | {fmt(qs[3]['normal_vs_quartile_auroc'])} |")
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())