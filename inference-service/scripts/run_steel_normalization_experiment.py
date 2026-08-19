"""Normalization/distance experiment for Steel PatchCore (Stage N).

Runs only after the feature-layer gate fails. On the frozen R0 layer setup
(layer2 + bilinear layer3, 1536-d):

- N0 = current semantics (concat then per-patch L2, cosine) == R0's numbers.
- N1 = explicit per-patch L2 + cosine; audited as code-equivalent to N0, so it
  is recorded as N0 without a redundant run.
- N2 = per-layer L2 normalization BEFORE concat, then per-patch L2, cosine.

Uses the same diagnostic subset, reservoir sampler (seed 42, budget 50000),
frozen tiling, and A0 global-max aggregation. No holdout, no frozen-bank writes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from steel_patchcore.lifecycle import lifecycle_enter  # noqa: E402
from steel_patchcore.representation import (  # noqa: E402
    NORMALIZATION_CANDIDATES,
    NORMALIZATION_GATE,
    normalization_gate_passed,
)
from steel_patchcore.recovery import sha256_file  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
SUBSET_MANIFEST = DS / "representation_diagnostic_manifest.json"
SUBSET_MANIFEST_SHA = DS / "representation_diagnostic_manifest.sha256"
RUN_ROOT = ROOT / "model-training/runs/steel-representation"
R_RESULTS = RUN_ROOT / "results.json"
OUT_JSON = RUN_ROOT / "normalization_results.json"
OUT_MD = RUN_ROOT / "normalization_results.md"

BANK_PATCHES = 50_000
SEED = 42


def _load_experiment_module():
    script = ROOT / "inference-service/scripts/run_steel_representation_experiment.py"
    spec = importlib.util.spec_from_file_location("_repr_experiment", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def n2_features(h2f: torch.Tensor, h3f: torch.Tensor) -> torch.Tensor:
    """Per-layer L2 before concat, then per-patch L2, cosine-ready (1536-d)."""
    l2 = torch.nn.functional.normalize(h2f, p=2, dim=1)
    l3 = torch.nn.functional.normalize(h3f, p=2, dim=1)
    cat = torch.cat([l2, l3], dim=1)
    return torch.nn.functional.normalize(cat, p=2, dim=1)


def build_n2_bank(module, model, device, train_ids):
    rng = np.random.default_rng(SEED)
    reservoir = None
    seen = 0
    for image_id in train_ids:
        tiles = module.load_normalized_tiles(module.IMG_DIR / f"{image_id}.jpg", device)
        for tile in tiles:
            h2, h3_up = module.extract_tile_spatial(model, tile)
            b = h2.shape[0]
            hh, ww = h2.shape[-2], h2.shape[-1]
            h2f = h2.permute(0, 2, 3, 1).reshape(b, hh * ww, -1)[0]
            h3f = h3_up.permute(0, 2, 3, 1).reshape(b, hh * ww, -1)[0]
            feats = n2_features(h2f, h3f).cpu().numpy().astype(np.float32)
            if reservoir is None:
                reservoir = np.zeros((BANK_PATCHES, feats.shape[1]), dtype=np.float32)
            for index in range(feats.shape[0]):
                seen += 1
                if seen <= BANK_PATCHES:
                    reservoir[seen - 1] = feats[index]
                else:
                    j = int(rng.integers(0, seen))
                    if j < BANK_PATCHES:
                        reservoir[j] = feats[index]
    return reservoir


def score_n2(module, model, device, bank, image_ids):
    bank_t = torch.from_numpy(bank).to(device)
    scores = []
    for image_id in image_ids:
        tile_max = []
        for tile in module.load_normalized_tiles(module.IMG_DIR / f"{image_id}.jpg", device):
            h2, h3_up = module.extract_tile_spatial(model, tile)
            b = h2.shape[0]
            hh, ww = h2.shape[-2], h2.shape[-1]
            h2f = h2.permute(0, 2, 3, 1).reshape(b, hh * ww, -1)[0]
            h3f = h3_up.permute(0, 2, 3, 1).reshape(b, hh * ww, -1)[0]
            emb = n2_features(h2f, h3f)
            sim = emb @ bank_t.T
            dist = 1.0 - sim.max(dim=1).values
            tile_max.append(float(dist.max()))
        scores.append(max(tile_max))
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-score", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("NORMALIZATION_EXPERIMENT_REQUIRES_GPU")
    device = torch.device("cuda:0")

    blocked = lifecycle_enter("representation_n2", "experimental_banks",
                              RUN_ROOT / "normalization_checkpoint.json", RUN_ROOT)
    if blocked is not None:
        return int(blocked)

    if sha256_file(SUBSET_MANIFEST) != SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip():
        raise RuntimeError("SUBSET_MANIFEST_SHA_MISMATCH")
    if not R_RESULTS.exists():
        raise RuntimeError("R results missing; run the R experiment first")

    subset = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    train_ids = list(subset["train_normal_subset"])
    val_ids = list(subset["validation_normal_subset"])
    dev_ids = list(subset["recovery_dev_anomaly_subset"])

    module = _load_experiment_module()
    model = None
    bank_path = RUN_ROOT / "N2" / "bank.npz"

    if not args.only_score:
        model = module.build_model(device)
        t0 = time.time()
        print("building N2 bank (per-layer L2 before concat)", flush=True)
        bank = build_n2_bank(module, model, device, train_ids)
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bank_path, features=bank.astype(np.float32))
        bank_sha = sha256_file(bank_path)
        json.dump({
            "candidate_id": "N2", "kind": "per_layer_l2_before_concat", "dim": int(bank.shape[1]),
            "rows": int(bank.shape[0]), "seed": SEED, "budget": BANK_PATCHES,
            "source": "train_normal_subset", "source_count": len(train_ids), "bank_sha256": bank_sha,
        }, open(bank_path.parent / "bank_manifest.json", "w", encoding="utf-8"), indent=2)
        print(f"N2 bank sha={bank_sha[:12]} in {time.time() - t0:.0f}s", flush=True)
    else:
        if not bank_path.exists():
            raise RuntimeError("N2 bank missing; run without --only-score first")

    model = model or module.build_model(device)
    bank = np.load(bank_path)["features"].astype(np.float32)

    # N0 == R0 (current semantics); N1 is audited code-equivalent to N0.
    r = json.loads(R_RESULTS.read_text(encoding="utf-8"))
    n0_auroc = float(r["candidates"]["R0"]["image_auroc"])
    _, quartiles = module.load_area_ratios(dev_ids)

    n2_train = score_n2(module, model, device, bank, train_ids)
    n2_val = score_n2(module, model, device, bank, val_ids)
    n2_dev = score_n2(module, model, device, bank, dev_ids)
    from steel_patchcore.aggregation import auroc, operating_point, normal_vs_quartile_auroc

    normal = np.asarray(n2_val, dtype=np.float64)
    anomaly = np.asarray(n2_dev, dtype=np.float64)
    n2_auroc = auroc(np.concatenate([normal, anomaly]),
                     np.concatenate([np.zeros(normal.size, dtype=int), np.ones(anomaly.size, dtype=int)]))
    threshold = float(np.max(n2_train))
    op = operating_point(normal, anomaly, threshold)
    quartiles_arr = np.asarray(quartiles, dtype=int)
    q_rows = []
    for q in (1, 2, 3, 4):
        qs = anomaly[quartiles_arr == q]
        q_rows.append({
            "quartile": q, "count": int(qs.size),
            "median_score": float(np.median(qs)) if qs.size else None,
            "recall": float((qs >= threshold).mean()) if qs.size else None,
            "normal_vs_quartile_auroc": normal_vs_quartile_auroc(normal, qs),
        })

    passed = normalization_gate_passed(n0_auroc, n2_auroc)
    results = {
        "schema_version": "steel_patchcore_representation_normalization_v1",
        "verdict": "NORMALIZATION_SIGNAL_FOUND" if passed else "NORMALIZATION_GATE_FAILED",
        "gate": NORMALIZATION_GATE,
        "n0_auroc": n0_auroc,
        "n1_note": "N1 (per-patch L2 + cosine) is code-equivalent to current semantics; recorded as N0.",
        "n2": {
            "image_auroc": n2_auroc,
            "delta_vs_n0": n2_auroc - n0_auroc,
            "threshold": threshold,
            "normal_median": float(np.median(normal)),
            "anomaly_median": float(np.median(anomaly)),
            "anomaly_minus_normal_median": float(np.median(anomaly) - np.median(normal)),
            "operating_point": op,
            "quartiles": q_rows,
            "gate_passed": passed,
        },
        "candidates": [c["id"] for c in NORMALIZATION_CANDIDATES],
        "holdout_access_count": 0,
    }
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    _render(results, OUT_MD)
    print(json.dumps({"verdict": results["verdict"], "n0": round(n0_auroc, 4),
                      "n2": round(n2_auroc, 4), "delta": round(n2_auroc - n0_auroc, 4)}))
    print(results["verdict"])
    return 0


def _render(results: dict, path: Path) -> None:
    lines = [
        "# Steel PatchCore — Representation Normalization Results (N0/N1/N2)",
        "",
        f"Verdict: `{results['verdict']}`",
        "",
        f"- N0 (current, =R0) AUROC: {results['n0_auroc']:.4f}",
        f"- {results['n1_note']}",
        f"- N2 AUROC: {results['n2']['image_auroc']:.4f} (Δ vs N0 = {results['n2']['delta_vs_n0']:+.4f})",
        "",
        "| Quartile | count | median score | recall | normal-vs-quartile AUROC |",
        "|---|---|---|---|---|",
    ]
    for row in results["n2"]["quartiles"]:
        lines.append(f"| Q{row['quartile']} | {row['count']} | {row['median_score']:.6f} | {row['recall']:.4f} | {row['normal_vs_quartile_auroc']:.4f} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())